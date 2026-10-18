"""Comparable cases, preserved failures and privileged evaluation outside policy calls."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from world_use import evaluation
from world_use.recorder import Tape
from world_use.tasks import digest, environment, json_bytes


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def evidence(folder):
    save(folder / "task/result.json", dict(recording_error=None, flight_recording_error=None,
        final_status=dict(enabled=False, power_uncertain=False, physics_recording=dict(error=None))))
    save(folder / "flight/session.json", dict(format_version=3))
    tape = Tape(1)
    tape.add(0, False, False, 0, [0], [0], None, None, None, None, None)
    (folder / "flight/tape").mkdir()
    np.savez_compressed(folder / "flight/tape/000000.npz", **tape.arrays())
    event = json_bytes(dict(kind="closed", level="info", t=0, message="closed"))
    # Journal events are single-line JSON.
    event = json_bytes(json.loads(event), indent=None)
    (folder / "flight/events.jsonl").write_bytes(event)
    save(folder / "flight/complete.json", dict(parts=1, events_bytes=len(event)))
    save(folder / "physics/complete.json", dict(parts=1, error=None))
    arrays = dict(time=np.array([0., .1, .2]), simulation_time=np.array([1., 1.1, 1.2]),
                  qpos=np.zeros((3, 1)), qvel=np.zeros((3, 1)), state=np.zeros((3, 2)))
    np.savez_compressed(folder / "physics/000000.npz", **arrays)
    np.savez_compressed(folder / "replay.npz", **arrays)


@pytest.fixture
def programs(tmp_path):
    task, evaluator, model = (tmp_path / name for name in ("policy.py", "audit.py", "model.dat"))
    task.write_text("def run(robot, params, record):\n    return {'verdict': 'pass'}\n")
    evaluator.write_text("def run(robot, params, record):\n    return {'verdict': 'pass'}\n")
    model.write_bytes(b"frozen model")
    return task, evaluator, model


@pytest.fixture
def fake_process(monkeypatch):
    calls = []

    def launch(task, output, *, parameters, **kwargs):
        output.mkdir()
        calls.append((task, parameters))
        save(output / "invocation.json", dict(environment=environment()))
        if "episode" in parameters:
            case = parameters["case"]
            result = dict(verdict="pass", valid_start=case["parameters"].get("valid_start", True),
                          reasons=[], interventions=[], evaluator_secret="only after execution")
        else:
            assert "evaluator_secret" not in parameters and "episode" not in parameters
            mode = parameters["parameters"].get("mode")
            if mode == "killed":
                return 9
            if mode == "interrupt":
                raise KeyboardInterrupt
            evidence(Path(parameters["output"]))
            if mode == "corrupt":
                (Path(parameters["output"]) / "flight/complete.json").unlink()
            result = dict(verdict="fail" if mode == "failed" else "pass")
        save(output / "result.json", dict(result=result, exception=None, recording_error=None))
        return 1 if result["verdict"] == "fail" else 0

    monkeypatch.setattr(evaluation, "launch", launch)
    return calls


def case(name="nominal", **parameters):
    return dict(name=name, role="development", seed=37, parameters=parameters)


def run(programs, output, **kwargs):
    task, evaluator, model = programs
    return evaluation.run_suite(task, evaluator, output, cases=kwargs.pop("cases", [case()]),
                                inputs={"model.dat": model}, scene_revision="a" * 64, **kwargs)


def test_case_identity_survives_policy_change_and_inputs_are_frozen(tmp_path, programs, fake_process):
    first = run(programs, tmp_path / "one")
    programs[0].write_text("def run(robot, params, record):\n    return {'verdict': 'fail'}\n")
    second = run(programs, tmp_path / "two")
    assert first["policy_revision"] != second["policy_revision"]
    assert first["trials"][0]["case"]["id"] == second["trials"][0]["case"]["id"]
    assert len(evaluation.compare(tmp_path / "one", tmp_path / "two")) == 1
    programs[2].write_bytes(b"changed original")
    assert (tmp_path / "one/inputs/model.dat").read_bytes() == b"frozen model"
    assert fake_process[0][1]["seed"] == 37
    assert first["scene_validation"]["status"] == "unvalidated"
    assert first["trials"][0]["simulation_s"] == pytest.approx(.2)


def test_policy_knowledge_is_frozen_and_comparable_on_identical_cases(tmp_path, programs, fake_process):
    skill = tmp_path / "shared-skill.md"
    skill.write_text("Verify the held object after lifting.")
    first = run(programs, tmp_path / "one", policy_context={"skills/grasp.md": skill})
    frozen = Path(fake_process[0][1]["context"]["skills/grasp.md"])
    assert frozen.read_text() == skill.read_text()
    assert "context" not in fake_process[1][1]  # Evaluator inputs stay independent.
    skill.write_text("Verify again after transport.")
    second = run(programs, tmp_path / "two", policy_context={"skills/grasp.md": skill})
    assert first["policy_revision"] != second["policy_revision"]
    assert first["policy_context"] != second["policy_context"]
    assert len(evaluation.compare(tmp_path / "one", tmp_path / "two")) == 1
    repeated = evaluation.rerun(tmp_path / "one", "nominal", tmp_path / "repeat")
    assert repeated["policy_revision"] == first["policy_revision"]
    assert repeated["policy_context"] == first["policy_context"]
    frozen.write_text("changed snapshot")
    with pytest.raises(ValueError, match="source snapshot changed"):
        evaluation.rerun(tmp_path / "one", "nominal", tmp_path / "changed")


@pytest.mark.parametrize("stage", ["policy", "evaluator"])
def test_mutated_suite_inputs_invalidate_attempt_and_prevent_later_execution(
    tmp_path, programs, fake_process, monkeypatch, stage
):
    launch = evaluation.launch

    def mutate(task, output, *, parameters, **kwargs):
        code = launch(task, output, parameters=parameters, **kwargs)
        if ("episode" in parameters) == (stage == "evaluator"):
            (tmp_path / "suite/inputs/model.dat").write_bytes(b"accidental overwrite")
        return code

    monkeypatch.setattr(evaluation, "launch", mutate)
    report = run(programs, tmp_path / "suite", cases=[case("first"), case("second")])
    assert report["counts"] == dict(completed=0, invalid=2, interrupted=0)
    assert all(any("recorded input changed" in r for r in trial["reasons"]) for trial in report["trials"])
    assert len(fake_process) == (1 if stage == "policy" else 2)


def test_mutated_policy_context_cannot_produce_a_valid_trial(tmp_path, programs, fake_process, monkeypatch):
    skill = tmp_path / "skill.md"
    skill.write_text("Initial guidance")
    launch = evaluation.launch

    def mutate(task, output, *, parameters, **kwargs):
        code = launch(task, output, parameters=parameters, **kwargs)
        Path(parameters["context"]["skill.md"]).write_text("changed during execution")
        return code

    monkeypatch.setattr(evaluation, "launch", mutate)
    report = run(programs, tmp_path / "suite", policy_context={"skill.md": skill})
    assert report["counts"]["invalid"] == 1
    assert "source snapshot changed" in report["trials"][0]["reasons"][0]
    assert len(fake_process) == 1


def test_invalid_interrupted_and_unsuccessful_cases_are_retained(tmp_path, programs, fake_process):
    report = run(programs, tmp_path / "suite", cases=[case("good"), case("bad-start", valid_start=False),
                 case("lost", mode="killed"), case("broken-record", mode="corrupt"), case("failed", mode="failed")])
    assert report["counts"] == dict(completed=2, invalid=2, interrupted=1)
    assert report["trials"][-1]["task"]["verdict"] == "fail"
    assert report["trials"][2]["policy_exit"] == 9
    assert "flight record is incomplete" in report["trials"][3]["reasons"]
    interrupted = run(programs, tmp_path / "interrupted", cases=[case("stop", mode="interrupt"), case("not-started")])
    assert interrupted["counts"]["interrupted"] == 2
    assert not (tmp_path / "interrupted/trials/not-started").exists()


@pytest.mark.parametrize("change", ["missing-chunk", "replay-changed", "missing-flight-close", "recording-error"])
def test_integrity_checks_committed_evidence_not_a_passing_verdict(tmp_path, change):
    evidence(tmp_path)
    assert evaluation.episode_integrity(tmp_path)["complete"]
    if change == "missing-chunk":
        (tmp_path / "physics/000000.npz").unlink()
    elif change == "replay-changed":
        with np.load(tmp_path / "replay.npz") as replay:
            arrays = dict(replay)
        arrays["qpos"][1, 0] = .3
        np.savez_compressed(tmp_path / "replay.npz", **arrays)
    elif change == "missing-flight-close":
        (tmp_path / "flight/complete.json").unlink()
    else:
        saved = json.loads((tmp_path / "task/result.json").read_text())
        saved["final_status"]["physics_recording"]["error"] = "disk full"
        save(tmp_path / "task/result.json", saved)
    assert not evaluation.episode_integrity(tmp_path)["complete"]


def test_stale_validation_and_privileged_comparisons_are_refused(tmp_path, programs, fake_process):
    validation = tmp_path / "validation.json"
    save(validation, dict(format_version=1, scene_revision="b" * 64, status="passed", reference=dict(kind="measured")))
    with pytest.raises(ValueError, match="stale"):
        run(programs, tmp_path / "stale", validation=validation)
    assert not (tmp_path / "stale").exists()
    run(programs, tmp_path / "privileged-one", observation_mode="privileged_diagnostic")
    run(programs, tmp_path / "privileged-two", observation_mode="privileged_diagnostic")
    with pytest.raises(ValueError, match="excluded"):
        evaluation.compare(tmp_path / "privileged-one", tmp_path / "privileged-two")


def test_rerun_pins_source_input_runtime_and_validation(tmp_path, programs, fake_process, monkeypatch):
    report = tmp_path / "source-report.json"
    asset = tmp_path / "evidence.dat"
    asset.write_bytes(b"reference")
    save(report, dict(format_version=1, scene_revision="a" * 64, status="unvalidated", reference=dict(kind="synthetic",
                     observations=[dict(path="evidence.dat", sha256=digest(asset.read_bytes()))])))
    original = run(programs, tmp_path / "one", validation=report)
    repeated = evaluation.rerun(tmp_path / "one", "nominal", tmp_path / "repeat")
    assert original["trials"][0]["case"]["id"] == repeated["trials"][0]["case"]["id"]
    assert (tmp_path / "repeat/validation/evidence.dat").read_bytes() == b"reference"
    monkeypatch.setattr(evaluation, "environment", lambda: dict(python="changed runtime"))
    with pytest.raises(ValueError, match="runtime changed"):
        evaluation.rerun(tmp_path / "one", "nominal", tmp_path / "wrong-runtime")
    (tmp_path / "one/inputs/model.dat").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="input changed"):
        evaluation.rerun(tmp_path / "one", "nominal", tmp_path / "wrong-input")


def test_dwell_rejects_transient_success_simulation_reset_and_observation_gap():
    times = np.arange(0., 2.1, .1)
    true = np.ones(len(times), dtype=bool)
    assert evaluation.held_for(times, true, seconds=2, max_gap_s=.11)
    transient = true.copy()
    transient[-2] = False
    assert not evaluation.held_for(times, transient, seconds=2, max_gap_s=.11)
    assert not evaluation.held_for([0., 2.], [True, True], seconds=2, max_gap_s=.11)
    assert not evaluation.held_for([0., 1., 0., 2.], [True] * 4, seconds=2, max_gap_s=1.1)
    assert not evaluation.held_for([0., 0.], [True, True], seconds=2, max_gap_s=1.1)


def test_frozen_policy_uses_session_and_evaluator_sees_only_completed_episode(tmp_path, programs):
    task, auditor, _ = programs
    task.write_text('''from pathlib import Path
from world_use import World, bodies
from world_use.simulation import SimulationSession
def run(robot, params, record):
    assert Path(params["context"]["skill.md"]).read_text() == "Use current observations"
    truth = World()
    truth.add_box("hidden_truth", "object", [.5, 0, .02], [.03, .03, .04])
    body = bodies.make("sim", truth, q=[.3782, .0015, -.0005, -.0011, .0625, .0051], gripper=1.)
    with SimulationSession(body, Path(params["output"]), cameras={}, sources={"policy.py": Path(__file__)},
                           entrypoint="policy.py") as session:
        assert "hidden_truth" not in session.client.world()["boxes"]
        assert "episode" not in params and "evaluator" not in params
        session.client.enable()
        session.client.run([dict(do="hold", seconds=.1)], wait=10, check=False)
        session.client.release()
        session.close(dict(verdict="inconclusive"))
    return dict(verdict="pass")
''')
    auditor.write_text('''import json
from pathlib import Path
def run(robot, params, record):
    episode = Path(params["episode"])
    assert (episode / "physics/complete.json").exists()
    assert not json.loads((episode / "task/result.json").read_text())["final_status"]["enabled"]
    return dict(verdict="pass", valid_start=True, reasons=[], interventions=[])
''')
    skill = tmp_path / "skill.md"
    skill.write_text("Use current observations")
    report = run(programs, tmp_path / "actual", policy_context={"skill.md": skill})
    assert report["counts"] == dict(completed=1, invalid=0, interrupted=0), report
    assert report["trials"][0]["task"]["verdict"] == "pass"
    assert report["trials"][0]["recording"]["power_off"]
