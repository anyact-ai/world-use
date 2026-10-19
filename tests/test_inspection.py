"""Shared evidence inspection, fresh simulation lifecycle and retained failed-build evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from world_use import World, bodies
from world_use.cameras import Frame
from world_use.config import load_robot, manifest_data
from world_use.inspection import SimulationWorkspace, frames, inspect
from world_use.simulation import SimulationSession
from world_use.tasks import Record, digest, prepare
from world_use.visualization import view


@pytest.fixture
def episode(tmp_path):
    folder = tmp_path / "episode"
    flight = folder / "flight"
    flight.mkdir(parents=True)
    manifest = load_robot(Path(__file__).parents[1] / "examples/adapters/planar.toml")
    (flight / "robot.urdf").write_bytes(manifest.urdf.read_bytes())
    (flight / "session.json").write_text(json.dumps(dict(
        body="test arm", mode="simulation", package_version="test", initial=dict(
            model=manifest_data(manifest) | {"urdf": "robot.urdf"}, world=World().to_dict(),
            q=[0, 0], q_start=[0, 0], gripper=None))))
    return folder


def validation(folder):
    folder.mkdir(parents=True)
    image = folder / "comparison.png"
    Image.new("RGB", (20, 10), "orange").save(image)
    report = dict(format_version=1, scene_revision="a" * 64, status="unvalidated", groups={},
                  reference=dict(kind="synthetic", observations=dict(one=dict(
                      kind="synthetic", split="held_out", images=dict(comparison=dict(
                          path=image.name, sha256=digest(image.read_bytes())))))))
    (folder / "scene-validation.json").write_text(json.dumps(report))
    return report


def rrd_rows(output, entity, component):
    from rerun.chunk import RrdReader
    rows = []
    for chunk in RrdReader(output).stream():
        if chunk.entity_path == entity:
            batch = chunk.to_record_batch()
            if component in batch.schema.names:
                rows.extend(batch.column(component).to_pylist())
    return rows


def test_standalone_validation_exports_labelled_comparison_without_a_robot(tmp_path, monkeypatch):
    rr = pytest.importorskip("rerun")
    rr.set_strict_mode(True)
    folder = tmp_path / "validation"
    validation(folder)
    monkeypatch.setattr(bodies, "make", lambda *a, **kw: pytest.fail("inspection opened a robot"))
    output = view(folder, output=tmp_path / "validation.rrd")
    assert rrd_rows(output, "/evidence/comparison/one", "EncodedImage:blob")
    report = rrd_rows(output, "/evidence/report", "TextDocument:text")
    assert any("synthetic" in row[0] and "held_out" in row[0] and "a" * 64 in row[0] for row in report)
    (folder / "comparison.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        view(folder, output=tmp_path / "changed.rrd")


def test_raw_policy_frames_keep_capture_clock_pixels_and_hash(episode):
    pytest.importorskip("rerun")
    task = prepare(episode / "task", {"test.py": Path(__file__)}, entrypoint="test.py", parameters={})
    record = Record(task)
    frame = Frame(Image.new("RGB", (10, 8), "red"), "side", timestamp=record.start + .25,
                  depth=np.full((8, 10), .4, dtype=np.float32))
    record.append("reply", result=frame)
    captured = list(frames(episode))
    assert captured[0][0] == pytest.approx(.25)
    assert captured[0][1].image.getpixel((0, 0)) == (255, 0, 0)
    assert captured[0][1].depth[0, 0] == pytest.approx(.4)
    output = view(episode, output=episode / "view.rrd")
    assert rrd_rows(output, "/evidence/raw/side", "Image:buffer")
    from rerun.chunk import RrdReader
    assert any("policy_elapsed" in chunk.to_record_batch().schema.names
               for chunk in RrdReader(output).stream() if chunk.entity_path == "/evidence/raw/side")


def test_phase_inspection_connects_observations_calls_and_job_outcomes(episode, monkeypatch, capsys):
    from world_use import cli

    task = prepare(episode / "task", {"test.py": Path(__file__)}, entrypoint="test.py", parameters={})
    record = Record(task)
    frame = Frame(Image.new("RGB", (10, 8), "red"), "side")
    with record.phase("place", context={"object": "block"}) as placement:
        record.invoke("http", dict(method="GET", path="/frame"), lambda: frame)
        with record.phase("verify") as verification:
            record.invoke("http", dict(method="GET", path="/jobs/7?wait=0"),
                          lambda: dict(id=7, status="failed"))
        lost = record.append("intent", operation="http", inputs=dict(method="POST", path="/run", body={}))
    with record.phase("unrelated"):
        record.append("note", text="not part of placement")
    (episode / "flight/events.jsonl").write_text(json.dumps(dict(kind="finished", level="info", t=1.,
        data=dict(job=7, status="failed", outcome=dict(status="failed", message="grip lost")))) + "\n")
    monkeypatch.setattr(bodies, "make", lambda *a, **kw: pytest.fail("inspection opened a robot"))
    overview = inspect(episode).report["task_trace"]
    assert "events" not in overview
    assert {p["name"] for p in overview["phases"]} == {"place", "verify", "unrelated"}
    detail = inspect(episode, phase=placement).report["task_trace"]
    assert detail["jobs"][7]["outcome"]["message"] == "grip lost"
    assert detail["unanswered_calls"] == [lost]
    assert any(e.get("phase") == verification for e in detail["events"])
    assert not any(e.get("text") == "not part of placement" for e in detail["events"])
    assert len(detail["artifacts"]) == 1
    assert (task / detail["artifacts"][0]["path"]).is_file()
    assert cli.main(["inspect", str(episode), "--phase", str(placement)]) == 0
    assert json.loads(capsys.readouterr().out)["task_trace"]["selected_phase"] == placement
    assert inspect(task, phase=verification).report["task_trace"]["job_ids"] == [7]
    with pytest.raises(ValueError, match="unknown task phase"):
        inspect(episode, phase=999)
    (task / detail["artifacts"][0]["path"]).write_text("changed evidence")
    inspect(episode, phase=verification)  # A different phase does not load the damaged frame.
    with pytest.raises(ValueError, match="artifact changed"):
        inspect(episode, phase=placement)


def suite(folder):
    folder.mkdir()
    value = dict(format_version=1, scene_revision="b" * 64, policy={"revision": "policy"},
                 evaluator={"revision": "audit"}, runtime_id="runtime", observation_mode="camera",
                 scene_validation={"status": "unvalidated"}, cases=[dict(name="never-started", id="case",
                     seed=3, role="held_out", parameters={})])
    (folder / "suite.json").write_text(json.dumps(value))
    return value


def test_unstarted_named_case_remains_visible_and_cannot_escape_suite(tmp_path):
    pytest.importorskip("rerun")
    root = tmp_path / "suite"
    suite(root)
    selected = inspect(root, case="never-started")
    assert selected.flight is None
    assert selected.report["trials"][0]["status"] == "interrupted"
    output = view(root, case="never-started", output=tmp_path / "unstarted.rrd")
    assert any("never-started" in row[0] for row in rrd_rows(output, "/evidence/report", "TextDocument:text"))
    with pytest.raises(ValueError, match="unknown case"):
        inspect(root, case="../../other")


def session_at(path):
    body = bodies.make("sim", World(), q=[.3782, .0015, -.0005, -.0011, .0625, .0051], gripper=1.)
    return SimulationSession(body, path, cameras={}, sources={"test.py": Path(__file__)}, entrypoint="test.py")


def test_live_reset_requires_power_off_and_failed_reload_keeps_current_revision(tmp_path):
    workspace = SimulationWorkspace(tmp_path / "workspace", session_at, scene_revision="a" * 64)
    first = workspace.session
    assert first is not None
    try:
        first.client.world(fact=dict(key="old", value=1, source="test"))
        first.client.enable()
        with pytest.raises(ValueError, match="power off"):
            workspace.reset()
        assert workspace.session is first and not first.closed
        first.client.release()

        def broken(_):
            raise KeyboardInterrupt("bad scene build")

        with pytest.raises(KeyboardInterrupt):
            workspace.reload(broken, scene_revision="b" * 64)
        saved = json.loads((workspace.path / "inspection.json").read_text())
        assert saved["scene_revision"] == "a" * 64 and "bad scene build" in saved["error"]
        assert workspace.session is first and not first.closed
        candidates = []

        def powered(path):
            candidate = session_at(path)
            candidates.append(candidate)
            candidate.client.enable()
            return candidate

        with pytest.raises(ValueError, match="power off"):
            workspace.reload(powered, scene_revision="b" * 64)
        assert candidates[0].closed and workspace.session is first and not first.closed
        second = workspace.reset("fresh")
        assert first.closed and second.kernel is not first.kernel
        assert "old" not in second.client.world()["facts"]
        assert inspect(workspace.path).report["scene_revision"] == "a" * 64
    finally:
        workspace.close()


def test_hardware_mode_candidate_is_never_sent_control_requests(tmp_path, monkeypatch):
    from world_use import inspection
    mutations = []

    class Session:
        def __init__(self, path, mode):
            self.output, self.closed, self.body = path, False, SimpleNamespace(simulated=True)
            self.client = SimpleNamespace(status=lambda: dict(session={"mode": mode}, enabled=False,
                                           power_uncertain=False), stop=lambda *a: mutations.append("stop"),
                                           release=lambda: mutations.append("release"))

        def close(self, *a, **kw):
            mutations.append("close")
            self.closed = True

    monkeypatch.setattr(inspection, "SimulationSession", Session)
    workspace = SimulationWorkspace(tmp_path / "workspace", lambda p: Session(p, "simulation"), scene_revision="a" * 64)
    old = workspace.session
    with pytest.raises(ValueError, match="hardware"):
        workspace.reload(lambda p: Session(p, "hardware"), scene_revision="b" * 64)
    assert workspace.session is old and mutations == []
    assert "no control request sent" in json.loads((workspace.path / "inspection.json").read_text())["error"]


def test_follow_keeps_last_valid_scene_and_displays_failed_build(tmp_path, monkeypatch):
    pytest.importorskip("rerun")
    from world_use import visualization
    workspace = tmp_path / "workspace"
    validation(workspace / "episodes/old")
    pointer = dict(format_version=1, scene_revision="a" * 64, error=None,
                   active=dict(episode="episodes/old", closed=False))
    (workspace / "inspection.json").write_text(json.dumps(pointer))
    output = tmp_path / "follow.rrd"
    monkeypatch.setattr(visualization._sdk(), "spawn", lambda **kw: kw["recording"].save(output))
    calls = 0

    def change(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            pointer["error"] = "ValueError: bad scene build"
            (workspace / "inspection.json").write_text(json.dumps(pointer))
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr(visualization.time, "sleep", change)
    view(workspace, follow=True)
    assert calls == 2
    assert rrd_rows(output, "/evidence/comparison/one", "EncodedImage:blob")
    assert any("bad scene build" in row[0] for row in rrd_rows(output, "/evidence/report", "TextDocument:text"))


@pytest.mark.parametrize("initial", [True, False])
def test_pointer_write_failure_cleans_candidate_and_retains_previous_identity(tmp_path, monkeypatch, initial):
    from world_use import inspection
    sessions = []

    class Session:
        def __init__(self, output):
            self.output, self.closed, self.body = output, False, SimpleNamespace(simulated=True)
            self.client = SimpleNamespace(status=lambda: dict(session={"mode": "simulation"}, enabled=False,
                                           power_uncertain=False), stop=lambda *a: None, release=lambda: None)
            sessions.append(self)

        def close(self, *a, **kw):
            self.closed = True

    monkeypatch.setattr(inspection, "SimulationSession", Session)
    workspace = None if initial else SimulationWorkspace(tmp_path / "workspace", Session, scene_revision="a" * 64)
    previous = sessions[-1] if sessions else None

    def fail(_):
        raise OSError("disk full")

    monkeypatch.setattr(SimulationWorkspace, "_publish", fail)
    with pytest.raises(OSError, match="disk full"):
        if initial:
            SimulationWorkspace(tmp_path / "workspace", Session, scene_revision="b" * 64)
        else:
            workspace.reload(Session, scene_revision="b" * 64)
    assert sessions[-1].closed
    if workspace is not None:
        assert workspace.session is previous and workspace.scene_revision == "a" * 64
        assert json.loads((workspace.path / "inspection.json").read_text())["scene_revision"] == "a" * 64


def test_named_cli_rerun_reports_completed_failure_and_allows_unstarted_case(tmp_path, monkeypatch, capsys):
    from world_use import cli, evaluation
    root = tmp_path / "suite"
    suite(root)
    called = []

    def rerun(source, case, output):
        called.append((source, case, output))
        return {"trials": [dict(status="completed", task={"verdict": "fail"})]}

    monkeypatch.setattr(evaluation, "rerun", rerun)
    output = tmp_path / "rerun"
    assert cli.main(["rerun", str(root), "never-started", "--out", str(output)]) == 4
    assert called == [(root, "never-started", output)]
    assert json.loads(capsys.readouterr().out)["trials"][0]["task"]["verdict"] == "fail"
