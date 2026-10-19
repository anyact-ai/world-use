"""Named simulation cases, frozen task programs and independent outcome checks.

Policy and evaluator are trusted Python programs using ``run(robot, params, record)``.
They execute in separate processes. The policy receives case inputs; only the evaluator
receives the completed episode for privileged inspection. This is not a Python sandbox.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from world_use.records import inspect as inspect_flight
from world_use.tasks import digest, environment, json_bytes, launch, prepare, write_new


def _identity(value) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def held_for(simulation_time, condition, *, seconds: float, max_gap_s: float) -> bool:
    """Whether a sampled condition remained true through the final simulated interval.

    Requires increasing finite simulation times and a bounded observation gap. A true
    final sample, wall-clock delay, or a long gap between two true samples is insufficient.
    """
    times = np.asarray(simulation_time, dtype=float)
    values = np.asarray(condition)
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("seconds and max_gap_s must be finite and positive")
    if times.ndim != 1 or values.shape != times.shape or values.dtype.kind != "b":
        raise ValueError("one boolean condition is required per simulation timestamp")
    if len(times) < 2 or not np.isfinite(times).all() or (np.diff(times) <= 0).any() or not values[-1]:
        return False
    failures = np.flatnonzero(~values)
    start = int(failures[-1] + 1) if len(failures) else 0
    # The observed interval must cover the requested duration, including its start.
    candidates = np.flatnonzero(times[start:] <= times[-1] - seconds + 1e-9)
    if not len(candidates):
        return False
    start += int(candidates[-1])
    return bool((np.diff(times[start:]) <= max_gap_s + 1e-9).all())


def _sources(task: Path, sources: Sequence[Path]) -> dict[str, Path]:
    task = task.resolve()
    paths = [task, *(Path(p).resolve() for p in sources)]
    if any(not p.is_relative_to(task.parent) for p in paths):
        raise ValueError("sources must be inside the entrypoint directory")
    return {str(p.relative_to(task.parent)): p for p in paths}


def _checked_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("artifact names must stay inside their directory")
    return path


def _validation(report: Path | None, output: Path, scene_revision: str) -> dict:
    if report is None:
        return dict(status="unvalidated", scene_revision=scene_revision, reference_kind=None,
                    reason="No reference validation report supplied")
    data = report.read_bytes()
    value = json.loads(data)
    if value.get("format_version") != 1 or value.get("status") not in ("passed", "failed", "unvalidated"):
        raise ValueError("unsupported scene validation report")
    if value.get("scene_revision") != scene_revision:
        raise ValueError("scene validation is stale: scene_revision does not match the evaluated scene")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("scene validation has no artifact manifest; regenerate the report")
    for name, expected in artifacts.items():
        payload = _checked_path(report.parent, name).read_bytes()
        if digest(payload) != expected:
            raise ValueError(f"scene validation evidence changed: {name}")
        write_new(_checked_path(output / "validation", name), payload)
    write_new(output / "validation/scene-validation.json", data)
    return dict(path="validation/scene-validation.json", sha256=digest(data), scene_revision=scene_revision,
                status=value["status"], reference_kind=value.get("reference", {}).get("kind"))


def _definition(output: Path, task: Path, sources: Sequence[Path], context: Mapping[str, Path] | None = None) -> dict:
    bundle = _sources(task, sources)
    context_names = {}
    for name, path in (context or {}).items():
        _checked_path(output / "source/__context__", name)
        relative = f"__context__/{name}"
        if relative in bundle:
            raise ValueError(f"policy context collides with source: {name}")
        bundle[relative] = Path(path)
        context_names[name] = relative
    prepare(output, bundle, entrypoint=task.name, parameters={})
    manifest = _read(output / "invocation.json")
    return dict(path=output.name, entrypoint=manifest["entrypoint"], sources=manifest["sources"],
                revision=manifest["revision"], context=context_names)


def run_suite(task: Path, evaluator: Path, output: Path, *, cases: Sequence[dict],
              inputs: Mapping[str, Path], scene_revision: str, sources: Sequence[Path] = (),
              evaluator_sources: Sequence[Path] = (), validation: Path | None = None,
              observation_mode: str = "camera", parent: str | None = None,
              policy_context: Mapping[str, Path] | None = None) -> dict:
    """Execute a fresh suite, preserving every declared case and attempt.

    Each case has ``name``, ``seed``, ``role`` (development/held_out), and ``parameters``.
    The policy receives those fields plus ``inputs`` (snapshotted files) and ``output``
    (its fresh SimulationSession episode directory). The independent evaluator receives
    ``episode`` and ``case`` and must return verdict, valid_start, reasons, interventions.
    Scene validity, recording integrity, task success and power cleanup remain separate.
    ``policy_context`` snapshots declared prompts/skills as part of the policy revision,
    independently of scene inputs and case identity. Their paths reach the policy as
    ``params["context"]``. Capturing them does not prove the authoring agent used them.
    """
    if observation_mode not in ("camera", "privileged_diagnostic"):
        raise ValueError("observation_mode must be camera or privileged_diagnostic")
    if not re.fullmatch(r"[0-9a-f]{64}", scene_revision):
        raise ValueError("scene_revision must be a SHA-256 identity of the declared scene")
    normalized, names = [], set()
    for case in cases:
        if set(case) != {"name", "seed", "role", "parameters"}:
            raise ValueError("cases require exactly name, seed, role, parameters")
        name = case["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in names:
            raise ValueError("case names must be unique safe directory names")
        if type(case["seed"]) is not int or not 0 <= case["seed"] < 2**64:
            raise ValueError("case seed must be an integer in [0, 2**64)")
        if case["role"] not in ("development", "held_out") or not isinstance(case["parameters"], dict):
            raise ValueError("case needs development/held_out role and object parameters")
        names.add(name)
        normalized.append(json.loads(json_bytes(case)))
    if not normalized:
        raise ValueError("at least one case is required")
    payloads = {name: Path(path).read_bytes() for name, path in inputs.items()}
    output = Path(output).resolve()
    for name in payloads:
        _checked_path(output / "inputs", name)
    # Check stale reports before starting a policy or creating an incomplete suite.
    if validation is not None and _read(validation).get("scene_revision") != scene_revision:
        raise ValueError("scene validation is stale: scene_revision does not match the evaluated scene")
    output.mkdir(parents=True, exist_ok=False)
    policy = _definition(output / "policy", Path(task), sources, policy_context)
    check = _definition(output / "evaluator", Path(evaluator), evaluator_sources)
    input_refs = {}
    for name, data in payloads.items():
        write_new(_checked_path(output / "inputs", name), data)
        input_refs[name] = dict(path=f"inputs/{name}", sha256=digest(data))
    hashes = {name: ref["sha256"] for name, ref in input_refs.items()}
    for case in normalized:
        case["id"] = _identity(dict(case=case, inputs=hashes, scene_revision=scene_revision))
    runtime = _read(output / "policy/invocation.json")["environment"]
    suite = dict(format_version=1, policy=policy, evaluator=check, cases=normalized,
                 inputs=input_refs, scene_revision=scene_revision,
                 scene_validation=_validation(validation, output, scene_revision),
                 runtime=runtime, runtime_id=_identity(runtime), observation_mode=observation_mode,
                 parent=parent)
    write_new(output / "suite.json", json_bytes(suite))
    for case in normalized:
        try:
            _run_case(output, suite, case)
        except KeyboardInterrupt:
            break
    report = summarize(output)
    write_new(output / "summary.json", json_bytes(report))
    return report


def _verify_sources(root: Path, definition: dict):
    source = root / definition["path"] / "source"
    for name, expected in definition["sources"].items():
        if digest(_checked_path(source, name).read_bytes()) != expected:
            raise ValueError(f"source snapshot changed: {name}")


def _verify_inputs(root: Path, suite: dict):
    for name, ref in suite["inputs"].items():
        if digest(_checked_path(root, ref["path"]).read_bytes()) != ref["sha256"]:
            raise ValueError(f"recorded input changed: {name}")


def _launch(root: Path, definition: dict, output: Path, parameters: dict) -> int:
    _verify_sources(root, definition)
    source = root / definition["path"] / "source"
    return launch(source / definition["entrypoint"], output, parameters=parameters,
                  sources=[source / name for name in definition["sources"]], parent=str(root))


def _run_case(root: Path, suite: dict, case: dict):
    trial = root / "trials" / case["name"]
    trial.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    write_new(trial / "started.json", json_bytes(dict(case_id=case["id"], observation_mode=suite["observation_mode"])))
    result: dict[str, Any] = dict(case_id=case["id"], status="interrupted", reasons=[], task=None,
                                  privileged_evaluator=True, policy_exit=None, evaluator_exit=None)
    interrupted = False
    try:
        _verify_inputs(root, suite)
        params = {key: case[key] for key in ("name", "seed", "role", "parameters")}
        params.update(inputs={name: str(root / ref["path"]) for name, ref in suite["inputs"].items()},
                      context={name: str(root / suite["policy"]["path"] / "source" / relative)
                               for name, relative in suite["policy"].get("context", {}).items()},
                      output=str(trial / "episode"), observation_mode=suite["observation_mode"])
        result["policy_exit"] = _launch(root, suite["policy"], trial / "policy", params)
        _verify_inputs(root, suite)
        _verify_sources(root, suite["policy"])
        policy_result = trial / "policy/result.json"
        if not policy_result.exists():
            result["reasons"].append("policy process ended without a terminal record")
        else:
            result["evaluator_exit"] = _launch(root, suite["evaluator"], trial / "evaluator",
                dict(episode=str(trial / "episode"), case=case))
            _verify_inputs(root, suite)
            _verify_sources(root, suite["evaluator"])
            evaluation = trial / "evaluator/result.json"
            if not evaluation.exists():
                result["reasons"].append("evaluator process ended without a terminal record")
            else:
                audit = _read(evaluation)
                outcome = audit.get("result")
                if not isinstance(outcome, dict) or not isinstance(outcome.get("valid_start"), bool):
                    result.update(status="invalid", reasons=["evaluator did not establish a valid initial state"])
                elif not isinstance(outcome.get("reasons"), list) or not isinstance(outcome.get("interventions"), list):
                    result.update(status="invalid", reasons=["evaluator omitted reasons or interventions"])
                else:
                    result["task"] = outcome
                    result["status"] = "completed" if outcome["valid_start"] else "invalid"
                    if not outcome["valid_start"]:
                        result["reasons"].append("invalid initial state")
                if audit.get("exception") or audit.get("recording_error"):
                    result["status"] = "invalid"
                    result["reasons"].append("evaluator failed or its record is incomplete")
        recording = episode_integrity(trial / "episode")
        result["recording"] = recording
        if not recording["complete"] and result["status"] != "interrupted":
            result["status"] = "invalid"
        result["reasons"].extend(recording["reasons"])
        result["simulation_s"] = recording["simulation_s"]
        # A crashing policy cannot be rescued by a passing physical snapshot.
        if policy_result.exists():
            saved = _read(policy_result)
            if saved.get("exception") or saved.get("recording_error"):
                result["status"] = "invalid"
                result["reasons"].append("policy failed or its source/decision record is incomplete")
            elif result["policy_exit"] != 0 and result["task"] is not None:
                result["task"]["verdict"] = "fail"
                result["reasons"].append("policy returned an unsuccessful outcome")
        for name in ("policy", "evaluator"):
            manifest = trial / name / "invocation.json"
            if manifest.exists() and _identity(_read(manifest)["environment"]) != suite["runtime_id"]:
                result["status"] = "invalid"
                result["reasons"].append(f"runtime changed before {name} execution")
    except KeyboardInterrupt:
        interrupted = True
        result["reasons"].append("evaluation interrupted")
    except Exception as exc:
        result.update(status="invalid")
        result["reasons"].append(f"{type(exc).__name__}: {exc}")
    finally:
        result["wall_s"] = time.monotonic() - start
        write_new(trial / "trial.json", json_bytes(result))
    if interrupted:
        raise KeyboardInterrupt


def episode_integrity(episode: Path) -> dict:
    """Inspect session evidence separately from a task's physical success function."""
    episode = Path(episode)
    reasons, simulation_s, power_off = [], None, False
    try:
        task = _read(episode / "task/result.json")
        status = task.get("final_status") or {}
        for error in (task.get("recording_error"), task.get("flight_recording_error"),
                      status.get("recording", {}).get("error"), status.get("physics_recording", {}).get("error")):
            if error:
                reasons.append(str(error))
        power_off = (status.get("enabled") is False and status.get("power_uncertain") is False
                     and not status.get("job") and not status.get("queued"))
        if not power_off:
            reasons.append("power cleanup is not confirmed")
        flight = inspect_flight(episode / "flight")
        if not flight["closed"] or flight["summary"].get("recording_lost"):
            reasons.append("flight record is incomplete")
        complete = _read(episode / "physics/complete.json")
        chunks = sorted((episode / "physics").glob("[0-9]*.npz"))
        if complete.get("error") or complete.get("parts") != len(chunks) or not chunks:
            reasons.append("physics recording is incomplete")
        with np.load(episode / "replay.npz", allow_pickle=False) as replay:
            times = replay["simulation_time"]
            if (len(times) < 2 or not np.isfinite(times).all() or (np.diff(times) <= 0).any()
                    or len(replay["time"]) != len(times) or not np.isfinite(replay["time"]).all()
                    or (np.diff(replay["time"]) <= 0).any()):
                reasons.append("simulation clocks are incomplete or invalid")
            else:
                simulation_s = float(times[-1] - times[0])
            offset = 0
            for path in chunks:
                with np.load(path, allow_pickle=False) as chunk:
                    size = len(chunk["simulation_time"])
                    for key in ("simulation_time", "time", "qpos", "qvel", "state"):
                        if (not np.isfinite(chunk[key]).all()
                                or not np.array_equal(chunk[key], replay[key][offset:offset+size])):
                            reasons.append(f"physics replay differs from committed chunk: {path.name}/{key}")
                    offset += size
            if offset != len(times):
                reasons.append("physics replay row count differs from committed chunks")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        reasons.append(f"missing or invalid session evidence: {exc}")
    return dict(complete=not reasons, reasons=reasons, power_off=power_off, simulation_s=simulation_s)


def summarize(root: Path) -> dict:
    """Read all declared cases, including those lost before a terminal write. No rates are inferred."""
    root = Path(root)
    suite = _read(root / "suite.json")
    rows = []
    for case in suite["cases"]:
        path = root / "trials" / case["name"] / "trial.json"
        trial = _read(path) if path.exists() else dict(status="interrupted", task=None,
                    reasons=["no terminal trial record; execution may not have started"])
        rows.append(dict(case=case, **trial))
    return dict(format_version=1, policy_revision=suite["policy"]["revision"],
                policy_context={name: suite["policy"]["sources"][relative]
                                for name, relative in suite["policy"].get("context", {}).items()},
                evaluator_revision=suite["evaluator"]["revision"], runtime_id=suite["runtime_id"],
                scene_validation=suite["scene_validation"], observation_mode=suite["observation_mode"],
                camera_comparison_eligible=suite["observation_mode"] == "camera", trials=rows,
                counts={status: sum(row["status"] == status for row in rows)
                        for status in ("completed", "invalid", "interrupted")})


def rerun(root: Path, case_name: str, output: Path) -> dict:
    """Recreate one recorded case in a fresh output; refuse changed sources, inputs or runtime."""
    root = Path(root).resolve()
    suite = _read(root / "suite.json")
    cases = [c for c in suite["cases"] if c["name"] == case_name]
    if len(cases) != 1:
        raise ValueError(f"unknown case: {case_name}")
    _verify_inputs(root, suite)
    definitions = []
    for key in ("policy", "evaluator"):
        definition = suite[key]
        source = root / definition["path"] / "source"
        _verify_sources(root, definition)
        context_names = set(definition.get("context", {}).values())
        definitions.append((source / definition["entrypoint"],
                            [source / name for name in definition["sources"] if name not in context_names]))
    if _identity(environment()) != suite["runtime_id"]:
        raise ValueError("runtime changed; restore the recorded versions before rerunning")
    validation = suite["scene_validation"]
    report = root / validation["path"] if "path" in validation else None
    if report is not None and digest(report.read_bytes()) != validation["sha256"]:
        raise ValueError("scene validation report changed")
    case = {key: value for key, value in cases[0].items() if key != "id"}
    return run_suite(definitions[0][0], definitions[1][0], output, cases=[case],
                     inputs={name: root / ref["path"] for name, ref in suite["inputs"].items()},
                     scene_revision=suite["scene_revision"], sources=definitions[0][1],
                     evaluator_sources=definitions[1][1], validation=report,
                     policy_context={name: root / suite["policy"]["path"] / "source" / relative
                                     for name, relative in suite["policy"].get("context", {}).items()},
                     observation_mode=suite["observation_mode"], parent=str(root))


def compare(left: Path, right: Path) -> list[dict]:
    """Pair identical camera cases; keep invalid and interrupted rows visible."""
    a, b = summarize(left), summarize(right)
    for key in ("runtime_id", "evaluator_revision", "observation_mode"):
        if a[key] != b[key]:
            raise ValueError(f"evaluations are not comparable: {key} differs")
    if not a["camera_comparison_eligible"]:
        raise ValueError("privileged diagnostic policies are excluded from camera-only comparisons")
    rows_a = {row["case"]["id"]: row for row in a["trials"]}
    rows_b = {row["case"]["id"]: row for row in b["trials"]}
    if rows_a.keys() != rows_b.keys():
        raise ValueError("evaluations must contain identical named seeded cases and scene inputs")
    return [dict(case_id=key, name=rows_a[key]["case"]["name"], left=rows_a[key], right=rows_b[key]) for key in rows_a]
