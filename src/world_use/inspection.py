"""Shared inspection of saved evidence and an explicitly simulation-only reload owner.

``SimulationWorkspace(path, factory, scene_revision=revision)`` calls
``factory(fresh_output) -> SimulationSession``. ``reset(name)`` rebuilds with the
same factory; ``reload(factory, scene_revision=revision, name=name)`` tries a new
one. Factories own cleanup if construction raises. Both operations require an
idle, confirmed unpowered current session and candidate. Ordinary task source
stays ordinary Python; this is not a plugin loader or a hardware control path.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from .simulation import SimulationSession
from .tasks import digest, read_trace


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def artifact(root: Path, ref: dict) -> Path:
    """Resolve a recorded artifact without escaping its evidence directory or losing its hash."""
    path = (root / ref["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("inspection artifact escapes its evidence directory")
    if ref.get("sha256") and digest(path.read_bytes()) != ref["sha256"]:
        raise ValueError(f"inspection artifact changed: {ref['path']}")
    return path


@dataclass
class Inspection:
    root: Path
    flight: Path | None = None
    episode: Path | None = None
    validation: Path | None = None
    report: dict = field(default_factory=dict)


def inspect(path: Path, *, case: str | None = None, phase: int | None = None) -> Inspection:
    """Read the same revision and evidence for a person or agent. Never connect to a robot."""
    root = Path(path).expanduser().resolve()
    if root.is_file():
        root = root.parent
    result = Inspection(root)
    task = None
    if (root / "inspection.json").exists():
        workspace = _read(root / "inspection.json")
        active = workspace.get("active")
        if active:
            result = inspect(artifact(root, {"path": active["episode"]}), phase=phase)
        elif phase is not None:
            raise ValueError("workspace has no active task to inspect")
        result.root = root
        result.report["workspace"] = workspace
        result.report["scene_revision"] = workspace.get("scene_revision")
        return result
    if (root / "suite.json").exists():
        from .evaluation import summarize
        suite = _read(root / "suite.json")
        result.report = dict(kind="evaluation", path=str(root), scene_revision=suite["scene_revision"],
                             **summarize(root))
        if case is not None:
            matches = [item for item in suite["cases"] if item["name"] == case]
            if len(matches) != 1:
                raise ValueError(f"unknown case: {case}")
            trial = artifact(root, {"path": f"trials/{case}"})
            result.episode = trial / "episode"
            result.report["selected_case"] = case
        validation = suite.get("scene_validation", {})
        if "path" in validation:
            result.validation = artifact(root, validation)
    elif (root / "trial.json").exists() or ((root / "started.json").exists() and root.parent.name == "trials"):
        result = inspect(root.parent.parent, case=root.name, phase=phase)
        result.root = root
        return result
    elif (root / "scene-validation.json").exists():
        result.validation = root / "scene-validation.json"
    elif (root / "flight/session.json").exists():
        result.episode = root
    elif (root / "invocation.json").exists():
        task = root
    elif (root / "session.json").exists():
        result.flight = root
        if (root / "external-scene.json").exists():
            external = _read(root / "external-scene.json")
            # This legacy marker deliberately points one directory up; validate the expected layout.
            if external != {"path": "..", "format": "mujoco-v1"}:
                raise ValueError("unsupported external-scene marker")
            result.episode = root.parent
    else:
        raise ValueError("inspection needs a flight record, task, episode, evaluation suite, validation or workspace")
    if result.episode is not None and (result.episode / "flight/session.json").exists():
        result.flight = result.episode / "flight"
    if result.validation is not None:
        report = _read(result.validation)
        if report.get("format_version") != 1:
            raise ValueError("unsupported scene validation report")
        if result.report.get("scene_revision", report["scene_revision"]) != report["scene_revision"]:
            raise ValueError("scene validation revision does not match inspected scene")
        result.report["validation"] = report
        result.report.setdefault("scene_revision", report["scene_revision"])
    if result.flight is not None:
        from .records import inspect as inspect_record
        result.report["flight"] = inspect_record(result.flight)
    if result.episode is not None and (result.episode / "task/result.json").exists():
        result.report["task"] = _read(result.episode / "task/result.json")
    if result.episode is not None and (result.episode / "task/invocation.json").exists():
        task = result.episode / "task"
    if task is not None:
        trace = task_trace(task, phase=phase)
        if phase is not None and result.flight is not None:
            # Job IDs come from recorded replies/polls; never guess the outcome of a lost submission.
            jobs = result.report["flight"]["jobs"]
            trace["jobs"] = {job: jobs.get(job, {}) for job in trace["job_ids"]}
        result.report["task_trace"] = trace
    elif phase is not None:
        raise ValueError("select an episode or named case with a task trace before selecting a phase")
    result.report.setdefault("path", str(root))
    return result


def task_trace(task: Path, *, phase: int | None = None) -> dict:
    """Index phases; expand one phase's evidence on demand, retaining unknown call outcomes."""
    manifest = _read(task / "invocation.json")
    events = read_trace(task)
    phases = {}
    for event in events:
        if event["kind"] == "phase_begin":
            phases[event["seq"]] = dict(id=event["seq"], parent=event.get("phase"), name=event["name"],
                context=event["context"], start_s=event["elapsed_s"], status="unknown")
        elif event["kind"] == "phase_end" and event.get("phase") in phases:
            phases[event["phase"]].update(status=event["status"], end_s=event["elapsed_s"],
                                          exception=event.get("exception"), message=event.get("message"))
    report = dict(path=str(task), invocation=manifest["invocation"], revision=manifest["revision"],
                  sources=manifest["sources"], phases=list(phases.values()))
    if phase is None:
        return report
    if phase not in phases:
        raise ValueError(f"unknown task phase: {phase}")
    selected = {phase}
    for item in phases.values():
        if item["parent"] in selected:
            selected.add(item["id"])
    selected_events = [e for e in events if e.get("phase") in selected
                       or (e["kind"] == "phase_begin" and e["seq"] in selected)]
    intents = {e["seq"]: e for e in selected_events if e["kind"] == "intent"}
    answered = {e["call"] for e in selected_events if e["kind"] in ("reply", "exception")}
    job_ids = set()
    for event in selected_events:
        intent = intents.get(event.get("call"), {})
        request = intent.get("inputs", {})
        if intent.get("operation") != "http" or event["kind"] != "reply":
            continue
        route = request.get("path", "").split("?")[0]
        if route in ("/run", "/answer", "/home", "/calibrate") or route.startswith("/jobs/"):
            job = event.get("result", {}).get("id")
            if type(job) is int:
                job_ids.add(job)
    artifacts = {}

    def collect(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                artifact(task, value)
                artifacts[value["path"]] = value
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(selected_events)
    report.update(selected_phase=phase, events=selected_events, artifacts=list(artifacts.values()),
                  unanswered_calls=sorted(intents.keys() - answered), job_ids=sorted(job_ids))
    return report


def frames(episode: Path, *, seen: set[str] | None = None):
    """Exact policy frame replies, on their own capture-time clock, not annotated look images."""
    from .cameras import Frame
    task = episode / "task"
    if not (task / "started.json").exists():
        return
    start = _read(task / "started.json")["monotonic"]
    if seen is None:
        seen = set()

    def refs(value):
        if isinstance(value, dict):
            if set(value) == {"frame"}:
                yield value["frame"]
            else:
                for item in value.values():
                    yield from refs(item)
        elif isinstance(value, list):
            for item in value:
                yield from refs(item)

    for event in read_trace(task):
        for ref in refs(event):
            if ref["path"] in seen:
                continue
            frame = Frame.from_dict(_read(artifact(task, ref)))
            seen.add(ref["path"])
            yield frame.timestamp - start, frame


class SimulationWorkspace:
    """One replaceable, independently recorded simulation; failed builds retain the active scene.

    ``wu view workspace --follow`` watches the saved pointer. The viewer remains
    read-only; reset/reload are explicit Python calls made by the scene owner.
    Do not issue concurrent control calls while resetting or reloading.
    """

    def __init__(self, path: Path, factory: Callable[[Path], SimulationSession], *, scene_revision: str):
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=False)
        self.session: SimulationSession | None = None
        self.factory = factory
        self.scene_revision: str | None = None
        self.error: str | None = None
        self.reload(factory, scene_revision=scene_revision, name="initial")

    @staticmethod
    def _ready(session: SimulationSession):
        if not isinstance(session, SimulationSession) or not getattr(session.body, "simulated", False):
            raise ValueError("inspection reset/reload accepts SimulationSession only")
        if session.closed:
            return
        status = session.client.status()
        if status.get("session", {}).get("mode") != "simulation":
            raise ValueError("inspection reset/reload cannot operate hardware")
        if status.get("enabled") is not False or status.get("power_uncertain") is not False or any(
            status.get(key) for key in ("job", "queued")
        ):
            raise ValueError("stop work and confirm power off before resetting or reloading simulation")

    def _publish(self):
        value = dict(format_version=1, scene_revision=self.scene_revision, error=self.error,
                     active=None if self.session is None else
                     dict(episode=str(self.session.output.relative_to(self.path)), closed=self.session.closed))
        temporary = self.path / f".inspection-{uuid4().hex}.json"
        try:
            temporary.write_text(json.dumps(value, indent=2) + "\n")
            temporary.replace(self.path / "inspection.json")
        finally:
            temporary.unlink(missing_ok=True)

    def reload(self, factory: Callable[[Path], SimulationSession], *, scene_revision: str,
               name: str = "reload") -> SimulationSession:
        """Build first, validate both sides, then retire the old episode. No implicit motion or enable."""
        if not re.fullmatch(r"[0-9a-f]{64}", scene_revision):
            raise ValueError("scene_revision must be a SHA-256 scene identity")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError("name must be a safe episode name")
        candidate = None
        previous = self.session, self.factory, self.scene_revision
        try:
            if self.session is not None:
                self._ready(self.session)
            output = self.path / "episodes" / f"{name}-{uuid4().hex[:12]}"
            candidate = factory(output)
            self._ready(candidate)
            if candidate.closed or candidate.output != output:
                raise ValueError("factory must return a fresh open session at the requested output")
            if self.session is not None:
                self._ready(self.session)
                if not self.session.closed:
                    self.session.close({"reason": f"inspection {name}"})
            self.session, self.factory, self.scene_revision = candidate, factory, scene_revision
            self.error = None
            self._publish()
            return candidate
        except BaseException as exc:
            self.session, self.factory, self.scene_revision = previous
            self.error = f"{type(exc).__name__}: {exc}"
            if isinstance(candidate, SimulationSession) and candidate is not self.session and not candidate.closed:
                try:
                    if not getattr(candidate.body, "simulated", False) or (
                        candidate.client.status().get("session", {}).get("mode") != "simulation"
                    ):
                        raise ValueError("candidate identity is not verified as simulation; no control request sent")
                    candidate.client.stop("rejected inspection candidate")
                    candidate.client.release()
                    candidate.close(exception=self.error)
                except Exception as cleanup:
                    self.error += f"; candidate cleanup failed at {candidate.output}: {cleanup}"
            try:
                self._publish()
            except OSError as publication:
                exc.add_note(f"inspection error could not be saved: {publication}")
            raise

    def reset(self, name: str = "reset") -> SimulationSession:
        """Rebuild a fresh kernel, measurements, clock and record under the same declared revision."""
        assert self.scene_revision is not None
        return self.reload(self.factory, scene_revision=self.scene_revision, name=name)

    def close(self):
        if self.session is not None and not self.session.closed:
            self._ready(self.session)
            self.session.close({"reason": "inspection closed"})
        self._publish()
