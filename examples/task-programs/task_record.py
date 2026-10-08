"""Example-local task evidence. One writer, explicit sources, no robot or task scheduler."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout, suppress
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import numpy as np

from world_use.cameras import Frame
from world_use.client import Client


class RecordingError(RuntimeError):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value, *, indent=2) -> bytes:
    return (json.dumps(value, indent=indent, allow_nan=False) + "\n").encode()


def write_new(path: Path, data: bytes):
    """Publish a fully written file without replacing an earlier artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_trace(root: Path) -> list[dict]:
    """Read the committed JSONL prefix after process loss; reject corruption in complete lines."""
    path = Path(root) / "trace.jsonl"
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        lines.pop()
    return [json.loads(line) for line in lines]


def prepare(output: Path, sources: Mapping[str, Path], *, entrypoint: str, parameters: dict,
            url: str | None = None, parent: str | None = None) -> Path:
    """Snapshot declared source files. Does not import task code or connect to a daemon."""
    if entrypoint not in sources:
        raise ValueError("entrypoint must be included in the source bundle")
    payloads = {}
    for name, path in sources.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source names must stay inside the bundle")
        payloads[name] = Path(path).read_bytes()
    # Validate parameters before creating the output folder.
    parameters = json.loads(json_bytes(parameters))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, data in payloads.items():
        write_new(output / "source" / name, data)
        hashes[name] = digest(data)
    import world_use
    package = Path(world_use.__file__).parent
    runtime = {str(p.relative_to(package)): digest(p.read_bytes()) for p in sorted(package.rglob("*.py"))}
    manifest = dict(format_version=1, invocation=uuid4().hex, created_at=datetime.now(UTC).isoformat(),
                    entrypoint=entrypoint, parameters=parameters, url=url, parent=parent,
                    sources=hashes, revision=digest(json_bytes(sorted(hashes.items()))),
                    environment=dict(python=sys.version, platform=platform.platform(),
                                     packages=sorted((d.metadata["Name"], d.version)
                                                     for d in importlib.metadata.distributions()
                                                     if d.metadata["Name"]), world_use_sources=runtime))
    write_new(output / "invocation.json", json_bytes(manifest))
    return output


class Record:
    """A single invocation. Reopening it for execution is deliberately refused."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "invocation.json").read_text())
        if self.manifest["entrypoint"] not in self.manifest["sources"]:
            raise ValueError("entrypoint is missing from the source bundle")
        for name, expected in self.manifest["sources"].items():
            path = (self.root / "source" / name).resolve()
            if not path.is_relative_to(self.root / "source") or digest(path.read_bytes()) != expected:
                raise ValueError(f"source bundle changed: {name}")
        self.start = time.monotonic()
        write_new(self.root / "started.json", json_bytes(dict(pid=os.getpid(), monotonic=self.start,
                                                            utc=datetime.now(UTC).isoformat())))
        self.sequence = 0
        self.error: str | None = None
        self.cell_source: str | None = None
        self.finished = False

    def artifact(self, data: bytes, suffix: str) -> dict:
        try:
            sha = digest(data)
            relative = f"artifacts/{sha}.{suffix}"
            path = self.root / relative
            if not path.exists():
                write_new(path, data)
            elif digest(path.read_bytes()) != sha:
                raise RecordingError(f"artifact changed: {relative}")
            return dict(path=relative, sha256=sha)
        except Exception as exc:
            self.error = str(exc)
            raise RecordingError(self.error) from exc

    def input_file(self, path: str | Path) -> Path:
        """Copy an external input before reading it; return the recorded copy."""
        path = Path(path)
        ref = self.artifact(path.read_bytes(), path.suffix.lstrip(".") or "bin")
        self.append("input", original=str(path.resolve()), artifact=ref)
        return self.root / ref["path"]

    def encode(self, value):
        if isinstance(value, Frame):
            return {"frame": self.artifact(json_bytes(value.to_dict()), "frame.json")}
        if isinstance(value, np.ndarray):
            stream = BytesIO()
            np.save(stream, value, allow_pickle=False)
            return {"array": self.artifact(stream.getvalue(), "npy")}
        if isinstance(value, np.generic):
            return self.encode(value.item())
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            # The transport response contains the exact raw Frame, including depth and calibration.
            if {"png", "id", "camera", "timestamp", "view", "depth", "calibration", "tool"} <= value.keys():
                return {"frame": self.artifact(json_bytes(value), "frame.json")}
            if not all(isinstance(key, str) for key in value):
                raise TypeError("recorded dictionaries need string keys")
            return {key: self.encode(v) for key, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.encode(v) for v in value]
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise TypeError(f"declare serializable inputs/outputs; cannot record {type(value).__name__}")

    def append(self, kind: str, **data) -> int:
        if self.error or self.finished:
            raise RecordingError(self.error or "invocation already finished")
        try:
            encoded = self.encode(data)
            self.sequence += 1
            event = dict(seq=self.sequence, elapsed_s=time.monotonic() - self.start, kind=kind,
                         cell=self.cell_source, **encoded)
            with (self.root / "trace.jsonl").open("ab") as stream:
                stream.write(json_bytes(event, indent=None))
                stream.flush()
                os.fsync(stream.fileno())
            return self.sequence
        except Exception as exc:
            self.error = str(exc)
            raise RecordingError(self.error) from exc

    def invoke(self, kind, inputs, function, *, recovery=False):
        """Record intent before calling, then reply or exception; never retry the function."""
        try:
            call = self.append("intent", operation=kind, inputs=inputs)
        except RecordingError:
            if not recovery:
                raise
            return function()
        try:
            result = function()
        except BaseException as exc:
            # Do not mask the original failure, especially during recovery.
            with suppress(RecordingError):
                self.append("exception", call=call, exception=type(exc).__name__, message=str(exc),
                            traceback=traceback.format_exc(), response=getattr(exc, "body", None))
            raise
        try:
            self.append("reply", call=call, result=result)
        except RecordingError:
            if not recovery:
                raise
        return result

    def call(self, function, /, *args, **kwargs):
        """Record explicit pure-computation inputs/results; no automatic source/dependency discovery."""
        name = f"{function.__module__}.{function.__qualname__}"
        return self.invoke(name, dict(args=args, kwargs=kwargs), lambda: function(*args, **kwargs))

    def cell(self, path: str | Path, namespace: dict, *, inputs: dict | None = None, outputs=()):
        """Execute saved source in the caller's namespace; capture only declared inputs/outputs."""
        source = Path(path).read_bytes()
        ref = self.artifact(source, "py")
        previous = self.cell_source
        self.cell_source = ref["path"]
        cell_number = self.sequence + 1
        stdout = self.root / "cells" / f"{cell_number:06d}.stdout.txt"
        stderr = stdout.with_name(f"{cell_number:06d}.stderr.txt")
        stdout.parent.mkdir(exist_ok=True)

        def execute():
            namespace.update(inputs or {})
            namespace["record"] = self
            with stdout.open("x", buffering=1) as out, stderr.open("x", buffering=1) as err:
                with redirect_stdout(out), redirect_stderr(err):
                    exec(compile(source, str(self.root / ref["path"]), "exec"), namespace)
                return {name: namespace[name] for name in outputs}

        try:
            return self.invoke("cell", dict(source=ref, inputs=inputs or {}, outputs=list(outputs),
                                            stdout=str(stdout.relative_to(self.root)),
                                            stderr=str(stderr.relative_to(self.root))), execute)
        finally:
            self.cell_source = previous

    def finish(self, result: dict | None, *, status: dict | None = None, exception: str | None = None) -> dict:
        if self.finished:
            raise RecordingError("invocation already finished")
        payload = dict(result=result, exception=exception, final_status=status,
                       recording_error=self.error, trace_events=self.sequence,
                       flight_recording_error=None if status is None else status.get("recording", {}).get("error"))
        # A final result is distinct from a completed flight record or a successful task.
        write_new(self.root / "result.json", json_bytes(self.encode(payload)))
        self.finished = True
        return payload


class RecordedClient(Client):
    """Client transport recording, including frames acquired inside Tracking(client, tracker)."""

    def __init__(self, url: str, record: Record):
        super().__init__(url)
        if not record.manifest["url"] or record.manifest["url"].rstrip("/") != self.url:
            raise ValueError("client must match the invocation's simulation endpoint")
        self.recording = record
        self.frames: set[str] = set()
        self.measurements: set[str] = set()
        self.flight: str | None = None

    def begin(self):
        if self.flight is not None:
            raise ValueError("client already began an invocation")
        status = self.status()
        if not status["recording"]["path"] or status["recording"].get("error"):
            raise ValueError("an intact daemon flight record is required")
        if status.get("session", {}).get("mode") != "simulation":
            raise ValueError("this runner requires session.mode=simulation")
        if any(status.get(key) for key in ("enabled", "power_uncertain", "faulted", "job", "queued")):
            raise ValueError("start with an idle, unpowered robot and no unresolved faults")
        if status.get("feedback", {}).get("stale"):
            raise ValueError("current robot feedback is required")
        self.recording.append("begin", status=status, world=self.world(), card=self.capabilities())
        self.flight = status["recording"]["path"]
        self.record(context=dict(invocation=self.recording.manifest["invocation"], phase="begin",
                                 revision=self.recording.manifest["revision"]))
        return status

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        parsed = urlsplit(path)
        route, query = parsed.path, parse_qs(parsed.query)
        request_body = body
        body = body or {}
        frame_id = query.get("id", [None])[0]
        if route == "/frame" and frame_id is not None and frame_id not in self.frames:
            raise ValueError("frame belongs to another invocation; capture a fresh frame")
        if route == "/measure" and body.get("frame") not in self.frames:
            raise ValueError("measure a frame acquired by this invocation")
        if route == "/run":
            for requirement in body.get("requires", []):
                if requirement["evidence"] not in self.measurements:
                    raise ValueError("required evidence belongs to another invocation")
        if route in ("/enable", "/run", "/answer", "/home"):
            status = self.status()
            if self.flight is None or status["recording"]["path"] != self.flight:
                raise ValueError("daemon record changed; begin a new invocation")
        # Recording trouble must not prevent inspection, stopping, or an explicit power recovery.
        recovery = method == "GET" and (route == "/status" or route.startswith("/jobs/"))
        recovery |= route in ("/stop", "/home", "/home_route", "/release", "/shutdown")
        result = self.recording.invoke("http", dict(method=method, path=path, body=body),
                                       lambda: super(RecordedClient, self)._call(method, path, request_body),
                                       recovery=recovery)
        if route == "/frame":
            self.frames.add(result["id"])
        elif route == "/measure":
            self.measurements.add(result["id"])
        elif route == "/withdraw":
            self.measurements.difference_update(body.get("measurements", []))
        return result
