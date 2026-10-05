"""Python client for the daemon's JSON API (stdlib only). The CLI and the MCP server are thin layers on it."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cameras import Frame
    from .perception import Measurement

DEFAULT_URL = os.environ.get("WORLD_USE_URL", "http://127.0.0.1:7431")


class DaemonError(RuntimeError):
    def __init__(self, code: int, body: dict):
        self.code, self.body = code, body
        msg = body.get("error") or (body.get("refused") or {}).get("message") or json.dumps(body)
        super().__init__(f"{code}: {msg}")


class Client:
    def __init__(self, url: str = DEFAULT_URL, timeout: float = 150.0):
        self.url, self.timeout = url.rstrip("/"), timeout

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise DaemonError(e.code, json.loads(e.read() or b"{}")) from None

    def alive(self) -> bool:
        try:
            self.status()
            return True
        except (urllib.error.URLError, ConnectionError, OSError):
            return False

    def status(self) -> dict:
        return self._call("GET", "/status")

    def card(self) -> str:
        return self._call("GET", "/card")["card"]

    def capabilities(self) -> dict:
        """The embodiment card plus structured cameras, supported effects and limits."""
        return self._call("GET", "/card")

    def run(self, spec=None, wait: float = 0.0, check: bool = True, *, requires: list[dict] | None = None,
            plan_id: str | None = None, request_id: str | None = None) -> dict:
        """Rehearse (unless check=False), then run. A plan the kernel would refuse comes back refused, unmoved.
        A prepared plan fixes data and criteria; it still rehearses against the current state.
        Retrying a prepared ID returns its original job. Literal plans can supply session_id:unique request_id."""
        return self._call("POST", "/run", dict(spec=spec, wait=wait, check=check,
                                               requires=[] if requires is None else requires,
                                               plan_id=plan_id, request_id=request_id))

    def job(self, job_id: int, wait: float = 0.0) -> dict:
        return self._call("GET", f"/jobs/{job_id}?wait={wait}")

    def check(self, spec, *, prepare=False, requires=None, max_age_s=None, effects=None) -> dict:
        return self._call("POST", "/check", dict(spec=spec, prepare=prepare, requires=requires,
                                                  max_age_s=max_age_s, effects=effects))

    def fit_geometry(self, evidence: str, *, kind="known_box", frame="work", size_m=None,
                     max_residual_m=.003) -> dict:
        return self._call("POST", "/fit_geometry", dict(evidence=evidence, kind=kind, frame=frame,
                                                         size_m=size_m, max_residual_m=max_residual_m))

    def verify_effect(self, job: int, after: str, *, effect=0) -> dict:
        return self._call("POST", "/verify_effect", dict(job=job, after=after, effect=effect))

    def invalidate_target(self, target: str) -> dict:
        return self._call("POST", "/invalidate_target", dict(target=target))

    def inspect_run(self, *, since=0, limit=50, job=None) -> dict:
        from urllib.parse import urlencode
        query = dict(since=since, limit=limit)
        if job is not None:
            query["job"] = job
        return self._call("GET", "/inspect?" + urlencode(query))

    def evidence_image(self, evidence: str) -> dict:
        from urllib.parse import urlencode
        return self._call("GET", "/evidence_image?" + urlencode(dict(evidence=evidence)))

    def answer(self, job_id: int, answer: str, wait: float = 0.0) -> dict:
        return self._call("POST", "/answer", dict(job=job_id, answer=answer, wait=wait))

    def stop(self, reason: str = "stop requested") -> dict:
        return self._call("POST", "/stop", dict(reason=reason))

    def events(self, since: int = 0, wait: float = 0.0) -> dict:
        return self._call("GET", f"/events?since={since}&wait={wait}")

    def enable(self) -> dict:
        return self._call("POST", "/enable", {})

    def release(self) -> dict:
        return self._call("POST", "/release", {})

    def reset(self) -> dict:
        return self._call("POST", "/reset", {})

    def home_route(self, steps: list | None, note: str = "") -> dict:
        return self._call("POST", "/home_route", dict(steps=steps, note=note))

    def home(self, wait: float = 0.0, *, request_id: str | None = None) -> dict:
        return self._call("POST", "/home", dict(wait=wait, request_id=request_id))

    def world(self, **change) -> dict:
        return self._call("POST", "/world", change) if change else self._call("GET", "/world")

    def box(self, name: str, kind: str, center, size, **extra) -> dict:
        """Tell the world model about a box (work frame by default): a surface, an object or a zone."""
        return self.world(box=dict(name=name, kind=kind, center=list(center), size=list(size), **extra))

    def remove(self, name: str) -> dict:
        return self.world(remove=name)

    def look(self, camera: str | None = None, spec=None, grid: bool = False) -> dict:
        """Save a picture from a camera (with the plan's path drawn on it, given a spec; with grid, a pixel ruler
        and nothing else); returns its path."""
        return self._call("POST", "/look", dict(camera=camera, spec=spec, grid=grid))

    def frame(self, camera: str | None = None, *, depth: bool = False) -> Frame:
        """Read an unannotated frame in memory, without recording it. Pixels are not downscaled."""
        from urllib.parse import urlencode

        from .cameras import Frame
        params = dict(camera=camera) if camera is not None else {}
        if depth:
            params["depth"] = "true"
        query = "?" + urlencode(params) if params else ""
        return Frame.from_dict(self._call("GET", "/frame" + query))

    def help(self) -> dict:
        return self._call("GET", "/help")["steps"]

    def calibrate(self, camera: str, points: int = 8, spread: float | None = None, wait: float = 0.0,
                  *, request_id: str | None = None) -> dict:
        """Start calibrating a camera from the arm: a job whose checkpoints ask where the tool point is."""
        return self._call("POST", "/calibrate", dict(camera=camera, points=points, spread=spread, wait=wait,
                                                      request_id=request_id))

    def record(self, *, context: dict | None = None, note: str = "", evidence: Measurement | None = None) -> dict:
        """Write the flight record so far, without stopping anything."""
        payload = dict(context=context, note=note)
        if evidence is not None:
            payload["evidence"] = evidence.request()
        return self._call("POST", "/record", payload)

    def shutdown(self) -> dict:
        return self._call("POST", "/shutdown", {})
