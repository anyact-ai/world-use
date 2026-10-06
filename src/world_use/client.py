"""Python client for the daemon's JSON API (stdlib only). The CLI and the MCP server are thin layers on it."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cameras import Frame

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
        except OSError:
            return False

    def status(self) -> dict:
        return self._call("GET", "/status")

    def card(self) -> str:
        return self._call("GET", "/card")["card"]

    def capabilities(self) -> dict:
        """The card's text, plus its cameras: which are calibrated and which give depth."""
        return self._call("GET", "/card")

    def run(self, spec, wait: float = 0.0, check: bool = True, *, requires: list[dict] | None = None) -> dict:
        """Rehearse (unless check=False), then run. A plan the kernel would refuse comes back refused, unmoved.
        Each submission includes its own plan. requires=[{"evidence": measurement ID, "max_age_s": seconds}]
        refuses the run, or its next step, once a measurement is older than that or its camera was recalibrated."""
        return self._call("POST", "/run", dict(spec=spec, wait=wait, check=check, requires=requires or []))

    def job(self, job_id: int, wait: float = 0.0) -> dict:
        return self._call("GET", f"/jobs/{job_id}?wait={wait}")

    def check(self, spec) -> dict:
        return self._call("POST", "/check", dict(spec=spec))

    def answer(self, job_id: int, answer: str, wait: float = 0.0) -> dict:
        return self._call("POST", "/answer", dict(job=job_id, answer=answer, wait=wait))

    def stop(self, reason: str = "stop requested") -> dict:
        return self._call("POST", "/stop", dict(reason=reason))

    def events(self, since: int = 0, wait: float = 0.0, limit: int | None = None) -> dict:
        """Events after number since, oldest first; with a limit, at most that many and whether there are more."""
        return self._call("GET", f"/events?since={since}&wait={wait}" + ("" if limit is None else f"&limit={limit}"))

    def enable(self) -> dict:
        return self._call("POST", "/enable", {})

    def release(self) -> dict:
        return self._call("POST", "/release", {})

    def reset(self) -> dict:
        return self._call("POST", "/reset", {})

    def home_route(self, steps: list | None, note: str = "") -> dict:
        return self._call("POST", "/home_route", dict(steps=steps, note=note))

    def home(self, wait: float = 0.0) -> dict:
        return self._call("POST", "/home", dict(wait=wait))

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

    def frame(self, camera: str | None = None, *, depth: bool = False, id: str | None = None) -> Frame:
        """An unannotated frame at native resolution, not recorded: a new one from a camera (depth=True adds
        aligned metric depth, from simulated cameras), or with id, one the daemon still holds."""
        from urllib.parse import urlencode

        from .cameras import Frame
        if id is not None:
            params = dict(id=id)
        else:
            params = ({} if camera is None else dict(camera=camera)) | (dict(depth="true") if depth else {})
        return Frame.from_dict(self._call("GET", "/frame" + ("?" + urlencode(params) if params else "")))

    def measure(self, frame: Frame | str, *, point=None, box=None, mask=None, target: str | None = None,
                plane: dict | None = None) -> dict:
        """Measure the visible surface under a point [x, y], a box [left, top, right, bottom] (right and bottom
        exclusive) or a boolean mask of a frame (or its id), from its depth. Returns the measurement in work-frame
        metres, with the path of the frame's picture with the measured pixels drawn on it. For a point,
        plane={"box": [...], "max_error_m": ...} projects onto a plane fitted to that visible depth patch."""
        body = dict(frame=frame if isinstance(frame, str) else frame.id, point=point, box=box, target=target,
                    plane=plane)
        if mask is not None:
            import numpy as np

            from .cameras import pack
            body["mask"] = pack(np.packbits(np.asarray(mask, bool)).tobytes())
        return self._call("POST", "/measure", body)

    def withdraw(self, measurements: list[str], reason: str) -> dict:
        """Mark measurements as no longer true (a tracker lost their target): runs that require them stop."""
        return self._call("POST", "/withdraw", dict(measurements=measurements, reason=reason))

    def help(self) -> dict:
        return self._call("GET", "/help")["steps"]

    def calibrate(self, camera: str, points: int = 8, spread: float | None = None, wait: float = 0.0) -> dict:
        """Start calibrating a camera from the arm: a job whose checkpoints ask where the tool point is."""
        return self._call("POST", "/calibrate", dict(camera=camera, points=points, spread=spread, wait=wait))

    def record(self, *, context: dict | None = None, note: str = "") -> dict:
        """Write the flight record so far, without stopping anything."""
        return self._call("POST", "/record", dict(context=context, note=note))

    def shutdown(self) -> dict:
        return self._call("POST", "/shutdown", {})
