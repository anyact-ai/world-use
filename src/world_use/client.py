"""Python client for the daemon's JSON API (stdlib only). The CLI and the MCP server are thin layers on it."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

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

    def run(self, spec, wait: float = 0.0) -> dict:
        return self._call("POST", "/run", dict(spec=spec, wait=wait))

    def job(self, job_id: int, wait: float = 0.0) -> dict:
        return self._call("GET", f"/jobs/{job_id}?wait={wait}")

    def check(self, spec) -> dict:
        return self._call("POST", "/check", dict(spec=spec))

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

    def home_route(self, steps: list, note: str = "") -> dict:
        return self._call("POST", "/home_route", dict(steps=steps, note=note))

    def home(self, wait: float = 0.0) -> dict:
        return self._call("POST", "/home", dict(wait=wait))

    def world(self, **change) -> dict:
        return self._call("POST", "/world", change) if change else self._call("GET", "/world")

    def shutdown(self) -> dict:
        return self._call("POST", "/shutdown", {})
