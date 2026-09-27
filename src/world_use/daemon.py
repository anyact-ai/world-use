"""The daemon: one process per robot that owns the kernel and outlives every policy session.

A policy's tool calls come and go (and get interrupted); the robot must not. The daemon runs the control loop
in its own thread and serves a small JSON API on localhost, which the CLI, the Python client and the MCP
server all use.

    python -m world_use.daemon --body sim --port 7431
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
import tomllib
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import bodies, views
from .errors import Refused
from .kernel import Kernel
from .plan import check
from .world import World

DEFAULT_PORT = 7431
MAX_WAIT_S = 120.0


class Daemon:
    def __init__(self, kernel: Kernel, host: str = "127.0.0.1", port: int = DEFAULT_PORT):
        self.k = kernel
        self.host, self.port = host, port
        self.stop_loop = threading.Event()
        self.http = ThreadingHTTPServer((host, port), _handler(self))
        self.http.daemon_threads = True
        self.control = threading.Thread(target=self.k.loop, args=(self.stop_loop,), name="control", daemon=True)

    def start(self):
        self.control.start()
        threading.Thread(target=self.http.serve_forever, name="http", daemon=True).start()
        self.k.emit("daemon", f"serving on http://{self.host}:{self.port}")

    def shutdown(self, force: bool = False) -> dict:
        """Stop serving. Refuses while torque is on away from rest: an arm without brakes would drop."""
        k = self.k
        if k.enabled:
            k.release()                       # raises Refused unless idle at rest
        self.stop_loop.set()
        self.control.join(timeout=2.0)
        summary = k.close()
        threading.Thread(target=self.http.shutdown, daemon=True).start()
        return summary

    # -- API ----------------------------------------------------------------------------------------
    def api(self, method: str, path: str, query: dict, body: dict) -> tuple[int, object]:
        k = self.k
        route = path.strip("/").split("/")
        wait = min(float(body.get("wait", query.get("wait", 0)) or 0), MAX_WAIT_S)
        if method == "GET" and route == ["status"]:
            with k.lock:
                return 200, views.status(k)
        if method == "GET" and route == ["card"]:
            return 200, dict(card=views.card(k))
        if method == "GET" and route == ["events"]:
            since = int(query.get("since", 0))
            events = k.events.wait(since, wait) if wait else k.events.since(since)
            return 200, dict(events=events, last=k.events.seq)
        if method == "GET" and route[0] == "jobs" and len(route) == 2:
            return self._job(int(route[1]), wait)
        if method == "GET" and route == ["world"]:
            return 200, k.world.to_dict()
        if method != "POST":
            return 404, dict(error=f"no route {method} /{path.strip('/')}")
        if route == ["run"]:
            job = k.submit(body["spec"])
            return self._job(job.id, wait)
        if route == ["check"]:
            report = check(body["spec"], k)          # snapshots the kernel under its lock, then runs unlocked
            return 200, dict(report.to_dict(), text=str(report))
        if route == ["answer"]:
            k.answer(int(body["job"]), body["answer"])
            return self._job(int(body["job"]), wait)
        if route == ["stop"]:
            k.stop(body.get("reason", "stop requested"))
            time.sleep(3 * k.clock.dt if hasattr(k.clock, "dt") else 0.03)
            return 200, dict(line=views.state_line(k))
        if route == ["enable"]:
            k.enable()
            return 200, dict(line=views.state_line(k))
        if route == ["release"]:
            k.release()
            return 200, dict(line=views.state_line(k))
        if route == ["reset"]:
            k.reset()
            return 200, dict(line=views.state_line(k))
        if route == ["home_route"]:
            k.set_home_route(body.get("steps", []), body.get("note", ""))
            return 200, dict(home=views.status(k)["home"])
        if route == ["home"]:
            job = k.submit({"do": "seq", "steps": k.home_plan(), "label": "home"})
            return self._job(job.id, wait)
        if route == ["world"]:
            return self._world(body)
        if route == ["shutdown"]:
            return 200, dict(summary=self.shutdown())
        return 404, dict(error=f"no route POST /{path.strip('/')}")

    def _job(self, job_id: int, wait: float) -> tuple[int, dict]:
        job = self.k.jobs.get(job_id)
        if job is None:
            return 404, dict(error=f"no job {job_id}")
        if wait and not job.finished and job.status != "waiting":
            job.attention.wait(wait)
        d = job.to_dict()
        d["line"] = views.state_line(self.k)
        if job.outcome is not None and not job.outcome.ok:
            d["incident"] = views.incident(self.k, job)
        return 200, d

    def _world(self, body: dict) -> tuple[int, dict]:
        w = self.k.world
        if "fact" in body:
            f = body["fact"]
            w.assert_fact(f["key"], f["value"], f.get("source", "policy"), f.get("note", ""))
        if "box" in body:
            b = dict(body["box"])
            w.add_box(b.pop("name"), b.pop("kind"), b.pop("center"), b.pop("size"), b.pop("frame", "work"),
                      b.pop("yaw_deg", 0.0), source=b.pop("source", "policy"), **b)
        if "remove" in body:
            w.boxes.pop(body["remove"], None)
        if "frame" in body:
            f = body["frame"]
            T = np.eye(4)
            T[:3, 3] = f.get("origin", [0, 0, 0])
            if "yaw_deg" in f:
                a = np.radians(f["yaw_deg"])
                T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
            w.add_frame(f["name"], T, f.get("source", "policy"))
        return 200, dict(boxes=sorted(w.boxes), facts=sorted(w.facts), frames=sorted(w.frames))


def _handler(d: Daemon):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, obj):
            data = json.dumps(obj, default=_plain).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _call(self, method):
            u = urlparse(self.path)
            query = {k: v[-1] for k, v in parse_qs(u.query).items()}
            body = {}
            if method == "POST":
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except json.JSONDecodeError as e:
                    return self._reply(400, dict(error=f"bad JSON: {e}"))
            try:
                code, obj = d.api(method, u.path, query, body)
            except Refused as e:
                code, obj = 409, dict(refused=e.to_dict())
            except (KeyError, ValueError, TypeError) as e:
                code, obj = 400, dict(error=f"{type(e).__name__}: {e}")
            self._reply(code, obj)

        def do_GET(self):
            self._call("GET")

        def do_POST(self):
            self._call("POST")

        def log_message(self, *args):
            pass
    return Handler


def _plain(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def load_workcell(path: Path | None) -> dict:
    """A workcell file (TOML): body, body options, boxes, facts and operator overrides. Boxes and facts are
    added after connecting, so they may use frames the body defines (like "work")."""
    if path is None:
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def apply_workcell(cell: dict, k: Kernel):
    for b in cell.get("box", []):
        b = dict(b)
        k.world.add_box(b.pop("name"), b.pop("kind"), b.pop("center"), b.pop("size"), b.pop("frame", "work"),
                        b.pop("yaw_deg", 0.0), source="workcell", **b)
    for f in cell.get("fact", []):
        k.world.assert_fact(f["key"], f["value"], f.get("source", "workcell"), f.get("note", ""))
    env = cell.get("envelope", {})
    if "max_excursion_deg" in env:
        k.envelope.override("max_excursion", np.radians(env["max_excursion_deg"]),
                            env.get("reason", "set in the workcell file by the operator"))
        k.emit("override", f"max excursion {env['max_excursion_deg']} deg: {env.get('reason', 'workcell file')}", "warn")


def main(argv=None):
    ap = argparse.ArgumentParser(description="world-use daemon: owns one robot and serves the policy API")
    ap.add_argument("--body", help="sim | sim:<robot> | rebot (default: the workcell's, else sim)")
    ap.add_argument("--workcell", type=Path, help="TOML file with body options, boxes and facts")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--runs", type=Path, default=Path("runs"), help="where flight records go")
    ap.add_argument("--enable", action="store_true", help="switch torque on right away")
    a = ap.parse_args(argv)
    sys.setswitchinterval(0.001)            # keep the control thread's ticks regular while other threads work
    cell = load_workcell(a.workcell)
    name = a.body or cell.get("body", "sim")
    world = World()
    options = dict(cell.get("body_options", {}))
    if name.startswith("sim") and "start_deg" in options:
        options["q"] = np.radians(options.pop("start_deg"))
    body = bodies.make(name, world, **options)
    run_dir = a.runs / f"{datetime.now():%Y%m%d-%H%M%S}-{name.replace(':', '-')}"
    k = Kernel(body, world, run_dir=run_dir)
    k.connect()
    apply_workcell(cell, k)
    if a.enable:
        k.enable()
    d = Daemon(k, port=a.port)
    d.start()
    print(f"world-use daemon: {k.manifest.name} on http://127.0.0.1:{a.port} (flight record: {run_dir})", flush=True)
    done = threading.Event()

    def on_signal(signum, _frame):
        try:
            d.shutdown()
            done.set()
        except Refused as e:                  # never exit with torque on away from rest
            k.emit("shutdown_refused", f"{signal.Signals(signum).name} ignored: {e}", "alarm")
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    while not done.is_set() and d.control.is_alive():
        done.wait(0.5)
        if d.stop_loop.is_set():
            break


if __name__ == "__main__":
    main()
