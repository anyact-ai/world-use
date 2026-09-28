"""The daemon: one process per robot that owns the kernel and outlives every policy session.

A policy's tool calls come and go (and get interrupted); the robot must not. The daemon runs the control loop
in its own thread and serves a small JSON API on localhost, which the CLI, the Python client and the MCP
server all use. It also owns the cameras.

    python -m world_use.daemon --body sim --port 7431

With a simulated body the daemon keeps two worlds: the simulator's truth, and the kernel's model of it. A workcell
box is in both unless it says `known = false`; what a policy adds (`wu box`) goes into the model only.
"""
import argparse
import json
import signal
import sys
import tempfile
import threading
import time
import tomllib
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import bodies, cameras, views
from .behaviors import REGISTRY
from .errors import Refused
from .kernel import Kernel
from .plan import Report, check
from .world import World

DEFAULT_PORT = 7431
MAX_WAIT_S = 120.0


class Daemon:
    def __init__(self, kernel: Kernel, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 cams: dict[str, cameras.Camera] | None = None):
        self.k = kernel
        self.host, self.port = host, port
        self.cameras = dict(cams or {})
        kernel.cameras = self.cameras                    # the card lists them
        self.shots = 0
        self.checked = None                              # the last plan `check` rehearsed: `run --checked` runs it
        self.stop_loop = threading.Event()
        self.done = threading.Event()                    # set once the shutdown reply has gone out
        self.http = ThreadingHTTPServer((host, port), _handler(self))
        self.http.daemon_threads = True
        self.control = threading.Thread(target=self.k.loop, args=(self.stop_loop,), name="control", daemon=True)

    def start(self):
        self.control.start()
        threading.Thread(target=self.http.serve_forever, name="http", daemon=True).start()
        self.k.emit("daemon", f"serving on http://{self.host}:{self.port}")

    def shutdown(self) -> dict:
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
        if method == "GET" and route == ["help"]:
            return 200, dict(steps={kind: cls.help() for kind, cls in REGISTRY.items()})
        if method == "GET" and route == ["events"]:
            since = int(query.get("since", 0))
            events = k.events.wait(since, wait) if wait else k.events.since(since)
            return 200, dict(events=events, last=k.events.seq)
        if method == "GET" and route[0] == "jobs" and len(route) == 2:
            return self._job(int(route[1]), wait)
        if method == "GET" and route == ["world"]:
            return 200, dict(k.world.to_dict(), text=views.world_text(k))
        if method != "POST":
            return 404, dict(error=f"no route {method} /{path.strip('/')}")
        if route == ["run"]:
            if body.get("checked"):
                if self.checked is None:
                    raise Refused("no plan has been checked yet", "spec", "check one first, or give the plan")
                return self._run(self.checked, wait, bool(body.get("check", True)))
            return self._run(body["spec"], wait, bool(body.get("check", True)))
        if route == ["look"]:
            return 200, self.look(body.get("camera"), body.get("spec"))
        if route == ["check"]:
            report = check(body["spec"], k)          # snapshots the kernel under its lock, then runs unlocked
            self.checked = body["spec"]
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
            self._settle()
            return 200, dict(line=views.state_line(k))
        if route == ["release"]:
            k.release()
            self._settle()
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

    def _run(self, spec, wait: float, rehearse: bool) -> tuple[int, dict]:
        """Rehearse on a twin first (when the robot is idle), and refuse the whole plan, with every problem named,
        if the kernel would refuse any step of it. Otherwise run it."""
        k = self.k
        report: Report | None = None
        note = ""
        if rehearse and k.enabled and not k.faulted:
            with k.lock:
                busy = k.active is not None or bool(k.queue)
            if busy:
                note = "not rehearsed: another job is running or queued"
            else:
                report = check(spec, k)
                if report.refused:
                    text = "refused in rehearsal, so nothing moved:\n" + str(report) + "\n" + views.reach_line(k)
                    return 200, dict(id=None, status="refused", incident=text, rehearsal=report.to_dict(),
                                     line=views.state_line(k))
        job = k.submit(spec)
        code, d = self._job(job.id, wait)
        if report is not None:
            d["rehearsal"] = dict(seconds=report.seconds, moving_s=report.moving_s, ok=report.ok)
            if not report.ok:
                d["warning"] = (f"in rehearsal this ended {report.outcome.status}: {report.outcome.message} "
                                "(the world model may be incomplete; running it anyway)")
        elif note:
            d["rehearsal"] = note
        return code, d

    def look(self, camera: str | None = None, spec=None) -> dict:
        """One picture from a camera, with the tool, the known boxes and (given a plan) its path drawn on it, saved
        to the flight record. Returns the file's path: a model reads the image from there."""
        k = self.k
        if not self.cameras:
            raise Refused("no cameras: add [[camera]] entries to the workcell", "no_camera")
        name = camera or next(iter(self.cameras))
        cam = self.cameras.get(name)
        if cam is None:
            raise Refused(f"no camera {name!r}; cameras: {', '.join(self.cameras)}", "no_camera")
        report = check(spec, k) if spec is not None else None
        img = cam.snap(k)
        tool = k.world.from_base("work", k.chain.fk(k.state.q)[:3, 3])
        drawn = ("magenta cross = tool point; green outlines = the boxes the kernel knows; F/L/U = work axes"
                 + ("; blue = the plan's tool path" if report is not None else "")
                 + ("; floor grid: 10 cm squares" if isinstance(cam, cameras.SimCamera) else "")) \
            if cam.view is not None else "no calibration for this camera, so nothing is drawn on it"
        caption = f"{name} | t+{k.clock.now() - k.t0:.0f}s | tool F{tool[0]:+.3f} L{tool[1]:+.3f} U{tool[2]:+.3f}"
        img = cameras.overlay(img, cam.view, k, None if report is None else report.tool_path, caption)
        self.shots += 1
        folder = (k.run_dir or Path(tempfile.gettempdir()) / "world-use") / "views"
        folder.mkdir(parents=True, exist_ok=True)
        path = (folder / f"{self.shots:04d}-{name}.{'png' if isinstance(cam, cameras.SimCamera) else 'jpg'}").resolve()
        img.save(path, quality=88) if path.suffix == ".jpg" else img.save(path)
        k.emit("look", f"{name}: {path.name}", camera=name)
        out = dict(path=str(path), camera=name, size=list(img.size), drawn=drawn)
        if report is not None:
            out["check"] = str(report)
        return out

    def _settle(self):
        """Let the control loop read the body once or twice, so the reply shows the new state."""
        time.sleep(3 * getattr(self.k.clock, "dt", 0.01))

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
        out = {}
        if "box" in body:
            b = dict(body["box"])
            box = w.add_box(b.pop("name"), b.pop("kind"), b.pop("center"), b.pop("size"), b.pop("frame", "work"),
                            b.pop("yaw_deg", 0.0), source=b.pop("source", "policy"), **b)
            self.k.emit("world", f"box {views.box_line(self.k, box)}")
            out["line"] = views.box_line(self.k, box)
        if "remove" in body:
            if w.boxes.pop(body["remove"], None) is None:
                raise KeyError(f"no box {body['remove']!r}; boxes: {sorted(w.boxes)}")
            self.k.emit("world", f"removed box {body['remove']!r}")
            out["line"] = f"removed {body['remove']!r}"
        if "frame" in body:
            f = body["frame"]
            T = np.eye(4)
            T[:3, 3] = f.get("origin", [0, 0, 0])
            if "yaw_deg" in f:
                a = np.radians(f["yaw_deg"])
                T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
            w.add_frame(f["name"], T, f.get("source", "policy"))
        return 200, dict(out, boxes=sorted(w.boxes), facts=sorted(w.facts), frames=sorted(w.frames))


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
            except (OSError, RuntimeError) as e:          # a camera that did not answer, for instance
                code, obj = 502, dict(error=f"{type(e).__name__}: {e}")
            self._reply(code, obj)
            if code == 200 and u.path.strip("/") == "shutdown":
                d.done.set()

        def do_GET(self):
            self._call("GET")

        def do_POST(self):
            self._call("POST")

        def log_message(self, format, *args):
            pass
    return Handler


def _plain(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


WORKCELLS = Path(__file__).parent / "workcells"


def load_workcell(path: Path | None) -> dict:
    """A workcell file (TOML): body, body options, boxes, cameras, facts and operator overrides. Boxes and facts
    are added after connecting, so they may use frames the body defines (like "work"). A bare name ("block")
    means one of the workcells that ship with world-use."""
    if path is None:
        return {}
    path = Path(path)
    if not path.exists() and (WORKCELLS / f"{path.name}.toml").exists():
        path = WORKCELLS / f"{path.name}.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


def apply_workcell(cell: dict, k: Kernel, truth: World | None = None):
    """Boxes, facts and overrides from a workcell. With a simulator's truth world, boxes go there as well, and a box
    marked `known = false` goes only there: part of the scene the policy has to discover."""
    for b in cell.get("box", []):
        b = dict(b)
        known = b.pop("known", True)
        args = (b.pop("name"), b.pop("kind"), b.pop("center"), b.pop("size"), b.pop("frame", "work"),
                b.pop("yaw_deg", 0.0))
        if known:
            k.world.add_box(*args, source="workcell", **b)
        if truth is not None and truth is not k.world:
            truth.add_box(*args, source="workcell", **b)
    for f in cell.get("fact", []):
        k.world.assert_fact(f["key"], f["value"], f.get("source", "workcell"), f.get("note", ""))
    env = cell.get("envelope", {})
    if "max_excursion_deg" in env:
        k.envelope.override("max_excursion", np.radians(env["max_excursion_deg"]),
                            env.get("reason", "set in the workcell file by the operator"))
        why = env.get("reason", "workcell file")
        k.emit("override", f"max excursion {env['max_excursion_deg']} deg: {why}", "warn")


def make_cameras(cell: dict, k: Kernel, body, truth: World | None) -> dict[str, cameras.Camera]:
    """Cameras from the workcell. On a simulator, an entry without url or command renders the simulation from its
    eye/look_at; with no entries at all, a simulator gets three views (side, front, top)."""
    cams: dict[str, cameras.Camera] = {}
    for c in cell.get("camera", []):
        if "url" in c or "command" in c:
            cams[c["name"]] = cameras.from_config(c, k.world)
        elif truth is not None and (view := cameras.view_from_config(c, k.world)) is not None:
            cams[c["name"]] = cameras.SimCamera(c["name"], view, body)
        else:
            raise ValueError(f"camera {c.get('name')!r} needs a url or a command (or, on a simulator, eye and look_at)")
    if not cams and truth is not None:
        cams = cameras.sim_cameras(body, truth)
    return cams


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
    simulated = name.startswith("sim")
    truth = World() if simulated else None            # the simulator's scene; `world` is the kernel's model of it
    options = dict(cell.get("body_options", {}))
    if simulated and "start_deg" in options:
        options["q"] = np.radians(options.pop("start_deg"))
    body = bodies.make(name, truth if simulated else world, **options)
    run_dir = a.runs / f"{datetime.now():%Y%m%d-%H%M%S}-{name.replace(':', '-')}"
    k = Kernel(body, world, run_dir=run_dir)
    k.connect()
    if truth is not None:
        truth.frames.update(world.frames)
    apply_workcell(cell, k, truth)
    if a.enable:
        k.enable()
    d = Daemon(k, port=a.port, cams=make_cameras(cell, k, body, truth))
    d.start()
    print(f"world-use daemon: {k.manifest.name} on http://127.0.0.1:{a.port} (flight record: {run_dir})", flush=True)
    def on_signal(signum, _frame):
        try:
            d.shutdown()
            d.done.set()
        except Refused as e:                  # never exit with torque on away from rest
            k.emit("shutdown_refused", f"{signal.Signals(signum).name} ignored: {e}", "alarm")
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    while not d.done.is_set() and (d.control.is_alive() or d.stop_loop.is_set()):
        d.done.wait(0.5)


if __name__ == "__main__":
    main()
