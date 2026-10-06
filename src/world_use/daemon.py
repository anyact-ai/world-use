"""The daemon: one process per robot that owns the kernel and outlives every policy session.

A policy's tool calls come and go (and get interrupted); the robot must not. The daemon runs the control loop
in its own thread and serves a small JSON API on localhost, which the CLI, the Python client and the MCP
server all use. It also owns the cameras.

    python -m world_use.daemon --body sim --port 7431

With a simulated body the daemon keeps two worlds: the simulator's truth, and the kernel's model of it. A workcell
box is in both unless it says `known = false`; what a policy adds (`wu box`) goes into the model only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import tempfile
import threading
import time
import traceback
from copy import deepcopy
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import bodies, calibrate, cameras, views
from .behaviors import REGISTRY, build
from .config import _vector, load_robot, load_workcell
from .errors import Refused, explain
from .events import _plain
from .fit import load as load_fit
from .kernel import Kernel
from .perception import Measurements
from .plan import Report
from .request_models import WorldChange, http_request
from .worker import Rehearser
from .world import World

DEFAULT_PORT = 7431
MAX_WAIT_S = 120.0


class Daemon:
    def __init__(self, kernel: Kernel, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 cams: dict[str, cameras.Camera] | None = None, rehearser: Rehearser | None = None, *,
                 session: dict, config: dict | None = None):
        """session is the startup identity from session_identity(): `wu up` reuses a daemon only when it matches."""
        self.k = kernel
        self.host, self.port = host, port
        self.session = session
        self.cameras = dict(cams or {})
        kernel.cameras = self.cameras                    # the card lists them
        self.measurements = Measurements(kernel, self.cameras)
        kernel.record_session(session=self.session, config=config or {})
        self.shots = 0
        self.calibrations: dict[int, dict] = {}         # job id -> camera, picture size and, once solved, the result
        self._solving = threading.Lock()
        self.stop_loop = threading.Event()
        self.done = threading.Event()                    # set once the shutdown reply has gone out
        self._closing = threading.Lock()                 # held once a shutdown has begun
        self.http = ThreadingHTTPServer((host, port), _handler(self))
        self.http.daemon_threads = True
        # rehearsals and reach probes run in a worker process: in this one they took the control loop's ticks
        self.rehearser = rehearser if rehearser is not None else Rehearser()
        self._own_rehearser = rehearser is None
        kernel.planner = self.rehearser
        self.control = threading.Thread(target=self.k.loop, args=(self.stop_loop,), name="control", daemon=True)

    def start(self):
        self.control.start()
        threading.Thread(target=self.http.serve_forever, name="http", daemon=True).start()
        self.k.emit("daemon", f"serving on http://{self.host}:{self.port}")

    def shutdown(self) -> dict:
        """Stop serving. Refuses while torque is on away from rest: an arm without brakes would drop. Runs once:
        a second Ctrl+C during the release ramp must not start another."""
        if not self._closing.acquire(blocking=False):
            raise Refused("already shutting down", "busy")
        try:
            k = self.k
            k.begin_shutdown()                # closes queued admission before releasing on the control thread
            self.stop_loop.set()
            self.control.join(timeout=2.0)
            if self.control.is_alive():
                raise RuntimeError("power is off, but the control loop has not stopped; retry shutdown")
            summary = k.close()
            if self._own_rehearser:
                self.rehearser.close()
        except BaseException:
            self._closing.release()           # not down after all: a later shutdown may try again
            raise
        threading.Thread(target=self.http.shutdown, daemon=True).start()
        return summary

    # -- API ----------------------------------------------------------------------------------------
    def api(self, method: str, path: str, query: dict, body: dict) -> tuple[int, object]:
        k = self.k
        route = path.strip("/").split("/")
        query, body = http_request(method, route, query, body)
        wait = min(body.get("wait", query.get("wait", 0.0)), MAX_WAIT_S)
        if method == "GET" and route == ["status"]:
            with k.lock:
                status = views.status(k)
                status.update(session=self.session, line=f"{self.session['adapter']} | {status['line']}")
                return 200, status
        if method == "GET" and route == ["card"]:
            # The worker probes reach before the lock is taken: the control loop must never wait for it.
            reach = self.rehearser.reach_line(k) if k.enabled and k.active is None else ""
            with k.lock:
                card = views.card(k, reach=lambda _: reach)
            return 200, dict(card=card, session=self.session,
                             cameras={name: dict(depth=isinstance(cam, cameras.SimCamera),
                                                 calibrated=cam.view is not None)
                                      for name, cam in self.cameras.items()},
                             collision_coverage=dict(checked=["padded_link_segments", "tool_point_surfaces"],
                                 unchecked=["fingers", "payload", "pedestal", "self_collision"]))
        if method == "GET" and route == ["help"]:
            from .validation import FIELDS, schema
            return 200, dict(steps={kind: dict(cls.help(), schema=schema(kind) if kind in FIELDS else None)
                                   for kind, cls in REGISTRY.items()})
        if method == "GET" and route == ["events"]:
            since = query.get("since", 0)
            limit = query.get("limit")
            events = k.events.wait(since, wait) if wait else k.events.since(since)
            more = limit is not None and len(events) > limit
            events = events[:limit]
            return 200, dict(events=events, last=events[-1]["seq"] if events else since,
                             missed=max(0, events[0]["seq"] - since - 1) if events else 0, more=more)
        if method == "GET" and route[0] == "jobs" and len(route) == 2:
            return self._job(int(route[1]), wait)
        if method == "GET" and route == ["world"]:
            with k.lock:
                return 200, dict(k.world.to_dict(), text=views.world_text(k))
        if method == "GET" and route == ["frame"]:
            if "id" in query:
                return 200, self.measurements.frame(query["id"]).to_dict()
            _, cam = self._camera(query.get("camera"))
            if query.get("depth", False):
                if not isinstance(cam, cameras.SimCamera):
                    raise Refused(f"camera {cam.name!r} has no aligned metric depth", "missing_depth",
                                  "depth comes from simulated (MuJoCo) cameras")
                frame = cam.capture(k, depth=True)
            else:
                frame = cam.capture(k)
            return 200, self.measurements.keep(frame).to_dict()
        if method != "POST":
            return 404, dict(error=f"no route {method} /{path.strip('/')}")
        if route == ["run"]:
            return self._run(body["spec"], wait, body.get("check", True), body.get("requires"))
        if route == ["look"]:
            return 200, self.look(body.get("camera"), body.get("spec"), body.get("grid", False))
        if route == ["check"]:
            report = self.rehearser.check(body["spec"], k)
            return 200, dict(report.to_dict(), text=str(report))
        if route == ["answer"]:
            k.answer(body["job"], body["answer"])
            return self._job(body["job"], wait)
        if route == ["stop"]:
            k.stop(body.get("reason", "stop requested"))
            self._settle()
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
            self._settle()                    # a fault still present latches again on the next tick
            return 200, dict(line=views.state_line(k))
        if route == ["home_route"]:
            if "steps" not in body:
                raise ValueError("home_route needs steps: a list of moves, [] to fold straight home from here, "
                                 "or null to clear the route")
            return 200, self._home_route(body["steps"], body.get("note", ""))
        if route == ["home"]:
            return self._job(k.submit({"do": "seq", "steps": k.home_plan(), "label": "home"}).id, wait)
        if route == ["world"]:
            with k.lock:
                return self._world(body)
        if route == ["calibrate"]:
            return self.calibrate(body["camera"], body.get("points", 8), body.get("spread"), wait)
        if route == ["measure"]:
            return 200, self.measurements.measure(body["frame"], point=body.get("point"), box=body.get("box"),
                                                  mask=body.get("mask"), target=body.get("target"),
                                                  plane=body.get("plane"))
        if route == ["withdraw"]:
            return 200, dict(withdrawn=self.measurements.withdraw(body["measurements"], body.get("reason", "")))
        if route == ["record"]:
            if body.get("context") or body.get("note"):
                k.emit("annotation", body.get("note", "agent context"), context=body.get("context", {}))
            return 200, dict(summary=k.save_record(), run=str(k.run_dir) if k.run_dir else None)
        if route == ["shutdown"]:
            return 200, dict(summary=self.shutdown(), run=None if k.run_dir is None else str(k.run_dir))
        return 404, dict(error=f"no route POST /{path.strip('/')}")

    def _run(self, spec, wait: float, rehearse: bool, requires=None) -> tuple[int, dict]:
        """Rehearse an idle snapshot, then admit only while that snapshot is still current. Required measurements
        are checked before rehearsal, at admission and before each step, with or without rehearsal."""
        build(spec)                  # malformed plans are request errors, before rehearsal or queueing
        k = self.k
        guard = self.measurements.guard(requires or [])
        if guard is not None:
            guard()
        report: Report | None = None
        admission = None
        if rehearse:
            snap, admission = k.checked_start()
            report = self.rehearser.check(spec, k, snap=snap)
            if report.refused or report.outcome.status not in ("done", "surprise"):
                text = ("refused in rehearsal, so nothing moved:\n" + str(report) + "\n"
                        + self.rehearser.reach_line(k))
                return 200, dict(id=None, status="refused", incident=text, rehearsal=report.to_dict(),
                                 line=views.state_line(k))
        job = k.submit(spec, admission=admission, guard=guard)
        code, d = self._job(job.id, wait)
        if report is not None:
            d["rehearsal"] = dict(seconds=report.seconds, moving_s=report.moving_s, ok=report.ok)
            if not report.ok:
                d["warning"] = (f"in rehearsal this ended {report.outcome.status}: {report.outcome.message} "
                                "(the world model may be incomplete; running it anyway)")
        return code, d

    def _camera(self, camera: str | None):
        if not self.cameras:
            raise Refused("no cameras: add [[camera]] entries to the workcell", "no_camera")
        name = camera or next(iter(self.cameras))
        cam = self.cameras.get(name)
        if cam is None:
            raise Refused(f"no camera {name!r}; cameras: {', '.join(self.cameras)}", "no_camera")
        return name, cam

    def look(self, camera: str | None = None, spec=None, grid: bool = False) -> dict:
        """One picture from a camera, with the tool, the known boxes and (given a plan) its path drawn on it, saved
        to the flight record. Returns the file's path: a model reads the image from there. With grid, a pixel ruler
        and nothing the kernel believes: for reading off where something is, e.g. while calibrating."""
        k = self.k
        name, cam = self._camera(camera)
        if grid or self._calibrating(name):            # what the kernel believes must not anchor an answer
            return self._save(k, name, cam, cameras.ruler(cam.picture(k)),
                              "a pixel grid every 100 px (x across, y down, from the top left); nothing else drawn"
                              + ("" if grid else ", because this camera is being calibrated"))
        report = self.rehearser.check(spec, k) if spec is not None else None
        img = cam.picture(k)
        tool = k.world.from_base("work", k.chain.fk(k.state.q)[:3, 3])
        drawn = ("magenta cross = tool point; green outlines = the boxes the kernel knows; F/L/U = work axes"
                 + ("; blue = the plan's tool path" if report is not None else "")
                 + ("; MuJoCo simulation" if isinstance(cam, cameras.SimCamera) else "")) \
            if cam.view is not None else "no calibration for this camera, so nothing is drawn on it"
        caption = f"{name} | t+{k.clock.now() - k.t0:.0f}s | tool F{tool[0]:+.3f} L{tool[1]:+.3f} U{tool[2]:+.3f}"
        img = cameras.overlay(img, cam.view, k, None if report is None else report.tool_path, caption)
        out = self._save(k, name, cam, img, drawn)
        if report is not None:
            out.update(check=str(report), ok=report.ok)
        return out

    def _save(self, k, name: str, cam, img, drawn: str) -> dict:
        self.shots += 1
        folder = (k.run_dir or Path(tempfile.gettempdir()) / "world-use") / "views"
        folder.mkdir(parents=True, exist_ok=True)
        path = (folder / f"{self.shots:04d}-{name}.{'png' if isinstance(cam, cameras.SimCamera) else 'jpg'}").resolve()
        img.save(path, quality=88) if path.suffix == ".jpg" else img.save(path)
        k.emit("look", f"{name}: {path.name}", camera=name, path=f"views/{path.name}",
               drawn=drawn, size=list(img.size))
        return dict(path=str(path), camera=name, size=list(img.size), drawn=drawn)

    def _settle(self):
        """Let the control loop read the body once or twice, so the reply shows the new state."""
        time.sleep(3 * getattr(self.k.clock, "dt", 0.01))

    def _home_route(self, steps: list | None, note: str) -> dict:
        """Rehearse a candidate without replacing the installed route, then atomically check and install it."""
        k = self.k
        rehearsal = {}
        if steps is None:
            k.set_home_route(None, note)
        else:
            steps = deepcopy(steps)
            with k.lock:
                snap, admission = k.checked_start(require_enabled=False)
                spec = {"do": "seq", "steps": k.home_plan(steps=steps), "label": "home"}
            report = self.rehearser.check(spec, k, snap=snap)
            if report.ok:
                k.set_home_route(steps, note, admission=admission)
            rehearsal = dict(ok=report.ok, problems=report.problems, text=str(report))
        with k.lock:
            return dict(home=views.status(k)["home"], **rehearsal)

    def calibrate(self, camera: str, points: int = 8, spread: float | None = None, wait: float = 0.0):
        """Start a calibration tour for a camera (see calibrate.py). The reply to its last answer carries the fit."""
        k = self.k
        if not k.enabled:
            raise Refused("torque is off: enable first", "off", "enable")
        with k.lock:
            busy = k.active is not None or bool(k.queue)
        if busy:
            raise Refused("a job is running: calibrate while the arm is idle", "busy")
        cam = self.cameras.get(camera)
        if cam is None:
            raise Refused(f"no camera {camera!r}; cameras: {', '.join(self.cameras)}", "no_camera")
        w, h = cam.picture(k).size                       # fails early on a dead camera
        s = min(1.0, cameras.MAX_SIDE / max(w, h))        # the answers count pixels of the picture `wu look` saves
        size = (round(w * s), round(h * s))
        steps, a = calibrate.tour(k, camera, size, points, (float(spread),) if spread else (0.12, 0.09, 0.06),
                                  check=self.rehearser.check)
        job = k.submit({"do": "seq", "steps": steps, "label": f"calibrate {camera} ({2 * a * 100:.0f} cm box)"})
        self.calibrations[job.id] = dict(camera=camera, size=size)
        return self._job(job.id, wait)

    def _calibrating(self, name: str) -> bool:
        return any(c["camera"] == name and not self.k.jobs[i].finished for i, c in self.calibrations.items())

    def _calibrated(self, job) -> dict:
        c = self.calibrations[job.id]
        with self._solving:
            if "result" not in c:
                c["result"] = self._fit(job, c["camera"], c["size"])
        return c["result"]

    def _fit(self, job, name: str, size) -> dict:
        k, cam = self.k, self.cameras[name]
        answers = (job.outcome.data if job.outcome else {}).get("answers", [])
        pixels = [calibrate.parse(a["answer"], size) for a in answers]
        seen = [i for i, px in enumerate(pixels) if px is not None]
        points = np.array([k.world.to_base("work", answers[i]["tool"]) for i in seen]).reshape(-1, 3)
        cut = cam if isinstance(cam, cameras.EquirectCut) else None
        try:
            fit = calibrate.solve(points, [pixels[i] for i in seen], size, fov_deg=cam.fov_deg or 60.0,
                                  focal=None if cut is None else cut.f * size[0] / cut.size[0])
        except Refused as e:
            return dict(installed=False, text=f"calibration of {name!r} from {len(answers)} answers: not installed: "
                        f"{e}" + (f" ({e.hint})" if e.hint else ""))
        why = calibrate.installable(fit, points, size)
        lines = calibrate.keep(fit, points, cut)
        if why is None:
            with k.lock:
                k.changed()
                if cut is None:
                    cam.view = fit.view
                else:
                    pose = np.eye(4)
                    pose[:3, :3], pose[:3, 3] = fit.R.T @ cut.R.T, fit.C
                    cut.install(pose)
                k.world.assert_fact(f"camera.{name}", lines.replace("\n", "; "),
                                    f"wu calibrate, job {job.id}: {len(fit.used)} points, fit {fit.rms:.0f} px")
                k.emit("calibrated",
                       f"{name}: fit {fit.rms:.0f} px, {fit.loo_rms:.0f} px each from the others; installed")
        return dict(installed=why is None, lines=lines,
                    text=calibrate.describe(name, fit, answers, pixels, size, why, lines, cut is not None))

    def _job(self, job_id: int, wait: float) -> tuple[int, dict]:
        job = self.k.jobs.get(job_id)
        if job is None:
            return 404, dict(error=f"no job {job_id}")
        if wait and not job.finished and job.status != "waiting":
            job.attention.wait(wait)
        d = job.to_dict()
        d["line"] = views.state_line(self.k)
        if job.outcome is not None and not job.outcome.ok:
            d["incident"] = views.incident(self.k, job, reach=self.rehearser.reach_line)
        if job.finished and job_id in self.calibrations:
            d["calibration"] = self._calibrated(job)
        return 200, d

    def _world(self, body: dict) -> tuple[int, dict]:
        """Change the world model. A request's changes are made on a copy first: all of them apply, or none."""
        k, w = self.k, self.k.world
        body = WorldChange.model_validate(body).model_dump(exclude_unset=True)
        trial, lines = deepcopy(w), []
        if "frame" in body:
            f = body["frame"]
            if f["name"] == "base":
                raise ValueError("frame: needs a name other than base")
            trial.add_frame(f["name"], frame_pose(f), f.get("source", "policy"))
            lines.append(views.frame_line(trial.frame(f["name"])))
        if "fact" in body:
            f = body["fact"]
            fact = trial.assert_fact(f["key"], f["value"], f.get("source", "policy"), f.get("note", ""))
            lines.append(f"fact {fact.key} = {fact.value} (from {fact.source})")
        box = None if "box" not in body else add_box(trial, body["box"], "policy")
        if "remove" in body and trial.boxes.pop(body["remove"], None) is None:
            raise KeyError(f"no box {body['remove']!r}; boxes: {sorted(w.boxes)}")
        k.changed()
        w.frames, w.boxes, w.facts = trial.frames, trial.boxes, trial.facts
        if box is not None:
            lines.append(views.box_line(k, box))
            k.emit("world", f"box {lines[-1]}")
        if "remove" in body:
            lines.append(f"removed {body['remove']!r}")
            k.emit("world", f"removed box {body['remove']!r}")
        k.emit("world_state", "world updated", change=body, world=w.to_dict())
        return 200, dict(line="\n".join(lines), boxes=sorted(w.boxes), facts=sorted(w.facts), frames=sorted(w.frames))


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
            port = d.http.server_port
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            host = self.headers.get("Host")
            if host not in hosts or self.headers.get("Origin") not in (None, f"http://{host}"):
                return self._reply(403, dict(error="use the local daemon address and a same-origin client"))
            u = urlparse(self.path)
            query = {k: v[-1] for k, v in parse_qs(u.query, keep_blank_values=True).items()}
            body = {}
            if method == "POST":
                if self.headers.get_content_type() != "application/json":
                    return self._reply(415, dict(error="POST requests require application/json"))
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    if not 0 <= n <= 1024 * 1024:
                        return self._reply(413, dict(error="request body must be at most 1 MiB"))
                    body = json.loads(self.rfile.read(n) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError("request body must be a JSON object")
                except (ValueError, UnicodeError) as e:
                    return self._reply(400, dict(error=str(e)))
            try:
                code, obj = d.api(method, u.path, query, body)
            except Refused as e:
                code, obj = 409, dict(refused=e.to_dict())
            except (KeyError, ValueError, TypeError) as e:
                code, obj = 400, dict(error=explain(e))
            except (OSError, RuntimeError) as e:          # a camera that did not answer, for instance
                code, obj = 502, dict(error=explain(e))
            except Exception as e:                        # a bug: answer anyway, and keep the traceback in the log
                traceback.print_exc()
                code, obj = 500, dict(error=explain(e))
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


def session_identity(name: str, cell: dict) -> dict:
    """The selected adapter and startup configuration, distinct from the robot manifest and live world."""
    adapter = "sim:rebot" if name == "sim" and "robot" not in cell else name
    config = dict(cell)
    for key in ("fit", "robot"):
        if key in config:
            config[f"{key}_sha256"] = hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
    if "robot" in config:
        model = load_robot(config["robot"])
        config["urdf_sha256"] = hashlib.sha256(model.urdf.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
    return dict(adapter=adapter, mode="simulation" if bodies.simulated(name) else "hardware",
                workcell_digest=digest)


def frame_pose(frame: dict) -> np.ndarray:
    """A frame's pose in the base frame, from its origin (m) and rpy_deg as a workcell gives them."""
    from .geometry import rpy
    T = np.eye(4)
    T[:3, 3] = _vector(frame.get("origin", [0, 0, 0]), 3, "frame.origin")
    T[:3, :3] = rpy(*np.radians(_vector(frame.get("rpy_deg", [0, 0, 0]), 3, "frame.rpy_deg")))
    return T


def add_box(world: World, spec: dict, source: str):
    """A box from a workcell or a request: name, kind, center and size (three numbers each, metres), optional frame,
    yaw_deg and source, and the parameters of its kind."""
    b = dict(spec)
    for key in ("name", "kind", "center", "size"):
        if key not in b:
            raise ValueError(f"box: missing {key}")
    return world.add_box(b.pop("name"), b.pop("kind"), b.pop("center"), b.pop("size"),
                         b.pop("frame", "work"), b.pop("yaw_deg", 0.0),
                         source=b.pop("source", source), **b)


def apply_workcell(cell: dict, k: Kernel, truth: World | None = None):
    """Boxes, facts, overrides and a fitted robot model from a workcell. With a simulator's truth world, boxes go
    there as well, and a box marked `known = false` goes only there: part of the scene the policy has to discover."""
    if "fit" in cell:
        k.use_fit(load_fit(cell["fit"]))
    for frame in cell.get("frame", []):
        k.world.add_frame(frame["name"], frame_pose(frame), source="workcell")
    if truth is not None:
        truth.frames.update(k.world.frames)
    for b in cell.get("box", []):
        b = dict(b)
        if b.pop("known", True):
            add_box(k.world, b, "workcell")
        if truth is not None and truth is not k.world:
            add_box(truth, b, "workcell")
    for f in cell.get("fact", []):
        k.world.assert_fact(f["key"], f["value"], f.get("source", "workcell"), f.get("note", ""))
    env = cell.get("envelope", {})
    if "max_excursion_deg" in env:
        k.envelope.override("max_excursion", np.radians(env["max_excursion_deg"]),
                            env.get("reason", "set in the workcell file by the operator"))
    if "turn_height_m" in env:
        k.envelope.override("turn_height", env["turn_height_m"],
                            env.get("turn_reason", "set in the workcell file by the operator"))


def make_cameras(cell: dict, k: Kernel, body, truth: World | None) -> dict[str, cameras.Camera]:
    """Cameras from the workcell. On a simulator, an entry without url or command renders the simulation from its
    eye/look_at; with no entries at all, a simulator gets three views (side, front, top)."""
    cams: dict[str, cameras.Camera] = {}
    for c in cell.get("camera", []):
        if "path" in c or "url" in c or "command" in c:
            cams[c["name"]] = cameras.from_config(c, k.world)
        elif truth is not None and (view := cameras.view_from_config(c, k.world)) is not None:
            cams[c["name"]] = cameras.SimCamera(c["name"], view, body)
        else:
            raise ValueError(f"camera {c.get('name')!r} needs a path, a url or a command "
                             "(or, on a simulator, eye and look_at)")
    if not cams and truth is not None:
        cams = cameras.sim_cameras(body, truth)
    return cams


def make_body(name: str, cell: dict, world: World | None = None):
    model = load_robot(cell["robot"]) if "robot" in cell else None
    options = dict(cell.get("body_options", {}))
    if name == "sim" or name.startswith("sim:"):
        # Keep existing simulation workcells working. A hardware driver's options never go to SimBody.
        selected = cell.get("body", "sim")
        if selected != "sim" and not selected.startswith("sim:"):
            options = {}
        if options.keys() & cell.get("simulation", {}).keys():
            raise ValueError("simulation options must not be repeated in body_options")
        options.update(cell.get("simulation", {}))
        if "start_deg" in options:
            if "q" in options:
                raise ValueError("simulation: choose start_deg or q, not both")
            options["q"] = np.radians(options.pop("start_deg"))
    try:
        return bodies.make(name, world, manifest=model, **options)
    except TypeError as e:
        raise ValueError(f"{name} options: {e}") from e


def main(argv=None):
    ap = argparse.ArgumentParser(description="world-use daemon: owns one robot and serves the policy API")
    ap.add_argument("--body", help="sim | sim:<built-in> | rebot | module:Class (default: workcell's, else sim)")
    ap.add_argument("--workcell", help="TOML file with robot, connection options and scene")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--runs", type=Path, default=Path("runs"), help="where flight records go")
    ap.add_argument("--enable", action="store_true", help="switch torque on right away")
    a = ap.parse_args(argv)
    sys.setswitchinterval(0.001)            # keep the control thread's ticks regular while other threads work
    cell = load_workcell(a.workcell)
    name = a.body or cell.get("body", "sim")
    world = World()
    # Only the built-in twin renders cameras and simulates contacts with the truth world.
    truth = World() if name == "sim" or name.startswith("sim:") else None
    body = make_body(name, cell, truth)
    run_dir = a.runs / f"{datetime.now():%Y%m%d-%H%M%S-%f}-{name.replace(':', '-')}"
    k = Kernel(body, world, run_dir=run_dir)
    k.connect()
    if truth is not None:
        truth.frames.update(world.frames)
    apply_workcell(cell, k, truth)
    d = Daemon(k, port=a.port, cams=make_cameras(cell, k, body, truth),
               session=session_identity(name, cell), config=cell)
    try:
        d.rehearser.check([], k)            # build the twin once: a robot it cannot model must not start
    except Exception as e:
        if getattr(e, "rule", None) != "worker_extension":    # registered plugin steps are refused per plan
            k.close()
            d.rehearser.close()             # its worker outlives an exiting parent
            d.http.server_close()
            raise ValueError(f"the rehearsal twin cannot model this robot: {e}") from e
    d.start()
    if a.enable:
        try:
            k.enable()
        except Exception as e:
            # Keep the daemon available for status and recovery after a partial power transition.
            print(f"enable failed: {explain(e)}", file=sys.stderr, flush=True)
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
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as e:
        sys.exit(f"world-use startup: {explain(e)}")
