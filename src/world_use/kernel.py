"""Kernel: the only code that talks to the body.

Each control tick: read the body, run the watchdog, advance the active behavior, send the command, record.
Its rules are the lessons of running a slow policy on real hardware:

- Idle means holding still. The kernel never moves on its own, except along a home route the policy set and
  that nothing has touched since (a heat emergency may use it).
- A surprise holds where the arm really is, cancels anything queued behind it, and waits for the policy.
- A refused command moves nothing.
- The policy may be slow, crash or restart: the kernel keeps holding until it is told something new.
- Once the control loop runs, only its thread talks to the body. Requests from other threads (switching torque
  on or off) are handed to it and waited for, so a driver is never called from two threads at once.
"""
import itertools
import json
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .behaviors import STATUSES, Behavior, ContactSense, Outcome, Residuals, Sequence, build
from .body import Body, JointState
from .envelope import MARGIN, Envelope, Trip
from .errors import Refused, explain
from .events import EventLog
from .kinematics import Chain
from .motion import Timing
from .recorder import Tape, save_summary
from .world import World

TERMINAL = (*STATUSES, "cancelled")            # a queued job that never started ends "cancelled"


class RealClock:
    def __init__(self, rate_hz: float):
        self.dt, self._next = 1.0 / rate_hz, None

    def now(self) -> float:
        return time.monotonic()

    def wait(self):
        now = time.monotonic()
        self._next = max((self._next or now) + self.dt, now)
        time.sleep(max(0.0, self._next - time.monotonic()))


class VirtualClock:
    """Simulated time: each tick advances it by one period without sleeping (twin checks run fast)."""

    def __init__(self, rate_hz: float):
        self.dt, self.t = 1.0 / rate_hz, 0.0

    def now(self) -> float:
        return self.t

    def wait(self):
        self.t += self.dt


@dataclass
class Command:
    q: np.ndarray
    dq: np.ndarray
    gripper: float | None
    gripper_v: float = 0.0


@dataclass
class Job:
    id: int
    behavior: Behavior
    spec: dict | list
    status: str = "queued"            # queued, running, waiting, then one of TERMINAL
    outcome: Outcome | None = None
    question: dict | None = None
    answer: str | None = None
    t_start: float | None = None
    t_end: float | None = None
    attention: threading.Event = field(default_factory=threading.Event)   # set on waiting or finished

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL

    def to_dict(self) -> dict:
        d: dict = dict(id=self.id, status=self.status, what=self.behavior.describe())
        if self.question:
            d["question"] = self.question
        if self.outcome:
            d["outcome"] = self.outcome.to_dict()
        if self.t_start and self.t_end:
            d["seconds"] = round(self.t_end - self.t_start, 2)
        return d


class Heat:
    """Temperature trend per joint from the readings since the torque last came on (at most the last minute):
    how long until the limit at this rate. Readings from before a switch would dilute the trend."""

    def __init__(self):
        self.samples: deque[tuple[float, np.ndarray]] = deque(maxlen=60)
        self.last_t = -np.inf
        self.on = False

    def update(self, t: float, temp, on: bool = True):
        if on != self.on:
            self.samples.clear()
            self.on = on
        if temp is not None and t - self.last_t >= 1.0:
            self.samples.append((t, np.asarray(temp, float)))
            self.last_t = t

    def slope_per_min(self) -> np.ndarray | None:
        if len(self.samples) < 5:
            return None
        t = np.array([s[0] for s in self.samples])
        T = np.array([s[1] for s in self.samples])
        tc = t - t.mean()
        return (tc @ (T - T.mean(0))) / (tc @ tc) * 60.0

    def minutes_left(self, limit: float) -> tuple[int, float, float] | None:
        """(joint index, temperature, minutes until limit) for the joint that gets there first, if rising."""
        if not self.samples:
            return None
        T, slope = self.samples[-1][1], self.slope_per_min()
        if slope is None:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            left = np.where(slope > 0.05, (limit - T) / slope, np.inf)
        i = int(np.nanargmin(left))
        return (i, float(T[i]), float(left[i])) if np.isfinite(left[i]) else None


class Kernel:
    state: JointState                                # the latest measurement; set by connect()
    cmd: Command                                     # what the body is told each tick; set by connect()
    q_start: np.ndarray                              # joints at the session start
    envelope: Envelope
    t0: float

    def __init__(self, body: Body, world: World | None = None, clock=None, run_dir: Path | None = None,
                 ik_weights=None, auto_answer: bool = False):
        self.body, self.manifest = body, body.manifest
        self.chain = Chain(self.manifest.urdf, self.manifest.tool_link)
        if self.chain.n != self.manifest.n:
            raise ValueError(f"URDF chain to {self.manifest.tool_link} has {self.chain.n} joints; "
                             f"manifest has {self.manifest.n}")
        self.world = world or World()
        self.clock = clock or RealClock(self.manifest.rate_hz)
        m = self.manifest
        self.timing = Timing(m.rate_hz, m.speed, m.auto_accel, m.min_move_s)
        self.ik_weights = ik_weights
        self.auto_answer = auto_answer             # twin checks: assume the expected answer at checkpoints
        self.run_dir = Path(run_dir) if run_dir else None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events = EventLog(self.run_dir / "events.jsonl" if self.run_dir else None)
        self.tape = Tape(m.n)
        self.heat = Heat()
        self.lock = threading.RLock()
        self.jobs: dict[int, Job] = {}
        self.queue: deque[Job] = deque()
        self.active: Job | None = None
        self._ids = itertools.count(1)
        self._stop: str | None = None
        self.enabled = self.faulted = False
        self.cameras: dict = {}                     # name -> camera, when a daemon owns some (the card lists them)
        self.last_touch = 0                        # event seq of the last contact (0 = none this session)
        self.home_route: tuple[list, int] | None = None   # (specs, event seq when set)
        self._hot_alarm_t = -np.inf
        self._warned: set[int] = set()
        self._sense: ContactSense | None = None    # collision check for every move (guarded moves add their own)
        self.held_at: float | None = None          # where the fingers closed on something, known to the world or not
        self.grip_start: float | None = None       # the gripper as the session found it: home puts it back
        self.residuals = Residuals(m.rate_hz)      # torque the model does not explain: contact checks judge against it
        self.still = 0                             # ticks the command has not changed for
        self._last_q_cmd = np.zeros(0)
        self._loop_thread: threading.Thread | None = None     # set by loop(): from then on the body's only caller
        self._posted: queue.SimpleQueue[tuple[Callable[[], object], Future]] = queue.SimpleQueue()

    # -- lifecycle -------------------------------------------------------------------------------
    def connect(self) -> JointState:
        st = self.body.connect()
        self.state, self.q_start = st, np.asarray(st.q, float).copy()
        self.cmd = Command(self.q_start.copy(), np.zeros(self.manifest.n), st.gripper)
        self.grip_start = st.gripper
        self.envelope = Envelope(self.manifest, self.chain, self.world, self.q_start)
        for name, T in (self.manifest.frames(self.chain, self.q_start) if self.manifest.frames else {}).items():
            if name not in self.world.frames:        # a twin inherits the real session's frames, never recomputes them
                self.world.add_frame(name, T, source=f"{self.manifest.name}, at session start")
        if "work" not in self.world.frames:
            self.world.add_frame("work", np.eye(4), source="default: the base frame")
        self.t0 = self.clock.now()
        self.emit("connected", f"{self.manifest.name} connected at joints (deg) "
                  f"{np.round(np.degrees(self.q_start), 1).tolist()}")
        for w in getattr(self.body, "warnings", []):
            self.emit("warning", w, "warn")
        return st

    def enable(self):
        """Torque on at the measured pose. Nothing if it is on already."""
        self._on_loop(self._enable)

    def release(self):
        """Torque off. Only where that moves nothing (the manifest's rest pose), and only when idle."""
        self._on_loop(self._release)

    def _enable(self):
        if self._loop_thread is not None and not self._loop_thread.is_alive():
            raise Refused("the control loop has stopped, so nothing would command the motors", "no_loop",
                          "restart the daemon")
        if self.enabled:
            return
        try:                          # not under the lock: an engage takes a second or two, and status must answer
            self.body.enable()
        except Exception as e:
            self.emit("enable_failed", explain(e), "warn" if isinstance(e, Refused) else "alarm")
            raise
        st = self.body.read()
        with self.lock:
            self.state = st
            self.cmd = Command(np.asarray(st.q, float).copy(), np.zeros(self.manifest.n), st.gripper)
            self.residuals.clear()                 # readings from before a release say nothing about now
            self.enabled = True
        self.emit("enabled", "torque on")

    def _release(self):
        with self.lock:
            if not self.enabled:
                return
            if self.active is not None:
                raise Refused("a job is running; stop it first", "busy")
            rest = self.manifest.rest
            if rest is not None and not rest.holds(self.state.q):
                raise Refused("the arm is not at its rest pose: releasing torque here would drop it", "not_at_rest",
                              "go home first")
            self.body.disable()
            self.enabled = False
        self.emit("released", "torque off")

    def _on_loop(self, fn: Callable[[], object]):
        """Run fn where the body may be called: on the control thread while the loop runs (this thread waits for
        it), else right here. Never call it holding self.lock: fn may need the lock on the control thread."""
        loop = self._loop_thread
        if loop is None or loop is threading.current_thread() or not loop.is_alive():
            return fn()
        f: Future = Future()
        self._posted.put((fn, f))
        while True:
            try:
                return f.result(timeout=0.5)
            except TimeoutError:
                if not loop.is_alive() and f.cancel():       # the loop ended before taking it: nothing was done
                    raise RuntimeError("the control loop stopped before it could do this") from None
            except CancelledError:
                raise RuntimeError("the control loop stopped before it could do this") from None

    def _run_posted(self, run: bool = True):
        """Do what other threads handed over; with run=False (the loop has stopped) refuse it instead: switching
        torque on now would leave nothing to command the motors."""
        while True:
            try:
                fn, f = self._posted.get_nowait()
            except queue.Empty:
                return
            if not run:
                f.cancel()
            elif f.set_running_or_notify_cancel():
                try:
                    f.set_result(fn())
                except Exception as e:
                    f.set_exception(e)

    def close(self) -> dict:
        """Close the connection (never switches torque off by itself) and write the flight record."""
        self.body.close()
        summary = self.save_record()
        self.emit("closed", "connection closed")
        self.events.close()
        return summary

    def save_record(self) -> dict:
        """Write the flight record so far (tape, summary, world; events are written as they happen), without
        closing: a run can be studied while it goes on. Returns the summary."""
        rate = self.manifest.rate_hz
        if not self.run_dir:
            return dict(body=self.manifest.name, **self.tape.summary(rate))
        summary = dict(body=self.manifest.name, **self.tape.save(self.run_dir / "tape.npz", rate))
        summary["events"] = self.events.seq
        save_summary(self.run_dir / "summary.json", summary)
        with self.lock:                                  # the control thread moves held boxes about
            world = self.world.to_dict()
        (self.run_dir / "world.json").write_text(json.dumps(world, indent=1))
        return summary

    # -- requests (any thread) ---------------------------------------------------------------------
    def submit(self, spec) -> Job:
        """Queue a behavior. A malformed spec is refused here; limits are checked when it starts."""
        behavior = build(spec)
        with self.lock:
            job = Job(next(self._ids), behavior, spec if not isinstance(spec, Behavior) else spec.spec())
            self.jobs[job.id] = job
            if self.faulted:
                self._end(job, Outcome("refused", behavior.kind, "the kernel is faulted: an operator must reset it",
                                       hint="check the hardware, then reset"))
            elif not self.enabled:
                self._end(job, Outcome("refused", behavior.kind, "torque is off: enable first", hint="enable"))
            else:
                self.queue.append(job)
        return job

    def stop(self, reason: str = "stop requested"):
        with self.lock:
            self._stop = reason

    def answer(self, job_id: int, text: str):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or job.status != "waiting":
                raise Refused(f"job {job_id} is not waiting for an answer", "no_question")
            job.answer, job.status = str(text), "running"      # resumes on the next tick
            job.attention.clear()

    def reset(self):
        """Operator: clear a fault after checking the hardware."""
        with self.lock:
            self.faulted = False
        self.emit("reset", "fault cleared by operator", "warn")

    def set_home_route(self, specs: list, note: str = "") -> None:
        """The policy's way out from here, checked against the scene it can see now. [] = fold straight home.
        It stays valid until something is touched."""
        for s in specs:
            build(s)
        with self.lock:
            self.home_route = (list(specs), self.events.seq + 1)
        self.emit("home_route", f"home route set ({len(specs)} moves, then fold)" + (f": {note}" if note else ""))

    def home_plan(self) -> list:
        """Route + fold back to the session's start pose, or Refused if nothing valid is known."""
        route = self.home_route
        if route is None:
            raise Refused("no home route set, so there is no known-clear way back", "no_home_route",
                          "look at the scene, then set a home route ([] = fold straight home from here)")
        if self.last_touch >= route[1]:
            raise Refused("the arm has touched something since the home route was set; the way back may be blocked",
                          "home_route_stale", "look again, then set the home route again")
        q0 = self.q_start.copy()
        rest = self.manifest.rest
        carry = set(rest.joints) if rest else set()
        # A folded arm rests on its stops, and the start pose was measured with torque off, sagged into them.
        # Powered, it meets them a little earlier: folding to that angle pushes into the stop. Fold to just off it.
        if rest is not None:
            for i in rest.stops:
                s = rest.off_stop(i, self.manifest.joints[i])
                q0[i] = rest.q[i] + s * max(MARGIN, s * (q0[i] - rest.q[i]))
        # full precision elsewhere: a rounded target would differ from where the session started
        free = {str(i + 1): float(np.degrees(q0[i])) for i in range(self.manifest.n) if i not in carry}
        fold = [{"do": "joints", "target_deg": free, "label": "turn back while high"}] if free else []
        if carry:
            fold.append({"do": "joints", "target_deg": {str(i + 1): float(np.degrees(q0[i])) for i in sorted(carry)},
                         "label": "fold"})
        plan = route[0] + [f for f in fold if self._differs(f)]
        g = self.manifest.gripper       # a gripper left open past pi comes back a turn low on the reBot
        if (g is not None and self.grip_start is not None and self.held_at is None and self.cmd.gripper is not None
                and abs(self.cmd.gripper - self.grip_start) > 0.02):     # never while it holds something
            lo, hi = sorted((g.closed, g.open))
            plan.append({"do": "gripper", "to": round(float(np.clip(self.grip_start, lo, hi)), 3),
                         "label": "gripper as it was found"})
        return plan

    def _differs(self, spec) -> bool:
        goal = self.cmd.q.copy()
        for k, v in spec["target_deg"].items():
            goal[int(k) - 1] = np.radians(v)
        return bool(np.abs(goal - self.cmd.q).max() > 1e-4)

    # -- services for behaviors (control thread) ---------------------------------------------------
    def set(self, q, dq=None):
        self.cmd.q = np.asarray(q, float).copy()
        self.cmd.dq = np.zeros(self.manifest.n) if dq is None else np.asarray(dq, float).copy()

    def set_gripper(self, position: float, v: float = 0.0):
        self.cmd.gripper, self.cmd.gripper_v = float(position), float(v)

    def hold_here(self):
        """Command the measured joint positions: stops pressing into whatever blocked the arm.
        The gripper keeps its command, so whatever it holds stays held."""
        self.set(np.asarray(self.state.q, float))

    def ask(self, question: dict):
        job = self.active
        assert job is not None, "a question is asked by the running job"
        if self.auto_answer:
            job.answer = question.get("expect", "yes")
            self.emit("assumed", f"assumed '{job.answer}' for: {question['ask']}", "warn", **question)
            return
        job.question, job.status = question, "waiting"
        job.attention.set()
        self.emit("question", question["ask"], **question)

    def take_answer(self) -> str | None:
        job = self.active
        assert job is not None, "an answer is taken by the running job"
        if job.answer is None:
            return None
        answer, job.answer, job.question = job.answer, None, None
        if job.status == "waiting":
            job.status = "running"
        return answer

    def rebias(self):
        """Re-zero the collision check at the current pose: the gravity model's error changes across the
        workspace, so it is measured afresh whenever a new motion starts."""
        self._sense = ContactSense(self) if "torque" in self.manifest.sensing and self.state.tau is not None else None

    def _collision(self) -> Trip | None:
        if self._sense is None:
            return None
        dev = self._sense.deviation(self)
        if len(self._sense.hist) < (self._sense.hist.maxlen or 0):
            return None
        limit = np.array([j.contact_dtau for j in self.manifest.joints])
        tool = self.chain.fk(self.state.q)[:3, 3]
        for zone in self.world.zones_at(tool):
            if zone.kind == "fragile":
                limit = np.minimum(limit, float(zone.params.get("dtau", 0.3)))
        limit = self._sense.limits(limit)
        over = np.abs(dev) > limit
        if over.any():
            i = int(np.argmax(np.abs(dev) - limit))
            return Trip("contact", f"unexpected contact: {self.manifest.joints[i].name} torque moved {dev[i]:+.1f} Nm "
                        f"beyond what the arm's weight explains (limit {limit[i]:.1f})", i, float(dev[i]),
                        float(limit[i]))
        return None

    def touched(self, kind: str, message: str):
        e = self.emit(kind, message)
        self.last_touch = e["seq"]

    def gripped(self, contact: float, attach: bool = True) -> str | None:
        """The fingers closed on something at `contact`: it is held until the gripper opens past it, whatever the
        world knows. With attach, an object the world knows at the tool point moves with the tool meanwhile
        (not when the grip found the wrong width: then it is not the object planned). Returns its name, if known."""
        self.held_at = contact
        held = self.world.held
        if held is not None:
            return held[0]
        box = self.world.grab(self.tool) if attach else None
        return None if box is None else box.name

    def _track_held(self):
        """Keep a held object's box with the tool; let go of it when the gripper opens past where it closed."""
        g = self.manifest.gripper
        if self.held_at is None or g is None:
            return
        if self.cmd.gripper is not None and (self.cmd.gripper - self.held_at) * np.sign(g.open - g.closed) > 0.1:
            box = self.world.drop()
            self.held_at = None
            if box is None:
                self.emit("let_go", "let go of what it held")
            else:
                c = self.world.from_base("work", box.pose[:3, 3]) if "work" in self.world.frames else box.pose[:3, 3]
                self.emit("let_go",
                          f"let go of {box.name!r}; it should now stand at F{c[0]:+.3f} L{c[1]:+.3f} U{c[2]:+.3f}")
            return
        self.world.carry(self.tool)

    def emit(self, kind: str, message: str, level: str = "info", **data) -> dict:
        return self.events.emit(kind, message, level, **data)

    # -- the control tick --------------------------------------------------------------------------
    def tick(self):
        st = self.body.read()
        self.state = st
        if self.enabled and st.tau is not None:
            self.residuals.push(np.asarray(st.tau, float) - self.chain.gravity(st.q))
        now = self.clock.now()
        self.heat.update(now, st.temp, self.enabled)
        with self.lock:
            self._track_held()
            if self.enabled:
                trip = self.envelope.watch(st, self.cmd.q, self.cmd.gripper)
                b = self.active.behavior if self.active is not None and self.active.status == "running" else None
                if trip is None and b is not None and b.moves and not b.senses_contact:
                    trip = self._collision()
                if trip:
                    self._on_trip(trip, now)
                self._heat_warnings(st)
            if self._stop is not None:
                reason, self._stop = self._stop, None
                self._cancel_queue(f"stopped: {reason}")
                if self.active is not None:
                    self._end(self.active, self.active.behavior.stop(self, reason))
                elif self.enabled:
                    self.hold_here()
            # a job starts once there is a torque baseline to judge contact against (0.1 s after switching on)
            if self.active is None and self.queue and self.enabled and (st.tau is None or self.residuals.ready):
                self._start(self.queue.popleft(), now)
            job = self.active
        if job is not None and job.status in ("running", "waiting"):
            try:
                out = job.behavior.tick(self)
            except Refused as e:
                out = Outcome("refused", job.behavior.kind, str(e), hint=e.hint)
            except Exception as e:                       # a bug in a behavior: hold, never keep going blind
                self.hold_here()
                out = Outcome("faulted", job.behavior.kind, f"{type(e).__name__}: {e}")
            if out is not None:
                with self.lock:
                    self._end(job, out)
        if self.enabled:
            self.body.command(self.cmd.q, self.cmd.dq, self.cmd.gripper, self.cmd.gripper_v)
        self.still = self.still + 1 if np.array_equal(self.cmd.q, self._last_q_cmd) else 0
        self._last_q_cmd = self.cmd.q.copy()
        moving = bool(job is not None and not job.finished and job.status == "running" and job.behavior.moves)
        self.tape.add(now - self.t0, self.enabled, moving, job.id if job else 0, self.cmd.q, st.q, st.tau, st.temp,
                      self.cmd.gripper, st.gripper, st.gripper_tau)

    def _start(self, job: Job, now: float):
        job.t_start = time.time()
        self.active = job
        self.rebias()
        self.envelope.context = job.behavior.describe()
        try:
            job.behavior.start(self)
        except Refused as e:
            self._end(job, Outcome("refused", job.behavior.kind, str(e), dict(rule=e.rule, **e.data), hint=e.hint))
            return
        except Exception as e:
            self.hold_here()
            self._end(job, Outcome("faulted", job.behavior.kind, f"{type(e).__name__}: {e}"))
            return
        job.status = "running"
        self.emit("started", f"job {job.id}: {job.behavior.describe()}", job=job.id)

    def _end(self, job: Job, out: Outcome):
        job.outcome, job.status, job.t_end = out, out.status, time.time()
        if self.active is job:
            self.active = None
        level = "info" if out.ok else ("alarm" if out.status == "faulted" else "warn")
        self.emit("finished", f"job {job.id} {out.status}: {out.message}", level, job=job.id, status=out.status)
        if not out.ok:
            self._cancel_queue(f"job {job.id} ended {out.status}")
            if out.status in ("surprise", "faulted"):
                stale = self.world.invalidate(f"job {job.id} {out.status}: {out.message}")
                if stale:
                    self.emit("facts_stale", f"facts to re-check: {', '.join(stale)}", "warn")
        job.attention.set()

    def _cancel_queue(self, why: str):
        while self.queue:
            j = self.queue.popleft()
            self._end(j, Outcome("cancelled", j.behavior.kind, f"not started: {why}"))

    def _on_trip(self, trip: Trip, now: float):
        if trip.isolate:                             # gripper only: freeze it where it is, the arm carries on
            self.cmd.gripper, self.cmd.gripper_v = self.state.gripper, 0.0
            self.emit("gripper_trip", trip.message, "warn")
            job = self.active
            step = job.behavior if job is not None else None
            while isinstance(step, Sequence) and step.current is not None:     # the step a plan is on
                step = step.current
            if job is not None and step is not None and step.kind in ("gripper", "grip"):
                seq = job.behavior
                where = f"step {seq.i + 1}/{len(seq.steps)}: " if isinstance(seq, Sequence) else ""
                self._end(job, Outcome("surprise", step.kind, where + trip.message, hint="look at the gripper"))
            return
        if trip.kind == "hot":
            self._on_hot(trip, now)
            return
        self.hold_here()
        if trip.kind in ("blocked", "overload", "contact"):
            self.touched("contact", f"watchdog: {trip.message}")
        status = "faulted" if trip.kind == "fault" else "surprise"
        if trip.kind == "fault":
            self.faulted = True
        if self.active is not None:
            self._end(self.active, Outcome(status, self.active.behavior.kind, trip.message,
                                           dict(trip=trip.kind, joint=None if trip.joint is None else trip.joint + 1),
                                           hint="holding where the arm is; look before the next move"))
        else:
            self.emit("trip", trip.message, "alarm" if status == "faulted" else "warn", trip=trip.kind)

    def _on_hot(self, trip: Trip, now: float):
        if self.active is not None and self.active.behavior.label == "home: motor hot":
            return                                    # already going home
        if self.active is not None:
            self._end(self.active, Outcome("stopped", self.active.behavior.kind, trip.message, dict(trip="hot")))
        rest = self.manifest.rest
        if rest is None or rest.holds(self.state.q):
            # at rest, switching torque off moves nothing, and it is how a motor cools fastest
            self._cancel_queue("motor hot")
            self.body.disable()
            self.enabled = False
            self.emit("hot", f"{trip.message}: torque released at rest to cool", "alarm")
            return
        try:
            plan = self.home_plan()
        except Refused as e:
            self.hold_here()
            if now - self._hot_alarm_t > 10.0:
                self._hot_alarm_t = now
                self.emit("hot", f"{trip.message} and {e}: holding. Send it home.", "alarm")
            return
        self.emit("hot", f"{trip.message}: going home along the home route", "alarm")
        job = Job(next(self._ids), build({"do": "seq", "steps": plan, "label": "home: motor hot"}), plan)
        self.jobs[job.id] = job
        self._start(job, now)

    def _heat_warnings(self, st: JointState):
        if st.temp is None:
            return
        temp = np.nan_to_num(np.asarray(st.temp, float))
        for i in np.where(temp > self.manifest.temp_warn_c)[0]:
            if int(i) not in self._warned:
                self._warned.add(int(i))
                left = self.heat.minutes_left(self.manifest.temp_limit_c)
                extra = f"; about {left[2]:.1f} min to the limit at this rate" if left and left[0] == i else ""
                self.emit("heat", f"{self.manifest.joints[i].name} at {temp[i]:.0f} C{extra}", "warn")
        self._warned &= set(int(i) for i in np.where(temp > self.manifest.temp_warn_c - 3)[0])

    # -- convenience -------------------------------------------------------------------------------
    def run(self, spec, timeout_s: float = 600.0) -> Outcome:
        """Submit and tick until it ends (scripts, tests, twin checks). Not for use while a loop thread runs."""
        if self._loop_thread is not None and self._loop_thread.is_alive():
            raise RuntimeError("the control loop is running: submit() the job and wait for it instead")
        job = self.submit(spec)
        deadline = self.clock.now() + timeout_s
        while not job.finished:
            self.tick()
            self.clock.wait()
            if self.clock.now() > deadline and self._stop is None:
                self.stop(f"timeout after {timeout_s:.0f} s")
            if job.status == "waiting" and not self.auto_answer:
                raise RuntimeError(f"job {job.id} is waiting for an answer: {(job.question or {}).get('ask')}")
        assert job.outcome is not None
        return job.outcome

    def loop(self, stop: threading.Event):
        """The control loop for a daemon thread. From here on this thread is the only one that calls the body."""
        if self._loop_thread is not None and self._loop_thread.is_alive():
            raise RuntimeError("a control loop is already running for this kernel")
        self._loop_thread = threading.current_thread()
        while not stop.is_set():
            self._run_posted()
            try:
                self.tick()
            except Exception as e:
                self._survive(e)
            self.clock.wait()
        self._run_posted(run=False)

    def _survive(self, e: Exception):
        """The body raised (an adapter unplugged, a value out of range): fault and say so, but keep the loop and
        the daemon alive, so the flight record survives and an operator sees why. The motors keep their last
        command until something works again or someone switches them off."""
        with self.lock:
            first = not self.faulted
            self.faulted = True
            self.hold_here()
            self._cancel_queue("the kernel faulted")
            if self.active is not None:
                self._end(self.active, Outcome("faulted", self.active.behavior.kind, explain(e),
                                               hint="check the hardware, then reset"))
        if first:
            self.emit("fault", f"control tick failed: {explain(e)}", "alarm")

    @property
    def tool(self) -> np.ndarray:
        """Measured tool pose (4x4, base frame)."""
        return self.chain.fk(self.state.q)
