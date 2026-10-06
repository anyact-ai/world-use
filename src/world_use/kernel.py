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
from __future__ import annotations

import itertools
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import CancelledError, Future
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field, replace
from numbers import Real
from pathlib import Path

import numpy as np

from .behaviors import (
    BUILTINS,
    STATUSES,
    Behavior,
    ContactSense,
    Gripper,
    Joints,
    Line,
    Lines,
    MoveTo,
    Outcome,
    Residuals,
    Sequence,
    build,
    fragile_dtau,
    walk,
)
from .body import Body, JointState
from .envelope import MARGIN, Envelope, Trip
from .errors import Refused, explain
from .events import EventLog
from .kinematics import Chain
from .motion import Timing
from .recorder import Journal, Tape, save_summary, session_record
from .validation import references
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
    t_start: float | None = None      # kernel clock
    t_end: float | None = None
    attention: threading.Event = field(default_factory=threading.Event)   # set on waiting or finished
    admission: Callable[[], None] | None = None    # revalidate a checked plan immediately before its first tick
    guard: Callable[[], None] | None = None        # raises Refused before a step once what the plan relied on is stale

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL

    def to_dict(self) -> dict:
        d: dict = dict(id=self.id, status=self.status, what=self.behavior.describe())
        if self.question:
            d["question"] = self.question
        if self.outcome:
            d["outcome"] = self.outcome.to_dict()
        if self.t_start is not None and self.t_end is not None:
            d["seconds"] = round(self.t_end - self.t_start, 2)
        return d


class Heat:
    """Temperature trend per joint from the readings since the torque last came on (at most the last minute):
    how long until the limit at this rate. Readings from before a switch would dilute the trend, and so would the
    first WARMUP_S after it: a motor driver's reading jumps as the current comes on (the reBot's elbow read 28 to
    37 C in 12 s, then flat), and a line through that said "1 min to 80 C"."""
    WARMUP_S = 20.0

    def __init__(self):
        self.samples: deque[tuple[float, np.ndarray]] = deque(maxlen=60)
        self.last_t = -np.inf
        self.on = False
        self.since = -np.inf                         # when the torque last came on or went off

    def update(self, t: float, temp, on: bool = True):
        if on != self.on:
            self.samples.clear()
            self.on, self.since = on, t
        if temp is not None and t - self.last_t >= 1.0:
            self.samples.append((t, np.asarray(temp, float)))
            self.last_t = t

    def minutes_left(self, limit: float) -> tuple[int, float, float] | None:
        """(joint index, temperature, minutes until limit) for the joint that gets there first, if rising. Request
        threads ask while the control thread updates, so it reads a copy of the samples."""
        settled = [s for s in list(self.samples) if s[0] >= self.since + self.WARMUP_S]
        if len(settled) < 5:
            return None
        t = np.array([s[0] for s in settled])
        T = np.array([s[1] for s in settled])
        tc = t - t.mean()
        slope = (tc @ (T - T.mean(0))) / (tc @ tc) * 60.0
        with np.errstate(divide="ignore", invalid="ignore"):
            left = np.where(slope > 0.05, (limit - T[-1]) / slope, np.inf)
        i = int(np.nanargmin(left))
        return (i, float(T[-1][i]), float(left[i])) if np.isfinite(left[i]) else None


class Kernel:
    # Exact behavior types whose internal motion boundaries call start_behavior/check_guard.
    # An embedded adapter may extend this set after auditing its own steps.
    guarded_steps = BUILTINS
    state: JointState                                # the latest measurement; set by connect()
    cmd: Command                                     # what the body is told each tick; set by connect()
    q_start: np.ndarray                              # joints at the session start
    envelope: Envelope
    t0: float

    def __init__(self, body: Body, world: World | None = None, clock=None, run_dir: Path | None = None,
                 auto_answer: bool = False):
        self.body, self.manifest = body, body.manifest
        self.chain = Chain(self.manifest.urdf, self.manifest.tool_link)
        if self.chain.n != self.manifest.n:
            raise ValueError(f"URDF chain to {self.manifest.tool_link} has {self.chain.n} joints; "
                             f"manifest has {self.manifest.n}")
        if [j.name for j in self.manifest.joints] != self.chain.joint_names:
            raise ValueError(f"manifest joints must follow URDF chain order: {self.chain.joint_names}")
        self.world = world or World()
        if getattr(body, "world", None) is self.world:
            self.world = World.from_dict(self.world.to_dict())
        self.clock = clock or RealClock(self.manifest.rate_hz)
        self.t0 = self.clock.now()
        m = self.manifest
        self.timing = Timing(m.rate_hz, m.speed, m.auto_accel, m.min_move_s)
        # Fewer than six joints cannot hold every orientation: unless the manifest says otherwise, such an arm keeps
        # the tool's tilt and lets its heading (yaw about the base's vertical) turn.
        free_yaw = m.ik_weights is None and m.n < 6
        self.ik_weights = (1.0, 1.0, 1.0, 1.0, 1.0, 0.0) if free_yaw else m.ik_weights
        self.auto_answer = auto_answer             # twin checks: assume the expected answer at checkpoints
        self.run_dir = Path(run_dir).expanduser().resolve() if run_dir else None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events = EventLog(clock=self.clock.now, t0=self.t0)
        self.tape = Tape(m.n)
        self.tape.mark_power(0.0, False)
        self.journal = Journal(self.tape, self.run_dir, events=self.events) if self.run_dir else None
        self.heat = Heat()
        self.lock = threading.RLock()
        self.control_revision = 0
        self.jobs: dict[int, Job] = {}
        self.queue: deque[Job] = deque()
        self.active: Job | None = None
        self._ids = itertools.count(1)
        self._stop: str | None = None
        self.enabled = self.faulted = False
        self._closing = False                   # closes admission even for enable calls already posted to the loop
        self.power_uncertain = False             # an incomplete enable/disable must never look like torque off
        self.feedback_at: float | None = None    # kernel-clock time of the last distinct successful sample
        self.feedback_error: str | None = None
        self._feedback_stamp: float | None = None
        self.cameras: dict = {}                     # name -> camera, when a daemon owns some (the card lists them)
        self.last_touch = 0                        # event seq of the last contact (0 = none this session)
        self.home_route: tuple[list, int] | None = None   # (specs, event seq when set)
        self._thermal_home: Job | None = None
        self._finding: tuple | None = None         # the watchdog finding of the last tick, reported once
        self._warned: set[int] = set()
        self._sense: ContactSense | None = None    # collision check for every move (guarded moves add their own)
        self.held_at: float | None = None          # where the fingers closed on something, known to the world or not
        self.grip_start: float | None = None       # the gripper as the session found it: home puts it back
        self.residuals = Residuals(m.rate_hz)      # torque the model does not explain: contact checks judge against it
        self.planner = None                        # a daemon's snapshot worker; embedded callers plan inline
        self.fit = None                            # a model fitted from flight records (fit.py), once one is in use
        self.still = 0                             # ticks the command has not changed for
        self._last_q_cmd = np.zeros(0)
        self._loop_thread: threading.Thread | None = None     # set by loop(): from then on the body's only caller
        self._posted: queue.SimpleQueue[tuple[Callable[[], object], Future]] = queue.SimpleQueue()

    # -- lifecycle -------------------------------------------------------------------------------
    def connect(self) -> JointState:
        st = self._validate_state(self.body.connect())
        self.state, self.q_start = st, np.asarray(st.q, float).copy()
        self.feedback_at, self._feedback_stamp = self.clock.now(), st.t
        self.cmd = Command(self.q_start.copy(), np.zeros(self.manifest.n), st.gripper)
        self.grip_start = st.gripper
        self.envelope = Envelope(self.manifest, self.chain, self.world, self.q_start, self.emit)
        for name, T in (self.manifest.frames(self.chain, self.q_start) if self.manifest.frames else {}).items():
            if name not in self.world.frames:        # a twin inherits the real session's frames, never recomputes them
                self.world.add_frame(name, T, source=f"{self.manifest.name}, at session start")
        if "work" not in self.world.frames:
            self.world.add_frame("work", np.eye(4), source="default: the base frame")
        self.emit("connected", f"{self.manifest.name} connected at joints (deg) "
                  f"{np.round(np.degrees(self.q_start), 1).tolist()}")
        for w in getattr(self.body, "warnings", []):
            self.emit("warning", w, "warn")
        self.record_session()
        return st

    def record_session(self, **context):
        """Record startup state after connecting and applying the workcell, before executing a task."""
        if self.run_dir:
            save_summary(self.run_dir / "session.json", session_record(self, **context))

    def enable(self):
        """Torque on at the measured pose. Nothing if it is on already; refused while the kernel is faulted."""
        self._on_loop(self._enable)

    def release(self):
        """Torque off. Only where that moves nothing (the manifest's rest pose), and only when idle."""
        self._on_loop(self._release)

    def begin_shutdown(self):
        """Close admission before releasing power. A failed release leaves the session available for recovery."""
        with self.lock:
            self._closing = True
            self.changed()
        try:
            self.release()
        except BaseException:
            with self.lock:
                self._closing = False
            raise

    def _accept_work(self):
        if self._closing:
            raise Refused("the kernel is shutting down; no new work can start", "closing")

    def _enable(self):
        with self.lock:
            self._accept_work()
            self.changed()
        if self._loop_thread is not None and not self._loop_thread.is_alive():
            raise Refused("the control loop has stopped, so nothing would command the motors", "no_loop",
                          "restart the daemon")
        if self.faulted:              # a fault with torque off (a failed read, say) is not cleared by switching on
            raise Refused("the kernel is faulted: an operator must reset it before the torque comes on", "faulted",
                          "check the hardware, then reset")
        if self.enabled:
            return
        self.power_uncertain = True
        self.tape.mark_power(self.clock.now() - self.t0, True)
        try:                          # not under the lock: an engage takes a second or two, and status must answer
            self.body.enable()
            self.enabled = True
            st = self._read_state()
        except Exception as e:
            with self.lock:
                self.faulted = True
                self._cancel_queue("enable failed; motor power is unconfirmed")
            self.emit("enable_failed", explain(e), "warn" if isinstance(e, Refused) else "alarm")
            raise
        with self.lock:
            self.state = st
            self.cmd = Command(np.asarray(st.q, float).copy(), np.zeros(self.manifest.n), st.gripper)
            self.residuals.clear()                 # readings from before a release say nothing about now
            self.enabled = True
            self.power_uncertain = False
        self.emit("enabled", "torque on")

    def _release(self):
        self.changed()
        with self.lock:
            if not self.enabled and not self.power_uncertain:
                return
            if self.active is not None or self.queue:
                raise Refused("a job is running; stop it first", "busy")
            # A failed enable may have left an old reading behind. Never release on that evidence.
            self.state = self._read_state()
            rest = self.manifest.rest
            if rest is not None and not rest.holds(self.state.q):
                raise Refused("the arm is not at its rest pose: releasing torque here would drop it", "not_at_rest",
                              "go home first")
            self.power_uncertain = True
            try:
                self.body.disable()
            except Exception:
                self.faulted = True
                raise
            self.enabled = False
            self.power_uncertain = False
            self.tape.mark_power(self.clock.now() - self.t0, False)
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
        try:
            self.body.close()
        except Exception as e:
            self.emit("adapter", f"closing the connection failed ({e}); saving the flight record anyway", "warn")
        self.emit("closed", "connection closed")
        try:
            if self.journal:
                self.journal.close()
            return self.save_record()
        except OSError as e:
            summary = (self.journal.summary(until=self.clock.now() - self.t0) if self.journal else
                       self.tape.summary(until=self.clock.now() - self.t0))
            return dict(body=self.manifest.name, recording_error=str(e),
                        **summary)

    def save_record(self) -> dict:
        """Write the flight record so far (tape, summary, world and events), without
        closing: a run can be studied while it goes on. Returns the summary."""
        until = self.clock.now() - self.t0
        if not self.run_dir:
            return dict(body=self.manifest.name, **self.tape.summary(until=until))
        assert self.journal is not None
        self.journal.flush()
        summary = dict(body=self.manifest.name, **self.journal.summary(until=until))
        summary["events"] = self.events.seq
        save_summary(self.run_dir / "summary.json", summary)
        with self.lock:                                  # the control thread moves held boxes about
            world = self.world.to_dict()
        save_summary(self.run_dir / "world.json", world)
        return summary

    # -- requests (any thread) ---------------------------------------------------------------------
    def checked_start(self, *, require_enabled: bool = True):
        """An idle snapshot and its admission check, also checked when the submitted job starts.

        Rehearsal runs outside the lock. Only the kernel accounts for intervening control changes and the
        checked job's own submission; observation events and ordinary encoder noise do not invalidate it.
        """
        from .plan import same_start, snapshot

        with self.lock:
            self._accept_work()
            if (require_enabled and not self.enabled) or self.faulted or self.power_uncertain:
                raise Refused("checked runs need torque on, confirmed power and a cleared fault", "not_ready",
                              "inspect status and resolve the power state before running")
            if self.active is not None or self.queue or self._stop is not None:
                raise Refused("checked runs need an idle robot; another job is running, queued or stopping",
                              "busy", "wait for it to finish, then retry")
            snap, revision, enabled = snapshot(self), self.control_revision, self.enabled

        def admission():
            gripper = self.state.gripper
            changed = (not same_start(snap, self)
                       or ((gripper is None) != (snap.gripper is None))
                       or (gripper is not None and snap.gripper is not None and abs(gripper - snap.gripper) > 0.01))
            own_job = self.active is not None and self.active.admission is admission
            if (changed or self.control_revision != revision + int(own_job)
                    or self.queue or self._stop is not None or (self.active is not None and not own_job)
                    or self.enabled != enabled or self.faulted or self.power_uncertain or self._closing):
                raise Refused("the robot or scene changed during rehearsal; nothing started", "stale_check",
                              "wait until idle, then retry so the plan is checked from the new state")

        return snap, admission

    def submit(self, spec, admission: Callable[[], None] | None = None, *,
               guard: Callable[[], None] | None = None) -> Job:
        """Queue a behavior. A malformed spec, or one naming a frame or joint this robot lacks, is refused here;
        limits are checked when it starts. A guard is called now and before each step, and refuses the step once
        what the plan relied on is stale. While a thermal return runs, the job ends refused at once."""
        behavior = build(spec if isinstance(spec, Behavior) else deepcopy(spec))
        if guard is not None and any(type(step) not in self.guarded_steps for step in walk(behavior)):
            raise Refused("a guarded plan needs built-in or explicitly audited steps; this custom step is unchecked",
                          "custom_step")
        with self.lock:
            self._accept_work()
            self._references(behavior)
            if guard is not None:
                guard()
            if admission is not None:
                admission()
            self.changed()
            job = Job(next(self._ids), behavior, deepcopy(behavior.spec()), admission=admission, guard=guard)
            self.jobs[job.id] = job
            self.emit("submitted", f"job {job.id}: {behavior.describe()}", job=job.id, spec=job.spec)
            if self.faulted or self.power_uncertain:
                self._end(job, Outcome("refused", behavior.kind, "the kernel is faulted or motor power is unconfirmed",
                                       hint="check the hardware, then reset"))
            elif not self.enabled:
                self._end(job, Outcome("refused", behavior.kind, "torque is off: enable first", hint="enable"))
            elif self._thermal_home is not None:
                self._end(job, Outcome("refused", behavior.kind, "a motor is too hot: the thermal return is running "
                                       "and ends with torque off at rest", hint="let the motors cool, then enable"))
            else:
                self.queue.append(job)
        return job

    def stop(self, reason: str = "stop requested"):
        with self.lock:
            self.changed()
            self._stop = reason

    def answer(self, job_id: int, text: str):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or job.status != "waiting":
                raise Refused(f"job {job_id} is not waiting for an answer", "no_question")
            self.emit("answer", f"job {job_id}: checkpoint answered", job=job_id, answer=str(text),
                      question=job.question)
            job.answer, job.status = str(text), "running"      # resumes on the next tick
            job.attention.clear()

    def reset(self):
        """Operator: clear a fault after checking the hardware."""
        with self.lock:
            self.changed()
            if self.power_uncertain:
                raise Refused("motor power is unconfirmed: release at a freshly measured rest pose first",
                              "power_uncertain")
            self.faulted = False
            self._finding = None                  # a fault still present is reported again, not re-latched silently
        self.emit("reset", "fault cleared by operator", "warn")

    def _home_steps(self, specs: list) -> list:
        """Emergency returns must not depend on answers, holds, contact, or arbitrary plugin behavior."""
        if not isinstance(specs, list):
            raise Refused("a home route must be a list of steps, or null to clear it", "home_route_step")
        route = build(specs)
        for step in walk(route):
            if type(step) not in (Sequence, Joints, Line, Lines, MoveTo, Gripper):
                raise Refused(f"{step.kind} is not allowed in a home route; use only motion and gripper steps",
                              "home_route_step", "resolve questions before setting the route; use null to clear it")
        self._references(route)
        return deepcopy(route.spec()["steps"])

    def set_home_route(self, specs: list | None, note: str = "", *,
                       admission: Callable[[], None] | None = None) -> None:
        """Store a route, with an optional check that its rehearsal is still current. [] = fold straight home.
        Embedded callers own the rehearsal; step limits still apply. None clears the route."""
        steps = None if specs is None else self._home_steps(specs)
        with self.lock:
            if steps is not None:
                self._accept_work()
            if admission is not None:
                admission()
            self.changed()
            self.home_route = None if steps is None else (steps, self.events.seq + 1)
        message = "home route cleared" if steps is None else f"home route set ({len(steps)} moves, then fold)"
        self.emit("home_route", message + (f": {note}" if note else ""), steps=steps, note=note)

    def home_plan(self, *, steps: list | None = None) -> list:
        """Route + supported rest pose. Explicit steps prepare a candidate without installing it."""
        if steps is None:
            route = self.home_route
            if route is None:
                raise Refused("no home route set, so there is no known-clear way back", "no_home_route",
                              "look at the scene, then set a home route ([] = fold straight home from here)")
            if self.last_touch >= route[1]:
                raise Refused("the arm has touched something since the home route was set; the way back may be blocked",
                              "home_route_stale", "look again, then set the home route again")
            steps = route[0]
        steps = self._home_steps(steps)
        q0 = self.q_start.copy()
        rest = self.manifest.rest
        carry = set(rest.joints) if rest else set()
        # Leave clearance from physical stops: a powered arm can meet them before its unpowered rest angle.
        if rest is not None:
            # Session-start bounds remain valid after any escape moves in the route.
            lo, hi = self.envelope.bounds(self.q_start)
            for i in sorted(carry):
                lo[i], hi[i] = max(lo[i], rest.q[i] - rest.tol), min(hi[i], rest.q[i] + rest.tol)
                if i in rest.stops:
                    s = rest.off_stop(i, self.manifest.joints[i])
                    if s > 0:
                        lo[i] = max(lo[i], rest.q[i] + MARGIN)
                    else:
                        hi[i] = min(hi[i], rest.q[i] - MARGIN)
                if lo[i] > hi[i]:
                    raise Refused(f"{self.manifest.joints[i].name}: no supported rest target satisfies the joint "
                                  "planning limits and stop clearance", "home_rest",
                                  "check the configured rest pose and tolerance")
                q0[i] = float(np.clip(rest.q[i], lo[i], hi[i]))
        # full precision elsewhere: a rounded target would differ from where the session started
        free = {str(i + 1): float(np.degrees(q0[i])) for i in range(self.manifest.n) if i not in carry}
        fold = [{"do": "joints", "target_deg": free, "label": "turn back while high"}] if free else []
        if carry:
            fold.append({"do": "joints", "target_deg": {str(i + 1): float(np.degrees(q0[i])) for i in sorted(carry)},
                         "label": "fold"})
        plan = steps + fold
        g = self.manifest.gripper       # a gripper left open past pi comes back a turn low on the reBot
        if g is not None and self.grip_start is not None and self.held_at is None:  # never while holding
            lo, hi = sorted((g.closed, g.open))
            plan.append({"do": "gripper", "to": round(float(np.clip(self.grip_start, lo, hi)), 3),
                         "label": "gripper as it was found"})
        return plan

    # -- services for behaviors (control thread) ---------------------------------------------------
    def changed(self):
        """Invalidate checked admission at a control mutation, even if it is subsequently reversed."""
        with self.lock:
            self.control_revision += 1

    def check_guard(self):
        """Before a step moves: if the running job's guard refuses, hold here and let the refusal end the job."""
        job = self.active
        if job is not None and job.guard is not None:
            try:
                job.guard()
            except Refused:
                self.hold_here()
                raise

    def start_behavior(self, behavior):
        """Start the running job's behavior or one of its steps. Every runner (a plan, a grasp, a grip's opening)
        starts steps here, so each passes the job's guard, re-zeroes the contact check and names its place in the
        plan for rehearsals."""
        self.check_guard()
        self.rebias()
        step, where = self._step()
        self.envelope.context = ": ".join([*where, (step or behavior).describe()])
        behavior.start(self)

    def _step(self) -> tuple[Behavior | None, list[str]]:
        """The running job's innermost current step, and where it is in the plan: ["step 2/3", "step 1/2"]."""
        step, where = (None if self.active is None else self.active.behavior), []
        while isinstance(step, Sequence) and step.current is not None:
            where.append(f"step {step.i + 1}/{len(step.steps)}")
            step = step.current
        return step, where

    def _references(self, behavior):
        for step in walk(behavior):
            if type(step) in BUILTINS:
                references(step.params, self.world, self.manifest.n)

    def set(self, q, dq=None):
        self.cmd.q = np.asarray(q, float).copy()
        self.cmd.dq = np.zeros(self.manifest.n) if dq is None else np.asarray(dq, float).copy()

    def set_gripper(self, position: float, v: float = 0.0):
        self.cmd.gripper, self.cmd.gripper_v = float(position), float(v)

    def hold_here(self):
        """Command the measured joint positions: stops pressing into whatever blocked the arm.
        The gripper keeps its command, so whatever it holds stays held."""
        self.set(np.asarray(self.state.q, float))
        self.cmd.gripper_v = 0.0

    def ask(self, question: dict):
        job = self.active
        assert job is not None, "a question is asked by the running job"
        if self.auto_answer:
            expect = question.get("expect", "yes")
            job.answer = "(any answer)" if expect is None else expect
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
        limit = self._sense.limits([j.contact_dtau for j in self.manifest.joints], fragile_dtau(self))
        over = np.abs(dev) > limit
        if over.any():
            i = int(np.argmax(np.abs(dev) - limit))
            doubt = self._sense.doubt(limit, [i])
            return Trip("contact", f"unexpected contact: {self.manifest.joints[i].name} torque moved {dev[i]:+.1f} Nm "
                        f"beyond what the arm's weight explains (limit {limit[i]:.1f}"
                        + (f"; {doubt}" if doubt else "") + ")", i, float(dev[i]), float(limit[i]))
        return None

    def use_fit(self, model):
        """Judge torque by a model fitted from this robot's flight records (fit.py) instead of the URDF alone: its
        links' masses and centres of mass for gravity everywhere (contact checks, load limits, rehearsals), and its
        friction at the commanded velocity. A body that computes gravity itself (the reBot's feedforward, the
        simulator's torques) takes it too. While the control loop runs, the change happens between two ticks."""
        if model.body != self.manifest.name:
            raise ValueError(f"that model was fitted for {model.body!r}, not {self.manifest.name!r}")
        self._on_loop(lambda: self._use_fit(model))
        self.emit("fit", model.headline(), fit=model.to_dict())

    def _use_fit(self, model):
        with self.lock:
            self.changed()
            model.apply(self.chain)
            use = getattr(self.body, "use_fit", None)
            if use is not None:
                use(model)
            self.fit = model
            self.residuals.clear()              # readings judged by the old model say nothing about the new one

    def expected_torque(self, q) -> np.ndarray:
        """Joint torque the model explains at q: the arm's own weight and, with a fitted model, friction at the
        commanded velocity. A contact check judges the rest."""
        tau = self.chain.gravity(q)
        return tau if self.fit is None else tau + self.fit.friction_torque(self.cmd.dq)

    def touched(self, kind: str, message: str):
        self.changed()
        e = self.emit(kind, message, world=self.world.to_dict())
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
        if box is not None:
            self.emit("attached", f"gripped {box.name}", world=self.world.to_dict())
        return None if box is None else box.name

    def _track_held(self):
        """Keep a held object's box with the tool; let go of it when the gripper opens past where it closed, by more
        than twice the squeeze (a grip itself commands one squeeze past it)."""
        g = self.manifest.gripper
        if self.held_at is None or g is None:
            return
        opened = 0.0 if self.cmd.gripper is None else (self.cmd.gripper - self.held_at) * np.sign(g.open - g.closed)
        if opened > 2 * g.squeeze:
            box = self.world.drop()
            self.held_at = None
            if box is None:
                self.emit("let_go", "let go of what it held")
            else:
                c = self.world.from_base("work", box.pose[:3, 3]) if "work" in self.world.frames else box.pose[:3, 3]
                self.emit("let_go",
                          f"let go of {box.name!r}; it should now stand at F{c[0]:+.3f} L{c[1]:+.3f} U{c[2]:+.3f}",
                          world=self.world.to_dict())
            return
        self.world.carry(self.tool)

    def emit(self, kind: str, message: str, level: str = "info", **data) -> dict:
        return self.events.emit(kind, message, level, **data)

    # -- the control tick --------------------------------------------------------------------------
    def _validate_state(self, st: JointState) -> JointState:
        """Check a sample before trusting it, and own its arrays even if the driver reuses its buffers."""
        fields = {}
        for name in ("q", "dq", "tau", "temp"):
            value = getattr(st, name)
            if value is None and name != "q":
                continue
            array = np.asarray(value)
            if array.shape != (self.manifest.n,) or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
                raise ValueError(f"feedback.{name} must contain {self.manifest.n} finite numbers")
            fields[name] = array.astype(float, copy=True)
        for name in ("t", "gripper", "gripper_tau"):
            value = getattr(st, name)
            if value is None and name != "t":
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"feedback.{name} must be a finite number")
        return replace(st, **fields)

    def _read_state(self) -> JointState:
        try:
            st = self._validate_state(self.body.read())
            if self._feedback_stamp is not None and st.t < self._feedback_stamp:
                raise ValueError("feedback.t must not move backwards")
            if st.t != self._feedback_stamp:
                self.feedback_at, self._feedback_stamp = self.clock.now(), st.t
            if self.feedback_at is None or self.clock.now() - self.feedback_at > 1.0:
                raise ConnectionError("the body returned no new feedback for over 1 second")
        except Exception as e:
            self.feedback_error = explain(e)
            self._survive(e)         # posted enable/release calls need the same fault handling as a tick
            raise
        self.feedback_error = None
        return st

    def feedback_status(self) -> dict:
        age = None if self.feedback_at is None else max(0.0, self.clock.now() - self.feedback_at)
        return dict(age_s=None if age is None else round(age, 3),
                    stale=self.feedback_error is not None or age is None or age > 1.0, error=self.feedback_error)

    def tick(self):
        """One control tick. Embedded callers get the same fault latch as the daemon loop."""
        try:
            self._tick()
        except Exception as e:
            self._survive(e)
            raise

    def _tick(self):
        st = self._read_state()
        self.state = st
        if self.enabled and st.tau is not None:
            self.residuals.push(np.asarray(st.tau, float) - self.expected_torque(st.q))
        now = self.clock.now()
        self.heat.update(now, st.temp, self.enabled)
        with self.lock:
            self._track_held()
            finding = None
            if self.enabled:
                trip = self.envelope.watch(st, self.cmd.q, self.cmd.gripper,
                                           self.active is not None and self.active is self._thermal_home)
                b = self.active.behavior if self.active is not None and self.active.status == "running" else None
                if (trip is None or trip.kind == "hot") and b is not None and b.moves and not b.senses_contact:
                    trip = self._collision() or trip
                if trip:
                    # Faults and keep-out zones name no joint: what they say tells one from another.
                    finding = (trip.kind, trip.joint, trip.message if trip.kind in ("fault", "keep_out") else None)
                    self._on_trip(trip, now, finding != self._finding)
                self._heat_warnings(st)
            self._finding = finding
            if self._stop is not None:
                reason, self._stop = self._stop, None
                self._cancel_queue(f"stopped: {reason}")
                if self.active is not None:
                    self._end(self.active, self.active.behavior.stop(self, reason))
                elif self.enabled:
                    self.hold_here()
            # a job starts once there is a torque baseline to judge contact against (0.1 s after switching on)
            if (self.active is None and self.queue and self.enabled and not self.faulted
                    and not self.power_uncertain and (st.tau is None or self.residuals.ready)):
                self._start(self.queue.popleft(), now)
            job = self.active
        if job is not None and job.status in ("running", "waiting"):
            try:
                out = job.behavior.tick(self)
            except Refused as e:
                self.hold_here()
                out = Outcome("refused", job.behavior.kind, str(e), dict(rule=e.rule, **e.data), hint=e.hint)
            except Exception as e:                       # a bug in a behavior: hold, never keep going blind
                self.hold_here()
                out = Outcome("faulted", job.behavior.kind, f"{type(e).__name__}: {e}")
            if out is not None:
                with self.lock:
                    self._end(job, out)
        if self.enabled and not self.power_uncertain:
            self.body.command(self.cmd.q, self.cmd.dq, self.cmd.gripper, self.cmd.gripper_v)
        self.still = self.still + 1 if np.array_equal(self.cmd.q, self._last_q_cmd) else 0
        self._last_q_cmd = self.cmd.q.copy()
        moving = bool(job is not None and not job.finished and job.status == "running" and job.behavior.moves)
        self.tape.add(now - self.t0, self.enabled or self.power_uncertain, moving, job.id if job else 0,
                      self.cmd.q, st.q, st.tau, st.temp,
                      self.cmd.gripper, st.gripper, st.gripper_tau)

    def _start(self, job: Job, now: float):
        job.t_start = now
        self.active = job
        try:
            if job.admission is not None:
                job.admission()
            self.start_behavior(job.behavior)
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
        thermal = job is self._thermal_home
        if thermal:
            self._thermal_home = None
        if self.active is job:
            self.active = None
            self.cmd.dq, self.cmd.gripper_v = np.zeros(self.manifest.n), 0.0      # no feedforward between jobs
        if thermal and out.ok:
            self._cancel_queue("the thermal return ends with torque off")
            try:
                self._release()                 # thermal completion includes confirmed torque-off
            except Exception as e:
                out = Outcome("faulted", job.behavior.kind, f"thermal return could not release: {explain(e)}",
                              hint="treat the arm as energized; resolve motor power before reset")
        if out.status == "faulted":
            self.faulted = True
        job.outcome, job.status, job.t_end = out, out.status, self.clock.now()
        level = "info" if out.ok else ("alarm" if out.status == "faulted" else "warn")
        self.emit("finished", f"job {job.id} {out.status}: {out.message}", level, job=job.id, status=out.status,
                  outcome=out.to_dict())
        if not out.ok:
            self._cancel_queue(f"job {job.id} ended {out.status}")
            if out.status in ("surprise", "faulted"):
                stale = self.world.invalidate(f"job {job.id} {out.status}: {out.message}")
                if stale:
                    self.emit("facts_stale", f"facts to re-check: {', '.join(stale)}", "warn")
        job.attention.set()
        if thermal:
            if out.ok:
                self.emit("hot", "thermal return complete: torque released at rest to cool", "alarm")
            else:
                self.set_home_route(None, "thermal return did not finish; inspect the scene before retrying")
                self.emit("hot", "thermal return failed; torque may remain on. Operator must resolve power now.",
                          "alarm")

    def _cancel_queue(self, why: str):
        cancelled = list(self.queue)
        self.queue.clear()                      # first: each _end below would cancel the rest again, recursively
        for j in cancelled:
            self._end(j, Outcome("cancelled", j.behavior.kind, f"not started: {why}"))

    def _on_trip(self, trip: Trip, now: float, new: bool):
        """Act on a watchdog finding: every tick it holds and cancels queued work, but a finding that persists
        (a box over the idle arm, a steady overload) is reported, and counted as a touch, only when it first shows."""
        self._cancel_queue(f"watchdog: {trip.message}")
        if trip.isolate:                             # gripper only: freeze it where it is, the arm carries on
            self.cmd.gripper, self.cmd.gripper_v = self.state.gripper, 0.0
            if new:
                self.emit("gripper_trip", trip.message, "warn")
            step, where = self._step()
            if self.active is not None and step is not None and step.kind in ("gripper", "grip", "grasp"):
                self._end(self.active, Outcome("surprise", step.kind, ": ".join([*where, trip.message]),
                                               hint="look at the gripper"))
            return
        if trip.kind == "hot" and not self.faulted and not self.power_uncertain:
            self._on_hot(trip, now, new)
            return
        self.hold_here()
        if new and trip.kind in ("blocked", "overload", "contact"):
            self.touched("contact", f"watchdog: {trip.message}")
        status = "faulted" if trip.kind == "fault" else "surprise"
        if trip.kind == "fault":
            self.faulted = True
        if self.active is not None:
            self._end(self.active, Outcome(status, self.active.behavior.kind, trip.message,
                                           dict(trip=trip.kind, joint=None if trip.joint is None else trip.joint + 1),
                                           hint="holding where the arm is; look before the next move"))
        elif new:
            self.emit("trip", trip.message, "alarm" if status == "faulted" else "warn", trip=trip.kind)

    def _on_hot(self, trip: Trip, now: float, new: bool):
        if self.active is not None:
            self._end(self.active, Outcome("stopped", self.active.behavior.kind, trip.message, dict(trip="hot")))
        rest = self.manifest.rest
        if rest is None or rest.holds(self.state.q):
            # at rest, switching torque off moves nothing, and it is how a motor cools fastest
            self._release()
            self.emit("hot", f"{trip.message}: torque released at rest to cool", "alarm")
            return
        try:
            plan = self.home_plan()
        except Refused as e:
            self.hold_here()
            if new:
                self.emit("hot", f"{trip.message} and {e}: holding with torque on. Operator must resolve power now; "
                          "if no clear return is available, support the arm and cut its motor supply.", "alarm")
            return
        self.emit("hot", f"{trip.message}: going home along the home route", "alarm")
        job = Job(next(self._ids), build({"do": "seq", "steps": plan, "label": "home: motor hot"}), plan)
        self.jobs[job.id] = job
        self.emit("submitted", f"job {job.id}: thermal return", job=job.id, spec=job.spec)
        self._thermal_home = job
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
        try:
            job = self.submit(spec)
        except Refused as e:
            return Outcome("refused", "spec", str(e), hint=e.hint)
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
            with suppress(Exception):      # tick latched the fault; stay alive for status and recovery
                self.tick()
            self.clock.wait()
        self._run_posted(run=False)

    def _survive(self, e: Exception):
        """The body raised (an adapter unplugged, a value out of range): fault and say so, but keep the loop and
        the daemon alive, so the flight record survives and an operator sees why. Motor power is unknown;
        commands must not resume on reconnection. Physical motor power may still be on."""
        with self.lock:
            first = not self.faulted
            self.faulted = True
            if self.enabled:
                self.power_uncertain = True
            self.hold_here()
            self._cancel_queue("the kernel faulted")
            if self.active is not None:
                self._end(self.active, Outcome("faulted", self.active.behavior.kind, explain(e),
                                               hint="treat the arm as energized; resolve motor power before reset"))
        if first:
            detail = "; motor power unconfirmed, commands suspended" if self.power_uncertain else ""
            self.emit("fault", f"control I/O failed: {explain(e)}{detail}", "alarm")

    @property
    def tool(self) -> np.ndarray:
        """Measured tool pose (4x4, base frame)."""
        return self.chain.fk(self.state.q)
