"""Behaviors: everything that moves the robot, under one contract.

    start(kernel)  plan from where the robot is; raise Refused to refuse (nothing moves)
    tick(kernel)   called every control tick; set the command, return None to continue or an Outcome to end

Every behavior can carry an expectation. When what happens is not what was expected, it ends with a
"surprise" and the kernel holds: the policy decides what to do next, with the facts in front of it.

A spec is plain JSON: {"do": "line", "up": 0.05}. A list is a sequence. `build(spec)` makes the behavior.
"""
import inspect
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from . import motion
from .errors import Refused

if TYPE_CHECKING:
    from .kernel import Kernel

STATUSES = ("done", "refused", "surprise", "stopped", "faulted")    # how a behavior can end


@dataclass
class Outcome:
    status: str                       # one of STATUSES
    kind: str                         # the behavior that ended
    message: str
    data: dict = field(default_factory=dict)
    expected: object = None
    observed: object = None
    hint: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "done"

    def to_dict(self) -> dict:
        d = dict(status=self.status, kind=self.kind, message=self.message)
        for k in ("data", "expected", "observed", "hint"):
            v = getattr(self, k)
            if v not in (None, "", {}):
                d[k] = v
        return d


class Behavior:
    """The docstring is the help a policy reads (`wu help STEP`): a summary line, then one line per parameter."""
    kind = "behavior"
    example: dict | None = None       # one spec that shows the typical use
    moves = True                      # counts as motion time (holding, waiting and asking do not)
    senses_contact = False            # True: it expects contact and judges it itself (the kernel's check steps aside)

    @classmethod
    def help(cls) -> dict:
        summary, _, params = inspect.cleandoc(cls.__doc__ or cls.kind).partition("\n")
        return dict(kind=cls.kind, summary=summary.strip(), params=inspect.cleandoc(params), example=cls.example)

    def __init__(self, label: str | None = None, **params):
        self.label, self.params = label, params

    def start(self, k: Kernel) -> None:
        pass

    def tick(self, k: Kernel) -> Outcome | None:
        return self.done("nothing to do")

    def stop(self, k: Kernel, reason: str) -> Outcome:
        """Interrupted from outside: hold where the arm really is, so nothing keeps pressing."""
        k.hold_here()
        return Outcome("stopped", self.kind, reason)

    def done(self, message: str = "", **data) -> Outcome:
        return Outcome("done", self.kind, message or self.describe(), data)

    def surprise(self, message: str, expected=None, observed=None, hint: str = "", **data) -> Outcome:
        return Outcome("surprise", self.kind, message, data, expected, observed, hint)

    def describe(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.kind}({args})" + (f" [{self.label}]" if self.label else "")

    def spec(self) -> dict:
        return {"do": self.kind, **self.params, **({"label": self.label} if self.label else {})}


# -- motion along a precomputed path ----------------------------------------------------------------

class PathBehavior(Behavior):
    """Plans a joint path at start, has the envelope check all of it, then plays it one row per tick."""
    allow_contact = False

    def plan(self, k: Kernel) -> np.ndarray:
        raise NotImplementedError

    def start(self, k):
        self.path = np.asarray(self.plan(k), float)
        self.info = k.envelope.check_path(self.path, k.cmd.q, allow_contact=self.allow_contact)
        self.vel = np.gradient(np.vstack([k.cmd.q, self.path]), axis=0)[1:] * k.manifest.rate_hz
        self.vel[-1] = 0.0
        self.i = 0

    def tick(self, k):
        if self.i >= len(self.path):
            return self.arrived(k)
        k.set(self.path[self.i], self.vel[self.i])
        self.i += 1
        return None

    def arrived(self, k) -> Outcome:
        return self.done(self.describe(), seconds=self.info["seconds"])


def _speed_timing(k: Kernel, speed=None):
    t = k.timing
    return t if speed is None else motion.Timing(t.rate_hz, float(speed), t.accel, t.min_s, t.ik_tol)


class Joints(PathBehavior):
    """Joint-space move: each joint turns straight to its target. The only step that changes the gripper's angle.

    target_deg  {joint number: degrees}, joints numbered from 1
    delta_deg   {joint number: degrees to add}
    duration    seconds (default: from the speed limit)
    speed       peak joint speed, rad/s
    """
    kind = "joints"
    example = {"do": "joints", "delta_deg": {"6": -90}}

    def plan(self, k):
        goal = k.cmd.q.copy()
        for key, v in (self.params.get("target_deg") or {}).items():
            goal[_joint(k, key)] = np.radians(float(v))
        for key, v in (self.params.get("delta_deg") or {}).items():
            goal[_joint(k, key)] += np.radians(float(v))
        if not self.params.get("target_deg") and not self.params.get("delta_deg"):
            raise Refused("joints needs target_deg or delta_deg", "spec")
        timing = _speed_timing(k, self.params.get("speed"))
        path, _ = motion.joint_move(k.cmd.q, goal, timing, self.params.get("duration"))
        return path


def _joint(k, key) -> int:
    i = int(key) - 1
    if not 0 <= i < k.manifest.n:
        raise Refused(f"joint numbers are 1..{k.manifest.n}, got {key!r}", "spec")
    return i


class Line(PathBehavior):
    """Straight line of the tool point; the gripper keeps its angle. At most one segment long (see the card).

    forward, left, up  metres along the frame's axes (negative: back, right, down)
    frame              whose axes (default: work)
    duration           seconds (default: from the speed limit)
    speed              peak joint speed, rad/s
    """
    kind = "line"
    example = {"do": "line", "forward": 0.05, "up": 0.02}

    def delta(self, k):
        fwd, left, up = k.world.frame(self.params.get("frame", "work")).axes
        p = self.params
        return p.get("forward", 0.0) * fwd + p.get("left", 0.0) * left + p.get("up", 0.0) * up

    def plan(self, k):
        d = self.delta(k)
        if np.linalg.norm(d) > k.manifest.max_segment_m:
            raise Refused(f"line is {100 * np.linalg.norm(d):.1f} cm; one segment may be at most "
                          f"{100 * k.manifest.max_segment_m:.0f} cm", "segment_length", "split it into shorter lines")
        path, _, _ = motion.line(k.chain, k.cmd.q, d, _speed_timing(k, self.params.get("speed")),
                                 self.params.get("duration"), *k.envelope.bounds(k.cmd.q), weights=k.ik_weights)
        return path


class Lines(PathBehavior):
    """Several straight legs as one smooth motion: corners are rounded, so the arm does not stop at each one.

    legs      [[forward, left, up], ...] metres, each leg from the end of the last
    blend     how much to round each corner, metres (default 0.02)
    frame, duration, speed  as for line
    """
    kind = "lines"
    example = {"do": "lines", "legs": [[0.05, 0, 0], [0, 0.03, 0]], "blend": 0.02}

    def plan(self, k):
        fwd, left, up = k.world.frame(self.params.get("frame", "work")).axes
        legs = [a * fwd + b * left + c * up for a, b, c in self.params["legs"]]
        for d in legs:
            if np.linalg.norm(d) > k.manifest.max_segment_m:
                raise Refused("a leg is longer than one segment may be", "segment_length", "split it")
        path, _, _ = motion.polyline(k.chain, k.cmd.q, legs, _speed_timing(k, self.params.get("speed")),
                                     self.params.get("blend", 0.02), self.params.get("duration"),
                                     *k.envelope.bounds(k.cmd.q), weights=k.ik_weights)
        return path


class MoveTo(PathBehavior):
    """Tool point to an absolute position, in a straight line; the gripper keeps its angle.

    to        [forward, left, up] metres in the frame (default: work)
    frame, duration, speed  as for line
    """
    kind = "move_to"
    example = {"do": "move_to", "to": [0.35, -0.05, 0.30]}

    def plan(self, k):
        target = k.world.to_base(self.params.get("frame", "work"), self.params["to"])
        here = k.chain.fk(k.cmd.q)[:3, 3]
        d = target - here
        if np.linalg.norm(d) > k.manifest.max_segment_m:
            raise Refused(f"target is {100 * np.linalg.norm(d):.1f} cm away; one move may be at most "
                          f"{100 * k.manifest.max_segment_m:.0f} cm", "segment_length", "go in shorter moves (lines)")
        path, _, _ = motion.line(k.chain, k.cmd.q, d, _speed_timing(k, self.params.get("speed")),
                                 self.params.get("duration"), *k.envelope.bounds(k.cmd.q), weights=k.ik_weights)
        return path


# -- contact ------------------------------------------------------------------------------------------

class Residuals:
    """Joint torque the arm's own weight does not explain (measured minus the gravity model), over the last few
    tenths of a second: the baseline a contact check judges against, and how noisy each joint is right now."""

    def __init__(self, rate_hz: float, window_s: float = 0.3, need_s: float = 0.1):
        self.samples: deque[np.ndarray] = deque(maxlen=max(1, int(window_s * rate_hz)))
        self.need = max(1, int(need_s * rate_hz))

    def push(self, r):
        self.samples.append(np.asarray(r, float))

    def clear(self):
        self.samples.clear()

    @property
    def ready(self) -> bool:
        return len(self.samples) >= self.need

    @property
    def window(self) -> int:
        return self.samples.maxlen or 1

    def baseline(self) -> tuple[np.ndarray, np.ndarray]:
        """(median, robust noise) per joint. Refuses when empty: a NaN bias would make every check pass."""
        if not self.samples:
            raise RuntimeError("no torque readings since the torque came on")
        s = np.array(self.samples)
        bias = np.median(s, axis=0)
        return bias, 1.4826 * np.median(np.abs(s - bias), axis=0)


class ContactSense:
    """Torque change that the arm's own weight does not explain.

    The bias (measured minus model) is the median of the kernel's last few tenths of a second, so the URDF's mass
    errors and cable loads cancel and one noisy reading does not shift every judgement after it. Their spread is
    the joint's noise, and no threshold is set below NOISE_K times it: a real reBot's loaded shoulder and elbow
    read +-0.5-1 Nm from one tick to the next while holding still. A short median filter rejects single-tick
    spikes. Without an explicit joint list it watches the joints with real leverage along the direction of
    motion: a vertical push barely loads a vertical base axis, whose friction would only add noise.

    A joint resting on the stop it folds onto (the manifest's rest stops) is not judged, and is re-zeroed until it
    leaves: the stop takes part of its load, so arriving there or lifting off moved ~2 Nm between the reBot's
    elbow motor and its stop with nothing touched.
    """
    NOISE_K = 3.5                     # holds of up to 54 s on the reBot stayed within 3.3x (2026-09-27)
    STOP_ZONE = 0.06                  # rad from a rest stop within which a joint is not judged

    def __init__(self, k: Kernel, joints=None, direction=None, window=5):
        st = k.state
        if st.tau is None:
            raise Refused("this robot does not report joint torque; guarded moves need it", "sensing")
        if joints is not None:
            self.joints = [int(j) - 1 for j in joints]
        elif direction is not None and np.linalg.norm(direction) > 0:
            lever = np.abs(k.chain.jacobian(st.q)[:3].T @ (np.asarray(direction) / np.linalg.norm(direction)))
            self.joints = [int(i) for i in np.where(lever >= 0.3 * lever.max())[0]]
        else:
            self.joints = list(range(k.manifest.n))
        if k.residuals.samples:
            self.bias, noise = k.residuals.baseline()
        else:                         # a heat emergency may start home before any reading: judge from this one
            self.bias, noise = np.asarray(st.tau, float) - k.chain.gravity(st.q), np.zeros(k.manifest.n)
        self.floor = self.NOISE_K * noise
        rest = k.manifest.rest
        self.stops = [] if rest is None else [(i, rest.q[i]) for i in rest.stops]
        self.hist = deque(maxlen=window)

    def deviation(self, k: Kernel) -> np.ndarray:
        st = k.state
        self.hist.append(np.asarray(st.tau, float) - k.chain.gravity(st.q))
        med = np.median(np.array(self.hist), axis=0)
        on = [i for i, stop in self.stops if abs(st.q[i] - stop) < self.STOP_ZONE]
        self.bias[on] = med[on]
        return med - self.bias

    def limits(self, limit) -> np.ndarray:
        """Per-joint thresholds: the requested ones, but never inside the joint's measured noise."""
        return np.maximum(np.broadcast_to(np.asarray(limit, float), self.floor.shape), self.floor)

    def exceeded(self, k, dtau) -> tuple[bool, np.ndarray]:
        dev = self.deviation(k)
        over = np.abs(dev[self.joints]) > self.limits(dtau)[self.joints]
        return bool(len(self.hist) == self.hist.maxlen and over.any()), dev


class Guarded(Line):
    """Slow straight line that stops the moment something pushes back; no contact by the end is a surprise.
    It first makes sure the arm has held still for a moment where it starts: that stillness is what contact is
    judged against.

    forward, left, up  metres along the frame's axes: the furthest it may go
    dtau               joint torque change that counts as contact, Nm (default 0.6; fragile zones use less)
    expect_contact     false to probe: then no contact is success
    speed_mps          tool speed, m/s (default 0.02)
    joints             joint numbers to watch (default: the ones with leverage along the motion)
    """
    kind = "guarded"
    example = {"do": "guarded", "forward": 0.03, "dtau": 0.6}
    allow_contact = True
    senses_contact = True

    OVERSHOOT = 0.02                  # plan at most this far into a surface the world knows about

    def start(self, k):
        self.d, self.cut = self._clip_to_surfaces(k, self.delta(k))
        self.seconds = max(1.0, float(np.linalg.norm(self.d)) / float(self.params.get("speed_mps", 0.02)) / 0.8)
        super().start(k)
        self.sense: ContactSense | None = None     # taken once the arm has held still for a whole baseline window

    def _clip_to_surfaces(self, k, d):
        """A known surface ends the search: no need to plan (or be able to reach) beyond it."""
        p0 = k.chain.fk(k.cmd.q)[:3, 3]
        length = float(np.linalg.norm(d))
        if length < 1e-9:
            raise Refused("guarded move has zero length", "zero_length")
        u = d / length
        for s in np.arange(0.005, length + 1e-9, 0.005):
            for box in k.world.solids():
                if box.depth(p0 + u * s) > self.OVERSHOOT:
                    return u * s, box.name
        return d, None

    def plan(self, k):
        if np.linalg.norm(self.d) > k.manifest.max_segment_m:
            raise Refused("guarded move is longer than one segment may be", "segment_length")
        # mostly constant speed (cruise profile), so acceleration torques do not look like contact
        path, _, _ = motion.line(k.chain, k.cmd.q, self.d, k.timing, self.seconds, *k.envelope.bounds(k.cmd.q),
                                 weights=k.ik_weights, shape=motion.cruise)
        return path

    def dtau(self, k) -> float:
        base = float(self.params.get("dtau", 0.6))
        tool = k.chain.fk(k.state.q)[:3, 3]
        zones = [b for b in k.world.zones_at(tool) if b.kind == "fragile"]
        return min([base] + [float(b.params.get("dtau", 0.3)) for b in zones])

    def tick(self, k):
        if self.sense is None:
            # straight after another move the recent torque is that move's slowing down, not this pose at rest
            if k.still < k.residuals.window:
                k.set(k.cmd.q)
                return None
            self.sense = ContactSense(k, self.params.get("joints"), direction=self.d)
        hit, dev = self.sense.exceeded(k, self.dtau(k))
        if hit:
            k.hold_here()
            moved = float(np.linalg.norm(k.chain.fk(k.state.q)[:3, 3] - k.chain.fk(self.path[0])[:3, 3]))
            k.touched("contact", f"contact after {100 * moved:.1f} cm")
            return self.done(f"contact after {100 * moved:.1f} cm{self._noisy(k)}", moved_m=round(moved, 4),
                             torque_change=np.round(dev, 2).tolist())
        return super().tick(k)

    def _noisy(self, k) -> str:
        """Say so when a joint's noise, not the request, set its threshold: it changes what counts as contact."""
        if self.sense is None:
            return ""
        want = self.dtau(k)
        used = self.sense.limits(want)
        raised = [i for i in self.sense.joints if used[i] > want + 1e-9]
        return (" (torque noise raised the threshold: " + ", ".join(f"j{i + 1} {used[i]:.1f} Nm" for i in raised) + ")"
                if raised else "")

    def arrived(self, k):
        if self.params.get("expect_contact", True):
            where = f", {100 * self.OVERSHOOT:.0f} cm past where {self.cut!r} should be" if self.cut else ""
            return self.surprise(f"reached the end of the guarded move without contact{where}{self._noisy(k)}",
                                 expected="contact", observed="no contact",
                                 hint="the world model is off here: look, then correct it" if self.cut else
                                 "the surface is further than planned, or something held turned in the grip instead "
                                 "of pushing back: look first. To set a held thing down at a known height, use "
                                 "guarded with expect_contact false")
        return self.done("no contact, as expected")


class Touchdown(Guarded):
    """Guarded move straight down until contact: find a table, set an object down.

    max      how far down it may go, metres (default 0.06); over a known surface, at most 2 cm past its top
    dtau, speed_mps, joints  as for guarded
    """
    kind = "touchdown"
    example = {"do": "touchdown", "max": 0.06}

    def delta(self, k):
        return np.array([0.0, 0.0, -float(self.params.get("max", 0.06))])


# -- gripper ------------------------------------------------------------------------------------------

class Gripper(Behavior):
    """Open or close the gripper to a given opening.

    aperture_mm  opening between the fingers, mm
    to           the same in the gripper's native units (see the card)
    seconds      how long to take (default: as fast as its speed limit allows)
    """
    kind = "gripper"
    example = {"do": "gripper", "aperture_mm": 60}

    def start(self, k):
        g = k.manifest.gripper
        if g is None:
            raise Refused("this robot has no gripper", "no_gripper")
        to = _native(g, self.params, "aperture_mm", "to")
        lo, hi = sorted((g.closed, g.open))
        if not lo <= to <= hi:
            raise Refused(f"gripper target {to:.2f} {g.unit} is outside {lo}..{hi}", "gripper_limit")
        start = k.cmd.gripper if k.cmd.gripper is not None else k.state.gripper
        seconds = float(self.params.get("seconds", 0.0))
        seconds = max(seconds, 1.875 * abs(to - start) / g.v_max, 0.3)
        n = max(2, int(seconds * k.manifest.rate_hz))
        s = motion.minjerk(np.arange(1, n + 1) / n)
        self.traj = start + (to - start) * s
        self.v = np.gradient(np.concatenate([[start], self.traj]))[1:] * k.manifest.rate_hz
        self.v[-1] = 0.0
        self.i, self.settle = 0, int(0.3 * k.manifest.rate_hz)

    def tick(self, k):
        if self.i < len(self.traj):
            k.set_gripper(self.traj[self.i], self.v[self.i])
            self.i += 1
            return None
        if self.settle > 0:
            self.settle -= 1
            return None
        g, st = k.manifest.gripper, k.state
        return self.done(f"gripper at {st.gripper:.2f} {g.unit}" + _mm(g, st.gripper), position=round(st.gripper, 3),
                         effort=None if st.gripper_tau is None else round(st.gripper_tau, 2))


def _mm(g, pos):
    a = g.aperture(pos)
    return "" if a is None else f" ({1000 * a:.0f} mm)"


def _native(g, params: dict, mm_key: str, native_key: str):
    """A gripper value from params, given in mm (mm_key) or native units (native_key); lists convert elementwise."""
    if mm_key in params:
        if g.m_per_unit is None:
            raise Refused("this gripper has no mm calibration; give it in native units", "spec", f"use {native_key}")
        v = params[mm_key]
        return [g.position(float(x) / 1000) for x in v] if isinstance(v, (list, tuple)) else g.position(float(v) / 1000)
    if native_key in params:
        v = params[native_key]
        return [float(x) for x in v] if isinstance(v, (list, tuple)) else float(v)
    raise Refused(f"give {mm_key} (or {native_key}, native units)", "spec")


class Grip(Behavior):
    """Close until the fingers meet something, squeeze a little and hold; the width it closes on is checked.

    expect_mm  [lo, hi] opening where the fingers should meet the object; outside it, or nothing, is a surprise
    expect     the same in the gripper's native units
    start_mm   open to this first (or start, native units)
    squeeze    how much further to close after contact, native units (default 0.1)
    effort     gripper effort that counts as contact (default 0.6)
    lag        how far the gripper may fall behind its command before that counts as contact (default 0.1)
    speed      closing speed, native units per second (default 0.3)
    min        close no further than this, native units (default: fully closed)
    """
    kind = "grip"
    example = {"do": "grip", "start_mm": 60, "expect_mm": [35, 45]}

    def start(self, k):
        g = k.manifest.gripper
        if g is None:
            raise Refused("this robot has no gripper", "no_gripper")
        p = self.params
        here = k.cmd.gripper if k.cmd.gripper is not None else k.state.gripper
        self.expect = sorted(_native(g, p, "expect_mm", "expect")) if ("expect_mm" in p or "expect" in p) else None
        start = _native(g, p, "start_mm", "start") if ("start_mm" in p or "start" in p) else None
        self.pre = None
        if start is not None and abs(start - here) > 0.02:             # open to the start width first, smoothly
            self.pre = Gripper(to=start)
            self.pre.start(k)
        self.speed = float(p.get("speed", 0.3)) * np.sign(g.closed - g.open)     # units/s, towards closed
        self.effort, self.lag = float(p.get("effort", 0.6)), float(p.get("lag", 0.1))
        self.floor = float(p.get("min", g.closed))
        self.phase, self.contact, self.wait = "open", None, 0

    def tick(self, k):
        g, st, p = k.manifest.gripper, k.state, self.params
        if self.phase == "open":
            if self.pre is not None and self.pre.tick(k) is None:
                return None
            self.phase = "close"
        if self.phase == "close":
            effort = 0.0 if st.gripper_tau is None else abs(st.gripper_tau)
            behind = abs(st.gripper - k.cmd.gripper)
            if effort > self.effort or behind > self.lag:
                self.contact = float(st.gripper)
                squeeze = float(p.get("squeeze", 0.1)) * np.sign(g.closed - g.open)
                k.set_gripper(self.contact + squeeze)
                self.phase, self.wait = "squeeze", int(0.3 * k.manifest.rate_hz)
                return None
            nxt = k.cmd.gripper + self.speed / k.manifest.rate_hz
            if (nxt - self.floor) * np.sign(g.open - g.closed) <= 0:
                k.set_gripper(self.floor)
                return self._no_contact(k)
            k.set_gripper(nxt, self.speed)
            return None
        self.wait -= 1                                    # squeeze: let it settle, then judge
        if self.wait > 0:
            return None
        contact = self.contact
        assert contact is not None, "the squeeze phase starts at a contact"
        k.touched("grip", f"grip contact at {contact:.2f} {g.unit}")
        data = dict(contact_at=round(contact, 3),
                    holding_effort=None if st.gripper_tau is None else round(st.gripper_tau, 2))
        a = g.aperture(contact)
        if a is not None:
            data["aperture_mm"] = round(1000 * a, 1)
        if self.expect is not None and not self.expect[0] <= contact <= self.expect[1]:
            lo, hi = self.expect
            mm = p.get("expect_mm")
            want = f"{mm[0]}..{mm[1]} mm" if mm else f"{lo:.2f}..{hi:.2f} {g.unit}"
            return self.surprise(f"fingers met something at {contact:.2f} {g.unit}{_mm(g, contact)}, "
                                 f"outside the expected {want}", expected=[lo, hi], observed=contact,
                                 hint="the object is not where, or not the size, planned: open and look", **data)
        name = k.gripped(contact)
        what = f"holding {name!r}" if name else "holding"
        return self.done(f"{what} at {contact:.2f} {g.unit}" + _mm(g, contact), **data)

    def _no_contact(self, k):
        return self.surprise("the gripper closed on nothing", expected="contact", observed="no contact",
                             hint="open, check the object's position in a camera, adjust and retry")


# -- waiting, asking, composing -----------------------------------------------------------------------

class Hold(Behavior):
    """Hold still.

    seconds  how long (omit it to hold until stopped or replaced)
    """
    kind = "hold"
    example = {"do": "hold", "seconds": 2}
    moves = False

    def start(self, k):
        seconds = self.params.get("seconds")
        self.left = None if seconds is None else int(float(seconds) * k.manifest.rate_hz)

    def tick(self, k):
        k.set(k.cmd.q, np.zeros(k.manifest.n))
        if self.left is None:
            return None
        self.left -= 1
        return self.done(f"held {self.params['seconds']} s") if self.left <= 0 else None


class Checkpoint(Behavior):
    """Stop and ask; the arm holds until `wu answer`. A check assumes the expected answer and says so.

    ask     the question, e.g. "is the black loop between the jaws?"
    view    the camera that answers it best (`wu look VIEW`); roi = [x0, y0, x1, y1] in that image
    expect  the answer that means carry on (default "yes"); any other answer ends the plan
    """
    kind = "checkpoint"
    example = {"do": "checkpoint", "ask": "is the block between the jaws?", "view": "side"}
    moves = False

    def start(self, k):
        self.asked = False

    def tick(self, k):
        if not self.asked:
            k.ask(dict(ask=self.params["ask"], view=self.params.get("view"), roi=self.params.get("roi"),
                       expect=str(self.params.get("expect", "yes"))))
            self.asked = True
        k.set(k.cmd.q, np.zeros(k.manifest.n))
        answer = k.take_answer()
        if answer is None:
            return None
        expect = str(self.params.get("expect", "yes"))
        if answer.strip().lower() == expect.lower():
            return self.done(f"{self.params['ask']} -> {answer}", answer=answer)
        return self.surprise(f"{self.params['ask']} -> {answer}", expected=expect, observed=answer)


class Sequence(Behavior):
    """Steps in order; the first one that does not end "done" ends the sequence. A plain list is one too.

    steps  the steps
    """
    kind = "seq"
    example = {"do": "seq", "label": "lift, wait", "steps": [{"do": "line", "up": 0.05}, {"do": "hold", "seconds": 1}]}

    def __init__(self, steps, label=None, **params):
        super().__init__(label, **params)
        self.steps = [s if isinstance(s, Behavior) else build(s) for s in steps]
        self.i, self.current = 0, None

    @property
    def moves(self):
        return self.current is not None and self.current.moves

    @property
    def senses_contact(self):
        return self.current is not None and self.current.senses_contact

    def start(self, k):
        self.i, self.current = 0, None
        self.results: list[Outcome] = []

    def tick(self, k):
        while True:
            if self.current is None:
                if self.i >= len(self.steps):
                    return self.done(f"{len(self.steps)} steps done", steps=[o.message for o in self.results])
                self.current = self.steps[self.i]
                k.emit("step", f"step {self.i + 1}/{len(self.steps)}: {self.current.describe()}", step=self.i + 1)
                k.rebias()
                k.envelope.context = f"step {self.i + 1}/{len(self.steps)}: {self.current.describe()}"
                try:
                    self.current.start(k)
                except Refused as e:
                    return Outcome("refused", self.current.kind, f"step {self.i + 1}/{len(self.steps)}: {e}",
                                   dict(step=self.i + 1, rule=e.rule), hint=e.hint)
            out = self.current.tick(k)
            if out is None:
                return None
            self.results.append(out)
            self.current = None
            self.i += 1
            if not out.ok:
                out.data = {**out.data, "step": self.i, "of": len(self.steps)}
                out.message = f"step {self.i}/{len(self.steps)}: {out.message}"
                return out
            k.emit("step_done", f"step {self.i}/{len(self.steps)}: {out.message}", step=self.i)

    def stop(self, k, reason):
        k.hold_here()
        return Outcome("stopped", self.kind, f"stopped at step {self.i + 1}/{len(self.steps)}: {reason}",
                       dict(step=self.i + 1))

    def describe(self):
        return self.label or f"sequence of {len(self.steps)}"

    def spec(self):
        return {"do": "seq", "steps": [s.spec() for s in self.steps], **({"label": self.label} if self.label else {})}


REGISTRY: dict[str, type[Behavior]] = {c.kind: c for c in (Joints, Line, Lines, MoveTo, Guarded, Touchdown, Gripper,
                                                            Grip, Hold, Checkpoint, Sequence)}


def register(cls: type[Behavior]) -> type[Behavior]:
    """Add a behavior kind (usable as a decorator). Plugins use this; specs then accept {"do": cls.kind}."""
    REGISTRY[cls.kind] = cls
    return cls


def build(spec) -> Behavior:
    """Behavior from a spec: a dict with "do", a list (= sequence), or a Behavior (returned as is)."""
    if isinstance(spec, Behavior):
        return spec
    if isinstance(spec, list):
        return Sequence(spec)
    if not isinstance(spec, dict) or "do" not in spec:
        raise Refused(f"a step must be a dict with a 'do' key or a list of steps, got {spec!r}", "spec")
    spec = dict(spec)
    kind = spec.pop("do")
    if kind not in REGISTRY:
        raise Refused(f"unknown behavior {kind!r}; known: {sorted(REGISTRY)}", "spec")
    label = spec.pop("label", None)
    if kind == "seq":
        return Sequence(spec.pop("steps"), label, **spec)
    return REGISTRY[kind](label, **spec)
