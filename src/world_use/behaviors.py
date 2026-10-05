"""Behaviors: everything that moves the robot, under one contract.

    start(kernel)  plan from where the robot is; raise Refused to refuse (nothing moves)
    tick(kernel)   called every control tick; set the command, return None to continue or an Outcome to end

Every behavior can carry an expectation. When what happens is not what was expected, it ends with a
"surprise" and the kernel holds: the policy decides what to do next, with the facts in front of it.

A spec is plain JSON: {"do": "line", "up": 0.05}. A list is a sequence. `build(spec)` makes the behavior.
"""
from __future__ import annotations

import inspect
import re
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from . import geometry, motion
from .envelope import TURN_EPS
from .errors import Refused
from .validation import FIELDS, validate
from .world import DIRECTIONS, along, heading

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
    """The docstring is the help a policy reads (`wu help STEP`): a summary sentence, any more it needs, then one line
    per parameter."""
    kind = "behavior"
    example: dict | None = None       # one spec that shows the typical use
    moves = True                      # counts as motion time (holding, waiting and asking do not)
    senses_contact = False            # True: it expects contact and judges it itself (the kernel's check steps aside)

    @classmethod
    def help(cls) -> dict:
        doc = inspect.cleandoc(cls.__doc__ or cls.kind)
        cut = end.end() if (end := re.search(r"\.\s|\.$|\n\n", doc)) else len(doc)    # first sentence or paragraph
        return dict(kind=cls.kind, summary=" ".join(doc[:cut].split()), params=doc[cut:].strip(),
                    example=cls.example)

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

    @property
    def moves(self):
        return getattr(self, "_pending", None) is None

    def start(self, k):
        self._pending = None
        if k.planner is None:
            self.prepare(k)
        else:
            self._pending, self._snapshot = k.planner.prepare(self.spec(), k)
            k.set(k.cmd.q)             # continue feedback and hold while the worker computes

    def prepare(self, k):
        self.path = np.asarray(self.plan(k), float)
        self.info = k.envelope.check_path(self.path, k.cmd.q, allow_contact=self.allow_contact)
        self.vel = np.gradient(np.vstack([k.cmd.q, self.path]), axis=0)[1:] * k.manifest.rate_hz
        self.vel[-1] = 0.0
        self.i = 0

    def ready(self, k):
        if self._pending is not None:
            if not self._pending.done():
                return False
            from .plan import same_start
            prepared = self._pending.result()
            if not same_start(self._snapshot, k):
                raise Refused("the scene or command changed while preparing this step", "stale_path",
                              "inspect the current state and replan")
            self.__dict__.update(prepared)
            self._pending = None
            k.rebias()
        if self.i == 0:
            k.check_guard()
        return True

    def tick(self, k):
        if not self.ready(k):
            return None
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
    """Joint-space move: each joint turns straight to its target. Also changes the gripper's angle.

    target_deg  {joint number: degrees}, joints numbered from 1
    delta_deg   {joint number: degrees to add}
    duration    seconds (default: from the speed limit)
    speed       peak joint speed, rad/s
    """
    kind = "joints"
    example = {"do": "joints", "delta_deg": {"6": -90}}

    def plan(self, k):
        goal = k.cmd.q.copy()
        for key, v in self.params.get("target_deg", {}).items():
            goal[int(key) - 1] = np.radians(v)
        for key, v in self.params.get("delta_deg", {}).items():
            goal[int(key) - 1] += np.radians(v)
        timing = _speed_timing(k, self.params.get("speed"))
        path, _ = motion.joint_move(k.cmd.q, goal, timing, self.params.get("duration"))
        return path


class Line(PathBehavior):
    """Straight line of the tool point; the gripper keeps its angle. At most one segment long (see the card).

    up, down, forward, back, left, right  metres along the frame's axes; at most one word of each pair
    frame              whose axes (default: work)
    duration           seconds (default: from the speed limit)
    speed              peak joint speed, rad/s
    """
    kind = "line"
    example = {"do": "line", "forward": 0.05, "up": 0.02}

    def delta(self, k):
        axes = k.world.frame(self.params.get("frame", "work")).T[:3, :3]
        return axes @ sum((np.multiply(DIRECTIONS[w], v) for w, v in self.params.items() if w in DIRECTIONS),
                          np.zeros(3))

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


def _direction(value) -> np.ndarray:
    """A unit direction from a word (down, forward, ...) or [f, l, u], in the step's frame."""
    v = np.asarray(DIRECTIONS[value] if isinstance(value, str) else value, float)
    return v / np.linalg.norm(v)


class MoveTo(PathBehavior):
    """Tool point to an absolute position in a straight line; with point, the gripper turns on the way.

    to          [forward, left, up] metres in the frame (default: work); leave it out to turn in place
    point       which way the gripper should end up pointing: down, up, forward, back, left, right, or [f, l, u]
    jaws        which way its jaws should open, the same way (default: as near to now as pointing allows)
    within_deg  how far from point it may end so the wrist and base can stay still below the turn height
                (default 5: near its base a real arm can seldom point exactly down without turning them)
    frame, duration, speed  as for line
    """
    kind = "move_to"
    example = {"do": "move_to", "to": [0.20, 0.0, 0.10], "point": "down"}

    def plan(self, k):
        p = self.params
        frame = p.get("frame", "work")
        F = k.world.frame(frame).T[:3, :3]
        T0 = k.chain.fk(k.cmd.q)
        target = k.world.to_base(frame, p["to"]) if "to" in p else T0[:3, 3].copy()
        d = target - T0[:3, 3]
        if np.linalg.norm(d) > k.manifest.max_segment_m:
            raise Refused(f"target is {100 * np.linalg.norm(d):.1f} cm away; one move may be at most "
                          f"{100 * k.manifest.max_segment_m:.0f} cm", "segment_length", "go in shorter moves (lines)")
        timing = _speed_timing(k, p.get("speed"))
        self.aimed = None
        if "point" not in p and "jaws" not in p:
            path, _, _ = motion.line(k.chain, k.cmd.q, d, timing, p.get("duration"), *k.envelope.bounds(k.cmd.q),
                                     weights=k.ik_weights)
            return path
        g = k.manifest.gripper
        if g is None:
            raise Refused("this robot has no gripper to point", "no_gripper")
        want = F @ _direction(p["point"]) if "point" in p else T0[:3, :3] @ np.asarray(g.approach, float)
        jaws = F @ _direction(p["jaws"]) if "jaws" in p else None
        try:
            R = geometry.aim(T0[:3, :3], g.approach, g.opens_along, want, jaws)
        except ValueError as e:
            raise Refused(str(e), "spec", "give jaws square to point") from None
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, target
        words = heading(F.T @ want)
        exact, failed, turn = None, None, None
        try:
            lo, hi = k.envelope.bounds(k.cmd.q)
            exact, _, _ = motion.cartesian(k.chain, k.cmd.q, T, timing, p.get("duration"), lo, hi,
                                           weights=k.ik_weights, turning=words)
            turn = k.envelope.turn_problem(np.vstack([k.cmd.q, exact]), k.manifest.rate_hz)
        except Refused as e:
            failed = e
        if exact is not None and turn is None:
            self.aimed = (want, False)
            return exact
        within = np.radians(float(p.get("within_deg", 5.0)))
        tilt = self._tilt(k, T0, target, want, jaws, timing, words)
        if tilt is not None and tilt[0] is not None and tilt[1] <= within:
            self.aimed = (want, True)
            return tilt[0]
        if exact is not None and turn is not None:      # only the turn below the turn height is in the way
            more = "" if tilt is None or tilt[0] is None else (
                f", or let it only tilt: that ends {np.degrees(tilt[1]):.0f} deg from {words} "
                f"(within_deg {np.ceil(np.degrees(tilt[1])):.0f})")
            raise Refused(str(turn), "turn_clearance", turn.hint + more, **turn.data)
        assert failed is not None
        raise failed

    def _tilt(self, k, T0, target, want, jaws, timing, words):
        """Point as near to `want` as the pitch joints alone can, with the joints the turn-clearance rule guards
        held still: (path, or None if unreachable; how far from want it ends, rad), or None where the arm has no
        such rule or its other joints do not share one axis."""
        rule = k.manifest.turn_clearance
        if rule is None:
            return None
        held = list(rule[0])
        free = [i for i in range(k.manifest.n) if i not in held]
        axes = k.chain.axes(k.cmd.q)
        n = axes[free[0]]
        if any(np.linalg.norm(np.cross(axes[i], n)) > 1e-6 for i in free[1:]):
            return None
        g = k.manifest.gripper
        a0 = T0[:3, :3] @ np.asarray(g.approach, float)
        pa, pw = a0 - (a0 @ n) * n, want - (want @ n) * n
        if np.linalg.norm(pa) < 1e-6 or np.linalg.norm(pw) < 1e-6:
            return None
        R = geometry.axis_angle(n, np.arctan2(n @ np.cross(pa, pw), pa @ pw)) @ T0[:3, :3]
        off = geometry.angle(R @ g.approach, want)
        if jaws is not None:
            j = R @ np.asarray(g.opens_along, float)
            off = max(off, min(geometry.angle(j, jaws), geometry.angle(-j, jaws)))
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, target
        q = np.asarray(k.cmd.q, float)
        lo, hi = k.envelope.bounds(q)
        for i in held:
            lo[i], hi[i] = max(lo[i], q[i] - TURN_EPS / 2), min(hi[i], q[i] + TURN_EPS / 2)
        try:
            path, _, _ = motion.cartesian(k.chain, q, T, timing, self.params.get("duration"), lo, hi,
                                          weights=k.ik_weights, turning=words)
        except Refused:
            return None, off
        return path, off

    def arrived(self, k) -> Outcome:
        if self.aimed is None:
            return super().arrived(k)
        want, tilted = self.aimed
        g, W = k.manifest.gripper, k.world.frame("work").T[:3, :3]
        R = k.chain.fk(k.cmd.q)[:3, :3]
        off = np.degrees(geometry.angle(R @ g.approach, want))
        why = f" ({off:.1f} deg off: the wrist stays still below the turn height)" if tilted and off >= 0.5 else ""
        return self.done(f"{self.describe()}: now pointing {heading(W.T @ R @ g.approach)}{why}, jaws open "
                         f"{along(W.T @ R @ g.opens_along)}", seconds=self.info["seconds"], off_deg=round(off, 1))


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
    read +-0.5-1 Nm from one tick to the next while holding still. Noise raises a threshold at most NOISE_CAP
    times, so a joint gone noisy stops on nothing rather than going numb, and never above a fragile zone's limit:
    that one is the operator's. A short median filter rejects single-tick spikes. Without an explicit joint list
    it watches the joints with real leverage along the direction of motion: a vertical push barely loads a
    vertical base axis, whose friction would only add noise.

    A joint resting on the stop it folds onto (the manifest's rest stops) is not judged, and is re-zeroed until it
    leaves: the stop takes part of its load, so arriving there or lifting off moved ~2 Nm between the reBot's
    elbow motor and its stop with nothing touched.
    """
    NOISE_K = 3.5                     # holds of up to 54 s on the reBot stayed within 3.3x (2026-09-27)
    NOISE_CAP = 2.0                   # noise raises a joint's threshold at most this many times
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
            self.bias, noise = np.asarray(st.tau, float) - k.expected_torque(st.q), np.zeros(k.manifest.n)
        self.floor = self.NOISE_K * noise
        rest = k.manifest.rest
        self.stops = [] if rest is None else [(i, rest.q[i]) for i in rest.stops]
        self.hist = deque(maxlen=window)

    def deviation(self, k: Kernel) -> np.ndarray:
        st = k.state
        self.hist.append(np.asarray(st.tau, float) - k.expected_torque(st.q))
        med = np.median(np.array(self.hist), axis=0)
        on = [i for i, stop in self.stops if abs(st.q[i] - stop) < self.STOP_ZONE]
        self.bias[on] = med[on]
        return med - self.bias

    def limits(self, limit, fragile: float | None = None) -> np.ndarray:
        """Per-joint thresholds: the requested ones, raised clear of the joint's measured noise (at most NOISE_CAP
        times), and never above a fragile zone's limit. There a noisy arm may stop on nothing: it must not press
        harder than the zone allows. `doubt` says when."""
        limit = np.broadcast_to(np.asarray(limit, float), self.floor.shape)
        raised = np.clip(self.floor, limit, self.NOISE_CAP * limit)
        return raised if fragile is None else np.minimum(raised, fragile)

    def doubt(self, used, joints=None) -> str:
        """Joints whose noise alone can cross the threshold used for them, in words: a stop there may be nothing."""
        loud = [i for i in (self.joints if joints is None else joints) if self.floor[i] > used[i] + 1e-9]
        return ("noise alone can cross the limit, so it may be nothing: "
                + ", ".join(f"j{i + 1} {self.floor[i]:.1f} Nm" for i in loud)) if loud else ""

    def exceeded(self, k, dtau, fragile: float | None = None) -> tuple[bool, np.ndarray]:
        dev = self.deviation(k)
        over = np.abs(dev[self.joints]) > self.limits(dtau, fragile)[self.joints]
        return bool(len(self.hist) == self.hist.maxlen and over.any()), dev


def fragile_dtau(k: Kernel) -> float | None:
    """The contact limit of the fragile zones the tool is in (the lowest), or None outside them."""
    tool = k.chain.fk(k.state.q)[:3, 3]
    limits = [float(b.params.get("dtau", 0.3)) for b in k.world.zones_at(tool) if b.kind == "fragile"]
    return min(limits) if limits else None


class Guarded(Line):
    """Slow straight line that stops the moment something pushes back; no contact by the end is a surprise.
    It first makes sure the arm has held still for a moment where it starts: that stillness is what contact is
    judged against.

    up, down, forward, back, left, right  as for line: the furthest it may go
    frame              whose axes (default: work)
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

    def prepare(self, k):
        self.d, self.cut = self._clip_to_surfaces(k, self.delta(k))
        self.seconds = max(1.0, float(np.linalg.norm(self.d)) / float(self.params.get("speed_mps", 0.02)) / 0.8)
        super().prepare(k)
        self.sense: ContactSense | None = None     # taken once the arm has held still for a whole baseline window

    def _clip_to_surfaces(self, k, d):
        """A known surface ends the search: no need to plan (or be able to reach) beyond it."""
        p0 = k.chain.fk(k.cmd.q)[:3, 3]
        length = float(np.linalg.norm(d))
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

    def tick(self, k):
        if not self.ready(k):
            return None
        if self.sense is None:
            # straight after another move the recent torque is that move's slowing down, not this pose at rest
            if k.still < k.residuals.window:
                k.set(k.cmd.q)
                return None
            self.sense = ContactSense(k, self.params.get("joints"), direction=self.d)
        hit, dev = self.sense.exceeded(k, float(self.params.get("dtau", 0.6)), fragile_dtau(k))
        if hit:
            k.hold_here()
            moved = float(np.linalg.norm(k.chain.fk(k.state.q)[:3, 3] - k.chain.fk(self.path[0])[:3, 3]))
            k.touched("contact", f"contact after {100 * moved:.1f} cm")
            return self.done(f"contact after {100 * moved:.1f} cm{self._noisy(k)}", moved_m=round(moved, 4),
                             torque_change=np.round(dev, 2).tolist())
        return super().tick(k)

    def _noisy(self, k) -> str:
        """Say so when a joint's noise, not the request, set its threshold, or when its noise alone can cross the
        threshold (a fragile zone's is never raised): either changes what counts as contact."""
        if self.sense is None:
            return ""
        asked, fragile = float(self.params.get("dtau", 0.6)), fragile_dtau(k)
        want = asked if fragile is None else min(asked, fragile)
        used = self.sense.limits(asked, fragile)
        raised = [i for i in self.sense.joints if used[i] > want + 1e-9]
        notes = ["torque noise raised the threshold: " + ", ".join(f"j{i + 1} {used[i]:.1f} Nm" for i in raised)
                 if raised else "", self.sense.doubt(used)]
        return f" ({'; '.join(n for n in notes if n)})" if any(notes) else ""

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
        seconds = max(self.params.get("seconds", 0.0), 1.875 * abs(to - start) / g.v_max, 0.3)
        n = max(2, int(np.ceil(seconds * k.manifest.rate_hz)))
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
    """A gripper value from params, given in mm (mm_key) or native units (native_key); lists convert elementwise.
    None when neither is given."""
    if mm_key in params:
        if g.m_per_unit is None:
            raise Refused("this gripper has no mm calibration; give it in native units", "spec", f"use {native_key}")
        v = params[mm_key]
        return [g.position(x / 1000) for x in v] if isinstance(v, (list, tuple)) else g.position(v / 1000)
    v = params.get(native_key)
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return None if v is None else float(v)


class Grip(Behavior):
    """Close until the fingers meet something, squeeze a little and hold; the width it closes on is checked.

    expect_mm  [lo, hi] opening where the fingers should meet the object; outside it, or nothing, is a surprise
    expect     the same in the gripper's native units
    start_mm   open to this first (or start, native units)
    squeeze    how much further to close after contact, native units (at most the gripper's, on the card)
    hold_effort  least gripper effort (absolute) that counts as a firm hold once squeezed; less is a surprise, a grip on
               an edge or a taper that would slip on the lift (default: not checked; requires gripper_effort sensing)
    """
    kind = "grip"
    example = {"do": "grip", "start_mm": 60, "expect_mm": [35, 45]}

    def start(self, k):
        g = k.manifest.gripper
        if g is None:
            raise Refused("this robot has no gripper", "no_gripper")
        p = self.params
        here = k.cmd.gripper if k.cmd.gripper is not None else k.state.gripper
        lo, hi = sorted((g.closed, g.open))
        self.expect, start = _native(g, p, "expect_mm", "expect"), _native(g, p, "start_mm", "start")
        if self.expect is not None:
            if not all(lo <= x <= hi for x in self.expect):
                raise Refused(f"expected grip width must be within {lo}..{hi} {g.unit}", "gripper_limit")
            self.expect.sort()
        if start is not None and not lo <= start <= hi:
            raise Refused(f"grip start must be within {lo}..{hi} {g.unit}", "gripper_limit")
        self.squeeze, self.hold = p.get("squeeze", g.squeeze), p.get("hold_effort")
        if self.squeeze > g.squeeze:
            raise Refused(f"grip squeeze must be at most {g.squeeze} {g.unit}", "gripper_limit")
        if self.hold is not None and self.hold > g.tau_max:
            raise Refused(f"grip hold_effort must be at most {g.tau_max}", "gripper_limit")
        if self.hold is not None and "gripper_effort" not in k.manifest.sensing:
            raise Refused("hold_effort requires gripper_effort sensing", "sensing")
        # Contact, closing speed and the pre-open margin scale with the gripper's own limits and units. On the reBot
        # (rad): contact at 0.6 effort or 0.1 behind the command, closing at 0.3/s, pre-opening beyond 0.02.
        self.effort, self.lag = 0.15 * g.tau_max, g.track_tol / 6
        self.pre = None
        if start is not None and abs(start - here) > 0.0045 * (hi - lo):     # open to the start width first, smoothly
            self.pre = Gripper(to=start)
            k.start_behavior(self.pre)
        self.speed = g.v_max / 15 * np.sign(g.closed - g.open)     # units/s, towards closed
        self.phase, self.contact, self.wait = "open", None, 0
        self.closed_wait = int(np.ceil(.2 * k.manifest.rate_hz))

    def tick(self, k):
        g, st, p = k.manifest.gripper, k.state, self.params
        if self.hold is not None and (st.gripper_tau is None or not np.isfinite(st.gripper_tau)):
            raise Refused("hold_effort needs a finite gripper effort reading; sensing is unavailable", "sensing")
        if self.phase == "open":
            if self.pre is not None and self.pre.tick(k) is None:
                return None
            k.check_guard()
            self.phase = "close"
        if self.phase == "close":
            effort = 0.0 if st.gripper_tau is None else abs(st.gripper_tau)
            behind = abs(st.gripper - k.cmd.gripper)
            if effort > self.effort or behind > self.lag:
                self.contact = float(st.gripper)
                squeeze = self.squeeze * np.sign(g.closed - g.open)
                self.squeeze_target = float(np.clip(self.contact + squeeze, *sorted((g.closed, g.open))))
                self.phase, self.wait = "squeeze", int(0.3 * k.manifest.rate_hz)
                return None
            nxt = k.cmd.gripper + self.speed / k.manifest.rate_hz
            if (nxt - g.closed) * np.sign(g.open - g.closed) <= 0:
                k.set_gripper(g.closed)
                # Read the response to the last command before deciding the fingers found nothing.
                self.closed_wait -= 1
                return self._no_contact(k) if self.closed_wait <= 0 else None
            k.set_gripper(nxt, self.speed)
            return None
        # Contact can leave the fingers behind the command. Approach the bounded squeeze target smoothly.
        delta = self.squeeze_target - k.cmd.gripper
        step = float(np.clip(delta, -g.v_max / k.manifest.rate_hz, g.v_max / k.manifest.rate_hz))
        k.set_gripper(k.cmd.gripper + step, step * k.manifest.rate_hz if abs(delta) > abs(step) else 0.0)
        if abs(delta) > abs(step) + 1e-9:
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
        if self.hold is not None and st.gripper_tau is not None and abs(st.gripper_tau) < self.hold:
            k.gripped(contact, attach=False)        # it may hold something all the same: going home must not let go
            return self.surprise(f"weak grip: {abs(st.gripper_tau):.2f} of effort squeezed at {contact:.2f} {g.unit}"
                                 f"{_mm(g, contact)}, less than the {self.hold:.2f} a firm hold needs",
                                 expected=self.hold, observed=round(abs(st.gripper_tau), 2),
                                 hint="the fingers met an edge or a taper: open, move a little and grip again", **data)
        if self.expect is not None and not self.expect[0] <= contact <= self.expect[1]:
            lo, hi = self.expect
            mm = p.get("expect_mm")
            want = f"{mm[0]}..{mm[1]} mm" if mm else f"{lo:.2f}..{hi:.2f} {g.unit}"
            k.gripped(contact, attach=False)        # it holds something all the same: going home must not let go
            return self.surprise(f"fingers met something at {contact:.2f} {g.unit}{_mm(g, contact)}, "
                                 f"outside the expected {want}", expected=[lo, hi], observed=contact,
                                 hint="the object is not where, or not the size, planned: open and look", **data)
        name = k.gripped(contact)
        what = f"holding {name!r}" if name else "holding"
        return self.done(f"{what} at {contact:.2f} {g.unit}" + _mm(g, contact), **data)

    def _no_contact(self, k):
        return self.surprise("the gripper closed on nothing", expected="contact", observed="no contact",
                             hint="open, check the object's position in a camera, adjust and retry")


class Grasp(Behavior):
    """A grip that searches nearby on a miss instead of ending the plan.
    The kernel knows at once when the fingers close on nothing (or on the wrong width); rather than a round trip to
    the policy, the fingers reopen, lift a little, shift to the next offset, come back down and grip again. Done at
    the first grip within expect_mm; a surprise after the last offset, holding where it tried last.

    expect_mm, start_mm  as for grip, or expect and start in native units; a start is required: each retry reopens
                         to it
    squeeze, hold_effort  as for grip (a weak grip is a miss too)
    search_mm  [[across, along], ...] offsets from where the grasp began, mm: across the jaws (the direction the
               fingers are thin in) and along them (the way they open), both level (default: 6 mm each way across,
               then 6 mm each way along)
    lift_mm    how far to lift before shifting, at most 50 (default 8; more where the base may only turn higher up,
               as the shift turns it)
    """
    kind = "grasp"
    example = {"do": "grasp", "start_mm": 30, "expect_mm": [10, 22], "search_mm": [[6, 0], [-6, 0]]}

    @property
    def moves(self):
        return self.current is not None and self.current.moves

    @property
    def senses_contact(self):
        return self.current is not None and self.current.senses_contact

    def start(self, k):
        p = self.params
        if k.manifest.gripper is None:
            raise Refused("this robot has no gripper", "no_gripper")
        search = p.get("search_mm", [[6, 0], [-6, 0], [0, 6], [0, -6]])
        self.offsets = [np.asarray(o, float) / 1000 for o in search]
        self.lift = float(p.get("lift_mm", 8.0)) / 1000
        self.grip = {key: p[key] for key in FIELDS["grip"] if key in p}
        opens = k.tool[:3, :3] @ np.asarray(k.manifest.gripper.opens_along, float)
        along = np.array([opens[0], opens[1], 0.0])
        if np.linalg.norm(along) < 0.2:                    # jaws opening up and down: search level anyway
            along = np.array([0.0, 1.0, 0.0])
        self.along = along / np.linalg.norm(along)
        self.across = np.cross([0.0, 0.0, 1.0], self.along)
        self.at, self.tries, self.misses = np.zeros(2), 0, []
        self.queue: deque[Behavior] = deque([Grip(**self.grip)])
        self.current: Behavior | None = None

    def _shift(self, k, offset):
        """One smooth motion from the current offset to the next: up, across, down. A sideways move turns the base,
        so where the body only turns above a height the lift goes at least that high (corners rounded above it)."""
        d = (offset - self.at)[0] * self.across + (offset - self.at)[1] * self.along
        fwd, left, _ = k.world.frame("work").axes if "work" in k.world.frames else np.eye(3)
        self.at = offset
        lift, blend = self.lift, min(0.004, self.lift / 2)
        need = k.envelope.turn_height()
        if need is not None:
            p = k.chain.fk(k.cmd.q)[:3, 3]
            up = float(k.world.from_base("work", p)[2]) if "work" in k.world.frames else float(p[2])
            lift = max(lift, need + blend + 0.002 - up)
        return Lines(legs=[[0.0, 0.0, lift], [float(d @ fwd), float(d @ left), 0.0], [0.0, 0.0, -lift]], blend=blend)

    def tick(self, k):
        while True:
            if self.current is None:
                self.current = self.queue.popleft()
                k.start_behavior(self.current)
            out = self.current.tick(k)
            if out is None:
                return None
            self.current = None
            if out.kind != "grip":
                if not out.ok:
                    return out
                continue
            self.tries += 1
            if out.ok:
                return self.done(f"{out.message} (try {self.tries}"
                                 + (f", {1000 * self.at[0]:+.0f} mm across, {1000 * self.at[1]:+.0f} mm along)"
                                    if self.tries > 1 else ")"),
                                 **out.data, tries=self.tries, offset_mm=[round(1000 * x, 1) for x in self.at])
            self.misses.append(out.message)
            if self.tries > len(self.offsets):
                return self.surprise(f"no grip after {self.tries} tries: {'; '.join(self.misses)}",
                                     expected=out.expected, observed=out.observed,
                                     hint="look at the object from above and plan the grasp again", tries=self.tries)
            nxt = self.offsets[self.tries - 1]
            k.emit("grasp_retry", f"missed ({out.message}); retrying {1000 * nxt[0]:+.0f} mm across, "
                                  f"{1000 * nxt[1]:+.0f} mm along", offset_mm=[round(1000 * x, 1) for x in nxt])
            opening = {"to": self.grip["start"]} if "start" in self.grip else {"aperture_mm": self.grip["start_mm"]}
            self.queue.extend([Gripper(**opening), self._shift(k, nxt), Grip(**self.grip)])

    def stop(self, k, reason):
        k.hold_here()
        return Outcome("stopped", self.kind, f"stopped after {self.tries} grip tries: {reason}")


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
    """Stop and ask; the arm holds until the question is answered. A check assumes the expected answer and says so.

    ask     the question, e.g. "is the black loop between the jaws?"
    view    the camera that answers it best (look at it); roi = [x0, y0, x1, y1] in that image
    expect  the answer that means carry on (default "yes"); any other answer ends the plan. null: any answer carries
            on, and is kept in the outcome with where the tool was (a measurement, like "512,300" in a picture)
    """
    kind = "checkpoint"
    example = {"do": "checkpoint", "ask": "is the block between the jaws?", "view": "side"}
    moves = False

    def start(self, k):
        self.asked = False

    def tick(self, k):
        expect = self.params.get("expect", "yes")
        expect = None if expect is None else str(expect)
        if not self.asked:
            k.ask(dict(ask=self.params["ask"], view=self.params.get("view"), roi=self.params.get("roi"),
                       expect=expect))
            self.asked = True
        k.set(k.cmd.q, np.zeros(k.manifest.n))
        answer = k.take_answer()
        if answer is None:
            return None
        if expect is None:
            tool = k.world.from_base("work", k.chain.fk(k.state.q)[:3, 3])
            return self.done(f"{self.params['ask']} -> {answer}", answer=answer, ask=self.params["ask"],
                             tool=np.round(tool, 4).tolist())
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
        self.answers: list[dict] = []     # what free checkpoints were told, however deep, in order

    def tick(self, k):
        while True:
            n = len(self.steps)
            if self.current is None and self.i >= n:
                return self.done(f"{n} step{'' if n == 1 else 's'} done", steps=[o.message for o in self.results],
                                 **self._answers())
            where = f"step {self.i + 1}/{n}"
            try:
                if self.current is None:
                    self.current = self.steps[self.i]
                    k.emit("step", f"{where}: {self.current.describe()}", step=self.i + 1)
                    k.start_behavior(self.current)
                out = self.current.tick(k)
            except Refused as e:             # the kernel holds and ends the job; say where in the plan it was
                self.answers += [{**a, "step": self.i + 1} for a in e.data.get("answers", [])]
                raise Refused(f"{where}: {e}", e.rule, e.hint, **{**e.data, "step": self.i + 1, "of": n,
                                                                   **self._answers()}) from None
            if out is None:
                return None
            self.results.append(out)
            self.current = None
            self.i += 1
            if "tool" in out.data and "answer" in out.data:
                self.answers.append(dict(step=self.i, ask=out.data["ask"], answer=out.data["answer"],
                                         tool=out.data["tool"]))
            self.answers += [{**a, "step": self.i} for a in out.data.get("answers", [])]     # this plan's step
            if not out.ok:
                out.data = {**out.data, "step": self.i, "of": n, **self._answers()}
                out.message = f"{where}: {out.message}"
                return out
            k.emit("step_done", f"{where}: {out.message}", step=self.i)

    def _answers(self) -> dict:
        return {"answers": self.answers} if self.answers else {}

    def stop(self, k, reason):
        k.hold_here()
        return Outcome("stopped", self.kind, f"stopped at step {self.i + 1}/{len(self.steps)}: {reason}",
                       dict(step=self.i + 1))

    def describe(self):
        return self.label or f"sequence of {len(self.steps)}"

    def spec(self):
        return {"do": "seq", "steps": [s.spec() for s in self.steps], **({"label": self.label} if self.label else {})}


REGISTRY: dict[str, type[Behavior]] = {c.kind: c for c in (Joints, Line, Lines, MoveTo, Guarded, Touchdown, Gripper,
                                                            Grip, Grasp, Hold, Checkpoint, Sequence)}
BUILTINS = frozenset(REGISTRY.values())


def register(cls: type[Behavior]) -> type[Behavior]:
    """Add a behavior kind (usable as a decorator). Plugins use this; specs then accept {"do": cls.kind}."""
    REGISTRY[cls.kind] = cls
    return cls


def walk(b: Behavior):
    """b and every step of the sequences in it, however deep."""
    yield b
    for step in b.steps if isinstance(b, Sequence) else ():
        yield from walk(step)


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
    if not isinstance(kind, str):
        raise Refused("do must name a behavior", "spec")
    if kind not in REGISTRY:
        raise Refused(f"unknown behavior {kind!r}; known: {sorted(REGISTRY)}", "spec")
    label = spec.pop("label", None)
    if label is not None and not isinstance(label, str):
        raise Refused("label must be a string", "spec")
    if REGISTRY[kind].__module__ == __name__:
        validate(kind, spec)
    if kind == "seq":
        return Sequence(spec.pop("steps"), label, **spec)
    return REGISTRY[kind](label, **spec)
