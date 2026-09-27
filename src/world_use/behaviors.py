"""Behaviors: everything that moves the robot, under one contract.

    start(kernel)  plan from where the robot is; raise Refused to refuse (nothing moves)
    tick(kernel)   called every control tick; set the command, return None to continue or an Outcome to end

Every behavior can carry an expectation. When what happens is not what was expected, it ends with a
"surprise" and the kernel holds: the policy decides what to do next, with the facts in front of it.

A spec is plain JSON: {"do": "line", "up": 0.05}. A list is a sequence. `build(spec)` makes the behavior.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from . import motion
from .errors import Refused

if TYPE_CHECKING:
    from .kernel import Kernel

STATUSES = ("done", "refused", "surprise", "stopped", "faulted")


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
    kind = "behavior"
    moves = True                      # counts as motion time (holding, waiting and asking do not)
    senses_contact = False            # True: it expects contact and judges it itself (the kernel's check steps aside)

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
    """Joint-space move. target_deg / delta_deg map joint number (1-based) to degrees."""
    kind = "joints"

    def plan(self, k):
        goal = k.cmd.q.copy()
        for key, v in (self.params.get("target_deg") or {}).items():
            goal[_joint(k, key)] = np.radians(float(v))
        for key, v in (self.params.get("delta_deg") or {}).items():
            goal[_joint(k, key)] += np.radians(float(v))
        if not self.params.get("target_deg") and not self.params.get("delta_deg"):
            raise Refused("joints needs target_deg or delta_deg", "spec")
        path, _ = motion.joint_move(k.cmd.q, goal, _speed_timing(k, self.params.get("speed")), self.params.get("duration"))
        return path


def _joint(k, key) -> int:
    i = int(key) - 1
    if not 0 <= i < k.manifest.n:
        raise Refused(f"joint numbers are 1..{k.manifest.n}, got {key!r}", "spec")
    return i


class Line(PathBehavior):
    """Straight tool-point line, orientation held. forward/left/up in metres along `frame`'s axes."""
    kind = "line"

    def delta(self, k):
        f, l, u = k.world.frame(self.params.get("frame", "work")).axes
        p = self.params
        return p.get("forward", 0.0) * f + p.get("left", 0.0) * l + p.get("up", 0.0) * u

    def plan(self, k):
        d = self.delta(k)
        if np.linalg.norm(d) > k.manifest.max_segment_m:
            raise Refused(f"line is {100 * np.linalg.norm(d):.1f} cm; one segment may be at most "
                          f"{100 * k.manifest.max_segment_m:.0f} cm", "segment_length", "split it into shorter lines")
        path, _, _ = motion.line(k.chain, k.cmd.q, d, _speed_timing(k, self.params.get("speed")),
                                 self.params.get("duration"), *k.envelope.bounds(k.cmd.q), weights=k.ik_weights)
        return path


class Lines(PathBehavior):
    """Several legs [[forward, left, up], ...] as one blended motion: no stop at each corner."""
    kind = "lines"

    def plan(self, k):
        f, l, u = k.world.frame(self.params.get("frame", "work")).axes
        legs = [a * f + b * l + c * u for a, b, c in self.params["legs"]]
        for d in legs:
            if np.linalg.norm(d) > k.manifest.max_segment_m:
                raise Refused("a leg is longer than one segment may be", "segment_length", "split it")
        path, _, _ = motion.polyline(k.chain, k.cmd.q, legs, _speed_timing(k, self.params.get("speed")),
                                     self.params.get("blend", 0.02), self.params.get("duration"),
                                     *k.envelope.bounds(k.cmd.q), weights=k.ik_weights)
        return path


class MoveTo(PathBehavior):
    """Tool point to an absolute position [x, y, z] (metres) in `frame`, in a straight line, orientation held."""
    kind = "move_to"

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

class ContactSense:
    """Torque change that the arm's own weight does not explain.

    The bias (measured minus model) is captured while holding still before the move, so the URDF's mass errors
    and cable loads cancel. A short median filter rejects single-tick spikes. Without an explicit joint list it
    watches the joints with real leverage along the direction of motion: a vertical push barely loads a vertical
    base axis, whose friction would only add noise.
    """

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
        self.bias = np.asarray(st.tau, float) - k.chain.gravity(st.q)
        self.hist = deque(maxlen=window)

    def deviation(self, k: Kernel) -> np.ndarray:
        st = k.state
        dev = np.asarray(st.tau, float) - k.chain.gravity(st.q) - self.bias
        self.hist.append(dev)
        return np.median(np.array(self.hist), axis=0)

    def exceeded(self, k, dtau) -> tuple[bool, np.ndarray]:
        dev = self.deviation(k)
        sel = np.abs(dev[self.joints])
        return bool(len(self.hist) == self.hist.maxlen and sel.max() > dtau), dev


class Guarded(Line):
    """A slow straight line that stops the moment something pushes back (joint torque change > dtau Nm).

    expect_contact=True (the default): reaching the end without contact is a surprise, not success.
    Fragile zones lower dtau automatically.
    """
    kind = "guarded"
    allow_contact = True
    senses_contact = True

    OVERSHOOT = 0.02                  # plan at most this far into a surface the world knows about

    def start(self, k):
        self.d, self.cut = self._clip_to_surfaces(k, self.delta(k))
        self.sense = ContactSense(k, self.params.get("joints"), direction=self.d)
        self.seconds = max(1.0, float(np.linalg.norm(self.d)) / float(self.params.get("speed_mps", 0.02)) / 0.8)
        super().start(k)

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
        hit, dev = self.sense.exceeded(k, self.dtau(k))
        if hit:
            k.hold_here()
            moved = float(np.linalg.norm(k.chain.fk(k.state.q)[:3, 3] - k.chain.fk(self.path[0])[:3, 3]))
            k.touched("contact", f"contact after {100 * moved:.1f} cm")
            return self.done(f"contact after {100 * moved:.1f} cm", moved_m=round(moved, 4),
                             torque_change=np.round(dev, 2).tolist())
        return super().tick(k)

    def arrived(self, k):
        if self.params.get("expect_contact", True):
            where = f", {100 * self.OVERSHOOT:.0f} cm past where {self.cut!r} should be" if self.cut else ""
            return self.surprise(f"reached the end of the guarded move without contact{where}", expected="contact",
                                 observed="no contact",
                                 hint="the world model is off here: look, then correct it" if self.cut else
                                 "the surface is further than planned: look first, then go further")
        return self.done("no contact, as expected")


class Touchdown(Guarded):
    """Guarded move straight down (at most `max` metres) until contact: find a table, put an object down."""
    kind = "touchdown"

    def delta(self, k):
        return np.array([0.0, 0.0, -float(self.params.get("max", 0.06))])


# -- gripper ------------------------------------------------------------------------------------------

class Gripper(Behavior):
    """Move the gripper to `to` (native units) or `aperture_mm`, over `seconds` or at its speed limit."""
    kind = "gripper"
    moves = True

    def start(self, k):
        g = k.manifest.gripper
        if g is None:
            raise Refused("this robot has no gripper", "no_gripper")
        to = g.position(self.params["aperture_mm"] / 1000) if "aperture_mm" in self.params else float(self.params["to"])
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


class Grip(Behavior):
    """Close until the fingers meet something, then squeeze a little and hold.

    Contact = gripper effort over `effort`, or the gripper falling `lag` behind its command. `expect` = [lo, hi]
    (native units): contact outside it, or no contact, is a surprise: the wrong thing, or nothing, is in the hand.
    """
    kind = "grip"

    def start(self, k):
        g = k.manifest.gripper
        if g is None:
            raise Refused("this robot has no gripper", "no_gripper")
        p = self.params
        here = k.cmd.gripper if k.cmd.gripper is not None else k.state.gripper
        self.pre = None
        if "start" in p and abs(float(p["start"]) - here) > 0.02:     # open to the start width first, smoothly
            self.pre = Gripper(to=float(p["start"]))
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
        k.touched("grip", f"grip contact at {self.contact:.2f} {g.unit}")
        data = dict(contact_at=round(self.contact, 3), holding_effort=None if st.gripper_tau is None else round(st.gripper_tau, 2))
        a = g.aperture(self.contact)
        if a is not None:
            data["aperture_mm"] = round(1000 * a, 1)
        lo, hi = p.get("expect", (None, None))
        if lo is not None and not lo <= self.contact <= hi:
            return self.surprise(f"fingers met something at {self.contact:.2f} {g.unit}, outside the expected {lo}..{hi}",
                                 expected=[lo, hi], observed=self.contact,
                                 hint="the object is not where, or not the size, planned: open and look", **data)
        return self.done(f"holding at {self.contact:.2f} {g.unit}" + _mm(g, self.contact), **data)

    def _no_contact(self, k):
        return self.surprise("the gripper closed on nothing", expected="contact", observed="no contact",
                             hint="open, check the object's position in a camera, adjust and retry")


# -- waiting, asking, composing -----------------------------------------------------------------------

class Hold(Behavior):
    """Hold the current command for `seconds` (forever if omitted, until stopped or replaced)."""
    kind = "hold"
    moves = False

    def start(self, k):
        self.left = None if self.params.get("seconds") is None else int(float(self.params["seconds"]) * k.manifest.rate_hz)

    def tick(self, k):
        k.set(k.cmd.q, np.zeros(k.manifest.n))
        if self.left is None:
            return None
        self.left -= 1
        return self.done(f"held {self.params['seconds']} s") if self.left <= 0 else None


class Checkpoint(Behavior):
    """Stop and ask. The arm holds while the question waits for an answer.

    ask     the question, e.g. "is the black loop between the jaws?"
    view    which camera answers it best; roi = region of interest [x0, y0, x1, y1] in that image
    expect  the answer that means "carry on" (default "yes"); any other answer ends the plan with a surprise
    In a twin check the expected answer is assumed and flagged as an assumption.
    """
    kind = "checkpoint"
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
    """Run steps in order. The first step that does not end "done" ends the sequence with its outcome."""
    kind = "seq"

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
                try:
                    self.current.start(k)
                except Refused as e:
                    return Outcome("refused", self.current.kind, f"step {self.i + 1}: {e}", dict(step=self.i + 1),
                                   hint=e.hint)
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
        return Outcome("stopped", self.kind, f"stopped at step {self.i + 1}/{len(self.steps)}: {reason}", dict(step=self.i + 1))

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
