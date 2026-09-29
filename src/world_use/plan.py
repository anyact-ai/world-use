"""Plans: code builds data, data gets checked, the kernel runs it.

    p = Plan("put it down")
    p.line(up=0.05)
    p.touchdown(max=0.07)
    p.gripper(aperture_mm=60)
    report = check(p, kernel)        # the same kernel code, run on a twin from the robot's measured state
    print(report)                    # durations, contacts, refusals, heat - before anything real moves

Any registered behavior is a method (p.grip(...), p.checkpoint(...)); plugins' behaviors appear automatically.
"""
from dataclasses import dataclass, field

import numpy as np

from . import motion
from .behaviors import REGISTRY, Outcome
from .body import Manifest
from .errors import Refused
from .geometry import rotation_log
from .kernel import Kernel, VirtualClock
from .world import DIRECTIONS, World, along, heading


class Plan:
    def __init__(self, label: str | None = None, steps=None):
        self.label, self.steps = label, list(steps or [])

    def add(self, spec) -> Plan:
        self.steps.append(spec)
        return self

    def __getattr__(self, kind):
        if kind.startswith("_") or kind not in REGISTRY:
            raise AttributeError(kind)

        def step(label=None, **params):
            return self.add({"do": kind, **params, **({"label": label} if label else {})})
        return step

    def spec(self) -> dict:
        return {"do": "seq", "steps": list(self.steps), **({"label": self.label} if self.label else {})}

    def __len__(self):
        return len(self.steps)


@dataclass
class Report:
    outcome: Outcome                  # how the rehearsal ended
    seconds: float                    # simulated time from start to finish
    moving_s: float
    steps: list[str] = field(default_factory=list)
    contacts: list[str] = field(default_factory=list)
    assumed: list[str] = field(default_factory=list)
    tool_end: list | None = None
    temp_rise: dict | None = None
    problems: list[dict] = field(default_factory=list)   # every limit the plan would break: step, message, hint, rule
    tool_path: list | None = None     # tool positions along the rehearsal (base frame), for drawing on images
    gripper_end: dict | None = None   # where the gripper ends pointing, when the plan turned it more than 2 deg

    @property
    def ok(self) -> bool:
        return self.outcome.ok and not self.problems

    @property
    def refused(self) -> bool:
        """The kernel would refuse this plan (as opposed to it ending differently than planned)."""
        return bool(self.problems) or self.outcome.status == "refused"

    def to_dict(self) -> dict:
        return dict(ok=self.ok, refused=self.refused, outcome=self.outcome.to_dict(), problems=self.problems,
                    seconds=self.seconds, moving_s=self.moving_s, steps=self.steps, contacts=self.contacts,
                    assumed=self.assumed, tool_end=self.tool_end, temp_rise=self.temp_rise,
                    gripper_end=self.gripper_end)

    def __str__(self) -> str:
        n = len(self.problems)
        if n:
            head = (f"check FAILED: {n} limit{'s' if n > 1 else ''} would be broken, so the kernel would refuse this "
                    f"plan and nothing would move")
        else:
            head = "check passed" if self.ok else f"check FAILED ({self.outcome.status})"
        lines = [head]
        for q in self.problems:
            lines.append(f"  {q['step']}: {q['message']}" + (f" (hint: {q['hint']})" if q.get("hint") else ""))
        verb = "rehearsed past them" if n else "rehearsed"
        lines.append(f"  {verb}: {self.seconds:.1f} s simulated, {self.moving_s:.1f} s of it moving")
        if not n:
            lines += [f"  {s}" for s in self.steps]
        if not self.outcome.ok:
            hint = f" (hint: {self.outcome.hint})" if self.outcome.hint else ""
            lines.append(f"  -> {self.outcome.message}{hint}")
        lines += [f"  contact: {c}" for c in self.contacts]
        lines += [f"  ASSUMED: {a}" for a in self.assumed]
        if self.tool_end:
            f, left, u = self.tool_end
            lines.append(f"  ends with the tool at F{f:+.3f} L{left:+.3f} U{u:+.3f} (work)")
        if self.gripper_end:
            g = self.gripper_end
            lines.append(f"  the gripper ends pointing {g['points']}, jaws open {g['jaws']} "
                         f"(turned {g['turned_deg']} deg)")
        if self.temp_rise:
            r = self.temp_rise
            lines.append(f"  heat: j{r['joint']} +{r['rise_c']:.1f} C, to about {r['end_c']:.0f} C")
        return "\n".join(lines)


@dataclass(frozen=True)
class Snapshot:
    """A robot's kernel as plain data: all a twin needs, so a rehearsal can run in another process."""
    body: str | None                  # the manifest's name in bodies.manifests(); None if it is not registered
    world: dict
    q: list[float]                    # measured: the twin starts where the robot really is
    gripper: float | None
    temp: list[float] | None
    q_cmd: list[float]                # commanded: where the kernel plans from (reach probes start there)
    grip_cmd: float | None
    q_start: list[float]
    max_excursion: float | None
    overrides: dict
    home_route: list | None           # the steps, while the robot's route is still valid
    held_at: float | None
    grip_start: float | None
    ik_weights: list[float] | None
    fit: dict | None = None           # the fitted robot model the kernel judges torque by (fit.py), if any


def snapshot(k: Kernel) -> Snapshot:
    """k's state as plain data, taken under its lock so it is consistent."""
    from . import bodies
    with k.lock:
        st, env, route = k.state, k.envelope, k.home_route
        return Snapshot(
            body=next((name for name, m in bodies.manifests().items() if m is k.manifest), None),
            world=k.world.to_dict(), q=np.asarray(st.q, float).tolist(), gripper=st.gripper,
            temp=None if st.temp is None else np.asarray(st.temp, float).tolist(),
            q_cmd=np.asarray(k.cmd.q, float).tolist(), grip_cmd=k.cmd.gripper,
            q_start=np.asarray(k.q_start, float).tolist(), max_excursion=env.max_excursion,
            overrides=dict(env.overrides),
            home_route=list(route[0]) if route is not None and k.last_touch < route[1] else None,
            held_at=k.held_at, grip_start=k.grip_start,
            ik_weights=None if k.ik_weights is None else list(k.ik_weights),
            fit=None if k.fit is None else k.fit.to_dict())


def twin_from(s: Snapshot, manifest: Manifest | None = None) -> Kernel:
    """A simulated kernel at the snapshot's state: same manifest, world, limits, session start and home route."""
    from . import bodies
    from .bodies.sim import SimBody
    if manifest is None:
        if s.body is None:
            raise ValueError("this snapshot's body is not registered, so its manifest must be given")
        manifest = bodies.manifests()[s.body]
    world = World.from_dict(s.world)
    body = SimBody(manifest, world, q=s.q, gripper=s.gripper, temp_c=s.temp)
    t = Kernel(body, world, VirtualClock(manifest.rate_hz), ik_weights=s.ik_weights, auto_answer=True)
    t.connect()                       # the world already holds the session's frames, so they are kept
    if s.fit is not None:             # the twin weighs its links, and feels friction, as the robot's own fit says
        from .fit import Model
        t.use_fit(Model.from_dict(s.fit))
    t.q_start = np.asarray(s.q_start, float)
    t.envelope.q_start = t.q_start
    t.envelope.max_excursion, t.envelope.overrides = s.max_excursion, dict(s.overrides)
    if s.home_route is not None:      # valid from here on, until something in the twin is touched
        t.home_route = (list(s.home_route), t.events.seq + 1)
    t.held_at, t.grip_start = s.held_at, s.grip_start
    t.residuals.need = 1              # noise-free, and the robot's own baseline is warm by the time a plan runs
    t.enable()
    if s.grip_cmd is not None:
        t.cmd.gripper = s.grip_cmd
    return t


def twin(k: Kernel) -> Kernel:
    """A simulated copy of a kernel at the robot's measured state: same manifest, world, limits and session start."""
    return twin_from(snapshot(k), k.manifest)


def check(spec, k: Kernel, timeout_s: float = 900.0) -> Report:
    """Rehearse spec on a twin of k. Checkpoints assume their expected answer, and the report says so.

    Limits a step would break are recorded and the rehearsal carries on past them, so one check lists every
    problem in the plan. It stops early only where a step cannot be planned at all (out of reach, a bad spec).
    """
    return rehearse(spec, twin(k), timeout_s)


def rehearse(spec, t: Kernel, timeout_s: float = 900.0) -> Report:
    """Run spec on the twin t (see check) and report what happened."""
    if isinstance(spec, Plan):
        spec = spec.spec()
    t.envelope.rehearsal = []
    R0 = t.chain.fk(t.cmd.q)[:3, :3]
    temp0 = None if t.state.temp is None else np.asarray(t.state.temp, float).copy()
    seq0 = t.events.seq
    out = t.run(spec, timeout_s)
    events = t.events.since(seq0)
    problems = [dict(step=where, message=str(p), hint=p.hint, rule=p.rule, **p.data)
                for where, found in t.envelope.rehearsal for p in found]
    steps = [e["message"] for e in events if e["kind"] == "step_done"]
    if not steps and out.ok:
        steps = [out.message]
    summary = t.tape.summary(t.manifest.rate_hz)
    tool = t.world.from_base("work", t.chain.fk(t.state.q)[:3, 3])
    turned = None
    g = t.manifest.gripper
    R1 = t.chain.fk(t.state.q)[:3, :3]
    deg = float(np.degrees(np.linalg.norm(rotation_log(R1 @ R0.T))))
    if g is not None and deg > 2.0:
        W = t.world.frame("work").T[:3, :3]
        turned = dict(points=heading(W.T @ R1 @ np.asarray(g.approach, float)),
                      jaws=along(W.T @ R1 @ np.asarray(g.opens_along, float)), turned_deg=round(deg))
    rise = None
    if temp0 is not None and t.state.temp is not None:
        d = np.asarray(t.state.temp, float) - temp0
        i = int(d.argmax())
        rise = dict(joint=i + 1, rise_c=round(float(d[i]), 1), end_c=round(float(t.state.temp[i]), 1))
    tape = t.tape.arrays()
    path = None
    if tape:
        q = tape["q"][:: max(1, len(tape["q"]) // 200)]
        path = np.round([t.chain.fk(row)[:3, 3] for row in q], 4).tolist()
    return Report(out, round(float(t.clock.now() - t.t0), 2), summary.get("moving_s", 0.0), steps,
                  [e["message"] for e in events if e["kind"] in ("contact", "grip")],
                  [e["message"] for e in events if e["kind"] == "assumed"],
                  np.round(tool, 3).tolist(), rise, problems, path, turned)


def reach(k: Kernel, step: float = 0.03) -> dict[str, Refused | None]:
    """Which short straight lines (work frame) the kernel would accept from where the arm is: None, or the refusal.
    A quick plan at coarse resolution; nothing moves and nothing is recorded."""
    out: dict[str, Refused | None] = {}
    fwd, left, up = k.world.frame("work").axes
    q = k.cmd.q.copy()
    for name, (f, s, u) in DIRECTIONS.items():
        try:
            path, _, _ = motion.line(k.chain, q, step * (f * fwd + s * left + u * up), k.timing, None,
                                     *k.envelope.bounds(q), weights=k.ik_weights, knots=6)
            k.envelope.check_path(path, q)
            out[name] = None
        except Refused as e:
            out[name] = e
    return out
