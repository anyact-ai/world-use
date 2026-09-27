"""Plans: code builds data, data gets checked, the kernel runs it.

    p = Plan("put it down")
    p.line(up=0.05)
    p.touchdown(max=0.07)
    p.gripper(to=3.0)
    report = check(p, kernel)        # the same kernel code, run on a twin from the robot's measured state
    print(report)                    # durations, contacts, refusals, heat - before anything real moves

Any registered behavior is a method (p.grip(...), p.checkpoint(...)); plugins' behaviors appear automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .behaviors import REGISTRY, Outcome
from .kernel import Kernel, VirtualClock
from .world import World


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
    outcome: Outcome
    seconds: float                    # simulated time from start to finish
    moving_s: float
    steps: list[str] = field(default_factory=list)
    contacts: list[str] = field(default_factory=list)
    assumed: list[str] = field(default_factory=list)
    tool_end: list | None = None
    temp_rise: dict | None = None

    @property
    def ok(self) -> bool:
        return self.outcome.ok

    def to_dict(self) -> dict:
        return dict(ok=self.ok, outcome=self.outcome.to_dict(), seconds=self.seconds, moving_s=self.moving_s,
                    steps=self.steps, contacts=self.contacts, assumed=self.assumed, tool_end=self.tool_end,
                    temp_rise=self.temp_rise)

    def __str__(self) -> str:
        head = "check passed" if self.ok else f"check FAILED ({self.outcome.status})"
        lines = [f"{head}: {self.seconds:.1f} s simulated, {self.moving_s:.1f} s of it moving"]
        lines += [f"  {s}" for s in self.steps]
        if not self.ok:
            lines.append(f"  -> {self.outcome.message}" + (f" (hint: {self.outcome.hint})" if self.outcome.hint else ""))
        lines += [f"  contact: {c}" for c in self.contacts]
        lines += [f"  ASSUMED: {a}" for a in self.assumed]
        if self.tool_end:
            lines.append(f"  ends with the tool at F{self.tool_end[0]:+.3f} L{self.tool_end[1]:+.3f} U{self.tool_end[2]:+.3f} (work)")
        if self.temp_rise:
            lines.append(f"  heat: j{self.temp_rise['joint']} +{self.temp_rise['rise_c']:.1f} C, to about {self.temp_rise['end_c']:.0f} C")
        return "\n".join(lines)


def twin(k: Kernel) -> Kernel:
    """A simulated copy of a kernel at the robot's measured state: same manifest, world, limits and session start."""
    from .bodies.sim import SimBody
    with k.lock:                              # a consistent snapshot; the twin itself runs without the lock
        world = World.from_dict(k.world.to_dict())
        st, q_start, env = k.state, k.q_start.copy(), k.envelope
        grip_cmd = None if k.cmd is None else k.cmd.gripper
        route = k.home_route
    body = SimBody(k.manifest, world, q=st.q, gripper=st.gripper, temp_c=st.temp)
    t = Kernel(body, world, VirtualClock(k.manifest.rate_hz), ik_weights=k.ik_weights, auto_answer=True)
    t.connect()
    t.q_start = q_start
    t.envelope.q_start, t.envelope.z_start = env.q_start, env.z_start
    t.envelope.max_excursion, t.envelope.overrides = env.max_excursion, dict(env.overrides)
    t.home_route, t.last_touch = route, 0
    t.enable()
    if grip_cmd is not None:
        t.cmd.gripper = grip_cmd
    return t


def check(spec, k: Kernel, timeout_s: float = 900.0) -> Report:
    """Rehearse spec on a twin of k. Checkpoints assume their expected answer, and the report says so."""
    if isinstance(spec, Plan):
        spec = spec.spec()
    t = twin(k)
    temp0 = None if t.state.temp is None else np.asarray(t.state.temp, float).copy()
    seq0 = t.events.seq
    out = t.run(spec, timeout_s)
    events = t.events.since(seq0)
    steps = [e["message"] for e in events if e["kind"] in ("step_done",)]
    if not steps and out.ok:
        steps = [out.message]
    summary = t.tape.summary(t.manifest.rate_hz)
    tool = t.world.from_base("work", t.chain.fk(t.state.q)[:3, 3])
    rise = None
    if temp0 is not None and t.state.temp is not None:
        d = np.asarray(t.state.temp, float) - temp0
        i = int(d.argmax())
        rise = dict(joint=i + 1, rise_c=round(float(d[i]), 1), end_c=round(float(t.state.temp[i]), 1))
    return Report(out, round(float(t.clock.now() - t.t0), 2), summary.get("moving_s", 0.0), steps,
                  [e["message"] for e in events if e["kind"] in ("contact", "grip")],
                  [e["message"] for e in events if e["kind"] == "assumed"],
                  np.round(tool, 3).tolist(), rise)
