"""Views: the kernel's state rendered for a policy. Short by default; details on request.

The state line is what a policy reads after every step, so every character in it has to earn its place.
"""
from __future__ import annotations

import numpy as np

from .errors import Refused


def _xyz(k, p, frame="work") -> str:
    f = k.world.from_base(frame, p)
    return f"F{f[0]:+.3f} L{f[1]:+.3f} U{f[2]:+.3f}"


def _gripper(k, pos, effort=None) -> str:
    g = k.manifest.gripper
    if g is None or pos is None:
        return ""
    a = g.aperture(pos)
    s = f"grip {pos:.2f}{g.unit}" + ("" if a is None else f" ({1000 * a:.0f}mm)")
    return s + ("" if effort is None else f" {effort:+.1f}")


def state_line(k) -> str:
    st = k.state
    parts = [f"t+{k.clock.now() - k.t0:.0f}s"]
    job = k.active
    if k.faulted:
        parts.append("FAULTED, holding")
    elif job is not None:
        parts.append(f"job {job.id} {job.status}: {job.behavior.describe()}"[:80])
    else:
        parts.append("idle, holding" if k.enabled else "torque off")
    parts.append("tool " + _xyz(k, k.chain.fk(st.q)[:3, 3]))
    g = _gripper(k, st.gripper, st.gripper_tau)
    if g:
        parts.append(g)
    if st.tau is not None:
        parts.append("tau " + " ".join(f"{t:+.1f}" for t in np.asarray(st.tau)))
    if st.temp is not None:
        temp = np.nan_to_num(np.asarray(st.temp, float))
        i = int(temp.argmax())
        left = k.heat.minutes_left(k.manifest.temp_limit_c)
        s = f"hottest j{i + 1} {temp[i]:.0f}C"
        if left is not None and left[2] < 30:
            s += f" ({left[2]:.1f} min to {k.manifest.temp_limit_c:.0f}C, j{left[0] + 1})"
        parts.append(s)
    return " | ".join(parts)


def status(k) -> dict:
    st = k.state
    tool = k.chain.fk(st.q)
    d = dict(body=k.manifest.name, enabled=k.enabled, faulted=k.faulted, line=state_line(k),
             joints_deg=np.round(np.degrees(st.q), 2).tolist(),
             tool=dict(base=np.round(tool[:3, 3], 4).tolist(),
                       **{name: np.round(k.world.from_base(name, tool[:3, 3]), 4).tolist()
                          for name in k.world.frames if name != "base"}))
    if st.tau is not None:
        d["torque_nm"] = np.round(np.asarray(st.tau), 2).tolist()
    if st.temp is not None:
        d["temp_c"] = np.round(np.asarray(st.temp, float), 0).tolist()
        left = k.heat.minutes_left(k.manifest.temp_limit_c)
        if left:
            d["heat"] = dict(joint=left[0] + 1, temp_c=round(left[1], 1), minutes_to_limit=round(left[2], 1))
    if k.manifest.gripper is not None and st.gripper is not None:
        g = k.manifest.gripper
        d["gripper"] = dict(position=round(st.gripper, 3), unit=g.unit, commanded=None if k.cmd.gripper is None else round(k.cmd.gripper, 3),
                            aperture_mm=None if g.aperture(st.gripper) is None else round(1000 * g.aperture(st.gripper), 1),
                            effort=None if st.gripper_tau is None else round(st.gripper_tau, 2))
    if k.active is not None:
        d["job"] = k.active.to_dict()
    if k.queue:
        d["queued"] = [j.id for j in k.queue]
    try:
        d["home"] = f"ready: {len(k.home_plan())} moves"
    except Refused as e:
        d["home"] = f"not available: {e}"
    if k.envelope.overrides:
        d["overrides"] = k.envelope.overrides
    if k.world.facts:
        d["facts"] = {key: (f.value if f.stale is None else f"{f.value} (STALE: {f.stale})") for key, f in k.world.facts.items()}
    d["events"] = k.events.seq
    return d


def incident(k, job) -> str:
    """What went differently from the plan, in one block a policy can act on."""
    out = job.outcome
    lines = [f"job {job.id} {out.status}: {out.message}"]
    if out.expected is not None or out.observed is not None:
        lines.append(f"expected: {out.expected}   observed: {out.observed}")
    if out.hint:
        lines.append(f"hint: {out.hint}")
    lines.append(state_line(k))
    recent = [e for e in k.events.since(max(0, k.events.seq - 8)) if e["level"] != "info" or e["kind"] in ("contact", "grip")]
    for e in recent[-4:]:
        lines.append(f"  [{e['seq']}] {e['kind']}: {e['message']}")
    return "\n".join(lines)


def card(k) -> str:
    """The embodiment card: what this robot is and what it can do, for the top of a policy's context."""
    m, c = k.manifest, k.chain
    lines = [f"# {m.name}", f"{m.n} joints, control at {m.rate_hz:.0f} Hz; senses: {', '.join(sorted(m.sensing))}."]
    lines.append("joints (deg): " + "; ".join(f"j{i + 1} {j.name} {np.degrees(j.lower):.0f}..{np.degrees(j.upper):.0f}"
                                          for i, j in enumerate(m.joints)))
    if m.gripper:
        g = m.gripper
        span = "" if g.m_per_unit is None else f" = 0..{1000 * abs(g.aperture(g.open) or 0):.0f} mm opening"
        lines.append(f"gripper: {g.closed}..{g.open} {g.unit} (closed..open){span}.")
    reach = np.linalg.norm(c.fk(np.zeros(c.n))[:3, 3] - c.points(np.zeros(c.n))[1])
    lines.append(f"reach about {reach:.2f} m from the shoulder; one Cartesian move at most {100 * m.max_segment_m:.0f} cm.")
    lines.append(f"default peak joint speed {m.speed} rad/s; motors warn at {m.temp_warn_c:.0f} C, stop at {m.temp_limit_c:.0f} C.")
    if m.rest:
        lines.append("torque can only be released at the rest pose (the arm has no brakes).")
    if m.max_excursion is not None or k.envelope.max_excursion is not None:
        lines.append(f"each joint may travel at most {np.degrees(k.envelope.max_excursion):.0f} deg from the session start.")
    frames = [f for f in k.world.frames if f != "base"]
    lines.append("frames: " + ", ".join(frames) + ". Moves take forward/left/up along a frame's x/y/z (default: work).")
    for b in k.world.boxes.values():
        extra = ", ".join(f"{a}={v}" for a, v in b.params.items())
        lines.append(f"{b.kind} {b.name!r}: centre {np.round(b.pose[:3, 3], 3).tolist()} size {np.round(b.size, 3).tolist()}"
                     + (f" ({extra})" if extra else ""))
    for n in m.notes:
        lines.append(f"note: {n}")
    return "\n".join(lines)
