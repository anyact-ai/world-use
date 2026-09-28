"""Views: the kernel's state rendered for a policy. Short by default; details on request.

The state line is what a policy reads after every step, so every character in it has to earn its place.
"""
import time

import numpy as np

from .errors import Refused
from .plan import reach


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
        if k.enabled and left is not None and left[2] < 30:
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
        a = g.aperture(st.gripper)
        d["gripper"] = dict(position=round(st.gripper, 3), unit=g.unit,
                            commanded=None if k.cmd.gripper is None else round(k.cmd.gripper, 3),
                            aperture_mm=None if a is None else round(1000 * a, 1),
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
    if k.world.held is not None:
        d["holding"] = k.world.held[0]
    elif k.held_at is not None:
        d["holding"] = "something the world has no box for"
    if k.world.facts:
        d["facts"] = {key: (f.value if f.stale is None else f"{f.value} (STALE: {f.stale})")
                      for key, f in k.world.facts.items()}
    d["events"] = k.events.seq
    return d


def incident(k, job, reach=None) -> str:
    """What went differently from the plan, in one block a policy can act on."""
    out = job.outcome
    lines = [f"job {job.id} {out.status}: {out.message}"]
    if out.expected is not None or out.observed is not None:
        lines.append(f"expected: {out.expected}   observed: {out.observed}")
    if out.hint:
        lines.append(f"hint: {out.hint}")
    lines.append(state_line(k))
    if out.status == "refused" and k.enabled and k.active is None:
        lines.append((reach or reach_line)(k))
    recent = [e for e in k.events.since(max(0, k.events.seq - 8))
              if e["level"] != "info" or e["kind"] in ("contact", "grip")]
    for e in recent[-4:]:
        lines.append(f"  [{e['seq']}] {e['kind']}: {e['message']}")
    return "\n".join(lines)


def reach_line(k, step: float = 0.03) -> str:
    """Which short lines the kernel would take from here, and why not the others: saves a refusal per direction."""
    ok, bad = [], {}
    for direction, refusal in reach(k, step).items():
        if refusal is None:
            ok.append(direction)
        else:
            bad.setdefault(_why(k, refusal), []).append(direction)
    text = f"from here a {100 * step:.0f} cm line can go {', '.join(ok) if ok else 'nowhere'}"
    if bad:
        text += "; not " + "; ".join(f"{' or '.join(ds)} ({why})" for why, ds in bad.items())
    return text + "."


def _why(k, refusal: Refused) -> str:
    reasons = []
    for p in refusal.problems:
        if p.rule == "reach":
            r = "out of reach with the gripper at this angle"
        elif p.rule == "joint_limit":
            r = f"{k.manifest.joints[p.data['joint'] - 1].name} at its limit"
        elif p.rule == "turn_clearance":
            r = f"turning needs the tool at U{p.data['need_up']:+.3f}"
        elif p.rule in ("surface", "keep_out"):
            r = f"{p.rule.replace('_', '-')} {p.data.get('box')!r}"
        else:
            r = p.rule or str(p)
        reasons.append(r)
    return " and ".join(dict.fromkeys(reasons))


def _heading(v) -> str:
    """A direction in the work frame, in words: 'forward, level', 'straight down', 'left, tilted 30 deg down'."""
    v = np.asarray(v, float) / np.linalg.norm(v)
    elev = float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))
    if abs(elev) > 80:
        return "straight up" if elev > 0 else "straight down"
    names = ("forward", "forward-left", "left", "back-left", "back", "back-right", "right", "forward-right")
    name = names[int(np.round(np.degrees(np.arctan2(v[1], v[0])) / 45.0)) % 8]
    return f"{name}, " + ("level" if abs(elev) < 5 else f"tilted {abs(elev):.0f} deg {'up' if elev > 0 else 'down'}")


def _axis(v) -> str:
    v = np.abs(np.asarray(v, float)) / np.linalg.norm(v)
    i = int(v.argmax())
    name = ("forward and back", "left and right", "up and down")[i]
    return name if v[i] > 0.94 else f"roughly {name}"


def tool_line(k) -> str | None:
    """Which way the gripper points and opens, in the work frame: what a policy needs to plan an approach."""
    g = k.manifest.gripper
    if g is None:
        return None
    R = k.world.frame("work").T[:3, :3].T @ k.chain.fk(k.cmd.q)[:3, :3]
    return (f"tool: the gripper points {_heading(R @ np.asarray(g.approach))}; its jaws open "
            f"{_axis(R @ np.asarray(g.opens_along))}; the tool point (the position the state line reports) is "
            f"{g.tool_point}. line, lines and move_to keep this angle; only joints moves change it.")


def box_line(k, b, frame: str = "work") -> str:
    f = k.world.in_frame(b, frame)
    c, s = np.where(np.abs(f["centre"]) < 5e-4, 0.0, f["centre"]), f["size"]      # no "-0.000"
    parts = [f"{b.kind} {b.name!r}: centre F{c[0]:+.3f} L{c[1]:+.3f} U{c[2]:+.3f}, "
             f"size {s[0]:.3f} x {s[1]:.3f} x {s[2]:.3f}"]
    if abs(f["yaw_deg"]) > 0.5:
        parts.append(f"turned {f['yaw_deg']:.0f} deg")
    if b.kind == "surface":
        parts.append(f"top at U{f['top']:+.3f}")
    if b.kind == "object":
        parts.append(f"from U{2 * c[2] - f['top']:+.3f} to U{f['top']:+.3f}")
        parts.append(f"{1000 * b.grip_width:.0f} mm across the jaws")
    parts += [f"{key}={v}" for key, v in b.params.items() if key != "grip_width"]
    if b.source not in ("workcell", "config"):
        parts.append(f"from {b.source}")
    if k.world.held is not None and k.world.held[0] == b.name:
        parts.append("in the gripper")
    return ", ".join(parts)


def world_text(k) -> str:
    """The world model as a policy reads it: frames, boxes (work frame), facts with their sources and age."""
    w = k.world
    lines = ["frames: " + ", ".join(w.frames) + ". Boxes are in the work frame, metres; sizes are forward x left x up."]
    lines += [box_line(k, b) for b in w.boxes.values()] or ["no boxes: nothing is known about the scene yet"]
    now = time.time()
    for key, f in w.facts.items():
        stale = f" STALE ({f.stale})" if f.stale else ""
        lines.append(f"fact {key} = {f.value} (from {f.source}, {now - f.t:.0f} s ago){stale}")
    return "\n".join(lines)


def card(k, reach=None) -> str:
    """The embodiment card: what this robot is and what it can do, for the top of a policy's context."""
    m, c = k.manifest, k.chain
    sim = bool(getattr(k.body, "simulated", False))
    lines = [f"# {m.name}" + (" (simulated)" if sim else ""),
             f"{m.n} joints, control at {m.rate_hz:.0f} Hz; senses: {', '.join(sorted(m.sensing))}."]
    if sim:
        lines.append("This is a simulation: the joints follow commands exactly, with no sag or noise. Lines marked "
                     "'hardware:' describe the real robot.")
    lines.append("joints (deg): " + "; ".join(f"j{i + 1} {j.name} {np.degrees(j.lower):.0f}..{np.degrees(j.upper):.0f}"
                                          for i, j in enumerate(m.joints)))
    if m.gripper:
        g = m.gripper
        span = "" if g.m_per_unit is None else f" = 0..{1000 * abs(g.aperture(g.open) or 0):.0f} mm opening"
        mm = " gripper and grip also take millimetres (aperture_mm, start_mm, expect_mm)." if g.m_per_unit else ""
        lines.append(f"gripper: {g.closed}..{g.open} {g.unit} (closed..open){span}.{mm} grip squeezes "
                     f"{g.squeeze} {g.unit} past contact.")
        lines.append(tool_line(k) or "")
    reach_m = np.linalg.norm(c.fk(np.zeros(c.n))[:3, 3] - c.points(np.zeros(c.n))[1])
    shoulder = c.points(k.cmd.q)[1]
    s = k.world.from_base("work", shoulder) + 0.0
    now = np.linalg.norm(c.fk(k.cmd.q)[:3, 3] - shoulder)
    lines.append(f"reach about {reach_m:.2f} m from the shoulder at F{s[0]:+.3f} L{s[1]:+.3f} U{s[2]:+.3f} "
                 f"(the tool is {now:.2f} m from it now); one Cartesian move at most {100 * m.max_segment_m:.0f} cm.")
    lines.append(f"default peak joint speed {m.speed} rad/s; motors warn at {m.temp_warn_c:.0f} C, "
                 f"stop at {m.temp_limit_c:.0f} C.")
    if m.rest:
        lines.append("torque can only be released at the rest pose (the arm has no brakes).")
    if m.max_excursion is not None or k.envelope.max_excursion is not None:
        lines.append(f"each joint may travel at most {np.degrees(k.envelope.max_excursion):.0f} deg "
                     "from the session start.")
    need = k.envelope.turn_height()
    if need is not None:
        joints, above = m.turn_clearance
        names = ", ".join(f"j{j + 1}" for j in joints)
        sideways = "; every left or right move turns j1, so lift first, then move sideways" if 0 in joints else ""
        lines.append(f"turning {names} needs the tool at U{need:+.3f} or higher ({100 * above:.0f} cm above where it "
                     f"started){sideways}.")
    others = [f for f in k.world.frames if f not in ("base", "work")]
    lines.append("frames: work = forward, left, up from the base, pointing where the arm pointed at the session start"
                 + (f"; also {', '.join(others)}" if others else "") + ". Positions here are work-frame metres; "
                 "moves take forward/left/up along a frame's axes (default: work).")
    lines += [box_line(k, b) for b in k.world.boxes.values()]
    cams = k.cameras
    if cams:
        lines.append(f"cameras: {', '.join(cams)}. `wu look NAME` saves an image and prints its path.")
    if k.enabled and k.active is None:
        lines.append((reach or reach_line)(k))
    lines += [f"note: {n}" for n in m.notes]
    lines += [f"hardware: {n}" for n in m.hardware_notes]
    return "\n".join(lines)
