"""Views: the kernel's state rendered for a policy. Short by default; details on request.

The state line is what a policy reads after every step, so every character in it has to earn its place.
"""
from __future__ import annotations

import time

import numpy as np

from .errors import Refused
from .plan import reach
from .world import along, heading


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
    if k.power_uncertain:
        parts.append("FAULTED, motor power unconfirmed")
    elif k.faulted:
        parts.append("FAULTED, holding" if k.enabled else "FAULTED, torque off")
    elif job is not None:
        parts.append(f"job {job.id} {job.status}: {job.behavior.describe()}"[:80])
    else:
        parts.append("idle, holding" if k.enabled else "torque off")
    feedback = k.feedback_status()
    if feedback["stale"]:
        age = "unknown age" if feedback["age_s"] is None else f"{feedback['age_s']:.1f}s old"
        parts.append(f"STALE feedback ({age}); values below are last known")
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
        if k.enabled and not k.power_uncertain and not feedback["stale"] and left is not None and left[2] < 30:
            s += f" ({left[2]:.1f} min to {k.manifest.temp_limit_c:.0f}C, j{left[0] + 1})"
        parts.append(s)
    return " | ".join(parts)


def status(k) -> dict:
    st = k.state
    tool = k.chain.fk(st.q)
    feedback = k.feedback_status()
    d = dict(body=k.manifest.name, enabled=k.enabled, power_uncertain=k.power_uncertain,
             feedback=feedback,
             recording=dict(path=str(k.run_dir) if k.run_dir else None,
                            error=k.journal.error if k.journal else None),
             faulted=k.faulted, line=state_line(k),
             joints_deg=np.round(np.degrees(st.q), 2).tolist(),
             tool=dict(base=np.round(tool[:3, 3], 4).tolist(),
                       **{name: np.round(k.world.from_base(name, tool[:3, 3]), 4).tolist()
                          for name in k.world.frames if name != "base"}))
    if st.tau is not None:
        d["torque_nm"] = np.round(np.asarray(st.tau), 2).tolist()
    if st.temp is not None:
        d["temp_c"] = np.round(np.asarray(st.temp, float), 0).tolist()
        left = k.heat.minutes_left(k.manifest.temp_limit_c)
        if left and k.enabled and not k.power_uncertain and not feedback["stale"]:
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
        d["home"] = f"set: {len(k.home_plan())} moves"
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
              if (e["level"] != "info" or e["kind"] in ("contact", "grip"))
              and not (e["kind"] == "finished" and e.get("data", {}).get("job") == job.id)]     # the headline
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


def tool_line(k) -> str | None:
    """Which way the gripper points and opens, in the work frame: what a policy needs to plan an approach. With
    torque off it describes the measured pose (the command is then where the torque last went off)."""
    g = k.manifest.gripper
    if g is None:
        return None
    R = k.world.frame("work").T[:3, :3].T @ k.chain.fk(k.cmd.q if k.enabled else k.state.q)[:3, :3]
    keep = ("line and lines keep its tilt, while its heading turns as the arm moves sideways"
            if k.ik_weights is not None and k.ik_weights[5] == 0 else "line and lines keep this angle")
    return (f"tool: the gripper points {heading(R @ np.asarray(g.approach))}; its jaws open "
            f"{along(R @ np.asarray(g.opens_along))}; the tool point (the position the state line reports) is "
            f"{g.tool_point}. {keep}; move_to with \"point\" turns it, and so do joints moves.")


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


def frame_line(frame) -> str:
    """Describe a resolved frame compactly, using its actual transform and source: the yaw of its x axis from the
    base's (positive towards L), and where its z points when that is not straight up."""
    origin = frame.T[:3, 3]
    x, _, z = frame.axes
    yaw = int(np.round(np.degrees(np.arctan2(x[1], x[0]))))
    tilt = "" if z[2] > np.cos(np.radians(0.5)) else f", z {heading(z)}"
    return (f"{frame.name} in base: origin F{origin[0]:+.3f} L{origin[1]:+.3f} U{origin[2]:+.3f} m, "
            f"yaw {yaw:+d} deg{tilt} (from {frame.source})")


def card(k, reach=None) -> str:
    """The embodiment card: what this robot is and what it can do, for the top of a policy's context."""
    from .bodies.sim import SimBody
    m = k.manifest
    sim = bool(getattr(k.body, "simulated", False))
    lines = [f"# {m.name}" + (" (simulated)" if sim else ""),
             f"{m.n} joints, control at {m.rate_hz:.0f} Hz; senses: {', '.join(sorted(m.sensing))}."]
    if isinstance(k.body, SimBody):
        lines.append("This is a simulation: MuJoCo models gravity and frictional contacts with approximate actuators.")
    if sim and m.hardware_notes:
        lines.append("Lines marked 'hardware:' describe the real robot.")
    if "torque" not in m.sensing:
        lines.append("no joint torque sensing: touchdown, guarded and contact monitoring are unavailable.")
    lines.append("joints (deg): " + "; ".join(f"j{i + 1} {j.name} {np.degrees(j.lower):.0f}..{np.degrees(j.upper):.0f}"
                                          for i, j in enumerate(m.joints)))
    if m.gripper:
        g = m.gripper
        span = "" if g.m_per_unit is None else f" = 0..{1000 * abs(g.aperture(g.open) or 0):.0f} mm opening"
        mm = " gripper and grip also take millimetres (aperture_mm, start_mm, expect_mm)." if g.m_per_unit else ""
        lines.append(f"gripper: {g.closed}..{g.open} {g.unit} (closed..open){span}.{mm} grip squeezes "
                     f"{g.squeeze} {g.unit} past contact.")
        lines.append(tool_line(k) or "")
    lines.append(f"one Cartesian move at most {100 * m.max_segment_m:.0f} cm; default peak joint speed "
                 f"{m.speed} rad/s.")
    if "temperature" in m.sensing:
        lines.append(f"motors warn at {m.temp_warn_c:.0f} C, stop at {m.temp_limit_c:.0f} C.")
    if m.rest:
        lines.append("torque can only be released at the rest pose (the arm has no brakes).")
    if k.envelope.max_excursion is not None:
        lines.append(f"each joint may travel at most {np.degrees(k.envelope.max_excursion):.0f} deg "
                     "from the session start.")
    need = k.envelope.turn_height()
    if need is not None:
        joints, above = m.turn_clearance
        names = ", ".join(f"j{j + 1}" for j in joints)
        over = k.envelope.overrides.get("turn_height")
        why = f"operator override: {over['reason']}" if over else f"{100 * above:.0f} cm above where it started"
        lines.append(f"turning {names} needs the tool at U{need:+.3f} or higher ({why}); lift before any move that "
                     "turns them.")
    others = [f for f in k.world.frames if f not in ("base", "work")]
    lines.append("frames: " + frame_line(k.world.frame("work"))
                 + (f"; also {', '.join(others)}" if others else "")
                 + ". Positions use work-frame metres (x=forward, y=left, z=up); moves default to work.")
    lines += [box_line(k, b) for b in k.world.boxes.values()]
    cams = k.cameras
    if cams:
        lines.append(f"cameras: {', '.join(cams)}. `wu look NAME` saves an image and prints its path.")
    if k.enabled and k.active is None:
        lines.append((reach or reach_line)(k))
    if k.fit is not None:
        lines.append(f"torque model: {k.fit.headline()}; contact checks, load limits and rehearsals use it.")
    lines += [f"note: {n}" for n in m.notes]
    lines += [f"hardware: {n}" for n in m.hardware_notes]
    return "\n".join(lines)


# -- replies as the CLI and MCP print them; each interface adds its own way to act on them ---------------

def job_text(d: dict, *, answer: str, wait: str) -> str:
    """A job reply as a policy reads it: the outcome, or the question it waits on, then the state line. answer
    and wait are the interface's commands for answering a checkpoint and for waiting longer; {id} is the job."""
    lines = [f"warning: {d['warning']}"] if d.get("warning") else []
    if d.get("calibration"):
        lines.append(d["calibration"]["text"])
    if d.get("incident"):
        return "\n".join(lines + [d["incident"]])
    out, hint = d.get("outcome"), []
    if out:
        lines.append(f"job {d['id']} {out['status']}: {out['message']}")
    elif d["status"] == "waiting":
        q = d["question"]
        where = f" (look at: {q['view']}" + (f" {q['roi']}" if q.get("roi") else "") + ")" if q.get("view") else ""
        lines.append(f"job {d['id']} is waiting at a checkpoint: {q['ask']}{where}")
        hint = ["answer with: " + answer.format(id=d["id"])]
    else:
        lines.append(f"job {d['id']} is still {d['status']}: {d['what']}")
        hint = ["keep waiting with: " + wait.format(id=d["id"])]
    return "\n".join(lines + [d["line"]] + hint)


def home_text(r: dict) -> str:
    """A home-route reply: the route and, once one is set, whether its rehearsal from here passes."""
    line = f"home: {r['home']}"
    if "ok" not in r:
        return line
    return line + ("; rehearsed from here, it passes" if r["ok"] else "\n" + r["text"])


def event_line(e: dict) -> str:
    return f"[{e['seq']}] {e['t']:>7.1f}s {e['level']:5s} {e['kind']}: {e['message']}"


def record_line(s: dict) -> str:
    """A flight record's summary: powered and moving time, peak motor temperatures, a failed recording."""
    share = s.get("moving_share")
    return (f"powered {s.get('powered_s', 0)} s, moving {s.get('moving_s', 0)} s"
            + (f" ({100 * share:.0f}%)" if share is not None else "") + f"; max temps {s.get('max_temp_c')}"
            + (f"; recording failed: {s['recording_error']}" if s.get("recording_error") else ""))


def steps_text(steps: dict, step: str | None = None) -> str:
    """The steps a plan can use, each with an example; with a step's name, its parameters."""
    import json
    if step:
        h = steps.get(step)
        if h is None:
            raise ValueError(f"no step {step!r}; steps: {', '.join(steps)}")
        return "\n".join([f"{step}: {h['summary']}", h["params"], 'every step also takes "label"',
                          f"e.g. {json.dumps(h['example'])}" if h.get("example") else ""]).strip()
    lines = ["steps (a plan is a JSON list of them; help for one step lists its parameters):"]
    for kind, h in steps.items():
        lines.append(f"  {kind:10s} {h['summary']}")
        if h.get("example"):
            lines.append(f"  {'':10s} {json.dumps(h['example'])}")
    return "\n".join(lines)


def parse_value(text: str):
    """A value typed as text: JSON when it parses (0.205, true, [1, 2]), otherwise the text itself."""
    import json
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text
