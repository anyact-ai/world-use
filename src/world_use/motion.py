"""Joint trajectories: straight tool lines, blended polylines, Cartesian moves and joint moves.

Pure math, sampled at the control rate. A path is an (N, n) array of joint positions, one row per tick,
starting one tick after the current pose and ending exactly at the goal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import Refused
from .geometry import interpolate_rotation, rotation_log
from .kinematics import Chain

KNOTS = 300                           # most IK knots per straight stretch (about one per millimetre)


@dataclass(frozen=True)
class Timing:
    rate_hz: float = 100.0
    speed: float = 0.25               # rad/s: automatic durations make the fastest joint peak at this
    accel: float = 3.0                # rad/s^2: ...and keep peak joint acceleration under this
    min_s: float = 0.5                # no move is shorter than this
    ik_tol: float = 3e-3              # combined m / rad residual accepted along a Cartesian path


def minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def cruise(tau, ramp=0.2):
    """0..1 position profile: min-jerk-shaped speed-up over `ramp` of the time, constant speed, slow-down.

    Peak speed is 1 / (1 - ramp) times the average (1.875 for plain min-jerk), so long paths finish sooner
    at the same joint-speed cap. Acceleration and jerk start and end at zero.
    """
    tau = np.clip(tau, 0.0, 1.0)
    ramp = float(np.clip(ramp, 1e-3, 0.5))
    def area(x):                                           # integral of minjerk from 0 to x
        return 2.5 * x**4 - 3 * x**5 + x**6

    total = 1.0 - ramp
    out = np.where(tau < ramp, ramp * area(tau / ramp),
                   np.where(tau > 1 - ramp, total - ramp * area((1 - tau) / ramp), 0.5 * ramp + (tau - ramp)))
    return out / total


def resample(knots, duration, rate_hz, shape=minjerk):
    n = max(2, round(duration * rate_hz))
    s = shape(np.arange(1, n + 1) / n) * (len(knots) - 1)
    lo = np.clip(np.floor(s).astype(int), 0, len(knots) - 2)
    frac = (s - lo)[:, None]
    return knots[lo] * (1 - frac) + knots[lo + 1] * frac


def time_scale(knots, duration, timing: Timing, shape=minjerk):
    """Time a geometric path (knots uniform in path length). Without a duration, pick the shortest one that
    keeps the actual peak joint speed and acceleration inside `timing`: measure a trial timing and rescale,
    since speeds scale as 1/T and accelerations as 1/T^2."""
    knots = np.asarray(knots, float)
    if duration is None:
        trial = 10.0
        full = np.vstack([knots[:1], resample(knots, trial, timing.rate_hz, shape)])
        peak_v = np.abs(np.diff(full, axis=0)).max() * timing.rate_hz
        peak_a = np.abs(np.diff(full, n=2, axis=0)).max() * timing.rate_hz ** 2
        duration = max(timing.min_s, trial * peak_v / timing.speed, trial * np.sqrt(peak_a / timing.accel))
    return resample(knots, duration, timing.rate_hz, shape), float(duration)


def joint_move(q0, q1, timing: Timing, duration=None):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if np.abs(q1 - q0).max() < 1e-6:
        return np.repeat(q0[None, :], 2, axis=0), 2 / timing.rate_hz
    knots = q0 + (q1 - q0) * np.linspace(0, 1, 41)[:, None]
    return time_scale(knots, duration, timing)


def _solve(chain: Chain, q, targets, lower, upper, timing: Timing, weights=None, turning: str | None = None):
    """IK along a list of 4x4 targets, each seeded by the last. Returns (knots incl. q, worst residual). With
    `turning` (words for where the gripper is being turned to point), a refusal says how far the turn got."""
    q = np.asarray(q, float)
    p0 = chain.fk(q)[:3, 3]
    knots, worst, reached = [q.copy()], 0.0, None
    for i, T in enumerate(targets):
        q, res = chain.ik(T, q, lower, upper, weights)
        if res > timing.ik_tol and reached is None:
            reached = i                              # the first target it could not reach
        worst = max(worst, res)
        if worst > 10 * timing.ik_tol:            # fail fast instead of grinding through the rest
            break
        knots.append(q.copy())
    if worst > timing.ik_tol:
        total = float(np.linalg.norm(targets[-1][:3, 3] - p0))
        ok = 0.0 if not reached else float(np.linalg.norm(targets[reached - 1][:3, 3] - p0))
        if turning is not None:
            R0 = chain.fk(knots[0])[:3, :3]
            whole = np.degrees(np.linalg.norm(rotation_log(targets[-1][:3, :3] @ R0.T)))
            got = 0.0 if not reached else whole * reached / len(targets)
            if total < 1e-4:
                raise Refused(f"the gripper cannot turn to point {turning} here: it gets {got:.0f} of the "
                              f"{whole:.0f} deg (IK residual {worst * 1000:.1f} mm)", "reach",
                              "move the tool first: nose-down, for one, is reachable low and near the base",
                              residual_mm=round(worst * 1000, 1))
            raise Refused(f"only the first {100 * ok:.1f} cm of this {100 * total:.1f} cm move is reachable while the "
                          f"gripper turns to point {turning} (it gets {got:.0f} of {whole:.0f} deg)", "reach",
                          "turn where it can, then move, or the other way round", reachable_m=round(ok, 4),
                          length_m=round(total, 4))
        if total < 1e-4:
            raise Refused(f"that orientation is not reachable from here (IK residual {worst * 1000:.1f} mm)", "reach",
                          "turn less, or move the tool first", residual_mm=round(worst * 1000, 1))
        raise Refused(f"only the first {100 * ok:.1f} cm of this {100 * total:.1f} cm straight line is reachable with "
                      f"the gripper held at its current angle", "reach",
                      f"stop after {np.floor(100 * ok):.0f} cm, go another way, or first turn the gripper "
                      "(move_to with point, or a joints move)", reachable_m=round(ok, 4), length_m=round(total, 4))
    return np.array(knots), worst


def _bounds(chain: Chain, q0, lower=None, upper=None):
    """Start-pose contact (a joint resting on its stop) is allowed: never plan past where we started."""
    lo = chain.lower if lower is None else lower
    hi = chain.upper if upper is None else upper
    return np.minimum(lo, q0), np.maximum(hi, q0)


def cartesian(chain: Chain, q0, T_goal, timing: Timing, duration=None, lower=None, upper=None, weights=None,
              shape=minjerk, knots=None, turning: str | None = None):
    """Tool point along a straight line to T_goal; orientation turns along the shortest rotation. `knots` overrides
    the IK resolution (about one per millimetre), e.g. for a quick feasibility probe."""
    q0 = np.asarray(q0, float)
    T0 = chain.fk(q0)
    delta = T_goal[:3, 3] - T0[:3, 3]
    turn = np.linalg.norm(np.asarray(T_goal[:3, :3] @ T0[:3, :3].T) - np.eye(3))
    length = float(np.linalg.norm(delta))
    if length < 1e-6 and turn < 1e-6:
        raise Refused("move has zero length", "zero_length")
    n = knots or int(np.clip(max(length / 0.001, turn / 0.005), 40, KNOTS))
    targets = []
    for k in range(1, n + 1):
        T = np.eye(4)
        T[:3, :3] = interpolate_rotation(T0[:3, :3], T_goal[:3, :3], k / n)
        T[:3, 3] = T0[:3, 3] + delta * k / n
        targets.append(T)
    lo, hi = _bounds(chain, q0, lower, upper)
    knots, worst = _solve(chain, q0, targets, lo, hi, timing, weights, turning)
    path, duration = time_scale(knots, duration, timing, shape)
    return path, duration, worst


def line(chain: Chain, q0, delta, timing: Timing, duration=None, lower=None, upper=None, weights=None, shape=minjerk,
         knots=None):
    """Tool point moves by `delta` (metres, base frame) along a straight line; orientation is held."""
    T = chain.fk(q0).copy()
    T[:3, 3] += np.asarray(delta, float)
    return cartesian(chain, q0, T, timing, duration, lower, upper, weights, shape, knots)


def polyline(chain: Chain, q0, legs, timing: Timing, blend=0.02, duration=None, lower=None, upper=None,
             weights=None, sharp_deg=120.0):
    """Several straight legs (metres, base frame, each relative to the end of the last) as one motion.

    Corners are rounded by `blend` metres so the joints keep moving; turns sharper than `sharp_deg` cannot be
    rounded and stop instead. Each stretch between stops speeds up once, cruises and slows down once.
    """
    q0 = np.asarray(q0, float)
    T0 = chain.fk(q0)
    pts = [T0[:3, 3].copy()]
    for leg in legs:
        d = np.asarray(leg, float)
        if np.linalg.norm(d) < 1e-6:
            raise Refused("a leg has zero length", "zero_length")
        pts.append(pts[-1] + d)
    pts = np.array(pts)
    stretches, first = [], 0
    for k in range(1, len(pts) - 1):
        a, b = pts[k] - pts[k - 1], pts[k + 1] - pts[k]
        turn = np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
        if turn > sharp_deg:
            stretches.append(pts[first:k + 1])
            first = k
    stretches.append(pts[first:])
    lengths = [float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) for p in stretches]
    lo, hi = _bounds(chain, q0, lower, upper)
    q, worst, pieces, total = q0.copy(), 0.0, [], 0.0
    for p, length in zip(stretches, lengths, strict=True):
        targets = []
        for x in blended_curve(p, blend):
            T = T0.copy()
            T[:3, 3] = x
            targets.append(T)
        knots, w = _solve(chain, q, targets, lo, hi, timing, weights)
        worst = max(worst, w)
        share = None if duration is None else duration * length / sum(lengths)
        piece, d = time_scale(knots, share, timing, shape=cruise)
        pieces.append(piece)
        total += d
        q = piece[-1].copy()
    return np.vstack(pieces), total, worst


def blended_curve(pts, blend):
    """Points about 1 mm apart along the polyline, each interior corner rounded by a quadratic Bezier."""
    curve = [pts[0]]
    for k in range(1, len(pts)):
        a, b = pts[k - 1], pts[k]
        seg = b - a
        if k < len(pts) - 1:
            nxt = pts[k + 1] - b
            r = min(blend, 0.5 * float(np.linalg.norm(seg)), 0.5 * float(np.linalg.norm(nxt)))
            c_in, c_out = b - seg / np.linalg.norm(seg) * r, b + nxt / np.linalg.norm(nxt) * r
            n_line = max(2, int(np.linalg.norm(c_in - curve[-1]) / 0.001))
            curve.extend(curve[-1] + (c_in - curve[-1]) * np.linspace(0, 1, n_line + 1)[1:, None])
            s = np.linspace(0, 1, max(8, int(4 * r / 0.001)) + 1)[1:, None]
            curve.extend((1 - s) ** 2 * c_in + 2 * (1 - s) * s * b + s ** 2 * c_out)
        else:
            n_line = max(2, int(np.linalg.norm(b - curve[-1]) / 0.001))
            curve.extend(curve[-1] + (b - curve[-1]) * np.linspace(0, 1, n_line + 1)[1:, None])
    curve = np.array(curve)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])
    s = np.linspace(0, arc[-1], int(np.clip(arc[-1] / 0.001, 40, KNOTS * (len(pts) - 1))) + 1)[1:]
    return np.array([np.interp(s, arc, curve[:, i]) for i in range(3)]).T
