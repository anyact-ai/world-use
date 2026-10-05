"""Find where a camera is from the arm itself: the tool visits the corners of a box, the policy says where it sees
the tool point in the camera's picture, and a pinhole camera is fitted to those pairs.

A model names a pixel to within about 10-20 px. With eight corners of a box 12-24 cm across and a camera about a
metre away, that fixes the camera's direction and height well, but not how far away it is, which trades against its
focal length: left free on the simulator the focal length came out 36% wrong. So the fit leans on the field of view
the camera is said to have (its workcell fov_deg, else 60 deg) and says how much it did. On the simulator, with
answers scattered by 12 px, what it then drew in the workspace landed 7-9 px from where it belongs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .cameras import View
from .errors import Refused
from .geometry import axis_angle

NOISE_PX = 15.0                       # how far off a model's pixel answers are taken to be
PRIOR_SD = 0.25                       # how far (in log focal length) the camera may be from its stated field of view
MIN_POINTS = 6
GRAY = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 0, 1), (0, 0, 1))   # one axis per move


# -- the tour ------------------------------------------------------------------------------------------

def tour(k, camera: str, size, points: int = 8, spreads=(0.12, 0.09, 0.06), check=None) -> tuple[list, float]:
    """A plan that takes the tool to `points` corners of a box around where it is (above the height where the base
    may turn) and asks at each where the tool point is in the camera's picture; then back to the start. The largest
    box of `spreads` (half-widths, m) the rehearsal passes is used: bigger fixes the camera better."""
    from .plan import check as check_here
    check = check or check_here
    if not MIN_POINTS <= points <= len(GRAY):
        raise Refused(f"a calibration visits {MIN_POINTS} to {len(GRAY)} points", "spec")
    here = k.world.from_base("work", k.chain.fk(k.cmd.q)[:3, 3])
    need = k.envelope.turn_height()
    bottom = here[2] if need is None else max(here[2], need + 0.01)
    w, h = size
    where = f" ({k.manifest.gripper.tool_point})" if k.manifest.gripper else ""
    problems = ""
    for a in spreads:
        steps: list[dict] = []
        for i, (fi, li, ui) in enumerate(GRAY[:points]):
            corner = [here[0] + (2 * fi - 1) * a, here[1] + (2 * li - 1) * a, bottom + 1.5 * a * ui]
            steps.append({"do": "move_to", "to": [round(float(c), 4) for c in corner],
                          "label": f"calibration point {i + 1}"})
            steps.append({"do": "checkpoint", "view": camera, "expect": None,
                          "ask": f"calibrating {camera!r} ({i + 1}/{points}): where is the tool point{where} when "
                                 f"you look at {camera} with the grid? answer x,y in pixels of that {w}x{h} "
                                 "picture (0,0 top left), or unseen"})
        steps.append({"do": "move_to", "to": [round(float(c), 4) for c in here], "label": "back to where it started"})
        report = check(steps, k)
        if not report.refused:
            return steps, a
        problems = "; ".join(p["message"] for p in report.problems) or report.outcome.message
    raise Refused(f"no calibration tour fits around the tool here ({problems})", "calibration_tour",
                  "move the tool to open space the camera sees well, above the turn height, and try again")


def parse(answer: str, size) -> tuple[float, float] | None:
    """A pixel from an answer like "512,300" or "x=512 y=300"; None for "unseen", no two numbers, or off the
    picture."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(answer))
    if len(nums) < 2:
        return None
    x, y = float(nums[0]), float(nums[1])
    return (x, y) if 0 <= x <= size[0] and 0 <= y <= size[1] else None


# -- the fit -------------------------------------------------------------------------------------------

@dataclass
class Fit:
    view: View
    R: np.ndarray                      # world (base) to camera
    C: np.ndarray                      # camera centre, base frame
    f: float
    rms: float                         # px, over the points used
    loo_rms: float                     # px, each point predicted from the others: the honest accuracy
    f_sd: float                        # relative uncertainty of the focal length
    from_prior: bool                   # whether the focal length came mostly from the stated field of view
    used: list[int]
    left_out: list[int] = field(default_factory=list)
    predicted: np.ndarray | None = None    # where the fit puts every point (px)

    @property
    def fov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.view.width / 2 / self.f)))


def solve(points, pixels, size, fov_deg: float = 60.0, focal: float | None = None) -> Fit:
    """A pinhole camera (rotation, centre, focal length; optical centre in the middle, square pixels) from points
    in the base frame and the pixels they were seen at. With `focal` the focal length is known and not fitted (a cut
    from a 360). Refuses what cannot fix a camera: too few points, points nearly in one plane or on one line, or a
    tool that barely moves in the picture (a camera on the arm)."""
    X, x = np.asarray(points, float), np.asarray(pixels, float)
    w, h = size
    if len(X) < MIN_POINTS:
        raise Refused(f"only {len(X)} answers placed the tool in the picture; at least {MIN_POINTS} are needed",
                      "calibration", "move the tool where the camera sees it, and calibrate again")
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    if s[1] < 0.1 * s[0]:
        raise Refused("the points seen lie nearly on one line", "calibration", "the camera must see the tool at "
                      "corners of the box on both layers")
    if s[2] < 0.1 * s[0]:
        raise Refused("the points seen lie nearly in one plane", "calibration", "the camera must see the tool at "
                      "corners of the box on both layers")
    if np.linalg.norm(x.std(0)) < 0.05 * np.hypot(w, h) and np.linalg.norm(X.std(0)) > 0.03:
        raise Refused("the tool hardly moves in this camera's picture: a camera on the arm moves with it, so it "
                      "cannot be calibrated this way", "calibration")
    c = np.array([w / 2, h / 2])
    f_prior = (w / 2) / np.tan(np.radians(fov_deg) / 2) if focal is None else focal
    R, C, f = _dlt(X, x, c)
    centre = X.mean(0)
    C = centre + (C - centre) * f_prior / f               # keep the picture's scale, take the stated focal length
    used = list(range(len(X)))
    R, C, f, sd = _refine(R, C, f_prior, X, x, c, f_prior, fixed=focal is not None)
    left_out: list[int] = []
    while len(used) > MIN_POINTS:
        miss = _leave_one_out(R, C, f, X, x, c, f_prior, focal is not None, used)
        worst = int(np.argmax(miss))
        if miss[worst] <= max(40.0, 3 * float(np.median(miss))):
            break
        left_out.append(used.pop(worst))
        R, C, f, sd = _refine(R, C, f, X[used], x[used], c, f_prior, fixed=focal is not None)
    miss = _leave_one_out(R, C, f, X, x, c, f_prior, focal is not None, used)
    uv, _ = _project(R, C, f, c, X)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R.T, C
    view = View(T, f, f, c[0], c[1], int(w), int(h))
    rms = float(np.sqrt(np.mean(np.sum((uv[used] - x[used]) ** 2, axis=1))))
    return Fit(view, R, C, f, rms, float(np.sqrt(np.mean(miss ** 2))), sd,
               focal is None and sd > 0.6 * PRIOR_SD, used, left_out, uv)


def _project(R, C, f, c, X) -> tuple[np.ndarray, np.ndarray]:
    Y = (np.asarray(X, float) - C) @ R.T
    return f * Y[:, :2] / Y[:, 2:3] + c, Y[:, 2]


def _rot(d) -> np.ndarray:
    n = float(np.linalg.norm(d))
    return np.eye(3) if n < 1e-12 else axis_angle(d / n, n)


def _dlt(X, x, c) -> tuple[np.ndarray, np.ndarray, float]:
    """A first camera from the points by the normalised direct linear transform, split into rotation, centre and
    focal length."""
    Xm, xm = X.mean(0), x.mean(0)
    sX = np.sqrt(3) / np.mean(np.linalg.norm(X - Xm, axis=1))
    sx = np.sqrt(2) / np.mean(np.linalg.norm(x - xm, axis=1))
    T3, T2 = np.diag([sX, sX, sX, 1.0]), np.diag([sx, sx, 1.0])
    T3[:3, 3], T2[:2, 2] = -sX * Xm, -sx * xm
    Xh = np.c_[X, np.ones(len(X))] @ T3.T
    xh = np.c_[x, np.ones(len(x))] @ T2.T
    A = []
    for Xi, (u, v, _) in zip(Xh, xh, strict=True):
        A.append(np.r_[Xi, np.zeros(4), -u * Xi])
        A.append(np.r_[np.zeros(4), Xi, -v * Xi])
    P = np.linalg.inv(T2) @ np.linalg.svd(np.array(A))[2][-1].reshape(3, 4) @ T3
    if np.median((np.c_[X, np.ones(len(X))] @ P.T)[:, 2]) < 0:
        P = -P                                             # the points are in front of the camera
    M = P[:, :3]
    flip = np.flipud(np.eye(3))                            # RQ from QR
    Q, U = np.linalg.qr((flip @ M).T)
    K, R = flip @ U.T @ flip, flip @ Q.T
    S = np.diag(np.sign(np.diag(K)))
    K, R = K @ S, S @ R
    if np.linalg.det(R) < 0:
        raise Refused("the answers do not fit any camera (they fit only its mirror image)", "calibration",
                      "check that x counts across and y down, from the top left")
    K = K / K[2, 2]
    C = -np.linalg.solve(M, P[:, 3])
    return R, C, float((K[0, 0] + K[1, 1]) / 2)


def _refine(R0, C0, f0, X, x, c, f_prior, fixed=False, iterations=60):
    """Damped least squares on the rotation, the centre and (unless fixed) the log focal length, with the stated
    field of view as a soft prior. Returns (R, C, f, relative sd of f)."""
    def residuals(p):
        f = f0 if fixed else np.exp(p[6])
        uv, z = _project(_rot(p[:3]) @ R0, p[3:6], f, c, X)
        r = ((uv - x) / NOISE_PX).ravel()
        if np.any(z <= 0):
            r = r + 1e3                                    # behind the camera: never a solution
        return r if fixed else np.r_[r, (p[6] - np.log(f_prior)) / PRIOR_SD]

    p = np.r_[np.zeros(3), C0, np.log(f0)]
    n = 6 if fixed else 7
    lam, r = 1e-3, residuals(p)
    J = np.zeros((len(r), n))
    for _ in range(iterations):
        for j in range(n):
            e = np.zeros_like(p)
            e[j] = 1e-6
            J[:, j] = (residuals(p + e) - residuals(p - e)) / 2e-6
        A = J.T @ J
        step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -J.T @ r)
        trial = p.copy()
        trial[:n] += step
        rt = residuals(trial)
        if rt @ rt < r @ r:
            p, r, lam = trial, rt, lam / 3
            if np.linalg.norm(step) < 1e-9:
                break
        else:
            lam *= 5
    sd = 0.0
    if not fixed:
        s2 = max(1.0, float(r[:-1] @ r[:-1]) / max(1, len(r) - 1 - n))
        sd = float(np.sqrt(np.linalg.pinv(J.T @ J)[6, 6] * s2))
    return _rot(p[:3]) @ R0, p[3:6], float(f0 if fixed else np.exp(p[6])), sd


def _leave_one_out(R, C, f, X, x, c, f_prior, fixed, used) -> np.ndarray:
    """For each point used, how far the camera fitted to the others misses it (px)."""
    miss = []
    for i in used:
        rest = [j for j in used if j != i]
        Ri, Ci, fi, _ = _refine(R, C, f, X[rest], x[rest], c, f_prior, fixed, iterations=15)
        uv, _ = _project(Ri, Ci, fi, c, X[i:i + 1])
        miss.append(float(np.linalg.norm(uv[0] - x[i])))
    return np.array(miss)


def installable(fit: Fit, points, size) -> str | None:
    """Why a fit should not be used, or None if it may."""
    _, z = _project(fit.R, fit.C, fit.f, np.array(size) / 2, np.asarray(points, float)[fit.used])
    if fit.rms > 0.02 * size[0]:
        return f"the fit misses by {fit.rms:.0f} px rms, more than 2% of the picture's width"
    if np.any(z <= 0):
        return "some points would be behind the camera"
    if not 20.0 <= fit.fov_deg <= 120.0:
        return f"a field of view of {fit.fov_deg:.0f} deg is not a camera's"
    return None


def keep(fit: Fit, points, cut=None) -> str:
    """The workcell lines that keep this calibration. They are in the base frame: the work frame turns with the
    arm's rest angle, which differs from one session to the next. For a cut from a 360 it is the 360's pose."""
    centre = np.asarray(points, float)[fit.used].mean(0)

    def v(p):
        return "[" + ", ".join(f"{x:.4f}" for x in p) + "]"
    if cut is None:
        z = fit.R[2]                                        # the optical axis, base frame
        return "\n".join(['frame = "base"', f"eye = {v(fit.C)}", f"look_at = {v(fit.C + ((centre - fit.C) @ z) * z)}",
                          f"up = {v(-fit.R[1])}", f"fov_deg = {fit.fov_deg:.1f}"])
    pano = fit.R.T @ cut.R.T                                # the 360's own frame, in the base frame
    return "\n".join(['frame = "base"', f"eye = {v(fit.C)}", f"facing = {v(-pano[:, 0])}", f"up = {v(pano[:, 2])}",
                      f"look_at = {v(centre)}"])


def describe(name: str, fit: Fit, answers: list[dict], pixels: list, size, why: str | None, lines: str,
             focal_known: bool = False) -> str:
    """The calibration as a policy reads it: every point, what the fit makes of it, and whether it was installed."""
    seen = [i for i, px in enumerate(pixels) if px is not None]
    out = [f"calibration of {name!r} from {len(answers)} answers ({len(fit.used)} used"
           + (f", {len(fit.left_out)} left out as outliers" if fit.left_out else "") + ", pixels of a "
           f"{size[0]}x{size[1]} picture):", "  point  tool F, L, U (work)      answer      fit        miss"]
    for i, a in enumerate(answers):
        f, left, u = a["tool"]
        where = f"  {i + 1:>5}  {f:+.3f} {left:+.3f} {u:+.3f}   "
        if pixels[i] is None:
            out.append(where + f"{str(a['answer'])[:10]:<11} (unseen or off the picture)")
            continue
        j = seen.index(i)
        (x, y), (px, py) = pixels[i], fit.predicted[j] if fit.predicted is not None else (np.nan, np.nan)
        note = " left out" if j in fit.left_out else ""
        out.append(where + f"{x:>4.0f},{y:<4.0f}   {px:>4.0f},{py:<4.0f}   {np.hypot(px - x, py - y):4.0f} px{note}")
    out.append(f"  fit {fit.rms:.1f} px rms; each point predicted from the others, {fit.loo_rms:.1f} px: the accuracy "
               "to expect")
    fov = f"field of view {fit.fov_deg:.1f} deg (focal length {fit.f:.0f} px)"
    out.append("  " + fov + (", known" if focal_known else f" +-{100 * fit.f_sd:.0f}%"
                             + (", mostly from the field of view it was said to have" if fit.from_prior else "")))
    if why is None:
        out.append("  installed for this session. To keep it, put these lines in the camera's workcell entry:")
        out += ["    " + line for line in lines.splitlines()]
    else:
        out.append(f"  not installed: {why}. Nothing changed.")
    return "\n".join(out)
