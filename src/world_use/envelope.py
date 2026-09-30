"""The envelope: limits every motion stays inside. Paths are checked whole before anything moves; measured
state is checked on every tick while it moves. A policy can tighten it; loosening needs an operator override,
which is scoped, has a reason, and shows up in the log."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from numbers import Real

import numpy as np

from .body import JointState, Manifest
from .errors import Refused
from .kinematics import Chain
from .world import World

MARGIN = 0.02                         # rad kept clear of the URDF limits
TURN_EPS = np.radians(2.0)            # a turn-clearance joint moving less than this in a path is not a turn
POINT_EVERY = 40                      # sample gravity loads and surface clearance; zones check every tick


@dataclass
class Trip:
    """A watchdog finding on measured state. `isolate` = only the gripper is affected; the arm carries on."""
    kind: str                          # fault, blocked, overload, contact, hot, keep_out, gripper
    message: str
    joint: int | None = None
    value: float | None = None
    limit: float | None = None
    isolate: bool = False


class Envelope:
    def __init__(self, manifest: Manifest, chain: Chain, world: World, q_start):
        self.m, self.chain, self.world = manifest, chain, world
        self.q_start = np.asarray(q_start, float)
        self.max_excursion = manifest.max_excursion
        self.overrides: dict[str, dict] = {}
        self.rehearsal: list[tuple[str, list[Refused]]] | None = None   # a twin's record of problems, else None
        self.context = ""                                                  # the step being planned, for that record
        if not np.isfinite(manifest.link_radius_m) or manifest.link_radius_m < 0:
            raise ValueError("link_radius_m must be finite and nonnegative")

    def override(self, key: str, value, reason: str):
        """Operator-only loosening, e.g. max_excursion for one task. Recorded with its reason. turn_height is the
        lowest tool height (work frame, m) at which the turn-clearance joints may turn, instead of start + clearance."""
        if key == "max_excursion":
            self.overrides[key] = dict(value=value, reason=reason, was=self.max_excursion)
            self.max_excursion = value
        elif key == "turn_height":
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError("turn_height must be a finite number")
            was = None if self.m.turn_clearance is None else self._turn_height(self.m.turn_clearance[1])
            self.overrides[key] = dict(value=float(value), reason=reason, was=was)
        else:
            raise KeyError(f"no override named {key!r}")

    # -- before motion ----------------------------------------------------------------------------
    def bounds(self, q_from) -> tuple[np.ndarray, np.ndarray]:
        """Joint bounds for planning: the URDF limits minus a margin, but never tighter than the pose the move
        starts from or the pose the session started in (a folded arm rests on its stops; going back is safe)."""
        q_from = np.asarray(q_from, float)
        lo = np.minimum(self.m.lower + MARGIN, np.minimum(q_from, self.q_start))
        hi = np.maximum(self.m.upper - MARGIN, np.maximum(q_from, self.q_start))
        return lo, hi

    def check_path(self, path, q_from, allow_contact=False) -> dict:
        """Check the whole path against every limit before anything moves. Raises one Refused that names every
        limit it would break, with the numbers; in a rehearsal (a twin), records them and lets it run on, so one
        check finds every problem in a plan. Returns a few plain numbers."""
        path = np.asarray(path, float)
        q_from = np.asarray(q_from, float)
        if not np.all(np.isfinite(path)):
            raise Refused("path contains NaN or infinite values", "finite")
        problems: list[Refused] = []
        rate = self.m.rate_hz
        full = np.vstack([q_from, path])
        v = np.abs(np.diff(full, axis=0)).max(axis=0) * rate
        a = np.abs(np.diff(full, n=2, axis=0)).max(axis=0) * rate ** 2 if len(full) > 2 else np.zeros(len(q_from))
        for i, j in enumerate(self.m.joints):
            if v[i] > j.v_max + 1e-9:
                problems.append(Refused(f"{j.name} would move at {v[i]:.2f} rad/s; limit {j.v_max}", "speed",
                                        "give it a longer duration", joint=i + 1))
            if a[i] > j.a_max + 1e-9:
                problems.append(Refused(f"{j.name} would accelerate at {a[i]:.1f} rad/s^2; limit {j.a_max}", "accel",
                                        "give it a longer duration", joint=i + 1))
        lo, hi = self.bounds(q_from)
        for i in np.where((path.min(0) < lo - 1e-9) | (path.max(0) > hi + 1e-9))[0]:
            worst = path[:, i].min() if path[:, i].min() < lo[i] - 1e-9 else path[:, i].max()
            problems.append(Refused(f"{self.m.joints[i].name} would reach {np.degrees(worst):.1f} deg; its range is "
                                    f"{np.degrees(lo[i]):.1f}..{np.degrees(hi[i]):.1f} deg", "joint_limit",
                                    "approach from another direction", joint=int(i) + 1))
        if self.max_excursion is not None:
            swing = np.abs(path - self.q_start)
            swing[:, [i for i, j in enumerate(self.m.joints) if j.excursion_exempt]] = 0.0
            if swing.max() > self.max_excursion:
                i = int(np.unravel_index(swing.argmax(), swing.shape)[1])
                problems.append(Refused(f"{self.m.joints[i].name} would travel {np.degrees(swing.max()):.0f} deg from "
                                        f"the session start; limit {np.degrees(self.max_excursion):.0f} deg",
                                        "excursion",
                                        "fold back part of the way first, or ask the operator for an override",
                                        joint=i + 1))
        sample = np.vstack([path[:: max(1, len(path) // POINT_EVERY)], path[-1:]])
        hold_max = np.array([j.tau_hold_max for j in self.m.joints])
        if np.isfinite(hold_max).any() and self.chain.links:
            g = np.abs([self.chain.gravity(q) for q in sample]).max(axis=0)
            for i in np.where(g > hold_max)[0]:
                problems.append(Refused(f"{self.m.joints[i].name} would hold {g[i]:.1f} Nm against gravity; limit "
                                        f"{hold_max[i]}", "gravity_load", "stay closer to the base", joint=int(i) + 1))
        tool_z, clearance, deepest = [], np.inf, {}
        tool0 = self.chain.fk(q_from)[:3, 3]
        # a path may start in contact (the tool resting on a table) and move away, never deeper than it began
        allowed = {b.name: max(0.0, b.depth(tool0)) + 0.001 for b in self.world.solids()}
        for q in sample:
            pts = self.chain.points(q)
            tool = pts[-1]
            tool_z.append(tool[2])
            for box in self.world.solids():
                depth = box.depth(tool)
                if depth > allowed[box.name] and not allow_contact:
                    deepest[box.name] = max(depth, deepest.get(box.name, 0.0))
                if np.isfinite(depth):
                    clearance = min(clearance, -depth)
        for name, depth in deepest.items():
            problems.append(Refused(f"the tool would go {1000 * depth:.0f} mm into {name!r}", "surface",
                                    "stop above it, or use a guarded move (touchdown) to make contact", box=name))
        keep_out, slow = self.world.of_kind("keep_out"), self.world.of_kind("slow")
        entered, too_fast = set(), set()
        previous, previous_tool = q_from, tool0
        if keep_out or slow:
            for q in full:
                pts = self.chain.points(q)
                margin = self.m.link_radius_m + self.chain.motion_bound(previous, q)
                for box in keep_out:
                    if box.name not in entered and self._intersects(box, pts, margin):
                        entered.add(box.name)
                        problems.append(Refused(f"the arm would enter keep-out zone {box.name!r}", "keep_out",
                                                "go around it", box=box.name))
                speed = float(np.linalg.norm(pts[-1] - previous_tool) * rate)
                for box in slow:
                    if (box.name not in too_fast and speed > box.params["speed"] + 1e-9
                            and box.intersects_segment(previous_tool, pts[-1])):
                        too_fast.add(box.name)
                        problems.append(Refused(f"tool speed {speed:.3f} m/s in slow zone {box.name!r}; "
                                                f"limit {box.params['speed']:.3f}", "slow",
                                                "give the move a longer duration", box=box.name))
                previous, previous_tool = q, pts[-1]
        turn = self.turn_problem(full, rate)
        if turn is not None:
            problems.append(turn)
        if problems:
            if self.rehearsal is None:
                raise Refused.several(problems)
            self.rehearsal.append((self.context, problems))
        return dict(seconds=round(len(path) / rate, 2), peak_speed=round(float(np.max(v)), 3),
                    peak_accel=round(float(np.max(a)), 2), lowest_tool_z=round(float(min(tool_z)), 4),
                    surface_clearance=None if not np.isfinite(clearance) else round(float(clearance), 4))

    def turn_height(self) -> float | None:
        """Lowest tool height (work frame) at which the turn-clearance joints may turn, if the robot has the rule."""
        if self.m.turn_clearance is None:
            return None
        return self._turn_height(self.m.turn_clearance[1])

    def _turn_height(self, above: float) -> float:
        if "turn_height" in self.overrides:
            return float(self.overrides["turn_height"]["value"])
        return self._up(self.chain.fk(self.q_start)[:3, 3]) + above

    def _up(self, p) -> float:
        return float(self.world.from_base("work", p)[2]) if "work" in self.world.frames else float(p[2])

    def turn_problem(self, full, rate) -> Refused | None:
        if self.m.turn_clearance is None:
            return None
        joints, above = self.m.turn_clearance
        turn = full[:, list(joints)]
        if np.abs(turn - turn[0]).max() < TURN_EPS:        # incidental: a straight line nudges these a little
            return None
        turning = np.abs(np.diff(turn, axis=0)).max(axis=1) * rate > 1e-3
        poses = full[1:][turning][:: max(1, int(turning.sum()) // 40)]
        z = min(self._up(self.chain.fk(q)[:3, 3]) for q in poses)
        need = self._turn_height(above)
        if z >= need:
            return None
        names = "/".join(f"j{j + 1}" for j in joints)
        lift = max(1, int(np.ceil(100 * (need - z) - 1e-3)))
        why = (f"an operator override: {self.overrides['turn_height']['reason']}" if "turn_height" in self.overrides
               else f"{100 * above:.0f} cm above the start height")
        return Refused(f"this turns {names} with the tool at U{z:+.3f}; turning needs U{need:+.3f} or higher "
                       f"({why}), or the gripper sweeps across the table",
                       "turn_clearance", f"lift at least {lift} cm more first", tool_up=round(z, 4),
                       need_up=round(need, 4))

    # -- during motion ----------------------------------------------------------------------------
    def watch(self, st: JointState, q_cmd, grip_cmd=None, ignore_heat=False) -> Trip | None:
        """Watchdog on one measurement against what was commanded. Returns the first finding, or None."""
        if st.faults:
            return Trip("fault", "; ".join(st.faults))
        if st.q is None or not np.all(np.isfinite(st.q)):
            return Trip("fault", "no position feedback")
        err = np.abs(np.asarray(st.q) - np.asarray(q_cmd))
        for i, j in enumerate(self.m.joints):
            if err[i] > j.track_tol:
                return Trip("blocked",
                            f"{j.name} is {np.degrees(err[i]):.1f} deg behind its command: something is in the way",
                            i, float(err[i]), j.track_tol)
        if st.tau is not None:
            tau = np.abs(np.asarray(st.tau, float))
            for i, j in enumerate(self.m.joints):
                if tau[i] > j.tau_max:
                    return Trip("overload", f"{j.name} torque {tau[i]:.1f} Nm; limit {j.tau_max}", i, float(tau[i]),
                                j.tau_max)
        zones = self.world.of_kind("keep_out")
        if zones:
            pts = self.chain.points(st.q)
            for box in zones:
                if self._intersects(box, pts, self.m.link_radius_m):
                    return Trip("keep_out", f"the arm entered keep-out zone {box.name!r}")
        g = self.m.gripper
        if g is not None and grip_cmd is not None and st.gripper is not None:
            pushing = st.gripper_tau is None or abs(st.gripper_tau) > 0.5 * g.tau_max
            if abs(st.gripper - grip_cmd) > g.track_tol and pushing:
                return Trip("gripper", f"gripper is {st.gripper - grip_cmd:+.2f} {g.unit} off its command",
                            isolate=True)
            if st.gripper_tau is not None and abs(st.gripper_tau) > g.tau_max:
                return Trip("gripper", f"gripper effort {st.gripper_tau:.1f}; limit {g.tau_max}", isolate=True)
        if st.temp is not None and not ignore_heat:
            temp = np.nan_to_num(np.asarray(st.temp, float), nan=0.0)
            if temp.max() > self.m.temp_limit_c:
                i = int(np.argmax(temp))
                return Trip("hot", f"{self._name(i)} is at {temp[i]:.0f} C", i, float(temp[i]), self.m.temp_limit_c)
        return None

    @staticmethod
    def _intersects(box, points, margin):
        # The fixed base pedestal is excluded; all links from the first joint through the tool are included.
        return any(box.intersects_segment(a, b, margin) for a, b in pairwise(points[1:]))

    def _name(self, i):
        return self.m.joints[i].name if i < self.m.n else "gripper"
