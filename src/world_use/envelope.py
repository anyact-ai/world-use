"""The envelope: limits every motion stays inside. Paths are checked whole before anything moves; measured
state is checked on every tick while it moves. A policy can tighten it; loosening needs an operator override,
which is scoped, has a reason, and shows up in the log."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .body import JointState, Manifest
from .errors import Refused
from .kinematics import Chain
from .world import World

MARGIN = 0.02                         # rad kept clear of the URDF limits
POINT_EVERY = 40                      # check about this many poses along a path against the world


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
        self.z_start = float(chain.fk(self.q_start)[2, 3])
        self.max_excursion = manifest.max_excursion
        self.overrides: dict[str, dict] = {}

    def override(self, key: str, value, reason: str):
        """Operator-only loosening, e.g. max_excursion for one task. Recorded with its reason."""
        if key != "max_excursion":
            raise KeyError(f"no override named {key!r}")
        self.overrides[key] = dict(value=value, reason=reason, was=self.max_excursion)
        self.max_excursion = value

    # -- before motion ----------------------------------------------------------------------------
    def bounds(self, q_from) -> tuple[np.ndarray, np.ndarray]:
        """Joint bounds for planning: the URDF limits minus a margin, but never tighter than the pose the move
        starts from or the pose the session started in (a folded arm rests on its stops; going back is safe)."""
        q_from = np.asarray(q_from, float)
        lo = np.minimum(self.m.lower + MARGIN, np.minimum(q_from, self.q_start))
        hi = np.maximum(self.m.upper - MARGIN, np.maximum(q_from, self.q_start))
        return lo, hi

    def check_path(self, path, q_from, allow_contact=False) -> dict:
        """Refuse (raise Refused) unless the whole path is inside every limit. Returns a few plain numbers."""
        path = np.asarray(path, float)
        q_from = np.asarray(q_from, float)
        if not np.all(np.isfinite(path)):
            raise Refused("path contains NaN or infinite values", "finite")
        rate = self.m.rate_hz
        full = np.vstack([q_from, path])
        v = np.abs(np.diff(full, axis=0)).max(axis=0) * rate
        a = np.abs(np.diff(full, n=2, axis=0)).max(axis=0) * rate ** 2 if len(full) > 2 else np.zeros(len(q_from))
        for i, j in enumerate(self.m.joints):
            if v[i] > j.v_max + 1e-9:
                raise Refused(f"{j.name} would move at {v[i]:.2f} rad/s; limit {j.v_max}", "speed",
                              "give it a longer duration", joint=i + 1)
            if a[i] > j.a_max + 1e-9:
                raise Refused(f"{j.name} would accelerate at {a[i]:.1f} rad/s^2; limit {j.a_max}", "accel",
                              "give it a longer duration", joint=i + 1)
        lo, hi = self.bounds(q_from)
        bad = np.where((path.min(0) < lo - 1e-9) | (path.max(0) > hi + 1e-9))[0]
        if len(bad):
            names = ", ".join(self.m.joints[i].name for i in bad)
            raise Refused(f"joint limit: {names}", "joint_limit", "approach from another direction", joints=[int(i) + 1 for i in bad])
        if self.max_excursion is not None:
            swing = np.abs(path - self.q_start)
            swing[:, [i for i, j in enumerate(self.m.joints) if j.excursion_exempt]] = 0.0
            if swing.max() > self.max_excursion:
                i = int(np.unravel_index(swing.argmax(), swing.shape)[1])
                raise Refused(f"{self.m.joints[i].name} would travel {np.degrees(swing.max()):.0f} deg from the session "
                              f"start; limit {np.degrees(self.max_excursion):.0f} deg", "excursion",
                              "fold back part of the way first, or ask the operator for an override", joint=i + 1)
        sample = np.vstack([path[:: max(1, len(path) // POINT_EVERY)], path[-1:]])
        hold_max = np.array([j.tau_hold_max for j in self.m.joints])
        if np.isfinite(hold_max).any() and self.chain.links:
            g = np.abs([self.chain.gravity(q) for q in sample]).max(axis=0)
            over = np.where(g > hold_max)[0]
            if len(over):
                i = int(over[0])
                raise Refused(f"{self.m.joints[i].name} would hold {g[i]:.1f} Nm against gravity; limit {hold_max[i]}",
                              "gravity_load", "stay closer to the base", joint=i + 1)
        tool_z, clearance = [], np.inf
        tool0 = self.chain.fk(q_from)[:3, 3]
        # a path may start in contact (the tool resting on a table) and move away, never deeper than it began
        allowed = {b.name: max(0.0, b.depth(tool0)) + 0.001 for b in self.world.solids()}
        for q in sample:
            pts = self.chain.points(q)
            tool = pts[-1]
            tool_z.append(tool[2])
            for box in self.world.of_kind("keep_out"):
                for p in pts[2:]:
                    if box.contains(p):
                        raise Refused(f"the arm would enter keep-out zone {box.name!r}", "keep_out", box=box.name)
            for box in self.world.solids():
                depth = box.depth(tool)
                if depth > allowed[box.name] and not allow_contact:
                    raise Refused(f"the tool would go {1000 * depth:.0f} mm into {box.name!r}", "surface",
                                  "stop above it, or use a guarded move (touchdown) to make contact", box=box.name)
                if np.isfinite(depth):
                    clearance = min(clearance, -depth)
        if self.m.turn_clearance is not None:
            joints, above = self.m.turn_clearance
            turning = np.abs(np.diff(full[:, list(joints)], axis=0)).max(axis=1) * rate > 1e-3
            if turning.any():
                z = min(float(self.chain.fk(q)[2, 3]) for q in full[1:][turning][:: max(1, int(turning.sum()) // 40)])
                if z < self.z_start + above:
                    raise Refused(f"this turns joints {[j + 1 for j in joints]} with the tool near its start height; "
                                  "that sweeps the gripper across the table", "turn_clearance",
                                  f"lift at least {100 * above:.0f} cm first")
        return dict(seconds=round(len(path) / rate, 2), peak_speed=round(float(v.max()), 3),
                    peak_accel=round(float(a.max()), 2), lowest_tool_z=round(float(min(tool_z)), 4),
                    surface_clearance=None if not np.isfinite(clearance) else round(float(clearance), 4))

    # -- during motion ----------------------------------------------------------------------------
    def watch(self, st: JointState, q_cmd, grip_cmd=None) -> Trip | None:
        """Watchdog on one measurement against what was commanded. Returns the first finding, or None."""
        if st.faults:
            return Trip("fault", "; ".join(st.faults))
        if st.q is None or not np.all(np.isfinite(st.q)):
            return Trip("fault", "no position feedback")
        if st.temp is not None:
            temp = np.nan_to_num(np.asarray(st.temp, float), nan=0.0)
            if temp.max() > self.m.temp_limit_c:
                i = int(temp.argmax())
                return Trip("hot", f"{self._name(i)} is at {temp[i]:.0f} C", i, float(temp[i]), self.m.temp_limit_c)
        err = np.abs(np.asarray(st.q) - np.asarray(q_cmd))
        for i, j in enumerate(self.m.joints):
            if err[i] > j.track_tol:
                return Trip("blocked", f"{j.name} is {np.degrees(err[i]):.1f} deg behind its command: something is in the way",
                            i, float(err[i]), j.track_tol)
        if st.tau is not None:
            tau = np.abs(np.asarray(st.tau, float))
            for i, j in enumerate(self.m.joints):
                if tau[i] > j.tau_max:
                    return Trip("overload", f"{j.name} torque {tau[i]:.1f} Nm; limit {j.tau_max}", i, float(tau[i]), j.tau_max)
        tool = self.chain.fk(st.q)[:3, 3]
        for box in self.world.of_kind("keep_out"):
            if box.contains(tool):
                return Trip("keep_out", f"the tool entered keep-out zone {box.name!r}")
        g = self.m.gripper
        if g is not None and grip_cmd is not None and st.gripper is not None:
            if abs(st.gripper - grip_cmd) > g.track_tol and (st.gripper_tau is None or abs(st.gripper_tau) > 0.5 * g.tau_max):
                return Trip("gripper", f"gripper is {st.gripper - grip_cmd:+.2f} {g.unit} off its command", isolate=True)
            if st.gripper_tau is not None and abs(st.gripper_tau) > g.tau_max:
                return Trip("gripper", f"gripper effort {st.gripper_tau:.1f}; limit {g.tau_max}", isolate=True)
        return None

    def _name(self, i):
        return self.m.joints[i].name if i < self.m.n else "gripper"
