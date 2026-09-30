"""SimBody: a kinematic twin of any manifest, good enough to rehearse plans and to test the kernel.

Joints follow commands with a short lag. Measured torque is the arm's own gravity load plus servo stiffness
times tracking error, so pressing on something looks like it does on a real position-controlled arm. With a model
fitted from the real robot's records (use_fit), the links weigh what it says and the joints have its friction.
Surfaces in the world push the tool back. Objects stop the gripper at their width and ride along once gripped.
Motors heat while they carry load. It is not a physics engine: nothing tips, slides or bounces.
"""
from __future__ import annotations

import numpy as np

from ..body import JointState, Manifest
from ..kinematics import Chain
from ..world import World

THERMAL = dict(heat=0.0027, cool_on=0.001, cool_off=0.0068)   # C/s per Nm^2; 1/s energised; 1/s off


class SimBody:
    simulated = True

    def __init__(self, manifest: Manifest, world: World | None = None, q=None, gripper=None, *, lag_s: float = 0.02,
                 stiffness: float = 150.0, noise: float = 0.0, seed: int = 0, temp_c=None, ambient_c: float = 25.0):
        self.manifest = manifest
        self.chain = Chain(manifest.urdf, manifest.tool_link)
        self.world = world if world is not None else World()
        n = manifest.n
        self.q = np.zeros(n) if q is None else np.asarray(q, float).copy()
        self.q_cmd = self.q.copy()
        g = manifest.gripper
        self.grip = None if g is None else float(g.closed if gripper is None else gripper)
        self.grip_cmd = self.grip
        self.lag, self.K, self.noise = lag_s, stiffness, noise
        self.rng = np.random.default_rng(seed)
        self.ambient = ambient_c
        self.temp = np.full(n, ambient_c) if temp_c is None else np.asarray(temp_c, float).copy()
        self.thermal = {**THERMAL, **manifest.thermal}
        self.enabled, self.t = False, 0.0
        self.dt = 1.0 / manifest.rate_hz
        self.friction = None              # joint velocities -> friction torque, from a fitted model (use_fit)

    def use_fit(self, model):
        """Weigh the links and feel friction as a model fitted from the real robot's records says (fit.py)."""
        model.apply(self.chain)
        self.friction = model.friction_torque

    # -- Body contract ----------------------------------------------------------------------------
    def connect(self) -> JointState:
        return self._state(np.zeros(self.manifest.n), None)

    def enable(self):
        self.enabled = True
        self.q_cmd, self.grip_cmd = self.q.copy(), self.grip

    def command(self, q, dq, gripper, gripper_v=0.0):
        self.q_cmd = np.asarray(q, float).copy()
        if gripper is not None:
            self.grip_cmd = float(gripper)

    def disable(self):
        self.enabled = False

    def close(self):
        pass

    def read(self) -> JointState:
        self.t += self.dt
        if not self.enabled:
            self._cool(np.zeros(self.manifest.n), off=True)
            return self._state(np.zeros(self.manifest.n), None)
        a = min(1.0, self.dt / self.lag) if self.lag > 0 else 1.0
        before = self.q
        self.q = self._push_back(self.q + a * (self.q_cmd - self.q))
        tau = self.chain.gravity(self.q) + self.K * (self.q_cmd - self.q)
        if self.friction is not None:
            tau = tau + self.friction((self.q - before) / self.dt)
        if self.noise:
            tau = tau + self.rng.normal(0, self.noise, len(tau))
        grip_tau = self._gripper(a)
        self.world.carry(self.chain.fk(self.q))
        self._cool(tau, off=False)
        return self._state(tau, grip_tau)

    # -- internals --------------------------------------------------------------------------------
    def _state(self, tau, grip_tau) -> JointState:
        s = self.manifest.sensing
        return JointState(self.t, self.q.copy(), None, tau.copy() if "torque" in s else None,
                          self.temp.copy() if "temperature" in s else None, self.grip,
                          grip_tau if "gripper_effort" in s else None)

    def _contact_points(self, q):
        """Points that can touch a surface: the tool point, and the bottom of whatever is held."""
        T = self.chain.fk(q)
        pts = [T[:3, 3]]
        held = self.world.held
        if held is not None and held[0] in self.world.boxes:
            P = T @ held[1]
            pts.append(P[:3, 3] - P[:3, 2] * self.world.boxes[held[0]].size[2] / 2)
        return pts

    def _push_back(self, q):
        """Surfaces do not let the tool through: move the measured joints back out, along the surface normal."""
        for _ in range(3):
            worst = None
            for p in self._contact_points(q):
                for box in self.world.solids():
                    d = box.depth(p)
                    if d > 1e-5 and (worst is None or d > worst[0]):
                        worst = (d, box.pose[:3, 2], p)
            if worst is None:
                return q
            d, normal, p = worst
            J = self.chain.jacobian(q)
            # the contact point rides on the tool: v = v_tool + w x r, so its Jacobian is J_lin - [r]x J_ang
            r = p - self.chain.fk(q)[:3, 3]
            Jp = J[:3] - np.cross(r, J[3:].T).T
            q = q + np.linalg.lstsq(Jp, d * normal, rcond=None)[0]
        return q

    def _gripper(self, a) -> float | None:
        g, grip, target = self.manifest.gripper, self.grip, self.grip_cmd
        if g is None or grip is None or target is None:
            return None
        closing = np.sign(g.closed - g.open)
        stop = self._object_stop()
        if stop is not None and (target - stop) * closing > 0:          # the object is in the way
            self.grip = grip + a * (stop - grip)
            if self.world.held is None:
                self.world.grab(self.chain.fk(self.q))
            return float(15.0 * (target - self.grip))             # squeezing: effort pushes towards closed
        if self.world.held is not None and stop is not None and (grip - stop) * closing < -0.02:
            self.world.drop()                                             # opened past the object: let go
        self.grip = grip + a * (target - grip)
        return float(self.rng.normal(0, 0.02)) if self.noise else 0.0

    def _object_stop(self) -> float | None:
        """Gripper position where the fingers meet the object between them, if there is one."""
        g = self.manifest.gripper
        held = self.world.held
        box = self.world.boxes.get(held[0]) if held is not None else self.world.object_at(self.chain.fk(self.q)[:3, 3])
        if box is None or g is None or g.m_per_unit is None:
            return None
        return g.position(box.grip_width)

    def _cool(self, tau, off: bool):
        th = self.thermal
        cooling = th["cool_off"] if off else th["cool_on"]
        rate = th["heat"] * np.asarray(tau) ** 2 - cooling * (self.temp - self.ambient)
        self.temp = self.temp + self.dt * rate
