"""MuJoCo rigid-body simulation, including actuators, gravity and frictional grasps.

The world here is simulation truth. The kernel maintains a separate estimated world.
Temperature is an explicit approximate motor heating model, not a MuJoCo measurement.
"""
from __future__ import annotations

import threading
from typing import Any

import numpy as np

from ..body import JointState, Manifest
from ..kinematics import Chain
from ..world import World
from .mujoco_scene import build, mj, quat

THERMAL = dict(heat=0.0027, cool_on=0.001, cool_off=0.0068)


class SimBody:
    simulated = True

    def __init__(self, manifest: Manifest, world: World | None = None, q=None, gripper=None, *,
                 noise: float = 0.0, seed: int = 0, temp_c=None, ambient_c: float = 25.0):
        self.manifest = manifest
        self.chain = Chain(manifest.urdf, manifest.tool_link)
        self.world = world if world is not None else World()
        n = manifest.n
        initial = manifest.rest.q if manifest.rest else np.clip(np.zeros(n), manifest.lower, manifest.upper)
        self.q = np.asarray(initial if q is None else q, float).copy()
        if self.q.shape != (n,) or not np.isfinite(self.q).all():
            raise ValueError(f"simulation q: expected {n} finite joint positions")
        self.q_cmd, self.dq_cmd = self.q.copy(), np.zeros(n)
        g = manifest.gripper
        self.grip = None if g is None else float(g.closed if gripper is None else gripper)
        self.grip_cmd = self.grip
        self.noise, self.rng = noise, np.random.default_rng(seed)
        self.ambient = ambient_c
        self.temp = np.broadcast_to(ambient_c if temp_c is None else temp_c, (n,)).astype(float).copy()
        if not np.isfinite(self.temp).all() or not np.isfinite(ambient_c):
            raise ValueError("simulation temperature: expected finite values")
        if not np.isfinite(noise) or noise < 0:
            raise ValueError("simulation noise must be finite and nonnegative")
        self.thermal = {**THERMAL, **manifest.thermal}
        self.enabled, self.t = False, 0.0
        self.dt = 1.0 / manifest.rate_hz
        self.substeps = max(1, int(np.ceil(self.dt / .002)))
        self.lock = threading.RLock()
        self.model: Any = None
        self.data: Any = None
        self._scene = None
        self._fit = None
        self._renderer = None
        self._closed = False

    def _aperture(self, value) -> float:
        g = self.manifest.gripper
        aperture = None if g is None else g.aperture(value)
        if aperture is None or not np.isfinite(aperture):
            raise ValueError("MuJoCo grippers need a calibrated aperture")
        return aperture

    def _signature(self):
        return tuple((name, box.kind, tuple(box.size), tuple(sorted(box.params.items())))
                     for name, box in self.world.boxes.items() if box.kind in ("object", "surface"))

    def _ensure_scene(self):
        if self._closed:
            raise RuntimeError("simulation is closed")
        scene = self._signature()
        if self.model is not None and scene == self._scene:
            return
        self.model, fingers = build(self.manifest, self.world, self.dt / self.substeps)
        self.data = mj.MjData(self.model)
        joints = [self.model.joint(j.name) for j in self.manifest.joints]
        self._q = np.array([int(j.qposadr[0]) for j in joints])
        self._v = np.array([int(j.dofadr[0]) for j in joints])
        self._fingers = [self.model.joint(name) for name in fingers]
        self._finger_bodies = {int(j.bodyid[0]) for j in self._fingers}
        self._objects = {name: self.model.body(f"box/{name}").id for name, b in self.world.boxes.items()
                         if b.kind == "object"}
        self.data.qpos[self._q] = self.q
        if self._fingers:
            aperture = self._aperture(self.grip)
            for joint in self._fingers:
                self.data.qpos[joint.qposadr] = aperture / 2
        self._gains = self.model.actuator_gainprm.copy(), self.model.actuator_biasprm.copy()
        self._scene = scene
        if self._fit is not None:
            self._apply_fit()
        mj.mj_forward(self.model, self.data)

    def use_fit(self, model):
        with self.lock:
            model.apply(self.chain)
            self._fit = model
            if self.model is not None:
                self._apply_fit()

    def _apply_fit(self):
        assert self._fit is not None
        for name, (mass, com) in self._fit.links.items():
            body = self.model.body(name)
            ratio = mass / body.mass[0]
            body.mass[:] = mass
            body.ipos[:] = com
            body.inertia[:] *= ratio
        kind = mj.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mj.mj_stateSize(self.model, kind))
        mj.mj_getState(self.model, self.data, state, kind)
        mj.mj_setConst(self.model, self.data)
        mj.mj_setState(self.model, self.data, state, kind)
        mj.mj_forward(self.model, self.data)

    def connect(self) -> JointState:
        return self._state(np.zeros(self.manifest.n), None, np.zeros(self.manifest.n))

    def enable(self):
        with self.lock:
            self._ensure_scene()
            self.enabled = True
            self.q_cmd, self.grip_cmd = self.q.copy(), self.grip
            self.dq_cmd[:] = 0

    def command(self, q, dq, gripper, gripper_v=0.0):
        with self.lock:
            self.q_cmd, self.dq_cmd = np.asarray(q, float).copy(), np.asarray(dq, float).copy()
            if gripper is not None:
                self.grip_cmd = float(gripper)

    def disable(self):
        with self.lock:
            self.enabled = False
            if self.model is not None:
                self.model.actuator_gainprm[:] = 0
                self.model.actuator_biasprm[:] = 0
                self.data.ctrl[:] = 0
                self.data.qfrc_applied[:] = 0

    def close(self):
        with self.lock:
            renderer, self._renderer = self._renderer, None
            self.model = self.data = None
            self._fingers = []
            self._closed = True
        if renderer is not None:
            renderer.close()

    def read(self) -> JointState:
        with self.lock:
            self._ensure_scene()
            m, d = self.model, self.data
            if self.enabled:
                m.actuator_gainprm[:], m.actuator_biasprm[:] = self._gains
            else:
                m.actuator_gainprm[:] = 0
                m.actuator_biasprm[:] = 0
            for _ in range(self.substeps):
                d.qfrc_applied[:] = 0
                if self.enabled:
                    # Gravity/velocity feed-forward, with actuator feedback integrated implicitly by MuJoCo.
                    d.ctrl[:self.manifest.n] = self.q_cmd + (d.qfrc_bias[self._v] + 10 * self.dq_cmd) / 150
                    if self._fingers:
                        d.ctrl[self.manifest.n:] = self._aperture(self.grip_cmd) / 2
                    if self._fit is not None:
                        d.qfrc_applied[self._v] = -self._fit.friction_torque(d.qvel[self._v])
                mj.mj_step(m, d)
            mj.mj_forward(m, d)
            self.t += self.dt
            self.q = d.qpos[self._q].copy()
            tau = d.qfrc_actuator[self._v].copy()
            if self.noise:
                tau += self.rng.normal(0, self.noise, len(tau))
            grip_tau = None
            g = self.manifest.gripper
            if self._fingers and g is not None and g.m_per_unit is not None:
                aperture = sum(float(d.qpos[j.qposadr[0]]) for j in self._fingers)
                self.grip = g.position(aperture)
                grip_tau = float(np.mean(d.actuator_force[self.manifest.n:]) * g.m_per_unit)
            self._update_world()
            self._cool(tau, off=not self.enabled)
            return self._state(tau, grip_tau, d.qvel[self._v].copy())

    def reset(self, q, gripper=None):
        """Explicitly reset a simulation or pose a recorded scene, without advancing physics."""
        with self.lock:
            self.q = np.asarray(q, float).copy()
            if self.q.shape != (self.manifest.n,) or not np.isfinite(self.q).all():
                raise ValueError("invalid simulation reset joint positions")
            self.grip = gripper
            self._ensure_scene()
            mj.mj_resetData(self.model, self.data)
            self.data.qpos[self._q] = self.q
            if self._fingers:
                for joint in self._fingers:
                    self.data.qpos[joint.qposadr] = self._aperture(gripper) / 2
            for name, box in self.world.boxes.items():
                if box.kind == "object":
                    j = self.model.joint(f"free/{name}")
                    i = int(j.qposadr[0])
                    self.data.qpos[i:i + 3] = box.pose[:3, 3]
                    self.data.qpos[i + 3:i + 7] = quat(box.pose[:3, :3])
                elif box.kind == "surface":
                    body = self.model.body(f"box/{name}")
                    body.pos[:] = box.pose[:3, 3]
                    body.quat[:] = quat(box.pose[:3, :3])
            self.q_cmd, self.grip_cmd = self.q.copy(), gripper
            self.dq_cmd[:] = 0
            mj.mj_forward(self.model, self.data)

    def render(self, view):
        return self._capture(view, depth=False)[0]

    def capture(self, view, *, feedback=False):
        """RGB-D and tool pose from one physics snapshot, without advancing the scene."""
        (rgb, depth), tool, timestamp, aperture = self._capture(view, depth=True)
        result = rgb, depth, tool, timestamp
        return (*result, aperture) if feedback else result

    def _capture(self, view, *, depth):
        import time

        from .mujoco_render import CameraRenderer
        with self.lock:
            self._ensure_scene()
            model = self.model
            data = mj.MjData(model)
            mj.mj_copyData(data, model, self.data)
            timestamp = time.monotonic()
            tool = self.chain.fk(self.q).copy()
            aperture = 1000 * self._aperture(self.grip) if self.manifest.gripper is not None else None
            if self._renderer is None:
                self._renderer = CameraRenderer()
            renderer = self._renderer
        return renderer.render(model, data, view, depth=depth), tool, timestamp, aperture

    def _update_world(self):
        d, m = self.data, self.model
        for name, index in self._objects.items():
            box = self.world.boxes[name]
            box.pose[:3, 3] = d.xpos[index]
            box.pose[:3, :3] = d.xmat[index].reshape(3, 3)
        # A truth label only: it never constrains an object or moves its pose.
        self.world.held = None
        for name, index in self._objects.items():
            touching = set()
            for contact in d.contact:
                a, b = m.geom_bodyid[contact.geom1], m.geom_bodyid[contact.geom2]
                if a == index and b in self._finger_bodies:
                    touching.add(b)
                if b == index and a in self._finger_bodies:
                    touching.add(a)
            if len(touching) == 2:
                self.world.held = (name, np.linalg.inv(self.chain.fk(self.q)) @ self.world.boxes[name].pose)
                break

    def _state(self, tau, grip_tau, dq) -> JointState:
        s = self.manifest.sensing
        return JointState(self.t, self.q.copy(), dq, tau.copy() if "torque" in s else None,
                          self.temp.copy() if "temperature" in s else None, self.grip,
                          grip_tau if "gripper_effort" in s else None)

    def _cool(self, tau, off: bool):
        th = self.thermal
        cooling = th["cool_off"] if off else th["cool_on"]
        self.temp += self.dt * (th["heat"] * np.asarray(tau) ** 2 - cooling * (self.temp - self.ambient))
