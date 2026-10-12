"""world-use joint I/O for the exported RoboCasa Panda scene.

The mobile base is parked. MuJoCo simulates the original meshes and contacts;
the adapter adds joint servos, a brake and calibrated RGB-D capture.
"""
from __future__ import annotations

import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import mujoco as mj
import numpy as np

from world_use.bodies.mujoco_render import CameraRenderer
from world_use.body import GripperSpec, JointSpec, JointState, Manifest
from world_use.cameras import View
from world_use.kinematics import Chain

SCENE = Path('runs/robocasa-small-cup/scene')


def numbers(values):
    return ' '.join(f'{x:.12g}' for x in values)


def euler(q):
    r = np.empty(9)
    mj.mju_quat2Mat(r, q)
    r = r.reshape(3, 3)
    return [np.arctan2(r[2, 1], r[2, 2]), np.arctan2(-r[2, 0], np.hypot(r[0, 0], r[1, 0])),
            np.arctan2(r[1, 0], r[0, 0])]


def export_urdf(m, path):
    root = ET.Element('robot', name='robocasa_panda')
    ids = range(m.body('robot0_link0').id, m.body('gripper0_right_finger_joint2_tip').id + 1)
    for i in ids:
        name = m.body(i).name
        link = ET.SubElement(root, 'link', name=name)
        if m.body_mass[i] > 0:
            inertial = ET.SubElement(link, 'inertial')
            q = m.body_iquat[i]
            rpy = euler(q)
            ET.SubElement(inertial, 'origin', xyz=numbers(m.body_ipos[i]), rpy=numbers(rpy))
            ET.SubElement(inertial, 'mass', value=str(m.body_mass[i]))
            a, b, c = m.body_inertia[i]
            ET.SubElement(inertial, 'inertia', ixx=str(a), iyy=str(b), izz=str(c), ixy='0', ixz='0', iyz='0')
        if i == ids.start:
            continue
        ji = m.body_jntadr[i] if m.body_jntnum[i] else -1
        kind = 'fixed' if ji < 0 else ('revolute' if m.jnt_type[ji] == mj.mjtJoint.mjJNT_HINGE else 'prismatic')
        joint = ET.SubElement(root, 'joint', name=f'fixed_{name}' if ji < 0 else m.joint(ji).name, type=kind)
        ET.SubElement(joint, 'parent', link=m.body(m.body_parentid[i]).name)
        ET.SubElement(joint, 'child', link=name)
        q = m.body_quat[i]
        rpy = euler(q)
        ET.SubElement(joint, 'origin', xyz=numbers(m.body_pos[i]), rpy=numbers(rpy))
        if ji >= 0:
            assert np.allclose(m.jnt_pos[ji], 0)
            sign = -1 if m.jnt_range[ji, 1] == 0 else 1
            ET.SubElement(joint, 'axis', xyz=numbers(m.jnt_axis[ji] * sign))
            lo, hi = sorted(m.jnt_range[ji] * sign)
            ET.SubElement(joint, 'limit', lower=str(lo), upper=str(hi), effort='80', velocity='1')
    site = m.site('gripper0_right_grip_site')
    ET.SubElement(root, 'link', name='tool')
    joint = ET.SubElement(root, 'joint', name='tool_fixed', type='fixed')
    ET.SubElement(joint, 'parent', link=m.body(site.bodyid[0]).name)
    ET.SubElement(joint, 'child', link='tool')
    rpy = euler(site.quat)
    ET.SubElement(joint, 'origin', xyz=numbers(site.pos), rpy=numbers(rpy))
    ET.indent(root)
    path.write_text(ET.tostring(root, encoding='unicode'))


def warm_lighting(m):
    m.vis.headlight.ambient[:] = [.20, .19, .18]
    m.vis.headlight.diffuse[:] = [.30, .30, .31]
    m.light_diffuse[:] = [.65, .55, .44]
    m.light_ambient[:] = [.04, .025, .012]
    m.light_specular[:] = [.14, .085, .04]
    m.light_pos[0] = [-.5, -2.5, 3.1]
    m.light_dir[0] = [.5, .7, -.6]
    m.light_type[0] = mj.mjtLightType.mjLIGHT_DIRECTIONAL
    m.vis.quality.shadowsize = 4096
    m.vis.quality.offsamples = 4
    m.vis.global_.offwidth = 1600
    m.vis.global_.offheight = 1000
    m.geom_group[m.geom_group == 0] = 3
    m.site_rgba[:, 3] = 0


class PandaBody:
    simulated = True

    def __init__(self, output, scene=SCENE, *, park_left_m=.20):
        self.model = m = mj.MjModel.from_xml_path(str(scene / 'scene.xml'))
        self.data = d = mj.MjData(m)
        state = np.load(scene / 'state.npz')
        d.qpos[:] = state['qpos']
        # Park 20 cm toward this task's left-hand counter before the session starts.
        # This is a setup pose, not a base motion commanded by world-use.
        d.qpos[1] += park_left_m
        d.qvel[:] = 0
        m.opt.timestep = .002
        m.opt.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
        warm_lighting(m)
        # Park the upstream mobile base with passive brakes, leaving the arm articulated.
        m.dof_frictionloss[:4] = 100000
        m.dof_frictionloss[4:11] = 1000
        self.gripper_friction = m.dof_frictionloss[11:13].copy()
        self.gripper_gain = m.actuator_gainprm[11:13].copy()
        self.gripper_bias = m.actuator_biasprm[11:13].copy()
        m.dof_frictionloss[11:13] = 1000
        m.actuator_gainprm[11:13] = 0
        m.actuator_biasprm[11:13] = 0
        m.dof_solref[:13] = [.004, 1]
        m.dof_solimp[:13] = [.9999, .9999, .001, .5, 2]
        m.dof_damping[11:13] = 5
        mj.mj_forward(m, d)
        self.base = np.eye(4)
        self.base[:3, :3] = d.body('robot0_link0').xmat.reshape(3, 3)
        self.base[:3, 3] = d.body('robot0_link0').xpos
        urdf = Path(output) / 'panda.urdf'
        export_urdf(m, urdf)
        joints = tuple(JointSpec(f'robot0_joint{i+1}', *m.jnt_range[i+4], v_max=1.0, track_tol=.12)
                       for i in range(7))
        self.manifest = Manifest('RoboCasa Franka Panda', urdf.resolve(), 'tool', joints,
            speed=.5, max_segment_m=.4, gripper=GripperSpec(0, .08, (0, 0, 1), (1, 0, 0),
            unit='m', m_per_unit=1, v_max=.12, track_tol=.015, tau_max=30, squeeze=.006),
            sensing=frozenset({'position', 'gripper_effort'}), notes=(
                'Simulated Franka Panda in RoboCasa; mobile base parked.',
                'Use RGB-D observations. Kitchen geometry is not in the coarse rehearsal twin.',
                'Use rehearse=false; full-scene contact rehearsal is not implemented.'))
        self.chain = Chain(urdf, 'tool')
        self.lock = threading.RLock()
        self.renderer = CameraRenderer()
        self.enabled = False
        self.q_cmd = d.qpos[4:11].copy()
        self.dq_cmd = np.zeros(7)
        self.grip_cmd = float(d.qpos[11] - d.qpos[12])
        self.kp = np.array([450, 450, 400, 400, 250, 150, 100])
        self.kd = np.array([35, 35, 30, 30, 20, 12, 8])
        error = np.max(np.abs(self.chain.fk(self.q_cmd) - self.tool_pose()))
        if error > 1e-7:
            raise ValueError(f'Exported kinematics disagree with MuJoCo: {error}')

    def tool_pose(self):
        site = self.data.site('gripper0_right_grip_site')
        pose = np.eye(4)
        pose[:3, :3] = site.xmat.reshape(3, 3)
        pose[:3, 3] = site.xpos
        return np.linalg.inv(self.base) @ pose

    def state(self):
        d = self.data
        return JointState(time.monotonic(), d.qpos[4:11].copy(), d.qvel[4:11].copy(),
                          gripper=float(d.qpos[11]-d.qpos[12]),
                          gripper_tau=float(np.mean(np.abs(d.actuator_force[11:13]))))

    def connect(self):
        return self.state()

    def enable(self):
        with self.lock:
            self.q_cmd = self.data.qpos[4:11].copy()
            self.dq_cmd[:] = 0
            self.grip_cmd = float(self.data.qpos[11]-self.data.qpos[12])
            self.model.dof_frictionloss[4:11] = 0
            self.model.dof_frictionloss[11:13] = self.gripper_friction
            self.model.actuator_gainprm[11:13] = self.gripper_gain
            self.model.actuator_biasprm[11:13] = self.gripper_bias
            self.enabled = True

    def command(self, q, dq, gripper, gripper_v=0):
        with self.lock:
            self.q_cmd, self.dq_cmd = np.array(q), np.array(dq)
            if gripper is not None:
                self.grip_cmd = gripper

    def read(self):
        with self.lock:
            m, d = self.model, self.data
            for _ in range(5):
                d.ctrl[:] = 0
                if self.enabled:
                    torque = self.kp*(self.q_cmd-d.qpos[4:11]) + self.kd*(self.dq_cmd-d.qvel[4:11]) + d.qfrc_bias[4:11]
                    d.ctrl[:7] = np.clip(torque, m.actuator_ctrlrange[:7, 0], m.actuator_ctrlrange[:7, 1])
                    d.ctrl[11:13] = [self.grip_cmd/2, -self.grip_cmd/2]
                mj.mj_step(m, d)
            return self.state()

    def disable(self):
        with self.lock:
            self.enabled = False
            self.data.ctrl[:] = 0
            self.model.dof_frictionloss[4:13] = 1000
            self.model.actuator_gainprm[11:13] = 0
            self.model.actuator_biasprm[11:13] = 0

    def close(self):
        self.renderer.close()

    def lens(self, name='robot0_agentview_center'):
        cam = self.data.camera(name)
        pose = np.eye(4)
        pose[:3, :3] = cam.xmat.reshape(3, 3) @ np.diag([1, -1, -1])
        pose[:3, 3] = cam.xpos
        f = 768/2/np.tan(np.deg2rad(self.model.camera(name).fovy[0])/2)
        return View(np.linalg.inv(self.base) @ pose, f, f, 512, 384, 1024, 768)

    def capture(self, view):
        with self.lock:
            d = mj.MjData(self.model)
            mj.mj_copyData(d, self.model, self.data)
            timestamp, tool = time.monotonic(), self.tool_pose()
        image, depth = self.renderer.render(self.model, d, replace(view, T=self.base@view.T), depth=True)
        return image, depth, tool, timestamp

    def render(self, view):
        return self.capture(view)[0]
