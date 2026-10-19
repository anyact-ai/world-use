"""Panda joint I/O; the kernel owns IK, trajectories, limits and power transitions."""
from __future__ import annotations

import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco as mj
import numpy as np

from world_use.bodies.mujoco_render import CameraRenderer
from world_use.body import JointSpec, JointState, Manifest, ToolBox
from world_use.kinematics import Chain


def numbers(values):
    return " ".join(f"{x:.12g}" for x in values)


def export_chain(model, path):
    root = ET.Element("robot", name="panda_pusher")
    for i in range(model.body("robot0_base").id, model.body("push_tool").id+1):
        name = model.body(i).name
        link = ET.SubElement(root, "link", name=name)
        if model.body_mass[i] > 0:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass", value=str(model.body_mass[i]))
            rot = np.empty(9)
            mj.mju_quat2Mat(rot, model.body_iquat[i])
            rot = rot.reshape(3, 3)
            tensor = rot @ np.diag(model.body_inertia[i]) @ rot.T
            ET.SubElement(inertial, "origin", xyz=numbers(model.body_ipos[i]))
            ET.SubElement(inertial, "inertia", **{
                key: str(tensor[a, b]) for key, a, b in
                (("ixx", 0, 0), ("iyy", 1, 1), ("izz", 2, 2),
                 ("ixy", 0, 1), ("ixz", 0, 2), ("iyz", 1, 2))})
        if model.body_parentid[i] == 0:
            continue
        j = model.body_jntadr[i] if model.body_jntnum[i] else -1
        node = ET.SubElement(root, "joint", name=model.joint(j).name if j >= 0 else "fixed_" + name,
                             type="revolute" if j >= 0 else "fixed")
        ET.SubElement(node, "parent", link=model.body(model.body_parentid[i]).name)
        ET.SubElement(node, "child", link=name)
        rot = np.empty(9)
        mj.mju_quat2Mat(rot, model.body_quat[i])
        rot = rot.reshape(3, 3)
        angles = [np.arctan2(rot[2, 1], rot[2, 2]),
                  np.arctan2(-rot[2, 0], np.hypot(rot[0, 0], rot[1, 0])),
                  np.arctan2(rot[1, 0], rot[0, 0])]
        ET.SubElement(node, "origin", xyz=numbers(model.body_pos[i]), rpy=numbers(angles))
        if j >= 0:
            ET.SubElement(node, "axis", xyz=numbers(model.jnt_axis[j]))
            ET.SubElement(node, "limit", lower=str(model.jnt_range[j, 0]),
                          upper=str(model.jnt_range[j, 1]), effort="80", velocity="1")
    ET.indent(root)
    path.write_text(ET.tostring(root, encoding="unicode"))


class PandaPusher:
    simulated = True

    def __init__(self, scene, output, *, model_file="scene.xml"):
        scene, output = Path(scene), Path(output)
        path = (scene / model_file).resolve()
        if (Path(model_file).is_absolute() or ".." in Path(model_file).parts
                or not path.is_relative_to(scene.resolve()) or path.suffix not in (".xml", ".mjb")):
            raise ValueError("select an XML or compiled MJB model inside the scene directory")
        self.model = m = (mj.MjModel.from_binary_path(str(path)) if path.suffix == ".mjb"
                          else mj.MjModel.from_xml_path(str(path)))
        self.data = d = mj.MjData(m)
        state = np.load(scene / "initial.npz")
        d.qpos[:], d.qvel[:] = state["qpos"], state["qvel"]
        self.q_cmd = d.qpos[:7].copy()
        self.dq_cmd = np.zeros(7)
        self.gain, self.bias = m.actuator_gainprm.copy(), m.actuator_biasprm.copy()
        self.renderer = CameraRenderer()
        self.enabled = False
        self.lock = threading.RLock()
        self.max_penetration = 0.
        self.worst_contact = None
        self.contact_counts = {}
        self.unexpected_contacts = {}
        urdf = output / "panda.urdf"
        export_chain(m, urdf)
        self.chain = Chain(urdf, "push_tool")
        self.manifest = Manifest(
            "Panda with a 8 mm pushing pin (simulation)", urdf.resolve(), "push_tool",
            tuple(JointSpec(f"robot0_joint{i+1}", *m.jnt_range[i], v_max=1., track_tol=.08)
                  for i in range(7)), speed=.6, max_segment_m=.5,
            tool_boxes=(ToolBox("pushing pin", (0, 0, -.035), (.008, .008, .070),
                                "scene.py: declared capsule endpoints and 4 mm radius"),),
            notes=("Rigid letters with free joints. Move the pin through contact; no grasping.",
                   "Tool is the pin's bottom centre; the starting orientation points down.",
                   "No joint torque sensing; rehearsal does not reproduce this external scene."))
        self.disable()
        mj.mj_forward(m, d)
        if np.max(np.abs(self.chain.fk(d.qpos[:7]) - self.tool_pose())) > 1e-7:
            raise ValueError("URDF and simulated tool kinematics disagree")

    def tool_pose(self):
        body = self.data.body("push_tool")
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = body.xmat.reshape(3, 3), body.xpos
        return pose

    def state(self):
        return JointState(time.monotonic(), self.data.qpos[:7].copy(), self.data.qvel[:7].copy())

    def connect(self):
        return self.state()

    def enable(self):
        with self.lock:
            self.q_cmd = self.data.qpos[:7].copy()
            self.dq_cmd[:] = 0
            self.model.dof_frictionloss[:7] = 0
            self.model.actuator_gainprm[:] = self.gain
            self.model.actuator_biasprm[:] = self.bias
            self.enabled = True

    def command(self, q, dq, gripper, gripper_v=0):
        with self.lock:
            self.q_cmd, self.dq_cmd = np.array(q), np.array(dq)

    def read(self):
        with self.lock:
            m, d = self.model, self.data
            for _ in range(10):
                d.ctrl[:] = self.q_cmd + (60/900)*self.dq_cmd if self.enabled else 0
                mj.mj_step(m, d)
                for contact in d.contact:
                    names = [m.geom(int(g)).name or "" for g in (contact.geom1, contact.geom2)]
                    if -float(contact.dist) > self.max_penetration:
                        self.max_penetration = -float(contact.dist)
                        self.worst_contact = dict(time_s=d.time, geoms=names, penetration_m=self.max_penetration)
                    if "pusher" in names:
                        other = names[1] if names[0] == "pusher" else names[0]
                        self.contact_counts[other] = self.contact_counts.get(other, 0) + 1
                    elif any(name.startswith("letter_") for name in names) and any(
                            name.startswith("robot0_") for name in names):
                        pair = "/".join(names)
                        self.unexpected_contacts[pair] = self.unexpected_contacts.get(pair, 0) + 1
            return self.state()

    def disable(self):
        with self.lock:
            self.enabled = False
            self.data.ctrl[:] = 0
            self.model.actuator_gainprm[:] = 0
            self.model.actuator_biasprm[:] = 0
            self.model.dof_frictionloss[:7] = 1000

    def capture(self, view):
        with self.lock:
            copy = mj.MjData(self.model)
            mj.mj_copyData(copy, self.model, self.data)
            tool, stamp = self.tool_pose(), time.monotonic()
        image, depth = self.renderer.render(self.model, copy, view, depth=True)
        return image, depth, tool, stamp

    def render(self, view):
        return self.capture(view)[0]

    def close(self):
        self.renderer.close()
