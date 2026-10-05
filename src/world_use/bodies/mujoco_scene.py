"""Build a MuJoCo scene from the robot URDF and the simulation's physical world."""
from __future__ import annotations

import logging
import os
import sys
import xml.etree.ElementTree as ET
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from ..robot_assets import bundled, resolve
from . import manifests

# Respect an explicitly selected backend; EGL also supports Mesa on headless Linux.
if sys.platform == "linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

# MuJoCo exposes its API dynamically from binary extensions without public type stubs.
mj: Any = import_module("mujoco")
# Otherwise MuJoCo appends its warnings to MUJOCO_LOG.TXT in whatever folder the process runs in.
mj.set_mju_user_warning(logging.getLogger("mujoco").warning)

KP, KV = 150.0, 10.0      # arm servos, Nm/rad and Nm s/rad; SimBody feeds gravity forward through the same gains
FINGER = 3000.0, 20.0, 60.0   # each gripper finger's servo: kp N/m, kv N s/m, force limit N
GEARBOX = {mj.mjtJoint.mjJNT_HINGE: 0.3, mj.mjtJoint.mjJNT_SLIDE: 10.0}   # Nm, N: unpowered geared motors hold this
BRAKE = 1000.0            # Nm: brakes hold any arm load


def quat(rotation):
    out = np.empty(4)
    mj.mju_mat2Quat(out, np.asarray(rotation, float).ravel())
    return out


def robot_spec(manifest):
    root = ET.parse(manifest.urdf).getroot()
    extension = root.find("mujoco")
    if extension is None:
        extension = ET.SubElement(root, "mujoco")
    compiler = extension.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(extension, "compiler")
    compiler.attrib.update(discardvisual="false", fusestatic="false", strippath="false", balanceinertia="true")
    for mesh in root.findall(".//mesh"):
        path = resolve(Path(manifest.urdf), mesh.attrib["filename"])
        if not path.is_file():
            raise ValueError(f"simulation mesh does not exist: {path}")
        mesh.set("filename", str(path))
    spec = mj.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    spec.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mj.mjtCone.mjCONE_ELLIPTIC
    spec.option.iterations = 100
    spec.visual.global_.offwidth = 2048
    spec.visual.global_.offheight = 2048
    spec.visual.quality.shadowsize = 2048
    for body in spec.bodies:
        if body.parent is not None and body.parent.name != "world":
            spec.add_exclude(bodyname1=body.parent.name, bodyname2=body.name)
    for geom in spec.geoms:
        if geom.contype:
            geom.group = 3                       # collisions can be inspected in the native viewer
            geom.friction = [0.8, 0.005, 0.0001]
            geom.condim = 4
    rebot = manifests()["rebot"].urdf
    if bundled(Path(manifest.urdf)) == rebot:       # Seeed's model: component hulls and upstream finger segments
        for name in ("link2", "link3", "link4", "link5", "gripper_end"):
            body = spec.body(name)
            for geom in list(body.geoms):
                if geom.contype:
                    spec.delete(geom)
                else:
                    body.add_geom(type=mj.mjtGeom.mjGEOM_MESH, meshname=geom.meshname,
                                  pos=geom.pos, quat=geom.quat, group=3, condim=4,
                                  friction=[0.8, 0.005, 0.0001])
        _rebot_fingers(spec, rebot)
    return spec


def _rebot_fingers(spec, urdf):
    """Upstream convex finger segments are expressed in the tool frame, with opposite left/right names."""
    for link, side in (("gripper_left", "right"), ("gripper_right", "left")):
        body = spec.body(link)
        R = np.empty(9)
        mj.mju_quat2Mat(R, body.quat)
        R = R.reshape(3, 3)
        for geom in list(body.geoms):
            if geom.contype:
                spec.delete(geom)
        for part in ("front", "mid", "rear"):
            name = f"{side}_finger_{part}"
            spec.add_mesh(name=name, file=str(resolve(urdf, f"../meshes/mujoco_collision/{name}.stl")))
            body.add_geom(name=f"{link}_{part}", type=mj.mjtGeom.mjGEOM_MESH, meshname=name,
                          pos=-R.T @ body.pos, quat=quat(R.T), group=3,
                          friction=[1.0, 0.005, 0.0001], condim=4)


def build(manifest, world, timestep):
    """The scene and the gripper joints SimBody drives: two opposed prismatic fingers, one jaw joint, or none.

    Joint friction is what holds the robot while its torque is off: gearbox friction, or brakes on an arm without a
    rest pose. SimBody applies it only then; powered, the servos hold."""
    spec = robot_spec(manifest)
    spec.option.timestep = timestep
    arm = [joint.name for joint in manifest.joints]
    for name in arm:
        joint = spec.joint(name)
        joint.armature = 0.01
        joint.damping[0] = 0.05
        joint.frictionloss = GEARBOX[joint.type] if manifest.rest is not None else BRAKE
        actuator = spec.add_actuator(name=f"servo/{name}", target=name, trntype=mj.mjtTrn.mjTRN_JOINT)
        actuator.set_to_position(kp=KP, kv=KV)
    grip = [j for j in spec.joints if j.name not in arm]
    g = manifest.gripper
    fingers = len(grip) == 2 and all(j.type == mj.mjtJoint.mjJNT_SLIDE for j in grip)
    if g is None:
        # Nothing drives them: their links stay where the URDF's zero puts them.
        removed = {j.name for j in grip}
        for equality in list(spec.equalities):
            if equality.type == mj.mjtEq.mjEQ_JOINT and {equality.name1, equality.name2} & removed:
                spec.delete(equality)
        for joint in grip:
            spec.delete(joint)
        grip = []
    elif len(grip) != 1 and not fingers:
        raise ValueError("MuJoCo grippers need one driven joint, or two opposed prismatic fingers, outside the arm "
                         f"chain; the URDF has {[j.name for j in grip]}")
    elif fingers and g.m_per_unit is None:
        raise ValueError("a gripper with two fingers needs m_per_unit: each finger travels half the opening")
    for joint in grip:
        joint.frictionloss = GEARBOX[joint.type]
        lever = 1.0                               # m per joint unit: fingers and sliding jaws get the finger servo
        if joint.type == mj.mjtJoint.mjJNT_HINGE:
            lever = 0.05                          # a hinged jaw gets it 5 cm out,
            joint.armature = 0.04 * lever ** 2    # with a servo's inertia there, which keeps a light jaw stable
        kp, kv, limit = FINGER[0] * lever ** 2, FINGER[1] * lever ** 2, FINGER[2] * lever
        actuator = spec.add_actuator(name=f"servo/{joint.name}", target=joint.name, trntype=mj.mjtTrn.mjTRN_JOINT)
        actuator.set_to_position(kp=kp, kv=kv)
        actuator.forcelimited = True
        actuator.forcerange = [-limit, limit]
    for joint in spec.joints:
        joint.solref_friction = [2 * timestep, 1]          # stiff, so a held joint does not creep
        joint.solimp_friction = [0.9999, 0.9999, 0.001, 0.5, 2]
    for box in world.boxes.values():
        if box.kind not in ("object", "surface"):
            continue
        body = spec.worldbody.add_body(name=f"box/{box.name}", pos=box.pose[:3, 3], quat=quat(box.pose[:3, :3]))
        if box.kind == "object":
            body.add_freejoint(name=f"free/{box.name}")
        mass = float(box.params.get("mass_kg", 0.05))
        friction = float(box.params.get("friction", 0.8))
        if not np.isfinite(mass) or mass <= 0 or not np.isfinite(friction) or friction < 0:
            raise ValueError(f"simulation box {box.name!r}: mass_kg must be positive and friction nonnegative")
        body.add_geom(name=f"box/{box.name}", type=mj.mjtGeom.mjGEOM_BOX, size=box.size / 2,
                      mass=mass, friction=[friction, 0.005, 0.0001], condim=4, group=2,
                      rgba=[0.90, 0.40, 0.12, 1] if box.kind == "object" else [0.62, 0.68, 0.72, 1])
    spec.worldbody.add_geom(name="floor", type=mj.mjtGeom.mjGEOM_PLANE, size=[2, 2, .05],
                            rgba=[0.19, 0.23, 0.28, 1], group=2)
    spec.worldbody.add_light(pos=[0.5, -0.5, 1.8], dir=[-0.2, 0.2, -1], diffuse=[0.8, 0.8, 0.8])
    spec.worldbody.add_light(pos=[-0.5, 0.8, 1.2], dir=[0.4, -0.4, -1], diffuse=[0.4, 0.4, 0.4], castshadow=False)
    model = spec.compile()
    if fingers:
        data = mj.MjData(model)
        mj.mj_kinematics(model, data)
        axis = data.body(manifest.tool_link).xmat.reshape(3, 3) @ np.asarray(g.opens_along)
        axes = [data.xaxis[model.joint(j.name).id] for j in grip]
        opening = g.aperture(g.open)
        if (opening is None or opening <= 0 or not np.isclose(axes[0] @ axes[1], -1, atol=1e-5)
                or not np.isclose(abs(axes[0] @ axis), 1, atol=1e-5)
                or any(not np.isclose(j.range[0], 0) or j.range[1] < opening / 2 for j in grip)):
            raise ValueError("MuJoCo gripper joints must oppose along opens_along, start at zero, "
                             "and each travel half the calibrated aperture")
    elif grip:
        joint = model.joint(grip[0].name)
        lo, hi = joint.range if joint.limited[0] else (-np.inf, np.inf)
        if not (lo <= min(g.closed, g.open) and max(g.closed, g.open) <= hi):
            raise ValueError(f"gripper closed and open are {joint.name!r} positions, in the URDF's units and "
                             f"zero; they must lie within its range {lo:g}..{hi:g}")
    return model, [j.name for j in grip]
