"""Prepare the existing six-knob RoboCasa stove for contact-driven dial turning."""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco as mj
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'robocasa-counter'))
from adapter import export_urdf, warm_lighting

from world_use.bodies.mujoco_render import CameraRenderer
from world_use.cameras import View
from world_use.geometry import aim, axis_angle
from world_use.kinematics import Chain

SOURCE = Path('runs/robocasa-small-cup/scene')
ROOT = Path('runs/robocasa-dial/scene')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    old = mj.MjModel.from_xml_path(str(SOURCE/'scene.xml'))
    state = np.load(SOURCE/'state.npz')['qpos']
    scene = ET.parse(SOURCE/'scene.xml').getroot()
    for body in list(scene.find('worldbody')):
        if body.get('name') in ('obj_main', 'distr_counter_main', 'distr_sink_main'):
            scene.find('worldbody').remove(body)
    # Upstream interaction tuning uses friction 100 and armature 1 on tiny knobs.
    # Use ordinary rigid contact and a low-resistance passive rotary joint instead.
    for joint in scene.iter('joint'):
        if joint.get('name', '').startswith('stove_main_group_knob_'):
            joint.set('armature', '.001')
            joint.set('frictionloss', '.04')
            joint.set('damping', '.015')
    for geom in scene.iter('geom'):
        if geom.get('name', '').startswith('stove_main_group_knob_'):
            geom.set('friction', '1.2 .02 .002')
            geom.set('solref', '.004 1')
    ET.indent(scene)
    xml = root/'scene.xml'
    xml.write_text(ET.tostring(scene, encoding='unicode'))
    model = mj.MjModel.from_xml_path(str(xml))
    data = mj.MjData(model)
    for j in range(model.njnt):
        source = mj.mj_name2id(old, mj.mjtObj.mjOBJ_JOINT, model.joint(j).name)
        if source >= 0:
            count = {mj.mjtJoint.mjJNT_FREE: 7, mj.mjtJoint.mjJNT_BALL: 4}.get(model.jnt_type[j], 1)
            dst, src = model.jnt_qposadr[j], old.jnt_qposadr[source]
            data.qpos[dst:dst+count] = state[src:src+count]
    mj.mj_forward(model, data)
    base = data.body('robot0_link0').xpos.copy()
    data.qpos[0] += -1.18-base[1]
    data.qpos[1] -= 3.10-base[0]
    # An already-set oven: the pointer starts near the original 300 F marking.
    # Set only before the episode; contact is the sole way to turn it during use.
    initial_angle = np.radians(80)
    data.qpos[model.jnt_qposadr[model.joint('stove_main_group_knob_temp_joint').id]] = initial_angle
    # A parked, open hand in front of the panel. This is setup, not recorded motion.
    export_urdf(model, root/'panda.urdf')
    chain = Chain(root/'panda.urdf', 'tool')
    seed = np.array([0, .5, 0, -1.6, 2, 1.3, -.5])
    pose = chain.fk(seed)
    jaws = axis_angle([1, 0, 0], initial_angle) @ [0, 0, 1]
    pose[:3, :3] = aim(pose[:3, :3], [0, 0, 1], [1, 0, 0], [1, 0, 0], jaws)
    pose[:3, 3] = [.43, -.117, .22]
    q, error = chain.ik(pose, seed, iters=500)
    if error > 1e-5:
        raise ValueError(f'Cannot park the Panda: IK residual {error}')
    data.qpos[4:11] = q
    data.qpos[11:13] = [.025, -.025]
    data.qvel[:] = 0
    mj.mj_forward(model, data)
    np.savez_compressed(root/'state.npz', qpos=data.qpos, qvel=data.qvel)
    warm_lighting(model)
    model.vis.global_.offwidth, model.vis.global_.offheight = 1920, 1080
    mj.mj_saveModel(model, str(root/'scene.mjb'), None)
    rig = dict(side=dict(eye=[2.67, -1.43, .96], target=[3.15, -.68, .84], fov_deg=58),
               overview=dict(eye=[2.05, -2.50, 1.65], target=[3.15, -.79, .86], fov_deg=57))
    renderer = CameraRenderer()
    for name, spec in rig.items():
        renderer.render(model, data, View.look_at(**spec, size=(1600, 1000))).save(root/f'{name}.png')
    renderer.close()
    (root/'metadata.json').write_text(json.dumps(dict(
        source=str(SOURCE), task='Set the oven temperature to 180 C; leave the timer and burners unchanged.',
        controls=6, initial_temperature_joint_deg=80,
        labels='Original upstream Fahrenheit scale, timer and burner symbols',
        rig=rig, base=data.body('robot0_link0').xpos.tolist(),
        mechanism=dict(armature_kg_m2=.001, joint_friction_Nm=.04, damping_Nm_s=.015,
                       contact_friction=1.2, note='Passive knob joints; no knob actuator or attachment')),
        indent=2)+'\n')
    print(root.resolve())


if __name__ == '__main__':
    main()
