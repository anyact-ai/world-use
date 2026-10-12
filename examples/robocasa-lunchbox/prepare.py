"""Place upstream RoboCasa snack assets in the existing Panda kitchen.

Scene preparation only. No action policy, object attachment, or task-time pose edits.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robocasa-counter'))
from adapter import warm_lighting

ROOT = Path('runs/robocasa-lunchbox-compact/scene')
SOURCE = Path('runs/robocasa-small-cup/scene')
ASSETS = Path('runs/robocasa-preview/vendor/robocasa/robocasa/models/assets/objects/objaverse').resolve()


def num(x):
    return ' '.join(f'{v:.10g}' for v in x)


def append_object(scene, name, category, asset_id, scale, position, quaternion, color=None):
    path = ASSETS / category / f'{category}_{asset_id}' / 'model.xml'
    obj = ET.parse(path).getroot()
    names = {e.get('name'): name + '_' + e.get('name') for e in obj.iter() if e.get('name')}
    classes = {e.get('class'): name + '_' + e.get('class') for e in obj.iter('default') if e.get('class')}
    for e in obj.iter():
        for key in ('name', 'mesh', 'material', 'texture'):
            if e.get(key) in names:
                e.set(key, names[e.get(key)])
        for key in ('class', 'childclass'):
            if e.get(key) in classes:
                e.set(key, classes[e.get(key)])
        if e.get('file'):
            e.set('file', str(path.parent / e.get('file')))
        if e.tag == 'mesh':
            e.set('scale', num(np.fromstring(e.get('scale', '1 1 1'), sep=' ') * scale))
        if e.tag in ('geom', 'site', 'body'):
            for key in ('size', 'pos'):
                if e.get(key):
                    e.set(key, num(np.fromstring(e.get(key), sep=' ') * scale))
    # The default region boxes are metadata, not physical or visible geometry.
    for parent in obj.iter():
        for child in list(parent):
            if child.tag == 'geom' and child.get('class') == name + '_region':
                parent.remove(child)
    for e in obj.find('asset'):
        scene.find('asset').append(copy.deepcopy(e))
    default = obj.find('default')
    if default is not None:
        destination = scene.find('default')
        if destination is None:
            destination = ET.SubElement(scene, 'default')
        for e in default:
            destination.append(copy.deepcopy(e))
    body = copy.deepcopy(obj.find('worldbody/body/body'))
    body.set('name', name)
    body.set('pos', num(position))
    body.set('quat', num(quaternion))
    ET.SubElement(body, 'freejoint', name=name + '_free')
    for i, geom in enumerate(body.iter('geom')):
        geom.set('name', f'{name}_geom{i}')
        if geom.get('class') == name + '_collision' or geom.get('group') == '0':
            geom.set('friction', '.95 .03 .003')
            geom.set('solref', '.006 1')
            geom.set('density', '250')
        elif color is not None:
            geom.attrib.pop('material', None)
            geom.set('rgba', num(color))
    scene.find('worldbody').append(body)
    return dict(name=name, source=str(path), scale=scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--variant', choices=['nominal', 'shifted'], default='nominal')
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    source_model = mj.MjModel.from_xml_path(str(SOURCE / 'scene.xml'))
    source_state = np.load(SOURCE / 'state.npz')['qpos']
    scene = ET.parse(SOURCE / 'scene.xml').getroot()
    for body in list(scene.find('worldbody')):
        if body.get('name') in ('obj_main', 'distr_counter_main', 'distr_sink_main'):
            scene.find('worldbody').remove(body)
    # The unobstructed countertop between the sink and stove. All positions are setup only.
    sources = []
    shifted = args.variant == 'shifted'
    tray_angle = np.radians(6 if shifted else 0)
    sources.append(append_object(scene, 'lunchbox', 'tray', 0, 1.10,
                                [1.854 if shifted else 1.86, -.347 if shifted else -.335, .953],
                                [np.cos(tray_angle/2), 0, 0, np.sin(tray_angle/2)], [.48, .64, .67, 1]))
    angle = np.radians(72 if shifted else 65)
    sources.append(append_object(scene, 'long_snack', 'boxed_food', 7, 1.0,
                                [2.147 if shifted else 2.155, -.472 if shifted else -.465, .96],
                                [np.cos(angle/2), 0, 0, np.sin(angle/2)]))
    angle = np.radians(-33 if shifted else -25)
    # Lay the upstream cracker carton on its broad face; its original collision box is retained.
    qz = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
    qx = np.array([np.cos(np.pi/4), 0, np.sin(np.pi/4), 0])
    q = np.empty(4)
    mj.mju_mulQuat(q, qz, qx)
    sources.append(append_object(scene, 'cracker_box', 'boxed_food', 0, .85,
                                [2.156 if shifted else 2.15, -.239 if shifted else -.245, .945], q))
    xml = root / 'scene.xml'
    ET.indent(scene)
    xml.write_text(ET.tostring(scene, encoding='unicode'))
    model = mj.MjModel.from_xml_path(str(xml))
    data = mj.MjData(model)
    for j in range(model.njnt):
        name = model.joint(j).name
        old = mj.mj_name2id(source_model, mj.mjtObj.mjOBJ_JOINT, name)
        if old >= 0:
            count = {mj.mjtJoint.mjJNT_FREE: 7, mj.mjtJoint.mjJNT_BALL: 4}.get(model.jnt_type[j], 1)
            dst, src = model.jnt_qposadr[j], source_model.jnt_qposadr[old]
            data.qpos[dst:dst+count] = source_state[src:src+count]
    # Park the robot at the new counter station before recording.
    data.qpos[1] -= .68
    friction = model.dof_frictionloss[:13].copy()
    model.dof_frictionloss[:13] = 100000
    data.ctrl[:] = 0
    for _ in range(1500):
        mj.mj_step(model, data)
    model.dof_frictionloss[:13] = friction
    mj.mj_forward(model, data)
    np.savez_compressed(root / 'state.npz', qpos=data.qpos, qvel=data.qvel)
    warm_lighting(model)
    mj.mj_saveModel(model, str(root/'scene.mjb'), None)
    renderer = mj.Renderer(model, height=1000, width=1600)
    option = mj.MjvOption()
    option.geomgroup[3] = False
    option.sitegroup[:] = False
    camera = mj.MjvCamera()
    camera.lookat[:] = [2, -.43, .99]
    camera.azimuth, camera.elevation, camera.distance = 65, -38, 1.8
    renderer.update_scene(data, camera=camera, scene_option=option)
    Image.fromarray(renderer.render()).save(root/'preview.png')
    renderer.close()
    report = dict(task='Pack the two snacks compactly into the open lunchbox', variant=args.variant,
                  setup='Existing RoboCasa kitchen and upstream object meshes; base parked at counter',
                  sources=sources, base=data.body('robot0_link0').xpos.tolist(),
                  objects={n: data.body(n).xpos.tolist() for n in ('lunchbox','long_snack','cracker_box')},
                  finite=bool(np.isfinite(data.qpos).all()))
    (root/'metadata.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
