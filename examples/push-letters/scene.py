"""Rigid letter workcell using RoboSimStudio's public robosuite Panda assets."""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco as mj
import numpy as np

REVISION = "73739abe212f963e8a9f5862b27bca370df091a7"
ASSETS = Path("runs/push-letters-assets/panda")
HEIGHT = .012
RADIUS = .004
COLORS = ((.12, .36, .76), (.85, .32, .12), (.12, .58, .42),
          (.58, .22, .68), (.82, .59, .10), (.12, .56, .68), (.72, .20, .38))


def numbers(values):
    return " ".join(f"{x:.12g}" for x in values)


def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def bars(letter):
    """Centre, half lengths and yaw of each rectangular stroke, in metres."""
    def vertical(x):
        return x, 0, .006, .030, 0

    def horizontal(y):
        return 0, y, .024, .006, 0
    shapes = {
        "L": [vertical(-.018), horizontal(-.024)],
        "T": [vertical(0), horizontal(.024)],
        "U": [vertical(-.018), vertical(.018), horizontal(-.024)],
        "O": [vertical(-.018), vertical(.018), horizontal(-.024), horizontal(.024)],
        "A": [(-.012, 0, .006, .0315, -.38), (.012, 0, .006, .0315, .38),
              (0, -.006, .017, .005, 0)],
        "B": [vertical(-.018), (-.006, -.024, .018, .006, 0),
              (-.006, 0, .018, .006, 0), (-.006, .024, .018, .006, 0),
              (.018, .012, .006, .012, 0), (.018, -.012, .006, .012, 0)],
    }
    return np.array(shapes[letter], float)


def inside(points, strokes):
    points = np.asarray(points)
    hit = np.zeros(points.shape[:-1], bool)
    for x, y, hx, hy, a in strokes:
        local = (points - [x, y]) @ rotation(a)
        hit |= (np.abs(local[..., 0]) <= hx) & (np.abs(local[..., 1]) <= hy)
    return hit


def template(letter, spacing=.001):
    axis = np.arange(-.04, .041, spacing)
    grid = np.array(np.meshgrid(axis, axis)).reshape(2, -1).T
    return grid[inside(grid, bars(letter))]


def boundary(letter):
    """Exposed stroke edges with outward normals; holes are legitimate contacts."""
    strokes = bars(letter)
    points, normals = [], []
    for x, y, hx, hy, a in strokes:
        r = rotation(a)
        for axis, half, other in ((0, hx, hy), (1, hy, hx)):
            for sign in (-1, 1):
                n = np.zeros(2)
                n[axis] = sign
                for t in np.linspace(-other * .85, other * .85, 7):
                    p = np.zeros(2)
                    p[axis], p[1-axis] = sign * half, t
                    p, n_world = p @ r.T + [x, y], n @ r.T
                    if not inside(p + n_world * .0001, strokes):
                        points.append(p)
                        normals.append(n_world)
    return np.array(points), np.array(normals)


def add_letter(parent, letter, *, name, center, yaw, color, physical, friction=.5):
    body = ET.SubElement(parent, "body", name=name, pos=numbers(center),
                         quat=numbers([np.cos(yaw/2), 0, 0, np.sin(yaw/2)]))
    if physical:
        ET.SubElement(body, "freejoint", name=name + "_free")
    strokes = bars(letter)
    for i, (x, y, hx, hy, a) in enumerate(strokes):
        ET.SubElement(body, "geom", name=f"{name}_{i}", type="box",
                      pos=numbers([x, y, 0 if physical else i*.000002]),
                      size=numbers([hx, hy, HEIGHT/2 if physical else .00012]),
                      quat=numbers([np.cos(a/2), 0, 0, np.sin(a/2)]), rgba=numbers([*color, 1]),
                      density="350", friction=numbers([friction, .001, .0001]),
                      solref=".004 1", solimp=".99 .99 .001", condim="3",
                      contype="1" if physical else "0", conaffinity="1" if physical else "0")
    return body


def build(output, assets=ASSETS, *, letters="LAB", variant="nominal"):
    output, assets = Path(output), Path(assets).resolve()
    scene = ET.parse(assets / "panda_nohand_standalone.xml").getroot()
    output.mkdir(parents=True, exist_ok=False)
    scene.set("model", "world_use_push_letters")
    scene.find("compiler").set("meshdir", str(assets))
    scene.find("compiler").attrib.pop("assetdir", None)
    scene.find("option").set("timestep", ".001")
    scene.find("option").set("integrator", "implicitfast")
    for elem in scene.findall(".//site"):
        elem.set("rgba", "0 0 0 0")
    for geom in scene.findall(".//geom"):
        if geom.get("group") == "0":
            geom.set("group", "3")
    for joint in scene.findall(".//joint"):
        joint.set("armature", ".1")
        joint.set("damping", "2")
        joint.set("frictionloss", "0")
    # Position servos retain upstream effort limits. Existing gravcomp handles weight.
    for actuator in scene.find("actuator"):
        actuator.set("kp", "900")
        actuator.set("kv", "60")
    hand = scene.find(".//body[@name='robot0_right_hand']")
    tool = ET.SubElement(hand, "body", name="push_tool", pos="0 0 .085", gravcomp="1")
    ET.SubElement(tool, "geom", name="pusher", type="capsule", fromto="0 0 -.066 0 0 -.004",
                  size=str(RADIUS), rgba=".15 .17 .19 1", mass=".04",
                  friction=".35 .001 .0001", solref=".004 1", solimp=".99 .99 .001")
    world = scene.find("worldbody")
    ET.SubElement(world, "geom", name="bench", type="box", pos=".38 0 -.025",
                  size=".52 .42 .025", rgba=".66 .65 .62 1", friction=".5 .001 .0001",
                  solref=".004 1", solimp=".99 .99 .001")
    ET.SubElement(world, "geom", name="mat", type="box", pos=".49 0 .00015",
                  size=".25 .34 .00015", rgba=".76 .75 .72 1", contype="0", conaffinity="0")
    ET.SubElement(world, "light", pos=".1 -.5 1.6", dir=".2 .3 -1", directional="true",
                  diffuse=".45 .45 .45", ambient=".10 .10 .10", specular=".05 .05 .05")
    visual = ET.SubElement(scene, "visual")
    ET.SubElement(visual, "global", offwidth="1600", offheight="1000")
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="4")
    ET.SubElement(visual, "headlight", ambient=".22 .22 .22", diffuse=".22 .22 .22", specular="0 0 0")
    metadata = dict(letters=[], variant=variant, upstream_revision=REVISION,
                    simulation="MuJoCo rigid letters; not the RSS Newton/sponge task")
    for i, letter in enumerate(letters):
        y = (len(letters)-1)/2 * .088 - i * .088
        target = [.49, y]
        angle = np.radians((22, -25, 18, -20, 28, -18, 20)[i])
        start = [.36 if i % 2 == 0 else .62, y + (.01 if i % 2 else -.01)]
        if variant == "shifted":
            start = np.array(start) + [.012, -.008]
            angle += np.radians(8 if i % 2 else -7)
        friction = .7 if variant == "friction" else .5
        color = COLORS[i]
        name = f"letter_{i}_{letter}"
        add_letter(world, letter, name=name, center=[*start, HEIGHT/2 + .001],
                   yaw=-np.pi/2+angle, color=color, physical=True, friction=friction)
        add_letter(world, letter, name="goal_"+name, center=[*target, .0005],
                   yaw=-np.pi/2, color=tuple(.52 + .08*v for v in color), physical=False)
        metadata["letters"].append(dict(name=name, letter=letter, color=color,
                                        target_xy=target, target_yaw=-np.pi/2))
    ET.indent(scene)
    path = output / "scene.xml"
    path.write_text(ET.tostring(scene, encoding="unicode"))
    model = mj.MjModel.from_xml_path(str(path))
    data = mj.MjData(model)
    data.qpos[:7] = [0, -.45, 0, -2.15, 0, 1.70, .7854]
    data.ctrl[:] = data.qpos[:7]
    for _ in range(800):
        mj.mj_step(model, data)
    np.savez_compressed(output / "initial.npz", qpos=data.qpos, qvel=data.qvel)
    (output / "task.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--letters", choices=["T", "LAB", "AUTOLAB"], default="LAB")
    parser.add_argument("--variant", choices=["nominal", "shifted", "friction"], default="nominal")
    args = parser.parse_args()
    print(build(args.output, args.assets, letters=args.letters, variant=args.variant))
