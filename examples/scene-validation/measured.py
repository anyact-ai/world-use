"""Diagnose a pinned MIT push with an independently constructed MuJoCo scene.

This previously inspected hex trajectory is development evidence. Use trials.py
for the separate recorded pushes selected before reading their measurements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from validate import LIMITS

from world_use.bodies.mujoco_scene import mj
from world_use.scene_validation import Observation, Validation, scene_identity
from world_use.tasks import digest, json_bytes, write_new

HERE = Path(__file__).resolve().parent
REFERENCE = json.loads((HERE / "reference.json").read_text())
SOURCES = ("measured.py", "reference.json", "validate.py")
FRICTIONS = (0.1, 0.2, 0.3, 0.4, 0.6, 0.8)


def vertices(geometry):
    if geometry["shape"] == "hex":
        side = geometry["hex_side_m"]
        return side * np.array([[np.cos(i * np.pi / 3), np.sin(i * np.pi / 3)] for i in range(6)])
    half = geometry["width_m"] / 2
    return np.array([[half, half], [-half, half], [-half, -half], [half, -half]])


def body_points(poses, geometry=REFERENCE["geometry"]):
    """Labelled polygon vertices expose both translation and yaw errors."""
    xy, theta = poses[:, :2], poses[:, 2]
    c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
    points = vertices(geometry)
    x, y = points[:, 0], points[:, 1]
    return np.stack([xy[:, 0, None] + c * x - s * y, xy[:, 1, None] + s * x + c * y,
                     np.zeros((len(poses), len(points)))], axis=-1)


def build_model(tip_xy, initial, friction, geometry=REFERENCE["geometry"]):
    height, mass = geometry["thickness_m"], geometry["mass_kg"]
    mesh = " ".join(str(v) for z in (-height / 2, height / 2) for x, y in vertices(geometry) for v in (x, y, z))
    izz = geometry["polar_inertia_kg_m2"]
    ixy = izz / 2 + mass * height**2 / 12
    probe = REFERENCE["probe"]
    radius, length, bottom = (probe[key] for key in ("radius_m", "length_m", "bottom_m"))
    probe_z = bottom + length / 2
    probe_mu = probe["sliding_friction"]
    density = REFERENCE["assumptions"]["probe_density_kg_m3"]
    # Explicit pairs prevent the fitted support friction from changing probe friction.
    # Translation joints preserve the recorded vertical probe orientation. A free
    # probe welded at its centre can tilt and displace its contact tip unobservably.
    return mj.MjModel.from_xml_string(f"""<mujoco>
      <option timestep="0.0005" integrator="implicitfast" cone="elliptic" iterations="100"/>
      <default><geom contype="0" conaffinity="0"/>
        <pair condim="3" solref=".001 1" solimp=".99 .99 .001"/>
      </default>
      <asset><mesh name="polygon" vertex="{mesh}"/></asset>
      <worldbody>
        <geom name="support" type="plane" size="1 1 .01"/>
        <body name="object" pos="{initial[0]} {initial[1]} {height / 2}"
              quat="{np.cos(initial[2] / 2)} 0 0 {np.sin(initial[2] / 2)}">
          <freejoint name="object_pose"/>
          <inertial pos="0 0 0" mass="{mass}" diaginertia="{ixy} {ixy} {izz}"/>
          <geom name="object_shape" type="mesh" mesh="polygon"/>
        </body>
        <body name="pusher" pos="{tip_xy[0]} {tip_xy[1]} {probe_z}">
          <joint name="probe_x" type="slide" axis="1 0 0"/>
          <joint name="probe_y" type="slide" axis="0 1 0"/>
          <geom name="probe" type="cylinder" size="{radius} {length / 2}" density="{density}"/>
        </body>
        <body name="target" mocap="true" pos="{tip_xy[0]} {tip_xy[1]} {probe_z}"/>
      </worldbody>
      <contact>
        <pair geom1="support" geom2="object_shape" friction="{friction} {friction} 0 0 0"/>
        <pair geom1="probe" geom2="object_shape" friction="{probe_mu} {probe_mu} 0 0 0"/>
      </contact>
      <equality><weld body1="pusher" body2="target" solref=".001 1" solimp=".99 .99 .001"/></equality>
    </mujoco>""")


def simulate(tip, initial, duration, friction, geometry=REFERENCE["geometry"]):
    at_start = [np.interp(0, tip[:, 0], tip[:, axis]) for axis in (1, 2)]
    model = build_model(at_start, initial, friction, geometry)
    data = mj.MjData(model)
    obj_id, probe_id = model.body("object").id, model.body("pusher").id
    for _ in range(400):
        mj.mj_step(model, data)
    data.time = 0
    rows, states, penetration, xyz, probe_xy, probe_velocity = [], [], [], [], [], []
    # Simulate through the end to cover the final asynchronous reference timestamp.
    for _ in range(int(np.ceil(duration / model.opt.timestep)) + 1):
        for index in range(2):
            data.mocap_pos[0, index] = np.interp(data.time, tip[:, 0], tip[:, index + 1])
        mj.mj_forward(model, data)
        yaw = np.arctan2(data.xmat[obj_id, 3], data.xmat[obj_id, 0])
        rows.append([data.time, *data.xpos[obj_id, :2], yaw])
        states.append(np.r_[data.qpos, data.qvel])
        xyz.append(data.xpos[obj_id].copy())
        probe_xy.append(data.xpos[probe_id, :2].copy())
        probe_velocity.append(data.qvel[model.jnt_dofadr[model.joint("probe_x").id]:][:2].copy())
        penetration.append(max((max(0, -contact.dist) for contact in data.contact), default=0))
        mj.mj_step(model, data)
    samples = np.array(rows)
    samples[:, 3] = np.unwrap(samples[:, 3])
    return samples, dict(state=np.array(states), penetration_m=np.array(penetration),
                         xyz=np.array(xyz), support_error_m=np.array(xyz)[:, 2] - geometry["thickness_m"] / 2,
                         probe_xy=np.array(probe_xy), probe_velocity=np.array(probe_velocity))


def load(path, expected_sha256=REFERENCE["trajectory_sha256"]):
    payload = path.read_bytes()
    if digest(payload) != expected_sha256:
        raise ValueError("the measured input does not match the pinned reference hash")
    data = json.loads(payload)
    streams = [np.array(data[key], float) for key in ("tip_pose", "object_pose")]
    for stream in streams:
        if (stream.ndim != 2 or stream.shape[1] != 4 or len(stream) < 2 or not np.isfinite(stream).all()
                or np.any(np.diff(stream[:, 0]) <= 0)):
            raise ValueError("measured poses need finite, strictly increasing timestamp,x,y,yaw samples")
        stream[:, 3] = np.unwrap(stream[:, 3])
    tip, obj = streams
    start, end = max(tip[0, 0], obj[0, 0]), min(tip[-1, 0], obj[-1, 0])
    if end <= start:
        raise ValueError("measured pusher and object streams have no common interval")
    initial = np.array([np.interp(start, obj[:, 0], obj[:, i]) for i in range(1, 4)])
    tip[:, 0] -= start
    obj[:, 0] -= start
    obj = obj[(obj[:, 0] >= 0) & (obj[:, 0] <= end - start)]
    return tip, obj, initial, end - start, start


def motion_error(reference, samples, geometry):
    points = body_points(samples[:, 1:4], geometry).reshape(len(samples), -1)
    predicted = np.column_stack([np.interp(reference[:, 0], samples[:, 0], points[:, i])
                                 for i in range(points.shape[1])])
    expected = body_points(reference[:, 1:4], geometry)
    return float(np.max(np.linalg.norm(predicted.reshape(expected.shape) - expected, axis=-1)))


def compare(output, input_path, trajectory, friction, *, geometry=REFERENCE["geometry"],
            split="fit", origin_url=REFERENCE["trajectory_url"], sources=SOURCES):
    tip, obj, initial, duration, origin = trajectory
    parameters = dict(geometry=geometry, probe=REFERENCE["probe"], assumptions=REFERENCE["assumptions"],
                      friction=friction,
                      initial_pose=initial.tolist(), timestep_s=0.0005, solver_iterations=100,
                      mujoco_version=mj.__version__, reference_time_origin_s=origin,
                      motion_metric="maximum labelled-vertex displacement in xy, includes yaw error")
    scene = scene_identity({name: HERE / name for name in sources}, parameters)
    alignment = dict(frame="dataset robot xy; metres, yaw radians", image_transform="no image reference",
                     time_offset_s=0.0, reference_time_origin_s=origin)
    observations = [Observation("object_motion", input_path, origin_url, "measured", split, alignment,
                                checks=("motion", "stability"), interval_s=(0, duration)),
                    Observation("probe_motion", input_path, origin_url, "measured", split, alignment,
                                checks=("motion",), interval_s=(0, duration))]
    validation = Validation(output, scene=scene, observations=observations, tolerances=LIMITS,
                            limitations=tuple(REFERENCE["evidence_limits"]))
    samples, stability = simulate(tip, initial, duration, friction, geometry)
    validation.motion("object_motion", obj[:, 0], body_points(obj[:, 1:4], geometry),
                      samples[:, 0], body_points(samples[:, 1:4], geometry))
    validation.stability("object_motion", state=stability["state"], penetration_m=stability["penetration_m"],
                         support_error_m=stability["support_error_m"], rest_xyz_m=None)
    selected = tip[(tip[:, 0] >= 0) & (tip[:, 0] <= duration)]
    validation.motion("probe_motion", selected[:, 0], np.c_[selected[:, 1:3], np.zeros(len(selected))],
                      samples[:, 0], np.c_[stability["probe_xy"], np.zeros(len(samples))])
    return validation.finish()


def summarize(report, split):
    checks = report["groups"]["physical_response"][split]["checks"]
    motion = next(check for check in checks if check["name"] == "motion" and check["observation"] == "object_motion")
    probe = next(check for check in checks if check["observation"] == "probe_motion")
    stability = next(check for check in checks if check["name"] == "stability")
    return dict(status=report["status"], friction=report["scene"]["parameters"]["friction"],
                max_vertex_error_m=motion["metrics"]["motion_m"], max_probe_error_m=probe["metrics"]["motion_m"],
                **stability["metrics"], stability_status=stability["status"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="local raw JSON matching reference.json's hash")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    trajectory = load(args.input)
    tip, obj, initial, duration, _ = trajectory
    args.output.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        write_new(args.output / "source" / name, (HERE / name).read_bytes())
    cut = 0.5 * duration
    fit = obj[obj[:, 0] <= cut]
    scores = []
    for friction in FRICTIONS:
        samples, _ = simulate(tip, initial, cut, friction)
        scores.append(dict(friction=friction, max_vertex_error_m=motion_error(fit, samples, REFERENCE["geometry"])))
    chosen = min(scores, key=lambda row: row["max_vertex_error_m"])["friction"]
    write_new(args.output / "fit-selection.json", json_bytes(dict(
        observations="first half of a previously inspected development push; full push is diagnostic only",
        scores=scores, chosen_friction=chosen, reference_sha256=REFERENCE["trajectory_sha256"])))
    reports = {name: compare(args.output / name, args.input, trajectory, friction)
               for name, friction in (("initial", 0.4), ("refined", chosen))}
    summary = {name: summarize(report, "fit") for name, report in reports.items()}
    write_new(args.output / "comparison.json", json_bytes(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
