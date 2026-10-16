"""Independent outcome checks from recorded simulation states, not camera estimates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np


def evaluate(model, data, task, replay, telemetry):
    time = replay["simulation_time"] if "simulation_time" in replay else replay["time"]
    qpos, qvel = replay["qpos"], replay["qvel"]
    finite = bool(len(time) > 1 and np.isfinite(qpos).all() and np.isfinite(qvel).all()
                  and np.isfinite(time).all() and (np.diff(time) > 0).all())
    settled = time >= time[-1]-2 if len(time) else np.array([], bool)
    dwell = bool(settled.sum() >= 35 and time[-1]-time[settled][0] >= 1.9) if len(time) else False
    letters = []
    for item in task["letters"]:
        body = data.body(item["name"])
        joint = model.joint(item["name"]+"_free").id
        qadr, vadr = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        rotation = body.xmat.reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        angle = (yaw-item["target_yaw"]+np.pi) % (2*np.pi)-np.pi
        if item["letter"] == "O":
            angle = (angle+np.pi/2) % np.pi-np.pi/2
        position_error = float(np.linalg.norm(body.xpos[:2]-item["target_xy"]))
        tilt = float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1))))
        drift = float(np.max(np.linalg.norm(qpos[settled, qadr:qadr+3]-body.xpos, axis=1))) if dwell else None
        speed = float(np.max(np.linalg.norm(qvel[settled, vadr:vadr+3], axis=1))) if dwell else None
        supported = False
        for contact in data.contact:
            geoms = [int(contact.geom1), int(contact.geom2)]
            if any(model.geom(g).name == "bench" for g in geoms):
                supported |= any(model.geom_bodyid[g] == body.id for g in geoms)
        touched = any(name.startswith(item["name"]+"_") and count > 0
                      for name, count in telemetry["pusher_contacts"].items())
        passes = (position_error < .005 and abs(np.degrees(angle)) < 5 and tilt < 2
                  and supported and touched and drift is not None and drift < .001
                  and speed is not None and speed < .001)
        letters.append(dict(name=item["name"], passed=bool(passes), position_error_mm=position_error*1000,
                            yaw_error_deg=float(np.degrees(angle)), tilt_deg=tilt,
                            supported=bool(supported), pushed=touched, drift_m=drift, speed_m_s=speed))
    tool_clear = bool(data.body("push_tool").xpos[2] > .09)
    physical = bool(telemetry["max_penetration_m"] < .001 and not telemetry.get("unexpected_contacts")
                    and not telemetry.get("solver_warnings"))
    passed = bool(finite and dwell and physical and tool_clear and telemetry["torque_off"]
                  and telemetry["error"] is None and all(item["passed"] for item in letters))
    return dict(passed=passed, finite=finite, settled_window=dwell, contacts_ok=physical,
                withdrawn=tool_clear, torque_off=telemetry["torque_off"], letters=letters,
                max_penetration_mm=telemetry["max_penetration_m"]*1000,
                criteria="<5 mm XY, <5 deg yaw, <2 deg tilt, table support, <1 mm drift and 1 mm/s over 2 s; "
                         "pin contact, <1 mm penetration, no other robot/letter contact, no solver warnings, power off")


def audit(folder):
    folder = Path(folder)
    model = mj.MjModel.from_binary_path(str(folder / "scene.mjb"))
    data = mj.MjData(model)
    replay = np.load(folder / "replay.npz")
    final = np.load(folder / "final-state.npz")
    data.qpos[:], data.qvel[:] = final["qpos"], final["qvel"]
    mj.mj_forward(model, data)
    result = evaluate(model, data, json.loads((folder / "task.json").read_text()), replay,
                      json.loads((folder / "result.json").read_text()))
    (folder / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    report = audit(args.folder)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
