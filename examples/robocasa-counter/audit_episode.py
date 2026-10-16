"""Check final physical evidence after an episode; simulator state is not policy input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    root = parser.parse_args().recording
    replay = np.load(root / 'replay.npz')
    model = mujoco.MjModel.from_binary_path(str(root / 'scene.mjb'))
    data = mujoco.MjData(model)
    data.qpos[:] = replay['qpos'][-1]
    mujoco.mj_forward(model, data)
    cup = model.body('obj_main').id
    start = model.jnt_qposadr[model.body_jntadr[cup]]
    final_position = replay['qpos'][-1, start:start + 3]
    tail = replay['qpos'][replay['time'] >= replay['time'][-1] - 10, start:start + 3]
    drift = float(np.linalg.norm(tail - final_position, axis=1).max())
    contacts = []
    for contact in data.contact:
        if cup in model.geom_bodyid[contact.geom]:
            contacts.append(dict(geoms=[model.geom(int(g)).name for g in contact.geom],
                                 distance_m=float(contact.dist)))
    sink_contact = any(any(g.startswith('sink_main_group_') for g in c['geoms']) for c in contacts)
    tool_distance = float(np.linalg.norm(data.site('gripper0_right_grip_site').xpos - data.xpos[cup]))
    final = json.loads((root / 'final.json').read_text())
    torque_off = final['torque_off']
    cup_axis = data.xmat[cup].reshape(3, 3)[:, 2]
    result = dict(
        scope='Post-run simulator audit; not supplied to the agent as an observation',
        finite_replay=bool(np.isfinite(replay['qpos']).all()),
        cup_world_position_m=final_position.tolist(), sink_contact=sink_contact,
        contacts=contacts, cup_motion_last_10_wall_seconds_m=drift,
        cup_axis_degrees_from_vertical=float(np.degrees(np.arccos(np.clip(cup_axis[2], -1, 1)))),
        tool_distance_from_cup_m=tool_distance, torque_off=torque_off,
        recording_error=final.get('recording_error'),
    )
    result['passed'] = bool(result['finite_replay'] and sink_contact and drift < .001
                            and tool_distance > .25 and torque_off and not result['recording_error'])
    (root / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit('Final physical checks failed')


if __name__ == '__main__':
    main()
