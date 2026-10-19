"""Post-run contact and stability checks; simulator truth is not a policy observation."""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import mujoco as mj
import numpy as np


def vertices(model, data, geom_id):
    if model.geom_type[geom_id] == mj.mjtGeom.mjGEOM_BOX:
        points = np.array(list(product([-1, 1], repeat=3))) * model.geom_size[geom_id]
    elif model.geom_type[geom_id] == mj.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[geom_id])
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        points = model.mesh_vert[start:start+count]
    else:
        raise ValueError('Expected upstream mesh or box collision geometry')
    return points @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]


def contact_windows(root, phases):
    calls = json.loads((root/'calls.json').read_text())
    events = [json.loads(line) for line in (root/'flight/events.jsonl').read_text().splitlines()]
    windows = {}
    for name, phase in phases.items():
        call = calls[phase['place']]
        job = call['result']['id']
        submitted = next(e for e in events if e['kind']=='submitted' and e['data']['job']==job)
        finished = next(e for e in events if e['kind']=='finished' and e['data']['job']==job)
        lift = next(e for e in events if e['kind']=='step' and e['data']['step']==3
                    and submitted['seq'] < e['seq'] < finished['seq'])
        opened = call['start'] + lift['t'] - submitted['t']
        windows[name] = (calls[phase['grip']]['start'], opened)
    return windows


def motion_audit(root, model, data, replay, phases):
    """Check every saved state; finger contacts are allowed only during the declared pickup/place phase."""
    windows = contact_windows(root, phases)
    robot = {i for i in range(model.ngeom) if model.geom_contype[i] and
             (model.geom(i).name.startswith('robot0_link') or model.geom(i).name.startswith('gripper'))}
    pads = {i for i in robot if '_pad_collision' in model.geom(i).name}
    objects = {i: model.body(model.geom_bodyid[i]).name for i in range(model.ngeom)
               if model.geom_contype[i] and model.body(model.geom_bodyid[i]).name in phases}
    unexpected = {}
    for stamp, qpos in zip(replay['time'], replay['qpos'], strict=True):
        data.qpos[:] = qpos
        mj.mj_forward(model, data)
        for contact in data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if a not in robot and b not in robot and not (a in objects and b in objects and objects[a] != objects[b]):
                continue
            if b in robot and a not in robot:
                a, b = b, a
            if a in pads and b in objects:
                lo, hi = windows[objects[b]]
                if lo <= stamp <= hi:
                    continue
            key = ' / '.join(sorted([model.geom(a).name, model.geom(b).name]))
            entry = unexpected.setdefault(key, dict(first_s=float(stamp), last_s=float(stamp),
                                                   contact_samples=0, max_penetration_m=0.0))
            entry['last_s'] = float(stamp)
            entry['contact_samples'] += 1
            entry['max_penetration_m'] = max(entry['max_penetration_m'], float(-contact.dist))
    movement = {}
    for name in phases:
        adr = model.jnt_qposadr[model.joint(name+'_free').id]
        start = windows[name][1]+1  # Allow one second to settle after opening the fingers.
        positions = replay['qpos'][replay['time'] >= start, adr:adr+3]
        movement[name] = float(np.max(np.linalg.norm(positions-positions[0], axis=1)))
    report = dict(post_run_only=True, sampled_states=len(replay['time']),
                  contact_scope='Every replay sample; this is not continuous collision certification',
                  allowed_finger_windows_s=windows, unexpected_contacts=unexpected,
                  displacement_after_placement_m=movement)
    (root/'motion-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    assert not unexpected, report
    assert max(movement.values()) < .001, report
    print(f'PASS: {len(replay["time"])} replay states contain no unintended contacts; '
          'both placed snacks move less than 1 mm afterward.')


def audit(root):
    model = mj.MjModel.from_binary_path(str(root/'scene.mjb'))
    data = mj.MjData(model)
    replay = np.load(root/'replay.npz')
    final = json.loads((root/'final.json').read_text())
    assert not final.get('recording_error'), final
    windows = contact_windows(root, final['object_phases']) if 'object_phases' in final else {}
    data.qpos[:] = replay['qpos'][-1]
    mj.mj_forward(model, data)
    floor = model.geom('lunchbox_geom16').id
    container = data.body('lunchbox')
    rotation = container.xmat.reshape(3, 3)
    local_floor = (vertices(model, data, floor)-container.xpos) @ rotation
    bounds = np.array([local_floor[:, :2].min(0), local_floor[:, :2].max(0)])
    report = dict(post_run_only=True, torque_off=json.loads((root/'final.json').read_text())['torque_off'], objects={})
    for name in ('long_snack', 'cracker_box'):
        body_id = model.body(name).id
        geom_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body_id and model.geom_contype[i]]
        points = np.concatenate([vertices(model, data, i) for i in geom_ids])
        local = (points-container.xpos) @ rotation
        xy = np.array([local[:, :2].min(0), local[:, :2].max(0)])
        contacts = [[model.geom(c.geom1).name, model.geom(c.geom2).name] for c in data.contact
                    if (c.geom1 in geom_ids or c.geom2 in geom_ids)]
        on_floor = any(floor in (c.geom1, c.geom2) and (c.geom1 in geom_ids or c.geom2 in geom_ids)
                       for c in data.contact)
        fingers_clear = not any(any('gripper' in n for n in pair) for pair in contacts)
        adr = model.jnt_qposadr[model.joint(name+'_free').id]
        start = max(replay['time'][-1]-20, windows[name][1]+1) if windows else replay['time'][-1]-20
        tail_mask = replay['time'] > start
        tail = replay['qpos'][tail_mask, adr:adr+3]
        duration = float(np.ptp(replay['time'][tail_mask]))
        drift = float(np.linalg.norm(tail-tail[-1], axis=1).max())
        inside = bool((xy[0]>=bounds[0]-.001).all() and (xy[1]<=bounds[1]+.001).all())
        report['objects'][name] = dict(on_floor=on_floor, inside_floor_bounds=inside,
                                      fingers_clear=fingers_clear, settled_observation_s=duration, drift_m=drift,
                                      contacts=contacts, footprint_in_container_m=xy.tolist())
        assert on_floor and inside and fingers_clear and drift < .001 and duration >= 3, report
    assert report['torque_off']
    (root/'physics-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PASS: both snacks contact the container floor, remain inside it, are clear of the fingers,')
    print('and drift less than 1 mm after settling (observed durations in physics-audit.json); robot power is off.')
    if 'object_phases' in final:
        motion_audit(root, model, data, replay, final['object_phases'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    audit(parser.parse_args().recording)
