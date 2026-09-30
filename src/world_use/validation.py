"""Validate built-in plan data before queueing; robot-specific limits remain in each behavior."""
from __future__ import annotations

import math
from numbers import Real

from .errors import Refused

MOTION = {'frame', 'duration', 'speed'}
FIELDS = {
    'joints': {'target_deg', 'delta_deg', 'duration', 'speed'},
    'line': MOTION | {'forward', 'left', 'up'},
    'lines': MOTION | {'legs', 'blend'},
    'move_to': MOTION | {'to', 'point', 'jaws', 'within_deg'},
    'guarded': {'frame', 'forward', 'left', 'up', 'dtau', 'expect_contact', 'speed_mps', 'joints'},
    'touchdown': {'max', 'dtau', 'speed_mps', 'joints'},
    'gripper': {'aperture_mm', 'to', 'seconds'},
    'grip': {'expect_mm', 'expect', 'start_mm', 'start', 'squeeze', 'effort', 'lag', 'speed', 'min', 'hold_effort'},
    'grasp': {'expect_mm', 'expect', 'start_mm', 'start', 'squeeze', 'effort', 'lag', 'speed', 'min', 'search_mm',
              'lift_mm', 'hold_effort'},
    'hold': {'seconds'},
    'checkpoint': {'ask', 'view', 'roi', 'expect'},
    'seq': {'steps'},
}
POSITIVE = {'duration', 'speed', 'speed_mps', 'dtau', 'effort', 'lag', 'max', 'hold_effort', 'lift_mm'}
NONNEGATIVE = {'seconds', 'blend', 'within_deg', 'squeeze', 'aperture_mm', 'start_mm'}
SCALARS = POSITIVE | NONNEGATIVE | {'forward', 'left', 'up', 'start', 'min'}


def number(value, field):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise Refused(f'{field} must be a finite number', 'spec')
    return value


def vector(value, field, length):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise Refused(f'{field} must contain {length} numbers', 'spec')
    for v in value:
        number(v, field)


def validate(kind: str, p: dict):
    unknown = p.keys() - FIELDS[kind]
    if unknown:
        raise Refused(f"{kind}: unknown parameter(s) {', '.join(sorted(unknown))}", 'spec',
                      f"parameters: {', '.join(sorted(FIELDS[kind]))}")
    for name, value in p.items():
        if value is None and (name == 'hold_effort' or (name == 'seconds' and kind == 'hold')):
            continue
        if name in SCALARS:
            number(value, name)
            if name in POSITIVE and value <= 0:
                raise Refused(f'{name} must be positive', 'spec')
            if name in NONNEGATIVE and value < 0:
                raise Refused(f'{name} must be nonnegative', 'spec')
        if name in ('frame', 'ask', 'view') and (not isinstance(value, str) or not value.strip()):
            raise Refused(f'{name} must be a nonempty string', 'spec')
        if name == 'expect_contact' and not isinstance(value, bool):
            raise Refused('expect_contact must be true or false', 'spec')
    for a, b in (('target_deg', 'delta_deg'), ('aperture_mm', 'to'), ('start_mm', 'start'), ('expect_mm', 'expect')):
        if a in p and b in p:
            raise Refused(f'give {a} or {b}, not both', 'spec')
    for name in ('target_deg', 'delta_deg'):
        if name in p:
            values = p[name]
            if not isinstance(values, dict) or not values:
                raise Refused(f'{name} must map joint numbers to degrees', 'spec')
            for key, value in values.items():
                if not str(key).isdigit() or int(key) < 1:
                    raise Refused(f'invalid joint number {key!r}', 'spec')
                number(value, name)
    if kind == 'joints' and not p.keys() & {'target_deg', 'delta_deg'}:
        raise Refused('joints needs target_deg or delta_deg', 'spec')
    if 'joints' in p:
        if not isinstance(p['joints'], list) or not p['joints']:
            raise Refused('joints must be a nonempty list of joint numbers', 'spec')
        for j in p['joints']:
            if isinstance(j, bool) or not isinstance(j, int) or j < 1:
                raise Refused('joints must be positive integers', 'spec')
    if 'to' in p:
        vector(p['to'], 'to', 3) if kind == 'move_to' else number(p['to'], 'to')
    for name in ('point', 'jaws'):
        if name in p and not isinstance(p[name], str):
            vector(p[name], name, 3)
            if not any(p[name]):
                raise Refused(f'{name} cannot be a zero vector', 'spec')
    for name in ('expect_mm', 'expect'):
        if name in p and kind in ('grip', 'grasp'):
            vector(p[name], name, 2)
    if kind == 'grasp':
        if not p.keys() & {'start_mm', 'start'}:
            raise Refused('grasp needs start_mm (or start): every retry reopens to it', 'spec')
        if p.get('lift_mm', 8) > 50:
            raise Refused('grasp lift_mm must be in (0, 50]', 'spec')
        if 'search_mm' in p:
            if not isinstance(p['search_mm'], (list, tuple)):
                raise Refused('search_mm must be a list of [across, along] pairs, mm', 'spec')
            for offset in p['search_mm']:
                vector(offset, 'search_mm offset', 2)
    if 'roi' in p and p['roi'] is not None:
        vector(p['roi'], 'roi', 4)
    if kind == 'lines':
        if not isinstance(p.get('legs'), list) or not p['legs']:
            raise Refused('lines needs a nonempty list of legs', 'spec')
        for leg in p['legs']:
            vector(leg, 'leg', 3)
    if kind == 'checkpoint' and 'ask' not in p:
        raise Refused('checkpoint needs ask', 'spec')
    if kind == 'checkpoint' and p.get('expect') is not None and not isinstance(p['expect'], str):
        raise Refused('checkpoint expect must be a string or null', 'spec')
    if kind == 'seq' and not isinstance(p.get('steps'), list):
        raise Refused('seq needs a list of steps', 'spec')
