"""Built-in steps' parameters, in one table: it checks plan data before it is queued and describes each step to
policies as JSON Schema (`wu help`). Limits of the robot itself (joint ranges, gripper range, reach) are checked
when a step starts."""
from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Any

from .errors import Refused
from .world import DIRECTIONS


def _finite(v) -> bool:
    return isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(v)


def _numbers(v, n: int) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == n and all(map(_finite, v))


@dataclass(frozen=True)
class Param:
    """A parameter's type: its JSON Schema, the check validate() applies, and what that check wants, in words."""
    schema: dict
    ok: Callable[[Any], bool]
    wants: str

    def nullable(self) -> Param:
        return Param(dict(anyOf=[self.schema, dict(type="null")]), lambda v: v is None or self.ok(v),
                     f"{self.wants}, or null")


def array(n: int, nonzero: bool = False) -> Param:
    schema: dict = dict(type="array", items=dict(type="number"), minItems=n, maxItems=n)
    if nonzero:
        schema["not"] = dict(const=[0] * n)
    return Param(schema, lambda v: _numbers(v, n) and (any(v) or not nonzero),
                 f"{n} finite numbers" + (", not all zero" if nonzero else ""))


def listing(item: Param, wants: str, least: int = 0) -> Param:
    return Param(dict(type="array", items=item.schema, minItems=least),
                 lambda v: isinstance(v, (list, tuple)) and len(v) >= least and all(map(item.ok, v)), wants)


JOINT = "[1-9][0-9]*"
NUMBER = Param(dict(type="number"), _finite, "a finite number")
POSITIVE = Param(dict(type="number", exclusiveMinimum=0), lambda v: _finite(v) and v > 0, "a positive number")
NONNEGATIVE = Param(dict(type="number", minimum=0), lambda v: _finite(v) and v >= 0, "a number, 0 or more")
TEXT = Param(dict(type="string", minLength=1), lambda v: isinstance(v, str) and bool(v.strip()), "a nonempty string")
FLAG = Param(dict(type="boolean"), lambda v: isinstance(v, bool), "true or false")
DEGREES = Param(dict(type="object", minProperties=1, patternProperties={f"^{JOINT}$": dict(type="number")},
                     additionalProperties=False),
                lambda v: isinstance(v, dict) and bool(v) and all(re.fullmatch(JOINT, str(j)) and _finite(x)
                                                                  for j, x in v.items()),
                'joint numbers mapped to degrees, like {"2": 30}')
JOINTS = Param(dict(type="array", items=dict(type="integer", minimum=1), minItems=1),
               lambda v: isinstance(v, list) and bool(v) and all(type(j) is int and j >= 1 for j in v),
               "a nonempty list of joint numbers, from 1")
ARROW = array(3, nonzero=True)
DIRECTION = Param(dict(anyOf=[dict(enum=list(DIRECTIONS)), ARROW.schema]),
                  lambda v: v in DIRECTIONS if isinstance(v, str) else ARROW.ok(v),
                  f"one of {', '.join(DIRECTIONS)}, or [forward, left, up] not all zero")
LIFT = Param(dict(type="number", exclusiveMinimum=0, maximum=50), lambda v: _finite(v) and 0 < v <= 50,
             "a number above 0 and at most 50")
STEPS = Param(dict(type="array", items=dict(type=["object", "array"]), description="each item is a step"),
              lambda v: isinstance(v, list), "a list of steps")

MOTION = dict(frame=TEXT, duration=POSITIVE, speed=POSITIVE)
DISTANCES = dict.fromkeys(DIRECTIONS, NUMBER)          # metres along the frame's axes; back, right, down negate
GRIP = dict(expect_mm=array(2), expect=array(2), start_mm=NONNEGATIVE, start=NUMBER, squeeze=NONNEGATIVE,
            hold_effort=POSITIVE.nullable())
FIELDS: dict[str, dict[str, Param]] = {
    "joints": dict(target_deg=DEGREES, delta_deg=DEGREES, duration=POSITIVE, speed=POSITIVE),
    "line": dict(DISTANCES, **MOTION),
    "lines": dict(legs=listing(ARROW, "a nonempty list of legs, each [forward, left, up] not all zero", 1),
                  blend=NONNEGATIVE, **MOTION),
    "move_to": dict(to=array(3), point=DIRECTION, jaws=DIRECTION, within_deg=NONNEGATIVE, **MOTION),
    "guarded": dict(DISTANCES, frame=TEXT, dtau=POSITIVE, expect_contact=FLAG, speed_mps=POSITIVE, joints=JOINTS),
    "touchdown": dict(max=POSITIVE, dtau=POSITIVE, speed_mps=POSITIVE, joints=JOINTS),
    "gripper": dict(aperture_mm=NONNEGATIVE, to=NUMBER, seconds=NONNEGATIVE),
    "grip": GRIP,
    "grasp": dict(GRIP, search_mm=listing(array(2), "a list of [across, along] offsets"), lift_mm=LIFT),
    "hold": dict(seconds=NONNEGATIVE.nullable()),
    "checkpoint": dict(ask=TEXT, view=TEXT, roi=array(4).nullable(), expect=TEXT.nullable()),
    "seq": dict(steps=STEPS),
}
NEEDS = {"joints": ("target_deg", "delta_deg"), "line": tuple(DIRECTIONS), "lines": ("legs",),
         "move_to": ("to", "point", "jaws"), "guarded": tuple(DIRECTIONS), "gripper": ("aperture_mm", "to"),
         "grasp": ("start_mm", "start"), "checkpoint": ("ask",), "seq": ("steps",)}      # at least one of these
PAIRS = (("target_deg", "delta_deg"), ("aperture_mm", "to"), ("start_mm", "start"), ("expect_mm", "expect"),
         ("up", "down"), ("forward", "back"), ("left", "right"))                       # never both


def schema(kind: str) -> dict:
    """A built-in step as JSON Schema, from the table validate() checks; only validate() also refuses blank text
    and a line or guarded move of zero length."""
    fields = FIELDS[kind]
    out: dict = dict(type="object", properties=dict(do=dict(const=kind), label=dict(type="string"),
                                                    **{name: t.schema for name, t in fields.items()}),
                     required=["do"], additionalProperties=False)
    needs = NEEDS.get(kind, ())
    if len(needs) == 1:
        out["required"].append(needs[0])
    elif needs:
        out["anyOf"] = [dict(required=[name]) for name in needs]
    pairs = [list(pair) for pair in PAIRS if set(pair) <= fields.keys()]
    if pairs:
        out["allOf"] = [{"not": dict(required=pair)} for pair in pairs]
    return out


def validate(kind: str, p: dict):
    """Refuse a built-in step's parameters that are unknown, of the wrong type, missing or contradictory."""
    fields = FIELDS[kind]
    if unknown := p.keys() - fields.keys():
        raise Refused(f"{kind}: unknown parameter(s) {', '.join(sorted(map(str, unknown)))}", "spec",
                      f"parameters: {', '.join(sorted(fields))}")
    for name, value in p.items():
        if not fields[name].ok(value):
            raise Refused(f"{kind}: {name} must be {fields[name].wants}", "spec")
    needs = NEEDS.get(kind, ())
    if needs and not p.keys() & set(needs):
        names = f"{', '.join(needs[:-1])} or {needs[-1]}" if len(needs) > 1 else needs[0]
        raise Refused(f"{kind} needs {names}", "spec")
    for a, b in PAIRS:
        if a in p and b in p:
            raise Refused(f"{kind}: give {a} or {b}, not both", "spec")
    if kind in ("line", "guarded") and not any(p.get(word) for word in DIRECTIONS):
        raise Refused(f"{kind} needs a distance that is not zero", "spec")


def references(p: dict, world, joints: int):
    """Refuse a built-in step that names a frame the world lacks or a joint the robot lacks."""
    if "frame" in p:
        world.frame(p["frame"])
    for j in (*p.get("target_deg", ()), *p.get("delta_deg", ()), *p.get("joints", ())):
        if int(j) > joints:
            raise Refused(f"no joint {j}: this robot's joints are numbered 1 to {joints}", "spec")
