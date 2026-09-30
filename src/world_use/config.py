"""Robot descriptions and workcells. Paths belong to their TOML file, not the launching shell."""
from __future__ import annotations

import math
import tomllib
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from .body import GripperSpec, JointSpec, Manifest, Rest
from .kinematics import Chain

WORKCELLS = Path(__file__).parent / "workcells"


def _keys(data, allowed, where):
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a table")
    if extra := data.keys() - set(allowed):
        raise ValueError(f"{where}: unknown fields: {', '.join(sorted(extra))}")


def _make(cls, data, where):
    _keys(data, (f.name for f in fields(cls)), where)
    try:
        return cls(**data)
    except TypeError as e:
        raise ValueError(f"{where}: {e}") from e


def _number(value, where, *, positive=False, infinity=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or math.isnan(value) or (not infinity and not math.isfinite(value))
            or (positive and value <= 0)):
        kind = "number" if infinity else "finite number"
        raise ValueError(f"{where}: expected a {'positive ' if positive else ''}{kind}")


def _vector(value, size, where):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{where}: expected {size} numbers")
    for x in value:
        _number(x, where)
    return tuple(value)


def _path(value, folder):
    if not isinstance(value, str) or not value:
        raise ValueError("expected a nonempty file path")
    return (folder / Path(value).expanduser()).resolve()


def manifest_data(m: Manifest) -> dict:
    """A portable description. Session frames are recorded separately in the world."""
    data = {k: v for k, v in asdict(m).items() if v is not None and k != "frames"}
    data["urdf"], data["sensing"] = str(m.urdf.absolute()), sorted(m.sensing)
    if m.rest is None:
        data["self_supporting"] = True
    else:
        for key in ("joints", "stops"):
            data["rest"][key] = [m.joints[i].name for i in getattr(m.rest, key)]
    if m.turn_clearance:
        joints, height = m.turn_clearance
        data["turn_clearance"] = dict(joints=[m.joints[i].name for i in joints], height_m=height)
    return data


def manifest_from_data(data: dict, folder: Path = Path(".")) -> Manifest:
    data = dict(data)
    support = data.pop("self_supporting", False)
    if type(support) is not bool or (support and "rest" in data) or (not support and "rest" not in data):
        raise ValueError("robot: specify either self_supporting = true or a [rest] pose")
    _keys(data, {f.name for f in fields(Manifest)} - {"frames"}, "robot")
    try:
        data["urdf"] = _path(data["urdf"], folder)
        joints = tuple(_make(JointSpec, j, f"joint[{i}]") for i, j in enumerate(data["joints"]))
    except KeyError as e:
        raise ValueError(f"robot: missing {e.args[0]}") from e
    names = [j.name for j in joints]
    if not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError("robot: joint names must be nonempty and unique")
    data["joints"] = joints
    for j in joints:
        for key in ("lower", "upper"):
            _number(getattr(j, key), f"{j.name}.{key}")
        if j.lower >= j.upper:
            raise ValueError(f"{j.name}: lower must be less than upper")
        for key in ("v_max", "a_max", "track_tol", "tau_max", "tau_hold_max", "contact_dtau"):
            _number(getattr(j, key), f"{j.name}.{key}", positive=True, infinity=key.startswith("tau_"))
        if type(j.excursion_exempt) is not bool:
            raise ValueError(f"{j.name}.excursion_exempt: expected a boolean")

    def indices(values, where):
        if not isinstance(values, (list, tuple)) or any(v not in names for v in values):
            raise ValueError(f"{where}: expected joint names from {names}")
        if len(set(values)) != len(values):
            raise ValueError(f"{where}: repeated joint names")
        return tuple(names.index(v) for v in values)

    if "rest" in data:
        rest = dict(data["rest"])
        for key in ("joints", "stops"):
            rest[key] = indices(rest.get(key, []), f"rest.{key}")
        r = _make(Rest, rest, "rest")
        r = Rest(_vector(r.q, len(joints), "rest.q"), r.joints, r.tol, r.stops)
        _number(r.tol, "rest.tol", positive=True)
        if not r.joints or not set(r.stops) <= set(r.joints):
            raise ValueError("rest: name the load-bearing joints; stops must be among them")
        if any(not j.lower <= q <= j.upper for j, q in zip(joints, r.q, strict=True)):
            raise ValueError("rest.q: pose must be inside the joint limits")
        data["rest"] = r
    if "gripper" in data:
        g = _make(GripperSpec, data["gripper"], "gripper")
        for key in ("closed", "open", "squeeze"):
            _number(getattr(g, key), f"gripper.{key}")
        for key in ("v_max", "track_tol", "tau_max"):
            _number(getattr(g, key), f"gripper.{key}", positive=True)
        if g.open == g.closed or g.squeeze < 0:
            raise ValueError("gripper: open and closed must differ; squeeze cannot be negative")
        if g.m_per_unit is not None:
            _number(g.m_per_unit, "gripper.m_per_unit")
            if (g.open - g.closed) * g.m_per_unit <= 0:
                raise ValueError("gripper.m_per_unit: opening must increase toward the open position")
        for key in ("approach", "opens_along"):
            axis = _vector(getattr(g, key), 3, f"gripper.{key}")
            if not np.isclose(np.linalg.norm(axis), 1):
                raise ValueError(f"gripper.{key}: expected a unit vector")
        data["gripper"] = g
    if "turn_clearance" in data:
        turn = data["turn_clearance"]
        _keys(turn, ("joints", "height_m"), "turn_clearance")
        selected = indices(turn.get("joints", []), "turn_clearance.joints")
        height = turn.get("height_m")
        _number(height, "turn_clearance.height_m", positive=True)
        if not selected:
            raise ValueError("turn_clearance.joints: select at least one joint")
        data["turn_clearance"] = (selected, height)
    for key in ("notes", "hardware_notes", "sensing"):
        if key in data:
            if not isinstance(data[key], (list, tuple)) or any(not isinstance(s, str) for s in data[key]):
                raise ValueError(f"robot.{key}: expected a list of strings")
            data[key] = frozenset(data[key]) if key == "sensing" else tuple(data[key])
    m = _make(Manifest, data, "robot")
    for key in ("name", "tool_link"):
        if not isinstance(getattr(m, key), str) or not getattr(m, key):
            raise ValueError(f"robot.{key}: expected a nonempty name")
    for key in ("rate_hz", "speed", "auto_accel", "min_move_s", "max_segment_m", "link_radius_m"):
        _number(getattr(m, key), f"robot.{key}", positive=True)
    if m.max_excursion is not None:
        _number(m.max_excursion, "robot.max_excursion", positive=True)
    for key in ("temp_warn_c", "temp_limit_c"):
        _number(getattr(m, key), f"robot.{key}")
    if m.temp_warn_c >= m.temp_limit_c:
        raise ValueError("robot: temp_warn_c must be below temp_limit_c")
    if "position" not in m.sensing or m.sensing - {"position", "torque", "temperature", "gripper_effort"}:
        raise ValueError("robot.sensing: requires position; optional torque, temperature, gripper_effort")
    _keys(m.thermal, ("heat", "cool_on", "cool_off"), "thermal")
    for key, value in m.thermal.items():
        for x in value if isinstance(value, (list, tuple)) else [value]:
            _number(x, f"thermal.{key}", positive=True)
        if isinstance(value, (list, tuple)) and len(value) != m.n:
            raise ValueError(f"thermal.{key}: expected one value per joint")
    return m


def load_robot(path: Path | str) -> Manifest:
    path = Path(path).expanduser().resolve()
    with path.open("rb") as f:
        m = manifest_from_data(tomllib.load(f), path.parent)
    chain = Chain(m.urdf, m.tool_link)
    if [j.name for j in m.joints] != chain.joint_names:
        raise ValueError(f"{path}: joints must follow the URDF chain order: {chain.joint_names}")
    if np.any(m.lower < chain.lower) or np.any(m.upper > chain.upper):
        raise ValueError(f"{path}: joint limits exceed the URDF limits")
    return m


def load_workcell(path: Path | str | None) -> dict:
    if path is None:
        return {}
    raw = str(path)
    path = Path(path).expanduser()
    if not path.exists() and raw == path.name and not path.suffix:
        bundled = WORKCELLS / f"{raw}.toml"
        if bundled.is_file():
            path = bundled
    with path.open("rb") as f:
        cell = tomllib.load(f)
    _keys(cell, ("body", "robot", "body_options", "simulation", "fit", "frame", "box", "camera", "fact", "envelope"),
          str(path))
    if "body" in cell and not isinstance(cell["body"], str):
        raise ValueError("workcell.body: expected an adapter name")
    for key in ("robot", "fit"):
        if key in cell:
            cell[key] = str(_path(cell[key], path.parent))
    for key in ("body_options", "simulation"):
        if key in cell and not isinstance(cell[key], dict):
            raise ValueError(f"{key}: expected a table of constructor options")
    sections = dict(
        frame=("name", "origin", "rpy_deg"),
        box=("name", "kind", "center", "size", "frame", "yaw_deg", "known", "grip_width", "dtau", "speed"),
        camera=("name", "path", "url", "command", "max_age_s", "rotate", "projection", "eye", "look_at", "up",
                "frame", "fov_deg", "size", "facing", "yaw_deg", "pitch_deg"),
        fact=("key", "value", "source", "note"))
    for section, allowed in sections.items():
        entries = cell.get(section, [])
        if not isinstance(entries, list):
            raise ValueError(f"{section}: use [[{section}]] for each entry")
        names = set()
        for entry in entries:
            _keys(entry, allowed, section)
            name = entry.get("key" if section == "fact" else "name")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError(f"{section}: names must be nonempty and unique")
            names.add(name)
    for camera in cell.get("camera", []):
        if len(camera.keys() & {"path", "url", "command"}) > 1:
            raise ValueError(f"camera {camera['name']}: choose one of path, url or command")
        if "path" in camera:
            camera["path"] = str(_path(camera["path"], path.parent))
    for frame in cell.get("frame", []):
        if frame["name"] == "base":
            raise ValueError("frame: the URDF base cannot be redefined")
        _vector(frame.get("origin", [0, 0, 0]), 3, "frame.origin")
        _vector(frame.get("rpy_deg", [0, 0, 0]), 3, "frame.rpy_deg")
    env = cell.get("envelope", {})
    _keys(env, ("max_excursion_deg", "reason"), "envelope")
    if "max_excursion_deg" in env:
        _number(env["max_excursion_deg"], "envelope.max_excursion_deg", positive=True)
    return cell
