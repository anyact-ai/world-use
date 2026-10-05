"""Bodies: robot adapters. `make(name)` builds one: "sim" (a twin of the reBot, or of the workcell's robot),
"sim:<built-in>", a built-in driver ("rebot"), or an installed "module:Class"."""
from __future__ import annotations

import importlib

from ..body import Body, Manifest

DRIVERS = {"rebot": "world_use.bodies.rebot:ReBotBody"}


def manifests() -> dict[str, Manifest]:
    from .rebot import MANIFEST as REBOT
    return {"rebot": REBOT}


def _adapter(name):
    module, sep, attribute = DRIVERS.get(name, name).partition(":")
    if not sep:
        raise ValueError(f"unknown body {name!r}; use sim, rebot, or an installed module:Class")
    try:
        return getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"cannot load adapter {name!r}: {e}") from e


def simulated(name: str) -> bool:
    return name == "sim" or name.startswith("sim:") or bool(getattr(_adapter(name), "simulated", False))


def make(name: str, world=None, *, manifest: Manifest | None = None, **options):
    """A body by name. Simulated bodies take q (start joints, rad), gripper and temp_c options. A built-in driver
    defaults to its robot's manifest; any other adapter needs the workcell's robot file."""
    kind, _, robot = name.partition(":")
    if kind == "sim":
        from .sim import SimBody
        if manifest is not None and robot:
            raise ValueError("choose --body sim with a robot file, or sim:<built-in>, not both")
        known = manifests()
        manifest = manifest or known.get(robot or "rebot")
        if manifest is None:
            raise ValueError(f"no manifest for {robot!r}; known: {sorted(known)}")
        return SimBody(manifest, world, **options)
    manifest = manifest or manifests().get(name)
    if manifest is None:
        raise ValueError(f"adapter {name!r} needs a robot TOML file in the workcell")
    body = _adapter(name)(manifest=manifest, **options)
    if not isinstance(body, Body) or body.manifest != manifest:
        raise ValueError(f"adapter {name!r} must implement Body and use the supplied manifest")
    return body
