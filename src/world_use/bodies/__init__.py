"""Bodies: robot adapters. `make(name)` builds one: "sim" (a twin of the reBot), "sim:<robot>", or "<robot>"."""
from __future__ import annotations

import importlib

from ..body import Body, Manifest


def manifests() -> dict[str, Manifest]:
    from .rebot import MANIFEST as REBOT
    return {"rebot": REBOT}


def _adapter(name):
    if name == "rebot":
        from .rebot import ReBotBody
        return ReBotBody
    module, sep, attribute = name.partition(":")
    if not sep:
        raise ValueError(f"unknown body {name!r}; use sim, rebot, or an installed module:Class")
    try:
        return getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"cannot load adapter {name!r}: {e}") from e


def simulated(name: str) -> bool:
    return name == "sim" or name.startswith("sim:") or bool(getattr(_adapter(name), "simulated", False))


def make(name: str, world=None, *, manifest: Manifest | None = None, **options):
    """A body by name. Simulated bodies take q (start joints, rad), gripper and temp_c options."""
    kind, _, robot = name.partition(":")
    if kind == "sim":
        from .sim import SimBody
        if manifest is not None and robot:
            raise ValueError("choose --body sim with a robot file, or sim:<built-in>, not both")
        if manifest is None:
            known = manifests()
            if (robot or "rebot") not in known:
                raise ValueError(f"no manifest for {robot!r}; known: {sorted(known)}")
            manifest = known[robot or "rebot"]
        return SimBody(manifest, world, **options)
    if name == "rebot":
        if manifest is not None:
            raise ValueError("the rebot driver uses its built-in model; use a custom adapter for a different robot")
        return _adapter(name)(**options)
    if manifest is None:
        raise ValueError(f"adapter {name!r} needs a robot TOML file in the workcell")
    body = _adapter(name)(manifest=manifest, **options)
    if not isinstance(body, Body) or body.manifest != manifest:
        raise ValueError(f"adapter {name!r} must implement Body and use the supplied manifest")
    return body
