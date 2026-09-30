"""Bodies: robot adapters. `make(name)` builds one: "sim" (a twin of the reBot), "sim:<robot>", or "<robot>"."""
from __future__ import annotations

from ..body import Manifest


def manifests() -> dict[str, Manifest]:
    from .rebot import MANIFEST as REBOT
    return {"rebot": REBOT}


def make(name: str, world=None, **options):
    """A body by name. Simulated bodies take q (start joints, rad), gripper and temp_c options."""
    kind, _, robot = name.partition(":")
    if kind == "sim":
        from .sim import SimBody
        known = manifests()
        robot = robot or "rebot"
        if robot not in known:
            raise KeyError(f"no manifest for {robot!r}; known: {sorted(known)}")
        return SimBody(known[robot], world, **options)
    if kind == "rebot":
        from .rebot import ReBotBody
        return ReBotBody(**options)
    raise KeyError(f"unknown body {name!r}: use 'sim', 'sim:<robot>' or a robot name ({sorted(manifests())})")
