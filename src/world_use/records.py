"""Offline inspection and visual replay of recorded measurements. Never opens a robot adapter."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import ImageDraw, ImageFont

from . import bodies, cameras
from .config import manifest_from_data
from .kinematics import Chain
from .recorder import Tape, load_tape
from .world import World


def events(folder: Path) -> list[dict]:
    path = folder / "events.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    out = []
    for i, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:          # only a final interrupted write can be ignored
                raise
    return out


def inspect(folder: Path | str) -> dict:
    folder = Path(folder)
    if not folder.is_dir() or not any((folder / name).exists() for name in ("session.json", "tape", "tape.npz")):
        raise ValueError(f"not a flight record: {folder}")
    session = json.loads((folder / "session.json").read_text()) if (folder / "session.json").exists() else {}
    log = events(folder)
    jobs = {}
    for e in log:
        data = e.get("data", {})
        if e["kind"] == "submitted":
            jobs.setdefault(data["job"], {}).update(spec=data["spec"])
        elif e["kind"] == "finished":
            jobs.setdefault(data["job"], {}).update(outcome=data.get("outcome", {"status": data["status"]}))
    a = load_tape(folder)
    return dict(run=str(folder.resolve()), session=session,
                closed=any(e["kind"] == "closed" for e in log),
                summary=Tape._summary(a, None), jobs=jobs,
                incidents=[e for e in log if e["level"] in ("warn", "alarm")],
                observations=[e for e in log if e["kind"] in ("look", "annotation", "answer")])


def describe(record: dict) -> str:
    s, meta = record["summary"], record["session"]
    lines = [f"{record['run']}",
             f"{meta.get('body', 'unknown body')} | {meta.get('mode', 'unknown mode')} | "
             f"world-use {meta.get('package_version', 'unknown')} | "
             + ("closed normally" if record["closed"] else "open or interrupted record"),
             f"{s.get('ticks', 0)} samples; powered {s.get('powered_s', 0)} s; moving {s.get('moving_s', 0)} s"]
    for job, data in record["jobs"].items():
        out = data.get("outcome", {})
        lines.append(f"job {job}: {out.get('status', 'no recorded outcome')} {out.get('message', '')}".rstrip())
    for e in record["incidents"]:
        lines.append(f"{e['t']:.1f}s {e['kind']}: {e['message']}")
    return "\n".join(lines)


def robot_of(folder: Path | str):
    """Use the recorded model; retain support for older records of built-in robots."""
    folder = Path(folder)
    meta = json.loads((folder / "session.json").read_text())
    if model := meta.get("initial", {}).get("model"):
        return manifest_from_data(model, folder)
    manifest = bodies.manifests().get(meta.get("adapter"))
    if manifest is None:
        raise ValueError("this older record has no robot description; replay requires a known built-in robot")
    return manifest


def replay(folder: Path | str, output: Path | str, fps: int = 12, speed: float = 1.0) -> Path:
    """Render measured joints and the recorded world model to a GIF, at an explicit playback speed."""
    folder, output = Path(folder), Path(output)
    if not 1 <= fps <= 60 or not 0 < speed <= 100:
        raise ValueError("fps must be 1..60 and speed must be greater than 0 and at most 100")
    meta = json.loads((folder / "session.json").read_text())
    manifest = robot_of(folder)
    a = load_tape(folder)
    if not len(a.get("t", [])):
        raise ValueError("this record has no committed telemetry")
    world = World.from_dict(meta["initial"]["world"])
    chain = Chain(manifest.urdf, manifest.tool_link)
    view = cameras.View.look_at(world.to_base("work", [.7, -.8, .65]),
                                world.to_base("work", [.30, 0, .18]), size=(800, 500))
    changes = [e for e in events(folder) if "world" in e.get("data", {})]
    change, frames = 0, []
    times = np.arange(a["t"][0], a["t"][-1] + 0.5 * speed / fps, speed / fps)
    if len(times) > 7200:
        raise ValueError("replay exceeds 7200 frames; increase --speed")
    for t in times:
        i = min(int(np.searchsorted(a["t"], t)), len(a["t"]) - 1)
        while change < len(changes) and changes[change]["t"] <= a["t"][i]:
            world = World.from_dict(changes[change]["data"]["world"])
            change += 1
        world.carry(chain.fk(a["q"][i]))
        g = manifest.gripper
        opening = g.aperture(a["grip"][i]) if g is not None and np.isfinite(a["grip"][i]) else None
        img = cameras.render(view, world, chain, a["q"][i], g, opening)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, img.width, 34), fill=(246, 246, 246))
        draw.text((14, 8), f"RECORDED JOINTS + WORLD MODEL  |  {t:.1f}s  |  {speed:g}x",
                  fill=cameras.INK, font=ImageFont.load_default(size=15))
        frames.append(img)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0)
    return output.resolve()
