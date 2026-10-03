"""A complete simulated task: approach, grip, transfer, release, verify, return.

This is a scripted reference policy, not a live model or a physics benchmark. The kernel's scene and the
simulator's truth stay separate. Success is measured from truth after releasing and withdrawing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import ImageDraw, ImageFont

from .. import Kernel, VirtualClock, World, bodies, cameras, check
from ..daemon import apply_workcell, load_workcell
from ..recorder import save_summary

SCENARIOS = ("nominal", "shifted", "missing", "misplaced")
TARGET = np.array([.34, -.07, .20])


def setup(scenario="nominal", output: Path | None = None):
    cell = load_workcell(Path("block"))
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
    if scenario == "shifted":
        cell["box"][1]["center"] = [.35, .02, .20]
    world, truth = World(), World()
    body = bodies.make("sim", truth, q=np.radians(cell["body_options"]["start_deg"]), gripper=1.0)
    k = Kernel(body, world, VirtualClock(100), run_dir=output)
    k.connect()
    truth.frames.update(world.frames)
    apply_workcell(cell, k, truth)
    if scenario == "missing":
        truth.boxes.pop("block")
    if scenario == "misplaced":
        truth.boxes["block"].pose[:3, 3] = truth.to_base("work", [.34, .12, .20])
    k.record_session(config=cell, scenario=scenario, policy="scripted reference; no model calls",
                     simulation_truth=truth.to_dict())
    return k, truth


def pickup(k) -> list[dict]:
    block = k.world.from_base("work", k.world.boxes["block"].pose[:3, 3])
    # The URDF tool frame is at the fingertips. Put the block inside the 30 mm pads.
    approach = k.world.frame("work").T[:3, :3].T @ k.tool[:3, :3] @ np.asarray(k.manifest.gripper.approach)
    block = block + .02 * approach
    high = float(k.world.from_base("work", k.tool[:3, 3])[2] + .08)
    return [{"do": "line", "up": .08}, {"do": "gripper", "aperture_mm": 65},
            {"do": "move_to", "to": [float(block[0]), float(block[1]), high]},
            {"do": "move_to", "to": [float(block[0]), float(block[1]), .22]},
            {"do": "grip", "expect_mm": [35, 45]}]


def placement(k) -> list[dict]:
    tool = k.world.from_base("work", k.tool[:3, 3])
    block = k.world.from_base("work", k.world.boxes["block"].pose[:3, 3])
    target = TARGET + tool - block
    return [{"do": "line", "up": .08},
            {"do": "move_to", "to": [float(target[0]), float(target[1]), float(tool[2] + .08)]},
            {"do": "line", "up": -.07}, {"do": "gripper", "aperture_mm": 65},
            {"do": "line", "up": .07}]


def success(k, truth: World) -> dict:
    block = truth.boxes.get("block")
    center = None if block is None else truth.from_base("work", block.pose[:3, 3])
    tool = truth.from_base("work", k.tool[:3, 3])
    placed = center is not None and np.linalg.norm(center - TARGET) < .01
    released = truth.held is None
    withdrawn = center is not None and tool[2] - center[2] > .08
    return dict(success=bool(placed and released and withdrawn), placed=bool(placed), released=released,
                withdrawn=bool(withdrawn), target=TARGET.tolist(),
                observed=None if center is None else np.round(center, 4).tolist())


def run(output: Path, scenario="nominal", *, video=True) -> dict:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"use an empty output directory: {output}")
    k, truth = setup(scenario, output)
    frames, next_frame, phase = [], 0.0, "Approach and grip"
    lens = cameras.View.look_at(truth.to_base("work", [.72, -.85, .65]),
                                truth.to_base("work", [.20, 0, .25]), size=(720, 450))
    camera = cameras.SimCamera("demo", lens, k.body)

    def execute(spec):
        nonlocal next_frame
        report = check(spec, k)
        # A missing object can pass the model's check. The live grip still has to meet its expectation.
        if report.refused or report.outcome.status not in ("done", "surprise"):
            return report.outcome
        job = k.submit(spec)
        deadline = k.clock.now() + 120
        while not job.finished:
            k.tick()
            if video and k.clock.now() + 1e-9 >= next_frame:
                img = camera.picture(k)
                observations = output / "views"
                observations.mkdir(exist_ok=True)
                path = observations / f"{len(frames):04d}-demo.jpg"
                img.save(path, quality=90)
                k.emit("look", "demo camera: simulation truth", camera="demo", path=f"views/{path.name}",
                       drawn="MuJoCo simulation", size=list(img.size))
                draw = ImageDraw.Draw(img)
                draw.rectangle((0, 0, img.width, 40), fill=(248, 248, 248))
                draw.text((16, 11), f"SIMULATION  /  {phase}  /  {k.clock.now():.1f}s  /  3x playback",
                          fill=cameras.INK, font=ImageFont.load_default(size=15))
                frames.append(img)
                next_frame += .21          # 0.21 simulated seconds per 70 ms GIF frame: exactly 3x
            k.clock.wait()
            if k.clock.now() >= deadline:
                k.stop("example timeout")
        return job.outcome

    try:
        k.enable()
        picked = execute(pickup(k))
        outcomes = [picked.to_dict()]
        if picked.ok:
            phase = "Transfer, release and withdraw"
            outcomes.append(execute(placement(k)).to_dict())
        else:
            # No contact disproves the assumed object at the jaws. Retire that belief before checking a retreat.
            k.world.boxes.pop("block", None)
            k.emit("world_state", "missed grasp: discard the assumed block pose", world=k.world.to_dict())
            # This example's open tray permits a straight retreat. A real scene needs a verified route.
            phase = "Missed grip: open and retreat"
            outcomes.append(execute([{"do": "gripper", "aperture_mm": 65}, {"do": "line", "up": .08}]).to_dict())
        result = success(k, truth)
        k.emit("task_result", "placement verified" if result["success"] else "placement not completed", **result)
        phase = "Return to rest"
        k.set_home_route([], "open tray: return from above the turn height")
        returned = execute(k.home_plan())
        if returned.ok:
            k.release()
        result.update(scenario=scenario, outcomes=outcomes, return_outcome=returned.to_dict(),
                      torque_off=not k.enabled and not k.power_uncertain)
        save_summary(output / "result.json", result)
        if frames:
            frames[0].save(output / "demo.gif", save_all=True, append_images=frames[1:], duration=70, loop=0)
        return result
    finally:
        k.close()
