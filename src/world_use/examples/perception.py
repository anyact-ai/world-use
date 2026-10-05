"""Measure, move, measure again: a block transfer through the MCP tools an agent uses, on a live simulation.

The procedure sees only tool replies and pictures. Picking the block's pixels stands in for an agent's eye: its
orange colour in the picture, or with --model edgetam, EdgeTAM following a first selection. The runner owns the
simulator, keeps physics running in real time, and alone judges the result against the simulator's truth, after
the release and before homing.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .. import Kernel, RealClock, World, bodies, cameras
from ..config import load_workcell
from ..daemon import Daemon, apply_workcell, session_identity
from ..mcp_server import build
from ..recorder import save_summary
from .pick_place import TARGET, success

SIZE = [.04, .04, .10]              # the procedure knows the block's shape: upright, 4 x 4 x 10 cm
GRASP = np.array([.02, 0, -.03])    # tool point from the top-face centre: fingertips past it, pads below it
TRAY = .22                          # tool height that sets a held block down on the tray
MAX_AGE_S = 45                      # how old a measurement a phase may rely on, for this task


def seen(image) -> tuple[list[float], list[int]] | None:
    """The middle of the block's lit top face and the box around the whole block, judged by colour alone: the
    block is orange, its top face the brightest orange. No simulator data is used."""
    rgb = np.asarray(image.convert("RGB")).astype(float)
    orange = (rgb[..., 0] > 65) & (rgb[..., 0] > 1.6 * rgb[..., 1]) & (rgb[..., 1] > 1.3 * rgb[..., 2])
    ys, xs = np.nonzero(orange & (rgb[..., 0] > 200))
    if len(xs) < 16:
        return None
    point = [float(np.median(xs)), float(np.median(ys))]
    ys, xs = np.nonzero(orange)
    return point, [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def at(*xyz) -> list[float]:
    """A work-frame position as a plan states it, to a tenth of a millimetre."""
    return np.round(xyz, 4).tolist()


def lifted(before: dict, after: dict) -> bool:
    """The block rose with the tool: its offset from the tool held while its top face rose."""
    held = np.linalg.norm(np.subtract(after["from_tool"], before["from_tool"])) < .015
    return bool(held and after["surface_center"][2] - before["surface_center"][2] > .035)


def placed(after: dict, aperture_mm: float) -> bool:
    """The block stands at the target, the gripper is open and the tool is clear above the block."""
    top, expected = np.asarray(after["surface_center"]), TARGET + [0, 0, SIZE[2] / 2]
    clearance = SIZE[2] / 2 - after["from_tool"][2]              # tool above the block's centre
    return bool(np.linalg.norm(top[:2] - expected[:2]) < .01 and abs(top[2] - expected[2]) < .008
                and clearance > .08 and aperture_mm >= 60)


async def procedure(server, evaluate, *, tracking=False) -> dict:
    """One attempt without re-grasps; every waypoint stays over the known clear tray."""
    results: dict = dict(selector="EdgeTAM" if tracking else "colour", measurements=[], outcomes=[],
                         lift="unknown", placement="unknown")

    async def tool(which, /, **arguments):
        return (await server.call_tool(which, arguments)).structured_content

    async def measure() -> dict | None:
        """The block measured in a new frame: at its middle pixel, or by the tracker after the first selection."""
        if tracking and results["measurements"]:
            m = (await tool("observe_targets", targets=["block"]))["measurements"][0]
        else:
            reply = await server.call_tool("camera_frame", dict(camera="overhead", depth=True))
            image = next(c for c in reply.content if c.type == "image")
            block = seen(Image.open(BytesIO(base64.b64decode(image.data))))
            if block is None:
                return None
            point, box = block
            frame = reply.structured_content["frame"]
            m = await (tool("select_target", frame=frame, target="block", box=box) if tracking
                       else tool("measure_pixels", frame=frame, target="block", point=point))
        results["measurements"].append({key: m.get(key) for key in ("id", "valid", "reason", "surface_center",
                                                                     "from_tool", "tracking")})
        return m if m["valid"] else None

    async def run(plan, relies_on=None):
        requires = [] if relies_on is None else [dict(evidence=relies_on["id"], max_age_s=MAX_AGE_S)]
        outcome = await tool("run", plan=plan, wait_s=120, requires=requires)
        results["outcomes"].append(outcome)
        if outcome["status"] != "done":
            raise RuntimeError(outcome.get("incident") or f"phase {outcome['status']}")

    await tool("policy")
    status = await tool("status")
    if status["session"]["mode"] != "simulation":
        raise ValueError("this example is only for simulation")
    await tool("card")
    block = await measure()
    if block is None:
        results.update(reason="no block measured; nothing was powered", torque_off=not status["enabled"])
        evaluate()
        return results
    top = np.asarray(block["surface_center"])
    await tool("add_box", name="block", kind="object", center=(top - [0, 0, SIZE[2] / 2]).tolist(), size=SIZE,
               source=f"measurement {block['id']}, assuming the known upright block")
    await tool("enable")
    try:
        # The unpowered wrist may have settled while the procedure looked: set the grasp orientation first.
        await run([dict(do="line", up=.08), dict(do="gripper", aperture_mm=65),
                   dict(do="move_to", point="forward", jaws="left", within_deg=0)])
        high = (await tool("status"))["tool"]["work"][2]
        await run(dict(do="move_to", to=at(*(top + GRASP)[:2], high)), block)
        block = await measure()
        if block is None:
            raise RuntimeError("the block is no longer measurable below the gripper")
        await run([dict(do="move_to", to=at(*(np.asarray(block["surface_center"]) + GRASP))),
                   dict(do="grip", expect_mm=[35, 45])], block)
        before = await measure()
        await run(dict(do="line", up=.06))
        after = await measure()
        if before is None or after is None:
            raise RuntimeError("the block was not measurable around the lift")
        results["lift"] = "pass" if lifted(before, after) else "fail"
        if results["lift"] != "pass":
            raise RuntimeError("the block did not rise with the tool")
        here = (await tool("status"))["tool"]["work"]
        # The block keeps its offset from the tool, so with the tool here its top is above the target's.
        to = TARGET + [0, 0, SIZE[2] / 2] - np.asarray(after["from_tool"])
        await run(dict(do="move_to", to=at(to[0], to[1], here[2])), after)
        # Use the measured placement height; a fixed relative descent can release above the tray.
        await run([dict(do="move_to", to=at(*to)), dict(do="gripper", aperture_mm=65),
                   dict(do="line", up=.08)], after)
        final = await measure()
        aperture = (await tool("status"))["gripper"]["aperture_mm"]
        results["placement"] = "pass" if final and placed(final, aperture) else "fail"
    except Exception as e:
        results["reason"] = str(e)
        # Over this known open tray, put the block down before opening; never release at height.
        await tool("stop", reason="the procedure's expectation failed")
        here = (await tool("status"))["tool"]["work"]
        lower = [dict(do="move_to", to=at(here[0], here[1], TRAY))] if abs(here[2] - TRAY) > .001 else []
        await run([*lower, dict(do="gripper", aperture_mm=65), dict(do="line", up=.08)])
    finally:
        evaluate()          # independent, after the release and before homing changes the withdrawal
        await tool("home_route", steps=[], note="known clear tray; return from above the turn height")
        results["return_outcome"] = await tool("home", wait_s=120)
        if results["return_outcome"]["status"] == "done":
            await tool("release")
    results["torque_off"] = not (await tool("status"))["enabled"]
    return results


def run(output: Path, *, scenario="displaced", model="color", device="cpu", speed=1.0) -> dict:
    """Own the simulator, its truth and the evaluator; the procedure gets none of them. speed > 1 ticks the
    control loop, and with it physics, that many times faster than real time."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"use an empty output directory: {output}")
    cell = load_workcell(Path("block"))
    if scenario == "displaced":
        cell["box"][1]["center"] = [.34, .10, .20]
    elif scenario == "missing":
        cell["box"].pop(1)
    elif scenario != "nominal":
        raise ValueError("scenario must be nominal, displaced or missing")
    for box in cell["box"]:
        if box["kind"] == "object":
            box["known"] = False
    if model not in ("color", "edgetam"):
        raise ValueError("model must be color or edgetam")
    tracker = None
    if model == "edgetam":                              # load the model before any motor is powered
        from ..vision_worker import TrackerProcess
        tracker = TrackerProcess(device=device)
    world, truth = World(), World()
    body = bodies.make("sim", truth, q=np.radians(cell["body_options"]["start_deg"]), gripper=1.0)
    k = Kernel(body, world, RealClock(body.manifest.rate_hz * speed), run_dir=output)
    k.connect()
    truth.frames.update(world.frames)
    apply_workcell(cell, k, truth)
    lens = cameras.View.look_at(world.to_base("work", [.34, 0, .90]), world.to_base("work", [.34, 0, .15]),
                                size=(512, 512), fov_deg=40)
    d = Daemon(k, port=0, cams={"overhead": cameras.SimCamera("overhead", lens, body)},
               session=session_identity("sim", cell))
    d.start()
    evaluated: dict = {}

    def evaluate():
        with k.lock, body.lock:
            evaluated.update(success(k, truth))
        k.emit("task_result", "independent simulator evaluation", **evaluated)

    try:
        server = build(f"http://127.0.0.1:{d.http.server_address[1]}", tracker=tracker)
        result = asyncio.run(procedure(server, evaluate, tracking=tracker is not None))
    finally:
        if tracker is not None:
            tracker.close()
        d.stop_loop.set()
        d.control.join(5)
        d.http.shutdown()
        d.http.server_close()
        recording = k.close()
        d.rehearser.close()
    result.update(scenario=scenario, evaluation=evaluated, recording=recording)
    save_summary(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=("nominal", "displaced", "missing"), default="displaced")
    parser.add_argument("--model", choices=("color", "edgetam"), default="color")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
