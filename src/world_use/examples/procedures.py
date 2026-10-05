"""A reusable, verified block-transfer procedure using only public MCP tools.

The simulation runner alone owns truth. The procedure reads rendered images and
the same registered geometry, checked phases and outcome tools available to an LLM.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from ..cameras import Frame
from ..mcp_server import build
from ..observations import Perception
from ..vision import Observation
from ..vision_worker import TrackerProcess
from ..world import World
from .perception import SIZE, orange_mask
from .perception import run as simulation
from .pick_place import TARGET


class ColorFixture:
    """Rendered-color resegmentation for reproducible integration tests, not learned tracking."""
    ready = True

    def select(self, target, frame, **prompt):
        return self.update(target, frame)

    def update(self, target, frame):
        mask = orange_mask(frame)
        y, x = np.nonzero(mask)
        bbox = (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1) if len(x) else None
        return Observation(frame.camera, frame.id, frame.timestamp, bbox, None, mask if len(x) else None, 15)

    def forget(self, target):
        pass

    def close(self):
        pass


def procedure(client, *, condition="verify", tracker=None, evaluate=lambda: None):
    if condition != "verify":
        raise ValueError("this procedure always verifies outcomes")
    perception = Perception(client, tracker if tracker is not None else ColorFixture())
    server = build(client.url, perception=perception)
    try:
        return asyncio.run(_procedure(server, evaluate, "EdgeTAM" if tracker is not None else "RGB color fixture"))
    finally:
        perception.close()


async def _procedure(server, evaluate, selector):
    results: dict = dict(condition="verify", selector=selector, interface="MCP", observations=[], outcomes=[],
                   lift="unknown", placement="unknown", calls=[])

    async def tool(tool_name, **arguments):
        started = time.monotonic()
        reply = await server.call_tool(tool_name, arguments)
        results["calls"].append(dict(tool=tool_name, seconds=time.monotonic() - started))
        if reply.is_error:
            raise RuntimeError(str(reply.content))
        return reply.structured_content, reply

    async def data(name, **arguments):
        value, _ = await tool(name, **arguments)
        return value

    await tool("policy")
    status = await data("status")
    if status["session"]["mode"] != "simulation":
        raise ValueError("this example is only for simulation")
    await tool("card")
    world = World.from_dict(await data("world"))
    latest = None
    target = None

    def center(geometry):
        return world.from_base("work", geometry["components"]["center"]["value"])

    async def estimate(observation):
        nonlocal latest
        receipt = observation.get("evidence")
        if receipt is None:
            raise RuntimeError("target lost; dependent work ended")
        geometry = await data("fit_geometry", evidence=receipt["id"], kind="known_box",
                              size_m=SIZE.tolist(), frame="work")
        results["observations"].append(dict(evidence=receipt["id"], geometry=geometry["id"],
                                             valid=geometry["valid"], reason=geometry["reason"]))
        if not geometry["valid"]:
            raise RuntimeError(f"block geometry unknown: {geometry['reason']}")
        latest = geometry
        return geometry

    async def observe():
        update = await data("observe_targets", targets=[target], depth=True)
        return await estimate(update["observations"][0])

    async def assert_block(geometry):
        await tool("add_box", name="block", kind="object", center=center(geometry).tolist(), size=SIZE.tolist(),
                   source=f"known upright shape; geometry {geometry['id']}; evidence {geometry['evidence']}")

    def destination(geometry, offset):
        return dict(geometry=geometry["id"], component="center", offset_m=list(offset), offset_frame="work")

    async def execute(plan, *, dependent=True, effects=None):
        requires = ([dict(evidence=e, max_age_s=45) for e in latest["evidence"]]
                    if dependent and latest is not None else [])
        checked = await data("check", plan=plan, requires=requires, max_age_s=45, effects=effects)
        if "prepared" not in checked:
            raise RuntimeError(checked["text"])
        outcome = await data("run", plan_id=checked["prepared"]["id"], wait_s=120)
        results["outcomes"].append(outcome)
        if outcome["status"] != "done":
            raise RuntimeError(outcome.get("incident", f"phase {outcome['status']}"))
        return outcome

    # Task criteria are fixed here. A baseline reference is attached before each corresponding action.
    lift = dict(kind="lift", frame="work", min_up_m=.035, max_error_m=.015, max_age_s=45)
    placement = dict(kind="placement", frame="work", target_m=TARGET.tolist(), position_tolerance_m=.01,
                     support_z_m=.15, height_tolerance_m=.008, min_clearance_m=.08, min_aperture_mm=60, max_age_s=45)
    metadata, captured = await tool("camera_frame", camera="overhead", depth=True)
    image = next(item for item in captured.content if item.type == "image")
    rgb = Image.open(BytesIO(base64.b64decode(image.data))).convert("RGB")
    ys, xs = np.nonzero(orange_mask(Frame(rgb, "selection")))
    if not len(xs):
        results.update(reason="no block selected; no powered work", torque_off=not status["enabled"])
        evaluate()
        return results
    selected = await data("select_target", frame=metadata["frame"], label="block",
                           box=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
    target = selected["target"]
    try:
        geometry = await estimate(selected)
    except RuntimeError as e:
        results.update(reason=str(e), torque_off=not status["enabled"])
        evaluate()
        return results
    await assert_block(geometry)
    await tool("enable")
    try:
        # Establish orientation after unpowered settling, above the known tray.
        await execute([dict(do="line", up=.08), dict(do="gripper", aperture_mm=65),
                       dict(do="move_to", point="forward", jaws="left", within_deg=0)], dependent=False)
        high = (await data("status"))["tool"]["work"][2]
        await execute(dict(do="move_to", to=destination(geometry, [.02, 0, high - center(geometry)[2]])))
        geometry = await observe()
        await assert_block(geometry)
        await execute(dict(do="move_to", to=destination(geometry, [.02, 0, .02])))
        await execute(dict(do="grip", expect_mm=[35, 45]))
        before = await observe()
        lifted = await execute(dict(do="line", up=.06), effects=[dict(lift, before=before["id"])])
        after = await observe()
        verified = await data("verify_effect", job=lifted["id"], after=after["id"])
        results["lift"] = verified["status"]
        if verified["status"] != "pass":
            raise RuntimeError(f"lift not verified: {verified['status']}")
        tool_after = world.from_base("work", np.asarray(after["tool"])[:3, 3])
        # Stable carry relation is an explicit task assumption, checked by the short lift above.
        to = TARGET + tool_after - center(after)
        to[2] = tool_after[2]
        await execute(dict(do="move_to", to=destination(after, to - center(after))))
        placed = await execute([dict(do="line", up=-.05), dict(do="gripper", aperture_mm=65), dict(do="line", up=.08)],
                               effects=[dict(placement, before=after["id"])])
        after = await observe()
        verified = await data("verify_effect", job=placed["id"], after=after["id"])
        results["placement"] = verified["status"]
    except Exception as e:
        results["reason"] = str(e)
        await tool("stop", reason="procedure expectation failed")
        current = (await data("status"))["tool"]["work"]
        recovery = ([dict(do="move_to", to=[current[0], current[1], .22])]
                    if abs(current[2] - .22) > .001 else [])
        await execute([*recovery, dict(do="gripper", aperture_mm=65), dict(do="line", up=.08)], dependent=False)
    finally:
        evaluate()
        await tool("home_route", steps=[], note="known clear tray; return from above turn height")
        returned = await data("home", wait_s=120, request_id=status["session_id"] + ":procedure-return")
        results["return_outcome"] = returned
        if returned["status"] == "done":
            await tool("release")
    results["torque_off"] = not (await data("status"))["enabled"]
    history = await data("inspect_run", limit=100)
    results["history"] = dict(next_cursor=history["next_cursor"], more=history["more"], missed=history["missed"])
    await tool("record", note="MCP procedure outcome",
               context=dict(lift=results["lift"], placement=results["placement"]))
    return results


def run(output, *, scenario="shifted", model="color", device="cpu"):
    return simulation(output, scenario=scenario, condition="verify", model=model, device=device,
                      procedure_fn=procedure, tracker_factory=TrackerProcess)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=("nominal", "shifted", "displaced", "missing"), default="shifted")
    parser.add_argument("--model", choices=("color", "edgetam"), default="color")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
