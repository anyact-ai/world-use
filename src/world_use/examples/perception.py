"""RGB-D block transfer with evidence prerequisites, running against a live MuJoCo daemon.

The procedure receives only a Client, known shape and tray geometry. The runner alone owns
the simulator's truth and independent evaluator. This is a constrained example, not a grasp generator.
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .. import Kernel, RealClock, World, bodies, cameras
from ..client import Client, DaemonError
from ..config import load_workcell
from ..daemon import Daemon, apply_workcell
from ..perception import measure
from ..recorder import save_summary
from .pick_place import TARGET, success

SIZE = np.array([.04, .04, .10])
CONDITIONS = ("nominal", "depth", "track", "verify")


def orange_mask(frame):
    """Fixture selector using only rendered RGB; no instance IDs, depth, or hidden coordinates."""
    rgb = np.asarray(frame.image).astype(float)
    return (rgb[..., 0] > 65) & (rgb[..., 0] > 1.6 * rgb[..., 1]) & (rgb[..., 1] > 1.3 * rgb[..., 2])


def upright_box(measurement, world):
    """Task assumption: known dimensions, upright, work-aligned, substantially visible top face."""
    if not measurement.valid:
        return None
    points = np.array([world.from_base("work", p) for p in measurement.points])
    top = np.quantile(points[:, 2], .95)
    surface = points[np.abs(points[:, 2] - top) < .003]
    if len(surface) < 16:
        return None
    lo, hi = np.quantile(surface[:, :2], [.02, .98], axis=0)
    # Missing sides, merged masks and occlusion are unknown, not an invented object centre.
    if np.any(hi - lo < SIZE[:2] * .65) or np.any(hi - lo > SIZE[:2] * 1.15):
        return None
    return np.array([*(lo + hi) / 2, top - SIZE[2] / 2])


def lift_result(before, after, tool_delta):
    if before is None or after is None:
        return "unknown"
    moved = after - before
    return "pass" if moved[2] > .035 and np.linalg.norm(moved - tool_delta) < .015 else "fail"


def placement_result(center, tool, aperture_mm):
    if center is None or aperture_mm is None:
        return "unknown"
    return "pass" if (np.linalg.norm(center - TARGET) < .01 and tool[2] - center[2] > .08
                      and abs(center[2] - .20) < .008 and aperture_mm >= 60) else "fail"


def procedure(c: Client, *, condition="verify", tracker=None, evaluate=lambda: None):
    """One attempt, no automatic re-grasps. All motion uses checked, finite phases on a known clear tray."""
    if c.status()["session"]["mode"] != "simulation":
        raise ValueError("this example is only for simulation")
    world = World.from_dict(c.world())
    results: dict = dict(condition=condition, selector="EdgeTAM" if tracker is not None else "RGB color fixture",
                   observations=[], outcomes=[], lift="unknown", placement="unknown")
    seeded = False
    latest: dict = {}

    def observe():
        nonlocal seeded, latest
        frame = c.frame("overhead", depth=True)
        started = time.monotonic()
        mask = orange_mask(frame)
        if tracker is not None:
            if not seeded:
                y, x = np.nonzero(mask)
                if len(x):
                    observation = tracker.select(frame, box=[int(x.min()), int(y.min()),
                                                            int(x.max()) + 1, int(y.max()) + 1])
                    seeded = True
                else:
                    observation = None
            else:
                observation = tracker.update(frame)
            mask = (observation.mask if observation is not None and observation.status == "tracked"
                    else np.zeros_like(mask))
        measurement = measure(frame, mask=mask, target="block")
        latest = c.record(evidence=measurement)
        center = upright_box(measurement, world)
        if frame.tool is None:
            raise ValueError("this procedure needs a synchronized simulation tool pose")
        tool = world.from_base("work", frame.tool[:3, 3])
        results["observations"].append(dict(evidence=latest["id"], inference_s=time.monotonic() - started,
                                             center=None if center is None else center.tolist(),
                                             reason=measurement.reason, age_s=frame.age_s))
        return center, tool, frame

    def execute(spec, dependent=True):
        requires = ([dict(evidence=latest["id"], max_age_s=45)]
                    if dependent and condition != "nominal" and latest is not None else [])
        result = c.run(spec, wait=120, requires=requires)
        results["outcomes"].append(result)
        if result["status"] != "done":
            raise RuntimeError(result.get("incident", f"phase {result['status']}"))

    def assert_block(center):
        source = ("fixed nominal estimate" if condition == "nominal"
                  else f"known upright shape; evidence {latest['id']}")
        c.box("block", "object", center, SIZE, source=source)

    center, _, _ = observe()
    if condition == "nominal":
        center = np.array([.34, .03, .20])
    elif center is None:
        results["reason"] = "block geometry unknown; no dependent phase"
        results["torque_off"] = not c.status()["enabled"]
        evaluate()
        return results
    assert_block(center)
    # The tool frame is at the fingertips; this known upright grasp uses the inside of the pads.
    offset = np.array([.02, 0, 0])
    c.enable()
    try:
        # The unpowered wrist can settle during inference. Establish the grasp orientation above the tray.
        execute([{"do": "line", "up": .08}, {"do": "gripper", "aperture_mm": 65},
                 {"do": "move_to", "point": "forward", "jaws": "left", "within_deg": 0}], dependent=False)
        high = c.status()["tool"]["work"][2]
        execute({"do": "move_to", "to": [float(center[0] + offset[0]), float(center[1] + offset[1]), high]})
        if condition in ("track", "verify"):
            center, _, _ = observe()
            if center is None:
                raise RuntimeError("target lost or shape ambiguous before descent")
            assert_block(center)
        goal = center + offset
        goal[2] = center[2] + .02
        execute({"do": "move_to", "to": goal.tolist()})
        execute({"do": "grip", "expect_mm": [35, 45]})
        before, tool_before, _ = (observe() if condition == "verify"
                                  else (center, np.array(c.status()["tool"]["work"]), None))
        execute({"do": "line", "up": .06})
        if condition in ("track", "verify"):
            after, tool_after, _ = observe()
            if condition == "verify":
                results["lift"] = lift_result(before, after, tool_after - tool_before)
                c.record(note="visual lift verification", context=dict(result=results["lift"],
                         evidence=latest["id"], assumptions="upright block; substantially visible top"))
                if results["lift"] != "pass":
                    raise RuntimeError("lift not visually verified")
            if after is None:
                raise RuntimeError("target geometry unknown after lift")
            target = TARGET + tool_after - after
        else:
            target = TARGET + np.array(c.status()["tool"]["work"]) - (center + [0, 0, .06])
        target[2] = c.status()["tool"]["work"][2]
        execute({"do": "move_to", "to": target.tolist()})
        execute([{"do": "line", "up": -.05}, {"do": "gripper", "aperture_mm": 65},
                 {"do": "line", "up": .08}])
        if condition == "verify":
            placed, withdrawn, _ = observe()
            aperture = c.status()["gripper"]["aperture_mm"]
            results["placement"] = placement_result(placed, withdrawn, aperture)
            c.record(note="visual placement verification", context=dict(result=results["placement"],
                     evidence=latest["id"], aperture_mm=aperture, assumptions="known tray and block size"))
    except (RuntimeError, DaemonError) as e:
        results["reason"] = str(e)
        # All task waypoints stay over this known open tray. Put down before opening; never release at height.
        c.stop("perception attempt ended")
        current = c.status()["tool"]["work"]
        execute([{"do": "move_to", "to": [current[0], current[1], .22]},
                 {"do": "gripper", "aperture_mm": 65}, {"do": "line", "up": .08}], dependent=False)
    finally:
        evaluate()   # independent after-release evaluator, before homing changes the withdrawal measurement
        c.home_route([], "known open tray; return from above turn height")
        returned = c.home(wait=120)
        results["return_outcome"] = returned
        if returned["status"] == "done":
            c.release()
    results["torque_off"] = not c.status()["enabled"]
    return results


def run(output: Path, *, scenario="shifted", condition="verify", model="color", device="cpu"):
    """Runner owns truth, continuous physics, lifecycle and evaluation; none is passed to the procedure."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"use an empty output directory: {output}")
    cell = load_workcell(Path("block"))
    if scenario == "shifted":
        cell["box"][1]["center"] = [.35, .02, .20]
    elif scenario == "displaced":
        cell["box"][1]["center"] = [.34, .10, .20]
    elif scenario == "missing":
        cell["box"].pop(1)
    elif scenario != "nominal":
        raise ValueError("unknown scenario")
    for box in cell["box"]:
        if box["kind"] == "object":
            box["known"] = False
    if condition not in CONDITIONS or model not in ("color", "edgetam"):
        raise ValueError("unknown condition or selector")
    # Load inference before starting a powered session.
    if model == "edgetam":
        from ..vision import EdgeTAM
        selector = EdgeTAM(device=device, max_age_s=15)
    else:
        selector = nullcontext(None)
    with selector as tracker:
        world, truth = World(), World()
        body = bodies.make("sim", truth, q=np.radians(cell["body_options"]["start_deg"]), gripper=1.0)
        k = Kernel(body, world, RealClock(100), run_dir=output)
        k.connect()
        truth.frames.update(world.frames)
        apply_workcell(cell, k, truth)
        lens = cameras.View.look_at(world.to_base("work", [.34, 0, .90]),
                                    world.to_base("work", [.34, 0, .15]), size=(512, 512), fov_deg=40)
        d = Daemon(k, port=0, cams={"overhead": cameras.SimCamera("overhead", lens, body)})
        d.start()
        c = Client(f"http://127.0.0.1:{d.http.server_address[1]}")
        evaluated = {}

        def evaluate():
            with k.lock, body.lock:
                evaluated.update(success(k, truth))
            k.emit("task_result", "independent simulator evaluation", **evaluated)

        try:
            result = procedure(c, condition=condition, tracker=tracker, evaluate=evaluate)
            result.update(scenario=scenario, evaluation=evaluated)
            c.shutdown()
            result["recording"] = k.save_record()
            save_summary(output / "result.json", result)
            return result
        finally:
            d.stop_loop.set()
            d.control.join(5)
            d.http.shutdown()
            d.http.server_close()
            if not k.evidence_closed:
                k.close()
            d.rehearser.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=("nominal", "shifted", "displaced", "missing"), default="shifted")
    parser.add_argument("--condition", choices=CONDITIONS, default="verify")
    parser.add_argument("--model", choices=("color", "edgetam"), default="color")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
