"""Exercise the pinned CPU EdgeTAM model through the live MuJoCo example; save everything it produced.

Requires the vision and rerun extras and access to the public Hugging Face checkpoint.
This is a constrained integration check, not an object-tracking or robotics benchmark.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image

from world_use import Kernel, VirtualClock, World, bodies, cameras
from world_use.cameras import Frame
from world_use.config import load_workcell
from world_use.daemon import apply_workcell
from world_use.examples.perception import run, seen
from world_use.records import inspect
from world_use.vision import MODEL, REVISION, EdgeTAM
from world_use.vision_worker import MAX_TARGETS, TrackerProcess


def picture() -> Image.Image:
    """The example's overhead view of the displaced block, rendered without a daemon."""
    cell = load_workcell(Path("block"))
    cell["box"][1]["center"] = [.34, .10, .20]
    world, truth = World(), World()
    body = bodies.make("sim", truth, q=np.radians(cell["body_options"]["start_deg"]), gripper=1.0)
    k = Kernel(body, world, VirtualClock(100))
    k.connect()
    truth.frames.update(world.frames)
    apply_workcell(cell, k, truth)
    lens = cameras.View.look_at(world.to_base("work", [.34, 0, .90]), world.to_base("work", [.34, 0, .15]),
                                size=(512, 512), fov_deg=40)
    try:
        return cameras.SimCamera("overhead", lens, body).picture(k)
    finally:
        k.close()


def orange(image) -> np.ndarray:
    rgb = np.asarray(image).astype(float)
    return (rgb[..., 0] > 65) & (rgb[..., 0] > 1.6 * rgb[..., 1]) & (rgb[..., 1] > 1.3 * rgb[..., 2])


def replay(image: Image.Image, output: Path) -> dict:
    """Point and box prompts, and forward history past the model's retention window."""
    expected = orange(image)
    point, box = seen(image)
    output.mkdir()
    image.save(output / "rgb.png")
    Image.fromarray(expected).save(output / "color-reference.png")
    samples = []
    with EdgeTAM(device="cpu", max_age_s=15) as tracker:
        for prompt in ({"point": tuple(point)}, {"box": box}):
            seed = None
            for index in range(20):
                frame = Frame(image.copy(), "replay")
                started = time.monotonic()
                observation = tracker.select(frame, **prompt) if index == 0 else tracker.update(frame)
                elapsed = time.monotonic() - started
                assert observation.status == "tracked", observation.to_dict()
                assert observation.mask is not None
                overlap = float(np.count_nonzero(observation.mask & expected)
                                / np.count_nonzero(observation.mask | expected))
                if seed is None:
                    seed = observation.mask.copy()
                stability = float(np.count_nonzero(observation.mask & seed)
                                  / np.count_nonzero(observation.mask | seed))
                assert stability > .9, f"static selection drifted: IoU={stability:.3f}"
                # A point can select the shadow too; only the box constrains the object outline.
                if "box" in prompt:
                    assert overlap > .75, f"fixture mask drifted: IoU={overlap:.3f}"
                if index in (0, 19):
                    Image.fromarray(observation.mask).save(output / f"{next(iter(prompt))}-{index}.png")
                # Exercise the real upstream session, which the offline contract tests cannot load.
                session = tracker._session
                assert not session.processed_frames
                for outputs in session.output_dict_per_obj.values():
                    assert len(outputs["non_cond_frame_outputs"]) <= tracker._recent
                    assert len(outputs["cond_frame_outputs"]) == 1
                samples.append(dict(prompt=next(iter(prompt)), index=index, inference_s=elapsed,
                                    iou=overlap, seed_iou=stability))
    return dict(frames=samples)


def worker_replay(image: Image.Image) -> dict:
    """Independent targets share loaded weights but not history; a full worker takes a name again, not a new one."""
    expected = orange(image)
    _, box = seen(image)
    samples = []
    with TrackerProcess(device="cpu", max_age_s=15) as worker:
        names = [f"target-{i}" for i in range(MAX_TARGETS)]
        for name in names:
            assert worker.select(name, Frame(image.copy(), "worker-replay"), box=box).status == "tracked"
        assert worker.select(names[0], Frame(image.copy(), "worker-replay"), box=box).status == "tracked"
        try:
            worker.select("one too many", Frame(image.copy(), "worker-replay"), box=box)
            raise AssertionError("a full worker accepted another target")
        except ValueError as e:
            assert "at most" in str(e), e
        for _ in range(3):
            frame = Frame(image.copy(), "worker-replay")
            for name in names[:2]:
                result = worker.update(name, frame)
                assert result.status == "tracked"
                overlap = float(np.count_nonzero(result.mask & expected) / np.count_nonzero(result.mask | expected))
                assert overlap > .75, (name, overlap)
                samples.append(dict(target=name, frame=frame.id, iou=overlap))
    return dict(observations=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    root = parser.parse_args().output
    root.mkdir(parents=True, exist_ok=False)
    report = dict(model=MODEL, revision=REVISION, device="cpu", python=platform.python_version(),
                  packages={name: version(name) for name in ("torch", "torchvision", "transformers", "timm")},
                  runs={}, passed=False)
    try:
        for scenario in ("displaced", "missing"):
            folder = root / scenario
            result = run(folder, scenario=scenario, model="edgetam", device="cpu")
            record = inspect(folder)
            report["runs"][scenario] = dict(result=result, telemetry=record["summary"])
            print(json.dumps(dict(case=scenario, evaluation=result["evaluation"], lift=result["lift"],
                                  placement=result["placement"], torque_off=result["torque_off"],
                                  reason=result.get("reason"))), flush=True)
            assert result["torque_off"] and record["closed"], result
            assert "recording_lost" not in record["summary"], record["summary"]
            for m in result["measurements"]:
                assert all((folder / "perception" / f"{m['id']}{suffix}").is_file() for suffix in (".png", ".npz"))
            if scenario == "missing":
                assert not result["outcomes"] and record["summary"]["powered_s"] == 0, result
                assert not result["evaluation"]["success"], result
            else:
                assert all(m["tracking"] == "tracked" for m in result["measurements"]), result["measurements"]
                assert result["lift"] == result["placement"] == "pass", result
                assert result["evaluation"]["success"], result
        image = picture()
        report["replay"] = replay(image, root / "replay")
        report["worker_replay"] = worker_replay(image)
        from rerun.chunk import RrdReader
        rrd = root / "displaced.rrd"
        subprocess.run([sys.executable, "-m", "world_use", "view", str(root / "displaced"), "--out", str(rrd)],
                       check=True)
        recording = RrdReader(rrd)
        assert recording.recordings() and recording.blueprints()
        report["passed"] = True
    finally:
        (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Real EdgeTAM + MuJoCo integration passed; evidence: {root}")


if __name__ == "__main__":
    main()
