"""Exercise the pinned CPU EdgeTAM model and live MuJoCo procedure; save full evidence.

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

from world_use.cameras import Frame
from world_use.examples.perception import CONDITIONS, orange_mask, run
from world_use.examples.procedures import run as run_mcp
from world_use.records import inspect
from world_use.vision import MODEL, REVISION, EdgeTAM
from world_use.vision_worker import TrackerProcess


def replay(image_path: Path, output: Path) -> dict:
    """Check point/box prompts and forward history past the model's retention window."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    expected = orange_mask(Frame(image, "replay"))
    ys, xs = np.nonzero(expected)
    assert len(xs), "the replay fixture must contain the visible block"
    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    output.mkdir()
    image.save(output / "rgb.png")
    Image.fromarray(expected).save(output / "color-reference.png")
    samples = []
    with EdgeTAM(device="cpu", max_age_s=15) as tracker:
        for prompt in ({"point": (float(np.median(xs)), float(np.median(ys)))}, {"box": box}):
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
                # A point can select the shadow too; only the task's box constrains the object outline.
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


def worker_replay(image_path: Path) -> dict:
    """Two independent prompts share loaded weights without sharing target history."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    expected = orange_mask(Frame(image, "worker-replay"))
    ys, xs = np.nonzero(expected)
    bounds = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    samples = []
    with TrackerProcess(device="cpu", max_age_s=15) as worker:
        for target in ("first", "second"):
            result = worker.select(target, Frame(image.copy(), "worker-replay"), box=bounds)
            assert result.status == "tracked"
        for _ in range(3):
            frame = Frame(image.copy(), "worker-replay")
            for target in ("first", "second"):
                result = worker.update(target, frame)
                assert result.status == "tracked"
                overlap = float(np.count_nonzero(result.mask & expected) / np.count_nonzero(result.mask | expected))
                assert overlap > .75, (target, overlap)
                samples.append(dict(target=target, frame=frame.id, iou=overlap))
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
        cases = [(condition, "displaced") for condition in CONDITIONS] + [("verify", "missing")]
        for condition, scenario in cases:
            name = f"{condition}-{scenario}"
            folder = root / name
            result = run(folder, condition=condition, scenario=scenario, model="edgetam", device="cpu")
            record = inspect(folder)
            report["runs"][name] = dict(result=result, telemetry=record["summary"])
            print(json.dumps(dict(case=name, evaluation=result["evaluation"], lift=result["lift"],
                                  placement=result["placement"], torque_off=result["torque_off"],
                                  reason=result.get("reason"))), flush=True)
            assert result["torque_off"], result
            assert record["closed"] and "recording_lost" not in record["summary"], record["summary"]
            for observation in result["observations"]:
                evidence = folder / "perception" / observation["evidence"]
                assert all((evidence / file).is_file()
                           for file in ("measurement.json", "rgb.png", "overlay.png", "surfaces.npz"))
            if scenario == "missing":
                assert not result["outcomes"] and record["summary"]["powered_s"] == 0, result
                assert not result["evaluation"]["success"], result
            elif condition == "verify":
                assert result["evaluation"]["success"], result
                assert result["lift"] == result["placement"] == "pass", result
            # The other conditions are ablations: measure their misses rather than require success.

        verified = root / "verify-displaced"
        first = report["runs"]["verify-displaced"]["result"]["observations"][0]["evidence"]
        report["replay"] = replay(verified / "perception" / first / "rgb.png", root / "replay")
        report["worker_replay"] = worker_replay(verified / "perception" / first / "rgb.png")
        for scenario in ("displaced", "missing"):
            folder = root / f"mcp-{scenario}"
            result = run_mcp(folder, scenario=scenario, model="edgetam", device="cpu")
            record = inspect(folder)
            report["runs"][f"mcp-{scenario}"] = dict(result=result, telemetry=record["summary"])
            print(json.dumps(dict(case=f"mcp-{scenario}", evaluation=result["evaluation"],
                                  lift=result["lift"], placement=result["placement"],
                                  torque_off=result["torque_off"], reason=result.get("reason"))), flush=True)
            assert result["torque_off"] and record["closed"] and "recording_lost" not in record["summary"]
            if scenario == "displaced":
                assert result["lift"] == result["placement"] == "pass", result
                assert result["evaluation"]["success"], result
            else:
                assert not result["outcomes"] and record["summary"]["powered_s"] == 0, result
        from rerun.chunk import RrdReader
        for name in ("verify-displaced", "mcp-displaced"):
            rrd = root / f"{name}.rrd"
            subprocess.run([sys.executable, "-m", "world_use", "view", str(root / name), "--out", str(rrd)], check=True)
            recording = RrdReader(rrd)
            assert recording.recordings() and recording.blueprints()
        report["passed"] = True
    finally:
        (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Real EdgeTAM + MuJoCo integration passed; evidence: {root}")


if __name__ == "__main__":
    main()
