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
from world_use.records import inspect
from world_use.vision import MODEL, REVISION, EdgeTAM


def replay(image_path: Path) -> dict:
    """Check point/box prompts and forward history past the model's retention window."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    expected = orange_mask(Frame(image, "replay"))
    ys, xs = np.nonzero(expected)
    assert len(xs), "the replay fixture must contain the visible block"
    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    samples = []
    with EdgeTAM(device="cpu", max_age_s=15) as tracker:
        for prompt in ({"point": (float(np.median(xs)), float(np.median(ys)))}, {"box": box}):
            for index in range(20):
                frame = Frame(image.copy(), "replay")
                started = time.monotonic()
                observation = tracker.select(frame, **prompt) if index == 0 else tracker.update(frame)
                elapsed = time.monotonic() - started
                assert observation.status == "tracked", observation.to_dict()
                assert observation.mask is not None
                overlap = float(np.count_nonzero(observation.mask & expected)
                                / np.count_nonzero(observation.mask | expected))
                assert overlap > .75, f"fixture mask drifted: IoU={overlap:.3f}"
                # Exercise the real upstream session, which the offline contract tests cannot load.
                session = tracker._session
                assert not session.processed_frames
                for outputs in session.output_dict_per_obj.values():
                    assert len(outputs["non_cond_frame_outputs"]) <= tracker._recent
                    assert len(outputs["cond_frame_outputs"]) == 1
                samples.append(dict(prompt=next(iter(prompt)), index=index, inference_s=elapsed, iou=overlap))
    return dict(frames=samples)


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
            print(json.dumps(dict(case=name, **report["runs"][name])), flush=True)
            assert result["torque_off"], result
            assert record["closed"] and "recording_lost" not in record["summary"], record["summary"]
            for observation in result["observations"]:
                evidence = folder / "perception" / observation["evidence"]
                assert all((evidence / file).is_file()
                           for file in ("measurement.json", "rgb.png", "overlay.png", "surfaces.npz"))
            expected_success = condition != "nominal" and scenario != "missing"
            assert result["evaluation"]["success"] == expected_success, result
            if scenario == "missing":
                assert not result["outcomes"] and record["summary"]["powered_s"] == 0, result
            elif condition == "verify":
                assert result["lift"] == result["placement"] == "pass", result

        verified = root / "verify-displaced"
        first = report["runs"]["verify-displaced"]["result"]["observations"][0]["evidence"]
        report["replay"] = replay(verified / "perception" / first / "rgb.png")
        rrd = root / "verify-displaced.rrd"
        subprocess.run([sys.executable, "-m", "world_use", "view", str(verified), "--out", str(rrd)], check=True)
        from rerun.chunk import RrdReader
        recording = RrdReader(rrd)
        assert recording.recordings() and recording.blueprints()
        report["passed"] = True
    finally:
        (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Real EdgeTAM + MuJoCo integration passed; evidence: {root}")


if __name__ == "__main__":
    main()
