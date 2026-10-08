"""Recompute saved planar alignments offline; never connect to a robot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from world_use.geometry import align_planar


def run(robot, params, record):
    if robot is not None:
        raise ValueError("alignment analysis takes saved inputs only")
    source = record.input_file(params["geometry"])
    entries = json.loads(source.read_text())
    if not entries:
        raise ValueError("no saved alignments")
    comparisons = []
    for entry in entries:
        if entry["function"] != "world_use.geometry.align_planar":
            raise ValueError("unsupported calculation in geometry.json")
        result = record.call(align_planar, np.asarray(entry["source"]), np.asarray(entry["target"]),
                             np.asarray(entry["tool"]), max_error_m=entry["max_error_m"])
        original = entry["result"]
        matches = result["valid"] == original["valid"]
        for key in ("yaw_delta_deg", "translation_xy_m", "errors_m", "max_error_m", "height_error_m", "tool_pose"):
            left, right = result[key], original[key]
            matches &= left is right if left is None or right is None else bool(
                np.allclose(left, right, rtol=1e-9, atol=1e-10))
        comparisons.append(dict(matches=bool(matches), result=result))
    return dict(verdict="pass" if all(c["matches"] for c in comparisons) else "fail",
                operation="saved alignment comparison; no robot execution",
                comparisons=comparisons)


def main():
    from run_task import launch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    return launch(Path(__file__), args.output, parameters=dict(geometry=str(args.geometry.resolve())))


if __name__ == "__main__":
    raise SystemExit(main())
