"""Fit two recorded MIT pushes, freeze the candidate, then evaluate three reserved trials."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from measured import FRICTIONS, HERE, REFERENCE, SOURCES, compare, load, motion_error, simulate, summarize
from validate import LIMITS

from world_use.bodies.mujoco_scene import mj
from world_use.tasks import digest, json_bytes, write_new

TRIAL_SOURCES = (*SOURCES, "trials.py", "rect1-reference.json")


def verify_sources(sources):
    changed = [name for name, sha in sources.items() if digest((HERE / name).read_bytes()) != sha]
    if changed:
        raise ValueError(f"candidate sources changed during evaluation: {', '.join(changed)}")


def run(input_dir, output):
    manifest = json.loads((HERE / "rect1-reference.json").read_text())
    geometry = REFERENCE["rect1_geometry"]
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name in TRIAL_SOURCES:
        payload = (HERE / name).read_bytes()
        sources[name] = digest(payload)
        write_new(output / "source" / name, payload)
    # No held-out measurements are loaded until the model and selection are frozen.
    fit = {case["name"]: load(input_dir / case["path"], case["sha256"])
           for case in manifest["cases"] if case["role"] == "fit"}
    scores = []
    for friction in FRICTIONS:
        errors = {}
        for name, (tip, obj, initial, duration, _) in fit.items():
            samples, _ = simulate(tip, initial, duration, friction, geometry)
            errors[name] = motion_error(obj, samples, geometry)
        scores.append(dict(friction=friction, errors_m=errors, worst_error_m=max(errors.values())))
    chosen = min(scores, key=lambda row: row["worst_error_m"])["friction"]
    write_new(output / "frozen-candidate.json", json_bytes(dict(
        selection=manifest, sources=sources, geometry=geometry, probe=REFERENCE["probe"],
        assumptions=REFERENCE["assumptions"], mujoco_version=mj.__version__, tolerances=LIMITS,
        selection_rule="minimum worst labelled-vertex error across the two fit trials; fixed grid",
        scores=scores, chosen_friction=chosen, initial_friction=0.4,
        held_out_access="no reserved measurements read before writing this candidate")))
    results = {}
    for case in manifest["cases"]:
        verify_sources(sources)
        name, split = case["name"], case["role"]
        trajectory = fit[name] if name in fit else load(input_dir / case["path"], case["sha256"])
        origin = f"{case['source_url']} (member: {case['archive_member']})"
        results[name] = dict(split=split, candidates={})
        for candidate, friction in (("initial", 0.4), ("refined", chosen)):
            report = compare(output / candidate / name, input_dir / case["path"], trajectory, friction,
                             geometry=geometry, split=split, origin_url=origin, sources=TRIAL_SOURCES)
            verify_sources(sources)
            results[name]["candidates"][candidate] = summarize(report, split)
        # Retain each completed case even if a later run fails.
        write_new(output / "results" / f"{name}.json", json_bytes(results[name]))
    write_new(output / "comparison.json", json_bytes(results))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="directory containing the five pinned rect1 JSON files")
    parser.add_argument("output", type=Path, help="fresh result directory")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
