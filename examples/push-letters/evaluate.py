"""Run fixed seeded camera cases, or recreate a named case from its saved sources and inputs."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mujoco as mj
import numpy as np

from world_use.evaluation import rerun, run_suite
from world_use.tasks import digest, json_bytes


def run(robot, params, record):
    # This setup program constructs the simulator. Only episode.observe supplies policy feedback.
    from episode import run as episode
    folder = Path(params["output"]).parent / "scene"
    folder.mkdir()
    for name, path in params["inputs"].items():
        shutil.copyfile(record.input_file(path), folder / name)
    model = mj.MjModel.from_binary_path(str(folder / "scene.mjb"))
    data = mj.MjData(model)
    with np.load(folder / "initial.npz") as state:
        data.qpos[:], data.qvel[:] = state["qpos"], state["qvel"]
    data.ctrl[:] = data.qpos[:7]
    generator = np.random.default_rng(params["seed"])
    task = json.loads((folder / "task.json").read_text())
    options = params["parameters"]
    for letter in task["letters"]:
        joint = model.joint(letter["name"] + "_free").id
        qadr = model.jnt_qposadr[joint]
        data.qpos[qadr:qadr + 2] += generator.uniform(-options["jitter_m"], options["jitter_m"], 2)
        yaw = generator.uniform(-options["jitter_deg"], options["jitter_deg"]) * np.pi / 180
        rotated = np.empty(4)
        mj.mju_mulQuat(rotated, np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), data.qpos[qadr + 3:qadr + 7])
        data.qpos[qadr + 3:qadr + 7] = rotated
    for _ in range(800):
        mj.mj_step(model, data)
    reasons = []
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        reasons.append("non-finite initial state")
    if any(c.dist < -.001 for c in data.contact):
        reasons.append("initial penetration exceeds 1 mm")
    if any(w.number for w in data.warning):
        reasons.append("initial solver warning")
    initial = dict(valid=not reasons, reasons=reasons, seed=params["seed"], parameters=options,
                   simulation_settle_s=float(data.time), mujoco=mj.__version__,
                   model_sha256=digest((folder / "scene.mjb").read_bytes()))
    (folder / "starting-state.json").write_bytes(json_bytes(initial))
    np.savez_compressed(folder / "initial.npz", qpos=data.qpos, qvel=data.qvel)
    record.input_file(folder / "starting-state.json")
    record.input_file(folder / "initial.npz")
    if reasons:
        return dict(verdict="inconclusive", reasons=reasons)
    result = episode(folder, Path(params["output"]), model_file="scene.mjb",
                     max_pushes=options["max_pushes"],
                     feedback="camera" if params["observation_mode"] == "camera" else "state")
    return dict(verdict="pass" if result["success"] else "fail", reason=result["error"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, help="Existing pushing scene directory, including XML and initial state")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation", type=Path, help="Reference validation report for this exact scene identity")
    parser.add_argument("--case", choices=["development", "held-out"], default="development")
    parser.add_argument("--rerun", type=Path, help="Existing evaluation suite to recreate")
    parser.add_argument("--case-name", help="Named recorded case for --rerun")
    parser.add_argument("--privileged", action="store_true",
                        help="State feedback diagnostic; excluded from camera comparison")
    args = parser.parse_args()
    if args.rerun:
        if not args.case_name:
            parser.error("--rerun requires --case-name")
        report = rerun(args.rerun, args.case_name, args.output)
    else:
        if args.scene is None:
            parser.error("--scene is required")
        # The compiled model contains meshes. Preserve XML/asset provenance too; MJB remains version-specific.
        import tempfile

        from world_use.scene_validation import scene_identity
        with tempfile.TemporaryDirectory(prefix="world-use-evaluation-") as staging:
            model = mj.MjModel.from_xml_path(str(args.scene / "scene.xml"))
            binary = Path(staging) / "scene.mjb"
            mj.mj_saveModel(model, str(binary))
            inputs = {name: args.scene / name for name in ("scene.xml", "initial.npz", "task.json")}
            inputs["scene.mjb"] = binary
            identity = scene_identity(inputs, dict(mujoco=mj.__version__,
                camera=dict(position=[.49, 0, 1.05], target=[.49, 0, 0], fov_deg=46, size=[1000, 1000]),
                initial_variation=dict(jitter_m=[.002, .008], jitter_deg=[2, 7])))
            held_out = args.case == "held-out"
            case = dict(name="shifted-seed29" if held_out else "nominal-seed17", seed=29 if held_out else 17,
                        role="held_out" if held_out else "development",
                        parameters=dict(jitter_m=.008 if held_out else .002,
                                        jitter_deg=7 if held_out else 2, max_pushes=20))
            root = Path(__file__).parent
            report = run_suite(Path(__file__), root / "evaluation_audit.py", args.output, cases=[case],
                inputs=inputs, scene_revision=identity["revision"], validation=args.validation,
                sources=[root / name for name in ("episode.py", "adapter.py", "policy.py", "scene.py", "audit.py")],
                evaluator_sources=[root / "audit.py"],
                observation_mode="privileged_diagnostic" if args.privileged else "camera")
    print(json.dumps(report, indent=2))
    return 0 if all(t["status"] == "completed" and t["task"]["verdict"] == "pass" for t in report["trials"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
