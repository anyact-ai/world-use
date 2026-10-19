"""Seeded starting state for one evaluation case, prepared before the camera policy starts."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import mujoco as mj
import numpy as np


def run(robot, params, record):
    folder = Path(params["output"])
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
    np.savez_compressed(folder / "initial.npz", qpos=data.qpos, qvel=data.qvel)
    return dict(verdict="inconclusive" if reasons else "pass", valid_start=not reasons, reasons=reasons,
                seed=params["seed"], parameters=options, simulation_settle_s=float(data.time),
                mujoco=mj.__version__, initial_state=record.artifact((folder / "initial.npz").read_bytes(), "npz"))
