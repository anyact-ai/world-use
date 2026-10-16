"""Privileged outcome evaluator, run after the camera policy has exited."""
from __future__ import annotations

import json
from pathlib import Path

import mujoco as mj
import numpy as np
from audit import audit

from world_use.evaluation import held_for


def run(robot, params, record):
    folder = Path(params["episode"])
    initial = json.loads(record.input_file(folder.parent / "scene/starting-state.json").read_text())
    if not initial["valid"]:
        return dict(verdict="inconclusive", valid_start=False, reasons=initial["reasons"], interventions=[])
    physical = audit(folder)
    replay = np.load(folder / "replay.npz")
    model = mj.MjModel.from_binary_path(str(folder / "scene.mjb"))
    task = json.loads((folder / "task.json").read_text())
    condition = np.ones(len(replay["simulation_time"]), dtype=bool)
    for letter in task["letters"]:
        joint = model.joint(letter["name"] + "_free").id
        qadr, vadr = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        q = replay["qpos"][:, qadr:qadr + 7]
        w, x, y, z = q[:, 3:].T
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        angle = (yaw - letter["target_yaw"] + np.pi) % (2 * np.pi) - np.pi
        if letter["letter"] == "O":
            angle = (angle + np.pi / 2) % np.pi - np.pi / 2
        condition &= np.linalg.norm(q[:, :2] - letter["target_xy"], axis=1) < .005
        condition &= np.abs(angle) < np.radians(5)
        condition &= np.linalg.norm(replay["qvel"][:, vadr:vadr + 3], axis=1) < .001
    settled = held_for(replay["simulation_time"], condition, seconds=1.9, max_gap_s=.1)
    reasons = []
    if not physical["passed"]:
        reasons.append("independent contact, support, pose or cleanup audit failed")
    if not settled:
        reasons.append("goal conditions did not remain true over the final 1.9 simulated seconds")
    telemetry = json.loads((folder / "result.json").read_text())
    if telemetry["error"]:
        reasons.append(telemetry["error"])
    actions = json.loads((folder / "actions.json").read_text())
    interventions = [a["label"] for a in actions if a["label"].startswith("Recovery")]
    return dict(verdict="pass" if not reasons else "fail", valid_start=True, reasons=reasons,
                interventions=interventions, metrics=dict(physical=physical, goal_held=settled),
                evidence_role="privileged evaluator; never returned to the policy")
