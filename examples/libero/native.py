"""One counted execution boundary over the published LIBERO embodiment and tools."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots_agent._llm import ToolCall
from inspect_robots_agent._tools import build_toolset
from PIL import Image

from world_use.behaviors import Outcome
from world_use.cameras import Frame


class EpisodeEnded(RuntimeError):
    pass


def fingerprint(observation) -> str:
    h = hashlib.sha256()
    for key, value in sorted(observation.images.items()):
        h.update(key.encode())
        h.update(np.asarray(value).tobytes())
    for key, value in sorted(observation.state.items()):
        h.update(key.encode())
        h.update(np.asarray(value, dtype=np.float64).tobytes())
    h.update(str(observation.state_time).encode())
    return h.hexdigest()


class Frames:
    """Reuse capture identity when the simulator is paused, without refreshing its age."""

    def __init__(self):
        self.images = None
        self.current = {}
        self.cache = OrderedDict()

    def sync(self, observation):
        if observation.images is self.images:
            return
        self.images = observation.images
        self.current = {name: Frame(Image.fromarray(rgb), name) for name, rgb in observation.images.items()}
        self.cache.update((frame.id, frame) for frame in self.current.values())
        while len(self.cache) > 8:
            self.cache.popitem(last=False)

    def frame(self, camera=None, *, id=None, depth=False):
        if depth:
            raise ValueError("LIBERO policy observations are RGB only")
        return self.cache[id] if id is not None else self.current[camera]


class Driver:
    """All native, settling, and procedure steps pass through this same gate."""

    def __init__(self, embodiment, observation, output: Path, *, max_steps=900, deadline=None):
        self.embodiment = embodiment
        self.observation = observation
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.deadline = deadline or (time.monotonic() + 1800)
        self.steps = 0
        self.success = False
        self.stop_reason = None
        self.store = {}
        space = embodiment.info.action_space
        self.approver = ChainApprover(ClampApprover(space), DeltaLimitApprover(space))
        self.tools = build_toolset(space, embodiment.info.observation_space, control_hz=20)
        self.frames = Frames()
        self.frames.sync(observation)
        self.last_action = None
        self.record_observation()

    def record(self, filename, row):
        with (self.output / filename).open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")

    def record_observation(self):
        for name, frame in self.frames.current.items():
            frame.image.save(self.output / f"{self.steps:04d}-{name}.png")

    def check_open(self):
        if self.success or self.stop_reason:
            raise EpisodeEnded(self.stop_reason or "success")
        if self.steps >= self.max_steps:
            self.stop_reason = "step_limit"
            raise EpisodeEnded(self.stop_reason)
        if time.monotonic() >= self.deadline:
            self.stop_reason = "episode_deadline"
            raise EpisodeEnded(self.stop_reason)

    def step(self, action, *, source):
        self.check_open()
        requested_stop = bool(action.meta.get("request_stop"))
        raw = np.asarray(action.data)
        if raw.shape != (10,) or not np.isfinite(raw).all():
            raise ValueError("invalid native action")
        approved = self.approver.review(action, self.store)
        result = self.embodiment.step(approved)
        self.steps += 1
        self.observation = replace(result.observation, extra={"env_step": self.steps})
        self.frames.sync(self.observation)
        self.success = bool(result.terminated and result.info.get("success"))
        self.last_action = approved
        actual_steps = self.embodiment.last["steps"]
        if actual_steps != self.steps:
            raise RuntimeError(f"unaccounted simulator steps: {actual_steps} != {self.steps}")
        self.record(
            "steps.jsonl",
            dict(
                step=self.steps,
                source=source,
                requested=raw.tolist(),
                approved=np.asarray(approved.data).tolist(),
                state={k: v.tolist() for k, v in self.observation.state.items()},
                success=self.success,
            ),
        )
        # The published rollout executes the terminal hold once before stopping.
        if requested_stop and not self.success:
            self.stop_reason = action.meta.get("stop_reason", "policy_stop")
            raise EpisodeEnded(self.stop_reason)
        return self.observation

    def aperture_mm(self):
        q = self.observation.state["gripper_qpos"]
        return float(abs(q[0] - q[1]) * 1000)

    def measure(self):
        return dict(
            env_step=self.steps, eef_state=self.observation.state["eef_state"].tolist(), aperture_mm=self.aperture_mm()
        )

    def execute(self, command):
        """Native interpolation, optionally followed by a bounded measured settling check.

        Settling repeats the same commanded pose through the approvers; every repeat
        counts toward the same 900-step horizon. It never selects a new target pose.
        """
        kind = command["kind"]
        if kind == "confirm_state":
            return Outcome(
                "done" if fingerprint(self.observation) == command["fingerprint"] else "refused",
                kind,
                "inspected state comparison",
                self.measure(),
            )
        if kind == "check_grip":
            lo, hi = command["expect_mm"]
            observed = self.aperture_mm()
            return Outcome(
                "done" if lo <= observed <= hi else "surprise",
                kind,
                "measured aperture check",
                self.measure(),
                [lo, hi],
                observed,
                "inspect the fresh images; aperture does not establish object identity or retention",
            )
        if kind != "move":
            raise ValueError("unknown command")
        targets = command["targets"]
        result = self.tools.execute(
            ToolCall(
                id="native",
                name="move_to",
                arguments=json.dumps(
                    dict(targets=targets, note=command.get("note", "Execute the policy-selected bounded stage."))
                ),
            ),
            self.observation,
        )
        if result.error:
            return Outcome("refused", kind, result.error)
        before = self.steps
        monitored = command.get("monitor_mm")
        settle = command.get("settle_steps", 0)
        stable = 0
        initial_aperture = self.aperture_mm()

        def checks():
            state = self.observation.state["eef_state"]
            pos_error = float(np.linalg.norm(state[:3] - result.target[:3]))
            rot_error = float(np.max(np.abs(state[3:9] - result.target[3:9])))
            aperture = self.aperture_mm()
            opening = command.get("expect_mm")
            aperture_ok = opening is None or opening[0] <= aperture <= opening[1]
            # Closing must reach a stable measured opening, not merely pass through it.
            return pos_error <= 0.012 and rot_error <= 0.12 and aperture_ok, pos_error, rot_error, aperture

        prior_aperture = initial_aperture
        for index in range(len(result.chunk.actions) + settle):
            action = result.chunk.actions[min(index, len(result.chunk.actions) - 1)]
            self.step(action, source=command.get("stage", kind))
            if self.success:
                raise EpisodeEnded("success")
            okay, pos_error, rot_error, aperture = checks()
            if monitored and not monitored[0] <= aperture <= monitored[1]:
                return Outcome(
                    "surprise",
                    kind,
                    "grip range violated during motion",
                    self.measure(),
                    monitored,
                    aperture,
                    "inspect before any further carry or release",
                )
            if index >= len(result.chunk.actions) - 1:
                stable = stable + 1 if okay and abs(aperture - prior_aperture) <= 0.3 else 0
                if settle and stable >= 3:
                    break
            prior_aperture = aperture
        self.record_observation()
        data = dict(
            self.measure(),
            executed_steps=self.steps - before,
            position_error_m=pos_error,
            rotation_column_error=rot_error,
            settled=bool(stable >= 3),
        )
        if settle and stable < 3:
            return Outcome(
                "stopped",
                kind,
                "bounded settling ended without measured convergence",
                data,
                observed=data,
                hint="inspect and choose another command; later stages did not run",
            )
        return Outcome("done", kind, "native commands executed; inspect measured result", data)
