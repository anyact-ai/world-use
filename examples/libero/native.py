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
from inspect_robots.types import Action
from inspect_robots_agent._llm import ToolCall
from inspect_robots_agent._tools import build_toolset
from PIL import Image

from world_use.behaviors import Outcome
from world_use.cameras import Frame
from world_use.checks import Settling

HZ = 20
ARM_TOLERANCE = (0.012, 0.12)  # metres, largest rot6d column difference
OPEN_MM = 80.0  # both fingers at their 0.04 m travel limit
CLOSED_END_MM = 3.0  # empty fingers creep from here toward 1.4 mm; at or below it nothing is held
OPEN_END_MM = 3.0  # the fingers creep over the last millimetres to their open limit
GRIP_STOP_MM = 0.3  # less travel than this over a settling window is a stop, not motion
WINDOW_S = 3.5 / HZ  # four 20 Hz intervals, clear of a sample boundary
STABLE_S = 2 / HZ  # three consecutive samples
# A free arm closes about a third of the distance to its command each step. A hand that gains
# under a millimetre toward a command 1.5 cm or more away, twice running, is held by something.
LEAD_M, PROGRESS_M, HELD_STEPS = 0.015, 0.001, 2
RELIEF_STEPS = 12


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
        self.changed = []  # (step, approver flags) for each applied action that differs from its request
        self.aperture_change = 0.0
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
        aperture = self.aperture_mm()
        result = self.embodiment.step(approved)
        self.steps += 1
        if approved is not action:
            self.changed.append((self.steps, [k for k in ("clamped", "delta_clamped") if approved.meta.get(k)]))
        self.observation = replace(result.observation, extra={"env_step": self.steps})
        self.frames.sync(self.observation)
        self.success = bool(result.terminated and result.info.get("success"))
        self.last_action = approved
        self.aperture_change = self.aperture_mm() - aperture
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
        """Measured state. A nonzero last-step aperture change means the fingers are still moving."""
        return dict(
            env_step=self.steps,
            eef_state=self.observation.state["eef_state"].tolist(),
            aperture_mm=self.aperture_mm(),
            aperture_change_mm=round(self.aperture_change, 2),
        )

    def applied(self, target, since):
        """How the approvers changed this command, or None when every request was applied unchanged."""
        changed = [flags for step, flags in self.changed if step > since]
        if not changed:
            return None
        last = np.asarray(self.last_action.data)
        offset = np.abs(last - target)
        # The hand follows only which side of 0.5 its command is on, and an unnamed gripper
        # entry is filled from a command state that may still be ramping.
        same_hand = (last[9] > 0.5) == (target[9] > 0.5) and (last[9] < 0.5) == (target[9] < 0.5)
        return dict(
            changed_steps=len(changed),
            reasons=sorted({flag for flags in changed for flag in flags}),
            last_command_position_offset_m=round(float(np.linalg.norm(offset[:3])), 4),
            last_command_reached_request=bool(np.all(offset[:9] <= 1e-9) and same_hand),
        )

    def execute(self, command):
        """Native interpolation, optionally followed by a bounded measured settling check.

        Settling repeats the same commanded pose through the approvers; every repeat
        counts toward the same 900-step horizon. It returns as soon as the measurements
        settle the question: the arm arrived or stopped short, the fingers reached their
        end or stopped apart. Motion still under way at the bound is reported as such,
        not as a stable result. A settled stage also ends once its command leads a hand
        that no longer follows, and a hand that stopped short is commanded to where it
        is, so that it stops pushing and the next move starts from rest.

        A stage may ask for four things a raw move does not do. travel_z first rises
        straight up to that height when the hand is below it. above_first moves level
        to above the pose and settles there before descending, so the hand comes straight
        down. contact_ok accepts a descent that contact stopped directly above the pose.
        rise_m replaces the target height with the measured height plus that rise.
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
        targets, started = command["targets"], self.steps
        if command.get("rise_m") is not None:
            targets = {**targets, "z": float(self.observation.state["eef_state"][2]) + command["rise_m"]}
        legs = {**command, "travel_z": None, "above_first": False, "contact_ok": False, "rise_m": None}

        def leg(part):
            out = self.execute({**legs, "targets": part})
            if "above_request_m" in out.data:
                # A leg that stalled is short of the stage's request, not of its own.
                state = self.observation.state["eef_state"]
                aside = [state[i] - targets[k] for i, k in enumerate("xy") if k in targets]
                out.data.update(
                    above_request_m=round(float(state[2] - targets.get("z", state[2])), 4),
                    aside_m=round(float(np.linalg.norm(aside)), 4) if aside else 0.0,
                )
            return out

        travel = command.get("travel_z")
        if travel is not None and travel - float(self.observation.state["eef_state"][2]) > ARM_TOLERANCE[0]:
            rise = leg({"z": travel})
            if not rise.ok:
                return rise
        if command.get("above_first") and "z" in targets and len(targets) > 1:
            first = leg({k: v for k, v in targets.items() if k != "z"})
            if not first.ok:
                return first
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
        before = started
        stage = command.get("stage", kind)
        monitored = command.get("monitor_mm")
        target, final = result.target, result.chunk.actions[-1]

        def errors():
            state = self.observation.state["eef_state"]
            return [float(np.linalg.norm(state[:3] - target[:3])), float(np.max(np.abs(state[3:9] - target[3:9])))]

        def report(arm=None, gripper=None):
            pos_error, rot_error = errors()
            data = dict(
                self.measure(),
                executed_steps=self.steps - before,
                position_error_m=pos_error,
                rotation_column_error=rot_error,
            )
            # A raw move checks no arrival, so it reports none.
            data.update({k: v for k, v in (("arm", arm), ("gripper", gripper)) if v})
            if (applied := self.applied(target, before)) is not None:
                data["applied"] = applied
            return data

        def advance(action):
            self.step(action, source=stage)
            if self.success:
                raise EpisodeEnded("success")
            aperture = self.aperture_mm()
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
            return None

        settle = command.get("settle_steps", 0)
        held, before_step = 0, self.observation.state["eef_state"][:3].copy()

        def held_up():
            nonlocal held, before_step
            here = self.observation.state["eef_state"][:3]
            lead = np.asarray(self.last_action.data)[:3] - before_step
            distance = float(np.linalg.norm(lead))
            gained = float((here - before_step) @ lead) / distance if distance > LEAD_M else np.inf
            held, before_step = (held + 1 if gained < PROGRESS_M else 0), here.copy()
            return held >= HELD_STEPS

        arm_state = hand_state = None
        for action in result.chunk.actions:
            if violated := advance(action):
                return violated
            if settle and held_up():
                arm_state = "stalled"
                break
        if not settle:
            self.record_observation()
            data = report()
            message = "native commands executed; inspect measured result"
            if "applied" in data and not data["applied"]["last_command_reached_request"]:
                message = "native commands executed, but the approvers held the last command short of the request"
            return Outcome("done", kind, message, data)

        # Settling repeats the final request. The half-step margin keeps the last counted
        # sample inside the deadline; the loop bound is what limits the steps.
        seconds = (settle + 0.5) / HZ
        arm = Settling(ARM_TOLERANCE, seconds, WINDOW_S, stable_s=STABLE_S)
        # Above 0.5 the native hand keeps closing and below it keeps opening. Its travel
        # toward that end either reaches it, stops on something, or is still under way.
        closing = "gripper" in targets and final.data[9] > 0.5
        opening = "gripper" in targets and final.data[9] < 0.5
        end = CLOSED_END_MM if closing else OPEN_END_MM
        hand = Settling([end], seconds, WINDOW_S, GRIP_STOP_MM / end, stable_s=STABLE_S) if closing or opening else None
        for index in range(0 if arm_state else settle + 1):
            if index and (violated := advance(final)):
                return violated
            if index and held_up():
                arm_state = "stalled"
                break
            # A wait cannot conclude while an approver is still changing what is applied:
            # that is neither arrival nor a stall.
            requested = self.last_action is final
            arm_state = arm.update(self.steps / HZ, errors(), command_active=requested)
            if hand is not None:
                # The hand's own command ramps toward its end. While closing, the ramp only
                # tightens fingers that have already stopped. While opening, it does not move
                # them until it passes their position, so that pause is not a stop.
                ramped = closing or self.observation.state["eef_state"][9] < 1e-9
                aperture = self.aperture_mm()
                hand_state = hand.update(
                    self.steps / HZ,
                    [aperture if closing else max(0.0, OPEN_MM - aperture)],
                    command_active=requested and ramped,
                )
            if arm_state in ("stalled", "timeout") or (arm_state == "arrived" and (hand is None or hand_state)):
                break
        later = "inspect and choose another command; later stages did not run"
        if arm_state == "stalled":
            state = self.observation.state["eef_state"]
            level = float(np.linalg.norm(state[:2] - target[:2]))
            above = float(state[2] - target[2])
            # Stopped over the pose with only height left to go: the palm met the object or the
            # fingertips met the table. A hand pushed aside is still a failed approach.
            contact = (
                command.get("contact_ok")
                and level <= ARM_TOLERANCE[0]
                and errors()[1] <= ARM_TOLERANCE[1]
                and above > 0
            )
            # Stop pressing: command the hand to rest at its height, over the pose when it
            # stopped there and otherwise where it is.
            for _ in range(RELIEF_STEPS):
                rest = np.append(self.observation.state["eef_state"][:9], final.data[9])
                if contact:
                    rest[:2], rest[3:9] = target[:2], target[3:9]
                rest = Action(data=rest)
                if violated := advance(rest):
                    return violated
                if self.last_action is rest:
                    break
            if contact:
                self.record_observation()
                data = report("stopped_on_contact")
                data["above_request_m"] = round(above, 4)
                return Outcome("done", kind, "the descent met contact above the requested height", data)
        self.record_observation()
        if arm_state != "arrived":
            stalled = arm_state == "stalled"
            data = report("not_approaching" if stalled else "still_converging")
            if stalled:
                # Where the hand stopped relative to the request tells a blocked descent from a shove.
                state = self.observation.state["eef_state"]
                data["above_request_m"] = round(float(state[2] - target[2]), 4)
                data["aside_m"] = round(float(np.linalg.norm(state[:2] - target[:2])), 4)
            return Outcome(
                "surprise" if stalled else "stopped",
                kind,
                "the arm is no longer approaching the requested pose"
                if stalled
                else "the arm was still converging when bounded settling ended",
                data,
                dict(position_error_m=ARM_TOLERANCE[0], rotation_column_error=ARM_TOLERANCE[1]),
                {k: data[k] for k in ("position_error_m", "rotation_column_error")},
                (
                    "something is in the way or the request is out of reach; the hand rests where it stopped; "
                    if stalled
                    else ""
                )
                + later,
            )
        # A stalled hand is one whose fingers stopped apart: on something, or blocked.
        grip = hand and {"arrived": "reached_end", "stalled": "stopped_apart"}.get(hand_state, "still_moving")
        data = report("arrived", grip)
        expected = command.get("expect_mm")
        within = expected and expected[0] <= data["aperture_mm"] <= expected[1]
        # Fingers still creeping open inside the expected opening have let go. A close still
        # under way has not gripped.
        if grip == "still_moving" and not (opening and within):
            return Outcome(
                "stopped", kind, "the gripper was still moving when bounded settling ended", data, hint=later
            )
        if hand is not None and expected and not within:
            empty = closing and hand_state == "arrived"
            return Outcome(
                "surprise",
                kind,
                "the fingers closed fully: nothing is between them at this pose"
                if empty
                else "the gripper stopped outside the expected opening",
                data,
                list(expected),
                data["aperture_mm"],
                "the measured opening is stable; " + later,
            )
        return Outcome("done", kind, "native commands executed; inspect measured result", data)
