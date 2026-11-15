"""RGB-only World Use tool adapter; Sol supplies identity, poses and decisions."""

from __future__ import annotations

import copy
import json
import math
import uuid

import numpy as np
from inspect_robots.types import Action, ActionChunk
from inspect_robots_agent._tools import ToolResult
from native import fingerprint

from world_use import procedures
from world_use.behaviors import Outcome
from world_use.tracking import Tracking
from world_use.vision_worker import TrackerProcess

DOCS = """World Use additions: select named regions by pixel point or box in an available
RGB capture, then observe their tracked regions. You own object identity and all spatial
reasoning. EdgeTAM tracks regions only; it supplies no depth, metric position or semantic
confirmation. Tracking motion in the wrist camera also includes camera motion.
grasp and place execute your chosen poses with measured checks and early returns. Every
interpolation and settling step uses the native controller and counts toward the 900-step
episode limit. They never choose an object, grasp pose, support height or recovery for you.
Before a grasp, select the intended object. Choose its expected measured finger opening
range in millimetres. Inspect both raw cameras after the short lift to judge whether the
intended object followed. Aperture alone does not establish identity or retention.
place returns BEFORE opening, with fresh views and a one-use release checkpoint. Inspect
the object and support geometry in those views; call release_placement only when the pose
is suitable. Otherwise choose a correction. Motion or reselection invalidates the receipt.
Tool completion is not task success. Use fresh camera evidence to decide the next action.
"""


def schema(name, description, properties, required):
    return dict(
        type="function",
        function=dict(
            name=name,
            description=description,
            parameters=dict(type="object", properties=properties, required=required, additionalProperties=False),
        ),
    )


TEXT = dict(type="string")
TARGETS = dict(type="object", description="Native move_to target dimensions; metres and rot6d columns.")
RANGE = dict(
    type="array",
    items=dict(type="number"),
    minItems=2,
    maxItems=2,
    description="Expected measured finger separation [min,max] in mm, with 2 <= min < max <= 78.",
)
SCHEMAS = [
    schema(
        "select_target",
        "Bind your object identity claim to a point or box in a visible RGB capture.",
        dict(
            target=TEXT,
            frame=TEXT,
            point=dict(type="array", items=dict(type="number"), minItems=2, maxItems=2),
            box=dict(type="array", items=dict(type="number"), minItems=4, maxItems=4),
            note=TEXT,
        ),
        ["target", "frame", "note"],
    ),
    schema(
        "observe_targets",
        "Read current tracked image regions. The paused simulator does not advance.",
        dict(note=TEXT),
        ["note"],
    ),
    schema(
        "grasp",
        "Execute your grasp approach, close, check measured opening, and test lift by 3-5 cm. "
        "Stops on failed evidence or measured checks; does not carry automatically.",
        dict(object_target=TEXT, approach=TARGETS, lift=TARGETS, expect_grip_mm=RANGE, note=TEXT),
        ["object_target", "approach", "lift", "expect_grip_mm", "note"],
    ),
    schema(
        "place",
        "Move to your chosen release pose while monitoring grip, then return BEFORE opening "
        "so you can inspect fresh views. Withdrawal is only executed after a separate release decision.",
        dict(
            object_target=TEXT, destination_target=TEXT, move=TARGETS, withdraw=TARGETS, expect_grip_mm=RANGE, note=TEXT
        ),
        ["object_target", "destination_target", "move", "withdraw", "expect_grip_mm", "note"],
    ),
    schema(
        "release_placement",
        "Approve the inspected placement and open/withdraw. Receipt is valid "
        "once, only while the observed images, robot state and target selections remain unchanged.",
        dict(checkpoint=TEXT, placement_evidence=TEXT, note=TEXT),
        ["checkpoint", "placement_evidence", "note"],
    ),
]


class WorldTools:
    def __init__(self, driver, *, tracker=None):
        self.driver = driver
        self.native = driver.tools
        self.tracking = Tracking(
            driver.frames, tracker or TrackerProcess(device="cpu", max_age_s=1800, timeout_s=90, startup_timeout_s=120)
        )
        self.pending = None
        self.latest = None

    def schemas(self):
        return self.native.schemas() + copy.deepcopy(SCHEMAS)

    def state_labels(self):
        return self.native.state_labels()

    def residual(self, *args):
        return self.native.residual(*args)

    def observe(self):
        names = list(self.tracking.targets) + list(self.tracking.lost)
        targets = self.tracking.observe(names) if names else []
        result = dict(self.driver.measure(), targets=targets)
        self.driver.record("tracking.jsonl", result)
        return result

    def context(self):
        self.driver.frames.sync(self.driver.observation)
        return dict(
            captures=[
                dict(camera=k, frame=v.id, captured_at=v.timestamp, size=list(v.image.size))
                for k, v in self.driver.frames.current.items()
            ],
            pending_release=self.pending["id"] if self.pending else None,
        )

    def _result(self, data):
        self.latest = data
        self.driver.record("world-use.jsonl", data)
        self.driver.record_observation()
        # This marker returns control to the episode loop without a simulator step.
        return ToolResult(
            note=json.dumps(data),
            chunk=ActionChunk(
                actions=[Action(data=self.driver.observation.state["eef_state"].copy(), meta={"runner_no_step": True})]
            ),
        )

    def _move(self, targets, stage, *, monitor=None, expected=None):
        return dict(kind="move", targets=targets, stage=stage, settle_steps=20, monitor_mm=monitor, expect_mm=expected)

    @staticmethod
    def _range(value):
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(type(v) not in (float, int) or not math.isfinite(v) for v in value)
            or not 2 <= value[0] < value[1] <= 78
        ):
            raise ValueError("expect_grip_mm must satisfy 2 <= min < max <= 78")
        return value

    def _validate_pose(self, targets):
        if not isinstance(targets, dict) or not targets or "gripper" in targets:
            raise ValueError("procedure poses must be nonempty native targets without a gripper command")
        from inspect_robots_agent._llm import ToolCall

        result = self.native.execute(
            ToolCall(
                id="validate",
                name="move_to",
                arguments=json.dumps(
                    dict(targets=targets, note="Validate the policy-selected pose without executing.")
                ),
            ),
            self.driver.observation,
        )
        if result.error:
            raise ValueError(result.error)

    def execute(self, call, observation):
        if call.name in ("done", "give_up"):
            self.pending = None
            return self.native.execute(call, observation)
        try:
            args = json.loads(call.arguments)
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
            if not isinstance(args.get("note"), str) or not args["note"].strip():
                raise ValueError("note is required")
            if call.name == "move_to":
                self.pending = None
                result = self.driver.execute(
                    dict(kind="move", targets=args.get("targets"), note=args["note"], stage="move_to")
                )
                return self._result(result.feedback())
            if call.name == "select_target":
                self.pending = None
                value = self.tracking.select(
                    args["frame"], args["target"], point=args.get("point"), box=args.get("box"), space="image"
                )
                return self._result(value)
            if call.name == "observe_targets":
                return self._result(self.observe())
            if call.name == "grasp":
                self.pending = None
                grip = self._range(args["expect_grip_mm"])
                for key in ("approach", "lift"):
                    self._validate_pose(args[key])
                if not {"x", "y", "z"} <= args["approach"].keys() or not {"x", "y", "z"} <= args["lift"].keys():
                    raise ValueError("grasp approach and lift require explicit x, y, z")
                delta = np.array([args["lift"][k] - args["approach"][k] for k in ("x", "y", "z")])
                if not (0.03 <= delta[2] <= 0.05 and np.linalg.norm(delta[:2]) <= 0.005):
                    raise ValueError("test lift must be vertical 3-5 cm, with at most 5 mm lateral displacement")
                outcome = procedures.grasp(
                    self.driver.execute,
                    self.observe,
                    object_target=args["object_target"],
                    approach=self._move(args["approach"], "grasp_approach"),
                    close=self._move({"gripper": 1}, "grasp_close", expected=grip),
                    check=dict(kind="check_grip", expect_mm=grip),
                    lift=self._move(args["lift"], "grasp_test_lift", monitor=grip),
                )
                return self._result(procedures.feedback(outcome))
            if call.name == "place":
                self.pending = None
                grip = self._range(args["expect_grip_mm"])
                for key in ("move", "withdraw"):
                    self._validate_pose(args[key])

                def pause(checkpoint, observations):
                    if checkpoint == "before_release":
                        self.pending = dict(
                            id=uuid.uuid4().hex,
                            fingerprint=fingerprint(self.driver.observation),
                            args=copy.deepcopy(args),
                        )
                        return Outcome(
                            "stopped",
                            "placement_decision",
                            "inspect fresh cameras and explicitly approve release, or correct the pose",
                            dict(checkpoint=self.pending["id"], completion_verified=False),
                        )
                    return Outcome("done", "after", "inspect placement; completion unverified")

                outcome = self._place(args, self._move(args["move"], "place_approach", monitor=grip), pause)
                return self._result(procedures.feedback(outcome))
            if call.name == "release_placement":
                pending, self.pending = self.pending, None
                evidence = args["placement_evidence"]
                if not isinstance(evidence, str) or not 1 <= len(evidence.strip()) <= 2000:
                    raise ValueError("provide a bounded explanation of the observed placement evidence")
                if (
                    pending is None
                    or args["checkpoint"] != pending["id"]
                    or fingerprint(self.driver.observation) != pending["fingerprint"]
                ):
                    raise ValueError("release checkpoint is missing, stale or consumed")

                def approved(checkpoint, observations):
                    return Outcome("done", "policy_assessment", evidence, dict(verified=False, checkpoint=checkpoint))

                outcome = self._place(
                    pending["args"], dict(kind="confirm_state", fingerprint=pending["fingerprint"]), approved
                )
                return self._result(procedures.feedback(outcome))
            raise ValueError("unknown tool")
        except (KeyError, ValueError, TypeError) as exc:
            return self._result(dict(status="refused", message=str(exc)))

    def _place(self, args, move, assess):
        return procedures.place(
            self.driver.execute,
            self.observe,
            move=move,
            release=self._move({"gripper": 0}, "place_release", expected=[70, 85]),
            check=dict(kind="check_grip", expect_mm=[70, 85]),
            withdraw=self._move(args["withdraw"], "place_withdraw"),
            object_target=args["object_target"],
            destination_target=args["destination_target"],
            assess=assess,
        )

    def close(self):
        self.tracking.close()
