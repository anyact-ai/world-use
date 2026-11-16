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

DOCS = """World Use additions. You own object identity, poses and every decision. EdgeTAM only
follows the image regions you select: no depth, no metric position, no confirmation of what a
region is.
select_target binds a name to a pixel point or box in the current image of one camera. grasp and
place take the same selection as select and select_destination, which saves a call. Select in
agentview when the object is clearly visible there. That camera does not move, so the region
seen before the approach is compared with the region after the test lift, and across a place
move that is vertical to within 2 cm. A region that stayed in place while the hand moved returns
surprise: the hand holds something else or nothing, or the selection sits on another object, and
place issues no release checkpoint. A region that moved, was hidden or was lost concludes
nothing. Tracking is evidence, not a precondition: an unselected or lost region is reported and
the procedure still runs. Reselect it when you want the comparison back.
grasp moves level to above your approach pose and then straight down. The approach z is the
lowest height you allow: if the hand meets contact directly above the pose, with the palm on a
tall object or the fingertips on the table, it stops there, so ask for a low z when you cannot
judge height. It then closes, checks the measured opening against your expected range in
millimetres, lifts by lift z minus approach z from where it stopped, and checks again. It stops
at the first failed check and never carries, opens, retreats or retries. Judge from both cameras
whether the intended object followed.
With the start orientation the fingers open along world y and take objects up to 78 mm across.
Grasp a can, bottle or box at its centre with the fingers across its narrow side. A mug, bowl
or pot is wider than that: take it by the rim, offset by the rim radius along y so one finger
goes inside, with the approach z just below the rim top instead of a low z.
place moves level to above your release pose, descends with the grip monitored and stops early on
contact the same way, then returns BEFORE opening with a one-use checkpoint. Inspect the views,
then call release_placement to open and withdraw, or correct the pose. Any motion or reselection
voids the checkpoint.
Every interpolation and settling step counts toward the 900-step episode limit.
Results report measurements. arm: arrived; stopped_on_contact (above_request_m is how far above
the requested height); not_approaching (the remaining offset stopped shrinking and the hand is
not over the pose); still_converging. gripper: reached_end (fully closed or open, nothing between
the fingers); stopped_apart; still_moving. A nonzero aperture_change_mm means the fingers were
measured in motion. applied appears when the native approvers changed your commands: their
per-step limit is measured from the previous applied command, so a move issued while the arm
still trails it can end short, and repeating the move covers the rest.
Tool completion is not task success. Decide the next action from fresh camera evidence.
"""


# A camera that does not move and does not look along the vertical, so vertical travel of a held
# object shifts its image. The wrist camera rises with the hand; a straight-down view only rescales.
FIXED_CAMERAS = frozenset({"agentview"})
TRAVEL_M = 0.015  # shorter vertical hand travel is not compared
SIDEWAYS_M = 0.02  # nor is travel with more sideways motion: only vertical travel was measured
UNCHANGED_IOU = 0.9
# Image heights a region may shift per metre of vertical hand travel and still count as unchanged.
# In the recorded lifts held objects shifted at least 0.64 and untouched ones at most 0.18.
UNCHANGED_SHIFT_PER_M = 0.3


def region_evidence(target, reference, after, hand_from=None):
    """Did a selected region stay put while the hand measurably rose or sank?

    Compares the region in two observations from one of FIXED_CAMERAS, with no observation
    of it in between, against the hand's travel since hand_from (default: the reference).
    Returns None when they cannot be compared. An unchanged region did not travel with the
    hand: the hand holds something else or nothing, or the selection sits on another object.
    A changed one establishes nothing, since occlusion and tracking drift change regions
    too. Identity and retention stay the policy's judgement.
    """
    old = next((t for t in reference.get("targets", []) if t.get("target") == target), None)
    new = next((t for t in after.get("targets", []) if t.get("target") == target), None)
    if old is None or new is None or new.get("camera") not in FIXED_CAMERAS:
        return None
    motion = new.get("motion") or {}
    moved = np.subtract(after["eef_state"][:3], (hand_from or reference)["eef_state"][:3])
    travel = float(moved[2])
    if (
        abs(travel) < TRAVEL_M
        or np.linalg.norm(moved[:2]) > SIDEWAYS_M
        or motion.get("from_frame") != old.get("frame")
        or motion.get("mask_iou") is None
    ):
        return None
    shift = float(np.hypot(*motion["center_delta_px"]))
    limit = UNCHANGED_SHIFT_PER_M * abs(travel) * new["size"][1]
    return dict(
        camera=new["camera"],
        hand_vertical_travel_m=round(travel, 4),
        mask_iou=round(motion["mask_iou"], 3),
        center_shift_px=round(shift, 2),
        unchanged=bool(motion["mask_iou"] >= UNCHANGED_IOU and shift <= limit),
    )


# Left out of the policy's copy of a result: identifiers it never uses, state the observation
# message already shows, and constants the tool documentation states once.
OMITTED = frozenset(
    {
        "eef_state",
        "env_step",
        "frame",
        "from_frame",
        "captured_at",
        "age_s",
        "space",
        "size",
        "tracking",
        "kind",
        "identity_verified",
        "geometry_verified",
        "dominant_region",
        "camera_motion_compensated",
        "command",
        "intent",
        "policy_note",
    }
)


def brief(value):
    """The policy's copy of a result: the same facts without the keys above and without float noise.

    The episode record keeps the full result.
    """
    if isinstance(value, dict):
        return {k: brief(v) for k, v in value.items() if k not in OMITTED and v not in (None, [], {}, "")}
    if isinstance(value, (list, tuple)):
        return [brief(v) for v in value]
    if isinstance(value, float):
        return round(value, 4 if abs(value) < 10 else 1) + 0.0  # no negative zero
    return value


def procedure_brief(feedback, bound):
    """Procedure feedback for the policy: stage statuses, the latest view of its own targets, the deciding facts."""
    told = dict(feedback, stages={s["stage"]: s["status"] for s in feedback["stages"]})
    name, latest = list(feedback["observations"].items())[-1]
    told["observations"] = {
        name: dict(latest, targets=[t for t in latest.get("targets", []) if t.get("target") in bound])
    }
    told["assessments"] = [
        dict(
            checkpoint=a["checkpoint"], **{k: v for k, v in a["outcome"].items() if k in ("status", "message", "data")}
        )
        for a in feedback["assessments"]
        if a["outcome"]["kind"] != "not_assessed"
    ]
    last = told.pop("last_command", None)
    # A command that failed carries its measurements; one that passed before a failed assessment adds nothing.
    if last and last["outcome"]["status"] != "done":
        told["last_command"] = dict(stage=last["stage"], **last["outcome"].get("data", {}))
    return told


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
    description="Expected measured finger separation [min,max] in mm, with 4 <= min < max <= 78.",
)
SELECTION = dict(
    type="object",
    description="A region in one camera's current image: {camera, point: [x, y]} or {camera, box: [x0, y0, x1, y1]}.",
)
SCHEMAS = [
    schema(
        "select_target",
        "Bind your object identity claim to a point or box in the current image of one camera.",
        dict(
            target=TEXT,
            camera=TEXT,
            point=dict(type="array", items=dict(type="number"), minItems=2, maxItems=2),
            box=dict(type="array", items=dict(type="number"), minItems=4, maxItems=4),
            note=TEXT,
        ),
        ["target", "camera", "note"],
    ),
    schema(
        "observe_targets",
        "Read current tracked image regions. The paused simulator does not advance.",
        dict(note=TEXT),
        ["note"],
    ),
    schema(
        "grasp",
        "Move above your approach pose, descend to it or to contact above it, close, check measured "
        "opening, and test lift by 3-5 cm. Stops on failed evidence or measured checks; does not carry "
        "automatically. select binds object_target first, in the same call.",
        dict(object_target=TEXT, select=SELECTION, approach=TARGETS, lift=TARGETS, expect_grip_mm=RANGE, note=TEXT),
        ["object_target", "approach", "lift", "expect_grip_mm", "note"],
    ),
    schema(
        "place",
        "Move above your release pose and descend to it or to contact above it while monitoring grip, "
        "then return BEFORE opening so you can inspect fresh views. Withdrawal is only executed after a "
        "separate release decision. select_destination binds destination_target first, in the same call.",
        dict(
            object_target=TEXT,
            destination_target=TEXT,
            select_destination=SELECTION,
            move=TARGETS,
            withdraw=TARGETS,
            expect_grip_mm=RANGE,
            note=TEXT,
        ),
        ["object_target", "move", "withdraw", "expect_grip_mm", "note"],
    ),
    schema(
        "release_placement",
        "Approve the inspected placement and open/withdraw. Receipt is valid once, only while the "
        "observed images, robot state and target selections remain unchanged. State the placement "
        "evidence you saw in note.",
        dict(checkpoint=TEXT, note=TEXT),
        ["checkpoint", "note"],
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
        self.seen = None

    def schemas(self):
        return self.native.schemas() + copy.deepcopy(SCHEMAS)

    def state_labels(self):
        return self.native.state_labels()

    def residual(self, *args):
        return self.native.residual(*args)

    def observe(self):
        names = list(self.tracking.targets) + list(self.tracking.lost)
        # The simulator is paused between steps, so a second look at the same step is the same look.
        key = (self.driver.steps, tuple(names))
        if self.seen is None or self.seen[0] != key:
            result = dict(self.driver.measure(), targets=self.tracking.observe(names) if names else [])
            self.driver.record("tracking.jsonl", result)
            self.seen = (key, result)
        return copy.deepcopy(self.seen[1])

    def _select(self, target, where):
        if not isinstance(where, dict):
            raise ValueError("a selection names a camera and a point or box")
        self.driver.frames.sync(self.driver.observation)
        current = self.driver.frames.current
        if where.get("camera") not in current:
            raise ValueError(f"camera must be one of {sorted(current)}")
        self.seen = None
        return self.tracking.select(
            current[where["camera"]].id, target, point=where.get("point"), box=where.get("box"), space="image"
        )

    def _result(self, data, told=None):
        """Record the full result and return the policy its brief copy."""
        self.driver.record("world-use.jsonl", data)
        self.driver.record_observation()
        # This marker returns control to the episode loop without a simulator step.
        return ToolResult(
            note=json.dumps(brief(data if told is None else told), separators=(",", ":")),
            chunk=ActionChunk(
                actions=[Action(data=self.driver.observation.state["eef_state"].copy(), meta={"runner_no_step": True})]
            ),
        )

    def _procedure(self, outcome, *bound):
        feedback = procedures.feedback(outcome)
        return self._result(feedback, procedure_brief(feedback, bound))

    def _move(self, targets, stage, *, monitor=None, expected=None):
        return dict(kind="move", targets=targets, stage=stage, settle_steps=20, monitor_mm=monitor, expect_mm=expected)

    @staticmethod
    def _range(value):
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(type(v) not in (float, int) or not math.isfinite(v) for v in value)
            or not 4 <= value[0] < value[1] <= 78
        ):
            raise ValueError("expect_grip_mm must satisfy 4 <= min < max <= 78")
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
                # The observation that follows shows the state. A completed move whose last command
                # reached the request needs no sentence and no account of how the approvers got there.
                plain = result.ok and result.data.get("applied", {}).get("last_command_reached_request", True)
                told = dict(status=result.status, **result.data)
                if plain:
                    told.pop("applied", None)
                else:
                    told["message"] = result.message
                return self._result(result.feedback(), told)
            if call.name == "select_target":
                self.pending = None
                return self._result(self._select(args["target"], args))
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
                if "select" in args:
                    self._select(args["object_target"], args["select"])

                name = args["object_target"]
                ran = []

                def run(command):
                    outcome = self.driver.execute(command)
                    ran.append(outcome.ok)
                    return outcome

                def observe():
                    # While the hand is over the object it hides it from the fixed camera, and a
                    # region the tracker loses stays lost. Look at it before the approach and after
                    # a completed lift only.
                    if not ran or (len(ran) == 5 and all(ran)):
                        return self.observe()
                    return dict(self.driver.measure(), targets=[])

                def lifted(checkpoint, observations):
                    if checkpoint != "after":
                        return Outcome("done", "not_assessed", "the region is compared after the test lift")
                    facts = region_evidence(
                        name, observations["before"], observations["after"], observations["before_lift"]
                    )
                    if facts and facts["unchanged"]:
                        return Outcome(
                            "surprise",
                            "lift_evidence",
                            "the selected region stayed in place while the hand rose",
                            facts,
                            "a region that moves with the hand",
                            dict(mask_iou=facts["mask_iou"], center_shift_px=facts["center_shift_px"]),
                            "the hand may hold something else or nothing, or the selection may sit on another "
                            "object; the gripper is still closed. Inspect both cameras, then reselect or regrasp",
                        )
                    # A changed region, a lost one or a lift too short to compare proves nothing either way.
                    return None

                outcome = procedures.grasp(
                    run,
                    observe,
                    approach=dict(self._move(args["approach"], "grasp_approach"), above_first=True, contact_ok=True),
                    close=self._move({"gripper": 1}, "grasp_close", expected=grip),
                    check=dict(kind="check_grip", expect_mm=grip),
                    # The hand may have stopped on contact above the requested height; rise from there.
                    lift=dict(
                        self._move(
                            {k: v for k, v in args["lift"].items() if k != "z"}, "grasp_test_lift", monitor=grip
                        ),
                        rise_m=float(delta[2]),
                    ),
                    assess=lifted,
                )
                return self._procedure(outcome, name)
            if call.name == "place":
                self.pending = None
                grip = self._range(args["expect_grip_mm"])
                for key in ("move", "withdraw"):
                    self._validate_pose(args[key])
                if "select_destination" in args:
                    self._select(args["destination_target"], args["select_destination"])
                bound = [args[k] for k in ("object_target", "destination_target") if k in args]

                def pause(checkpoint, observations):
                    if checkpoint != "before_release":
                        return None
                    facts = region_evidence(args["object_target"], observations["before"], observations[checkpoint])
                    if facts and facts["unchanged"]:
                        return Outcome(
                            "surprise",
                            "carry_evidence",
                            "the selected object region stayed in place while the hand travelled",
                            facts,
                            "a region that moves with the hand",
                            dict(mask_iou=facts["mask_iou"], center_shift_px=facts["center_shift_px"]),
                            "no release checkpoint was issued: the hand may not hold the selected object, or "
                            "the selection may sit on another object. Inspect both cameras, then reselect "
                            "or regrasp",
                        )
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

                move = dict(self._move(args["move"], "place_approach", monitor=grip), above_first=True, contact_ok=True)
                return self._procedure(self._place(args, move, pause), *bound)
            if call.name == "release_placement":
                pending, self.pending = self.pending, None
                if (
                    pending is None
                    or args["checkpoint"] != pending["id"]
                    or fingerprint(self.driver.observation) != pending["fingerprint"]
                ):
                    raise ValueError("release checkpoint is missing, stale or consumed")

                def approved(checkpoint, observations):
                    if checkpoint != "before_release":
                        return None
                    return Outcome(
                        "done",
                        "policy_assessment",
                        "released on the policy's recorded decision",
                        dict(verified=False, policy_note=args["note"]),
                    )

                outcome = self._place(
                    pending["args"], dict(kind="confirm_state", fingerprint=pending["fingerprint"]), approved
                )
                kept = pending["args"]
                return self._procedure(
                    outcome, *(kept[k] for k in ("object_target", "destination_target") if k in kept)
                )
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
            assess=assess,
        )

    def close(self):
        self.tracking.close()
