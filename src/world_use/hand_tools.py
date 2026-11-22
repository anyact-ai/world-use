"""Locate, grasp, place and push for an agent: one call runs a measured sequence on any arm with a gripper.

A policy decides what every region is, where the hand goes and when to release; these tools run the motion
between those decisions and report what was measured. They move the arm only through an executor, which
turns each command into its robot's own motion, judges arrival and contact, and knows what its wrist can
reach. Commands are dicts:

- move: targets (the executor's own pose keys, from the policy or from `Executor.orientation`), stage, settle
  ("move", "close" or "pass": arrival without waiting to rest), expect_mm and monitor_mm (finger opening in
  millimetres, at the end and throughout), and optionally travel_z (rise first), via (a point reached through
  free space before a straight line to the pose, where contact short of the pose is an accepted end; empty
  for the point above the pose), rise_m (the target height is the measured height plus this) and in_place (a
  turn of the hand where it is, which the executor may make in parts).
- check_grip: expect_mm. confirm_state: fingerprint, a state the policy inspected.

Outcome data of a move reports arm (arrived, stopped_on_contact with short_m, not_approaching, still_converging)
and gripper (reached_end, stopped_apart, still_moving), as `checks.Settling` and `checks.Following` judge them.
"""

from __future__ import annotations

import copy
import json
import math
import uuid
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from . import procedures
from .behaviors import Outcome
from .cameras import View
from .client import DaemonError
from .parallax import Look, footprint, image_directions, locate
from .tracking import Tracking

DOWN = np.array([0.0, 0.0, -1.0])
UPRIGHT_COS = 0.995  # a hand whose approach axis is within about 6 degrees of straight down is upright
LEAN_DEG = 30  # a pushing hand leans this far back, which puts its body behind its fingertips
BACK_OFF_M = 0.1  # a push leaves what it pushed by this much, room for a leaning hand to stand up again
LONGER = 1.3  # a region this much longer than it is wide has a narrow side to take it by

# Lift and carry evidence from a camera that does not move and does not look along the vertical, so
# vertical travel of a held object shifts its image. Thresholds from region-evidence.jsonl in the LIBERO
# example: held objects shifted at least 0.64 image heights per metre of travel, untouched ones at most 0.18.
TRAVEL_M = 0.015  # shorter vertical hand travel is not compared
SIDEWAYS_M = 0.02  # nor is travel with more sideways motion: only vertical travel was measured
UNCHANGED_IOU = 0.9
UNCHANGED_SHIFT_PER_M = 0.3


@dataclass(frozen=True)
class Hand:
    """What the tools need to know about a gripper and its cameras."""

    opening_mm: float  # the widest measured opening the fingers can be expected to report
    open_mm: tuple[float, float]  # fingers this far apart have let go, or made room
    fit_mm: float  # the widest narrow side a grasp turns the fingers to take by itself
    arrive_m: float  # a hand this near a goal is there, for a look from it
    hand_camera: str | None  # the camera fixed to the hand, for locate and centred grasps
    fixed_camera: str | None  # a camera that neither moves nor looks along the vertical, for image evidence
    in_view_m: float  # from a hover the hand camera sees no further to the side than this
    hover: str  # the hover the help asks for, as "15-25 cm"
    max_lean_deg: float = 60
    standoff_m: float = 0.1  # a leaning hand comes in along its own axis from this far back
    # The hand's resting orientation points the fingers straight down, and the tools stand a leaning hand up
    # again to look down with its camera or after a push. An arm that grasps from the side keeps its lean.
    stands_up: bool = True


class Executor(Protocol):
    """An arm's side of the tools. Every move it makes counts toward its own limits and records.

    measure() is the hand's measured state as observations carry it, and position() reads the tool point
    from one. pose() is the tool point and approach axis now. orientation() turns an approach axis, and
    optionally a sign-free finger axis, into the executor's target keys, and returns the finger axis it chose.
    lean_refusal() says (message, hint) when the hand cannot lean so and follow path, the start and end of
    its line. hand_look() is the hand camera's view and the tool point when a frame (by id) was taken, or now;
    None without a hand camera. fingerprint() changes once anything may have commanded the arm, and noise in
    its readings leaves it alone. key() is equal in two calls only when nothing in the world can have changed
    between them, as in a paused simulator, and None when that cannot be known. failures are the exceptions
    that mean the executor lost its arm during a call: the call then ends stopped, with what ran before.
    """

    failures: tuple[type[Exception], ...]

    def execute(self, command: dict) -> Outcome: ...
    def measure(self) -> dict: ...
    def position(self, observation: dict) -> np.ndarray: ...
    def pose(self) -> tuple[np.ndarray, np.ndarray]: ...
    def orientation(self, point, jaws=None) -> tuple[dict, np.ndarray | None]: ...
    def lean_refusal(self, direction, tilt_deg: float, path) -> tuple[str, str] | None: ...
    def hand_look(self, frame: str | None = None) -> tuple[View, np.ndarray] | None: ...
    def cameras(self) -> list[str]: ...
    def frame(self, camera: str): ...
    def validate_pose(self, targets: dict) -> None: ...
    def fingerprint(self) -> str: ...
    def record(self, name: str, row: dict) -> None: ...
    def key(self) -> object | None: ...


def tilted(direction, tilt_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """The approach axis turned tilt_deg from straight down toward a level direction, and the level finger
    axis across that direction."""
    level, tilt = np.array([*direction[:2], 0.0]) / math.hypot(*direction[:2]), math.radians(tilt_deg)
    return math.cos(tilt) * DOWN + math.sin(tilt) * level, np.array([level[1], -level[0], 0.0])


def upright(axis) -> bool:
    return float(axis[2]) <= -UPRIGHT_COS


def region_evidence(target, reference, after, travel, camera):
    """Whether a selected region stayed put while the hand rose or sank by `travel` (metres, x y z).

    Compares the region in two observations from `camera`, with no observation of it in between. Returns
    None when they cannot be compared. An unchanged region did not travel with the hand: the hand holds
    something else or nothing, or the selection sits on another object. A changed one proves nothing, since
    occlusion and tracking drift change regions too.
    """
    old = next((t for t in reference.get("targets", []) if t.get("target") == target), None)
    new = next((t for t in after.get("targets", []) if t.get("target") == target), None)
    if old is None or new is None or new.get("camera") != camera:
        return None
    motion = new.get("motion") or {}
    vertical = float(travel[2])
    if (
        abs(vertical) < TRAVEL_M
        or np.linalg.norm(travel[:2]) > SIDEWAYS_M
        or motion.get("from_frame") != old.get("frame")
        or motion.get("mask_iou") is None
    ):
        return None
    shift = float(np.hypot(*motion["center_delta_px"]))
    limit = UNCHANGED_SHIFT_PER_M * abs(vertical) * new["size"][1]
    return dict(
        camera=new["camera"],
        hand_vertical_travel_m=round(vertical, 4),
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

    The record keeps the full result.
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
    told: dict = dict(feedback, stages={s["stage"]: s["status"] for s in feedback["stages"]})
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


def _move(targets, stage, *, monitor=None, expected=None, settle="move"):
    return dict(kind="move", targets=targets, stage=stage, settle=settle, monitor_mm=monitor, expect_mm=expected)


def _point(value, name):
    if (
        not isinstance(value, dict)
        or value.keys() != {"x", "y", "z"}
        or any(type(v) not in (float, int) or not math.isfinite(v) for v in value.values())
    ):
        raise ValueError(f"{name} is a point: x, y and z in metres")
    return {k: float(v) for k, v in value.items()}


def _travel(args):
    travel = args.get("travel_z")
    if travel is not None and (type(travel) not in (float, int) or not math.isfinite(travel)):
        raise ValueError("travel_z must be a height in metres")
    return travel


XYZ = dict(type="object", description="{x, y, z} in metres, in the arm's frame.")
RANGE = dict(
    type="array",
    items=dict(type="number"),
    minItems=2,
    maxItems=2,
    description="Expected measured finger opening [min, max] in mm.",
)
SELECTION = dict(
    type="object",
    description="{camera, point: [u, v]} or {camera, box: [u0, v0, u1, v1]} in the camera's native pixels.",
)
TEXT = dict(type="string")


def _schema(name, description, properties, required):
    return dict(
        name=name,
        description=description,
        parameters=dict(type="object", properties=properties, required=required, additionalProperties=False),
    )


def _schemas(hand_camera):
    hand = f"the {hand_camera} camera" if hand_camera else "a camera on the hand"
    tools = [
        _schema(
            "select_target",
            "Bind your claim of what an object is to a point or box in one camera's current "
            "image. The tracker follows the region; it does not confirm what it is.",
            dict(
                target=TEXT,
                camera=TEXT,
                point=dict(type="array", items=dict(type="number")),
                box=dict(type="array", items=dict(type="number")),
            ),
            ["target", "camera"],
        ),
        _schema("observe_targets", "Read the tracked regions now.", {}, []),
        _schema(
            "grasp",
            f"Open, move above the approach pose (over the region itself when it was selected in "
            f"{hand}), come down to it or to contact short of it, or with tilt_deg come in leaning along the "
            "hand's own axis; close, check the measured opening, and lift 3-5 cm to test the hold. Stops at "
            "the first failed check; never carries, retreats or retries.",
            dict(
                object_target=TEXT,
                select=SELECTION,
                approach=XYZ,
                lift=dict(XYZ, description="{z}: the test-lift height"),
                expect_grip_mm=RANGE,
                offset=dict(
                    type="array",
                    items=dict(type="number"),
                    description="[dx, dy] in metres from "
                    f"the centre of a region selected in {hand} to the grasp point.",
                ),
                yaw_deg=dict(
                    type="number",
                    description="Turn of the finger axis about the vertical, from +y toward +x, in [-90, 90]: 0 along "
                    "y, 45 along x+y, -45 along x-y, 90 along x. For fingers along (dx, dy), atan2(dx, dy) in "
                    "degrees, plus or minus 180 to bring it into the range: the axis has no sign.",
                ),
                tilt_deg=dict(
                    type="number", description="Lean of the hand from upright; it then comes in along its own axis."
                ),
                toward=dict(type="object", description="{x, y}: the point a leaning hand's fingertips point toward."),
            ),
            ["object_target", "approach", "lift", "expect_grip_mm"],
        ),
        _schema(
            "place",
            "Rise to travel_z, move above the release pose or to via, and come to the pose or to "
            "contact short of it with the grip monitored, then stop BEFORE opening with a one-use checkpoint.",
            dict(
                object_target=TEXT,
                destination_target=TEXT,
                select_destination=SELECTION,
                travel_z=dict(type="number"),
                via=XYZ,
                move=XYZ,
                withdraw=XYZ,
                expect_grip_mm=RANGE,
            ),
            ["object_target", "move", "withdraw", "expect_grip_mm"],
        ),
        _schema(
            "release_placement",
            "Open and withdraw after inspecting a place. The checkpoint is valid once, "
            "and only while nothing has moved or been reselected since.",
            dict(checkpoint=TEXT, note=TEXT),
            ["checkpoint", "note"],
        ),
        _schema(
            "push",
            "Close the fingers, go to start through free space, move the fingertips straight to end or "
            "to where contact stops them, and back off. For a drawer, a door or sliding an object.",
            dict(start=XYZ, end=XYZ, tilt_deg=dict(type="number"), travel_z=dict(type="number")),
            ["start", "end"],
        ),
    ]
    if hand_camera:
        tools.insert(
            2,
            _schema(
                "locate",
                f"Move the hand level until it is over a region selected in {hand} and "
                "return its x, y, z and size. select binds target first.",
                dict(target=TEXT, select=SELECTION),
                ["target"],
            ),
        )
    return tools


class HandTools:
    """select_target, observe_targets, locate, grasp, place, release_placement and push.

    `call(name, arguments)` returns the full record of what happened and the policy's brief copy as JSON
    text. Arguments are the policy's: poses are dicts of the executor's own target keys with x, y and z in
    metres; a selection is {camera, point: [u, v]} or {camera, box: [u0, v0, u1, v1]} in native pixels.
    """

    def __init__(self, executor: Executor, hand: Hand, tracking: Tracking):
        self.arm, self.hand, self.tracking = executor, hand, tracking
        self.pending = None  # the one-use release checkpoint of the last place
        self.seen = None
        self.located = {}  # where each target was last found; forgotten once it is reselected or taken
        self.outlined = set()  # targets selected with a box around them, whose region is the whole object

    @property
    def push_lean(self) -> float:
        """How far a push leans the hand back unless told otherwise."""
        return min(LEAN_DEG, self.hand.max_lean_deg)

    def schemas(self) -> list[dict]:
        """Each tool's name, description and JSON Schema parameters, for an agent loop to register."""
        return _schemas(self.hand.hand_camera)

    def docs(self) -> str:
        """What the policy needs to know beyond each tool's description, with this hand's numbers."""
        h = self.hand
        camera = (
            f"Positions of small objects come from the {h.hand_camera} camera: hover {h.hover} above, select "
            "the whole object there with a box, and locate or grasp it; the hand then moves level, looking "
            "from several places, until it is over the region. "
            if h.hand_camera
            else ""
        )
        offset = "; with an offset from a hand-camera region and no toward it leans toward the region's centre"
        offset = offset if h.hand_camera else ""
        return (
            "You decide what every region is, where the hand goes and when to release; the tools run the motion "
            "between those decisions and report measurements, not success. "
            + camera
            + f"The fingers open to {h.opening_mm:g} mm. A grasp's approach z is the lowest height you allow: on "
            "contact above it the hand stops there and approach_ended_short_m says how far above. To take a thing "
            f"by its handle or in under a roof, lean the hand with tilt_deg (up to {h.max_lean_deg:g}): it turns "
            f"where it is, then comes in along its own axis from {100 * h.standoff_m:g} cm back, and keeps the "
            f"lean{offset}. A pinch on a handle does not resist turning: look at how the object hangs before "
            "going on. A lean the arm cannot make is refused before it moves. place stops before opening with a "
            "one-use checkpoint; inspect the views, then call release_placement or correct the pose. push leans "
            f"the hand {self.push_lean:g} degrees back from a level line so the fingertips lead, and backs off up "
            f"to {100 * BACK_OFF_M:g} cm. Results: arm arrived, stopped_on_contact (short_m), not_approaching; "
            "gripper reached_end (nothing between the fingers), stopped_apart. A surprise is a measurement: "
            "correct its cause and try again."
        )

    def observe(self):
        names = list(self.tracking.targets) + list(self.tracking.lost)
        # Nothing moves between two looks with the same key, so the second is the first again.
        key = self.arm.key()
        key = None if key is None else (key, tuple(names))
        if key is None or self.seen is None or self.seen[0] != key:
            result = dict(self.arm.measure(), targets=self.tracking.observe(names) if names else [])
            self.arm.record("tracking.jsonl", result)
            self.seen = (key, result)
        return copy.deepcopy(self.seen[1])

    def select(self, target, where):
        if not isinstance(where, dict):
            raise ValueError("a selection names a camera and a point or box")
        cameras = self.arm.cameras()
        if where.get("camera") not in cameras:
            raise ValueError(f"camera must be one of {sorted(cameras)}")
        self.seen = None
        self.located.pop(target, None)
        self.outlined.discard(target)
        if where.get("box") is not None:
            self.outlined.add(target)
        return self.tracking.select(
            self.arm.frame(where["camera"]).id, target, point=where.get("point"), box=where.get("box"), space="image"
        )

    def _centre(self, name, stage, floor_z=None):
        """Move the hand level until it is over a region tracked in the hand camera.

        Returns the region's position and shape, or the Outcome that stopped the search. Every move ends
        once the hand is there: a look needs the pose it is taken from, not a hand at rest. Given the lowest
        height a grasp allows, the first move heads for a little above it and the moves also open the
        fingers; otherwise they leave them alone. A position found earlier for a region not reselected
        since is confirmed by one look instead of being worked out again.
        """
        hand = dict(gripper=0) if floor_z is not None else {}

        def region():
            return next((t for t in self.observe()["targets"] if t["target"] == name), {})

        def look():
            seen = region()
            visible = seen.get("valid") is True and bool(seen.get("center"))
            view, here = self._hand_look(seen.get("frame"))
            return Look(view, here, *((seen["center"], seen.get("bbox")) if visible else ()))

        short = []

        def move(goal):
            targets = dict(x=float(goal[0]), y=float(goal[1]), **hand)
            here, axis = self.arm.pose()
            if math.hypot(goal[0] - here[0], goal[1] - here[1]) > self.hand.in_view_m:
                # Looks from too near, or at a region the hand covers, meet far outside what the camera sees.
                return Outcome(
                    "stopped",
                    "locate",
                    f"the looks at the region do not place it within the {self.hand.hand_camera} camera's view",
                    hint=f"hover {self.hand.hover} above the object with all of it in the {self.hand.hand_camera} "
                    "image and select it there again; later stages did not run",
                )
            if self.hand.stands_up and not upright(axis):
                # The last look is taken from over the region, straight down: a hand that a grasp or a
                # carry left leaning stands up on its way there.
                targets.update(self.arm.orientation(DOWN)[0])
            command = _move(targets, stage, expected=self.hand.open_mm if hand else None, settle="pass")
            moved = self.arm.execute(command)
            if moved.data.get("arm") == "not_approaching" and not short:
                # A first guess can be past the arm's reach. The hand rests where it stopped, and a look
                # from there is still a new look. A second such move ends the search.
                short.append(moved)
                return Outcome("done", moved.kind, "stopped short of a guessed goal; looking again from here")
            return moved

        found = locate(
            move,
            look,
            arrive_m=self.hand.arrive_m,
            assumed_z=None if floor_z is None else floor_z + 0.05,
            known=self.located.get(name),
        )
        if not found.ok:
            return found
        data = found.data
        self.located[name] = [data[k] for k in "xyz"]
        view = self._hand_look()[0]
        (pixel,), (depth,) = view.project(self.located[name])
        if "pixel" in data and depth > 0:
            # Which way the image the policy sees next lies in the world, for an offset toward something in it.
            for side, step in zip(("image_right", "image_down"), image_directions(view, pixel, depth), strict=True):
                data[side] = [round(float(v), 2) for v in step]
        extent = data.pop("extent_m", None)
        if extent:
            data["extent_mm"] = [round(1000 * side) for side in extent]
            shape = footprint(view, region().get("dominant_region") or {}, depth)
            if shape:
                data.update(across_mm=round(1000 * shape["across_m"]), along_mm=round(1000 * shape["along_m"]))
                if shape["along_m"] >= LONGER * shape["across_m"]:
                    # The finger axis turned from base y toward base x, as grasp's yaw_deg is, in [-90, 90).
                    yaw = math.degrees(math.atan2(shape["across"][0], shape["across"][1]))
                    data["fit_yaw_deg"] = round((yaw + 90) % 180 - 90)
        return data

    def _hand_look(self, frame=None) -> tuple[View, np.ndarray]:
        look = self.arm.hand_look(frame)
        if look is None:
            raise ValueError("this arm has no camera on its hand")
        return look

    def _approach(self, command, facts):
        """Execute a stage, noting how far short of its pose an approach that met contact ended."""
        outcome = self.arm.execute(command)
        if "short_m" in outcome.data:
            # Where the approach ended says what it met: the object, the surface or something else.
            facts.update(approach_ended_short_m=outcome.data["short_m"])
        return outcome

    def _turned(self, yaw_deg):
        turn = math.radians(yaw_deg)
        return self.arm.orientation(DOWN, np.array([math.sin(turn), math.cos(turn), 0.0]))[0]

    def _lean(self, command, tilt, toward, offset, found):
        """A grasp approach with the hand leaning: it turns where it is, goes to a point its standoff back
        along its own axis and from there straight along that axis to the grasp point. Returns the command
        for those last two legs, or the Outcome of a lean that has no direction or a turn that failed."""
        point = command["targets"]
        # With no point to lean toward, the fingertips point from the grasp point back to the region's centre.
        level = [toward[k] - point[k] for k in "xy"] if toward else [-offset[0], -offset[1]]
        length = math.hypot(*level)
        if length < 0.01:
            return Outcome(
                "refused",
                command["kind"],
                "the lean has no direction: toward, or the region's centre, is less than 1 cm from the grasp point",
                hint="name a toward further along the way the fingertips should point; later stages did not run",
            )
        approach, across = tilted(level, tilt)
        back = np.array([point[k] for k in "xyz"]) - self.hand.standoff_m * approach
        if refusal := self.arm.lean_refusal(level, tilt, [back, [point[k] for k in "xyz"]]):
            message, hint = refusal
            return Outcome("refused", command["kind"], message, hint=f"{hint}; later stages did not run")
        turn = self.arm.orientation(approach, across)[0]
        # The hand turns where it is: turned while it travels, a raised arm can swing several centimetres, which
        # reads as a hand held back.
        command_turn = _move(dict(turn, gripper=0), "grasp_lean", expected=self.hand.open_mm, settle="pass")
        there = self.arm.execute(dict(command_turn, in_place=True))
        if not there.ok:
            return there
        found.update(leaning=dict(tilt_deg=tilt, toward=[round(float(v) / length, 2) for v in level]))
        return dict(command, targets=dict(point, **turn), via=dict(zip("xyz", map(float, back), strict=True)))

    def _push(self, args):
        """Fingers closed, to start through free space, then straight to end or to where contact stops the hand."""
        start, end = _point(args["start"], "start"), _point(args["end"], "end")
        line = np.array([end[k] - start[k] for k in "xyz"])
        length = float(np.linalg.norm(line))
        if length < 0.01:
            raise ValueError("the start and end of a push are at least 1 cm apart")
        tilt = args.get("tilt_deg", self.push_lean)
        if type(tilt) not in (float, int) or not 0 <= tilt <= self.hand.max_lean_deg:
            raise ValueError(f"tilt_deg is between 0 and {self.hand.max_lean_deg:g}")
        # Only a mostly level line turns the hand across it, and leans it unless the line points where
        # the wrist cannot lean. A steep one has no direction to lean back from, and stands a hand
        # upright that an earlier grasp left leaning.
        level = math.hypot(line[0], line[1]) >= abs(line[2])
        if level:
            path = [[start[k] for k in "xyz"], [end[k] for k in "xyz"]]
            if tilt and (refusal := self.arm.lean_refusal(line, tilt, path)):
                if not self.hand.stands_up:
                    raise ValueError(f"{refusal[0]}; {refusal[1]}")
                tilt = 0  # an upright hand pushes where the wrist cannot lean
            turn = self.arm.orientation(*tilted(line, tilt))[0]
        else:
            turn = {} if not self.hand.stands_up or upright(self.arm.pose()[1]) else self.arm.orientation(DOWN)[0]
        stages = [
            _move({"gripper": 1}, "push_close"),
            dict(_move({**end, **turn}, "push"), travel_z=_travel(args), via=start),
        ]
        for command in stages:
            outcome = self.arm.execute(command)
            if not outcome.ok:
                break
        short = outcome.data.get("short_m")
        here = self.arm.pose()[0]
        retreat = {}
        if command is stages[-1]:
            # Leave what was pushed, back along the line but never past where the push began, also when the
            # push failed on its way. The other tools expect an upright hand; standing up while backing off
            # keeps its body behind its fingertips all the way.
            unit = line / length
            along = float(np.clip((here - [start[k] for k in "xyz"]) @ unit, 0.0, BACK_OFF_M))
            retreat.update(zip("xyz", map(float, here - unit * along), strict=True))
            if level and tilt and self.hand.stands_up:
                retreat.update(self.arm.orientation(DOWN)[0])
        left = self.arm.execute(_move(retreat, "push_back_off")) if retreat else outcome
        if not outcome.ok:
            told = dict(status=outcome.status, message=outcome.message, **outcome.data)
            if retreat:
                told["back_off"] = dict(status=left.status, arm=left.data.get("arm"))
            return outcome.feedback(), told
        told = dict(
            status=left.status,
            pushed=dict(
                ended="reached_end" if short is None else "stopped_on_contact",
                travelled_mm=round(1000 * (length - (short or 0.0))),
                short_mm=round(1000 * (short or 0.0)),
            ),
            **self.observe(),
        )
        return told, told

    def _range(self, value):
        top = self.hand.opening_mm
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(type(v) not in (float, int) or not math.isfinite(v) for v in value)
            or not 4 <= value[0] < value[1] <= top
        ):
            raise ValueError(f"expect_grip_mm must satisfy 4 <= min < max <= {top:g}")
        return value

    def _procedure(self, outcome, *bound, **facts):
        feedback = procedures.feedback(outcome)
        return dict(feedback, **facts), dict(procedure_brief(feedback, bound), **facts)

    def call(self, name: str, args: dict) -> tuple[dict, str]:
        """Run one tool: the full record and the policy's brief copy as compact JSON."""
        try:
            record, told = self._call(name, args)
        except (KeyError, ValueError, TypeError) as exc:
            record = told = dict(status="refused", message=str(exc))
        except self.arm.failures as exc:
            record = told = dict(
                status="stopped",
                message=f"the arm's controller failed during this call: {exc}",
                hint="what ran before the failure may have moved the arm: observe before going on",
            )
        return record, json.dumps(brief(told), separators=(",", ":"))

    def _call(self, name, args):
        if name in ("select_target", "locate", "grasp", "place", "push"):
            self.pending = None  # any motion or reselection voids a release checkpoint
        if name == "select_target":
            selected = self.select(args["target"], args)
            return selected, selected
        if name == "observe_targets":
            seen = self.observe()
            return seen, seen
        if name == "locate":
            target = args["target"]
            camera = args["select"].get("camera") if isinstance(args.get("select"), dict) else None
            if (
                self.hand.hand_camera is None
                or (camera or self.tracking.targets.get(target, {}).get("camera")) != self.hand.hand_camera
            ):
                raise ValueError(f"locate needs the target selected in the {self.hand.hand_camera} camera")
            if "select" in args:
                self.select(target, args["select"])
            found = self._centre(target, "locate")
            if isinstance(found, Outcome):
                return found.feedback(), found.feedback()
            located = dict(status="done", located=found, **self.arm.measure())
            return located, located
        if name == "grasp":
            return self._grasp(args)
        if name == "place":
            return self._place_call(args)
        if name == "push":
            return self._push(args)
        if name == "release_placement":
            return self._release(args)
        raise ValueError("unknown tool")

    def _grasp(self, args):
        grip = self._range(args["expect_grip_mm"])
        for key in ("approach", "lift"):
            self.arm.validate_pose(args[key])
        name = args["object_target"]
        ran = []
        camera = args["select"].get("camera") if isinstance(args.get("select"), dict) else None
        hand_camera = self.hand.hand_camera
        wrist = hand_camera is not None and (camera or self.tracking.targets.get(name, {}).get("camera")) == hand_camera
        # Centring over a hand-camera region supplies x and y; nothing else does.
        if not ({"z"} if wrist else {"x", "y", "z"}) <= args["approach"].keys() or "z" not in args["lift"]:
            raise ValueError(
                "grasp needs approach z and lift z, and approach x, y unless the object is selected "
                f"in the {hand_camera} camera"
            )
        # To the micrometre: 0.54 - 0.49 is a little over 0.05 in binary.
        rise = round(float(args["lift"]["z"] - args["approach"]["z"]), 6)
        if not 0.03 <= rise <= 0.05:
            raise ValueError("the test lift rises 3-5 cm: lift z minus approach z")
        offset = args.get("offset", [0, 0])
        if (
            not isinstance(offset, list)
            or len(offset) != 2
            or any(type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 0.2 for v in offset)
        ):
            raise ValueError("offset is [dx, dy] in metres, each within 0.2")
        if any(offset) and not wrist:
            raise ValueError(f"offset is measured from a region selected in the {hand_camera} camera")
        approach = dict(args["approach"], gripper=0)
        if (yaw := args.get("yaw_deg")) is not None:
            if type(yaw) not in (float, int) or not -90 <= yaw <= 90:
                raise ValueError("yaw_deg is between -90 and 90")
            approach.update(self._turned(yaw))
        tilt, toward = args.get("tilt_deg", 0), args.get("toward")
        if type(tilt) not in (float, int) or not 0 <= tilt <= self.hand.max_lean_deg:
            raise ValueError(f"tilt_deg is between 0 and {self.hand.max_lean_deg:g}")
        if toward is not None and (
            not isinstance(toward, dict)
            or toward.keys() != {"x", "y"}
            or any(type(v) not in (float, int) or not math.isfinite(v) for v in toward.values())
        ):
            raise ValueError("toward is a level point: x and y in metres")
        if tilt and yaw is not None:
            raise ValueError("a leaning hand's finger axis lies across its lean: give tilt_deg or yaw_deg")
        if tilt and toward is None and not any(offset):
            raise ValueError(
                "tilt_deg needs toward, the point the fingertips lean toward, or an offset from a "
                f"{hand_camera} region, whose centre they then lean toward"
            )
        if toward is not None and not tilt:
            raise ValueError("toward is where tilt_deg leans the hand: give both or neither")
        if "select" in args:
            self.select(name, args["select"])

        found = {}

        def run(command):
            if command.get("stage") == "grasp_approach" and wrist:
                centred = self._centre(name, "grasp_centre", float(args["approach"]["z"]))
                if isinstance(centred, Outcome):
                    ran.append(False)
                    return centred
                # The image that follows a grasp is not the one these three describe.
                unseen = ("pixel", "image_right", "image_down")
                found.update(centred={k: v for k, v in centred.items() if k not in unseen})
                over = dict(x=centred["x"] + offset[0], y=centred["y"] + offset[1])
                fit = centred.get("fit_yaw_deg")
                fits = fit is not None and centred["across_mm"] <= self.hand.fit_mm
                if yaw is None and not tilt and not any(offset) and fits and name in self.outlined:
                    # Fingers across the narrow side close on it squarely instead of turning it first.
                    # A point may select only part of an object, whose narrow side is not the object's.
                    over.update(self._turned(fit))
                    found.update(fingers_turned_to_yaw_deg=fit)
                command = dict(command, targets=dict(command["targets"], **over))
            if command.get("stage") == "grasp_approach" and tilt:
                command = self._lean(command, tilt, toward, offset, found)
                if isinstance(command, Outcome):
                    ran.append(False)
                    return command
            outcome = self._approach(command, found)
            ran.append(outcome.ok)
            return outcome

        def observe():
            # While the hand is over the object it hides it from the fixed camera, and a region the
            # tracker loses stays lost. Look at it before the approach and after a completed lift only.
            if not ran or (len(ran) == 5 and all(ran)):
                return self.observe()
            return dict(self.arm.measure(), targets=[])

        def lifted(checkpoint, observations):
            if checkpoint != "after":
                return Outcome("done", "not_assessed", "the region is compared after the test lift")
            facts = self._evidence(name, observations["before"], observations["after"], observations["before_lift"])
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
            approach=dict(_move(approach, "grasp_approach", expected=self.hand.open_mm), via={}),
            # Fingers closing across a box that is not square to them turn it first, slowly.
            close=_move({"gripper": 1}, "grasp_close", expected=grip, settle="close"),
            check=dict(kind="check_grip", expect_mm=grip),
            # The hand may have been centred, and stopped on contact above the requested height:
            # rise straight up from where it is.
            lift=dict(_move({}, "grasp_test_lift", monitor=grip), rise_m=rise),
            assess=lifted,
        )
        if len(ran) > 1:
            self.located.pop(name, None)  # fingers that closed may have moved it
        return self._procedure(outcome, name, **found)

    def _evidence(self, target, reference, after, hand_from=None):
        if self.hand.fixed_camera is None:
            return None
        travel = self.arm.position(after) - self.arm.position(hand_from or reference)
        return region_evidence(target, reference, after, travel, self.hand.fixed_camera)

    def _place_call(self, args):
        if not isinstance(args.get("object_target"), str):
            raise ValueError("place needs object_target, the held object's selected target")
        grip = self._range(args["expect_grip_mm"])
        for key in ("move", "withdraw"):
            self.arm.validate_pose(args[key])
        travel = _travel(args)
        if "via" in args:
            _point(args["via"], "via")
        if "select_destination" in args:
            self.select(args["destination_target"], args["select_destination"])
        bound = [args[k] for k in ("object_target", "destination_target") if k in args]

        def pause(checkpoint, observations):
            if checkpoint != "before_release":
                return None
            facts = self._evidence(args["object_target"], observations["before"], observations[checkpoint])
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
            self.pending = dict(id=uuid.uuid4().hex, fingerprint=self.arm.fingerprint(), args=copy.deepcopy(args))
            return Outcome(
                "stopped",
                "placement_decision",
                "inspect fresh cameras and explicitly approve release, or correct the pose",
                dict(checkpoint=self.pending["id"], completion_verified=False),
            )

        move = dict(_move(args["move"], "place_approach", monitor=grip), travel_z=travel, via=args.get("via", {}))
        facts = {}
        return self._procedure(self._place(args, move, pause, facts), *bound, **facts)

    def _release(self, args):
        pending, self.pending = self.pending, None
        if pending is None or args["checkpoint"] != pending["id"] or self.arm.fingerprint() != pending["fingerprint"]:
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
            pending["args"], dict(kind="confirm_state", fingerprint=pending["fingerprint"]), approved, {}
        )
        kept = pending["args"]
        return self._procedure(outcome, *(kept[k] for k in ("object_target", "destination_target") if k in kept))

    def _place(self, args, move, assess, facts):
        return procedures.place(
            lambda command: self._approach(command, facts),
            self.observe,
            move=move,
            release=_move({"gripper": 0}, "place_release", expected=self.hand.open_mm),
            check=dict(kind="check_grip", expect_mm=self.hand.open_mm),
            withdraw=_move(args["withdraw"], "place_withdraw"),
            assess=assess,
        )

    def close(self):
        self.tracking.close()


FINISHED = ("done", "refused", "surprise", "stopped", "faulted", "cancelled")
ALIGNED_COS = math.cos(math.radians(0.5))  # a hand within this of a turn's axes already points so


class DaemonArm:
    """The tools' executor for a robot the world-use daemon runs, through its client. Positions are in the robot's
    base frame. Each command is one job, rehearsed before it is admitted; between jobs the arm holds where it is.

    Moves go in straight legs no longer than one of the robot's segments. A line that may end on contact is a guarded
    move over its last segment, which needs joint torque sensing: on an arm without it the line is a plain move and
    contact is a surprise. The jaw sign and what the wrist can reach are the kernel's: move_to chooses the sign, and
    a lean is refused when its rehearsal is.
    """

    failures = (DaemonError, OSError)

    def __init__(self, client, hand: Hand, *, record=None, wait_s: float = 120.0):
        self.client, self.hand, self.wait_s = client, hand, wait_s
        self.sink = record  # where rows are recorded, (name, row) -> None; none by default
        self.torque = "torque_nm" in client.status()
        # Pieces a little shorter than the limit: the kernel measures each from where the last one ended.
        self.leg_m = 0.95 * float(client.capabilities()["max_segment_m"])

    # -- what the tools read ----------------------------------------------------------------------------
    def measure(self):
        s = self.client.status()
        return dict(tool=s["tool"]["base"], aperture_mm=(s.get("gripper") or {}).get("aperture_mm"), events=s["events"])

    def position(self, observation):
        return np.asarray(observation["tool"], float)

    def pose(self):
        s = self.client.status()
        return np.asarray(s["tool"]["base"], float), np.asarray(s["tool_axes"]["approach"], float)

    def orientation(self, point, jaws=None):
        targets = dict(point=[float(v) for v in point])
        if jaws is not None:
            targets["jaws"] = [float(v) for v in jaws]
        return targets, None if jaws is None else np.asarray(jaws, float)

    def lean_refusal(self, direction, tilt_deg, path):
        """A rehearsal of the lean as the tools make it: a turn where the hand is, then the approach to path's start
        and along its line."""
        approach, across = tilted(direction, tilt_deg)
        here = self.pose()[0]
        turn = self._turn(dict(point=approach.tolist(), jaws=across.tolist()))
        report = self.client.check(turn + self._approach(here, path[0], path[-1], {})[0])
        if report.get("ok"):
            return None
        problems = list(dict.fromkeys(p["message"] for p in report.get("problems") or []))  # each said once
        reason = "; ".join(problems) or (report.get("outcome") or {}).get("message") or report.get("text", "")
        return f"the arm cannot lean that way at that point: {reason}", "lean another way or less, or grasp upright"

    def hand_look(self, frame=None):
        if self.hand.hand_camera is None:
            return None
        seen = self.client.frame(id=frame) if frame else self.client.frame(self.hand.hand_camera)
        return seen.view, np.asarray(seen.tool, float)[:3, 3]

    def cameras(self):
        return list(self.client.capabilities()["cameras"])

    def frame(self, camera):
        return self.client.frame(camera)

    def validate_pose(self, targets):
        if not isinstance(targets, dict) or not targets or not targets.keys() <= {"x", "y", "z"}:
            raise ValueError("a pose is x, y and z in metres, in the robot's base frame")
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in targets.values()):
            raise ValueError("a pose is x, y and z in metres, in the robot's base frame")

    def fingerprint(self):
        """The kernel's count of control changes, from this client or any other, with its power state."""
        s = self.client.status()
        return json.dumps([s["control_revision"], s["enabled"], s["faulted"], s["power_uncertain"]])

    def record(self, name, row):
        if self.sink is not None:
            self.sink(name, row)

    def key(self):
        return None  # the world goes on changing between jobs

    # -- commands -----------------------------------------------------------------------------------------
    def execute(self, command):
        kind = command["kind"]
        if kind == "confirm_state":
            same = command["fingerprint"] == self.fingerprint()
            return Outcome("done" if same else "refused", kind, "inspected state comparison", self.measure())
        if kind == "check_grip":
            return self._opening(command["expect_mm"], Outcome("done", kind, "measured opening", self.measure()))
        steps, line = self._steps(command)
        outcome = self._job(steps, kind, line, command.get("monitor_mm"))
        if outcome.ok and command["targets"].get("gripper") == 0 and command.get("expect_mm"):
            return self._opening(command["expect_mm"], outcome)
        return outcome

    def _opening(self, expect, outcome):
        """The outcome, or a surprise when the measured opening is outside expect (mm). A fully open hand reads its
        widest opening, which the kernel's own grip checks cannot take as an upper bound."""
        observed = self.measure()["aperture_mm"]
        if observed is not None and expect[0] - 0.5 <= observed <= expect[1] + 0.5:
            return outcome
        return Outcome(
            "surprise",
            outcome.kind,
            "the gripper stopped outside the expected opening",
            dict(outcome.data, aperture_mm=observed),
            list(expect),
            observed,
            "inspect the cameras; the opening does not show what is held",
        )

    def _steps(self, command):
        """Plan steps for a move, and the length of its guarded last step when that may end on contact."""
        targets, here = dict(command["targets"]), self.pose()[0]
        turn = {k: targets.pop(k) for k in ("point", "jaws") if k in targets}
        gripper = targets.pop("gripper", None)
        goal = np.array([targets.get(k, v) for k, v in zip("xyz", here, strict=True)], float)
        if command.get("rise_m") is not None:
            goal[2] = here[2] + command["rise_m"]
        steps = []
        if gripper is not None:
            if gripper and command.get("expect_mm"):
                steps.append(dict(do="grip", expect_mm=list(command["expect_mm"])))
            elif gripper:  # closed on nothing, as for a push
                steps.append(dict(do="gripper", aperture_mm=0.0))
            else:
                steps.append(dict(do="gripper", aperture_mm=float(self.hand.opening_mm), check_arrival=True))
        travel = command.get("travel_z")
        if travel is not None and travel - here[2] > self.hand.arrive_m:
            above = np.array([here[0], here[1], travel])
            steps += self._legs(here, [above], {})
            here = above
        via, line = command.get("via"), None
        if via is None:
            steps += self._legs(here, [goal], turn)
        else:
            start = np.array([via.get("x", goal[0]), via.get("y", goal[1]), via.get("z", here[2])], float)
            approach, line = self._approach(here, start, goal, turn)
            steps += approach
        if command.get("settle") == "move" and steps and steps[-1]["do"] == "move_to":
            steps.append(dict(do="settle"))
        return steps, line

    def _approach(self, here, start, goal, turn):
        """Steps level to above start, turning the hand on the way, straight to start, then along the line to goal,
        which may end on contact; and that line's guarded length, None when it has none."""
        steps = self._legs(here, [[start[0], start[1], here[2]], start], turn)
        d = np.asarray(goal, float) - start
        length = float(np.linalg.norm(d))
        if not self.torque or length <= 1e-4:
            return steps + self._legs(start, [goal], {}), None
        guarded = min(length, self.leg_m)
        before = start + d * (1 - guarded / length)
        words = dict(forward=d[0], left=d[1], up=d[2])
        steps += self._legs(start, [before], {})
        steps.append(
            dict(
                do="guarded",
                frame="base",
                expect_contact=False,
                **{k: float(v * guarded / length) for k, v in words.items() if abs(v) > 1e-9},
            )
        )
        return steps, guarded

    def _legs(self, here, points, turn):
        """move_to steps from here through points, in pieces no longer than leg_m. The hand turns to `turn` on the
        first leg, or where it is when that leg goes nowhere."""
        steps, at = [], np.asarray(here, float)
        for n, point in enumerate(points):
            point = np.asarray(point, float)
            length = float(np.linalg.norm(point - at))
            pieces = math.ceil(length / self.leg_m) if length > 1e-4 else 0
            steps += [dict(do="move_to", to=(at + (point - at) * i / pieces).tolist(), frame="base")
                      for i in range(1, pieces + 1)]
            if n == 0 and (turning := self._turn(turn)):
                if steps:
                    steps[0].update(turn)
                else:
                    steps += turning
            at = point
        return steps

    def _turn(self, turn):
        """A turn of the hand where it is, or no step when it already points along turn's point and, when given, its
        jaws open along turn's jaws either way round: the kernel refuses a move that goes nowhere."""
        if not turn:
            return []
        axes = self.client.status()["tool_axes"]  # rounded, so no longer quite unit length

        def cos(a, b):
            a, b = np.asarray(a, float), np.asarray(b, float)
            return float(a @ b) / float(np.linalg.norm(a) * np.linalg.norm(b))

        aligned = cos(axes["approach"], turn["point"]) >= ALIGNED_COS
        if "jaws" in turn:
            aligned = aligned and abs(cos(axes["opens"], turn["jaws"])) >= ALIGNED_COS
        return [] if aligned else [dict(do="move_to", frame="base", **turn)]

    def _job(self, steps, kind, line=None, monitor=None):
        if not steps:
            data = dict(self.measure(), arm="arrived")
            return Outcome("done", kind, "nothing to do: the hand is already there", data)
        spec: dict = dict(do="seq", steps=steps)
        if monitor:
            spec["expect_grip_mm"] = list(monitor)
        try:
            job = self.client.run(spec, wait=self.wait_s)
        except DaemonError as e:
            if e.code != 409:
                raise
            return Outcome("refused", kind, str(e), self.measure())
        while job.get("id") is not None and job["status"] not in FINISHED:
            job = self.client.job(job["id"], wait=self.wait_s)
        outcome = job.get("outcome") or {}
        data = self.measure()
        if job["status"] != "done":
            message = outcome.get("message") or job.get("incident", "")
            return Outcome(
                "stopped" if job["status"] == "cancelled" else job["status"],  # a queued job a stop ended
                kind,
                message,
                data,
                outcome.get("expected"),
                outcome.get("observed"),
                outcome.get("hint", ""),
            )
        last = (outcome.get("data") or {}).get("last_result") or {}
        moved = (last.get("data") or {}).get("moved_m")
        if line is not None and moved is not None and steps[-1]["do"] == "guarded":
            data.update(arm="stopped_on_contact", short_m=round(max(line - moved, 0.0), 4))
            return Outcome("done", kind, "the approach met contact short of the requested pose", data)
        data["arm"] = "arrived"
        return Outcome("done", kind, "done; inspect the measured result", data)
