"""RGB-only World Use tool adapter; Sol supplies identity, poses and decisions."""

from __future__ import annotations

import copy
import json
import math
import uuid

import numpy as np
from inspect_robots.types import Action, ActionChunk
from inspect_robots_agent._tools import ToolResult
from native import ARM_TOLERANCE, fingerprint

from world_use import procedures
from world_use.behaviors import Outcome
from world_use.cameras import View
from world_use.parallax import Look, footprint, locate
from world_use.tracking import Tracking
from world_use.vision_worker import TrackerProcess

DOCS = """World Use additions. You own object identity, poses and every decision. EdgeTAM only
follows the image regions you select: it does not confirm what a region is.
Positions come from the wrist camera. Hover 15-25 cm above the scene with the whole object in
the wrist image and select it there with a box around all of it; a point on the top face also
does for a small object.
- grasp with a wrist selection first moves the hand level, looking from several positions, until
it is over the region, and reports that position as centred. The approach then needs only z.
- locate does the same without descending or moving the fingers and returns located: the
region's x, y, z, its extent_mm along the finger axis and across it, and pixel, where that point
is in the wrist image you see next. Check there that it lies on what you meant. Use it once for
a destination such as a basket or plate before carrying anything there; the position holds
while that object is not moved. z reads low, by up to 3 cm on flat things and 6 cm on tall ones
in recorded trials: use it for clearance, not as a grasp or release height.
A position found for a target is kept until you select the target again or a grasp closes on it:
locate or grasp it again without a new selection and one look confirms it.
When the last look shows the whole region, centred and located also give across_mm and
along_mm, its narrow and long sides, and for a region clearly longer than wide fit_yaw_deg, the
yaw_deg that puts the fingers across the narrow side.
centred and located x, y are world coordinates of the region's centre, already corrected for
the camera's mount: pass them unchanged as targets. Never line objects up in the wrist image
yourself: the camera sits 5 cm from the grasp axis, so the point under the fingers is at column
128, row 41 at fingertip depth and row 96 on a surface 17 cm below, not at the image centre.
A selection in agentview is not centred: grasp then uses your x, y as given. That camera does not
move, so it adds one check instead. A region seen there before the approach is compared with the
region after the test lift, and across a place move that is vertical to within 2 cm. A region
that stayed in place while the hand moved returns surprise: the hand holds something else or
nothing, and place issues no release checkpoint. A region that moved, was hidden or was lost
concludes nothing. Tracking is evidence, not a precondition.
grasp opens the fingers, moves level to above the pose and then straight down. The approach z is
the lowest height you allow: if the hand meets contact directly above the pose, with the palm on
a tall object or the fingertips on the table, it stops there, so ask for a low z when you cannot
judge height, and approach_ended_short_m says how far above it: far above the object's own height
means something else held the hand up. It then closes, checks the measured opening against your
expected range in millimetres, rises by lift z minus approach z, and checks again. It stops at
the first failed check and never carries, retreats or retries. Judge from both cameras whether
the object followed.
With the start orientation the fingers open along world y and take objects up to 78 mm across.
grasp's yaw_deg turns that axis about the vertical, 90 to world x, and the hand keeps the turn
afterwards. Given no yaw_deg and no offset, a grasp of a region you selected in the wrist camera
with a box around all of the object turns to its fit_yaw_deg by itself when its narrow side is
at most 70 mm, and reports fingers_turned_to_yaw_deg; pass yaw_deg to choose the turn yourself,
as when a neighbour is in the way. A point may select only part of an object, so a point
selection never turns the hand. The hand body is 20 cm wide along the finger axis and starts 6
cm above the fingertips: a neighbour taller than that within 10 cm along the axis blocks the
descent, and a turn of 30-60 degrees away from it can clear it. Fingers closing across a box
that is not square to them turn it first, which takes tens of steps. Grasp a can, bottle or box
at its centre with the fingers across its narrow side. A mug, bowl or pot is wider than the
fingers: take it by the rim. Give grasp an offset of the rim radius along the finger axis from
the centre of the wrist region, on a side clear of any handle, so one finger goes inside, with
the approach z 2-3 cm below the rim top instead of a low z.
place rises straight up to travel_z when the hand is below it, moves level to above your release
pose, descends with the grip monitored and stops early on contact the same way, then returns
BEFORE opening with a one-use checkpoint. Its approach_ended_short_m tells a landing on
the surface you meant from one on something higher, such as a rim. Choose a travel_z that clears
everything on the way, the destination's rim included, by more than the held object hangs below
the fingertips. To come at the release pose from the side instead of from above, into a shelf or
an oven, give via: the hand moves level to over that point, down to it and then straight to the
pose, stopping early on contact along that line, and after the release it leaves through via
before going to your withdraw pose. Put via in front of the opening at the height of the pose,
far enough out that the hand and what it holds are clear of the frame. Inspect the views, then
call release_placement to open and withdraw, or correct the pose. Any motion or reselection
voids the checkpoint. Two objects for one container go side
by side along its longer extent: release the first a fifth of that extent to one side of the
located centre and the second a fifth to the other side, so neither lands on the other or on
a wall.
push closes the fingers and moves them in a straight line from start to end, to shut a drawer or
a door or to slide something along a surface. It goes to start through free space, rising to
travel_z first when you give one, so put start a few centimetres clear of what you push, on the
side you push from, with the fingertips at the height of its middle. The line ends at end or
where contact stops the hand: pushed says which, with travelled_mm and short_mm. Put end a few
centimetres past where the thing should stop, since a drawer or door that reaches its stop holds
the hand there. The hand then backs off 2 cm. A hinged door moves along an arc: push it along two
or three straight lines. yaw_deg turns the finger axis as in grasp.
Every interpolation and settling step counts toward the 900-step episode limit; one clean pick
and place takes about 200.
Results report measurements. arm: arrived; stopped_on_contact (short_m is how far short of the
requested pose, along the line of approach); not_approaching (something stopped or shoved the hand: above_request_m
and aside_m say where it rests relative to the request); still_converging. gripper: reached_end
(fully closed or open, nothing between the fingers); stopped_apart; still_moving. applied appears
when the native approvers changed your commands: a move issued while the arm still trails the
previous command can end short, and repeating the move covers the rest.
A surprise is a measurement, not the end of the task: correct its cause and try again while
calls and steps remain. Tool completion is not task success. Decide from fresh camera evidence.
"""


# A camera that does not move and does not look along the vertical, so vertical travel of a held
# object shifts its image. The wrist camera rises with the hand; a straight-down view only rescales.
FIXED_CAMERA = "agentview"
TRAVEL_M = 0.015  # shorter vertical hand travel is not compared
SIDEWAYS_M = 0.02  # nor is travel with more sideways motion: only vertical travel was measured
UNCHANGED_IOU = 0.9
# Image heights a region may shift per metre of vertical hand travel and still count as unchanged.
# In the recorded lifts held objects shifted at least 0.64 and untouched ones at most 0.18.
UNCHANGED_SHIFT_PER_M = 0.3


# The wrist camera in the tool frame, from the embodiment's camera mount: a 75 degree field of view
# looking along the approach axis from 5 cm along tool y and 9.7 cm behind the grasp point, so the
# grasp point is not at the image centre. Published wrist images have their rows flipped, which the
# negative focal length and mirrored row offset say.
WRIST = View(
    [[-1.0, 0, 0, 0], [0, -1.0, 0, 0.05], [0, 0, 1.0, -0.097], [0, 0, 0, 1.0]], 166.8128, -166.8128, 128, 127, 256, 256
)
OPEN_MM = [70, 85]  # fingers this far apart have let go, or made room, even while they still creep
FIT_MM = 70  # the widest narrow side the fingers are turned to take: 78 mm of opening less clearance
LONGER = 1.3  # a region this much longer than it is wide has a narrow side to take it by
CLOSE_STEPS = 120
PUSH_M = (0.02, 0.4)  # shortest and longest line of one push
BACK_OFF_M = 0.02
SETTLE_STEPS = 40  # raised high above the table the arm needs about 25 steps to stop swinging


def turned(yaw_deg):
    """Pose targets for the finger axis, tool x, turned from world y toward world x with the approach axis down."""
    turn = math.radians(yaw_deg)
    columns = [math.sin(turn), math.cos(turn), 0.0, math.cos(turn), -math.sin(turn), 0.0]
    return dict(zip(("r00", "r10", "r20", "r01", "r11", "r21"), columns, strict=True))


def wrist_view(eef_state):
    """The wrist camera, as its published image shows the scene, in the world frame for a measured tool pose."""
    a, b = np.asarray(eef_state[3:6], float), np.asarray(eef_state[6:9], float)
    a = a / np.linalg.norm(a)
    b = b - a * (a @ b)
    b = b / np.linalg.norm(b)
    tool = np.eye(4)
    tool[:3, :3], tool[:3, 3] = np.stack([a, b, np.cross(a, b)], axis=1), eef_state[:3]
    return WRIST.on(tool)


def region_evidence(target, reference, after, hand_from=None):
    """Did a selected region stay put while the hand measurably rose or sank?

    Compares the region in two observations from FIXED_CAMERA, with no observation
    of it in between, against the hand's travel since hand_from (default: the reference).
    Returns None when they cannot be compared. An unchanged region did not travel with the
    hand: the hand holds something else or nothing, or the selection sits on another object.
    A changed one establishes nothing, since occlusion and tracking drift change regions
    too. Identity and retention stay the policy's judgement.
    """
    old = next((t for t in reference.get("targets", []) if t.get("target") == target), None)
    new = next((t for t in after.get("targets", []) if t.get("target") == target), None)
    if old is None or new is None or new.get("camera") != FIXED_CAMERA:
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
TARGETS = dict(type="object", description="Native move_to pose dimensions, metres and rot6d columns; no gripper.")
RANGE = dict(
    type="array",
    items=dict(type="number"),
    minItems=2,
    maxItems=2,
    description="Expected measured finger separation [min,max] in mm, with 4 <= min < max <= 78.",
)
POINT = dict(type="object")
TRAVEL = dict(type="number", description="Height in metres to rise to before moving level.")
YAW = dict(type="number", description="Turn of the finger axis about the vertical: 0 along world y, 90 along world x.")
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
        "locate",
        "Move the hand level until it is over a region selected in the wrist camera and return the "
        "region's world x, y, z and size. select binds target first, in the same call.",
        dict(target=TEXT, select=SELECTION, note=TEXT),
        ["target", "note"],
    ),
    schema(
        "grasp",
        "Open, move above your approach pose (above the region itself when it was selected in the wrist "
        "camera), descend to the pose or to contact above it, close, check measured opening, and test "
        "lift by 3-5 cm. Stops on failed evidence or measured checks; does not carry automatically. "
        "select binds object_target first, in the same call.",
        dict(
            object_target=TEXT,
            select=SELECTION,
            offset=dict(
                type="array",
                items=dict(type="number"),
                minItems=2,
                maxItems=2,
                description="[dx, dy] in metres from the centre of a wrist-selected region to the grasp point.",
            ),
            yaw_deg=YAW,
            approach=TARGETS,
            lift=TARGETS,
            expect_grip_mm=RANGE,
            note=TEXT,
        ),
        ["object_target", "approach", "lift", "expect_grip_mm", "note"],
    ),
    schema(
        "place",
        "Rise to travel_z, move above your release pose, or over via when given, and approach the pose "
        "or contact short of it while monitoring grip, then return BEFORE opening so you can inspect fresh "
        "views. Withdrawal is only "
        "executed after a separate release decision. select_destination binds destination_target first, "
        "in the same call.",
        dict(
            object_target=TEXT,
            destination_target=TEXT,
            select_destination=SELECTION,
            travel_z=TRAVEL,
            via=dict(POINT, description="{x, y, z} to pass through just before the release pose and on the way out."),
            move=TARGETS,
            withdraw=TARGETS,
            expect_grip_mm=RANGE,
            note=TEXT,
        ),
        ["object_target", "move", "withdraw", "expect_grip_mm", "note"],
    ),
    schema(
        "push",
        "Close the fingers, go to start through free space, then move in a straight line to end or to "
        "where contact stops the hand, and back off 2 cm. For shutting a drawer or door or sliding an object.",
        dict(
            start=dict(POINT, description="{x, y, z} where the line begins, clear of what is pushed."),
            end=dict(POINT, description="{x, y, z} where the line ends, at or beyond where the pushing should stop."),
            yaw_deg=YAW,
            travel_z=TRAVEL,
            note=TEXT,
        ),
        ["start", "end", "note"],
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
        self.located = {}  # where each target was last found; forgotten once it is reselected or taken
        self.outlined = set()  # targets selected with a box around them, whose region is the whole object

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
        self.located.pop(target, None)
        self.outlined.discard(target)
        if where.get("box") is not None:
            self.outlined.add(target)
        return self.tracking.select(
            current[where["camera"]].id, target, point=where.get("point"), box=where.get("box"), space="image"
        )

    def _centre(self, name, stage, floor_z=None):
        """Move the hand level until it is over a region tracked in the wrist camera.

        Returns the region's position and shape, or the Outcome that stopped the search. Every
        move is a counted native move that ends once the hand is there: a look needs the pose it
        is taken from, not a hand at rest. Given the lowest height a grasp allows, the first move
        heads for a little above it and the moves also open the fingers; otherwise they leave
        them alone. A position found earlier for a region not reselected since is confirmed by
        one look instead of being worked out again.
        """
        hand = dict(gripper=0) if floor_z is not None else {}

        def region():
            return next((t for t in self.observe()["targets"] if t["target"] == name), {})

        def look():
            seen, state = region(), self.driver.observation.state["eef_state"]
            visible = seen.get("valid") is True and bool(seen.get("center"))
            return Look(wrist_view(state), state[:3], *((seen["center"], seen.get("bbox")) if visible else ()))

        short = []

        def move(goal):
            targets = dict(x=float(goal[0]), y=float(goal[1]), **hand)
            command = self._move(targets, stage, expected=OPEN_MM if hand else None)
            moved = self.driver.execute(dict(command, rest=False))
            if moved.data.get("arm") == "not_approaching" and not short:
                # A first guess can be past the arm's reach. The hand rests where it stopped, and a look
                # from there is still a new look. A second such move ends the search.
                short.append(moved)
                return Outcome("done", moved.kind, "stopped short of a guessed goal; looking again from here")
            return moved

        found = locate(
            move,
            look,
            arrive_m=ARM_TOLERANCE[0],
            assumed_z=None if floor_z is None else floor_z + 0.05,
            known=self.located.get(name),
        )
        if not found.ok:
            return found
        data = found.data
        self.located[name] = [data[k] for k in "xyz"]
        extent = data.pop("extent_m", None)
        if extent:
            data["extent_mm"] = [round(1000 * side) for side in extent]
            view = wrist_view(self.driver.observation.state["eef_state"])
            depth = float(view.project(self.located[name])[1][0])
            shape = footprint(view, region().get("dominant_region") or {}, depth)
            if shape:
                data.update(across_mm=round(1000 * shape["across_m"]), along_mm=round(1000 * shape["along_m"]))
                if shape["along_m"] >= LONGER * shape["across_m"]:
                    # The finger axis turned from world y toward world x, as grasp's yaw_deg is, in [-90, 90).
                    yaw = math.degrees(math.atan2(shape["across"][0], shape["across"][1]))
                    data["fit_yaw_deg"] = round((yaw + 90) % 180 - 90)
        return data

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

    def _procedure(self, outcome, *bound, **facts):
        feedback = procedures.feedback(outcome)
        return self._result(dict(feedback, **facts), dict(procedure_brief(feedback, bound), **facts))

    def _approach(self, command, facts):
        """Execute a stage, noting how far short of its pose an approach that met contact ended."""
        outcome = self.driver.execute(command)
        if "short_m" in outcome.data:
            # Where the approach ended says what it met: the object, the surface or something else.
            facts.update(approach_ended_short_m=outcome.data["short_m"])
        return outcome

    @staticmethod
    def _point(value, name):
        if (
            not isinstance(value, dict)
            or value.keys() != {"x", "y", "z"}
            or any(type(v) not in (float, int) or not math.isfinite(v) for v in value.values())
        ):
            raise ValueError(f"{name} is a point: x, y and z in metres")
        return {k: float(v) for k, v in value.items()}

    @staticmethod
    def _yaw(args):
        yaw = args.get("yaw_deg")
        if yaw is not None and (type(yaw) not in (float, int) or not -90 <= yaw <= 90):
            raise ValueError("yaw_deg is between -90 and 90")
        return yaw

    def _move(self, targets, stage, *, monitor=None, expected=None, settle=SETTLE_STEPS):
        return dict(
            kind="move", targets=targets, stage=stage, settle_steps=settle, monitor_mm=monitor, expect_mm=expected
        )

    @staticmethod
    def _travel(args):
        travel = args.get("travel_z")
        if travel is not None and (type(travel) not in (float, int) or not math.isfinite(travel)):
            raise ValueError("travel_z must be a height in metres")
        return travel

    def _push(self, args):
        """Fingers closed, to start through free space, then straight to end or to where contact stops the hand."""
        start, end = self._point(args["start"], "start"), self._point(args["end"], "end")
        line = np.array([end[k] - start[k] for k in "xyz"])
        length = float(np.linalg.norm(line))
        if length < PUSH_M[0] or length > PUSH_M[1]:
            raise ValueError(f"a push runs {PUSH_M[0] * 100:g}-{PUSH_M[1] * 100:g} cm from start to end")
        turn = {} if (yaw := self._yaw(args)) is None else turned(yaw)
        stages = [
            self._move({"gripper": 1}, "push_close"),
            dict(self._move({**end, **turn}, "push"), travel_z=self._travel(args), via=start),
        ]
        pushed = {}
        for command in stages:
            outcome = self._approach(command, pushed)
            if not outcome.ok:
                return self._result(outcome.feedback(), dict(status=outcome.status, **outcome.data))
        here = np.array(self.driver.observation.state["eef_state"][:3])
        short = pushed.get("approach_ended_short_m", 0.0)
        # Leave what was pushed: back along the line, but never past where the push began.
        back = here - line / length * min(BACK_OFF_M, max(length - short, 0.0))
        retreat = self.driver.execute(self._move(dict(zip("xyz", map(float, back), strict=True)), "push_back_off"))
        told = dict(
            status=retreat.status,
            pushed=dict(
                ended="stopped_on_contact" if "approach_ended_short_m" in pushed else "reached_end",
                travelled_mm=round(1000 * (length - short)),
                short_mm=round(1000 * short),
            ),
            **self.observe(),
        )
        return self._result(told)

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
            if call.name == "locate":
                self.pending = None
                name = args["target"]
                camera = args["select"].get("camera") if isinstance(args.get("select"), dict) else None
                if (camera or self.tracking.targets.get(name, {}).get("camera")) != "wrist":
                    raise ValueError("locate needs the target selected in the wrist camera")
                if "select" in args:
                    self._select(name, args["select"])
                found = self._centre(name, "locate")
                if isinstance(found, Outcome):
                    return self._result(found.feedback())
                return self._result(dict(status="done", located=found, **self.driver.measure()))
            if call.name == "grasp":
                self.pending = None
                grip = self._range(args["expect_grip_mm"])
                for key in ("approach", "lift"):
                    self._validate_pose(args[key])
                name = args["object_target"]
                ran = []
                camera = args["select"].get("camera") if isinstance(args.get("select"), dict) else None
                wrist = (camera or self.tracking.targets.get(name, {}).get("camera")) == "wrist"
                # Centring over a wrist region supplies x and y; nothing else does.
                if not ({"z"} if wrist else {"x", "y", "z"}) <= args["approach"].keys() or "z" not in args["lift"]:
                    raise ValueError(
                        "grasp needs approach z and lift z, and approach x, y unless the object is selected "
                        "in the wrist camera"
                    )
                # To the micrometre: 0.54 - 0.49 is a little over 0.05 in binary.
                rise = round(float(args["lift"]["z"] - args["approach"]["z"]), 6)
                if not 0.03 <= rise <= 0.05:
                    raise ValueError("the test lift rises 3-5 cm: lift z minus approach z")
                offset = args.get("offset", [0, 0])
                if (
                    not isinstance(offset, list)
                    or len(offset) != 2
                    or any(type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 0.1 for v in offset)
                ):
                    raise ValueError("offset is [dx, dy] in metres, each within 0.1")
                if any(offset) and not wrist:
                    raise ValueError("offset is measured from a region selected in the wrist camera")
                approach = dict(args["approach"], gripper=0)
                if (yaw := self._yaw(args)) is not None:
                    approach.update(turned(yaw))
                if "select" in args:
                    self._select(name, args["select"])

                found = {}

                def run(command):
                    if command.get("stage") == "grasp_approach" and wrist:
                        centred = self._centre(name, "grasp_centre", float(args["approach"]["z"]))
                        if isinstance(centred, Outcome):
                            ran.append(False)
                            return centred
                        found.update(centred={k: v for k, v in centred.items() if k != "pixel"})
                        over = dict(x=centred["x"] + offset[0], y=centred["y"] + offset[1])
                        fit = centred.get("fit_yaw_deg")
                        fits = fit is not None and centred["across_mm"] <= FIT_MM
                        if yaw is None and not any(offset) and fits and name in self.outlined:
                            # Fingers across the narrow side close on it squarely instead of turning it first.
                            # A point may select only part of an object, whose narrow side is not the object's.
                            over.update(turned(fit))
                            found.update(fingers_turned_to_yaw_deg=fit)
                        command = dict(command, targets=dict(command["targets"], **over))
                    outcome = self._approach(command, found)
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
                    approach=dict(self._move(approach, "grasp_approach", expected=OPEN_MM), via={}),
                    # Fingers closing across a box that is not square to them turn it first, slowly.
                    close=self._move({"gripper": 1}, "grasp_close", expected=grip, settle=CLOSE_STEPS),
                    check=dict(kind="check_grip", expect_mm=grip),
                    # The hand may have been centred, and stopped on contact above the requested height:
                    # rise straight up from where it is.
                    lift=dict(self._move({}, "grasp_test_lift", monitor=grip), rise_m=rise),
                    assess=lifted,
                )
                if len(ran) > 1:
                    self.located.pop(name, None)  # fingers that closed may have moved it
                return self._procedure(outcome, name, **found)
            if call.name == "place":
                self.pending = None
                grip = self._range(args["expect_grip_mm"])
                for key in ("move", "withdraw"):
                    self._validate_pose(args[key])
                travel = self._travel(args)
                if "via" in args:
                    self._point(args["via"], "via")
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

                move = dict(
                    self._move(args["move"], "place_approach", monitor=grip), travel_z=travel, via=args.get("via", {})
                )
                facts = {}
                return self._procedure(self._place(args, move, pause, facts), *bound, **facts)
            if call.name == "push":
                self.pending = None
                return self._push(args)
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
                    pending["args"], dict(kind="confirm_state", fingerprint=pending["fingerprint"]), approved, {}
                )
                kept = pending["args"]
                return self._procedure(
                    outcome, *(kept[k] for k in ("object_target", "destination_target") if k in kept)
                )
            raise ValueError("unknown tool")
        except (KeyError, ValueError, TypeError) as exc:
            return self._result(dict(status="refused", message=str(exc)))

    def _place(self, args, move, assess, facts):
        def run(command):
            if command.get("stage") == "place_withdraw" and "via" in args:
                # Out the way it came in, before anything else.
                back = self.driver.execute(self._move(args["via"], "place_withdraw"))
                if not back.ok:
                    return back
            return self._approach(command, facts)

        return procedures.place(
            run,
            self.observe,
            move=move,
            release=self._move({"gripper": 0}, "place_release", expected=OPEN_MM),
            check=dict(kind="check_grip", expect_mm=OPEN_MM),
            withdraw=self._move(args["withdraw"], "place_withdraw"),
            assess=assess,
        )

    def close(self):
        self.tracking.close()
