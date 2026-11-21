"""World Use tools for the LIBERO arm, from RGB images alone. The model decides what each region is,
where the hand goes and when to release.

The tools are world_use.hand_tools; this file gives them LIBERO's arm (its absolute-pose commands with rot6d
columns, its wrist camera and the wrist rules measured on it) and keeps the model-facing schemas and text."""

from __future__ import annotations

import copy
import json
import math

import numpy as np
from inspect_robots.types import Action, ActionChunk
from inspect_robots_agent._tools import ToolResult
from native import ARM_TOLERANCE, fingerprint

from world_use.cameras import View
from world_use.geometry import aim
from world_use.hand_tools import Hand, HandTools, brief
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
is in the wrist image you see next. Check there that it lies on what you meant. image_right and
image_down are the level world directions that right and down in that image point along: an
offset toward something you see there is its distance times that direction. Use it once for
a destination such as a basket or plate before carrying anything there; the position holds
while that object is not moved. z reads low, by up to 1.5 cm on flat things and 6 cm on tall ones
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
the centre of the wrist region, so one finger goes inside, with the approach z 2-3 cm below the
rim top instead of a low z. Choose a side clear of any handle and of neighbours, which the hand
body meets up to 10 cm along the finger axis, and turn the fingers with yaw_deg to reach it.
An upright hand stays above what it holds, 10 cm and more over the grip, so it cannot take a
thing in under a roof or shelf lower than that. grasp's tilt_deg, up to 60, leans the hand
instead: the fingertips point down and toward `toward`, a level point {x, y}, with the finger
axis level across that direction. The hand turns where it is, then comes in along its own axis
from 10 cm back to the approach x, y, z or to contact short of it, which takes about 50 steps
more than an upright grasp. Its body stays behind and above the grip, on the side it came from,
and it keeps the lean afterwards: what it holds is ahead of the fingertips and goes in under a
roof first, where a place with via takes it. The wrist leans toward +y, -y or -x, where the
robot's base is, and no further round toward +x than 10 degrees from +y or -y. In recorded
trials it reached 45 degrees each of those ways and 60 only toward -y. yaw_deg does not go with
tilt_deg. The result's leaning
gives tilt_deg and toward, the level unit direction the fingertips point.
To take an object by its handle, lean the hand. Locate the whole object in the wrist camera,
then give grasp an offset from the region's centre to the middle of the handle's top, outside
the body's wall: about half the region's along_mm toward the handle. With an offset and no
toward the hand leans toward the region's centre, so it comes at the handle from outside,
pinches it across its thickness and carries the object upright ahead of it. That pinch does
not resist turning: if what it holds meets anything on the way, it turns in the fingers with
no change in their opening, so look at how it hangs before going on. Use tilt_deg 45
and an approach z at the handle's top, 3-4 cm below a mug's rim: in recorded trials a mug's
handle held for offsets of 5 to 8 cm and grip heights 5.5 to 8.5 cm above the table, and with
an offset of 4 cm the fingers closed on its body. Without an offset, name toward yourself.
A leaning hand's wrist camera looks along the lean, not down; locate and a wrist-selected grasp
stand the hand upright again on their first move.
place rises straight up to travel_z when the hand is below it, moves level to above your release
pose, descends with the grip monitored and stops early on contact the same way, then returns
BEFORE opening with a one-use checkpoint. Its approach_ended_short_m tells a landing on
the surface you meant from one on something higher, such as a rim. Choose a travel_z that clears
everything on the way, the destination's rim included, by more than the held object hangs below
the fingertips. To come at the release pose from the side instead of from above, into a shelf or
an oven, give via: the hand moves level to over that point, down to it and then straight to the
pose, stopping early on contact along that line. Put via in front of the opening at the height
of the pose, far enough out that the hand and what it holds are clear of the frame. After the
release the hand goes straight to your withdraw pose, so choose one that takes the fingers off
the object before any move sideways, back along its own axis for a leaning hand, as far up as
back at 45 degrees, and leave an enclosure with move_to. Inspect the views, then
call release_placement to open and withdraw, or correct the pose. Any motion or reselection
voids the checkpoint. Two objects for one container go side
by side along its longer extent: release the first a fifth of that extent to one side of the
located centre and the second a fifth to the other side, so neither lands on the other or on
a wall.
push closes the fingers and moves the fingertips in a straight line from start to end, to shut a
drawer or a door or to slide something along a surface. For a line that runs more level than
up or down the hand turns its finger axis across it and leans 30 degrees back from it, so the
fingertips lead and the hand body stays behind them, clear of what is above the thing pushed;
tilt_deg sets another lean, from 0 for an upright hand to 60. A push whose line points further
round toward +x than 10 degrees from +y or -y stays upright, since the wrist does not lean that
way. It goes to start through free
space, rising to travel_z first when you give one, so put start a few centimetres clear of what
you push, on the side you push from, with the fingertips at the height of its middle. The line
ends at end or where contact stops the hand: pushed says which, with travelled_mm and short_mm.
Put end a few centimetres past where the thing should stop, since a drawer or door that reaches
its stop holds the hand there. The hand then backs off up to 10 cm along the line, standing
upright again as it goes, also after a push that failed on its way, whose result says how that
went in back_off. A hinged door moves along an arc: push it along two or three straight
lines.
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


# The wrist camera in the tool frame, from the embodiment's camera mount: a 75 degree field of view
# looking along the approach axis from 5 cm along tool y and 9.7 cm behind the grasp point, so the
# grasp point is not at the image centre. Published wrist images have their rows flipped, which the
# negative focal length and mirrored row offset say.
WRIST = View(
    [[-1.0, 0, 0, 0], [0, -1.0, 0, 0.05], [0, 0, 1.0, -0.097], [0, 0, 0, 1.0]], 166.8128, -166.8128, 128, 127, 256, 256
)
HAND = Hand(
    opening_mm=78,
    open_mm=(70, 85),  # fingers this far apart have let go, or made room, even while they still creep
    fit_mm=70,  # 78 mm of opening less clearance
    arrive_m=ARM_TOLERANCE[0],
    hand_camera="wrist",
    # Does not move and does not look along the vertical, so vertical travel of a held object shifts its image.
    fixed_camera="agentview",
    in_view_m=0.3,  # from a hover of 15-25 cm the wrist camera sees no further to the side than this
    hover="15-25 cm",
)
# Steps a settled command may take. Raised high above the table the arm needs about 25 steps to stop
# swinging; fingers closing across a box that is not square to them turn it first, which takes longer.
SETTLE_STEPS = {"move": 40, "pass": 40, "close": 120}
# Fingers along a direction and along its opposite close on the same thing, half a turn of the wrist
# joint apart, and that joint turns 166 degrees each way. For an upright hand it was measured near the
# middle of its range with the finger axis 20 to 70 degrees from world y toward -x, and 8 degrees from its
# stop with the axis along +x at the table's left side.
FINGERS_MIDDLE = (-math.sin(math.radians(45)), math.cos(math.radians(45)), 0.0)
FINGERS_DEG = 110  # fingers() keeps the finger axis it chooses within this turn of FINGERS_MIDDLE
# The wrist cannot bend the fingertips away from the robot's base, toward +x. Leaning 25 degrees that way
# over the middle of the table, the hand rested 14 mm off its pose; 45 degrees was out of reach. Between
# +x and the y axis one lean was measured, halfway round toward +y, which the arm reached; until more
# are, a lean stays within 10 degrees of the y axis on that side.
OUTWARD = 0.17  # the x part of a level unit direction beyond which a lean points away from the base
SWING_DEG = 120  # a finger axis turning further than this turns halfway first


ROTATION = ("r00", "r10", "r20", "r01", "r11", "r21")  # the finger axis, tool x, then tool y


def fingers(eef_state, along):
    """An upright hand's finger axis along a level direction or along its opposite: the one nearer its
    present finger axis, or nearer FINGERS_MIDDLE when both are a quarter turn away, and in either case the
    other one if that is more than FINGERS_DEG from FINGERS_MIDDLE. Chosen by nearness alone, turn after turn
    adds up to the wrist joint's stop; chosen with no regard to the present axis, the hand turns half round
    where it need not move."""
    axis = np.array([along[0], along[1], 0.0]) / math.hypot(along[0], along[1])
    near = float(axis @ np.asarray(eef_state[3:6], dtype=float))
    if (axis @ FINGERS_MIDDLE if abs(near) < 0.1 else near) < 0:
        axis = -axis
    return axis if axis @ FINGERS_MIDDLE >= math.cos(math.radians(FINGERS_DEG)) else -axis


def upright(axis):
    """Pose targets for an upright hand with its finger axis, tool x, along a level unit direction."""
    x, y = float(axis[0]), float(axis[1])
    return dict(zip(ROTATION, (x, y, 0.0, y, -x, 0.0), strict=True))


def approach_axis(eef_state):
    """The unit direction the fingers point along, tool z, for a measured or commanded tool pose."""
    a, b = np.asarray(eef_state[3:6], dtype=float), np.asarray(eef_state[6:9], dtype=float)
    axis = np.cross(a, b)
    return axis / np.linalg.norm(axis)


def wrist_view(eef_state):
    """The wrist camera, as its published image shows the scene, in the world frame for a measured tool pose."""
    a, b = np.asarray(eef_state[3:6], float), np.asarray(eef_state[6:9], float)
    a = a / np.linalg.norm(a)
    b = b - a * (a @ b)
    b = b / np.linalg.norm(b)
    tool = np.eye(4)
    tool[:3, :3], tool[:3, 3] = np.stack([a, b, np.cross(a, b)], axis=1), eef_state[:3]
    return WRIST.on(tool)


class Arm:
    """The tools' executor on LIBERO's arm: every command goes through the driver's counted steps."""

    failures = ()  # the simulator runs in this process: its errors stop the episode

    def __init__(self, driver):
        self.driver = driver
        self.native = driver.tools

    @property
    def state(self):
        return self.driver.observation.state["eef_state"]

    def execute(self, command):
        """Run one of the tools' own commands. A turn in place to a lean turns the fingers first, upright, and
        then leans: turned in one move the raised arm swings several centimetres, which reads as a hand held
        back."""
        if command.get("in_place"):
            targets = command["targets"]
            fingers_first = dict(command, targets=dict(targets, **upright([targets["r00"], targets["r10"]])))
            there = self._execute(fingers_first)
            if not there.ok:
                return there
        return self._execute(command)

    def _execute(self, command):
        """A finger axis that has to swing more than SWING_DEG first turns halfway, upright, where the hand is
        and before any leg of the command, round the side FINGERS_MIDDLE is on: rot6d columns interpolated
        across a half turn pass through zero, and the arm then turns either way round, into a joint's stop as
        often as not."""
        command = {k: v for k, v in command.items() if k != "in_place"}
        if "settle" in command:
            command = dict(command)
            settle = command.pop("settle")
            command["settle_steps"] = SETTLE_STEPS[settle]
            if settle == "pass":  # arrival at the pose, without waiting for the hand to rest there
                command["rest"] = False
        targets = command.get("targets") or {}
        if all(k in targets for k in ROTATION[:3]):
            now = np.asarray(self.state[3:6], dtype=float)
            goal = np.array([targets[k] for k in ROTATION[:3]], dtype=float)
            if float(now @ goal) < math.cos(math.radians(SWING_DEG)):
                between = np.array([now[0] + goal[0], now[1] + goal[1], 0.0])
                if np.linalg.norm(between) < 0.2:  # opposite axes: a quarter turn from both
                    between = np.array([-goal[1], goal[0], 0.0])
                between = between / np.linalg.norm(between) * (1 if between @ FINGERS_MIDDLE >= 0 else -1)
                halfway = self.driver.execute(
                    dict(
                        kind="move",
                        targets=upright(between),
                        stage=command["stage"],
                        settle_steps=SETTLE_STEPS["move"],
                        monitor_mm=None,
                        expect_mm=None,
                    )
                )
                if not halfway.ok:
                    return halfway
        return self.driver.execute(command)

    def measure(self):
        return self.driver.measure()

    def position(self, observation):
        return np.asarray(observation["eef_state"][:3], dtype=float)

    def pose(self):
        return np.asarray(self.state[:3], dtype=float), approach_axis(self.state)

    def orientation(self, point, jaws=None):
        """rot6d targets for an approach axis and a finger axis. Without one the hand keeps the finger axis it
        has. An upright hand's finger axis is chosen by fingers(); a leaning hand keeps tool y, the side its
        camera is on, underneath. That way round the wrist joint stayed within 1.9 rad of the middle of its
        range for every lean the arm reached. The other way round it ran into its stop leaning toward -y,
        and leaning toward +y at the height a grasp turns at it rested 13 mm off its pose, outside
        ARM_TOLERANCE."""
        point = np.asarray(point, dtype=float)
        if jaws is None:
            a, b = np.asarray(self.state[3:6], dtype=float), np.asarray(self.state[6:9], dtype=float)
            rotation = aim(np.column_stack([a, b, np.cross(a, b)]), [0, 0, 1], [1, 0, 0], point)
            return dict(zip(ROTATION, map(float, rotation[:, :2].T.ravel()), strict=True)), None
        jaws = np.asarray(jaws, dtype=float)
        if abs(float(np.cross(point, jaws)[2])) < 1e-9:  # upright
            axis = fingers(self.state, jaws)
            return upright(axis), axis
        if np.cross(point, jaws)[2] > 0:
            jaws = -jaws
        rotation = np.column_stack([jaws, np.cross(point, jaws), point])
        return dict(zip(ROTATION, map(float, rotation[:, :2].T.ravel()), strict=True)), jaws

    def lean_refusal(self, direction, tilt_deg, path):
        if direction[0] / math.hypot(direction[0], direction[1]) > OUTWARD:
            return (
                "the wrist cannot lean the fingertips away from the robot's base: "
                "this lean points more than 10 degrees round from +y or -y toward +x",
                "lean toward +y, -y or -x, or grasp upright",
            )
        return None

    def hand_look(self, frame=None):
        # The paused simulator's frames and state are of one step.
        return wrist_view(self.state), np.asarray(self.state[:3], dtype=float)

    def cameras(self):
        self.driver.frames.sync(self.driver.observation)
        return list(self.driver.frames.current)

    def frame(self, camera):
        self.driver.frames.sync(self.driver.observation)
        return self.driver.frames.current[camera]

    def validate_pose(self, targets):
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

    def fingerprint(self):
        return fingerprint(self.driver.observation)

    def record(self, name, row):
        self.driver.record(name, row)

    def key(self):
        return self.driver.steps


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
        "camera), descend to the pose or to contact above it, or with tilt_deg come in leaning along the "
        "hand's own axis, close, check measured opening, and test lift by 3-5 cm. Stops on failed evidence "
        "or measured checks; does not carry automatically. select binds object_target first, in the same call.",
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
            yaw_deg=dict(
                type="number",
                description="Turn of the finger axis about the vertical: 0 along world y, 90 along world x.",
            ),
            tilt_deg=dict(
                type="number",
                description="Lean of the hand from upright, 0 to 60; it then comes in along its own axis.",
            ),
            toward=dict(
                type="object",
                description="{x, y}: the level point the fingertips of a leaning hand point toward.",
            ),
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
            via=dict(POINT, description="{x, y, z} to pass through just before the release pose."),
            move=TARGETS,
            withdraw=TARGETS,
            expect_grip_mm=RANGE,
            note=TEXT,
        ),
        ["object_target", "move", "withdraw", "expect_grip_mm", "note"],
    ),
    schema(
        "push",
        "Close the fingers, go to start through free space, then move the fingertips in a straight line to "
        "end or to where contact stops the hand, and back off up to 10 cm. For shutting a drawer or door or sliding "
        "an object.",
        dict(
            start=dict(POINT, description="{x, y, z} where the line begins, clear of what is pushed."),
            end=dict(POINT, description="{x, y, z} where the line ends, at or beyond where the pushing should stop."),
            tilt_deg=dict(type="number", description="Lean of the hand back from the push, 0 to 60; 30 if omitted."),
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
        tracking = Tracking(
            driver.frames, tracker or TrackerProcess(device="cpu", max_age_s=1800, timeout_s=90, startup_timeout_s=120)
        )
        self.tools = HandTools(Arm(driver), HAND, tracking)

    def schemas(self):
        return self.native.schemas() + copy.deepcopy(SCHEMAS)

    def state_labels(self):
        return self.native.state_labels()

    def residual(self, *args):
        return self.native.residual(*args)

    def _result(self, record, text):
        """Record the full result and return the policy its brief copy."""
        self.driver.record("world-use.jsonl", record)
        self.driver.record_observation()
        # This marker returns control to the episode loop without a simulator step.
        return ToolResult(
            note=text,
            chunk=ActionChunk(
                actions=[Action(data=self.driver.observation.state["eef_state"].copy(), meta={"runner_no_step": True})]
            ),
        )

    def execute(self, call, observation):
        if call.name in ("done", "give_up"):
            self.tools.pending = None
            return self.native.execute(call, observation)
        try:
            args = json.loads(call.arguments)
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
            if not isinstance(args.get("note"), str) or not args["note"].strip():
                raise ValueError("note is required")
            if call.name == "move_to":
                self.tools.pending = None
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
                return self._result(result.feedback(), json.dumps(brief(told), separators=(",", ":")))
        except (KeyError, ValueError, TypeError) as exc:
            told = dict(status="refused", message=str(exc))
            return self._result(told, json.dumps(brief(told), separators=(",", ":")))
        return self._result(*self.tools.call(call.name, args))

    def close(self):
        self.tools.close()
