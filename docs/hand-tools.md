# Hand tools for an agent

`world_use.hand_tools` gives an agent eight tools: `select_target`, `observe_targets`, `locate`, `grasp`,
`close_grasp`, `place`, `release_placement` and `push`. Each call runs a measured sequence, such as a grasp from
approach to test lift, and returns what was measured. The agent still decides what every region is, where the
hand goes and when to release. These are the tools behind the [LIBERO-10 results](../examples/libero), where
they cut model calls by more than half; LIBERO's arm and any robot the daemon runs use the same code.

They are plain Python calls for an agent loop of your own:

```python
from world_use import Client
from world_use.hand_tools import DaemonArm, Hand, HandTools
from world_use.tracking import Tracking
from world_use.vision_worker import TrackerProcess

client = Client()
hand = Hand(opening_mm=89, open_mm=(80, 89), fit_mm=80, arrive_m=0.012, hand_camera="hand",
            fixed_camera="side", in_view_m=0.3, hover="15-25 cm", max_lean_deg=90, standoff_m=0.05,
            stands_up=False)
tools = HandTools(DaemonArm(client, hand), hand, Tracking(client, TrackerProcess()))

tools.schemas()   # name, description and JSON Schema parameters of each tool, to register with your agent
tools.docs()      # what the agent needs to know beyond each description, with this hand's numbers
record, reply = tools.call("grasp", {"object_target": "mug", ...})   # full record, and the agent's brief JSON
```

They are not MCP tools yet. Over MCP, the `run` tool's plans still give an agent the same motions one step at a
time.

## What the tools do

- **Positions from a camera on the hand.** `locate`, and a `grasp` of a region selected in the hand camera,
  move the hand level and look from several places until it is over the region
  ([parallax](perception.md#positions-from-a-camera-on-the-hand)). They report the region's position and
  size, which way the image lies in the world (`image_right`, `image_down`), and for a region clearly longer
  than wide the yaw that puts the fingers across its narrow side. A grasp turns the fingers that way by itself.
- **Approaches that may end on contact.** The approach z is the lowest height the agent allows; contact short
  of it ends the approach there, and `approach_ended_short_m` says how far short.
- **Leaning grasps.** With `tilt_deg` and `toward`, or an offset from a hand-camera region, the hand turns where
  it is, goes `standoff_m` back along its own axis and comes in along that axis. This is how it takes a mug by
  its handle or reaches in under a roof. Before the hand turns, the executor checks that it can make the lean
  along that path, and refuses it otherwise.
- **Lift and carry evidence.** A region tracked in `fixed_camera` that stayed where it was while the hand rose
  or travelled vertically means the hand holds something else or nothing: the grasp returns a `surprise` and
  `place` issues no checkpoint.
- **Checkpoints before closing and opening.** `place` stops before opening and returns a one-use checkpoint,
  void after any motion or reselection; `release_placement` opens and withdraws. A grasp with
  `stop_before_close` stops the same way with the fingers open at its approach pose, so the policy can look
  before committing; `close_grasp` closes, checks and lifts.
- **Pushes.** `push` closes the fingers, goes to a start point through free space, follows a line to its end or
  to contact, and backs off along it.

## Executors

Every motion goes through an executor, which turns the tools' commands into its arm's own motion and knows
what that arm can reach. The [`Executor`](../src/world_use/hand_tools.py) protocol and the module docstring
describe the commands.

- `DaemonArm(client, hand)` runs each command as one job on a robot the daemon runs, rehearsed before it is
  admitted. Positions are in the robot's base frame, and moves go in straight legs no longer than the robot's
  `max_segment_m`. `move_to` chooses which way round the jaws go, the one that keeps the wrist off its stops,
  and a lean is refused when its rehearsal is. An approach that may end on contact is a `guarded` move over
  its last segment, so it needs joint torque sensing; without it the approach is a plain move. Between jobs
  the arm holds where it is, as it does between any two agent calls. A release checkpoint is void once
  anything commands the arm, from any client, and every observation is taken afresh. A daemon that fails
  during a call ends it `stopped`, after whatever ran before.
- The [LIBERO executor](../examples/libero/world_tools.py) drives the published LIBERO arm's absolute-pose
  commands. Its finger and lean rules were measured on that arm, because LIBERO reports no joint positions.

`Hand` describes the gripper and cameras. `stands_up` says whether the hand's resting orientation points the
fingers straight down; with it, the tools stand a leaning hand up again to look down or after a push. An arm
that grasps from the side, like the reBot, keeps its lean.

## Limits

- A pinch on a handle does not resist turning: what it holds can turn in the fingers with no change in their
  opening. On the twin's reBot, a box-built mug taken by its handle swings about 27 degrees as it rises.
- The tools measure; they do not recognise objects or check that a task is done.
- Evidence from images needs a fixed camera that does not look along the vertical.
- The tool set's sequences run in the caller's process. If it stops mid-sequence, the arm holds at a job
  boundary.
