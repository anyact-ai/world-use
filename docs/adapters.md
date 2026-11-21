# Adding a robot

A robot needs a URDF, a TOML description, and a driver with six methods. Keep these
in your own project or Python package. The daemon loads the driver by its import
path; the planning worker uses the description, without loading hardware code.

The [Panda kitchen adapter](../examples/robocasa-counter/adapter.py) implements the same six methods over a
RoboCasa scene; its [setup guide](../examples/robocasa-counter) says what it supports and where it turns off
rehearsal. A MuJoCo task can get its lifecycle and recording from
[`SimulationSession`](simulation.md#external-simulation-sessions).

## Try the complete path

The [planar example](../examples/adapters) is a two-joint arm with an in-memory
position controller. From a checkout, first try its built-in simulation:

```sh
uv run wu up --workcell examples/adapters/workcell.toml --body sim
uv run wu card
uv run wu check '[{"do":"joints","delta_deg":{"1":10,"2":-5}}]'
uv run wu down
```

Then use the example's own adapter. `PYTHONPATH` makes the local example module
importable; an installed driver package doesn't need it.

```sh
PYTHONPATH=examples/adapters uv run wu up --workcell examples/adapters/workcell.toml
uv run wu enable
uv run wu run '[{"do":"joints","delta_deg":{"1":10,"2":-5}}]'
uv run wu release
uv run wu down
uv run wu inspect runs/YOUR-RUN
uv run wu replay runs/YOUR-RUN --out replay.gif
```

Neither run touches hardware: `--body sim` is MuJoCo, and the example's own adapter is an in-memory
controller. Together they exercise setup, rehearsal in the worker, execution and records. For an embedded
Python example, run `uv run python examples/adapters/planar.py`.

The planar arm has no gripper. [tests/jaw_arm.toml](../tests/jaw_arm.toml) describes a five-joint arm with a
revolute jaw, and its [simulation tests](../tests/test_mujoco.py) cover grips, finger pads on fixed child links,
release, and Cartesian moves that let the heading turn. To adapt a whole task before changing the robot, follow
the [block workcell recipe](../examples/pick-place/README.md#adapt-the-task): change the scene and destination,
compose the phases, and check the released object separately from whether the commands completed.

## Describe the robot

Copy [planar.toml](../examples/adapters/planar.toml) and substitute your measured
specifications. `urdf` is relative to that file. `tool_link` identifies the point
that `line`, `lines` and `move_to` steps control. Each `[[joints]]` entry must follow the URDF chain's
joint order; its limits may be tighter than the URDF's, never wider.

The worker prepares every motion on a MuJoCo twin, on hardware too, so the description must
load in MuJoCo: physically valid inertias, collision geometry, and STL or OBJ meshes.
Mesh paths are relative to the URDF. A ROS `package://name/...` path resolves in the
nearest folder called `name` that holds the URDF or sits beside a folder above it;
otherwise loading fails and names the path. Joint positions must follow the URDF's
zero and sign. A driver that reports another convention, such as LeRobot's
normalized -100..100 joints and 0..100 gripper, converts in `read` and `command`.

The runtime supports revolute and continuous arm joints. Prismatic arm joints are
rejected before a driver is loaded: joint steps, limits, telemetry and torque
monitoring use rotational units.

The fields follow [Manifest](../src/world_use/body.py):

| Fields | Units / meaning |
| --- | --- |
| Joint `lower`, `upper`, `track_tol` | Radians |
| Joint `v_max`, `a_max` | Radians/s and radians/s² |
| Joint `tau_max`, `tau_hold_max`, `contact_dtau` | Torque thresholds in Nm |
| Joint `excursion_exempt` | `true` exempts a joint that swings nothing, such as wrist roll, from `max_excursion` |
| `rate_hz`, `speed`, `auto_accel`, `min_move_s` | Control rate and default motion timing |
| `sensing` | `position`, optionally `torque`, `temperature`, `gripper_effort` |
| `[gripper]` | See [grippers](#grippers) |
| `temp_warn_c`, `temp_limit_c` | Temperature thresholds in °C |
| `max_segment_m`, `link_radius_m`, `max_excursion` | Cartesian segment length, collision padding, joint excursion in radians |
| `[turn_clearance]` | Joint names in `joints`, required clearance in `height_m` |
| `ik_weights` | Weights of x, y, z, rx, ry, rz (base frame) that Cartesian moves hold; 0 frees one |
| `notes`, `hardware_notes` | Lines for the card. `hardware_notes` describe the physical robot, not its simulation: name the motor supply there, which an operator switches off when the arm cannot go home |

An arm with fewer than six joints cannot hold every tool orientation. Unless
`ik_weights` says otherwise, it holds position and tilt and lets the heading (yaw
about the base's vertical axis) turn, so lines go sideways and `move_to` reaches
around the base. Set `ik_weights = [1, 1, 1, 1, 1, 1]` to keep the heading as well.

Advertise only sensing the driver returns; `None` means unavailable, and zero means
a measured zero. With position alone, the kernel has no torque to judge contact by:
`touchdown`, `guarded` and contact monitoring are unavailable, and a collision shows
only as tracking error beyond a joint's `track_tol`.

### Grippers

```toml
[gripper]
closed = -0.17                 # Native units: what the driver reads and commands.
open = 1.74
approach = [1.0, 0.0, 0.0]     # Tool-frame direction the fingers point.
opens_along = [0.0, 1.0, 0.0]  # Tool-frame axis the jaws open along.
# m_per_unit = 0.03            # Opening in metres per native unit; enables millimetre commands.
```

`approach` and `opens_along` are required: the card and `move_to` "point" use them.
`unit` (default `rad`) names the native unit on the card, and `tool_point` says in
words where the tool link sits (default: between the fingertips). Unset `v_max`,
`track_tol` and `squeeze` scale with the travel between `closed` and `open`;
`tau_max` is an absolute effort in the driver's units.

The simulation drives the joints outside the arm chain. One joint, revolute or
prismatic, is commanded in its URDF coordinate, so `closed` and `open` must lie in
its URDF range, and `m_per_unit` is optional. Two opposed prismatic fingers that
start at zero share the opening along `opens_along`; they need `m_per_unit`. Other
mechanisms need their own simulation model. Without `[gripper]`, joints outside the
arm chain stay at their URDF zero.

The steps that check the measured opening are described under [measured completion](#measured-completion).

### Collision bounds

Declare conservative boxes in the `tool_link` frame for tools and fingers that
extend beyond the tool point. Each finger box must cover its whole travel; separate
boxes can leave the opening between fingers free. For example, after measuring a
tool, replace these illustrative dimensions and source:

```toml
[[tool_boxes]]
name = "probe"
role = "tool"                  # tool or finger
center = [0.0, 0.0, -0.025]    # metres in the tool frame
size = [0.01, 0.01, 0.05]      # full extents, metres
rpy = [0.0, 0.0, 0.0]         # radians
source = "caliper measurement of mounted probe"
```

Planning and every measured tick check these boxes, and the box of a held object, against keep-out zones and
support surfaces. A held object's box keeps the offset from the tool it had when it was gripped, in the kernel's
estimated world, not in simulator truth. Planning includes the corners a rotating payload sweeps through.

A guarded move that approaches a support nearly along its normal may plan a short way past the surface, so its
contact search can reach it. Every part that reaches that support wholly above its footprint, such as both
fingers, shares this allowance; overhanging parts and other geometry keep their clearance. The allowance ends
with the step or its interruption.

The card and rehearsal report list each box's `source` and what is not checked. Without `tool_boxes`, no tool
or finger geometry is checked. Self-collision, the pedestal, arm links against surfaces, and collisions with
objects that are not held are never checked. The boxes do not change simulator geometry.

### Rest or brakes

For an arm without brakes, describe the pose where its weight is supported:

```toml
[rest]
q = [0.0, 0.5]                 # One position per joint, in manifest order.
joints = ["shoulder", "elbow"] # The load-bearing joints that must be at rest.
tol = 0.1
# stops = ["elbow"]            # Only if this joint rests on a physical stop.
```

Home targets `q` within `tol`, joint planning limits, and clearance from declared
stops. It refuses when no supported target satisfies those constraints.

An arm that cannot fall when its torque is off, such as a braked arm or the in-memory example, sets
`self_supporting = true` instead. A missing rest declaration is an error. Do not copy another arm's limits or
folding pose.

Check a rest declaration in simulation before using it. With torque off, a
simulated joint holds only about 0.3 Nm (gearbox friction), so a rest pose that
needs motor torque sags there as well:

```sh
uv run wu up --workcell my_workcell.toml --body sim
uv run wu enable
uv run wu run '[{"do":"line","up":0.05}]'
uv run wu home-route '[]' && uv run wu home
uv run wu release
uv run wu status               # Again after a few seconds: the joints should not move.
uv run wu down
```

## Configure the workcell

```toml
body = "my_robot.driver:Arm"
robot = "arm.toml"
# fit = "fit.json"             # Optional model fitted from this robot's records.

[body_options]
channel = "can0"               # Passed to Arm(manifest=..., channel="can0").

[simulation]
start_deg = [0.0, 28.65]        # Used by --body sim; q accepts radians.

[[frame]]
name = "work"
origin = [0.0, 0.0, 0.0]       # Metres in the URDF base frame.
rpy_deg = [0.0, 0.0, 0.0]

[[camera]]
name = "side"
path = "images/side.jpg"
max_age_s = 3.0
```

Paths for the robot, fit and camera images are relative to the workcell file;
`~` expands to your home directory. Camera commands run as supplied. See the
[reBot guide](rebot.md#configure-a-camera-and-workcell) for camera sources and
calibration. `[[box]]` and `[[fact]]` entries describe the scene; the
[block workcell](../src/world_use/workcells/block.toml) shows boxes, and
[scene limits](simulation.md#scene-and-model-assumptions) says what the simulation builds from them.

`fit` names a file written by [`wu fit`](records.md). It is refused if its link names, joint names or number of
friction entries do not match this robot's URDF.

`[envelope]` holds operator overrides, applied to execution and rehearsal.
`max_excursion_deg` with `reason` replaces the manifest's `max_excursion`. For robots
with a `turn_clearance` rule, `turn_height_m` sets an absolute tool height in the
work frame (metres), with `turn_reason` explaining the scene clearance; it replaces
the default height relative to the starting pose.

`--body sim` simulates the same robot description with the `[simulation]` settings
([simulation of other robots](simulation.md#other-robots)); a hardware driver's `body_options` are not passed
to it.

Unknown fields, duplicate names and conflicting camera sources are rejected.
`wu up` also checks the robot, URDF and fit contents before reusing a daemon.
Configuration is local, trusted input: selecting a driver imports Python code.

## Implement I/O

The driver constructor accepts `manifest` and its connection options. It stores
that manifest and opens no connection until `connect`. The
[Body contract](../src/world_use/body.py) defines the methods:

| Method | Responsibility |
| --- | --- |
| `connect` | Open read-only and return measured state; no motion or release |
| `enable` | Engage at the measured position; report incomplete power transitions |
| `read` | Return fresh measurements or a timestamp-preserving cache; raise on lost feedback |
| `command` | Send one bounded position/velocity/gripper command |
| `disable` | Confirm torque-off, or raise with the unconfirmed motors |
| `close` | Close the connection without changing motor power |

The kernel calls I/O from one thread. Keep `read` and `command` within the control
period. Freshness comes from the time the driver received a sample, or a sequence counter, not from whether the
values change. Test constant fresh samples, frozen caches, disconnects, and partial power transitions with a fake
transport before connecting hardware.

Check the card, frames, motions, stop, home, release and records in simulation first. Neither the simulation
nor the example says how your physical arm behaves: validate the driver on it in a supervised session.

## Measured completion

A joint or gripper trajectory ends once its last command has been sent, whether or not the hand has got there.
These steps check measured state instead. Each failure is a surprise: the arm holds where it is and the steps
after it are cancelled.

- `{"do": "settle"}` waits at the final joint command until every measured joint is within `tolerance_deg` of
  it (default: one sixth of the joint's `track_tol`), for at most `seconds` (default 1). It ends early as a
  surprise when the error stops shrinking, which can mean contact, controller lag or an unreachable target.
  The outcome gives the `reason` (`arrived`, `stalled` or `timeout`) and each joint's error. Kernel watchdogs
  keep running during the wait.
- `gripper` waits 0.3 s after its trajectory and ends `done` with `reason: "command_complete"` and the target
  and measured positions, wherever the fingers are. With `check_arrival: true`, it instead requires the
  measured opening to stay within one sixth of the gripper's `track_tol` for 0.1 s before those 0.3 s run out.
  `grip` opens to its start width this way before it closes.
- `check_grip` watches the measured opening for `seconds` (default 0.2). It fails if the opening leaves
  `expect_mm` (or `expect` in native units), or if its peak-to-peak change exceeds `max_drift_mm` (or
  `max_drift`). Millimetre values need the gripper's `m_per_unit`, and a gripper without a measured position
  is refused. The gripper keeps its command when the check fails.
- `grip` and `grasp` take `verify_s` to run `check_grip` for that long after squeezing, with the same drift
  options. Their `expect_mm` or `expect` range then applies to these readings too, so allow for the object
  giving under the squeeze. For `grasp`, a failed check is a miss like any other: it retries at its next
  `search_mm` offset.
- A `seq` with `expect_grip_mm` (or `expect_grip`) checks the measured opening against that range on every
  tick of its steps, checkpoint waits included. Put lift and carry steps in it, and open the gripper outside it.

Grip checks report `object_retention_verified: false`. The opening cannot tell whether the object is still
held: a steady opening can hold nothing, and a compliant or reseating object can change the opening without
being lost. Check the object in a camera. The
[block example](../examples/pick-place#check-a-grip-before-carrying) combines these checks with a visual one.

### Checks for an external controller

`world_use.checks` provides the same checks for a controller outside the kernel, such as a simulator's own.
They send no commands.

`Settling(tolerances, timeout_s, window_s, progress_fraction=0.05, stable_s=0, stall_s=0)` decides when a wait
for a final target ends. Call `update(stamp, errors, command_active=True)` with each new sample: `stamp` is
the sample's time on a monotonic clock, and `errors` the absolute remaining errors, one per tolerance and in
its units (one vector can mix metres and radians). It returns `None` to keep waiting, or:

- `"arrived"` once every error has been within its tolerance for `stable_s`;
- `"stalled"` when, over at least `window_s` and three samples, each error still outside its tolerance shrank
  by less than `progress_fraction` of that tolerance and its current rate would not bring it in before the
  deadline;
- `"timeout"` `timeout_s` after the first sample.

A sample whose `stamp` has been seen before counts as a cached reading and never ends a wait. Without new
samples nothing calls `update`, so the adapter's stale-data watchdog must still catch lost feedback. Pass
`command_active=False` while an approver or controller is still changing the applied target: the wait then
cannot arrive or stall, its progress window and arrival timer restart, and the deadline still applies.

On a lightly damped arm a tolerance alone is not enough. A hand passing through its target is within
tolerance without having stopped: add its travel over the last control step as one more error, with the
travel you accept at rest as its tolerance. An arm that swings past its target makes no progress while it
turns round: `stall_s` is how long a stall must last before `update` reports it.

`Following(position, lead_m=..., progress_m=...)` detects a position-controlled hand that something holds
back, without torque sensing. After each control step, call `update(position, command)` with the measured
position and the command that step applied. It returns `True` once, two steps running, the command was more
than `lead_m` ahead of the hand and the hand gained less than `progress_m` toward it. Use it to end a move
early instead of pressing until a timeout, or to end a descent on contact. It does not say what holds the
hand, and while the command is within `lead_m` it never reports. After each update, `travel_m` is the hand's
travel over the last step: the extra error for `Settling` above.

A hand that closes on an object has no target opening to arrive at. Give `Settling` the fingers' remaining
travel to fully closed instead: `"arrived"` then means the fingers closed on nothing, `"stalled"` that they
stopped apart, and `"timeout"` that they were still moving. Compare the settled opening with the expected range
afterwards. The [LIBERO runner](../examples/libero/native.py) uses these checks for its arm and fingers.

## Grasp and place procedures

`world_use.procedures` runs a grasp or a placement as a fixed series of commands, observing between them. The
[hand tools](hand-tools.md) build their grasp and place on it. It
runs in your procedure's process and sends nothing to the robot itself: each command is a dictionary passed
unchanged to your `execute(command)`, which validates it, runs it through the normal control path (for example
`Client.run` or the MCP `run` tool) and returns its result as an `Outcome`. `observe()` returns a fresh
observation as a dictionary.

| Call | Stages, in order | Observations |
| --- | --- | --- |
| `grasp(execute, observe, approach=, close=, check=, lift=)` | `approach`, `close`, `check_before_lift`, `test_lift`, `check_after_lift` | `before`, `before_close`, `before_lift`, `after` |
| `place(execute, observe, move=, release=, check=, withdraw=)` | `arrive`, `release`, `check_release`, `withdraw` | `before`, `before_release`, `after` |

`grasp` runs `check` twice, before and after the test lift. `before` is observed ahead of the first command,
and `after` after the last. The commands carry their own checks: end the approach or move with `settle`, run
the test lift (and, for `place`, the move) in a `seq` with `expect_grip_mm`, keep the lift short, and make
`check` a `check_grip`. For example:

```python
from world_use.procedures import place

outcome = place(execute, observe,
                move={"do": "seq", "expect_grip_mm": [35, 45],
                      "steps": [{"do": "move_to", "to": [0.34, -0.07, 0.12]}, {"do": "settle"}]},
                release={"do": "gripper", "aperture_mm": 65},
                check={"do": "check_grip", "expect_mm": [60, 70]},
                withdraw={"do": "line", "up": 0.08})
```

The first command that does not end `done` ends the procedure. It observes once more (as `after`) and returns
that command's status, expected, observed and hint, with its message prefixed by the stage. It never opens,
retreats or retries by itself, so an uncertain grip never leads to an automatic opening over a payload. An
exception from `execute`, `observe` or `assess` propagates at once, and no further command is sent.

A `done` result means every command and check passed, not that the object is held or placed:
`completion_verified` is always `False`. Judge that from the observations, and send any correction through the
same executor.

Pass `object_target="cup"`, and for `place` also `destination_target="shelf"`, to tie the procedure to tracked
targets. Each checkpoint's observation must then hold `targets`, a list of tracking results with `target`
and `valid` fields, such as `observe_targets` returns. If a named target is missing or not `valid`, the procedure is
refused before its first command or stopped at that observation. The name is your claim about which object
the region is; a tracker follows a region and does not recognise objects.

Pass `assess(checkpoint, observations)` to decide at each checkpoint after `before` whether to continue. It
receives the checkpoint's name and a copy of the observations so far, and returns an `Outcome`: `done` lets
the next stage run; return `stopped` when the observation cannot be judged and `surprise` for a measured
mismatch. Any status but `done` ends the procedure with that outcome, so a failed check at `before_release`
keeps the gripper closed. At `after`, where no stage follows, it may return `None` to abstain. The result's
`assessment` is `"passed_caller_checks"` when the assessor passed at `after`, and `"unknown"` without an
assessor or when it abstained. An assessment does not refresh the measurements behind it: keep them in the
`requires` of later commands so the kernel checks their age again before moving.

The [perception example](../examples/perception) assesses a placement with calibrated depth: before release it
compares the measured block with its target, and after withdrawal it checks where the block was left. When the
check before release fails, the task lifts, corrects the target and calls `place` again. With RGB tracking only, mask overlap
cannot show that the right object is in the right place; have the policy judge that from the images.

`outcome.data` holds every stage's command and outcome (`stages`), the `observations` by checkpoint, the bound
targets (`intent`), `assessment` and every `assessments` entry, the last `checkpoint`, and `next_stage`, the
first stage that has not completed (`None` once all have). `next_stage` is not a point to resume from: the
procedure never replays completed stages, and recovery is a new command or call through the executor. Archive
`outcome.to_dict()` to keep the whole history.

`procedures.feedback(outcome)` makes a short report for a policy: the status and message, each stage's status,
`checkpoint`, `next_stage`, the latest assessment, and two observations, the latest and a reference
(`before_lift` for a grasp when there is one, otherwise `before`), so the policy can compare the object before
and after. A failure adds the last command with its outcome, and the failure's `expected`, `observed` and
`hint`. Motor power stays the caller's responsibility.

`async_place(...)` takes the same arguments with awaitable `execute` and `observe`; `assess` may be either. It
runs in the caller's event loop. Cancelling it stops later stages but not a command that a remote executor has
already accepted: find out how that job ended and resolve motor power. Do not run a synchronous procedure in a
background thread, which could keep issuing commands after its caller is cancelled.

## Guarded custom steps

A plan that `requires` measurements runs with a guard: before each step starts to move, the kernel checks that
the measurements are still current ([freshness](perception.md#freshness)). Custom steps are refused in such a
plan, because their internal moves would go unchecked. Embedded code can admit a custom step type in a kernel
subclass once that type checks the guard itself:

```python
class MyKernel(Kernel):
    guarded_steps = Kernel.guarded_steps | {MyMove}
```

The set holds exact types; a subclass of a listed type is not admitted. The step must be started with
`kernel.start_behavior`, and each move inside it, including one after a wait or after background preparation,
must pass `kernel.start_behavior` or `kernel.check_guard()` before it commands motion. Keep perception and
planning out of `tick`.
