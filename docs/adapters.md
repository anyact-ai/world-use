# Adding a robot

A robot needs a URDF, a TOML description, and a driver with six methods. Keep these
in your own project or Python package. The daemon loads the driver by its import
path; the planning worker uses the description, without loading hardware code.

For an external simulation, the [Panda kitchen adapter](../examples/robocasa-counter/adapter.py)
shows the same I/O contract connected to a RoboCasa scene. Its
[setup guide](../examples/robocasa-counter) explains what the integration supports and where
it disables rehearsal. MuJoCo tasks can share lifecycle and recording through
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

Both runs are simulated. This exercises setup, worker rehearsal, execution and
portable records without connecting hardware. For an embedded Python example,
run `uv run python examples/adapters/planar.py`.

To adapt a complete task before substituting the robot, follow the
[block workcell recipe](../examples/pick-place/README.md#adapt-the-task): change
the initial scene and destination, compose phases, then evaluate the released
object separately from command completion. The planar example has no gripper;
it demonstrates the adapter path rather than a block-transfer task. A second
fixture, [jaw_arm.toml](../tests/jaw_arm.toml), supplies five joints and a revolute
jaw. Its [simulation tests](../tests/test_mujoco.py) exercise grips, fixed child
finger pads, release and Cartesian motion with free heading. Neither fixture
establishes the dynamics or safe power behavior of a physical robot.

## Guarded custom steps

Embedded kernels may extend `Kernel.guarded_steps` with audited behavior types:
`guarded_steps = Kernel.guarded_steps | {MyMove}` in a kernel subclass. Admission checks exact types, so this
does not admit subclasses automatically. A custom primitive must start through `kernel.start_behavior`;
every internal submove, including after waiting or background preparation, must pass `start_behavior` or
`check_guard` before commanding motion. Ordinary custom steps remain refused when a plan has a guard.
Keep perception and planning outside the control tick.

### Measured completion

Finishing a joint trajectory means its commands were sent. Add `{"do": "settle"}`
when the next step needs measured arrival. It waits at the final joint command
for at most one second, returning sooner on arrival or insufficient progress.
`seconds` changes the cap; `tolerance_deg` sets the allowed joint error. Without
it, each joint uses one sixth of its declared tracking tolerance. Stalls and
timeouts report residuals, hold the measured arm pose and cancel following work.
They do not establish whether contact, controller lag or an unreachable target
caused the error. Kernel watchdogs continue to run throughout the wait.

External controllers can use `world_use.checks.Settling(tolerances, timeout_s,
window_s, stable_s=0)` with their own measured residuals, including a vector mixing
metres and radians. `update(feedback_timestamp, absolute_errors,
command_active=...)` returns `None`, `"arrived"`, `"stalled"` or `"timeout"`.
Use a monotonic measurement timestamp and declare tolerances in the residuals'
units. Cached samples cannot complete the wait; missing feedback still needs the
adapter's stale-data watchdog. Set `command_active=False` while an approver or
controller is still changing the applied target: this clears the progress window
without extending the deadline. A full window with at least three measurements
is needed for a stall, and slow progress projected to arrive before the deadline
keeps its remaining time. This checker sends no commands and changes no controller.

## Describe the robot

Copy [planar.toml](../examples/adapters/planar.toml) and substitute your measured
specifications. `urdf` is relative to that file. `tool_link` identifies the point
that `line`, `lines` and `move_to` steps control. Each `[[joints]]` entry must follow the URDF chain's
joint order; its limits may be tighter than the URDF's, never wider.

The MuJoCo twin prepares every motion, also on hardware, so the description must
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
| `notes`, `hardware_notes` | Lines for the card. `hardware_notes` describe the physical robot, not its simulation: name its motor supply there, which an operator switches off when the arm cannot go home |

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

`gripper` now checks measured arrival after its trajectory, with a maximum settling
wait of 0.3 s. Arrival requires 0.1 s within one sixth of the gripper's tracking
tolerance; a plateau or timeout ends with a surprise and holds the measured hand
and arm positions. Previously this step always returned done after the wait. Use
`grip` for intentional object contact; a position command alone does not verify a grasp.

`check_grip` observes opening over a bounded interval (default 0.2 s), using an
explicit `expect_mm` range or native-unit `expect`. Optional `max_drift_mm` or
`max_drift` bounds the peak-to-peak change during that interval. Millimetre inputs
need calibration; missing measurements refuse the check. Range or drift failures
hold the arm and retain the gripper command, then cancel following work. Sensor
noise, reseating and compliance affect these readings: consistency does not prove
object retention, and a change does not prove loss. The feedback states that
object retention remains unverified.

`grip` and `grasp` accept `verify_s` to run the same check after squeezing, with
optional drift bounds. Their `expect_mm`/`expect` range then also applies to the
post-squeeze measurements, so allow for the object's compliance. A failed check
uses `grasp`'s existing bounded offsets; it does not add new search directions.
For subsequent lift/carry steps, `seq` accepts `expect_grip_mm` or `expect_grip`
and checks that range every tick, including checkpoint waits. Put intentional
opening outside that sequence. See the [block example](../examples/pick-place#check-a-grip-before-carrying)
for composing measured checks with an independent visual check.

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

Planning and measured ticks check these boxes against keep-out zones and support
surfaces. The estimated held-object box uses its declared attachment transform;
it does not reveal simulator state. Planning includes the swept corners of a
rotating payload. A guarded, nearly normal approach can select one support for
the existing bounded contact search. Every part that reaches it wholly above its
footprint, such as two fingers, shares that allowance; overhanging parts and other
geometry keep their clearance.
That allowance ends with the step or interruption. It is a search allowance under
contact monitoring, not evidence that physical penetration is acceptable.

The card and rehearsal report list declared sources and omissions. Empty
`tool_boxes` means no tool/finger bounds were declared. Self-collision, the
pedestal, arm-link/surface collision and unattached objects as obstacles remain
outside these checks. Declaring bounds does not change simulator geometry.

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

Otherwise explicitly set `self_supporting = true`. Use that only when disabling
cannot make the robot fall, such as a braked arm or the in-memory example. A
missing rest declaration is an error. Do not copy another arm's limits or folding
pose.

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
[block workcell](../src/world_use/workcells/block.toml) shows boxes.

Fit files reject unknown fields, nonfinite values, nonpositive link masses and negative friction. Applying
a fit checks its link names and requires one friction pair per robot joint, in URDF order. Older fits without
joint names still use that positional order; supplied joint names must match it exactly.

`[envelope]` holds operator overrides, applied to execution and rehearsal.
`max_excursion_deg` with `reason` replaces the manifest's `max_excursion`. For robots
with a `turn_clearance` rule, `turn_height_m` sets an absolute tool height in the
work frame (metres), with `turn_reason` explaining the scene clearance; it replaces
the default height relative to the starting pose.

`--body sim` uses MuJoCo with the same robot description and `[simulation]` settings;
see [simulation requirements](simulation.md#other-robots). Driver options stay with
the driver. The built-in reBot driver uses its own model; a different robot uses its
own adapter and description.

The native environment is a set of static surface boxes and free object boxes.
`known = false` on an object or surface puts it in physics while leaving it out of the kernel's
estimate. Adding a kernel box later changes only that estimate. A different
robot's URDF can supply shaped collision geometry, but articulated or shaped
environment objects require their own environment integration; see
[scene limits](simulation.md#scene-and-model-assumptions).

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

The kernel calls I/O from one thread. Keep `read` and `command` bounded to the
control period. The time the driver received a sample, or a sequence counter,
establishes freshness; changing values do not. Test constant fresh samples, frozen caches, disconnects,
and partial power transitions with a fake transport before connecting hardware.

Check the card, frames, motions, stop, home, release and records in simulation.
Then validate your driver on the actual arm in a supervised session. The example
proves the integration path; it does not establish another arm's physical behavior.
