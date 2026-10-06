# Adding a robot

A robot needs a URDF, a TOML description, and a driver with six methods. Keep these
in your own project or Python package. The daemon loads the driver by its import
path; the planning worker uses the description, without loading hardware code.

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
