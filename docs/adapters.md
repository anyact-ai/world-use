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

## Describe the robot

Copy [planar.toml](../examples/adapters/planar.toml) and substitute your measured
specifications. `urdf` is relative to that file. `tool_link` identifies the point
Cartesian actions control. Each `[[joints]]` entry must follow the URDF chain's
joint order; its limits may be tighter than the URDF's, never wider.

The runtime supports revolute and continuous arm joints. Prismatic joints are
rejected before a driver is loaded: joint actions, limits, telemetry and torque
monitoring use rotational units. `Chain` supports prismatic joints for offline
kinematics, but that does not provide runtime support for linear actuators.

The fields follow [Manifest](../src/world_use/body.py):

| Fields | Units / meaning |
| --- | --- |
| Joint `lower`, `upper`, `track_tol` | Radians |
| Joint `v_max`, `a_max` | Radians/s and radians/s² |
| Joint `tau_max`, `tau_hold_max`, `contact_dtau` | Torque thresholds in Nm |
| `rate_hz`, `speed`, `auto_accel`, `min_move_s` | Control rate and default motion timing |
| `sensing` | `position`, optionally `torque`, `temperature`, `gripper_effort` |
| `[gripper]` | Native-unit limits, aperture conversion, approach and jaw axes |
| `temp_warn_c`, `temp_limit_c` | Temperature thresholds in °C |
| `max_segment_m`, `link_radius_m`, `max_excursion` | Cartesian segment length, collision padding, joint excursion in radians |
| `[turn_clearance]` | Joint names in `joints`, required clearance in `height_m` |

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
pose. Advertise only sensing the driver returns; `None` means unavailable, and
zero means a measured zero.

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

For robots with a `turn_clearance` rule, an operator can set
`[envelope].turn_height_m` to an absolute tool height in the work frame (metres),
with `turn_reason` explaining the scene clearance. This replaces the default
height relative to the starting pose and applies to execution and rehearsal.

`--body sim` uses the same robot description with `[simulation]` settings. Driver
options stay with the driver. Existing `sim`, `sim:rebot`, `rebot` and
`[body_options]` simulation workcells continue to work. The built-in reBot driver
uses its own model; a different robot uses its own adapter and description.

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
control period. Driver receipt timestamps or sequence counters establish freshness;
changing values do not. Test constant fresh samples, frozen caches, disconnects,
and partial power transitions with a fake transport before connecting hardware.

Check the card, frames, motions, stop, home, release and records in simulation.
Then validate your driver on the actual arm in a supervised session. The example
proves the integration path; it does not establish another arm's physical behavior.
