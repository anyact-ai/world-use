# Changelog

## 0.3.0 (unreleased)

### Highlights

- **MuJoCo simulation.** The twin that checks every plan, on hardware too, is a MuJoCo model with the robot's
  meshes, frictional grasps and rendered cameras; objects can slip, tip and fall. `wu demo` runs a scripted block
  task in it, with a success check and failure scenarios.
- **Flight records and replay.** Every session records plans, outcomes, telemetry and pictures in chunks that
  survive a killed process. `wu inspect`, `wu replay`, `wu fit` and the optional Rerun viewer (`wu view`, the
  `rerun` extra) read them, a live run included. See [records](docs/records.md).
- **Measuring from pictures.** `camera_frame` and `measure_pixels` over MCP, or `Client.frame` and
  `Client.measure`, turn pixels into work-frame metres, and `run(requires=...)` stops a plan once a measurement is
  too old or its camera was calibrated again. `wu mcp --vision` adds EdgeTAM tracking. Depth comes from simulated
  cameras. See [measuring from pictures](docs/perception.md).
- **Other arms.** A workcell loads a robot from a URDF and a TOML description, with its own driver. Grippers with
  one joint, revolute or prismatic, and five-joint arms (the tool's heading turns; `ik_weights` sets what a move
  holds) work, and `package://` mesh paths resolve. See the [adapter guide](docs/adapters.md).
- **Python 3.11 or newer** (0.2.0 needed 3.14). The reBot driver extra needs 3.11 to 3.13.
- New commands: `wu policy`, `wu calibrate`, `wu record`, `wu inspect`, `wu replay`, `wu view`, `wu fit`,
  `wu demo`; new MCP tools: `policy`, `job`, `calibrate`, `reset`, `shutdown` and the perception tools.

### Breaking changes

- A job command's exit status is its outcome: 0 done, 4 refused, surprise, stopped, faulted or cancelled, 5
  waiting at a checkpoint, 6 still running. An invalid or refused request exits 2, and 1 means the daemon or the
  MCP server could not start.
- `wu run` and `wu check` take the plan as JSON or a file. `wu run --checked`, `Client.run(checked=True)` and the
  daemon's shared "last checked" plan are gone.
- `wu events` returns at most `--limit` events (40 by default) and says where to continue.
- `wu home-route` checks the route from where the arm is and exits 4 if the kernel would refuse it. The HTTP
  API requires `steps`; `null` clears the route.
- MCP `status` returns its one line, with the full state as structured content. `run`, `job`, `answer` and `home`
  return the same short text as the CLI, naming the tool to call next.
- `grip` and `grasp` no longer take `effort`, `lag`, `speed` or `min`: their thresholds come from the gripper
  description, whose `approach` and `opens_along` are now required.
- `line` and `guarded` also take `back`, `right` and `down`, and refuse both words of a pair. Unknown frames and
  joints are refused before anything moves instead of faulting the kernel.
- Prismatic arm joints are refused, also by `Chain`. `Kernel(ik_weights=...)` is gone: set `ik_weights` in the
  robot description.
- The kinematic simulator and its `lag_s` and `stiffness` options are gone. A custom URDF needs inertias and
  collision geometry, and headless camera rendering needs EGL or OSMesa.
- Records use format 3: telemetry chunks under `tape/` and a `session.json` with the robot model; a built-in
  robot's meshes are not copied. `wu inspect` and `wu fit` still read 0.2.0 records. `Tape.save` is gone.

### Fixes

- A thermal return switches torque off at rest even when a job was queued, and refuses new jobs while it runs.
- Simulated joints hold their pose with torque off (gearbox friction, or brakes on a self-supporting arm).
- The daemon answers every request, applies a world change completely or not at all, and refuses at startup a
  robot its twin cannot model. `wu up` waits for a slow start and says why one failed.
- A refused step inside a plan leaves no velocity feedforward behind, and long queues cancel cleanly.
- `wu fit` fits only friction that opposes motion. Slow-zone refusals give a duration that would pass.
- Camera capture timeouts are reported, and calibrating a resized 360-camera view is correct.
- An empty grasp is declared only after the final gripper feedback; fragile zones refuse invalid thresholds.
- Recording is bounded and runs off the control thread; a storage failure never blocks release or shutdown.
- The daemon refuses requests from foreign hosts and origins, and non-JSON commands.
- MuJoCo warnings go to Python logging instead of a stray `MUJOCO_LOG.TXT`.

Procedure tools developed after 0.2.0 were never released; the measuring flow above replaces them.

## 0.2.0

Previous public release. The September 29 thermal-recovery changes were merged after
that tag; 0.3.0 includes them as well as the changes above. See the
[hardware records](docs/hardware.md) for the observations behind the recovery contract.
