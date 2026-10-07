# Flight records

Every `wu up` session, `wu demo` and the perception example write a flight record: one folder under `./runs` (or
`$WORLD_USE_RUNS`) with the plans, outcomes, telemetry and pictures, and the robot model they ran on. The commands
below read records offline. None of them opens a robot driver or runs a plan again.

```sh
wu inspect runs/block-demo                            # a summary; --json for everything
wu replay runs/block-demo --out replay.gif --speed 3  # a GIF of the recorded joints and world model
wu view runs/block-demo                               # the record in Rerun
wu view runs/block-demo --out block-demo.rrd          # the same, saved to a file
wu fit runs/RUN-1 runs/RUN-2                          # link masses and joint friction, from recorded runs
```

`wu inspect` prints the robot, the mode, whether the record closed normally, the powered and moving time, and each
job's outcome. `powered` counts from each attempt to switch torque on to a confirmed switch-off, ramps and
unconfirmed intervals included. `moving` is the time spent in motion steps, not independently detected movement.
Both are elapsed time on one monotonic clock that events and telemetry share.
For current records, "closed normally" means the final completion marker matches the committed chunk and event
counts. A `closed` event alone means the robot connection closed; its final recording write may still have failed.
An open or interrupted record reports only the telemetry already committed to disk.

`wu replay` renders the measured joints and the recorded world model with the saved robot geometry, and labels the
frames as a reconstruction. The original camera pictures stay in `views/`. `wu fit` writes `fit.json`; the
[2026-09-28 hardware record](hardware.md#2026-09-28-a-model-fitted-from-flight-records) shows what a fit changes.

## View in Rerun

`wu view` needs the `rerun` extra. The README's install line exposes both `wu` and `rerun`; `wu view` also finds the
viewer inside its tool environment. Rerun runs in its own process and reads the record; closing it never stops a
job or changes motor power.

The window has a 3D view of the robot's own geometry, a camera pane, measured and commanded joint plots, torque,
temperature, gripper, power and the active job, and the events, all on the run's `elapsed` timeline. Drag the
timeline to inspect a grasp, a refusal or a recovery. Joint plots keep every committed sample; the world and the
last ten seconds of tool path update at up to 20 Hz.

The 3D scene shows **measured joints and the estimated world**: a carried box is what the runtime believed it held,
not recovered physical motion. Camera pictures show the captured scene, with any overlays `wu look` drew. A
[measurement](perception.md) adds its picture to its camera's pane at the time the picture was taken, and its
surface points to the 3D scene, in green, from the time it was measured.

The viewer moves the arm joints and a calibrated gripper with two prismatic fingers. Other gripper joints stay where
the URDF puts them, with a warning in the event pane.

## Follow a live run

```sh
wu up --workcell block
wu view
```

With no folder, `wu view` follows the local daemon's record, including edits made through `wu box` or other
world tools, usually one to two seconds behind control. It stays connected across completed sessions,
waits through daemon restarts and opens the next session as a separate named recording. Ctrl+C stops
following; the Rerun window stays open. The daemon must be running when you first start `wu view`.

Supplying a folder opens that saved recording. Add `--follow` to follow it until completion; it does not
switch to other runs. A portable export with `--out` also stays with one recording. Camera pictures arrive
when a client calls `wu look` or measures; nothing streams video. Scene-file edits take effect when loaded
by a new daemon, not merely when saved. The folder must be readable where the viewer runs: for a remote
daemon, copy or mount its record.

## Headless export

```sh
wu view runs/block-demo --out block-demo.rrd
rerun block-demo.rrd
```

Export needs no display, daemon or robot driver. The `.rrd` file holds the meshes, measurements, pictures, events
and layout; open it on any machine with Rerun 0.38. `--follow --out session.rrd` exports a growing run until it
completes or Ctrl+C. The interactive viewer needs a graphics backend: on a cloud host, export and open the file
locally. Simulated cameras have their own rendering needs; see [simulation](simulation.md).

## What a record holds

| file | contents |
| --- | --- |
| `session.json` | package version, source and URDF hashes, the body and its mode, the daemon's startup identity and workcell, and the initial snapshot: robot description, world, commanded and measured state, fitted model |
| `robot.urdf` | the geometry used by this run; its limits and gripper description are in `session.json` |
| `assets/`, `robot-assets.json` | a custom robot's meshes and their original paths; for a built-in robot, its license notices only |
| `events.jsonl` | submitted plans, outcomes, checkpoint questions and answers, contacts, world changes, picture paths, notes |
| `tape/000000.npz`, ... | telemetry and power transitions in chunks; together the complete history |
| `summary.json`, `world.json` | the summary and world model at the last explicit save or normal close |
| `complete.json` | the final chunk and event counts, written only when the record closed cleanly |
| `recording.json` | how many samples, events and power transitions were lost, only if a buffer overran |
| `views/` | saved camera pictures |
| `perception/<id>.png`, `.npz` | one pair per measurement: the picture with the measured pixels drawn on it, and the measured `points` (base frame, metres) with the `pixels` they came from |

A record stays usable after the run folder moves or the robot's driver is uninstalled. A built-in robot's meshes
ship with world-use, and a record finds them when its URDF matches the installed one exactly.

Records written by world-use 0.2.0 hold their telemetry in one `tape.npz` and have no `session.json`: `wu inspect`
and `wu fit` read them; replay and the viewer cannot. In Python, `world_use.recorder.load_tape(run_folder)` returns
the complete arrays of either kind.

## How it is written

A background thread saves events and telemetry once per second, off the control loop. A killed process loses at
most the current second and an unfinished event line; earlier chunks stay readable. Storage failures never fault the
robot or block release and shutdown; `wu status --json` reports them.

The daemon keeps telemetry it has not yet saved in a bounded buffer: 30,000 samples (five minutes at 100 Hz), 5,000
power transitions and the latest 5,000 events. If storage stays down long enough to fill it, the oldest entries are
dropped, status reports the loss at once, `recording.json` keeps the counts, and `wu inspect` and Rerun flag the
record as incomplete. Gaps show in the sample indices and event numbers, and durations do not count across them.

Summaries add up durations and extremes as the run goes. After 8,192 powered ticks, the tick median and p99 come
from a uniform sample of that size (`tick_percentile_sample`); the maximum stays exact, and `wu inspect` computes
exact percentiles from the saved chunks.

## What an agent adds

```python
robot.record(context={"model": "your-model", "prompt": "move the block", "role": "live policy"})
robot.record(note="Operator moved an obstacle before answering the checkpoint")
```

From the shell: `wu record --context inputs.json --note '...'`; MCP's `record` takes the same fields. They are
saved as given, so keep credentials out of them. A `done` outcome means the commands completed; the block example
saves its separate success check in `result.json` and a `task_result` event.

Each measurement adds a `measurement` event with what the agent received: its id, camera and frame, `valid` and
`reason`, `surface_center`, `visible_bounds` and `from_tool` in the work frame, the picture's capture time
(`capture_t`) and the path of its picture. Its two files are written before the agent gets the reply, so a storage
failure fails the measurement instead of losing it later. A `withdrawn` event lists measurements a tracker withdrew
after losing their target.
