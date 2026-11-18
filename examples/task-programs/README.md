# Record task programs

An LLM can work out a task with short Python snippets, then turn them into a program with parameters for repeated
use. `world_use.tasks` runs such a program from a snapshot of its source and records its parameters, inputs,
calls and result, so you can later see what it did and why. To repeat a task, run the program again so that it
measures the scene afresh; do not replay a saved list of robot commands.
[External simulation sessions](../../docs/simulation.md#external-simulation-sessions) use the same recorder.

Each invocation gets its own folder with the source snapshot, parameters, inputs and trace. A revised program gets
a new invocation; `--parent` links it to the failed attempt that prompted the revision. The runner attaches only to
simulations. It is trusted, single-writer Python, not a sandbox or a lock on the robot.

## Try it offline

From the repository root:

```sh
uv run python examples/task-programs/analyze_alignment.py \
  examples/task-programs/alignment.json --output runs/alignment-01
```

The included fixture is a synthetic translation with a known answer; no robot is involved. The program snapshots
itself and the input, recomputes `align_planar` and compares the result with the saved one within numerical
tolerances. It also accepts `geometry.json` from the kitchen packing experiment, which reproduces only that run's
alignment calculations: the [packing task](../robocasa-lunchbox/task.py) chooses the landmarks, and rerunning it
needs a live simulation and fresh observations. Use a new output directory for each invocation.

## Save a task

Write a normal `task.py` with one entrypoint:

```python
def total(values):
    return sum(values)


def run(robot, params, record):
    value = record.call(total, params["values"])
    return {"verdict": "pass", "total": value}
```

With `{"values": [1, 2, 3]}` in `parameters.json`:

```sh
uv run python -m world_use.tasks /path/to/task.py \
  --params /path/to/parameters.json --output runs/task-01
```

`run` must return a dict whose `verdict` is `pass`, `fail` or `inconclusive`. The launcher copies the task and
its own code into the invocation folder before importing anything, then runs that copy in a new Python process.
Add local imports and configuration with repeated `--source /path/to/helper.py`; they must be under the task's
directory, and a dependency lockfile is worth including. The snapshot directory is the working directory, so give
external inputs as absolute paths in the parameters. The launcher installs no dependencies, and an invocation
cannot run a second time.

`record.call(function, *args, **kwargs)` saves the arguments before the call and the return value or exception
after it. It records what was called, not the function's code, so call functions from declared source or an
installed dependency; other imports are not found for you. To read an external file, copy it first with
`path = record.input_file(original_path)` and read the copy. NumPy arrays are saved without pickle.

Group related calls into a named phase:

```python
with record.phase("place", context={"object": "block"}) as phase_id:
    frame = robot.frame("top", depth=True)
    # Measure, compute and submit through this recorded client.
```

Phases can nest, and every call, computation and artifact inside one carries its ID. `wu inspect TASK_FOLDER`
lists the phases and `--phase ID` expands one. A phase ends `completed` when its block returns and `failed` when
it raises; a process that dies leaves it without an end. `completed` does not mean the placement worked, so record
the result of your check separately. Phases only label the record: they do not retry calls or change control or
power. The [pushing example](../push-letters) uses a phase for each push, and the
[kitchen recorder](../robocasa-counter/agent_episode.py) one for each MCP call and the observations it makes.

For exploration, run a Python file as a cell in a persistent namespace:

```python
namespace = {"robot": robot}
result = record.cell("/path/to/fit.py", namespace,
                     inputs={"points": current_points}, outputs=["corners"])
```

This saves the source as run, the declared inputs and outputs, stdout, stderr and any exception; later edits to
the file do not change the saved copy. Nothing else in the namespace is recorded, so pass such values as function
inputs when you turn the cells into a repeatable task.

## Attach to a simulation

By default `robot` is `None`. To attach to a separately started simulation, pass `--url http://127.0.0.1:7431`.
The daemon must report `session.mode=simulation` and an intact flight record, with current feedback, no active or
queued job, no unresolved fault, and torque off.

The supplied `robot` is a recorded [`Client`](../../src/world_use/client.py). Start every invocation by capturing a
new frame:

```python
frame = robot.frame("top", depth=True)
measurement = robot.measure(frame, point=selected_pixel, target="item")
job = robot.run(plan, requires=[{"evidence": measurement["id"], "max_age_s": 30}])
```

The pixel, plan and age limit are the task's choices. The client refuses frames and measurement IDs from earlier
invocations, even when the daemon still knows them, and the kernel applies its usual
[age, calibration and withdrawal checks](../../docs/perception.md#freshness). Keep the source measurement IDs with
any geometry derived from them: a fit does not make a measurement fresh, and coordinates submitted without
`requires` cannot be traced to a measurement. Measure again after the grasp changes, after contact, after tracking
is lost and before checking a released placement. A new Python process does not reset the daemon's world, so
re-establish movable objects and occupancy.

Pass the same client to `Tracking(robot, tracker)` or `mcp_server.build(client=robot)` to record the frames they
capture too; calls through another client are not recorded. Each frame's native RGB, aligned depth, calibration
revision, timestamp and tool pose are saved as a `.frame.json` artifact. Read one offline with
`Frame.from_dict(json.loads(path.read_text()))`; a restored frame does not count as a current observation.

The task does all robot actions itself: polling jobs, recovery, its success checks and the
[power handoff](../../src/world_use/POLICY.md). The launcher never stops, retries, homes or releases the robot. A
timeout or disconnect does not cancel accepted work: poll the known job ID, and if a submission's reply was lost,
find the job rather than submitting it again.

## Read the record

| File | Contents |
| --- | --- |
| `invocation.json`, `source/` | Parameters, parent reference, source hashes and copies, installed environment |
| `started.json` | Written once when execution starts, so a second run is refused; UTC and monotonic clock anchors |
| `trace.jsonl` | Ordered intents, replies, exceptions and flight references |
| `artifacts/`, `cells/` | Saved inputs, arrays, frames, cell source and stdout/stderr |
| `result.json` | The task's verdict, execution and recording errors, and final robot status |

Begin and end annotations link the invocation to the daemon's flight record, and the trace holds the initial
world, card and status, event positions and the returned job IDs. Keep that flight folder with the invocation; the
launcher neither copies nor closes it.

The launcher exits with zero only if the task returned `pass` with no execution or recording error and, when
attached, the robot ends with no active or queued job and torque off. Otherwise it exits nonzero and keeps the
task's own verdict in `result.json`. A missing `result.json` means the run was interrupted, and an intent without
a reply has an unknown outcome. `read_trace(folder)` reads the trace up to its last complete line. After a
recording failure, new task calls are refused, but status, job polling, stop, home and release still work.

Offline analysis recomputes saved inputs, and `wu replay` shows recorded motion; a new invocation measures afresh
and may act differently. None of these gives a success rate or shows transfer to hardware. Source hashes detect a
changed snapshot at startup, so leave the record unedited afterwards. Dependencies are identified, not vendored,
and reads the task does not route through `record` are not captured.
