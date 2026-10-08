# Record task programs

An LLM can explore a task with short Python snippets, then extract a parameterized
program for repeated use. This example preserves the code and observations behind
those decisions. It uses the existing world-use client; nothing is added to the
kernel or the installed `wu` interface.

Each invocation gets its own source snapshot, parameters, inputs and trace.
Repeat the **program**, using current observations, rather than resending a saved
list of robot commands. Revisions get new invocation folders; `--parent` can link
a revision to the failed attempt that prompted it.

## Try it offline

From the repository root:

```sh
uv run python examples/task-programs/analyze_alignment.py \
  examples/task-programs/alignment.json --output runs/alignment-01
```

The included fixture is a synthetic translation with a known answer, not robot
execution evidence. The program snapshots itself and the input, recomputes
`align_planar`, and compares its output within numerical tolerances. It also
accepts `geometry.json` from the kitchen experiment. That reproduces the recorded
alignment calculation, not the missing code that selected its landmarks or a new
packing attempt. Use a new output directory for each invocation.

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
uv run python examples/task-programs/run_task.py /path/to/task.py \
  --params /path/to/parameters.json --output runs/task-01
```

The launcher copies the task and its own code before importing the task, then
executes that snapshot in a fresh Python process. Declare additional local
imports or configuration with repeated `--source /path/to/helper.py`; they must
be under the task's directory. Include a dependency lockfile when available.
Execution uses the snapshot directory as its working directory, so use absolute
paths for external inputs in parameters. Dependencies are not installed by the
launcher. An existing invocation cannot be executed a second time.

`record.call(function, *args, **kwargs)` saves inputs before calling and the return
value or exception afterward. To read an external file, first copy it with
`path = record.input_file(original_path)`, then read that saved path. NumPy arrays
are saved without pickle. Functions must come from declared local source or an
identified installed dependency; arbitrary imports are not discovered for you.

For exploration, execute a file-backed cell in a persistent namespace:

```python
namespace = {"robot": robot}
result = record.cell("/path/to/fit.py", namespace,
                     inputs={"points": current_points}, outputs=["corners"])
```

This saves the executed source, declared inputs/outputs, stdout/stderr and any
exception. Later edits do not replace the executed copy. Undeclared namespace
contents are not captured; replace them with explicit function inputs when
extracting a repeated task.

## Use current observations

By default `robot` is `None`. To attach to a separately started simulation, pass
`--url http://127.0.0.1:7431`. Its startup identity must report
`session.mode=simulation`, with an intact flight record, current feedback, no
active/queued work, and power off. Hardware is not supported by this example.

The supplied `robot` is a recorded [`Client`](../../src/world_use/client.py).
Start every invocation by capturing a new frame:

```python
frame = robot.frame("top", depth=True)
measurement = robot.measure(frame, point=selected_pixel, target="item")
job = robot.run(plan, requires=[{"evidence": measurement["id"], "max_age_s": 30}])
```

The pixel, plan and age limit are task decisions. Frames and required measurement
IDs from earlier invocations are refused, even when the daemon still knows them.
The kernel retains its [age, calibration and withdrawal checks](../../docs/perception.md#freshness).
Derived geometry must retain its source measurement IDs; a fit does not refresh
evidence. Coordinates submitted without `requires` cannot be checked for their
source automatically. Refresh after grasp changes, contact, loss of tracking,
and before verifying a released placement. Re-establish movable objects and
occupancy; a new Python process does not reset the daemon's world.

Pass this same client to `Tracking(robot, tracker)` to record its internal frame
acquisitions too. Calls through an independent client or MCP server are outside
this recorder's coverage. Full frame transport data, including native RGB,
aligned depth, calibration revision, timestamp and captured tool pose, is saved
in `.frame.json` artifacts. Read these offline using
`Frame.from_dict(json.loads(path.read_text()))`; restoring a file does not make
it current evidence.

Task code owns all robot actions: polling jobs, bounded recovery, task predicates
and the [power handoff](../../src/world_use/POLICY.md). The launcher does not
stop, retry, home or release the robot on its own. A timeout or disconnect does
not cancel accepted work. Poll the known job ID; reconcile a lost submission
reply rather than submitting it again. An active job or unresolved power at exit
produces a nonzero exit code, while preserving the task's original verdict.

## Read the evidence

| File | Contents |
| --- | --- |
| `invocation.json`, `source/` | Parameters, parent reference, source hashes and copies, installed environment |
| `started.json` | Exclusive execution marker and UTC/monotonic clock anchor |
| `trace.jsonl` | Ordered intents, replies, exceptions and flight references |
| `artifacts/`, `cells/` | Saved inputs, arrays, frames, cell source and stdout/stderr |
| `result.json` | The task's verdict, execution/recording errors and final robot status |

Begin/end annotations link the invocation to the daemon's existing flight record;
the trace includes initial world/card/status, event positions and returned job
IDs. Keep that flight folder with the invocation when retaining evidence. It is
not duplicated or closed by this launcher.

The launcher returns zero only for a passing task with no execution or recording
errors and, when connected, no active/queued work or unresolved power. A task's
`pass` is distinct from complete recording and successful cleanup. Missing
`result.json` means interrupted or incomplete execution. An unanswered intent has
an unknown outcome. `read_trace(folder)` reads the complete JSONL prefix, ignoring
only a final unterminated line. Recording failure blocks new task calls without
blocking explicit inspection, stop or power recovery.

Offline analysis recomputes saved inputs. `wu replay` visualizes recorded motion.
A new invocation obtains fresh inputs and may act differently. None of these
establishes a success rate or hardware transfer. This is trusted, single-writer
Python, not a sandbox or a robot ownership lock. Source hashes detect changed
snapshots at startup; keep evidence unedited afterward. Dependencies are
identified, not vendored. Uncaptured external reads remain outside the
reproducibility claim.
