# Reading a run

```sh
wu inspect runs/YOUR-RUN
wu inspect runs/YOUR-RUN --json
wu replay runs/YOUR-RUN --out replay.gif --speed 3
wu view runs/YOUR-RUN                         # requires world-use[rerun]
wu view runs/YOUR-RUN --out recording.rrd      # headless export
```

These commands run offline. Replay reconstructs measured joints and the recorded
world model with the saved robot geometry. It does not operate a robot or
re-run the plan. Original camera observations remain under `views/`; the rendered
replay is labeled separately. New records include the robot description, URDF, and meshes,
so they remain usable after moving the run folder or uninstalling its driver.
Older records without a saved model use the installed built-in geometry. A saved
reBot URDF without archived meshes can use the bundled meshes only when its URDF
matches exactly; custom records need their original assets.

The optional [Rerun viewer](visualization.md) synchronizes the 3D reconstruction,
telemetry, events, and saved observations. It can also follow a running recorder.

A run contains:

| File | Contents |
| --- | --- |
| `session.json` | Format/package version, source and URDF hashes, adapter, startup configuration, initial world/command/measurement snapshot, fitted model |
| `robot.urdf` | The geometry used by this run; its limits and gripper description are in `session.json` |
| `assets/`, `robot-assets.json` | Content-addressed meshes and their original URDF path mapping |
| `events.jsonl` | Submitted plan JSON, structured outcomes, checkpoint questions/answers, contacts, world changes, view paths and annotations |
| `tape/000000.npz`, … | Append-only telemetry and power-transition chunks; the complete history of a new run |
| `recording.json` | Missing sample, event, and power-transition counts, only if a recording buffer overran |
| `complete.json` | Final chunk count and event byte count, committed only after the journal finishes successfully |
| `summary.json`, `world.json` | Summary and world model at the last explicit save or normal close |
| `views/` | Saved camera observations |

Events and telemetry are flushed once per second on a background thread. A process kill may
lose the current interval and an unfinished event line; it should not lose earlier
committed chunks. Format 3 saves and closes flush these chunks without creating a
second complete `tape.npz`. Readers still accept older `tape.npz` snapshots and
`tape/power.npz` files. Python integrations should use
`world_use.recorder.load_tape(run_folder)` to read the complete arrays.
The `closed` event reports connection closure; `complete.json` confirms that the
final recording writes succeeded. Live readers wait for that marker and its
listed data before finishing, including when storage is slow.

The daemon retires committed telemetry from memory. Its pending buffer holds at
most 30,000 samples (five minutes at 100 Hz), 5,000 power transitions, and the most
recent 5,000 events. Short storage failures can recover from that buffer. If a
failure or producer overrun exhausts it, the oldest entries are dropped; status
reports the loss immediately and the recorder persists the counts when storage
recovers. `inspect` and Rerun flag incomplete records. Sample and power indices in
the chunks identify gaps; durations do not count across them. Event sequence
numbers likewise identify missing events. Offline readers retain every committed
entry; a live event poll only returns the recent window and reports skipped
entries in its `missed` field.

Saved summaries accumulate lifetime durations and extrema without loading the
history. After 8,192 powered tick intervals, their median and p99 use a uniform
sample of that size, identified by `tick_percentile_sample`; the maximum remains
exact. Offline `inspect` computes exact percentiles from committed samples.
Embedded kernels without a run folder retain their whole tape for rehearsal.

`status --json` reports recording errors; storage failures do not fault the robot
or prevent release and shutdown. This is local file persistence, not a guarantee
against storage-device failure.

Record the context an integration knows, without coupling the runtime to a model SDK:

```python
robot.record(context={"model": "your-model", "prompt": "move the block", "role": "live policy"})
robot.record(note="Operator moved an obstacle before answering the checkpoint")
```

CLI: `wu record --context inputs.json --note '...'`. MCP's `record` accepts the same
fields. These values are saved as supplied; avoid placing credentials in them.
A `done` outcome records command completion. The example saves its separate task
predicate in `result.json` and a `task_result` event.

The [procedure tools](procedure-tools.md) also emit `geometry`, `prepared`,
`phase_submitted`, `target_lost` and `verification` events. These retain source
receipt IDs, fitted assumptions, resolved numeric plans, criteria frozen before
action, and observed effects. `finished` includes the job's capture-clock window
for relating before/after observations to execution. Agent-authored annotations
remain distinct from these calculated results.

MCP `inspect_run` and `Client.inspect_run` page through the current session's
committed events plus the live tail without loading its full telemetry. Continue
from `next_cursor` while `more` is true. Missing sequence numbers and recorder
errors remain visible; `record_complete` describes the available history, while
`closed` separately reports session closure. Archived pixels are retrieved by
evidence ID and remain explicitly historical.
