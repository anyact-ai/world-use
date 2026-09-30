# Reading a run

```sh
wu inspect runs/YOUR-RUN
wu inspect runs/YOUR-RUN --json
wu replay runs/YOUR-RUN --out replay.gif --speed 3
```

Both commands run offline. Replay reconstructs measured joints and the recorded
world model with the saved robot geometry. It does not operate a robot or
re-run the plan. Original camera observations remain under `views/`; the rendered
replay is labeled separately. New records include the robot description and URDF,
so they remain usable after moving the run folder or uninstalling its driver.
Older records without a saved model use the installed built-in geometry.

A run contains:

| File | Contents |
| --- | --- |
| `session.json` | Format/package version, source and URDF hashes, adapter, startup configuration, initial world/command/measurement snapshot, fitted model |
| `robot.urdf` | The geometry used by this run; its limits and gripper description are in `session.json` |
| `events.jsonl` | Submitted plan JSON, structured outcomes, checkpoint questions/answers, contacts, world changes, view paths and annotations |
| `tape/*.npz` | Incremental telemetry chunks; `power.npz` keeps power transitions |
| `tape.npz` | Complete telemetry at the last explicit save or normal close |
| `summary.json`, `world.json` | Summary and world model at the last explicit save or normal close |
| `views/` | Saved camera observations |

Events and telemetry are flushed once per second on a background thread. A process kill may
lose the current interval and an unfinished event line; it should not lose earlier
committed chunks. `inspect`, `replay`, and fitting read the newer of the complete
tape and the chunks. `status --json` reports recording errors; storage failures do
not fault the robot or prevent release and shutdown. This is local file
persistence, not a guarantee against storage-device failure.

Record the context an integration knows, without coupling the runtime to a model SDK:

```python
robot.record(context={"model": "your-model", "prompt": "move the block", "role": "live policy"})
robot.record(note="Operator moved an obstacle before answering the checkpoint")
```

CLI: `wu record --context inputs.json --note '...'`. MCP's `record` accepts the same
fields. These values are saved as supplied; avoid placing credentials in them.
A `done` outcome records command completion. The example saves its separate task
predicate in `result.json` and a `task_result` event.
