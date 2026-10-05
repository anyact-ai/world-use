# Checked procedures through Python and MCP

The procedure tools connect selected image support to a numeric motion plan and
an observable outcome. Python and MCP use the same daemon-owned measurements,
geometry fits, prerequisite checks and verification functions. Motion continues
through the existing kernel and plan vocabulary.

## Run the complete example

From a checkout with development dependencies installed:

```sh
uv sync --locked
uv run python -m world_use.examples.procedures --scenario displaced --output runs/procedure
uv run wu inspect runs/procedure
uv run wu view runs/procedure --out runs/procedure.rrd
```

This scripted procedure calls the public MCP tools to select, estimate, approach,
grip, verify a lift, place, verify release and withdrawal, and return with torque
off. Its runner alone receives simulator truth for independent evaluation. The
default selector is a deterministic orange-color fixture, not learned tracking.
The setup, estimator and predicates are shared with the
[perception example](../examples/perception), rather than a second task simulator.
See [simulation setup](simulation.md) for headless EGL requirements.

With the optional vision dependencies installed, the same task uses a separate
EdgeTAM process:

```sh
uv run --extra vision python -m world_use.examples.procedures \
  --scenario displaced --model edgetam --device cpu --output runs/procedure-edgetam
```

Weights load before enabling motors. For a CPU-only development install, use
`uv pip install --python .venv/bin/python --torch-backend cpu --editable '.[vision]'`
and run `.venv/bin/python` directly. The model revision is pinned by `vision.py`;
the checkpoint must already be cached or downloadable. Neither this example nor
its tests authorize or operate physical hardware.

The task assumes a 4 × 4 × 10 cm upright block, a substantially visible top face,
a calibrated fixed overhead camera, and a known clear tray. Its 45-second evidence
budget and lowering recovery are specific to this simulation. Results are
integration evidence, not a general grasping or tracking benchmark.

## Connect an agent

The core MCP server needs the `mcp` extra. Region selection and refresh also need
the `vision` extra and explicit startup configuration:

```sh
wu up --workcell block
wu mcp --vision --device cpu
```

Read `policy`, `card` and `status` before operating. Start `--vision` while torque
is off; startup refuses powered or uncertain sessions. Without that flag, all
existing core tools remain available, along with geometry fitting, verification,
image inspection and history. `select_target` and `observe_targets` are advertised
only when a tracker is configured. `--model-path` can select a local checkpoint;
its weights are then reported as local rather than as the pinned public revision.

| Tool | Result and limits |
|---|---|
| `camera_frame` | Native pixels and a capture ID; optional MuJoCo aligned depth. |
| `select_target` | A point or box seeds one region and returns its target ID, exact-frame preview and registered support. Labels do not guide semantic selection. |
| `observe_targets` | Updates 1–4 targets with one capture per camera, or an explicit camera-to-capture mapping. Region visibility and valid metric evidence are separate. |
| `inspect_image` | A cached frame or committed evidence image, with optional crop and an exact mapping to native pixel edges. Archived images remain historical. |
| `fit_geometry` | `known_box`, `plane` or `axis` from registered support. Returns a geometry ID, base-metre components, assumptions and diagnostics. |
| `check` | Resolves references, rehearses an ordinary PlanSpec and freezes optional effect criteria. Returns structured problems and a prepared plan on a non-refused rehearsal. |
| `run` | Submits a literal plan or a prepared ID. Prepared plans always rehearse again from current state. |
| `verify_effect` | Evaluates a predeclared lift or placement criterion using new geometry and synchronized captured feedback; returns `pass`, `fail` or `unknown`. |
| `inspect_run` | Pages through this daemon session's committed events plus its live tail, including plans, outcomes, geometry and verification. Supports a job filter and reports missing records. |

Tool replies preserve concise text and include MCP `structuredContent`. `card`
reports capabilities, effect fields, storage limits and collision coverage.
`help(step)` adds a JSON schema for the built-in step's fields; runtime validation
remains authoritative. `events(since, limit)` now returns the first bounded page
after `since`; continue using `last` while `more` is true. Python `Client.events`
retains its existing full live-window response.

## Bind a phase to geometry

In MCP, `check` prepares by default. Python opts in using `Client.check(...,
prepare=True)`. These illustrative arguments assume G1 is a valid known-box
estimate from the current session:

```json
{
  "plan": {
    "do": "move_to",
    "frame": "work",
    "to": {
      "geometry": "G1",
      "component": "center",
      "offset_m": [0.02, 0, 0.02],
      "offset_frame": "work"
    }
  },
  "max_age_s": 5
}
```

The resolver performs frame conversion, emits numeric PlanSpec, and adds every
source receipt to the prerequisites. The result includes the exact plan,
derivations and assumptions. Inspect it before `run(plan_id=...)`.
`to` accepts position components; `point` and `jaws` accept direction components.
An unsigned fitted normal/axis requires an explicit `sign` of `-1` or `1`.
Direction references cannot take position offsets. There is no expression language
or automatic following of a moving target.

Prepared IDs freeze data, not execution permission. Changed reference frames,
calibration, expired evidence or reported target loss can refuse admission.
Reselecting uses a new target ID; `replace_target` invalidates the old selection.
Accepted jobs retain small prerequisite values and revocation signals, so loss
also blocks later dependent steps. No tracker runs in the control tick, and an
executing step is not interrupted merely because its evidence becomes old.
Unobserved changes to the scene remain undetectable.

A prepared ID has one submission key. Repeating it returns the original job,
including after the prepared-plan cache expires; check again for another execution.
Literal plans, `home` and `calibrate` can supply `request_id` formed as
`status.session_id + ':' + unique_value`. Retry with that same key and payload.
Changed payloads or keys from an earlier daemon session are refused. The daemon
retains at most 1,024 keyed submissions and refuses further keys rather than
evicting them and risking another execution. Existing unkeyed literal calls keep
their behavior and do not receive this retry guarantee.

## Declare and verify the effect

For a lift after a supported grip, the caller can freeze this criterion in
`check(..., effects=[...])` before moving:

```json
{
  "kind": "lift",
  "before": "G1",
  "frame": "work",
  "min_up_m": 0.035,
  "max_error_m": 0.015,
  "max_age_s": 15
}
```

After the job completes, capture/update and fit G2, then call
`verify_effect(job=JOB_ID, after="G2", effect=0)`. The verifier uses the original
criteria, the selected target's continuity claim, source calibration, capture
times and measured tool positions at that job's start and end. Lift observations
must agree with those job boundaries within the declared tolerance; motion
outside the job cannot supply its lift displacement. A changed target, shape
assumption, unavailable geometry or observation taken before job completion
produces `unknown`.

Placement additionally declares `target_m`, `position_tolerance_m`, `support_z_m`,
`height_tolerance_m`, `min_clearance_m` and `min_aperture_mm`, with a `before`
geometry baseline, `frame` and `max_age_s`. It checks the destination and known
support height, captured gripper opening and tool withdrawal. Missing synchronized
feedback is unknown. A successful command and the estimated `World.held` state
cannot supply that evidence. The effect outcome is separate from job outcome and
from the final power state.

## Lifetime and implementation limits

The client helper `world_use.observations.Perception` exposes the same capture,
selection, update and crop operations to Python procedures. Its optional
`TrackerProcess` shares loaded weights across four independent target histories.
One inference request runs at a time; concurrent requests return busy. Calls have
a deadline, and a timed-out worker is terminated without replaying its request.
Its targets require reselection after restart. MCP dispatch runs on worker threads,
so status and stop can proceed during inference. Closing a perception helper
invalidates its targets; it does not command a return or switch power off.

The daemon retains 256 receipts, 256 small geometry results, 64 prepared plans and
256 submitted phase specifications for live verification. Captures and registered
measurement support have separate 64 MiB limits; expired support cannot be fitted
again. Already fitted geometry may survive raw-support eviction while its compact
source receipt remains valid. Long-term evidence and criteria remain in the flight
record after live handles expire. For other or closed sessions, use the existing
offline `wu inspect` and Rerun tools.

Geometry fits and effect calculations are bounded pure Python/numpy functions,
called on daemon request threads over authoritative registered support. Learned
inference stays in the client-side process. The kernel only consumes numeric plans
and compact prerequisites. Current collision checks still omit fingers, payloads,
the pedestal and self-collision; these tools do not expand that coverage.

Text grounding, material-point tracking, cross-camera association and arbitrary
grasp proposal remain future candidates in the [design](agent-tools-design.md).
