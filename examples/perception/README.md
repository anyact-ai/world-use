# Measure, move, verify

This simulation-only example transfers a known upright block using calibrated
MuJoCo RGB-D. The procedure knows the block dimensions and the tray, but receives
no simulator object positions or instance masks. It measures visible surfaces,
asserts its shape estimate explicitly, and checks evidence before dependent motion.
Physics and feedback continue while the procedure captures, measures and records.

From the repository with Python 3.13+:

```sh
uv sync --locked
uv run python -m world_use.examples.perception --output runs/perception
uv run wu view runs/perception --out runs/perception.rrd
```

Open the `.rrd` locally with Rerun 0.38, or omit `--out` on a desktop. Run directories
must be new or empty. The example owns a local simulated daemon and returns home
and releases torque before closing; it never connects to physical hardware.
Headless rendering needs MuJoCo's [EGL setup](../../docs/simulation.md).

The default selector thresholds the orange fixture in rendered RGB. It makes the
entire measurement, execution and recording path reproducible without downloaded
weights. It is deliberately specific to this scene. To use the existing learned
tracker instead:

```sh
uv run --extra vision python -m world_use.examples.perception \
  --model edgetam --device cpu --output runs/perception-edgetam
```

EdgeTAM loads its pinned weights before enabling motors. The first selection is
seeded from the same visible RGB box; later masks use `EdgeTAM.update`. Model files
must be reachable from Hugging Face or already cached. The automated tests use
deterministic masks and the color fixture; their success does not establish
EdgeTAM accuracy or latency.

The Linux CI vision job downloads the pinned checkpoint and runs the four
displaced-block conditions plus a missing-block refusal. It checks the independent
outcomes, visual verification, torque release, recorded evidence, point/box prompts
and tracking beyond the history window, then exports Rerun. Its `vision-evidence`
artifact contains full runs and `validation.json`, including inference and control
timings. Run the same check with vision and Rerun installed:

```sh
OMP_NUM_THREADS=2 uv run --extra vision --extra rerun python scripts/vision_smoke.py \
  --output runs/vision-validation
```

These checks cover this rendered fixture on CPU; they do not establish general
object-tracking accuracy, GPU behavior or physical RGB-D performance.

## Inspect what happened

`result.json` separates procedure-side lift/placement checks from the independent
simulator evaluator. Both need more than a completed motion command. The evaluator
runs after release and before returning home, and is the only consumer of object
truth. Missing or occluded geometry produces an unknown visual result.

Each registered observation has source RGB, metric depth, sampled support,
calibration, a preview and metadata in `perception/<id>/`. Rerun shows source
images at capture time and derived points when the measurement became available.
Observed surfaces are distinct from the estimated planning boxes. Events link job
prerequisites, shape assumptions and verification decisions to evidence IDs.

The model assumes a 4 × 4 × 10 cm upright, work-aligned block with a substantially
visible top face. It uses a static tray and a 45-second evidence age limit for this
bounded simulation task. That value is not a hardware default. Tracking has its
own 15-second observation budget. There is one grasp attempt; recovery lowers over
this known clear tray, opens, withdraws and follows the checked return route.
Do not use that recovery in an arbitrary scene.

## Compare conditions

```sh
uv run python -m world_use.examples.perception \
  --condition nominal --scenario displaced --output runs/nominal
uv run python -m world_use.examples.perception \
  --condition depth --scenario displaced --output runs/depth
uv run python -m world_use.examples.perception \
  --condition track --scenario displaced --output runs/track
uv run python -m world_use.examples.perception \
  --condition verify --scenario displaced --output runs/verify
```

`nominal` uses a fixed position; `depth` localizes once; `track` refreshes before
descent and after lift; `verify` adds lift and placement checks. With the color
fixture, refresh is color resegmentation rather than learned tracking. Add
`--model edgetam` consistently to a learned-model comparison. Available scenarios
are `nominal`, `shifted`, `displaced` and `missing`.

Use the same scenario, selector, motion limits and after-release evaluator across
conditions. Keep full run folders when reporting outcomes, latency, tick delays
or powered holding time. These small fixture runs are integration checks, not a
held-out robotics benchmark. See the [design and contracts](../../docs/perception-design.md)
for validity, failure semantics, recording bounds and deferred work.
