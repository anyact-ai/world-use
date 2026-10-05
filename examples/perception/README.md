# Measure, move, measure again

A simulated reBot moves a block nobody told it about. The procedure uses only the MCP tools an agent has:
it takes pictures, measures the block, runs each phase requiring a fresh measurement, and checks the lift
and the placement by measuring again. The robot's world model starts without the block, and the procedure
reads no simulator data. See [measuring from pictures](../../docs/perception.md) for the tools.

## Run it

From a checkout of world-use:

```sh
uv sync --locked
uv run python -m world_use.examples.perception --output runs/perception
uv run wu inspect runs/perception
uv run wu view runs/perception --out runs/perception.rrd
```

The example starts its own simulated daemon with physics running in real time, and takes about a minute.
It never connects to physical hardware, and it goes home and switches torque off before closing. Use a new
output folder for each run; headless rendering needs MuJoCo's
[EGL setup](../../docs/simulation.md#cameras-and-replay). `--scenario missing` removes the block: nothing is
measured and the arm is never powered.

## What it does

1. `camera_frame` with depth. Finding the block's lit top face by its orange colour stands in for the
   agent looking at the picture.
2. `measure_pixels` at the middle of that face, then `add_box` with the block's known 4 x 4 x 10 cm shape.
3. `enable`, set the gripper's orientation above the tray, and move above the block, requiring the
   measurement.
4. Measure again, then descend and grip, requiring the new measurement.
5. Measure, lift 6 cm, measure: the lift passed if `from_tool` held while the top face rose.
6. Carry the block so its top lands on the target's, lower it, open and withdraw.
7. Measure: placed if the top face is within 1 cm of the target, the gripper is open and the tool is
   8 cm clear above the block.

A separate evaluator, the only code that reads simulator truth, judges the result after the gripper lets go
and before homing. Any failure stops the arm, lowers the block onto the tray, opens and withdraws: safe only
over this known clear tray. `result.json` holds the procedure's checks, its measurements, the job
outcomes and the evaluator's verdict; `perception/` holds each measurement's picture and points.

## With EdgeTAM

```sh
uv run --extra vision python -m world_use.examples.perception \
  --model edgetam --device cpu --output runs/perception-edgetam
```

For a CPU-only install, use `uv pip install --python .venv/bin/python --torch-backend cpu --editable '.[vision]'`
and run `.venv/bin/python` directly. The weights load before the arm is powered. The procedure selects the
block with `select_target`, seeded by the box around its orange pixels, and measures it later with
`observe_targets`. The CI vision job runs this with the pinned model for both scenarios, checks the
tracker on a replayed picture and the Rerun export, and keeps everything as its `vision-evidence` artifact:

```sh
OMP_NUM_THREADS=2 uv run --extra vision --extra rerun python scripts/vision_smoke.py --output runs/vision-validation
```

## Limits

The task assumes a known upright block, a fixed calibrated overhead camera, a clear tray and simulated
depth. The block tilts a few degrees in the pinch grasp and settles 5 to 9 mm short of the target when
let go, inside the 1 cm tolerance. These runs check that the pieces work together; they are not a
benchmark of tracking or grasping.
