# Measure, move, measure again

A scripted procedure finds a block that is missing from the robot's world model, picks it up and places it on
the target, measuring with the camera before each move and after the lift and release. It uses only the MCP tools
an agent gets: camera frames, depth measurements, motion plans and freshness checks. The procedure knows the
block's colour and size; only the evaluator reads the simulator's pose of it. See
[measuring from pictures](../../docs/perception.md) for the tools.

## Run it

From a checkout of world-use:

```sh
uv sync --locked
uv run python -m world_use.examples.perception --output runs/perception
uv run wu inspect runs/perception
uv run wu view runs/perception --out runs/perception.rrd
```

The example starts its own simulated daemon with physics in real time and takes about a minute. It never connects
to hardware, and it goes home and switches torque off before closing. Use a new output folder for each run;
headless rendering needs MuJoCo's [EGL setup](../../docs/simulation.md#cameras-and-replay). `--scenario missing`
removes the block: nothing is measured and the arm is never powered.

## What it does

1. `camera_frame` with depth. Finding the block's lit top face by its orange colour stands in for the agent
   looking at the picture.
2. `measure_pixels` at the middle of that face, then `add_box` with the block's known 4 x 4 x 10 cm shape.
3. `enable`, turn the gripper to its grasp orientation above the tray, and move above the block. The move
   requires the measurement.
4. Measure again, then descend and grip, requiring the new measurement.
5. Measure, lift 6 cm, measure. The lift passes if the block's offset from the tool (`from_tool`) held while its
   top face rose.
6. Carry above the target and lower to the estimated release height. Fit a small depth plane on the lit top
   face; its normal and the known block height give the block's centre even when it tilts. If that centre is more
   than 3 mm from the target, the shared `place` procedure does not open the gripper; the task lifts 6 cm,
   shifts by the measured error and approaches again. It allows three approaches in all and corrections of at
   most 3 cm. Once the check passes, open, check the opening and withdraw.
7. Measure: the placement is confirmed if the top-face centre is within 8 mm of the target in 3D, the gripper
   is open and the tool is 8 cm above the block. The 8 mm leaves 2 mm of the task's 1 cm tolerance for
   measurement error.

After the gripper lets go and before homing, a separate evaluator, the only code that reads simulator truth,
applies the [block task's success check](../pick-place/README.md): position within 1 cm, orientation within
5 degrees, released and withdrawn. The camera checks do not measure the block's full orientation. On failure, an
open gripper withdraws upward; if it may still hold the block, the procedure first lowers it onto the tray and
opens, which is safe only over this clear tray. `result.json` holds the procedure's checks and measurements, the
job outcomes, each `place` attempt, and the evaluator's verdict; `perception/` holds each measurement's picture
and points.

## With EdgeTAM

```sh
uv run --extra vision python -m world_use.examples.perception \
  --model edgetam --device cpu --output runs/perception-edgetam
```

For a CPU-only install, use `uv pip install --python .venv/bin/python --torch-backend cpu --editable '.[vision]'`
and run `.venv/bin/python` directly. The weights load before the arm is powered. The procedure selects the block
with `select_target`, seeded by the box around its orange pixels, and measures it later with `observe_targets`.
At release height both modes measure the lit top face with the colour cue and a depth plane; the tracker does not
give the block's pose by itself.

The CI vision job runs this with the pinned model for both scenarios, checks the tracker on a replayed picture
and that shared image encoding gives identical masks, checks the Rerun export, and uploads the output as its
`vision-evidence` artifact:

```sh
OMP_NUM_THREADS=2 uv run --extra vision --extra rerun python scripts/vision_smoke.py --output runs/vision-validation
```

## Limits

The task assumes a known upright block, a fixed calibrated overhead camera, a clear tray and simulated depth.
These runs check that the pieces work together; they are not a benchmark of tracking or grasping.

The block can tilt and slide in the pinch grasp during transport, which is why the procedure measures again
before opening. A heavier, 100 g block has both passed and missed the 1 cm tolerance in CI and local runs. The
camera check is stricter than the task, so it can reject a placement that truth accepts: `placement` reports the
camera check, `evaluation.success` the simulator truth. A failed check is reported as a failure, followed by
homing and torque off, and the evaluator still records the block's actual position and orientation.

The colour selector is deliberately simple and sensitive to camera angle and lighting. From an angled view the
block can fall outside its colour threshold; the procedure then reports a failed observation, puts the block back
on the tray and goes home with torque off. `tests/test_simulation_variations.py` checks this recovery alongside
successful transfers with other block positions, camera placements, masses and friction.
