# Lunchbox packing

An LLM-written Python program packs two snacks into a small open container using
fresh world-use measurements. The program runs without model calls during execution.

https://github.com/user-attachments/assets/f714e092-69e5-40ac-b7f6-bdeac1f4f6b9

## What world-use adds

- **Measurement.** EdgeTAM and depth locate package and container surfaces;
  `measure_pixels` supplies floor and counter heights for compact placement.
- **Alignment.** `align_planar` converts measured footprints into a tool position
  and yaw, accounting for an off-center grasp. The task calculates release height.
- **Verification.** Gripper feedback and `observe_targets` check retention.
  Fresh RGB-D checks released placements; `requires` limits evidence age for motion.

The [task program](task.py) owns the image regions, packing order, layout and
clearances. Inputs are RGB-D and robot feedback; simulator poses are reserved for the audit.

## Result and scope

Both packages were released inside the container, with a measured 12.8 mm gap
and no unintended contacts in the sampled replay. The **same program** also
passed a shifted, rotated layout: new measurements changed its destinations and
the cookie-box rotation from about 24° to 40°.

These are two development checks with scene knowledge and human feedback. The
scene uses rigid packages, an open container scaled to 1.10×, a cookie box scaled
to 0.85× and idealized RGB-D. Alignment assumes a rigid grasp; center checks can
miss rotation. The adapter has no joint torque sensing or full-scene collision
rehearsal (`rehearse=False`). Contact physics drives the objects; post-run audits
check support and stability separately from camera checks.

The 63-second video uses saved side/wrist observations and recorded motion at 2×,
with pauses cut. The opening uses a presentation camera; overlays show measured
geometry and the task's calculations.

## Run locally

Complete the [kitchen setup](../robocasa-counter/README.md#build-the-upstream-scene) and
[runtime install](../robocasa-counter/README.md#prepare-the-runtime), then run from the repository root:

```sh
.venv/bin/python examples/robocasa-lunchbox/prepare.py
.venv/bin/python examples/robocasa-lunchbox/task.py --output runs/lunchbox-new
.venv/bin/python examples/robocasa-lunchbox/audit_episode.py runs/lunchbox-new
.venv/bin/python examples/robocasa-lunchbox/render_episode.py runs/lunchbox-new
```

Use a new output directory for each run. `--measure-only` checks perception and
layout without powering the simulated robot.
The shared [simulation session](../../docs/simulation.md#external-simulation-sessions) saves
declared source and raw RGB-D in `task/`, full-scene state in `physics/` and
`replay.npz`, and normal robot records in `flight/`. The task retains its geometry
calculations and independent physical audit. `wu replay runs/lunchbox-new/flight
--out scene.gif --speed 3` renders the saved scene through the initial side camera.

<details>
<summary>Run the shifted-layout check</summary>

```sh
.venv/bin/python examples/robocasa-lunchbox/prepare.py \
  --output runs/robocasa-lunchbox-shifted/scene --variant shifted
.venv/bin/python examples/robocasa-lunchbox/task.py \
  --scene runs/robocasa-lunchbox-shifted/scene --output runs/lunchbox-shifted-new
.venv/bin/python examples/robocasa-lunchbox/audit_episode.py runs/lunchbox-shifted-new
```

</details>

Selected records: `runs/lunchbox-compact-05/` and
`runs/lunchbox-compact-shifted-03/`. They retain task source, measurements, calls,
replays and audits; failed trials remain separate. Raw records and downloaded
assets are not bundled. See the [asset attribution](../robocasa-counter/README.md#attribution).
