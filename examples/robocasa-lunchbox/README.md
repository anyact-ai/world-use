# Lunchbox packing

A Python program packs two snacks into a small open container in a simulated RoboCasa kitchen, measuring with
RGB-D before each move and after each release. An LLM agent wrote the program during development, with knowledge
of the scene and feedback from a person; no model is called while it runs.

https://github.com/user-attachments/assets/f714e092-69e5-40ac-b7f6-bdeac1f4f6b9

Both packages ended inside the container with a measured 12.8 mm gap and no unintended contacts in the sampled
replay. The same program also packed a shifted, rotated layout: the new measurements changed its destinations and
the cookie box's rotation from about 24° to 40°. These are two development runs, not a success rate.

## What world-use adds

- EdgeTAM and depth locate the packages and the container; `measure_pixels` gives the floor and counter heights
  for a compact placement.
- `align_planar` turns a measured footprint and its target into a tool position and yaw, allowing for an
  off-centre grasp. The program works out the release height itself.
- Gripper feedback and `observe_targets` check that a package is still held after the lift, and fresh RGB-D
  checks each released package. `requires` limits how old the measurements behind each motion may be.

The [task program](task.py) chooses the image regions, packing order, layout and clearances. Its inputs are RGB-D
and robot feedback; only the audit reads simulator poses.

## Limits

The scene has rigid packages, an open container scaled to 1.10×, a cookie box scaled to 0.85× and ideal RGB-D.
Alignment assumes the package does not move in the grasp, and the placement check compares centres, so it can
miss a rotation. There is no joint torque sensing or full-scene rehearsal (`rehearse=False`). Packages move only
through contact, and the audit checks support and stability from the recorded physics, separately from the
camera checks.

The 63-second video uses the saved side and wrist observations and the recorded motion at 2×, with pauses cut.
The opening uses a presentation camera; overlays show measured geometry and the program's calculations.

## Run locally

Complete the [kitchen setup](../robocasa-counter/README.md#build-the-upstream-scene) and
[runtime install](../robocasa-counter/README.md#prepare-the-runtime), then run from the repository root:

```sh
.venv/bin/python examples/robocasa-lunchbox/prepare.py
.venv/bin/python examples/robocasa-lunchbox/task.py --output runs/lunchbox-new
.venv/bin/python examples/robocasa-lunchbox/audit_episode.py runs/lunchbox-new
.venv/bin/python examples/robocasa-lunchbox/render_episode.py runs/lunchbox-new
```

Use a new output directory for each run. `--measure-only` checks perception and layout without powering the
simulated robot. The shared [simulation session](../../docs/simulation.md#external-simulation-sessions) saves the
declared source and raw RGB-D in `task/`, full-scene state in `physics/` and `replay.npz`, and the robot records
in `flight/`; the program adds its geometry calculations. `audit_episode.py` checks from the recorded physics that
both packages rest on the container floor, inside it and clear of the fingers, that they moved less than 1 mm
after settling, and that the replay has no unintended contacts.
`wu replay runs/lunchbox-new/flight --out scene.gif --speed 3` renders the saved scene through the initial side
camera.

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

Selected records: `runs/lunchbox-compact-05/` and `runs/lunchbox-compact-shifted-03/`. They keep the task source,
measurements, calls, replays and audits; failed trials are kept separately. Raw records and downloaded assets are
not bundled. See the [asset attribution](../robocasa-counter/README.md#attribution).
