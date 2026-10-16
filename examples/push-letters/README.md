# Push letters onto their outlines

A Panda arm with an 8 mm rounded pin pushes loose letters onto matching targets.
Each letter must finish within 5 mm and 5 degrees, remain supported and still on
the table, and have no robot contact except the pin. The arm withdraws and releases
power before the outcome is audited.

This is a World Use adaptation of the letter-pushing experiment shown by Shuangyu
Xie with [RoboSimStudio](https://github.com/robosimstudio/robosimstudio), related to
[Revisiting Push-T with Agentic Robotics](https://arxiv.org/abs/2608.18227).
The inspected public RSS checkout does not contain the screenshot's `auto_lab/push`
task or its policy. This example uses MuJoCo, a Panda, and rigid compound letters
60 mm tall and 12 mm thick. It does not reproduce the screenshot's sponge material,
robot, Newton solver or unpublished controller, and makes no sim-to-real claim.

## What controls the arm

Codex authored and revised the Python policy while inspecting development trials.
There are no model API calls while it runs. The controller has the manufactured
letter shapes, colours, target poses and calibrated overhead RGB-D. Colour masks
and silhouette fitting estimate each letter's pose; a quasi-static friction model
chooses a short point push, then the controller measures again. It assumes complete
visibility on a flat table. This is not general object recognition or a blind
evaluation of a fresh agent.

Every movement runs through World Use's kernel, IK, joint limits, trajectories,
tracking watchdog and flight recorder. No letter pose is changed after setup,
and no weld, attachment or object-following rule holds a letter to the pin.
Live object poses, velocities and contacts are used only by the audit and recorder
in the default camera mode. `--feedback state` is an explicitly privileged
diagnostic option; it is not the camera experiment.

The adapter declares a conservative box around the pin from the task's explicit
capsule dimensions. The capability report identifies this source. Its presence
does not add the table or letters to the kernel's estimated world or reproduce
their contact dynamics in rehearsal.

The external scene is not represented by World Use's box-only rehearsal twin.
The normal client runs plans with `check=False`, checking each motion as it starts; full-scene contact rehearsal
and joint torque sensing are unavailable. The rounded pin, contact-clearance
checks, independent physical audit and bounded number of pushes make failures
inspectable; they do not establish hardware safety.

## Run locally

From the World Use repository root, with Python 3.13 and local graphics access:

```sh
uv sync --locked
git clone https://github.com/robosimstudio/robosimstudio.git runs/push-letters-assets/rss
git -C runs/push-letters-assets/rss checkout 73739abe212f963e8a9f5862b27bca370df091a7
uv run python examples/push-letters/scene.py \
  --assets runs/push-letters-assets/rss/assets/embodiments/panda \
  --output runs/letters-scene --letters LAB
uv run python examples/push-letters/episode.py \
  --scene runs/letters-scene --output runs/letters-run
uv run python examples/push-letters/audit.py runs/letters-run
uv run python examples/push-letters/render.py runs/letters-run --speed 2
```

The RSS Python package, Newton, Warp and a CUDA GPU are not needed; only its public
Panda model assets are read. macOS rendering requires access to CoreGraphics;
headless Linux requires EGL or OSMesa. Video export also needs `ffmpeg` on PATH.
Use fresh output directories: existing runs are never overwritten by the episode.
The renderer overwrites only its generated `demo.mp4` and `render.json`.
The runner uses the shared [simulation session](../../docs/simulation.md#external-simulation-sessions)
with wall-clock control. `wu replay runs/letters-run/flight --out scene.gif --speed 3`
also renders the original scene from its saved physics state.

Choose `--letters T`, `LAB` or `AUTOLAB` when building. `--variant shifted` offsets
the starts by 12 mm / -8 mm and changes yaw by -7 or +8 degrees. `--variant friction`
raises the letters' sliding coefficient from 0.5 to 0.7; the bench stays at 0.5,
so MuJoCo's maximum-coefficient combination changes the actual contact friction.
`episode.py --preview` saves the untouched initial scene without enabling the arm.

## Compare policy revisions on fixed cases

The evaluation driver freezes the policy and a separate privileged evaluator, snapshots
the selected scene and initial state, and applies a recorded random seed to small initial
pose variations. Start with a single `T` scene for a short development case:

```sh
uv run python examples/push-letters/scene.py \
  --assets runs/push-letters-assets/panda --output runs/evaluation-scene --letters T
uv run python examples/push-letters/evaluate.py \
  --scene runs/evaluation-scene --output runs/evaluation-policy-a
uv run python examples/push-letters/evaluate.py \
  --rerun runs/evaluation-policy-a --case-name nominal-seed17 --output runs/evaluation-repeat
```

Rerun uses saved source and inputs and refuses a changed Python/package/runtime identity.
The compiled model includes its meshes but is tied to the recorded MuJoCo version; the
original XML and task metadata retain asset provenance. To compare a revised policy,
run the same scene and case selection into another fresh output, then use
`world_use.evaluation.compare(first, second)`. Case identities exclude policy revision
but include the seed, parameters, scene identity and input hashes.

`--case held-out` selects the reserved `shifted-seed29` variation. Once a case is used
to tune the policy, it is no longer independent validation. `--privileged` is explicitly
labelled state-feedback diagnosis and is refused by camera-only comparison. An optional
`--validation` links a reference report for the exact declared scene identity; without
one, scene accuracy remains unvalidated even when the task passes.

Each suite keeps `suite.json`, per-case `trials/<name>/trial.json`, frozen policy/evaluator
invocations, the original episode evidence and `summary.json`. Invalid starts, interrupted
or unstarted cases, incomplete recordings, failure reasons and recovery interventions stay
in the report. Wall and simulation time are separate. The evaluator requires final goal
conditions to hold over 1.9 simulated seconds with no observation gap above 100 ms, alongside
the original physical audit. `summarize(suite)` can inspect a suite interrupted before its
summary was written. The Python separation is a trusted-code API boundary, not a sandbox.

## Recorded results

The full **AUTOLAB** layout subsequently passed **7/7 letters in 34 pushes**,
with maximum final errors of **2.72 mm and 2.35°**, and 0.1158 mm maximum contact
penetration. It used the same contact-selection and perception policy, with a
wider camera-clear parking pose and high transit waypoints for the wider layout.
The complete episode took 442.99 simulated seconds, including settling. This is
one development success; the two preceding wide-layout attempts were refused
before their first push by segment-length and reach checks.

On October 15, 2026, the camera policy completed all three letters in each of
three development conditions with unchanged policy code:

| Condition | Pushes | Simulation time, including final settling | Largest position / yaw error |
| --- | ---: | ---: | --- |
| Nominal | 14 | 135.74 s | 3.51 mm / 3.51° |
| Shifted and rotated starts | 14 | 129.48 s | 2.51 mm / 3.06° |
| Higher friction | 14 | 134.70 s | 3.20 mm / 2.31° |

These are three single-condition trials, not a statistical success rate. Each
passed table support, two-second stability, pin withdrawal, finite states,
no solver warnings, no other robot/letter contacts and power-off checks. Maximum
contact penetration was 0.1233 mm across the three trials. Simulation ran with a
virtual clock; its duration is not wall-clock inference latency.

Early failures exposed a flat cylinder contact instability during withdrawal,
background pixels entering a colour mask, and an ambiguous initial B shape. The
final task uses a capsule pin, absolute colour-distance rejection, and an
asymmetric B. Those changes precede the three-condition comparison.

On October 16, 2026, the shared evaluation path completed these one-letter T
checks with unchanged camera-policy source:

| Named case | Position / yaw error | Simulation time | Audit |
| --- | --- | ---: | --- |
| `nominal-seed17` | 0.61 mm / 0.45° | 47.62 s | Pass |
| Pinned rerun of `nominal-seed17` | 0.51 mm / 0.69° | 47.24 s | Pass |
| Reserved `shifted-seed29` | 0.69 mm / 2.28° | 48.76 s | Pass |

All three preserved complete evidence, sustained the final goal, withdrew the
pin and confirmed power off. The first two have identical case identities and
passed `evaluation.compare`; normal wall-clock scheduling need not reproduce
identical trajectories. Artifacts are under the ignored local folders
`runs/push-letters-evaluation-01`, `runs/push-letters-evaluation-repeat-01` and
`runs/push-letters-evaluation-heldout-01`. These check the recording and evaluation
workflow on three trials, not a success rate or scene-reconstruction accuracy.

## Evidence and verification

Each episode retains:

- `task/source/`, `task/invocation.json`, `task/trace.jsonl`: declared controller,
  scene, adapter and audit source, parameters, hashes and client calls.
- `task/artifacts/`: exact policy frame transport, including RGB, depth and calibration.
- `observation-*.png` / `.npz`, `observations.json`, `views.json`: actual policy
  images, aligned depth, selected points, estimated poses and camera calibration.
- `actions.json` and `flight/`: chosen pushes, full plans, outcomes and World Use records.
- `scene.mjb`, `physics/`, `replay.npz`, `final-state.npz`: compiled geometry and sampled physics,
  including measured robot joints and free letter poses. Replays do not rerun physics.
- `result.json`, `audit.json`: independent final-pose, stability, support, contact
  and power checks. Both the episode and audit return a nonzero status on failure.
- `demo.mp4`, `render.json`: optional continuous simulation replay with labeled
  playback speed. The overhead inset is rendered from the recorded physics state.

Local retained runs are `runs/push-letters-nominal-01`,
`runs/push-letters-shifted-run-01` and `runs/push-letters-friction-run-01`.
The full layout is `runs/push-letters-autolab-run-03`; wide-layout refusals are
`runs/push-letters-autolab-run-01` and `-02`.
Earlier trials remain in `runs/push-letters-camera-01` through `-06`.
Generated evidence and downloaded assets are ignored by Git.
Those earlier virtual-clock runs retain their original `source/` snapshots and
record format; current sessions use the shared layout above.

```sh
uv run pytest tests/test_push_letters.py
uv run ruff check examples/push-letters tests/test_push_letters.py
```

The regressions cover shadowed colour selection, background rejection,
image-to-pose reconstruction including B orientation, and rejecting incorrect
placements, bad contacts, solver warnings, incomplete runs and unresolved power.

## Attribution

The asset revision is pinned above. RSS is Apache-2.0; its Panda MJCF and meshes
come from robosuite under MIT, with Franka mesh provenance documented in the
[upstream asset README](https://github.com/robosimstudio/robosimstudio/blob/73739abe212f963e8a9f5862b27bca370df091a7/assets/embodiments/panda/README.md).
The clone retains its LICENSE and NOTICE files. This example does not redistribute
the robot meshes or the upstream kinematics URDF: its planning URDF is exported
from the loaded MuJoCo arm and checked against simulator forward kinematics.
