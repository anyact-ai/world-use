# Push letters onto their outlines

A simulated Panda arm with an 8 mm rounded pin pushes loose letters onto matching outlines, using an overhead
RGB-D camera for feedback. Each letter must finish within 5 mm and 5 degrees of its target, flat on the table and
still, with no robot contact except the pin. The arm then withdraws and releases power, and an audit checks the
outcome from the recorded physics.

The pushing policy is a Python program that Codex wrote and revised while watching development trials; no model
is called while it runs. It knows the letters' shapes, colours and target poses, and the overhead camera's
calibration. It estimates each letter's pose from a colour mask and a silhouette fit, chooses a short push with a
quasi-static friction model, pushes, and measures again. It assumes every letter is fully visible on a flat
table. This is neither general object recognition nor a blind test of a fresh agent.

In simulation, the seven-letter `AUTOLAB` layout passed in 34 pushes, and the three-letter `LAB` layout passed
with nominal, shifted and higher-friction starts without changing the policy. Each is a single development
trial, not a success rate; see [recorded results](#recorded-results).

This adapts the letter-pushing experiment that Shuangyu Xie showed in a screenshot of
[RoboSimStudio](https://github.com/robosimstudio/robosimstudio) (RSS), related to
[Revisiting Push-T with Agentic Robotics](https://arxiv.org/abs/2608.18227). The public RSS checkout does not
contain that `auto_lab/push` task or its policy. This example uses MuJoCo, a Panda and rigid compound letters
60 mm tall and 12 mm thick. It does not reproduce the screenshot's sponge material, robot, Newton solver or
unpublished controller, and makes no sim-to-real claim.

## How the arm is controlled

Every move goes through world-use's kernel: IK, joint limits, trajectories, the tracking watchdog and the flight
recorder. Letters move only through contact with the pin; nothing edits their poses after setup or attaches them
to the pin. In the default camera mode, only the audit and the recorder read the letters' true poses, velocities
and contacts. `--feedback state` gives the policy the true poses instead, as a diagnostic; it is not the camera
experiment.

The adapter declares a 70 mm box around the pin for the kernel's tool checks, sized from the capsule in
`scene.py`. The table and letters are not in the kernel's world model, and its box-only rehearsal cannot represent
this scene, so the client runs plans with `check=False` and each motion is checked as it starts. There is no joint
torque sensing. The policy rejects pushes whose approach would bring the pin within 2 mm of another stroke of the
letter, and the episode gives up after 20 pushes on one letter (`--max-pushes`). None of this says anything about
safety on hardware.

## Run locally

From the world-use repository root, with Python 3.13 and local graphics access:

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

Only the public Panda model assets from RSS are read; its Python package, Newton, Warp and a CUDA GPU are not
needed. macOS rendering needs access to CoreGraphics; headless Linux needs EGL or OSMesa. Video export also needs
`ffmpeg` on PATH. The scene builder and the episode refuse an existing output directory; the renderer overwrites
only its own `demo.mp4` and `render.json`. Episodes run in wall-clock time in the shared
[simulation session](../../docs/simulation.md#external-simulation-sessions).
`wu replay runs/letters-run/flight --out scene.gif --speed 3` renders the scene from its saved physics state.

Build with `--letters T`, `LAB` or `AUTOLAB`. `--variant shifted` offsets the starts by 12 mm / -8 mm and changes
yaw by -7 or +8 degrees. `--variant friction` raises the letters' sliding coefficient from 0.5 to 0.7; the bench
stays at 0.5, and MuJoCo uses the larger of the two coefficients for each contact. `episode.py --preview` saves
the untouched initial scene without enabling the arm.

## Compare policy revisions on fixed cases

`evaluate.py` runs named cases through `world_use.evaluation`. It freezes the scene, the policy sources, a setup
program and a separate evaluator. For each case, `setup.py` runs first in its own process: it jitters the letters'
starting poses with the case's seed, lets them settle and rejects an invalid start. The camera policy then runs
from that prepared scene, and `evaluation_audit.py` judges the outcome afterwards. Start with a single `T` scene
for a short development case:

```sh
uv run python examples/push-letters/scene.py \
  --assets runs/push-letters-assets/rss/assets/embodiments/panda \
  --output runs/evaluation-scene --letters T
uv run python examples/push-letters/evaluate.py \
  --scene runs/evaluation-scene --output runs/evaluation-policy-a
uv run python examples/push-letters/evaluate.py \
  --rerun runs/evaluation-policy-a --case-name nominal-seed17 --output runs/evaluation-repeat
```

`--rerun` runs a recorded case again from its saved sources and inputs, and refuses to run under a different
Python, package set or runtime. The saved `scene.mjb` contains the meshes but loads only in the recorded MuJoCo
version; the saved XML and `task.json` record where the assets came from. To compare a revised policy, run the
same scene and case into another fresh output, then call `world_use.evaluation.compare(first, second)`. It pairs
cases by an identity built from the seed, parameters, scene, input hashes and setup revision, but not the policy,
so both revisions start each paired case from the same state.

`--case held-out` selects the reserved `shifted-seed29` case. Once a case has been used to tune the policy, it no
longer counts as independent validation. `--privileged` runs the state-feedback diagnostic, which `compare`
refuses to pair with camera runs. `--validation` attaches a scene-validation report for this exact scene; without
one, the scene's accuracy is recorded as unvalidated even when the task passes.

A suite folder holds `suite.json`, `trials/<name>/trial.json` for each case, the frozen setup, policy and
evaluator invocations, the prepared scene, the episode's own records and `summary.json`. The report keeps invalid
starts, interrupted or unstarted cases, incomplete recordings, failure reasons and recovery moves, and keeps wall
and simulation time apart. Besides the physical audit, the evaluator requires the goal conditions to hold for the
final 1.9 simulated seconds with no gap over 100 ms between samples. `world_use.evaluation.summarize(suite)` reads
a suite that was interrupted before writing its summary. The three programs are separate trusted Python
processes, not a sandbox.

## Recorded results

On October 15, 2026, the camera policy placed all three `LAB` letters in each of three development conditions,
with unchanged policy code:

| Condition | Pushes | Simulation time, including final settling | Largest position / yaw error |
| --- | ---: | ---: | --- |
| Nominal | 14 | 135.74 s | 3.51 mm / 3.51° |
| Shifted and rotated starts | 14 | 129.48 s | 2.51 mm / 3.06° |
| Higher friction | 14 | 134.70 s | 3.20 mm / 2.31° |

These are three single trials, not a statistical success rate. Each passed the audit: table support, two-second
stability, pin withdrawal, finite states, no solver warnings, no other robot or letter contacts, and power off.
Maximum contact penetration was 0.1233 mm across the three. Simulation ran on a virtual clock, so the durations are not wall-clock time or
inference latency.

Earlier trials exposed an unstable contact between a flat cylinder pin and the letters during withdrawal,
background pixels entering a colour mask, and an initial B shape whose orientation was ambiguous. The capsule pin,
absolute colour-distance rejection and asymmetric B were all in place before the three trials above.

The full seven-letter `AUTOLAB` layout then passed 7/7 letters in 34 pushes, with maximum final errors of 2.72 mm
and 2.35° and 0.1158 mm maximum contact penetration. It used the same perception and contact selection, with a
wider camera-clear parking pose and high transit waypoints for the wider layout. The episode took 442.99 simulated
seconds, including settling. This is one development success; the two wide-layout attempts before it were refused
before their first push by the segment-length and reach checks.

On October 16, 2026, the shared evaluation path ran these one-letter `T` cases with unchanged camera-policy source:

| Named case | Position / yaw error | Simulation time | Audit |
| --- | --- | ---: | --- |
| `nominal-seed17` | 0.61 mm / 0.45° | 47.62 s | Pass |
| Pinned rerun of `nominal-seed17` | 0.51 mm / 0.69° | 47.24 s | Pass |
| Reserved `shifted-seed29` | 0.69 mm / 2.28° | 48.76 s | Pass |

All three kept complete records, held the goal to the end, withdrew the pin and ended with power off. The first
two have the same case identity and passed `evaluation.compare`; their trajectories differ because episodes run on
wall-clock scheduling. These three runs test the recording and evaluation workflow, not a success rate or the
accuracy of the scene.

The runs are kept locally and ignored by Git, as are the downloaded assets: `runs/push-letters-nominal-01`,
`runs/push-letters-shifted-run-01` and `runs/push-letters-friction-run-01`; the full layout in
`runs/push-letters-autolab-run-03` and the refused attempts in `runs/push-letters-autolab-run-01` and `-02`;
earlier trials in `runs/push-letters-camera-01` through `-06`; and the evaluations in
`runs/push-letters-evaluation-01`, `runs/push-letters-evaluation-repeat-01` and
`runs/push-letters-evaluation-heldout-01`. Runs made on the virtual clock keep their original `source/` snapshots
and record format; current episodes use the layout below.

## What an episode saves

- `task/source/`, `task/invocation.json`, `task/trace.jsonl`: the controller, scene, adapter and audit source,
  parameters, hashes and client calls.
- `task/artifacts/`: each camera frame exactly as the policy received it, including RGB, depth and calibration.
- `observation-*.png` / `.npz`, `observations.json`, `views.json`: the policy's images, aligned depth, selected
  points, estimated poses and camera calibration.
- `actions.json` and `flight/`: chosen pushes, full plans, outcomes and world-use records.
- `scene.mjb`, `physics/`, `replay.npz`, `final-state.npz`: compiled geometry and sampled physics, including the
  robot's joints and the letters' free poses. Replays do not rerun physics.
- `result.json`, `audit.json`: the audit's final-pose, stability, support, contact and power checks. Both the
  episode and the audit exit with a nonzero status on failure.
- `demo.mp4`, `render.json`: an optional continuous replay labelled with its playback speed. The overhead inset is
  rendered from the recorded physics state.

```sh
uv run pytest tests/test_push_letters.py
uv run ruff check examples/push-letters tests/test_push_letters.py
```

The tests cover colour selection in shadow, background rejection, recovering a letter's pose (including the B's
orientation) from an image, and the audit rejecting wrong placements, bad contacts, solver warnings, incomplete
runs and unresolved power.

## Attribution

The asset revision is pinned above. RSS is Apache-2.0; its Panda MJCF and meshes come from robosuite under MIT,
with Franka mesh provenance documented in the
[upstream asset README](https://github.com/robosimstudio/robosimstudio/blob/73739abe212f963e8a9f5862b27bca370df091a7/assets/embodiments/panda/README.md).
The clone keeps its LICENSE and NOTICE files. This example does not redistribute the robot meshes or the upstream
kinematics URDF: its planning URDF is exported from the loaded MuJoCo arm and checked against the simulator's
forward kinematics.
