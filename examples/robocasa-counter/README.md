# Franka Panda in a RoboCasa kitchen

An LLM uses world-use to pick a cup from the counter and put it in the sink.
The room, fixtures, object meshes, robot
and task come from [RoboCasa](https://github.com/robocasa/robocasa)
and [robosuite](https://github.com/ARISE-Initiative/robosuite).
The Panda uses its upstream Omron base, parked throughout the episode.

The selected episode completes pickup, transfer, release and withdrawal. The cup
settles **on its side** in the sink. Codex chose actual world-use MCP calls
interactively, using RGB-D images, EdgeTAM selections, measured surface points
and gripper feedback. A fresh selection after lifting measured a 79 mm rise in
the visible cup surface. The video shows these actual returns beside the motion.

The target mesh is scaled to 0.75, giving a roughly 60 mm cup body that fits the
Panda's 80 mm jaw travel. The base is parked 20 cm left of the upstream reset
position before recording. These are setup changes, not agent-controlled actions.
Simulator object poses were not supplied as policy observations. The same agent
built the setup and chose the calls; this is a selected development trial, not a
blind evaluation or a repeatability result.

`adapter.py` connects the exported scene to world-use's joint I/O contract. It
uses the original rigid-body contact geometry, a parked mobile base, joint PD
servos and gravity compensation. The core library handles IK, trajectory limits,
feedback, evidence freshness and recording. Full kitchen contact rehearsal is
not implemented; the calls explicitly use `rehearse=False`. No torque sensing or
hardware transfer is demonstrated.

The selected local recording is `runs/robocasa-agent-06/`: `panda-world-use-clear.mp4`
is the edited video, `calls.json` contains requests and returns, `replay.npz`
contains sampled simulator configurations, and `flight/perception/` contains
the exact measured points and images. The edit retains a gripper command stop
and a rejected depth measurement followed by reselection. Motion plays at 1x;
decision pauses from the roughly 11-minute 40-second recording are cut and
held frames are labelled. The panel links each tool result to the recorded LLM
action, with labelled positions and a before/after lift check. The main scene
replays recorded physics with its original warm-white lighting.

`audit_episode.py` checks the final state after recording: the cup contacts the
sink, moves less than 1 mm over the last ten seconds, and is more than 25 cm from
the tool, with torque off. The selected run passes. This audit uses simulator
state after the episode; it was not an input to the agent.

Earlier recordings remain under `runs/`: 01–02 are adapter checks, 03 is an
approach preview, 04 is an unsuccessful larger-cup trial, and 05 is a setup check.
Generated recordings and downloaded assets are ignored by Git.

After building the scene, use world-use's Python 3.13 environment with its vision
dependencies to start the interactive recorder. The output directory must be new:

```sh
HF_HOME=/private/tmp/world-use-parts-tray-hf .venv/bin/python -i examples/robocasa-counter/agent_episode.py --output runs/robocasa-agent-new
```

The recorder exposes `act(tool, intent=..., **arguments)` and `finish()`. It
contains no pickup policy or prewritten action sequence. Read `wu policy`, inspect
the cameras and choose each call from its returns. Release the robot before
calling `finish(result="...")`. EdgeTAM weights download on first use; `HF_HOME`
is optional and can point to any writable model cache.

The renderer's edit is specific to episode 06; a new episode needs its own call
selection. It uses Pillow, ffmpeg and the macOS SF fonts. Native rendering requires
local graphics access.

```sh
.venv/bin/python examples/robocasa-counter/audit_episode.py runs/robocasa-agent-06
.venv/bin/python examples/robocasa-counter/render_episode.py runs/robocasa-agent-06 --output-name panda-world-use-clear
```

## Build the upstream scene

The environment requires an older MuJoCo than world-use. Use a separate
Python 3.11 environment to build and inspect it. These preview dependencies omit
upstream training, teleoperation and demonstration-generation extras. Do not
install them into world-use's environment.

From the world-use repository root:

```sh
uv venv runs/robocasa-preview/.venv --python 3.11
git clone https://github.com/robocasa/robocasa.git runs/robocasa-preview/vendor/robocasa
git clone https://github.com/ARISE-Initiative/robosuite.git runs/robocasa-preview/vendor/robosuite
git -C runs/robocasa-preview/vendor/robocasa checkout 456174f62b89b8fca99eaaf33949c29fec9cfc2a
git -C runs/robocasa-preview/vendor/robosuite checkout 5ce6643f3092639d08f7b0f90ed1c6a84f50552c
uv pip install --python runs/robocasa-preview/.venv/bin/python -r examples/robocasa-counter/requirements-preview.txt
uv pip install --python runs/robocasa-preview/.venv/bin/python --no-deps -e runs/robocasa-preview/vendor/robocasa -e runs/robocasa-preview/vendor/robosuite
```

Download `textures.zip`, `fixtures.zip` and `objaverse.zip` from the
[official RoboCasa asset repository](https://huggingface.co/datasets/robocasa/robocasa-assets).
Extract textures and fixtures into
`runs/robocasa-preview/vendor/robocasa/robocasa/models/assets/`, and objects into
its `objects/` directory. Each archive contains its own top-level folder.
Also extract the current `fixtures_lightwheel.zip` into `models/assets/`.
Its upstream location is recorded in RoboCasa's `box_links_assets.json`; if that
Box link is unavailable, the [fixture mirror](https://huggingface.co/datasets/IIFAN/robocasa365-lightwheel-fixtures)
provides the same named pack. The four archives total approximately 3.5 GB before
extraction. The official object mirror contains the newer region-based XML
format; it cannot be loaded by RoboCasa v0.2.

```sh
runs/robocasa-preview/.venv/bin/python examples/robocasa-counter/preview.py
```

The script subclasses the existing `PickPlaceCounterToSink` task to scale its
cup target, using layout 1, Scandinavian style 2 and seed 7. It settles the scene using the upstream
controller, then saves native MuJoCo overview and camera images, depth, state,
compiled model, composed MJCF and metadata to `runs/robocasa-small-cup/scene/`. Rendering requires
local graphics access. No Blender or learned rendering is involved.

Optional knife-block, scale, paper-towel, oil/soap and glass decorations are
disabled with RoboCasa's fixture configuration. Task objects and the kitchen's
furniture, sink, appliances and contact geometry remain upstream assets.

Source revisions used:

- RoboCasa 1.0.1: `456174f62b89b8fca99eaaf33949c29fec9cfc2a`.
- robosuite 1.5.2: `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`.

## Attribution

RoboCasa365 is by Soroush Nasiriany, Sepehr Nasiriany, Abhiram Maddukuri and
Yuke Zhu, building on the original RoboCasa team's work. Its code is MIT licensed;
its released assets are CC BY 4.0. Robosuite and the upstream robot assets retain
their respective notices. Source and downloaded assets stay in the ignored
`runs/` directory; this example does not redistribute their meshes or textures.
