# RoboCasa kitchen setup

Shared setup for the [lunchbox](../robocasa-lunchbox) and [oven-dial](../robocasa-dial)
demos: a Franka Panda from [robosuite](https://github.com/ARISE-Initiative/robosuite)
in a [RoboCasa](https://github.com/robocasa/robocasa) kitchen, with its Omron base parked.

[adapter.py](adapter.py) connects the scene to world-use's joint I/O contract
using upstream contact geometry, PD servos and gravity compensation. It provides
gripper feedback but no joint torque sensing. Full-scene rehearsal is disabled
(`rehearse=False`); world-use still supplies IK, trajectory limits, freshness checks and records.

## Build the upstream scene

Build the scene in a separate Python 3.11 environment: upstream requires an older
MuJoCo than world-use. Keep these dependencies out of the runtime environment.

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

The script exports `PickPlaceCounterToSink` with a scaled cup, layout 1, style 2
and seed 7 to `runs/robocasa-small-cup/scene/`: images, depth, state, model and
metadata. Optional decorations are disabled; furniture and contact geometry
remain upstream. Rendering needs local graphics access. The pinned revisions
are RoboCasa 1.0.1 and robosuite 1.5.2.

## Prepare the runtime

After exporting, install world-use with vision in the repository's Python 3.13 environment:

```sh
uv sync --locked --python 3.13 --extra vision
```

Run task scripts with `.venv/bin/python` from the repository root. EdgeTAM
downloads weights before powering the arm; `HF_HOME` can select another cache.
Video rendering also requires ffmpeg and macOS SF fonts.

Continue with [lunchbox packing](../robocasa-lunchbox#run-locally) or
[oven-dial control](../robocasa-dial#run-locally), or try the earlier cup task below.

## Cup-to-sink example

Codex used RGB-D, EdgeTAM and gripper feedback to choose MCP calls that placed
a cup in the sink, where it settled on its side. This was a development run:
the same agent built the scene and chose the calls. Simulator poses were audit-only.

Before recording, the cup was scaled to 0.75 (roughly 60 mm wide for 80 mm jaws)
and the base parked 20 cm left of its upstream reset position.

Start an interactive recorder with a new output directory:

```sh
.venv/bin/python -i examples/robocasa-counter/agent_episode.py --output runs/robocasa-agent-new
```

The recorder exposes `act(tool, intent=..., **arguments)` and `finish()`, with no
prewritten pickup policy. Read `wu policy`, inspect the cameras and choose calls
from their returns. Release the robot before `finish(result="...")`.
Each episode uses the shared [simulation session](../../docs/simulation.md#external-simulation-sessions):
`task/` retains declared source and exact client observations, while `physics/`
keeps incremental full-scene states. `calls.json` adds the task's MCP intents and presentation images.
Use `wu replay runs/robocasa-agent-new/flight --out scene.gif --speed 3` for a fixed-camera replay.

```sh
.venv/bin/python examples/robocasa-counter/audit_episode.py runs/robocasa-agent-new
```

The audit checks cup contact with the sink, less than 1 mm of movement in the last
ten seconds, more than 25 cm of tool clearance and torque off. Command completion
alone does not establish these results.

<details>
<summary>Selected recording and earlier trials</summary>

The local archive `runs/robocasa-agent-06/` contains `calls.json` (requests and
returns), `replay.npz` (sampled simulator states) and `flight/perception/`
(measured points and images). Its audit passes. A fresh selection after lifting
measured a 79 mm rise in the visible cup surface.

The archived `panda-world-use-clear.mp4` plays motion at 1×, with decision gaps
cut from an approximately 11-minute 40-second session and held frames labeled.
It retains a gripper command stop and a rejected depth measurement followed by
reselection. The renderer is specific to episode 06; a new episode needs its
own call selection.

Earlier trials 01–02 are adapter checks, 03 is an approach preview, 04 is an
unsuccessful larger-cup attempt, and 05 is a setup check. Raw recordings and
downloaded assets are not included in this repository.

</details>

## Attribution

RoboCasa365 is by Soroush Nasiriany, Sepehr Nasiriany, Abhiram Maddukuri and
Yuke Zhu, building on the original RoboCasa team's work. Its code is MIT licensed;
its released assets are CC BY 4.0. Robosuite and the upstream robot assets retain
their respective notices. Source and downloaded assets stay in the ignored
`runs/` directory; this example does not redistribute their meshes or textures.
