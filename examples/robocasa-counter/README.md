# RoboCasa kitchen setup

Shared setup for the [lunchbox](../robocasa-lunchbox) and [oven-dial](../robocasa-dial) demos: a Franka Panda from
[robosuite](https://github.com/ARISE-Initiative/robosuite) in a [RoboCasa](https://github.com/robocasa/robocasa)
kitchen, with its Omron base parked. It also holds an earlier [cup-to-sink example](#cup-to-sink-example), in which
Codex chose MCP calls one at a time to put a cup in the sink.

[adapter.py](adapter.py) drives the arm with PD servos and gravity compensation and keeps the upstream meshes and
contact geometry. It reports gripper position and effort, but no joint torques. world-use's rehearsal cannot model
the kitchen, so plans run with `rehearse=False`; world-use still provides IK, trajectory limits, freshness checks
and records.

## Build the upstream scene

Build the scene in a separate Python 3.11 environment, because upstream needs an older MuJoCo than world-use. Keep
these dependencies out of the runtime environment.

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
[official RoboCasa asset repository](https://huggingface.co/datasets/robocasa/robocasa-assets), and the current
`fixtures_lightwheel.zip` from the Box link in RoboCasa's `box_links_assets.json`; if that link is unavailable, the
[fixture mirror](https://huggingface.co/datasets/IIFAN/robocasa365-lightwheel-fixtures) has the same pack. The four
archives total about 3.5 GB before extraction, and each contains its own top-level folder. Extract the textures and
both fixture archives into `runs/robocasa-preview/vendor/robocasa/robocasa/models/assets/`, and the objects into its
`objects/` directory. The objects use the newer region-based XML format, which RoboCasa v0.2 cannot load.

```sh
runs/robocasa-preview/.venv/bin/python examples/robocasa-counter/preview.py
```

The script builds the upstream `PickPlaceCounterToSink` task with the cup scaled to 0.75, layout 1, style 2 and
seed 7, and exports images, depth, state, the compiled model and metadata to `runs/robocasa-small-cup/scene/`. It
disables the optional counter decorations; furniture and contact geometry stay as upstream. Rendering needs local
graphics access. The pinned revisions are RoboCasa 1.0.1 and robosuite 1.5.2.

## Prepare the runtime

After exporting, install world-use with vision in the repository's Python 3.13 environment:

```sh
uv sync --locked --python 3.13 --extra vision
```

Run task scripts with `.venv/bin/python` from the repository root. EdgeTAM downloads its weights before the arm is
powered; `HF_HOME` selects another cache. Video rendering also needs ffmpeg and the macOS SF fonts.

Continue with [lunchbox packing](../robocasa-lunchbox#run-locally) or
[oven-dial control](../robocasa-dial#run-locally), or try the cup task below.

## Cup-to-sink example

Codex chose MCP calls from RGB-D, EdgeTAM and gripper feedback and placed a cup in the sink, where it settled on
its side. This was a development run: the same agent built the scene and then chose the calls. Simulator poses were
used only by the audit. Before recording, the cup was scaled to 0.75 (roughly 60 mm wide, for 80 mm jaws) and the
base parked 20 cm left of its upstream reset position.

Start an interactive recorder with a new output directory:

```sh
.venv/bin/python -i examples/robocasa-counter/agent_episode.py --output runs/robocasa-agent-new
```

At the prompt, `act(tool, intent=..., **arguments)` makes one MCP call and `finish(result="...")` closes the
episode; there is no prewritten pickup policy. Read `wu policy`, inspect the cameras and choose calls from their
replies. Release the robot before `finish`. Episodes use the shared
[simulation session](../../docs/simulation.md#external-simulation-sessions): `task/` holds the declared source and
the client's observations, `physics/` the full-scene states, and `calls.json` each call's intent, reply and
returned images. `wu replay runs/robocasa-agent-new/flight --out scene.gif --speed 3` renders a fixed-camera
replay.

```sh
.venv/bin/python examples/robocasa-counter/audit_episode.py runs/robocasa-agent-new
```

The audit reads the recorded simulator state: the cup must touch the sink, move less than 1 mm over the last ten
wall-clock seconds and be more than 25 cm from the gripper, and torque must be off.

<details>
<summary>Selected recording and earlier trials</summary>

The local archive `runs/robocasa-agent-06/` contains `calls.json` (requests and replies), `replay.npz` (sampled
simulator states) and `flight/perception/` (measured points and images). Its audit passes. A fresh selection after
lifting measured a 79 mm rise in the visible cup surface.

The archived `panda-world-use-clear.mp4` plays motion at 1× and cuts the decision gaps from a session of about
11 minutes 40 seconds; held frames are labelled. It includes a gripper close that world-use stopped early on
contact, and a depth measurement rejected for mixing cup and background, followed by a new selection.
`render_episode.py` is written for episode 06's call order; a new episode needs its own selection of calls.

Earlier trials 01–02 are adapter checks, 03 is an approach preview, 04 is an unsuccessful larger-cup attempt, and
05 is a setup check. Raw recordings and downloaded assets are not included in this repository.

</details>

## Attribution

RoboCasa365 is by Soroush Nasiriany, Sepehr Nasiriany, Abhiram Maddukuri and Yuke Zhu, building on the original
RoboCasa team's work. Its code is MIT licensed; its released assets are CC BY 4.0. Robosuite and the upstream robot
assets keep their own notices. Source and downloaded assets stay in the ignored `runs/` directory; this example
does not redistribute their meshes or textures.
