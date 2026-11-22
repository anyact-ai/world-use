# RoboWild pilot

[RoboWild](https://github.com/xlang-ai/RoboWild) tests frontier models as robot-use agents in 341 simulated tasks
(Isaac Sim) on four mobile robots. This pilot runs world-use's [hand tools](../../docs/hand-tools.md) on a tenth of
them: the 34 cross-embodiment tasks, on the mobile Franka, against the same model on its own.

**Status:** on the 10 development tasks, GPT-6.2 Astra (low reasoning) solved none, with world-use or without.

| arm | solved | model requests | Robot API calls | est. USD |
| --- | --- | --- | --- | --- |
| alone | 0 of 10 | 50 | 36 | 4.07 |
| world-use | 0 of 10 | 72 | 98 | 5.32 |

- Every attempt in both arms ended with the model calling `done`, after at most 34 of its 100 or 200 calls: it
  stopped when it could not verify a clearance or a grasp point, or when contact stopped the mobile base.
- With world-use the model acted more: it called the hand tools in 6 of the 10 tasks (11 grasps, 3 placements,
  1 push), and it stopped in the middle of the manipulation instead of before it. In 37-01 it placed three chips
  and stopped when the fourth would have closed on two.
- The held-out tasks have not been run. [development-check.jsonl](development-check.jsonl) has a row per arm and
  task, with the model's closing words; costs are estimates from token counts at the rates in `cases.toml`.

## The comparison

Both arms run through [RoboWild-Harness](https://github.com/xlang-ai/RoboWild-Harness) `eval` with the same model,
the `react_move_points` Robot API profile (`move_robot`, `get_camera_point_3d`, `done`), frozen simulation, seed 0,
one round, and each task's own call budget: 100 Robot API calls for a short task, 200 for a long one.

- **alone** is the harness's `react` backend, unchanged.
- **world-use** is [agent.py](agent.py), run as a `command` backend: the same `react` loop, with `grasp`, `place`,
  `release_placement` and `push` added and their help appended to its instructions. They run through
  [arm.py](arm.py), whose moves are `move_robot` calls on the same budget.

[cases.toml](cases.toml) lists the tasks, their budgets and a development/held-out split (10 and 24 tasks, chosen
by hash within short and long tasks). Development tasks may be inspected and replayed to build the tools; held-out
tasks are not inspected for tuning.

## What the tools do here, and what they do not

- A grasp opens above its pose, comes straight down, closes, checks the measured finger opening and lifts 3-5 cm
  with the opening monitored; a lean comes in along the hand's axis. A place carries, comes down to its pose,
  and stops with a one-use checkpoint before opening. A push follows a line with the fingers closed.
- An approach that stops short along its line, as on a counter or the object, reports how far short. The harness
  reports no forces: this is judged from where the hand ended.
- Positions come from the policy, in robot_base, with `get_camera_point_3d`. There is no region tracking, no
  locating through the wrist camera and no lift or carry evidence from images yet.
- A lean the arm cannot make fails at its first move with the planner's reason; this profile has no planning call
  to check it before the hand turns.

## Set up

On Linux with an NVIDIA RTX-class GPU, install RoboWild with its
[installer](https://github.com/xlang-ai/RoboWild#installation) (about 20 minutes and 18.5 GB of assets), then the
agent's environment: Python 3.13 with world-use and the harness's agent code, without its simulator dependencies.

```bash
uv venv --python 3.13 agent
```

```bash
uv pip install --python agent/bin/python -e /path/to/world-use
```

```bash
uv pip install --python agent/bin/python --no-deps -e RoboWild/robowild-harness -e RoboWild/robowild-harness/packages/robot-sdk
```

```bash
uv pip install --python agent/bin/python requests psutil transforms3d packaging pyyaml
```

Three things the installer leaves to the machine:

- RoboWild looks for Isaac Lab beside its checkout: link `IsaacLab` there if the installer's workspace is elsewhere.
- The harness copies its agent-side interpreter into each task workspace. Interpreters that find `libpython`
  through a path baked in relative to themselves (Modal's, for one) fail there with exit status 127. Pass
  `--worker-python` an environment on a copyable Python 3.12, such as uv's, with the harness installed.
- A virtual environment links to its interpreter. On a machine whose local disk is replaced while a mounted volume
  keeps the environments, as a restarted container's is, install the interpreters again at the same paths first.

## Run

In RoboWild's environment (`source activate_robowild.sh`), check the setup with a scripted plan first; it calls no
model:

```bash
python run.py --arm world-use --split development --tasks 01-01 --scripted plan.json --agent-python agent/bin/python --robowild RoboWild --output results/check
```

A plan is a list of tool calls, such as `[{"name": "done", "arguments": {}}]`; an argument `"$checkpoint"` passes on
the release checkpoint the call before it issued. Then each arm, on each split:

```bash
python run.py --arm alone --split development --model MODEL --credentials creds.json --budget-usd 50 --robowild RoboWild --output results/development --authorization "..."
```

```bash
python run.py --arm world-use --split development --model MODEL --credentials creds.json --budget-usd 50 --agent-python agent/bin/python --robowild RoboWild --output results/development --authorization "..."
```

`creds.json` holds `{"base_url": ..., "key": ...}` for the model's Responses endpoint; `run.py` deletes it once
read and starts [model_proxy.py](model_proxy.py), which alone holds the key, forwards the agents' requests and stops
the arm's spending at `--budget-usd` (prices per model in `cases.toml`). Each task is one harness evaluation, and
the next starts only while the budget left covers the costliest task so far, so the cap does not cut an attempt
short; `run.py` names the tasks it did not run. `--authorization` records the approval of the paid run with it.
Each task's directory under the arm holds the harness's `report.json`, `summary.md` and the run's videos and
records; the world-use agent's own records (`world-use.jsonl`, `moves.jsonl`) are beside its command logs.

## Checks

```bash
uv run pytest examples/robowild/test_arm.py examples/robowild/test_model_proxy.py examples/robowild/test_run.py
```

They run the hand tools through `RoboWildArm` on a fake mobile Franka whose results follow the harness's schemas,
the model proxy against a stand-in endpoint, and `run.py` against a stand-in harness.
