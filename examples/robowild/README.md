# RoboWild pilot

[RoboWild](https://github.com/xlang-ai/RoboWild) tests frontier models as robot-use agents in 341 simulated tasks
(Isaac Sim) on four mobile robots. This pilot runs world-use's [hand tools](../../docs/hand-tools.md) on a tenth of
them: the 34 cross-embodiment tasks, on the mobile Franka, against the same model on its own.

**Status:** on the 10 development tasks, GPT-6.2 Astra at high reasoning solved 2 on its own, and 1 with world-use
both before and after the changes the first runs led to; at low reasoning neither arm solved one. All results are
from RoboWild's simulator, one attempt per task.

| reasoning | arm | solved | model requests | Robot API calls | est. USD |
| --- | --- | --- | --- | --- | --- |
| low | alone | 0 of 10 | 50 | 40 | 4.08 |
| low | world-use | 0 of 10 | 72 | 98 | 5.32 |
| high | alone | 2 of 10 (01-01, 37-01) | 560 | 551 | 324.92 |
| high | world-use | 1 of 10 (01-01) | 405 | 500 | 163.43 |
| high | world-use, rerun at 4977816 | 1 of 10 (29-06) | 589 | 736 | 256.35 |
| high | world-use, 06-06 and 22-10 at 7f328b9 | 1 of 2 (22-10) | 68 | 86 | 18.27 |

- At low reasoning every attempt ended with the model calling `done` after at most 34 of its 100 or 200 calls, when
  it could not verify a clearance or a grasp point, or when contact stopped the mobile base. With world-use it called
  the hand tools in 5 tasks (11 grasps, 3 placements, 1 push) and stopped in the middle of the manipulation instead
  of before it.
- At high reasoning the model drove the arm itself with `move_robot` in both arms, measuring with
  `get_camera_point_3d` between moves. With world-use it called the hand tools in 4 tasks (35 grasps, 7 placements)
  and spent half as much.
- 37-01, sorting poker chips into stacks, is the task the tools lost. A grasp makes 4 calls and a place with its
  release 4: the model used `place` for 7 chips and then carried each with 2 `move_robot` calls, and 11 of its 32
  grasps closed empty or slipped. It ran out of the task's 200 calls; on its own it sorted the chips in 155.
- The rerun, with the changes below, solved 29-06 and lost 01-01: with one attempt per task, the runs differ as much
  from each other as the arms do. In 37-01, with each tool's call count in its result, the model used `place` for
  20 chips and had all five denominations stacked in order, a few chips loose, at 197 calls and $20.
- In the first runs at high reasoning, 788 of the 965 requests paid the full input price, and made 97% of the cost:
  a request with 130,000 tokens of context cost about $1 instead of $0.10. Azure OpenAI caches no prompt prefix
  through the Responses API's `instructions` field, which the harness's loop uses for its rules, so only the tool
  definitions before it were cached. `model_proxy.py` now sends the instructions as a leading developer message
  instead, for both arms: in a loop-shaped test every request after the first was cached but for its newest
  tokens, where none had been. The runs above were made without that change.
- Runs that did not go as planned. The baseline's 29-06 attempt was ended by its run's spending cap after 69 calls,
  before the model called `done`: it counts as not solved. In the baseline's 23-03 and 37-01 and the rerun's 04-02
  the cap was raised during the attempt by restarting the model proxy on the same port and ledger, and the attempts
  carried on. `run.py` now starts a task only while the budget left covers the costliest so far.
- What the runs changed: `yaw_deg`'s help gives the direction of a turn; fingers stopped on something narrower than
  expected are `stopped_apart`, not `reached_end`; each result ends with the call count. After the rerun the grasp
  schema also states the rules behind five of its six refusals (the 4 mm floor of an expected opening, the 3-5 cm
  test lift, `tilt_deg` without `yaw_deg`) and that a rounded object reads under its visible width. Run once on
  the two tasks they bear on most, no grasp was refused; the model expected 5-38 mm for a sausage, whose grasps had
  read 5.5-8 mm against 12 mm or more expected in every earlier run, and its grasp passed; and 22-10, the cake dome,
  was solved for the first time in any run.
- The held-out tasks have not been run. [development-check.jsonl](development-check.jsonl) has a row per run and task
  from [report.py](report.py), with the model's closing words; costs are estimates from token
  counts at the rates in `cases.toml`.

## The comparison

Both arms run through [RoboWild-Harness](https://github.com/xlang-ai/RoboWild-Harness) `eval` with the same model,
the `react_move_points` Robot API profile (`move_robot`, `get_camera_point_3d`, `done`), frozen simulation, seed 0,
one round, and each task's own call budget: 100 Robot API calls for a short task, 200 for a long one.

- **alone** is the harness's `react` backend, unchanged.
- **world-use** is [agent.py](agent.py), run as a `command` backend: the same `react` loop, with `grasp`,
  `close_grasp`, `place`, `release_placement`, `push`, `tilt` and `move_sequence` added and their help appended to
  its instructions. They run through [arm.py](arm.py), whose moves are `move_robot` calls on the same budget. Two
  other differences came from the development runs: the help ends with one sentence that the attempt is judged on
  the scene as it ends, so the model should leave what the task asks for in place, and old images are dropped six at
  a time, not one per move, so the prompt cache holds (the model can then see up to six more images than the loop's
  rules say).

[cases.toml](cases.toml) lists the tasks, their budgets and a development/held-out split (10 and 24 tasks, chosen
by hash within short and long tasks). Development tasks may be inspected and replayed to build the tools; held-out
tasks are not inspected for tuning.

## What the tools do here, and what they do not

- A grasp opens above its pose, fully or to `open_mm`, comes straight down, closes, checks the measured finger
  opening and lifts 3-5 cm with the opening monitored: four `move_robot` calls. A lean comes in along the hand's
  axis. With `stop_before_close` it stops with the fingers open around the object and a one-use checkpoint, so the
  model can look first; `close_grasp` then closes and lifts. The close is its own call: sent with the approach, it
  runs while the arm is still settling, and in the runs that did so it nearly doubled grasps that closed on
  nothing. A place carries, comes down to its pose, and stops with a one-use checkpoint before opening. A push
  follows a line with the fingers closed.
- `tilt` turns the held object about a point of it, such as a container's mouth, at most 30 degrees per call with
  the grip checked after each, optionally after carrying that point over a target, and reports where it ended.
- `move_sequence` runs up to 8 `move_robot` calls in one turn, each with its own goals, and stops after the first
  that is refused, ends in a safety stop or leaves a goal untracked. The harness's loop takes one tool call per turn,
  and at high reasoning a turn took 25-81 s on average, so a long task's 200 calls, one per turn, would fill its
  5,400 s timeout. On real robots, [Robocurve](https://robocurve.org/blog/codex-vs-inspect-robots/) traced the same
  model's better scores in one harness to the tools it wrote there, and suspects that sending several moves per turn
  is what does the work.
- An approach that stops short along its line, as on a counter or the object, reports how far short. The harness
  reports no forces: this is judged from where the hand ended.
- Positions come from the policy, in robot_base, with `get_camera_point_3d`. There is no region tracking, no
  locating through the wrist camera and no lift or carry evidence from images yet.
- A lean the arm cannot make fails at its first move with the planner's reason; this profile has no planning call
  to check it before the hand turns.
- Each tool's result ends with `_harness`, the task's call count after it, as a Robot API result does.

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
records; the world-use agent's own records (`world-use.jsonl`, `moves.jsonl`) are beside its command logs. One row
per task, as in `development-check.jsonl`:

```bash
python report.py results/development/alone results/development/world-use > rows.jsonl
```

## Checks

```bash
uv run pytest examples/robowild
```

They run the hand tools through `RoboWildArm` on a fake mobile Franka whose results follow the harness's schemas,
the model proxy against a stand-in endpoint, `run.py` against a stand-in harness and `report.py` on a run's
records.
