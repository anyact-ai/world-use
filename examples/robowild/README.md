# RoboWild pilot

[RoboWild](https://github.com/xlang-ai/RoboWild) tests frontier models as robot-use agents in 341 simulated tasks
(Isaac Sim) on four mobile robots. This pilot runs world-use's [hand tools](../../docs/hand-tools.md) on a tenth of
them: the 34 cross-embodiment tasks, on the mobile Franka, against the same model on its own.

**Status:** on the 10 development tasks, GPT-6.2 Astra at high reasoning solved 2 on its own. With world-use, the
two full runs closest to the tools as they are solved 4 (at d3eed9f) and 1 (at e57e81e, this code); over all 59
world-use attempts at high reasoning, 13 were solved, 22%. All results are from RoboWild's simulator, one attempt per
task. The held-out tasks have not been run.

| reasoning | arm | world-use commit | solved | model requests | Robot API calls | est. USD |
| --- | --- | --- | --- | --- | --- | --- |
| low | alone | | 0 of 10 | 50 | 40 | 4.08 |
| low | world-use | bd36c12 | 0 of 10 | 72 | 98 | 5.32 |
| high | alone | | 2 of 10 (01-01, 37-01) | 560 | 551 | 324.92 |
| high | world-use | babe7cf | 1 of 10 (01-01) | 405 | 500 | 163.43 |
| high | world-use | 4977816 | 1 of 10 (29-06) | 589 | 736 | 256.35 |
| high | world-use, one or a few tasks per change | 7f328b9 to ba7f0bc | 5 of 13 | 873 | 1,225 | 124.54 |
| high | world-use, stopped after 6 tasks | 600d12d | 1 of 6 (01-01) | 382 | 415 | 50.40 |
| high | world-use | d3eed9f | 4 of 10 (01-01, 02-06, 09-06, 29-06) | 781 | 952 | 99.06 |
| high | world-use | e57e81e | 1 of 10 (37-01) | 854 | 1,108 | 108.35 |

- One attempt per task is too few to tell these changes apart. The runs at d3eed9f and e57e81e differ by
  `move_sequence`, the quaternion scaling, the recovery sentence (see the comparison below) and the pose a resumed
  `close_grasp` reports starting from. At e57e81e the four tasks solved at d3eed9f failed as single attempts do: the
  keychain ended on the table below its hook, a rim grip on the bottle slipped, 44% of the reagent reached the
  beaker, and the carrot lay in the basin outside the faucet's stream. By task, the 59 attempts solved 01-01 4 times
  in 7; 09-06, 22-10, 29-06 and 37-01 2 in 7; 02-06 once in 5; and 04-02, 06-06, 23-03 and 36-01 never. At 22% per
  attempt a run of 10 averages about 2.
- Cost: since the prompt-cache fix a full run costs $99-108 for 781-854 model requests, about $0.13 each, against
  $0.40-0.43 a request in the world-use runs before it and $0.58 in the model's own run, where 788 of 965 requests
  paid the full input price. Azure OpenAI caches no prompt prefix through the Responses API's `instructions` field,
  which the harness's loop uses for its rules: `model_proxy.py` sends them as a leading developer message instead,
  for both arms, and the world-use arm drops old images six at a time, so a move does not end the cached prefix. The
  model's own arm was not run again with the fix.
- What the changes since 4977816 measurably did:
  - A grasp that closed as its approach arrived, one call fewer, closed empty 27 times in 48 at 600d12d. Closing as
    its own call once the arm settles, as since d3eed9f, closed empty 14 times in 83 and 22 in 101 in the two full
    runs after it.
  - `tilt`: 09-06, the pour, was solved in 2 of its 5 attempts since, and in neither before.
  - No quaternion was refused at e57e81e, against 6 calls at d3eed9f; across all runs, 21 of the 26 refused
    `move_robot` calls were quaternions off unit length.
  - `move_sequence` ran 78 sequences at e57e81e, and model requests took 41 s on average there against 38 s at
    d3eed9f. It did not keep the long tasks inside the 5,400 s timeout: 06-06 and 23-03 reached it at 118 and 137
    calls.
  - The sentences on how an attempt is judged and that a setback does not end it showed no effect on when the model
    stops: at e57e81e it called `done` with 73 calls left in 22-10, the cake dome having fallen out of reach, and
    with 49 left in 29-06, taking the carrot for rinsed.
  - The grasp schema's rules (a 4 mm floor under an expected opening, the 3-5 cm test lift, `tilt_deg` only with
    `yaw_deg`, a rounded object reading under its visible width) left no grasp refused in the check run on the two
    tasks they bear on most, 06-06 and 22-10.
- Where attempts are lost: of the 22 and 23 slips in the two full runs, 20 and 18 came in moves the model sent
  itself, often short straight carries, which points at the grip it took more than at the speed. Nine in ten of the
  tools' free moves end within 2 mm of their goal, and approaches that may end on contact within 10 mm.
- At low reasoning every attempt ended with the model calling `done` after at most 34 of its 100 or 200 calls, when
  it could not verify a clearance or a grasp point, or when contact stopped the mobile base.
- Runs that did not go as planned. The baseline's 29-06 attempt was ended by its run's spending cap after 69 calls,
  before the model called `done`: it counts as not solved. In the baseline's 23-03 and 37-01 and the 4977816 run's
  04-02 the cap was raised during the attempt by restarting the model proxy on the same port and ledger, and the
  attempts carried on. `run.py` now starts a task only while the budget left covers the costliest so far. The run at
  600d12d was stopped during its seventh task, once its empty closes showed the regression above; that attempt is
  not counted, and the run spent $69.31 in all.
- [development-check.jsonl](development-check.jsonl) has a row per run and task from [report.py](report.py), with
  the model's closing words; costs are estimates from token counts at the rates in `cases.toml`.

## The comparison

Both arms run through [RoboWild-Harness](https://github.com/xlang-ai/RoboWild-Harness) `eval` with the same model,
the `react_move_points` Robot API profile (`move_robot`, `get_camera_point_3d`, `done`), frozen simulation, seed 0,
one round, and each task's own call budget: 100 Robot API calls for a short task, 200 for a long one.

- **alone** is the harness's `react` backend, unchanged.
- **world-use** is [agent.py](agent.py), run as a `command` backend: the same `react` loop, with `grasp`,
  `close_grasp`, `place`, `release_placement`, `push`, `tilt` and `move_sequence` added and their help appended to
  its instructions. They run through [arm.py](arm.py), whose moves are `move_robot` calls on the same budget. Other
  differences came from the development runs:
  - The help ends with two sentences: the attempt is judged on the scene as it ends, so the model should leave
    what the task asks for in place; and a refused move, a stop on contact or a dropped object does not end it, so
    while calls remain the model should recover rather than call `done`.
  - A `move_robot` quaternion within 3% of unit length is scaled to it. `move_robot` refuses one more than 0.1% off,
    and the refusal counts toward the budget: the model's, written to three or four decimals, were 21 of the 26
    refused calls in the development runs, in both arms.
  - Old images are dropped six at a time, not one per move, so the prompt cache holds. The model can then see up to
    six more images than the loop's rules say.

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
the checkpoint the call before it issued. Then each arm, on each split:

```bash
python run.py --arm alone --split development --model MODEL --reasoning-effort high --credentials creds.json --budget-usd 50 --robowild RoboWild --output results/development --authorization "..."
```

```bash
python run.py --arm world-use --split development --model MODEL --reasoning-effort high --credentials creds.json --budget-usd 50 --agent-python agent/bin/python --robowild RoboWild --output results/development --authorization "..."
```

`creds.json` holds `{"base_url": ..., "key": ...}` for the model's Responses endpoint; `run.py` deletes it once
read and starts [model_proxy.py](model_proxy.py), which alone holds the key, forwards the agents' requests and stops
the arm's spending at `--budget-usd` (prices per model in `cases.toml`). Each task is one harness evaluation, and
the next starts only while the budget left covers the costliest task so far, so the cap does not cut an attempt
short; `run.py` names the tasks it did not run. An attempt gets 5,400 s (`--agent-timeout`). `--authorization`
records the approval of the paid run with it.
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
