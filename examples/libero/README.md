# LIBERO-10: a model alone and with world-use

GPT-6.2 Sol, on Azure, is the policy of a simulated Franka Panda on the ten LIBERO-10 tasks, using the
published [LLM_LIBERO](https://github.com/IRVLUTD/LLM_LIBERO) simulator and agent. Each case, a task and
an initial state, is run as a matched pair: once with the upstream agent's tools alone and once with World
Use tools added. Over 50 pairs, 30 of them on held-out initial states, the model solved 15 episodes alone
and 30 with world-use, and made 74% fewer model calls (815 against 3,095). The evaluation ran at commit
`ee263a4`; [paid-results.jsonl](paid-results.jsonl) has every episode.

## Protocol

Both arms of a pair start from the same initial state; the runner stops a pair whose arms do not reset
to identical images and robot state. They use the same endpoint, controller, action limits, camera
conversion and LIBERO success check. An episode has at most 80 model calls and 900 simulator steps. The
model runs with low reasoning effort and a 3,072-token output cap, and sees the two most recent RGB
observations. Neither arm gets demonstrations, object coordinates, depth, task notes or anything learned
in an earlier run.

The 50 cases are five initial states of every task:

- **Development**, states 0 and 1: the tools were adjusted on these.
- **Held-out**, states 4, 5 and 6: never run before this evaluation, run once after the tools were
  finished, and their records were not read to change anything (see [Cases](#cases) for the rule).

## What each arm gets

**Alone** (`raw` in the results, `baseline` in the runner): the upstream agent's tools and prompt, plus
one sentence that both arms get: the front camera's image is mirrored, so left and right in an
instruction are the reverse of what that image shows. [Validation](#validate-before-paying) checks that
this arm acts as upstream's does and is told nothing else.

**With world-use**: the same tools and prompt, plus

- `select_target` and `observe_targets`: the model marks a region in either camera with a point or a
  box, and EdgeTAM follows it.
- `locate`: moves the hand over a region selected in the wrist camera and returns its position and size.
- `grasp`: open, centre over the region, descend or come in leaning along the hand's own axis, close and
  test-lift, in one call.
- `place` and `release_placement`: rise, carry, approach from above or from a point the model names, and
  stop before release. The release needs a one-use checkpoint, void after any motion or reselection.
- `push`: closed fingers along a straight line until its end or until contact stops them, for a drawer,
  a door or sliding an object. The hand leans back from the line so that the fingertips lead.

The model chooses the regions, heights, expected finger opening and where to put things; the tools move
the arm and report measurements. [`DOCS` in `world_tools.py`](world_tools.py) is everything the model is
told about them. It names no task, object or coordinate, but it does give the wrist camera's mount, the
hand's dimensions and which way the wrist can lean, which the raw arm is not told.

## Results

| cases | pairs | solved alone | solved with world-use | calls alone / WU | est. USD alone / WU | steps alone / WU |
|---|---|---|---|---|---|---|
| development, states 0 and 1 | 20 | 7 | 13 | 1,224 / 337 | 23.63 / 3.51 | 13,182 / 12,140 |
| held-out, states 4, 5 and 6 | 30 | 8 | 17 | 1,871 / 478 | 37.81 / 5.22 | 18,831 / 18,503 |
| all | 50 | 15 | 30 | 3,095 / 815 | 61.44 / 8.73 | 32,013 / 30,643 |

Of the 50 pairs, both arms solved 12, only world-use 18, only the model alone 3 and neither 17. On the 12
that both solved, world-use made 67% fewer calls, sent 65% fewer input tokens and took 34% more simulator
steps. Over all pairs, the median episode took 71.5 model calls alone and 13 with world-use, and 27.3
million input tokens became 6.2 million.

| how a failed episode ended | alone | with world-use |
|---|---|---|
| `done`, rejected by the success check | 6 | 11 |
| `give_up` called by the model | 5 | 8 |
| the 80-call limit, recorded as `give_up` | 18 | 0 |
| the 900-step limit | 6 | 1 |
| failed episodes | 35 | 20 |

| task | solved alone | with world-use | held-out, alone / WU | mean calls, alone / WU |
|---|---|---|---|---|
| 0: soup and tomato sauce into the basket | 1 of 5 | 3 of 5 | 0 / 2 of 3 | 74 / 12 |
| 1: cream cheese and butter into the basket | 5 of 5 | 5 of 5 | 3 / 3 of 3 | 41 / 14 |
| 2: turn on the stove, moka pot onto it | 1 of 5 | 3 of 5 | 1 / 1 of 3 | 66 / 26 |
| 3: bowl into the bottom drawer, close it | 0 of 5 | 1 of 5 | 0 / 1 of 3 | 80 / 14 |
| 4: two mugs onto their plates | 1 of 5 | 5 of 5 | 0 / 3 of 3 | 57 / 10 |
| 5: book into the caddy | 1 of 5 | 2 of 5 | 1 / 0 of 3 | 43 / 9 |
| 6: mug onto the plate, pudding right of it | 1 of 5 | 4 of 5 | 1 / 3 of 3 | 48 / 11 |
| 7: soup and cream cheese into the basket | 4 of 5 | 5 of 5 | 2 / 3 of 3 | 62 / 12 |
| 8: both moka pots onto the stove | 1 of 5 | 2 of 5 | 0 / 1 of 3 | 68 / 27 |
| 9: mug into the microwave, close it | 0 of 5 | 0 of 5 | 0 / 0 of 3 | 79 / 26 |

Where world-use fell short:

- Eleven of its twenty failures were a `done` the success check rejected: the book in the caddy three
  times, the bowl and drawer three times, the moka pots twice, the soup and sauce twice and the mug on its
  plate once.
- No episode of either arm put the mug in the microwave. Held from above by its rim, it does not pass
  under the oven's roof (see [Limits](#limits)).

[paid-results.jsonl](paid-results.jsonl) has one row per arm of every pair, with how the episode ended,
calls, counted steps, wall time, tokens and estimated cost; `report.py tables` prints the per-case table
and these totals from it. Costs are estimated from the model's published prices, not an Azure bill, and
exclude local compute. The evaluation's 53 attempts, including three that were run again and one test
request before each, cost an estimated $73.54. The campaign's spending guard, which reserves every request
at a conservative rate, charged $101.19; its ledger is not committed. An evaluation of GPT-6.1 Sol with
earlier tools is recorded at commit `02a3ba1`.

## Limits

- **One run per case.** Two world-use runs of the development cases, with tools that differed only in how
  a push leans the hand, solved 14 and 13 of 20, and five cases came out differently between them. A
  per-task difference of one or two episodes is within that spread.
- **Retried attempts.** Three attempts, of two cases, stopped when a request failed on a dropped
  connection, and the pairs were run again. A pair counts only when both arms finished in one run.
- **The model judges the outcome.** No tool checks what a selected region is, how an object rests or
  whether the task is done; the model decides that from the images. Most world-use failures came from
  that judgement.
- **The evaluated tools grasped only from above.** A mug held by its rim does not go under an oven's
  roof: in a scripted trial on a development state the hand met the frame with the mug half in. LIBERO's
  own 50 demonstrations of that task take the mug by its handle, with the hand leaning 33 to 70 degrees.
  `grasp` now leans the hand with `tilt_deg`. In scripted trials on the two development states, with
  simulator truth standing in for the model's estimates, the tools then put the mug in the oven upright and
  pushed the door shut, and the benchmark's check passed. No model has been evaluated with it.
- **This tree is newer than `ee263a4`.** Since the evaluation, `grasp` takes `tilt_deg` and `toward`, and
  offsets up to 20 cm; `locate` says which way its image lies in the world; an upright hand's fingers are
  turned the way that keeps the wrist joint off its stop, so a `yaw_deg` of 90 now puts them along -x; a
  push leans the hand only on a line more level than steep, stays upright toward +x and backs off along its
  line also after failing on the way; `place` checks `object_target` before the hand moves; and `DOCS`
  gives the height error on flat things as 1.5 cm instead of 3 and says a push backs off up to 10 cm. To
  rerun the evaluated version, fetch it with `git fetch origin pull/23/head` and check out `ee263a4`.

## How the tools work

`native.py` drives this embodiment's absolute-pose commands and `world_tools.py` holds the tools. The
wrist mount, the hand's dimensions, the arrival tolerance and the motion thresholds in them were measured
on this simulated arm.

- **Positions from the wrist camera.** `locate` and a grasp of a wrist selection move the hand level and
  place the region in world coordinates from looks at least 3 cm apart, each showing the whole region,
  with [`world_use.parallax`](../../docs/perception.md#positions-from-a-camera-on-the-hand). The camera
  sits 5 cm beside the grasp axis and its published rows are flipped; `WRIST` in `world_tools.py` is that
  mount. A position is kept until its target is selected again or a grasp closes on it.
- **Arrival and contact.** A hand has arrived when it is within 12 mm of its pose and at rest. The last leg
  of a grasp, a placement or a push is a straight line, down by default. Contact that stops the hand on
  that line short of its pose is accepted and reported, so the model can ask for a height it cannot
  judge, and the hand is then commanded to rest instead of pressing on. A hand that stops gaining on its
  command anywhere else is a `surprise`. Both tests are
  [`checks.Following` and `Settling`](../../docs/adapters.md#measured-completion).
- **Fingers across the narrow side.** For a region the model outlined with a box in the wrist camera,
  clearly longer than wide and at most 70 mm across, a grasp given no yaw and no offset turns the fingers
  across it.
- **Leaning grasps.** With `tilt_deg` the hand turns where it is, its fingers first and then its lean,
  goes to a point 10 cm back along its own axis and comes in along that axis. It leans toward a point the
  model names or, given an offset from a wrist region, toward the region's centre, which is how it comes at
  a handle from outside. The hand keeps the side its camera is on underneath, and a lean toward +x, away
  from the robot's base, is refused: the wrist does not bend that way.
- **The wrist's range.** Fingers along a direction and along its opposite close on the same thing, half a
  turn of the wrist joint apart. An upright hand takes the one nearer its present finger axis unless that
  is more than 110 degrees from the middle of the joint's range, and a turn of more than 120 degrees is
  made in two moves, since one move across a half turn can run the joint into its stop.
- **Front-camera check.** A region selected in the front camera is compared before a grasp's approach
  and after its test lift, and across a placement whose hand travel is vertical to within 2 cm. If it
  stayed where it was while the hand moved, the tool returns a `surprise` and `place` withholds its
  release checkpoint. A region that moved, was hidden or was lost is reported and stops nothing.
- **Short results.** The model gets the measurements that decide its next step; the full record of every
  stage goes to disk.

Three files hold the scripted simulator trials behind numbers in the code and in `DOCS`. They are scored
with simulator ground truth, which no policy sees.

- [`centring-evidence.jsonl`](centring-evidence.jsonl): 20 grasps of wrist selections on development
  states, from over the object and from 5 cm off. Cans were placed within 12 mm and flat boxes within 2 mm
  of their true centres; z read 0.6 to 5.8 cm low on cans and 0.3 to 1.4 cm low on boxes; 18 were lifted.
- [`region-evidence.jsonl`](region-evidence.jsonl): the lifts behind the thresholds that tell a region
  left behind from one that rose with the hand.
- [`lean-evidence.jsonl`](lean-evidence.jsonl): the trials behind the leaning grasp, on development
  states. How the 50 demonstrations of the mug-into-microwave task hold the mug; the wrist joint's angle
  in 15 upright and 37 leaning poses, behind the finger directions the tools choose and the refusal to lean
  toward +x; 12 grasps of that mug's handle with `tilt_deg` 45, of which the 11 with offsets of 5 to 8 cm
  and grip heights of 5.5 to 8.5 cm lifted it and the one 4 cm from the region's centre closed on its body;
  6 of two other mugs and a moka pot by their handles, in both directions the wrist leans sideways, all
  lifted; 4 upright rim grasps under the new finger rule; and the mug put in the oven upright and the door
  pushed shut on both states, in 609 and 603 of the 900 steps.

## Rebuild the runtime

Use Linux with `uv`, Git, CMake 3.x and working headless EGL software rendering. The simulator runs on
Python 3.12 and world-use on Python 3.13, in separate virtual environments; the two processes communicate
through the upstream worker protocol, so their MuJoCo versions are independent. The adjacent requirement
files freeze each environment's dependencies.

| Component | Revision |
|---|---|
| world-use library | this tree |
| IRVLUTD/LLM_LIBERO | `7db834afc6c202abf5fd93668e65141630408551` |
| robocurve/inspect-robots | `4d35fe4b81a643c0e61e8d287810b3c26f9d386a` |
| lerobot/libero-assets (HF dataset) | `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` |
| facebook/EdgeTAM | `f3a09791b2343c0733d456d08d73771d9363b69a` |

1. Clone the sources at those revisions. Create the two environments and install the requirement files
   from PyPI plus the official CPU PyTorch index:

   ```sh
   uv venv --python 3.12 /workspace/libero-runtime/sim-env
   uv venv --python 3.13 /workspace/libero-runtime/agent-env
   PATH=/workspace/libero-runtime/sim-env/bin:$PATH uv pip install \
     --python /workspace/libero-runtime/sim-env/bin/python \
     -r examples/libero/sim-requirements.txt \
     --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
   uv pip install --python /workspace/libero-runtime/agent-env/bin/python \
     -r examples/libero/agent-requirements.txt \
     --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
   ```

   If the host has no CMake, install it in the simulator environment first:
   `uv pip install --python /workspace/libero-runtime/sim-env/bin/python 'cmake>=3.25,<4'`.
2. Install the checked-out Inspect Robots core and its `plugins/inspect-robots-agent`, LLM_LIBERO and this
   world-use library in the agent environment with
   `uv pip install --no-deps --python ... <source paths>`.
3. Point `LIBERO_CONFIG_PATH` at a new directory, then initialize LIBERO's config there by importing
   `libero.libero` once in the simulator interpreter and answering `N`.
4. Download the pinned `lerobot/libero-assets` snapshot with
   `huggingface_hub.snapshot_download(repo_type='dataset', revision=...)` and link it as
   `libero/libero/assets` inside the simulator installation.
5. Download the pinned EdgeTAM snapshot into `HF_HOME`. EdgeTAM also loads `config.json` from
   `timm/repvit_m1.dist_in1k` (revision `55e64695f26bb1711730f916e557b464fb46478d` when last rebuilt),
   which its snapshot does not include, so load the tracker once online before enabling offline mode.

Keep the host's proxy and CA settings in the environment for both installs. If the CPU PyTorch index is
unreachable, the same `torch` and `torchvision` versions from PyPI run on CPU; the runtime manifest then
records the extra CUDA packages. If Xet downloads are blocked, set `HF_HUB_DISABLE_XET=1` for the
snapshot downloads.

These steps have not been tested as a script on a clean machine; validation checks the runtime they
produce.

## Validate before paying

Set these variables for both validation and paid runs, adapting the runtime paths:

```sh
export LIBERO_PYTHON=/workspace/libero-runtime/sim-env/bin/python
export LIBERO_CONFIG_PATH=/workspace/libero-runtime/libero-config
export HF_HOME=/workspace/libero-runtime/huggingface
export HF_HUB_OFFLINE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
/workspace/libero-runtime/agent-env/bin/python examples/libero/validate.py \
  --output /workspace/libero-validation
```

`PYTHONPATH` must name the `src` folder of the world-use tree under test when the library is not
installed in the agent environment. Validation makes no model requests: a scripted policy answers through
a mock transport, and no task success is claimed. It writes `validation.json`, and a paid run refuses one
that failed or was made for other runner sources, another world-use tree or another runtime. It requires:

- `test_campaign.py`, the tests of the spending guard, case roles and campaign, to pass;
- the raw arm's approved actions and states to match upstream `evaluate` on the same scripted responses,
  and its prompt to be upstream's plus the one sentence both arms get;
- both arms to reset to identical initial observations, and a selection and a later observation of the
  same capture to give the same regions;
- on real LIBERO and EdgeTAM, the world-use tools to pass these checks: an empty grasp stops before the
  lift with the hand reported fully closed; a reversed move reports the offset of its last applied
  command; a selected region is flagged when another object is lifted and passes when its own object is;
  from hovers 5 cm off, `locate` and a wrist-selected `grasp` each place a can within 1.5 cm of its
  fixture position, and that grasp holds it; a grasp with `tilt_deg` turns the hand where it is and ends
  at its point leaning 45 degrees the way it was told; a push through free air runs its whole line with
  the hand leaning and comes back upright.

The pixel and pose fixtures of the scripted policy exist only in `validate.py`.

## Cases

[`cases.toml`](cases.toml) lists every task and initial state the runner accepts, each with a role.
Development cases, states 0 and 1, are for iteration. Held-out cases, states 4 to 6, run only with
`--held-out`, as a test of finished tools; their records are never inspected to tune them. Once a
held-out state has tested one version of the tools it is spent: a later version needs states never run,
and LIBERO has 50 per task. States 2 and 3 tested earlier tools and are not listed. The runner refuses an
unlisted case, a held-out case without `--held-out` and a development case with it.

## Run paid pairs

Supply `FOUNDRY_API_KEY` through the environment's credential settings, and set `LIBERO_AZURE_BASE_URL`
to the approved Azure resource's `/openai/v1` endpoint. Do not store credentials in this repository or
pass them on a command line.

One pair:

```sh
/workspace/libero-runtime/agent-env/bin/python examples/libero/run.py \
  --validation /workspace/libero-validation/validation.json \
  --output /workspace/libero-paid-pilot \
  --task 0 --state 0 \
  --authorization "<the user's own words approving this scope>" \
  --execute-paid-rollouts
```

Several pairs under one campaign cap, here state 0 of the basket tasks twice, three pairs at a time:

```sh
/workspace/libero-runtime/agent-env/bin/python examples/libero/campaign.py \
  /workspace/libero-paid-campaign 0:0 1:0 7:0 --repeats 2 \
  --cap 45 --pair-cap 7 --arm-cap 3.4 --workers 3 \
  --validation /workspace/libero-validation/validation.json \
  --authorization "<the user's own words approving this scope>"
```

`--role development` or `--role held_out` selects every case of that role instead. Any held-out case also
needs `--held-out-authorization`, the user's words approving that test. `--arms world_use` runs only the
world-use arm, for a development check; `protocol.json` records the arms run.

The authorization is recorded verbatim in `protocol.json`. An approval given for an earlier runner or
scope does not cover this one. Output directories must be new.

The campaign admits a pair only while the conservative charge of its finished attempts plus the full caps
of running pairs leaves room for one more pair cap. A pair that did not finish both arms, stopped by the
transport, a lost simulator or an interruption, keeps its folder and its charge and is tried again in a
`_retryN` folder, three attempts in all. Ctrl-C, or any signal that stops a pair, stops the campaign
instead: it starts nothing more and waits for the running pairs to end. Pairs found complete in the folder
are skipped, and each repeat after the first has its own `_repeatN` folder. Each pair holds a simulator of
about 3 GB, so choose `--workers` to fit the machine's memory.

### Spending limits

A small live image and function-call probe precedes the episodes. A pair has a $7 conservative usage cap,
$3.40 per arm, a 6,000-second deadline and 3,000 seconds per episode unless `--total-budget`,
`--arm-budget`, `--deadline-seconds` and `--episode-seconds` set others; all are recorded in
`protocol.json`.

The conservative charge prices every input token at the dearest input rate, the cache write's. Each
request is reserved before it is sent, with the full output cap and an input estimate of half a token per
byte of its text, an eighth per byte of the sealed reasoning the model gets back and 8,192 per image. That
estimate is at least five times what the service counted in every recorded request, and a request
estimated above 200,000 tokens is refused. Once the response arrives, the reservation is replaced by the
charge for the tokens the service counted. A raw arm that uses all 80 calls needs about $2.70 of
conservative allowance.

A request whose connection was never made is sent again up to three times, since nothing was sent. Any
other failure is not retried: its usage is unknown, its reservation stays charged and the pair stops. A
response that used more than its reservation is charged in full, stops the pair and keeps the campaign
from admitting any more, since the size estimate behind every reservation was wrong. All requests,
responses and settled usage are journalled without authentication headers.

### Results and records

Append the campaign's pairs, labelled with the commit they ran, to the compact rows, and print the
comparison tables. The tables use each case's latest commit for each arm, report repeats as solved k of
n, and total every episode of those runs by role, overall and by task. A pair in which the transport
stopped a request has one row, for the arm that was running, and no result for either arm:

```sh
python examples/libero/report.py collect examples/libero/paid-results.jsonl \
  <commit>=/workspace/libero-paid-campaign
python examples/libero/report.py tables examples/libero/paid-results.jsonl
```

Each arm saves raw observation PNGs, native action and state records, simulator video, the policy
transcript and the result. world-use also saves tracking and procedure feedback. Keep the result folders
somewhere durable: the compact rows are a summary, not the record.
