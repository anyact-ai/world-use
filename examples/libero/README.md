# LIBERO-10: a model alone and with World Use

This example runs a language model as the policy of a simulated Franka Panda on
the ten LIBERO-10 tasks, once alone and once with World Use tools, on the
published [LLM_LIBERO](https://github.com/IRVLUTD/LLM_LIBERO) simulator and
agent. The model is GPT-6.2 Sol on Azure.

A run is one matched pair: both arms on one task and initial state, from the same
start. Each episode has 80 model calls, 900 simulator steps, low reasoning, the
two most recent RGB observations and a 3,072-token output cap. Both arms use the
same endpoint, controller, limits, camera conversion and LIBERO success check.
Neither gets demonstrations, object coordinates, depth, task notes or anything
learned in an earlier run. Initial images and robot state must match between the
arms.

## What each arm is given

**The raw arm** has the upstream agent's tools and its prompt, plus one sentence
that both arms get: the front camera's image is mirrored, so left and right in an
instruction are the reverse of what that image shows. Validation checks that its
approved actions and measured states are identical to upstream's and that its
prompt is upstream's plus that sentence.

**The World Use arm** has the same tools and prompt, and adds:

- `select_target` and `observe_targets`: the model marks a region in either
  camera with a point or a box and EdgeTAM follows it.
- `locate`: the position and size of a region selected in the wrist camera.
- `grasp`: open, centre over the region, descend, close, test-lift, in one call.
- `place` and `release_placement`: rise, carry, approach from above or from a
  point the model names, and pause before release; the release needs a one-use
  decision tied to the unchanged observations.
- `push`: closed fingers along a straight line until its end or until contact
  stops them, for a drawer, a door or sliding an object. The hand leans back
  from the line so that the fingertips lead.

The model chooses the regions, heights, expected finger opening and where to put
things. The tools measure and report; image overlap and finger opening do not
certify what an object is, that it is supported or that the task is done.
[`DOCS` in `world_tools.py`](world_tools.py) is everything the model is told
about them. It names no task, object or coordinate.

## How the tools work

- **Positions from the wrist camera.** A region is placed in world coordinates
  from looks at least 3 cm apart, each showing the whole region, by
  [`world_use.parallax`](../../docs/perception.md#positions-from-a-camera-on-the-hand).
  The camera sits 5 cm beside the grasp axis and its published rows are flipped;
  `WRIST` in `world_tools.py` is that mount. A position is remembered until its
  target is selected again or a grasp closes on it.
- **Arrival and contact.** A hand has arrived when it is within 12 mm of its pose
  and at rest. The last leg of a grasp, a placement or a push is a straight line,
  down by default, and contact that stops the hand on that line short of its
  pose is accepted and reported, so the model can ask for a height it cannot
  judge; the hand is then commanded to rest instead of pressing on. A hand that
  stops gaining on its command anywhere else is a surprise. These use
  [`checks.Following` and `Settling`](../../docs/adapters.md#measured-completion).
- **Fingers across the narrow side.** For a region the model outlined with a box
  in the wrist camera, clearly longer than wide and at most 70 mm across, a
  grasp given no yaw and no offset turns the fingers across it.
- **Evidence instead of gates.** A region selected in the front camera that stayed
  where it was while the hand rose returns a surprise. A lost or changed region
  is reported and concludes nothing.
- **Brief results.** The model gets the measurements that decide its next step;
  the full record of every stage goes to disk.

`native.py` is the driver for this embodiment's absolute-pose commands and
`world_tools.py` the tools. The wrist mount, the hand's dimensions, the arrival
tolerance and the motion thresholds in them were measured on this simulated arm.

Two files hold scripted simulator trials behind numbers the code and `DOCS` use.
Simulator truth scores them and never reaches a policy.

- [`centring-evidence.jsonl`](centring-evidence.jsonl): 20 grasps of wrist
  selections on development states, from over the object and from 5 cm off. Cans
  were placed within 12 mm and flat boxes within 2 mm of their true centres; z
  read 0.6 to 5.8 cm low on cans and 0.3 to 1.4 cm low on boxes; 18 were lifted.
- [`region-evidence.jsonl`](region-evidence.jsonl): the lifts behind the
  thresholds that tell a region left behind from one that rose with the hand.

## Limits

- Nothing checks what a selected region is.
- Only a push leans the hand. Grasps come from above, and a rim offset takes
  mugs, bowls and pots. A mug held that way does not go under an oven's roof. In
  a scripted trial on a development state the hand met the frame with the mug
  half in; with the hand leaning 50 degrees, given as raw rotation targets, the
  mug went 3 cm further and fell back out when released.
- The World Use arm is told the wrist camera's mount and the hand's dimensions.
  The raw arm is not.

## Results

GPT-6.2 Sol alone and with World Use at `ee263a4`, one matched pair per case:
the twenty development cases and thirty held-out cases, states 4, 5 and 6 of
every task, run once after the tools were finished.

| | pairs | Sol alone solved | with World Use | calls alone / WU | est. USD alone / WU | steps alone / WU |
|---|---|---|---|---|---|---|
| development states 0 and 1 | 20 | 7 | 13 | 1,224 / 337 | 23.63 / 3.51 | 13,182 / 12,140 |
| held-out states 4, 5 and 6 | 30 | 8 | 17 | 1,871 / 478 | 37.81 / 5.22 | 18,831 / 18,503 |
| all | 50 | 15 | 30 | 3,095 / 815 | 61.44 / 8.73 | 32,013 / 30,643 |

The median episode took 71.5 model calls alone and 13 with World Use, and 27.3
million input tokens became 6.2 million. On the 12 pairs both arms solved, World
Use made 67% fewer calls, sent 65% fewer input tokens and took 34% more
simulator steps. World Use alone solved 18 pairs and Sol alone 3. Of the
episodes that failed, Sol alone gave up in 23, claimed done in 6 and ran out of
steps in 6; World Use claimed done in 11, gave up in 8 and ran out of steps in 1.

| task | Sol alone | with World Use | held-out, alone / WU | mean calls, alone / WU |
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

Where World Use fell short:

- Eleven of its twenty failures were a `done` the success check rejected: the
  book in the caddy three times, the bowl and drawer three times, the moka
  pots twice, the soup and sauce twice, and the mug on its plate once. Nothing
  here checks what an object is or how it rests; that judgement is the model's.
- No episode of either arm put the mug in the microwave. Held from above by its
  rim, it does not pass under the oven's roof (see Limits).
- On pairs both arms solved it took a third more simulator steps.

What these results do not show:

- Each case was run once. Two World Use runs of the development cases, with
  tools that differed only in how a push leans the hand, solved 14 and 13 of
  20, and five cases came out differently between them. A per-task difference
  of one or two episodes is within that spread.
- The development states are the ones the tools were adjusted on. The held-out
  states had never been run and their records were not read to change anything.
- Three attempts stopped on a dropped connection or a lost simulator and were
  run again; a pair counts only when both arms finished in one run.

[paid-results.jsonl](paid-results.jsonl) has one row per arm of every pair, with
how the episode ended, calls, counted steps, wall time, tokens and estimated
cost; `report.py tables` prints the per-case table and these totals from it.
The evaluation's 53 attempts cost an estimated $73.54, $101.19 at the
conservative rate the transport reserves. An evaluation of GPT-6.1 Sol with
earlier tools is recorded at commit `02a3ba1`.

## Rebuild the runtime

Use Linux with `uv`, Git, CMake 3.x, and working headless EGL software rendering.
Keep the simulator on Python 3.12 and World Use on Python 3.13. The two processes
communicate through the upstream worker protocol; their MuJoCo versions are
independent. The simulator dependencies and agent dependencies are frozen in the
adjacent requirement files. The runtime used for a paid run is recorded and must
match the runtime that passed validation.

Pinned sources:

| Component | Revision |
|---|---|
| World Use library | this tree |
| IRVLUTD/LLM_LIBERO | `7db834afc6c202abf5fd93668e65141630408551` |
| robocurve/inspect-robots | `4d35fe4b81a643c0e61e8d287810b3c26f9d386a` |
| lerobot/libero-assets (HF dataset) | `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` |
| facebook/EdgeTAM | `f3a09791b2343c0733d456d08d73771d9363b69a` |

Create the two virtual environments, clone/check out those source revisions, and
install the requirement files using PyPI plus the official CPU PyTorch index:

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

Install CMake in the simulator environment first if the host has none:
`uv pip install --python /workspace/libero-runtime/sim-env/bin/python 'cmake>=3.25,<4'`.
Install the checked-out Inspect Robots core and its `plugins/inspect-robots-agent`,
LLM_LIBERO, and this World Use library in the agent environment with `uv pip install
--no-deps --python ... <source paths>`. Initialize LIBERO's config by importing
`libero.libero` once in the simulator interpreter and answering `N`, with
`LIBERO_CONFIG_PATH` pointing to a new runtime-owned directory.

Download the pinned `lerobot/libero-assets` snapshot with
`huggingface_hub.snapshot_download(repo_type='dataset', revision=...)`, and link the
snapshot as `libero/libero/assets` inside the simulator installation. Download the
pinned EdgeTAM snapshot into `HF_HOME` before enabling offline mode. Both runtime
installs must preserve the inherited proxy and CA settings.

If the CPU PyTorch index is unreachable, the same `torch` and `torchvision`
versions from PyPI run on CPU; the runtime manifest then records the extra CUDA
packages. If Xet downloads are blocked, set `HF_HUB_DISABLE_XET=1` for the snapshot
downloads. EdgeTAM also loads `config.json` from `timm/repvit_m1.dist_in1k`
(revision `55e64695f26bb1711730f916e557b464fb46478d` when last rebuilt), which is
not part of its pinned snapshot: load the tracker once online before enabling
offline mode.

The explicit preparation above documents the reconstruction; it has not been
packaged as a separately tested clean-machine installer. The validation below
tests the actual installed runtime and refuses a failed or stale validation.

## Validate before paying

Set these variables for both validation and the pilot, adapting runtime paths:

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

Validation runs the runner's regressions, compares the rebuilt baseline's approved
actions and measured states against upstream `evaluate`, and exercises actual
EdgeTAM and LIBERO with synthetic policy responses. It verifies identical initial
observations and repeated-capture tracking. An empty grasp must stop before lift
with the hand reported fully closed, a reversed move must report the offset of its
last applied command, and one selected region must be flagged when another object
is lifted and pass when its own object is. From hovers 5 cm off, `locate` and a
wrist-selected `grasp` must each place a can within 1.5 cm of its fixture
position, and that grasp must hold it. A push through free air must run its whole
line with the hand leaning and come back upright. The raw arm's prompt must be upstream's
plus the one sentence both arms are given. No model requests leave the mock
transport, and no task success is claimed. Pixel and pose fixtures exist only in
`validate.py`. `PYTHONPATH` must name the `src` folder of the World Use tree under
test when the library is not installed in the agent environment.

## Cases

[`cases.toml`](cases.toml) lists every task and initial state the runner accepts,
each with a role. Development cases, states 0 and 1 of every task, are for
iteration. Held-out cases, states 4 to 6, run only with `--held-out`, as a test of
finished tools, and their records are never inspected to tune them. Once a
held-out state has tested one version of the tools it is spent; a later version
needs states never run, and LIBERO has 50 per task. The runner refuses an
unlisted case, a held-out case without `--held-out` and a development case with
it.

## Run the authorized pilot

Supply `FOUNDRY_API_KEY` through the environment's credential settings. Set
`LIBERO_AZURE_BASE_URL` to the approved Azure resource's `/openai/v1` endpoint.
Do not store credentials in this repository or pass them on a command line.

One pair:

```sh
/workspace/libero-runtime/agent-env/bin/python examples/libero/run.py \
  --validation /workspace/libero-validation/validation.json \
  --output /workspace/libero-paid-pilot \
  --task 0 --state 0 \
  --authorization "<the user's own words approving this scope>" \
  --execute-paid-rollouts
```

Several pairs under one campaign cap, here state 0 of the basket tasks twice,
three pairs at a time:

```sh
/workspace/libero-runtime/agent-env/bin/python examples/libero/campaign.py \
  /workspace/libero-paid-campaign 0:0 1:0 7:0 --repeats 2 \
  --cap 45 --pair-cap 7 --arm-cap 3.4 --workers 3 \
  --validation /workspace/libero-validation/validation.json \
  --authorization "<the user's own words approving this scope>"
```

`--role development` or `--role held_out` selects every case of that role
instead. Any held-out case also needs `--held-out-authorization`, the user's
words approving that test. `--arms world_use` runs only the World Use arm, for a
development check; `protocol.json` records the arms run.

The campaign admits a pair only while the conservative charge of its finished
attempts plus the full caps of running pairs leaves room for one more pair cap.
A pair that did not finish both arms, stopped by the transport, a lost simulator
or an interruption, keeps its folder and its charge and is tried again in a
`_retryN` folder, three attempts in all. Ctrl-C, or any signal that stops a
pair, stops the campaign instead: it starts nothing more and waits for the
running pairs to end. Each pair holds a simulator of about 3 GB, so choose
`--workers` to fit the machine's memory. Pairs found complete in the folder are skipped, and each repeat
after the first has its own `_repeatN` folder.

Append the campaign's pairs, labelled with the commit they ran, to the compact
rows, and print the comparison tables. The tables use each case's latest commit
for each arm, report repeats as solved k of n, and total every episode of those
runs by role, overall and by task. A pair in which the transport stopped a
request has one row, for the arm that was running, and no result for either arm:

```sh
python examples/libero/report.py collect examples/libero/paid-results.jsonl \
  <commit>=/workspace/libero-paid-campaign
python examples/libero/report.py tables examples/libero/paid-results.jsonl
```

The authorization is recorded verbatim in `protocol.json`. An approval given for
an earlier runner or scope does not cover this one. A paid run needs a
`validation.json` made by `validate.py` in its own runtime, for exactly these
sources and this library tree.

A small live image and function-call probe precedes the episodes. A pair has a $7
conservative usage cap, $3.40 per arm, a 6,000-second deadline and 3,000 seconds
per episode unless `--total-budget`, `--arm-budget`, `--deadline-seconds` and
`--episode-seconds` set others; all are recorded in `protocol.json`. Output
directories must be new.

Each request is reserved before it is sent, at half a token per byte of its text,
an eighth per byte of the sealed reasoning the model gets back, and 8,192 per
image, and refused above 200,000 tokens; that is at least five times what the
service counted in every recorded request. A raw arm that uses
all 80 calls needs about $2.70 of conservative allowance. A request whose
connection was never made is sent again up to three times, since nothing was
sent. Any other failure is not retried: its usage is unknown, its reservation
stays charged and the pair stops. A response that used more than its
reservation is charged in full and stops the whole campaign, since the size
estimate behind every reservation was wrong. All requests, responses and settled usage are
journalled without authentication headers. Prices are the model's published
rates, not an Azure billing reconciliation, and local compute is excluded.

Each arm saves raw observation PNGs, native action and state records, simulator
video, the policy transcript and the result. World Use also saves tracking and
procedure feedback. Keep the result folders somewhere durable: the compact rows
are a summary, not the record.
