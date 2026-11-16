# Matched LIBERO development pilot

This runner compares GPT-6.1 Sol alone with Sol using World Use PR #22, on the
published [LLM_LIBERO](https://github.com/IRVLUTD/LLM_LIBERO) simulator and policy.
It was reconstructed after the earlier, separately stored benchmark adapter was
lost with a workspace reset. It is **not an exact restoration of that adapter**.
Run a fresh baseline alongside World Use; do not compare its costs directly with
earlier runs or infer a benchmark success rate from one pair.

Each invocation runs one matched pair: a LIBERO-10 task and initial state (task 0,
state 0 unless `--task` and `--state` say otherwise), low reasoning, 80 calls, 900
simulator steps, two recent RGB observations, and a 3,072-token output cap. Both arms use the
same model endpoint, native controller, limits, camera conversion and LIBERO
success predicate. No demonstrations, object coordinates, depth, task notes or
prior-run learning are provided. Initial RGB and robot-state hashes must match.
Initial state 2 is reserved held-out and refused.

The baseline uses the upstream agent's prompt and tools. The World Use arm adds
RGB target selection/tracking, compact measured feedback and the library's grasp
and place procedures. Sol chooses the regions, poses, expected aperture and
placement interpretation. A placement pauses before release and needs a one-use
decision tied to the unchanged observations. Image overlap and finger opening do
not certify object identity, support or task completion.

## Conditions

Three conditions must stay distinguishable in any comparison:

- **Raw Sol**: the baseline arm. This tree does not change what it executes or is
  told; validation compares its approved actions and measured states with upstream.
  Neither arm receives the upstream rollout's `approver: N step(s) modified`
  observation line. Restoring it would be a new raw condition.
- **Frozen World Use**: commit `ffaaaf6`, whose `native.py`, `world_tools.py` and
  `run.py` have SHA-256 prefixes `ad8b8fab`, `a21a23f8` and `90a9cfc5`. Evidence
  produced with it belongs to that source.
- **Candidate World Use**: this tree, recorded as `world-use-candidate-20261116`.
  It has no paid result. Its differences from the frozen runner are the five
  below, introduced by separate commits so that an earlier commit can be run as a
  narrower condition: settling with applied commands, then the region comparison,
  then the briefer results and merged selection, then the contact-terminated grasp.

**Settling.** Each leg of a procedure stage repeats its final native pose for at
most 20 more steps, all counted toward the 900-step limit. It uses the library's `Settling`
check and returns as soon as the measurements settle the question. The arm has
`arrived` after three consecutive samples within 12 mm and 0.12 rotation-column
error; it is `not_approaching` once that offset stops shrinking over four
intervals; otherwise it is `still_converging` at the bound. The fingers have
`reached_end` within 3 mm of fully closed or open, have
`stopped_apart` once they travel less than 0.3 mm over four intervals, and are
otherwise `still_moving`. A stable opening outside the policy's expected range is
a measured surprise, not a failure to converge. No wait concludes while an approver
is still changing what is applied, and an opening hand is not judged until its
command ramp has finished: the fingers do not move until the ramp passes their
position. The frozen runner judged stability from single-step aperture changes
inside the expected range, so an empty or unexpectedly wide grasp used the whole
bound and reported non-convergence.

**Applied commands.** The native per-step limit is measured from the previous
applied command, while each move interpolates from the measured pose. A move issued
while the arm still trails its last command can therefore end short of its request.
World Use results now carry `applied` whenever the approvers changed a command,
with the distance from the last applied command to the request. Raw moves retain
native interpolation and are not extended.

**Region comparison.** When the object was selected in `agentview`, its tracked
region is compared across the test lift and across a vertical approach to a release
pose, using the existing procedure assessment hooks. For a grasp the region seen
before the approach is compared with the region after the lift. The hand must have
travelled at least 1.5 cm vertically and at most 2 cm sideways over the lift or
the place move.
A region that kept at least 0.9 mask overlap and shifted less than 0.3 image
heights per metre of that travel did not move with the hand: `grasp` returns a
surprise after the lift, and `place` issues no release receipt. The hand then
holds something else or nothing, or the selection sits on another object. A region
that changed concludes nothing and leaves the assessment unknown, and a wrist
selection is not compared. The rule
assumes a static camera that does not look along the vertical. A straight-down
view only rescales a lifted object, so such a camera must not be compared.

These thresholds were chosen on development evidence, not held-out data: 43
scripted real-simulator, real-EdgeTAM episodes over tasks 0, 1, 4, 6 and 7 and
initial states 0 and 1, with simulator ground truth. Every lift there was a 4 cm
request measuring 2.8 to 3.5 cm, so shorter travel rests on the scaling above, not
on separate measurements. The place approach was exercised in two further scripted
episodes only: a held object received its receipt, and a region left behind
across a 5.4 cm descent did not. All 9 selected regions whose
object stayed behind were flagged and none of the 9 that were lifted. Over every
tracked `agentview` region, 41 of 50 that stayed were unchanged; the other 9 had
drifted onto the passing hand or changed through occlusion, which the rule leaves
unflagged. One of 20 lifted objects was flagged: its point prompt had landed on
the carton occluding it, so the tracked region was not the object.

The same rule was run once on the library's own MuJoCo block scene, with a
different arm, object and renderer. Two oblique views of a gripped, lifted block
changed (overlap 0.27 and 0.53, shift 12 px). Of two oblique views of a block left
behind, one was unchanged and one changed through hand occlusion. The overhead
view kept 0.86 overlap with under 1 px of shift for a 3.7 cm lift, which is why it
is excluded. [region-evidence.jsonl](region-evidence.jsonl) holds every row of
both checks; all of it is simulated.

**A grasp that does not need the height.** Given the true object centre and a
height estimated from its mask, the earlier `grasp` lifted 3 of 18 objects over
all ten tasks: nine approaches were blocked on the way down, two grasps stopped
only because the hand had hidden the tracked region, and two closed on nothing.
`grasp` and `place` now move level to above the pose and descend straight down.
The requested z is the lowest height allowed: a descent that contact stops
directly above the pose is accepted, the hand closes there, and the test lift
rises from where it stopped. The selected region is looked at before the approach
and after the lift, not while the hand hides it, and a lost region is reported
instead of stopping the procedure. A region that moved no longer counts as a
passed check; only one that stayed behind changes the result. Asking for 5 mm
above the table at the same 18 true centres, the grasp lifted 10: all six can
attempts and all four flat-box attempts. The four mugs, two moka pots, the bowl
and the book still failed, because a centre grasp from above is the wrong grasp
for them. The policy text now says to take hollow objects by the rim just below
its top; two scripted rim grasps of a mug at its true centre lifted it, while ten
rim attempts placed from the footprint centre with a low z all failed, so that
grasp needs the rim located well. A successful grasp took 52 to 61 counted steps
against 42 to 44 before.

Tracking is now evidence and no longer a precondition. A scripted run of task 0
with true positions placed the first can, then had its second `place` refused
because the tracker had lost the held can during the carry. With an unselected or
lost region reported instead, the same script satisfied LIBERO's success check
after 11 tool calls and 367 steps: two grasps that select in the call, two
placements, one release and six moves. [grasp-evidence.jsonl](grasp-evidence.jsonl)
holds every attempt. These are scripted, privileged diagnostics of the tools, not
of a model using them.

**Briefer results and fewer calls.** Each World Use turn used to add about five
times the text of a raw turn, and every later request carries it again. The policy
now receives a brief copy of each result: no state vector the observation message
already shows, no capture identifiers, rounded numbers, the latest observation
only, and no sentence for a plain completed move. The full result still goes to
`world-use.jsonl`. Selections name a camera's current image, so the per-turn
message listing capture identifiers is gone, and `grasp` and `place` accept the
selection in the same call. On the validator's 14-call episode a grasp result
fell from about 5,000 characters to 1,000, a move result from 600 to 140, and
context growth from about 2,500 characters per call to 900. A raw turn adds about
500. The system prompt and tool schemas remain about 4,000 characters larger than
the raw arm's. These are character counts from recorded mock requests, not billed
tokens, and say nothing about how many calls a model needs.

The native terminal hold step is executed and counted. No physics parameters or
success predicates are altered.

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
| World Use library | `182b78d8e023adb5abe90a121afec2955e48ec70` |
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
is lifted and pass when its own object is. No model requests leave the mock
transport, and no task success is claimed. Pixel and pose fixtures exist only in
`validate.py`. `PYTHONPATH` must name the `src` folder of the World Use tree under
test when the library is not installed in the agent environment.

## Run the authorized pilot

Supply `FOUNDRY_API_KEY` through the environment's credential settings. Set
`LIBERO_AZURE_BASE_URL` to the approved Azure resource's `/openai/v1` endpoint.
Do not store credentials in this repository or pass them on a command line.

```sh
/workspace/libero-runtime/agent-env/bin/python examples/libero/run.py \
  --validation /workspace/libero-validation/validation.json \
  --output /workspace/libero-paid-pilot \
  --authorization "<the user's own words approving this scope>" \
  --execute-paid-rollouts
```

The authorization is recorded verbatim in `protocol.json`. An approval given for
an earlier runner or scope does not cover this one. The committed
`validation.json` records an offline validation of this source in one runtime; a
paid run needs a validation made in its own runtime.

Admission checks exact source and installed-runtime fingerprints. A small live
image/function-call probe precedes both episodes. A pair has a $15 conservative
usage cap, $7.40 per arm, a two-hour deadline and a 30-minute episode deadline
unless `--total-budget`, `--arm-budget`, `--deadline-seconds` and
`--episode-seconds` set others; all are recorded in `protocol.json`. A campaign
over several pairs runs one invocation per pair and must keep its own total.
Output directories must be new. There are no automatic HTTP retries;
unknown usage retains its reservation and blocks the campaign. All requests,
responses and settled usage are journalled without authentication headers. Prices
are assumptions from the published benchmark's model price row, not an Azure
billing reconciliation. Local compute is excluded.

Each arm saves raw observation PNGs, native action/state records, simulator video,
policy transcript and result. World Use also saves tracking and procedure feedback.
Preserve this source branch and the result artifacts durably; a workspace-local
download link is not a backup.
