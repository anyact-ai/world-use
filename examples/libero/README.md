# Matched LIBERO development pilot

This runner compares GPT-6.1 Sol alone with Sol using World Use, on the
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
[`cases.toml`](cases.toml) gives every runnable task and state a role. A held-out
case runs only with `--held-out`, to test finished tools and never to tune them.

The baseline uses the upstream agent's tools and its prompt with one sentence
added, which both arms get: the front camera's image is mirrored, so left and
right in an instruction are the reverse of what that image shows. The World Use
arm adds RGB target selection and tracking, the position of a selected region
measured from the wrist camera, compact measured feedback and the library's
grasp and place procedures. Sol chooses the regions, heights, expected aperture and
placement interpretation. A placement pauses before release and needs a one-use
decision tied to the unchanged observations. Image overlap and finger opening do
not certify object identity, support or task completion.

## Conditions

Three conditions must stay distinguishable in any comparison:

- **Raw Sol**: the baseline arm. This tree does not change what it executes;
  validation compares its approved actions and measured states with upstream.
  Up to `ea1173a` it was told exactly what upstream tells it. Since then both
  arms are told one more sentence, that the agentview image is mirrored, and
  validation checks that the raw prompt differs from upstream's by that sentence
  alone. Paid results before and after it are different raw conditions.
  Neither arm receives the upstream rollout's `approver: N step(s) modified`
  observation line. Restoring it would be another raw condition.
- **Frozen World Use**: commit `ffaaaf6`, whose `native.py`, `world_tools.py` and
  `run.py` have SHA-256 prefixes `ad8b8fab`, `a21a23f8` and `90a9cfc5`. Evidence
  produced with it belongs to that source.
- **Candidate World Use**: this tree, recorded as `world-use-candidate-20261116`.
  Its differences from the frozen runner are described below. They were
  introduced by separate commits so that an earlier commit can be run as a
  narrower condition: settling with applied commands, then the region comparison,
  then the briefer results and merged selection, then the contact-terminated
  grasp, then positions from the wrist camera. A paid result belongs to the
  commit that produced it.

**Settling.** Each leg of a procedure stage repeats its final native pose for at
most 40 more steps, all counted toward the 900-step limit. It uses the library's
`Settling` check and returns as soon as the measurements settle the question.
The arm has `arrived` after three consecutive samples within 12 mm and 0.12
rotation-column error that moved under 2 mm per step; it is `not_approaching`
once that offset has stopped shrinking over four intervals for five steps
running; otherwise it is `still_converging` at the bound. The fingers have
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

**Positions from the wrist camera.** The first paid pair, task 0 state 0 at
`6d73def`, went against the candidate: raw Sol satisfied the success check in 49
calls and 653 steps, and the World Use arm gave up after 21 calls and five failed
grasps. Both arms had assumed the grasp point is the centre of the wrist image.
The wrist camera is mounted 5 cm from the grasp axis, so the point under the
fingers appears near the top of the image, and both arms' first grasps were about
5 cm off. Raw Sol worked the offset out after 21 calls and two failed grasps.

A `grasp` whose object was selected in the wrist camera now moves the hand level
until it is over the region and only then descends. One look gives a direction;
the hand looks again from at least 3 cm away and the region is placed where the
rays meet, using the camera's mount on the hand. `locate` does the same without
descending or moving the fingers and returns the position, for a destination
the policy will carry something to. The mount is a constant of the robot read
from its model, not of any task, and the raw arm is not told it; the docs given
to the World Use arm state where the grasp point appears in the wrist image.

Starting 5.4 and 5.8 cm off ten cans and flat boxes, with the object's projected
centre as the selection, the hand was centred within a median 2.4 mm and at most
10.7 mm, and 17 of 20 grasps lifted the object
([centring-evidence.jsonl](centring-evidence.jsonl)). Two failures were the hand
body landing on a milk carton 8 cm from the cream cheese box, which no vertical
grasp across that box avoids. In the third the tracker dropped a visible region
on the last 3 mm correction; a region lost after it was placed now keeps its
position. `grasp` also opens the fingers before it descends, so it can follow a
failed attempt directly; `place` takes a `travel_z` and rises to it before moving
level; and an approach that stalls reports how far above and aside of the request
the hand stopped. With those changes, scripted runs of tasks 0 and 7 from hovers
5 cm off every object satisfied the success check after 11 calls and 479 steps:
one `locate` of the basket within 3.5 mm, two wrist-selected grasps, two
placements, one release and five moves. These are scripted, privileged
diagnostics of the tools, not of a model using them.

**Stopping on contact, and using what was located.** The second paid pair, at
`da32947`, was closer and still went against the candidate. Raw Sol succeeded
again, in 44 calls and 557 steps. The World Use arm put the soup in the basket
and was carrying the tomato sauce when the 900-step limit ended it after 23
calls. It had located the basket within 2 cm from a point at the edge of the
wrist image, then released 4 cm from that position, onto the rim. Each blocked
descent kept pushing for 16 steps with the command up to 13 cm below the hand,
tilted the hand, and needed two more moves to back out, because the approvers
limit every step from the last applied command. Three stalled placements and
their recoveries used about 300 steps.

A settled stage now ends once the hand gains less than a millimetre toward a
command at least 1.5 cm away on two consecutive steps, and a hand that stopped
short is commanded to rest where it is: over the pose when it met contact there,
so that a palm resting on a can does not slide off it, and at its measured pose
otherwise. The next move then starts unclamped. `grasp` and `place` report
`descent_ended_above_request_m`, which tells a landing on the intended surface
from one on a rim or a neighbour; a level or travel leg that stalls reports the
hand's offset from the whole request, not from the leg's own target. `locate`
also returns the region's extent and the pixel where the located point appears
in the next wrist image, and the docs say to pass located coordinates unchanged
and to select with a box: a box around the basket placed it within 1.2 mm in
both states of task 1, where a point on its liner was 35 mm off in one.
Centring now ends once the hand has arrived within the arm's 12 mm tolerance of
the region, which the recorded looks show costs no accuracy and takes 1.9 moves
instead of 2.3, and a sidestep is long enough that an object straight below the
hand still gets two separated looks.
`grasp` needs only z in `lift`, takes an `offset` from the centre of a wrist
region for a rim grasp and a `yaw_deg` to turn the finger axis away from a
neighbour, and lets a close run 120 steps while the fingers are still moving;
a release whose fingers are still creeping inside the expected opening counts
as released.

With these changes the 20 centring trials lifted 18, the two failures again the
box beside the milk carton, and the scripted task 0 run took 471 steps. A grasp
turned 45 degrees clears that carton and lifted the box in four of four trials,
in 158 to 189 steps, most of them the fingers turning the box before they hold
it.

**Looks at the whole region.** The pairs at `b61545e`, tasks 0 and 7, were the
first to favour the candidate: the World Use arm succeeded in both, in 13 and 15
calls, where raw Sol took 49 calls on task 7 and on task 0 declared the task done
after 58 calls with the wrong object in the basket. Both World Use episodes
still used 762 and 818 of their 900 steps on five and six placements. `locate`
had been given the basket at the edge of the wrist image and returned a centre
within 2 cm but a size 1.5 times too large: the centre of a region that the
image border cuts off moves as more of it comes into view, so the looks did not
agree on a depth. The policy then released 7 to 9 cm from the centre, onto the
rim.

A region is now placed only from looks that contain all of it. Looks at a
cut-off region lead the hand toward it, in one move to where the look meets a
plane a quarter metre below the camera, and one centring move that stops short
of an unreachable first guess no longer ends the search. The docs advise
releasing the first of two objects a fifth of the container's longer extent to
one side of its centre and the second to the other side. Scripted from the start
pose that way, tasks 0 and 7 located the basket within 6.4 mm in 63 and 69 steps
and satisfied the success check after 9 calls and 478 and 456 steps.

The native terminal hold step is executed and counted. No physics parameters or
success predicates are altered.

**A hand at rest.** The scenes of the five kitchen tasks put the table 46 cm
higher. Hovering there the arm swings: a 5 cm level move throws the hand 2 cm
up and it takes about 25 steps to come back to rest. Up to `c418a71` such a
hand counted as arrived while it passed through its pose at 4 mm per step, and
the small move that followed, begun on the swing, was reported as stalled:
both scripted centre grasps in those scenes ended that way before any descent.
Arrival now also needs the hand to move under 2 mm per step, a stall has to
last five steps, since a swinging hand makes no progress either while it turns
round, and a leg may settle for 40 steps instead of 20. A hand that gains
nothing on its command is still taken for held at once when the command leads
it by 3 cm, or by 1.5 cm in a descent that may end on contact, which starts
from rest. A grasp of a wrist-selected object needs only z in its approach:
at `c418a71` the docs called x and y ignored, the policy left them out, and
every episode lost a call to the refusal.

With these changes the two kitchen centre grasps complete, and the 20 centring
trials in the first scene lift the same 18 in a median 76 steps instead of 66;
the scripted task 0 run takes 524 steps instead of 483.

**Into the library.** The geometry and the judgements above lived in this folder,
where no other robot could use them and no core test covered them. They are now
library code: [`world_use.parallax`](../../docs/perception.md#positions-from-a-camera-on-the-hand)
for positions from a camera on the hand, a `View` in the tool frame with a signed
focal length for the wrist camera's flipped image, and
[`checks.Following` and `Settling(stall_s=...)`](../../docs/adapters.md#measured-completion)
for a held hand and a lasting stall. The library's own simulated arm finds and picks
a block with them in its tests. The driver's loop and the tool schemas stay here:
they are shaped by this embodiment's absolute-pose commands and approvers, and one
robot is not enough to choose an interface from. The move was checked against the
scripted simulator episode in `validate.py`, which now records its step count and
hashes of the applied commands, the results told to the policy and the prompt. All
were unchanged except one reported size, 74 to 73 mm: the region's box is scaled by
distance along the optical axis instead of by height difference.

The hand-set motion thresholds were tested instead of replaced. On a fake arm that
closes 8 to 70 percent of its gap to the command each step and keeps none to 70
percent of its speed, they tell free motion, contact and a blocked move apart in
every case tried; they fail only for a slow arm that keeps 85 percent of its speed.
Measuring each arm's response was therefore not built.

**Four changes after the move.** Each is a new condition.

- *Remembered positions.* A position found for a target is kept until the target
  is selected again or a grasp closes on it. Locating or grasping it again confirms
  it with one look instead of working it out again.
- *Looks without waiting.* A look needs the pose it is taken from, not a hand at
  rest, so centring moves end on arrival, and the rise to a travel height runs into
  the level carry at that height. In the recorded development episodes 36 percent
  of the World Use arm's steps were waits.
- *Fingers across the narrow side.* `locate` and `grasp` report the narrowest box
  around the region (`across_mm`, `along_mm`). Given no yaw and no offset, a
  grasp of a region the policy outlined with a box in the wrist camera, clearly
  longer than wide and with a narrow side of at most 70 mm, turns the fingers
  across it. A point can select only part of an object, so a point selection
  never turns the hand. Grasps stay vertical.
- *A review before done.* LIBERO ends an episode by itself when its check passes,
  so a `done` from the policy is always a claim the check rejects. The tools cannot
  know that and do not use it. The first `done` is answered once with a review: the
  hand rises to the highest it has been, the result lists what was grasped and
  released, and the policy is asked to check each named object in the images that
  follow before calling `done` again.

Not built: re-seeding a lost region from its remembered position, because a region
that leaves the wrist view is needed again only through its coordinates; and a
left-behind check for wrist selections, which needs the front camera's pose, and
that differs between LIBERO scenes.

## Paid results

**Basket tasks, both arms told the image is mirrored.** Tasks 0, 1 and 7, one
matched pair per task and state: development states 0 and 1 at `f47ff9f`, and
the held-out state 2 at `4800819`, the commit before it, which still refused a
test lift of exactly 5 cm.

| | raw Sol | World Use |
|---|---|---|
| development states: satisfied the success check | 4 of 6 | 6 of 6 |
| held-out state: satisfied the success check | 2 of 3 | 2 of 3 |
| how the others ended | gave up 2, claimed done 1 | claimed done 1 |
| model calls over the nine pairs, total and median | 338, 36 | 105, 11 |
| counted steps | 4,530 | 6,111 |
| input and output tokens | 2,215,097 and 31,281 | 599,463 and 8,998 |
| estimated cost at the assumed prices | $3.94 | $0.67 |
| wall time | 3,381 s | 3,057 s |

On the six pairs in which both arms succeeded, the World Use arm used 62% fewer
calls and 60% fewer input tokens, and 55% more steps. On held-out task 7 both
arms declared the task done and the check disagreed. The World Use arm had
taken the tomato sauce for the alphabet soup; the raw arm took the right can,
and its record does not show what the check rejected.

**The same development pairs before that sentence.** At `c418a71` the raw arm
was told only what upstream tells it:

| | raw Sol | World Use |
|---|---|---|
| satisfied the success check | 2 of 6 | 6 of 6 |
| how the others ended | gave up 3, claimed done 1 | |
| model calls, total and median | 260, 43.5 | 71, 12.5 |
| counted steps | 3,651 | 3,659 |
| estimated cost at the assumed prices | $3.30 | $0.42 |

Raw Sol went from two successes to four on these six pairs, and from a median
43.5 calls to 35.5, once it was told the image is mirrored. With one episode
per cell that is a suggestion, not a measurement, but it is the likelier reading
that part of the earlier gap was the raw arm misreading a mirrored image.

**The other seven tasks.** At `e1f4c5e`, before the sentence, each was run once
on state 0:

| task | raw Sol | calls | est. USD | World Use | calls | est. USD |
|---|---|---|---|---|---|---|
| 2: turn on the stove, moka pot onto it | step limit | 56 | 0.90 | **success** | 15 | 0.09 |
| 3: bowl into the bottom drawer, close it | **success** | 49 | 0.77 | step limit | 34 | 0.44 |
| 4: two mugs onto their plates | gave up | 74 | 1.47 | claimed done | 14 | 0.10 |
| 5: book into the caddy | gave up | 54 | 0.82 | gave up | 19 | 0.16 |
| 6: mug onto the plate, pudding right of it | gave up | 61 | 1.06 | claimed done | 12 | 0.08 |
| 8: both moka pots onto the stove | gave up | 61 | 1.00 | gave up | 23 | 0.21 |
| 9: mug into the microwave, close it | gave up | 60 | 1.00 | gave up | 13 | 0.09 |

Each arm succeeded once, on different tasks. The World Use arm made 130 calls
for an estimated $1.17 where raw Sol made 415 for $7.01, and it twice declared
a task done that the success check rejects: in tasks 4 and 6 it rim-grasped
both mugs and the pudding and set them down on the side the mirrored image
shows. Tasks 4 and 6 were run again at `4800819` with the sentence. In task 4
the World Use arm succeeded in 13 calls and raw Sol gave up after 35. In task 6
both arms declared done and the check disagreed; the World Use arm had put the
pudding on the correct side, 5 cm beyond the strip beside the plate that the
task accepts.

Where the World Use arm fell short:

- Nothing in these tools checks what a selected region is. Two of its five
  false completions took one grocery for another: the butter for the tomato
  sauce on task 0 at `e1f4c5e`, the tomato sauce for the soup on held-out task
  7. Raw Sol's false completion at `b61545e` was the same kind of mistake.
- The bowl was taken by its rim on the fourth attempt and set in the drawer, and
  the steps ran out while the drawer was being pushed shut. One moka pot reached
  the stove and the second slipped when the two touched. The mug for the
  microwave was lifted by its rim and lost on the microwave's roof during the
  carry. The book was never held. A grasp from above, even with a rim offset,
  holds these objects poorly or not at all.
- It uses more simulator steps than raw Sol on the pairs both complete, and its
  successful episodes ended between 497 and 855 of the 900 allowed.
- On task 5 `locate` returned a point above the camera for a clipped selection.
  Looks that meet less than 5 cm below the camera are now discarded.

What these results do not show:

- Each cell is one episode. Sol's raw outcomes on task 0 state 0 across six runs
  were two successes, one false completion, two give-ups and one step limit.
- The development states are the ones the tools were written against. Task 0
  state 0 had been run with three earlier versions of the World Use arm, which
  failed twice. The held-out state was run once, after the tools were finished.
  The one change made after it, accepting a 5 cm test lift, removes a refusal
  that seven of the eight development episodes at `e1f4c5e` had also hit.
- The other seven tasks were run once and not iterated on, and only tasks 4 and
  6 have been run since both arms were told about the mirrored image.
- The World Use arm is told the wrist camera's mount and the hand's dimensions.
  The raw arm is not.

[paid-results.jsonl](paid-results.jsonl) has one row per arm of every paid pair,
with the commit that produced it, how the episode ended, calls, counted steps,
wall time, tokens and estimated cost at the assumed prices. The World Use arm
is the tree at that commit. The raw arm is one condition up to `ea1173a` and
another from `4800819`, as described under Conditions. Wall times were not
controlled for other load on the machine. A pair in which the transport refused
a request has one row, for the arm that was running, and no result for either
arm. All 32 pairs together cost an estimated $26.05, $39.92 at the conservative
rate the transport reserves.

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
is lifted and pass when its own object is. From hovers 5 cm off, `locate` and a
wrist-selected `grasp` must each place a can within 1.5 cm of its fixture
position, and that grasp must hold it. The raw arm's prompt must be upstream's
plus the one sentence both arms are given. No model requests leave the mock
transport, and no task success is claimed. Pixel and pose fixtures exist only in
`validate.py`. `PYTHONPATH` must name the `src` folder of the World Use tree under
test when the library is not installed in the agent environment.

## Cases

[`cases.toml`](cases.toml) lists every task and initial state the runner accepts,
each with a role. Development cases, states 0 and 1 of every task, are for
iteration. A held-out case runs only with `--held-out`, as a test of finished
tools, and is never inspected to tune them: state 2 of tasks 0, 1 and 7 has been
run and reported once, and state 3 of every task has never been run. The runner
refuses an unlisted case, a held-out case without `--held-out` and a development
case with it.

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
words approving that test. The campaign admits a pair only while the conservative charge of its finished
attempts plus the full caps of running pairs leaves room for one more pair cap,
and admits nothing more once any attempt has unknown usage. Pairs it finds
complete in its folder are skipped; an interrupted attempt is kept, charged and
retried in a new `_retryN` folder, and each repeat after the first has its own
`_repeatN` folder. `--arms world_use` runs only the World Use arm, for a
development check that reuses the raw arm's earlier result; `protocol.json`
records the arms run.

Append the campaign's pairs, labelled with the commit they ran, to the compact
rows, and print the comparison tables. The tables use each case's latest commit
for each arm and report repeats as solved k of n:

```sh
python examples/libero/report.py collect examples/libero/paid-results.jsonl \
  <commit>=/workspace/libero-paid-campaign
python examples/libero/report.py tables examples/libero/paid-results.jsonl
```

The authorization is recorded verbatim in `protocol.json`. An approval given for
an earlier runner or scope does not cover this one. The committed
`validation.json` records an offline validation of this source in one runtime; a
paid run needs a validation made in its own runtime.

Admission checks exact source and installed-runtime fingerprints. A small live
image/function-call probe precedes the episodes. A pair has a $15 conservative
usage cap, $7.40 per arm, a two-hour deadline and a 30-minute episode deadline
unless `--total-budget`, `--arm-budget`, `--deadline-seconds` and
`--episode-seconds` set others; all are recorded in `protocol.json`. A campaign
runs one invocation per pair, with 3,000-second episodes and a 6,000-second
pair deadline. Output directories must be new. There are no automatic HTTP retries;
unknown usage retains its reservation and blocks the campaign. All requests,
responses and settled usage are journalled without authentication headers. Prices
are assumptions from the published benchmark's model price row, not an Azure
billing reconciliation. Local compute is excluded.

Each request is reserved at a conservative size before it is sent and refused
above 200,000 tokens. The size was one token per byte of request text, about
twelve times what the service counted, which refused a raw arm's 61st request
although the arm is allowed 80 calls. It is now half a token per byte plus 8,192
per image: at least six times the count in every one of the 394 requests
recorded up to `72d0fca`. A raw arm that uses all 80 calls needs about $2.70 of
conservative allowance.

Each arm saves raw observation PNGs, native action/state records, simulator video,
policy transcript and result. World Use also saves tracking and procedure feedback.
Preserve this source branch and the result artifacts durably; a workspace-local
download link is not a backup.
