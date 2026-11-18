# Design

world-use separates an agent's decisions from a robot's control loop. The agent submits a phase of work;
the runtime checks motion limits, executes the plan, and reports the outcome. The design comes from trying
to make those phases useful while spending less powered time waiting for the next decision.

## What we measured

The initial design came from running Claude Opus 5.5 and GPT-6 Astra as policies on a Seeed reBot arm (six joints,
no brakes) and in MuJoCo workcells. One day of a real task (open a 3D printer's
glass door, take the print out, show it to a camera; 20 powered runs, 2 h 10 min with torque on) gave the
numbers that shaped it:

- The arm moved for 14% of its powered time. The rest it held still while the model thought (a median 9 s per
  decision, some over a minute, each reading a conversation of about 390k tokens).
- Holding still is not free. Every useful raised pose loaded the elbow at 6-8 Nm, its continuous rating, and
  the elbow heated about 8 C per minute. One run aborted at 86 C.
- Improvising move by move: 11% of powered time moving. Planning the task offline, checking it against the
  robot model and running it in guarded batches: 26%, and 41% in the final clean run.
- In simulation, a model choosing 2 cm steps at 6 Hz placed 0 of 2 blocks; a model choosing one target per
  camera look built a tower but moved for only 9% of its episode.
- Most retries were not reasoning errors. They came from positions misjudged in images, world state nobody
  checked (a door the human had already opened), no lookahead (a pose with no way back), unsafe automation
  (an idle timeout that folded the arm into an open glass door), and processes dying with the agent's tool
  call. A person's physical intuition fixed the remaining ones.
- Tooling decides how fast a model gets started. A fresh agent given only the brief moved a block in
  simulation in 14 calls and 409 s with v0.1, and lost most of that time to what the tool did not say: which
  way the gripper points, which limits apply, positions it had to re-derive. With v0.2 (a card that says what
  is possible, checks that name every problem, pictures), the same task took 11 calls and 241 s, with no
  refusals, from a harder start pose.

These are single-task, single-robot observations, not benchmarks. They are why the rules below exist.
The moving-time percentages above count ticks at the nominal control rate. Flight records measure elapsed
monotonic time, stalled ticks and power transitions included, so their percentages are not directly comparable.

## Rules

1. **Two clocks.** The robot runs at 100 Hz or faster; the model at a few seconds per decision. The model
   decides phases, targets and skills; the kernel runs everything below that. Think with the torque off.
2. **Plans are data, checked before they run.** Code builds a plan; the plan is JSON; the same kernel runs
   it on a twin first and on the robot second. By default, `wu run` requires an idle robot and checks the
   whole plan; when the job is accepted, and again at its first tick, the kernel verifies that the state it
   checked from is still current. `--no-check` and embedded `Kernel.run` check each step as it starts, so
   earlier steps may have moved. HTTP and MCP reject unknown arguments, invalid types and nonfinite numbers
   before dispatch, using shared Pydantic request models.
3. **Make refusals useful.** A refusal names the violated limit and, where available, a correction
   ("lift at least 2 cm more first"). The card says which way the gripper points and opens, where known
   things are in the frame moves use, and which short moves are possible from here. A model should never
   have to find the limits one refusal at a time.
4. **Every step says what it expects.** Contact within a distance, a grip within a width, an answer at a
   checkpoint. Anything else is a surprise: the robot holds where it really is, queued work is cancelled,
   and the model gets an incident report instead of a stream of numbers. A wait for arrival ends at a deadline
   or as soon as progress stalls, and grip checks compare the measured opening with an expected range; whether
   the object is still held, the policy checks from observations.
5. **Idle means holding.** Nothing moves on a timer. Holding still heats the motors and is not safe in every
   situation, so the agent resolves motor power, normally by going home and releasing torque, before it hands
   off or waits for a reply: software cannot safely release a raised arm that has no brakes. The one automatic
   motion is a thermal return. When a motor reaches its limit, the arm follows the home route the agent set and
   torque is released, unless contact has made the route stale or the kernel is faulted; without a usable
   route, the kernel holds and alarms for the operator at once. A home route holds only built-in motion and
   gripper steps, with no waits or contact steps. The daemon rehearses a new route while idle and installs it only if the rehearsal passes and its
   starting state is still current; otherwise the installed route stays as it was, stale or not. A failed
   thermal return clears the route and calls the operator.
6. **Monitor unexpected contact.** On adapters with joint torque sensing, readings are compared with what the arm's own weight explains;
   guarded moves use tighter thresholds, fragile zones tighter still. Filtering, sensing, and model error
   determine detection latency. The checks do not replace hardware protection or an operator.
7. **Facts remember their source.** A door angle, a table height, a camera pose: recorded with where they
   came from, and marked stale when a surprise shows the world may have changed.
8. **The robot process outlives the agent.** One daemon per robot. Agents, consoles and viewers are clients.
   An accepted plan continues after a client disconnects; a checkpoint waits for an answer. Between jobs,
   the runtime holds position. Client loss does not imply cancellation.
9. **Context is a budget.** One state line per step. Events, not sensor streams. The robot's card once.
   Incidents with the expected and the observed side by side. Pictures when asked for, with what the kernel
   believes drawn on them, so a wrong belief shows in one look.
10. **Failures are evidence.** Every run keeps a flight record (tape, events, pictures, world, summary) so the
    next attempt, or the next plan written offline, starts from what actually happened.

## The core

The core has six concepts:

| Concept | Responsibility |
| --- | --- |
| **Body** | A manifest (joints, limits, gripper, rest pose, sensing, notes) and six adapter methods: connect, enable, read, command, disable, close. |
| **World** | Frames, boxes (surfaces, objects, keep-out, fragile and slow zones) and facts with their sources. |
| **Behavior** | Anything that moves the robot, under one contract: `start` prepares and may refuse; `tick` runs one control step and returns an outcome when done. A line, a guarded touchdown, a grip, a checkpoint and a whole plan are all behaviors. |
| **Event** | One numbered stream of everything that happened. |
| **View** | State rendered for a reader: the state line, the status, an incident, the robot's card, a camera picture with the world drawn on it. |
| **Kernel** | The only code that talks to the body. Each tick: read, watch, advance the behavior, command, record. |

The envelope inside the kernel holds the limits: joint limits with a margin, speed and acceleration, the
excursion from the session's start pose, gravity load, keep-out zones, surfaces, and robot-specific rules (the reBot
may not turn its base while the gripper is at table height). An agent can tighten it; loosening it is an operator
override with a reason, and it is logged.

Keep-out checks cover segments from joint to joint, padded by the manifest's `link_radius_m` (3 cm by default),
declared tool and finger boxes, and the estimated box of a held object. Planning checks their swept motion,
payload rotation included, and every measured tick checks the same geometry. Surfaces constrain the tool point
and those boxes. A guarded move toward a support may plan a small allowance past its surface to find contact,
shared by every part that reaches it wholly above its footprint. The allowance ends with the step or its
interruption, leaving only the overlap already measured: that can shrink as the arm withdraws, but never grows
at step boundaries or in planning snapshots. A re-measured support or payload starts again from its measured
overlap, so withdrawal stays possible. Undeclared tools and fingers, the pedestal, self-collision, arm links
against surfaces and objects that are not held are not checked; the capability report lists the declared
geometry and these omissions ([collision bounds](docs/adapters.md#collision-bounds)). Slow zones cap the planned
tool speed.

Faults latch until an operator resets them, and a watchdog trip while idle cancels anything queued. An incomplete
power transition, or failed I/O while powered, marks motor power unconfirmed (`power_uncertain`): commands stay
suspended, even across reconnection, until a release succeeds at a freshly measured rest pose. Status reports
feedback age and read errors and marks cached values stale.

Feedback is checked before replacing trusted state: joint arrays must have the expected dimensions and finite
values, and timestamps must be finite and nondecreasing. Optional readings may be absent. Invalid feedback
suspends commands through the same fault path as a failed read. Shutdown closes admission before releasing
power, including enable calls already waiting on the control thread; a failed release leaves recovery available.

The control thread never plans. Rehearsals, reach probes and each step's path come from one worker process that
works from a snapshot of the kernel (robot description, world and state) without importing a hardware driver.
While it computes, the control loop keeps reading, checking limits and holding; before playback it checks that
the command, scene and limits still match. Every run keeps a [flight record](docs/records.md), written off the
control thread.

## Two loops, one runtime

Agents develop and test procedures in simulation, then execute them through the same runtime. Preparing work
offline reduces powered waiting. Unfamiliar tasks still need live observations and decisions, especially when
contact or an unexpected scene change invalidates the plan. So world-use serves both loops with the same kernel,
behaviors and records:

- **Offline**: an agent writes a plan, checks it on a twin, evaluates it, and ships it as data the kernel
  executes. No model runs while the robot moves.
- **Online**: the same plans run with checkpoints and surprise handling, and a model answers the questions and
  decides what to do after a surprise.
- **Between them**: online runs leave flight records; those records tune the twin and become test cases for the
  next offline iteration.

The [task-program example](examples/task-programs) records Python source, inputs and calculations alongside
the flight record. A repeated task runs the program again with fresh observations; replaying a recording
only visualizes the earlier motion.

The core uses MuJoCo for simulation, numpy for planning and Pillow for image overlays. MuJoCo renders the same scene
it simulates, with the robot's meshes, and grasps use frictional contacts. Rehearsals run on the kernel's estimated
world, never on the simulation's truth; servo parameters and motor heating are approximate. Physics, planning and
rehearsal need neither ROS nor a GPU; rendering simulated cameras needs OpenGL, from a GPU or from EGL or OSMesa on
a headless host ([simulation](docs/simulation.md)). The optional Rerun viewer reads flight records in its own
process; it never imports a hardware driver or checks a plan.

Perception turns pixels into numbers a plan can use. The daemon keeps the last few camera frames, unannotated
and unrecorded (`look` is the recorded, annotated view); a MuJoCo frame can add aligned metric depth. Measuring
a point or box of a frame gives the visible surface's position in the work frame and saves the picture and points
with the run. A run can require measurements: before each step the kernel refuses once one is too old, its
camera was calibrated again or a tracker withdrew it, judging only numbers copied when the run was accepted, so
the control loop reads no files. Measurements never change the world model; the agent checks an outcome by
measuring again. Optional EdgeTAM tracking runs in its own process, never in the control loop. See
[measuring from pictures](docs/perception.md).

## Where it goes

The [block example](examples/pick-place) provides a complete run, a separate success check, scene variations
and a recovery to inspect. The [Panda kitchen adapter](examples/robocasa-counter) extends this to an external
environment, while [shape sorting](docs/shape-sorting.md) tests a live model using measured alignment.
The next development cycle connects scene reconstruction, execution, evaluation and visual inspection.
Pushing and kitchen packing share a [simulation session](docs/simulation.md#external-simulation-sessions), and
reference checks, comparable evaluations, declared collision bounds and shared inspection each have a first
working version. The milestones below keep the acceptance criteria and limits of the whole cycle.

World-use's direction is to help agents turn observations into checked robot actions and improve from recorded
outcomes. Other projects, including [RoboSimStudio](https://robosimstudio.github.io/) and
[ASPIRE](https://research.nvidia.com/labs/gear/aspire/), are sources of inspiration.
Derive our interfaces, implementation and user experience from our own tasks and existing architecture; copying
another repository or pursuing feature parity is not the objective. Each addition should solve an observed
problem in setup, scene accuracy, execution or diagnosis, with a concrete way to tell whether it helped.

### Scene reconstruction quality

A useful scene must reproduce the observations and physical interactions that matter to the task.
For world-use, reconstruction quality should make observations, motion checks and predicted outcomes more
trustworthy. Keep separate results for three aspects of reconstruction:

- **Appearance and camera:** compare reference frames with renders at matching viewpoints and timestamps.
  Inspect silhouettes, occlusion, materials and lighting; measure landmark error and mask overlap where suitable.
  Keep original frames, calibration and any crop or time alignment so comparisons can be reproduced.
- **Geometry:** establish metric scale, object dimensions, poses, support surfaces and contact shapes.
  State which measurements came from calibration, depth, a known dimension or an assumption. An attractive
  render or a good silhouette match alone does not establish the geometry behind it.
- **Physical response:** check finite state, penetration, support and drift, then compare task-relevant motion
  such as sliding distance, rotation, settling or deformation. Compare forces only when the reference has force
  measurements. Record solver settings and parameter sources, and mark unmeasured properties as uncertain.

Define tolerances from the task and available measurements before tuning. Fit on selected frames or motions,
then validate on reserved observations; several views from the same clip do not establish transfer to another
setup. Keep visual fit, physical validity and task success as separate results. A scene with missing evidence
is unvalidated in that respect. Changing relevant geometry, camera or dynamics requires a new validation.

Start with one recorded rigid-object contact task with known scale and observable motion. A synthetic reference
can test the tooling but cannot validate real-to-sim accuracy. Reuse the camera fitting and projection code;
`wu fit` remains a robot mass/friction fitter, not a scene or material estimator. Broader contact, actuator and
thermal fitting needs corresponding measurements. No new hardware operation is part of this development plan.

### Ownership and interfaces

- **Scene setup** owns simulator construction, initial conditions, cameras and physical-state capture. Start
  with the existing MuJoCo integrations and extract shared code only as two tasks use it. Keep task geometry
  and simulator-specific details with their scene; keep the six-method `Body` contract.
- **The kernel** owns robot I/O, power, limits, faults and evidence freshness. Planning remains in its worker.
  Its estimated `World` stays separate from simulation truth. A reconstructed model may contribute declared
  calibration and prior geometry; live hidden object state and evaluator answers do not enter policy tools.
  Extend the existing capability report with rehearsal coverage, including omitted geometry and dynamics.
  A passing partial rehearsal must state what it checked; improved scene rendering alone cannot expand it.
- **Task programs and evaluation** use the existing client and task-program recorder. The policy receives
  observations and submits plans; an independent evaluator reads the physical outcome. Scene validity, task
  outcome, recording completeness and power cleanup remain distinct. Privileged diagnostic runs are labelled
  and excluded from camera-only comparisons. This is an API boundary for trusted code, not a Python sandbox.
  Agent orchestration and skill promotion remain outside the runtime. Declared prompts and skill documents
  are frozen with the policy revision; changing that knowledge preserves the scene's case identity.
- **Records and views** connect the executed source, scene revision, assets, calibration, parameters, seed,
  exact observations, actions and outcomes. Extend the existing run records with backend-specific scene/replay
  artifacts rather than maintaining a second generic recording system. Views show reference footage,
  reconstruction, policy observations and estimated geometry with their roles labelled.
  Named task phases group observations, computations and calls in the same trace. Inspection expands one
  phase's artifacts and known job outcomes; incomplete calls show an unknown outcome.

Perception, grasp proposals and planning are separate from the `Body` I/O contract. Add a provider boundary
when a task needs another implementation, using explicit frames, units, observation provenance and failure
results. Isolate heavyweight dependencies in an optional process or service and keep the CPU setup working.
Choosing a backend, or falling back to another, must never happen silently; the resulting actions still pass
through kernel checks.
Keep agent-facing help and request validation tied to the same contracts.

### Delivery sequence

Use these milestones to test the design against our tasks. Choose the smallest useful implementation at each
step, and revise the scope when the evidence shows that a different approach serves world-use better.

1. **Support the existing external scenes consistently.** Migrate pushing and kitchen packing onto a shared
   session and recording path, preserving their task programs and physical audits. Start independent episodes
   with a fresh kernel, measurement store, clock and output folder; a simulator reset alone is insufficient.
   Keep lifecycle and state capture outside the robot adapter where possible. Completion means both tasks can
   run through the normal client, preserve their evidence and replay the actual external scene, with their
   rehearsal coverage stated. Check interruption and cleanup as well as successful execution.
2. **Add reference comparison and scene validation.** Pin a reference clip, scale evidence and task-relevant
   tolerances. Produce aligned image comparisons, geometry residuals and physical-response measurements tied to
   a scene revision. Start with hand-picked correspondences and editable parameters. Completion means an agent can
   identify a mismatch, revise the scene and show improvement on reserved observations. Deliberately wrong scale,
   camera pose or unstable contacts must fail the relevant checks; missing measurements remain unvalidated.
   The [first implementation](examples/scene-validation) covers explicit landmarks, dimensions, trajectories
   and stability. Synthetic refinement passes reserved numerical checks. In five independently recorded square-object
   pushes, two reserved repeats meet the planar-motion tolerance, while a new contact position fails. The faster
   hex push remains a failing development case. Measured camera reconstruction, settling evidence and reliable
   response across new contacts remain; friction fitting alone has not resolved them.
3. **Make repeated evaluations comparable.** Build on the task-program launcher with seeded initial conditions,
   task-specific success functions and a small shared helper for conditions that must hold over simulated time.
   Record wall time and simulation time separately, failure reasons, interventions and scene-validation results.
   Use fixed development cases and held-out layout/parameter variations; retain invalid and interrupted attempts
   in the report. Completion means a failed episode can be recreated under pinned versions and policy revisions
   can be compared on identical cases. Tests must catch transient success, invalid starts, stale evidence and
   accidental exposure of evaluator state to the policy-facing interface.
   The [shared runner](src/world_use/evaluation.py) and [pushing cases](examples/push-letters) freeze source,
   inputs and runtime, retain every attempt, and compare camera cases with a separate physical audit. Broader
   performance across held-out scenes remains an experiment to run, not a claim from infrastructure tests.
4. **Extend collision coverage for concrete motions.** Begin with fingers, the tool and a carried object's
   conservative geometry, using the same declared geometry in planning and runtime checks. Handle intended
   support and contact transitions, and list the remaining omissions, including self-collision. Full
   external-scene rehearsal is separate work and must use a declared estimated model, not hidden scene state.
   Completion means the planner rejects representative finger/payload collisions while allowing validated
   placements and withdrawal. Add obstacle-avoiding search or an optional planner only when those cases need it.
   Declared boxes, swept payload checks and scoped support contact are implemented and regression-tested.
   Tool/finger bounds are opt-in; [adapter declarations](docs/adapters.md#collision-bounds) list the omissions.
5. **Unify live inspection.** Start reference/scene snapshots in step 2; add persistent viewing, rerun/reset and
   reload after the session lifecycle is stable. Reuse the current viewer where it fits. Rebuild only an idle,
   powered-off simulation; keep the last valid scene visible on a failed build and display the error. Never
   hot-reload hardware. Completion means an agent and a person can inspect the same revision, compare it with
   the reference, rerun a named episode and inspect its actual observations and failure without custom scripts.
   [Experiment inspection](docs/records.md#inspect-an-experiment) connects these saved artifacts in the
   existing CLI and viewer, including named reruns and optional recorded-scene snapshots. Persistent reset/reload
   is an embedded, simulation-only lifecycle API; interactive scene authoring and measured video alignment remain.

Keep scene definitions and evaluators as ordinary Python with small data files for parameters and results.
Choose shared interfaces during the two-task migration, before adding a plugin registry, task language or
workflow graph. Keep CPU MuJoCo usable without new GPU dependencies. Deformable or multiphysics backends should
be optional integrations justified by a specific reconstruction that needs them. Add learned perception,
reusable skills and multi-arm scheduling when a measured task exposes the need. Repeated performance across
changed scenes and later, separately authorized hardware trials remain the evidence for transfer.
