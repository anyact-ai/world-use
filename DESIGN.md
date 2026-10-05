# Design

world-use separates an agent's decisions from a robot's control loop. The agent submits a phase of work;
the runtime checks motion limits, executes the plan, and reports the outcome. The design comes from trying
to make those phases useful while spending less powered time waiting for the next decision.

## What we measured

The design comes from running Claude Opus 5.5 and GPT-6 Astra as policies on a Seeed reBot arm (six joints,
no brakes) and in MuJoCo workcells, not from first principles. One day of a real task (open a 3D printer's
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
The original moving-time percentages were computed from nominal-rate tick counts. Current records use
elapsed monotonic time, including stalled ticks and power transitions; do not compare the two accounting
methods as if they were the same measurement.

## Rules

1. **Two clocks.** The robot runs at 100 Hz or faster; the model at a few seconds per decision. The model
   decides phases, targets and skills; the kernel runs everything below that. Think with the torque off.
2. **Plans are data, checked before they run.** Code builds a plan; the plan is JSON; the same kernel runs
   it on a twin first and on the robot second. By default, `wu run` requires an idle robot and checks the
   whole plan; when the job is accepted, and again at its first tick, the kernel verifies that the state it
   checked from is still current. `--no-check` and embedded `Kernel.run` check each step as it starts, so
   earlier steps may have moved.
3. **Say what would pass.** A refusal names every limit a plan would break, with the number that would pass
   ("lift at least 2 cm more first"). The card says which way the gripper points and opens, where known
   things are in the frame moves use, and which short moves are possible from here. A model should never
   have to find the limits one refusal at a time.
4. **Every step says what it expects.** Contact within a distance, a grip within a width, an answer at a
   checkpoint. Anything else is a surprise: the robot holds where it really is, queued work is cancelled,
   and the model gets an incident report instead of a stream of numbers.
5. **Idle means holding.** No idle timeouts that move. Holding still can heat motors and is not safe in every
   situation. A heat trip may use a home route the agent set, only if no contact has made it stale and the
   kernel is not faulted. Home routes contain only built-in motion and gripper steps, with no waits or contact
   operations. Completion includes torque release; a failed thermal return clears the route. Without a route
   the kernel holds and alarms for immediate operator action. The agent must resolve motor power before
   handing off asynchronously; software cannot safely release a raised, unsupported arm.
6. **Monitor unexpected contact.** Joint torque is compared with what the arm's own weight explains;
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

| | |
|---|---|
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

Keep-out geometry is a coarse link model: segments from joint to joint, padded by the manifest's `link_radius_m`
(3 cm by default), widened while planning by how far a point can travel between samples, and checked again at
every measured tick. It is not mesh collision: the base pedestal, fingers, payloads and self-collision are missing,
and surfaces constrain only the tool point. Slow zones cap the planned tool speed.

Faults latch until an operator resets them, and a watchdog trip while idle cancels anything queued. An incomplete
power transition, or failed I/O while powered, marks motor power unconfirmed (`power_uncertain`): commands stay
suspended, even across reconnection, until a release succeeds at a freshly measured rest pose. Status reports
feedback age and read errors and marks cached values stale.

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

The core uses MuJoCo for simulation, numpy for planning and Pillow for image overlays. MuJoCo renders the same scene
it simulates, with the robot's meshes, and grasps use frictional contacts. Rehearsals run on the kernel's estimated
world, never on the simulation's truth; servo parameters and motor heating are approximate. Physics, planning and
rehearsal need neither ROS nor a GPU; rendering simulated cameras needs OpenGL, from a GPU or from EGL or OSMesa on
a headless host ([simulation](docs/simulation.md)). The optional Rerun viewer reads flight records in its own
process; it never imports a hardware driver or checks a plan.

Perception turns pixels into numbers a plan can use. The daemon keeps the last few camera frames, unannotated
and unrecorded (`look` is the recorded, annotated view); a MuJoCo frame can add aligned metric depth and the
calibration the camera had. Measuring a point or a box of a frame gives the visible surface's centre and extent in
the work frame and its offset from the tool, and saves the picture and points with the run. A run can require
measurements: before each step the kernel refuses once one is too old for the run, its camera was calibrated
again, or a tracker withdrew it, judging only numbers copied when the run was accepted. Measurements never change
the world model; checking an outcome means measuring again and comparing, in the agent's procedure. Optional
EdgeTAM tracking runs in its own process, never in the control loop, and returns the same measurements. See
[measuring from pictures](docs/perception.md).

## Where it goes

Start with repeatable tasks on one arm. The [block example](examples/pick-place) gives new users a complete run, a
separate success check, controlled scene variations, and a recovery to inspect. More useful tasks and a second
concrete adapter will test the design better than a larger abstraction layer.

Near-term work should follow experiments:

- Improve camera setup and estimation where users lose time getting a trustworthy scene.
- Fit the twin from recorded hardware behavior. `wu fit` already estimates link masses, centres of mass and
  friction; better contact, actuator and thermal parameters need measurements to justify them. On the reBot that
  means fitting the torque its position loop applies rather than the reported one, a heating model per motor, and
  contact found from tracking lag, which is smoother than torque ([hardware records](docs/hardware.md)).
- Compare model-authored plans and live checkpoint decisions on the same tasks, keeping prompts, observations,
  interventions and measured outcomes with the records.
- Run the [perception example](examples/perception) over held-out positions and appearances, then a second task
  such as opening a door. Add text grounding, point tracking or grasp proposals when those tasks need them, and
  physical RGB-D once a sensor's calibration and timing are measured.
- Try two arms in simulation when a task needs cooperation, before introducing scheduling or a multi-robot graph
  API.

Branches, retries, reusable skills and learned policies belong where repeated tasks need them. A hosted service, a
broad plugin system and a large benchmark suite are later choices, not prerequisites for a useful robot harness.
