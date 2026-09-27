# Design

world-use is the layer between a frontier model and a robot. The model decides; world-use keeps the robot
safe, spends as little of its powered time as possible waiting, and turns what happens into context the model
can use. This document explains why it is built the way it is.

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

These are single-task, single-robot observations, not benchmarks. They are why the rules below exist.

## Rules

1. **Two clocks.** The robot runs at 100 Hz or faster; the model at a few seconds per decision. The model
   decides phases, targets and skills; the kernel runs everything below that. Think with the torque off.
2. **Plans are data, checked before they run.** Code builds a plan; the plan is JSON; the same kernel runs
   it on a twin first (`check`) and on the robot second. Refusals happen before torque, not halfway.
3. **Every step says what it expects.** Contact within a distance, a grip within a width, an answer at a
   checkpoint. Anything else is a surprise: the robot holds where it really is, queued work is cancelled,
   and the model gets an incident report instead of a stream of numbers.
4. **Holding still is the safe state, and nothing moves on its own.** No idle timeouts that move. The only
   automatic motion is going home along a route the policy set, and only if nothing has been touched since.
5. **Every motion stops on unexpected contact.** Joint torque is compared with what the arm's own weight
   explains; guarded moves use tighter thresholds, fragile zones tighter still.
6. **Facts remember their source.** A door angle, a table height, a camera pose: recorded with where they
   came from, and marked stale when a surprise shows the world may have changed.
7. **The robot process outlives the agent.** One daemon per robot. Agents, consoles and viewers are clients;
   a crashed or interrupted agent leaves a robot that is holding still, not one that is mid-motion or dead.
8. **Context is a budget.** One state line per step. Events, not sensor streams. The embodiment card once.
   Incidents with the expected and the observed side by side.
9. **Failures are evidence.** Every run keeps a flight record (tape, events, world, summary) so the next
   attempt, or the next plan written offline, starts from what actually happened.

## The core

Six concepts. Everything else is a plugin.

| | |
|---|---|
| **Body** | What the robot is (a manifest: joints, limits, gripper, rest pose, sensing, notes) and a five-method adapter: connect, enable, read, command, disable, close. |
| **World** | Frames, boxes (surfaces, objects, keep-out, fragile and slow zones) and facts with their sources. |
| **Behavior** | Anything that moves the robot, under one contract: `start` plans and may refuse; `tick` runs one control step and returns an outcome when done. A line, a guarded touchdown, a grip, a checkpoint and a whole plan are all behaviors. |
| **Event** | One numbered stream of everything that happened. |
| **View** | State rendered for a reader: the state line, the status, an incident, the embodiment card. |
| **Kernel** | The only code that talks to the body. Each tick: read, watch, advance the behavior, command, record. |

The envelope inside the kernel holds the limits: joint limits with a margin, speed and acceleration, the
excursion from the session's start pose, gravity load, keep-out zones, surfaces, and robot-specific rules
(the reBot may not turn its base while the gripper is at table height). A policy can tighten it; loosening it
is an operator override with a reason, and it is logged.

## Two loops, one runtime

Agentic Robotics work such as [Graph-as-Policy](https://arxiv.org/abs/2607.05369) shows the strongest results
when agents write, test and improve robot programs offline in simulation and export lightweight code. Our
numbers agree: the frontier model does not belong in the control loop, and most of our gains came from moving
work offline. Our runs also show the other half: a one-off task in a scene nobody has modelled (a magnetic
door, a zip tie, a round neck that slips) is cheaper to supervise once than to simulate, and the remaining
failures only showed up live.

So world-use serves both loops with the same kernel, behaviors and records:

- **Offline**: an agent writes a plan, rehearses it on a twin, evaluates it, and ships it as data the kernel
  executes. No model runs while the robot moves.
- **Online**: the same plans run with checkpoints and surprise handling, and a model answers the questions
  and decides what to do after a surprise.
- **Between them**: online runs leave flight records; those records tune the twin and become test cases for
  the next offline iteration.

We do not require ROS, a GPU simulator or an industrial arm. The core is numpy and the standard library, so a
laptop and a low-cost arm are enough.

## Where it goes

Near term, in order:

1. **Graphs.** Plans grow branches on outcomes, retries and tunable parameters. Every node's success rate and
   time go into the flight record, so failures point at a node.
2. **Success criteria and evaluation.** Checkable task predicates, and `wu eval`: run a plan many times on the
   twin with poses and heights varied, and report success rate, cycle time and the failing node.
3. **A physics twin fitted from real runs.** A MuJoCo body alongside the kinematic one, with parameters an
   agent tunes by replaying recorded tapes: mass errors, friction, contact thresholds from torque noise,
   heat rates, surface heights from touchdowns. Joint torques are evidence a video cannot give.
4. **Evidence packages.** Every incident bundled with its tape window, events, camera clip and plan node, and
   every run recording what the policy was given: tools, views, twin, and human interventions.
5. **Cameras and pointing.** Calibration checked at every session start, plans drawn onto real camera images
   before they run, and clicks in a calibrated image turned into positions.
6. **More ways in.** An MCP server beside the CLI, a console where a person can stop, nudge and approve, and
   skills compatible with the open robot-skill libraries growing around this work.

Further out: learned policies (VLAs) as behaviors, a streaming interface for models fast enough to steer
continuously, and flight records exported as training data.
