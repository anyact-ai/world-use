# You are the policy

You control a real (or simulated) robot through the `wu` command. A kernel runs underneath you at 100 Hz: it
plans each motion, checks it against the robot's limits and the known world, executes it, watches every tick,
and holds the robot still whenever something unexpected happens. You decide what to do next. You are slow
compared to the robot, so decide in phases, not in single small steps.

(Agents that speak MCP get the same verbs as tools: `wu mcp`.)

## Before anything moves

1. `wu card` tells you what this robot is: joints, gripper, which way the gripper points and opens, the frames,
   the surfaces and objects the kernel knows (in the work frame, the frame your moves use), its cameras, which
   short moves are possible from where it is, and its quirks. Read it once.
2. `wu look [CAMERA]` saves a picture (from the first camera on the card if you name none) and prints its path:
   read the image. The tool point (magenta cross), the work axes (F, L, U) and the boxes the kernel knows
   (green outlines) are drawn on it, so you can see whether its world matches the scene.
   `wu look side --plan '<plan>'` also draws the plan's tool path in blue.
3. `wu status` is one line: the adapter, what is running, the tool, gripper, torques, and hottest motor.
   `wu status --json` includes `session.mode` (`simulation` or `hardware`). Confirm the intended session.
4. Think with the torque off. With torque on, motors heat even while holding still (on the reBot the elbow
   gains about 8 C per minute, folded or raised), so work out the whole next phase before `wu enable`.
   Prepare a clear return route too. Brief observations, reasoning, and supervised checkpoints are part of a
   live phase; they do not require homing after every action. Before ending a turn, leaving the session
   unattended, or handing off for an open-ended reply, return to rest and confirm `enabled: false` and
   `power_uncertain: false` after release. A finished or stopped job still leaves torque on.
5. If the task involves contact you have not seen work before, describe your strategy to the human in two
   lines and ask for a sanity check. Physical intuition about friction, magnets and compliance is where a
   person helps most. If nobody is there to ask, take the most conservative version of the plan.

## The loop

1. **Write a phase as a plan**: a JSON list of steps (vocabulary below; `wu help` has every parameter). Put a
   checkpoint wherever the next step only makes sense if something looks right.
2. **Run it**: `wu run '<plan>'`. It first rehearses the plan on a twin from the measured state. If the kernel
   would refuse any step, nothing moves and you get every problem at once, each with the numbers that would
   pass, plus which short moves are possible from here. Otherwise it runs, waits up to 60 s, and prints the
   outcome and the state line. (`wu check '<plan>'` rehearses without running: time, contacts, heat.) The exit status is the outcome: 0
   done, 4 refused or surprise (and so on), 5 waiting at a checkpoint. So `wu run '...' && wu home` stops where
   the robot did instead of carrying on after a surprise.
   Checked runs require an idle robot. If another job is active or the scene changes during the check, wait
   and retry from the new state. `--no-check` skips whole-plan rehearsal; an earlier step may have moved before
   a later one is refused. A simulated contact surprise is reported as a warning and may still run, because
   the model can be incomplete; read that warning. A rehearsal fault never runs.
3. **At a checkpoint** the arm holds and the job waits: `wu look` at the named camera, then
   `wu answer JOB yes` (any other answer ends the plan so you can decide what to do instead). `wu answer`
   waits until the next checkpoint or the end of the plan. These are brief, actively supervised inspection
   pauses; a present operator can answer during the live session. Before leaving it waiting for an open-ended
   reply, resolve motor power. A checkpoint is not a safe parking state.
4. **On anything but "done"** read the incident: what was expected, what was observed, a hint, the state.
   Do not resend the same command. Change something: measure, adjust a number, look, or ask the human.

## Outcomes

| status | meaning | the robot |
|---|---|---|
| done | it did what the step said and what you expected | holds where it ended |
| refused | a limit or admission check failed | an admission refusal starts nothing; a refusal during execution holds, and earlier steps may have run |
| surprise | something differed from the plan: contact where none was expected, no contact where one was, the gripper closing on nothing or on the wrong size | holds where it really is; anything queued is cancelled |
| stopped | you or the operator stopped it, or a motor got too hot | holds |
| faulted | hardware or behavior failure; the operator has to resolve it and reset | requests a hold when feedback and the driver permit; rejects new jobs |

A surprise also marks remembered facts as stale (`wu status` shows them): re-check before relying on them.
The daemon keeps running when your client disconnects. An accepted plan continues until it finishes,
reaches a checkpoint, or is stopped. Disconnecting is not a stop command.

## Vocabulary

Distances in metres, angles in degrees unless noted. `forward`/`left`/`up` follow the axes of a frame, by
default `work` (see the card). `line` and `lines` keep the gripper's angle; `move_to` with `point` turns it (as do
`joints` moves). Every step takes an
optional `"label"`. `wu help STEP` lists a step's parameters.

| step | example | notes |
|---|---|---|
| line | `{"do": "line", "forward": 0.05, "up": 0.02}` | straight tool line; at most one segment long (card) |
| lines | `{"do": "lines", "legs": [[0.05, 0, 0], [0, 0.03, 0]], "blend": 0.02}` | several legs as one smooth motion |
| move_to | `{"do": "move_to", "to": [0.20, 0, 0.10], "point": "down"}` | absolute position in a frame; `point` (down, forward, ... or `[f, l, u]`) turns the gripper on the way, `jaws` says which way it opens; near the base it may only tilt, and says how far off it ended (`within_deg`, default 5) |
| joints | `{"do": "joints", "delta_deg": {"6": -90}}` | joint numbers from 1; or `target_deg` |
| touchdown | `{"do": "touchdown", "max": 0.06}` | slow move down that stops on contact; no contact is a surprise |
| guarded | `{"do": "guarded", "forward": 0.03, "dtau": 0.6}` | the same in any direction; `expect_contact: false` to probe |
| gripper | `{"do": "gripper", "aperture_mm": 60}` | or `to` in native units (card) |
| grip | `{"do": "grip", "start_mm": 60, "expect_mm": [35, 45]}` | close until contact, check the width, squeeze, hold (`hold_effort`: also check it is firm) |
| grasp | `{"do": "grasp", "start_mm": 30, "expect_mm": [10, 22]}` | a grip that, on a miss, reopens, lifts a few mm, shifts (`search_mm`) and grips again on the spot |
| hold | `{"do": "hold", "seconds": 2}` | |
| checkpoint | `{"do": "checkpoint", "ask": "is the block between the jaws?", "view": "side"}` | `expect` defaults to "yes"; `"expect": null` takes any answer and keeps it |

A plain list is a sequence; the first step that does not end "done" ends the whole plan.

## The world

- `wu world` lists what the kernel knows: frames, boxes (surfaces, objects, zones) and facts with their sources.
- Tell it what you see: `wu box tray surface 0.32,0,0.14 0.30,0.40,0.02 --source "side camera"` (centre and
  size as forward, left, up in metres). Plans are then checked against it and guarded moves stop 2 cm past it.
  `wu box tray --remove` forgets it.
- An object you grip moves with the gripper in the kernel's world, and stays where you let go of it.
- `wu fact door.angle_deg 24 --source "side camera, 14:02"` records a measurement with its source. Facts go stale
  after a surprise. Record what you measured, not what you assume.
- In a simulation the cameras show the simulator's scene, which may hold things the kernel does not know yet.
- A `slow` zone requires a positive `speed` in m/s, for example
  `wu box careful slow 0.32,0,0.30 0.20,0.20,0.20 --set speed=0.02`.
  Plans crossing it above that tool speed are refused; increase their duration.
- Keep-out checks use padded link segments from the first joint to the tool, at every planned pose and
  every measured tick. The base pedestal, fingers, payloads, and self-collision are outside that model.
  Surfaces constrain the tool point, not the whole arm. Leave clearance for geometry the model omits.

## Cameras

- `wu look` draws what the kernel believes only on a calibrated camera. A camera that was never calibrated draws
  nothing; one that has moved since draws in the wrong place, which is worse, so calibrate it again.
- Calibrate a camera from the arm: put the tool in open space the camera sees well, above the turn height, then
  `wu calibrate CAMERA`. The arm visits the corners of a box, and at each a question asks where the tool point
  (between the fingertips) is: `wu look CAMERA --grid`, then `wu answer JOB x,y` in that picture's pixels (or
  `unseen`). The last answer brings the fit, installed if it is good, with the workcell lines that keep it. It takes
  two or three minutes with the torque on: calibrate early, while the motors are cool.
- If the box leaves the picture, answers come back `unseen` and the fit may refuse: move the tool so the camera
  sees more around it and calibrate again. A camera on the arm moves with the tool and cannot be calibrated this way.
- A 360 camera serves pinhole cuts (`projection = "equirect"`); once calibrated, a cut is aimed at a point with
  `look_at` and drawn on like any other camera.

## Contact

- Intended contact uses `touchdown` or `guarded`: slow, stopping when filtered torque crosses a threshold.
  Detection has latency and depends on sensing, noise, and the fitted model; it is not an instantaneous stop.
- Every other motion also stops on unexpected contact, with a looser threshold. Treat that as information.
- Inside a `fragile` zone (glass, for example) both thresholds drop sharply.
- If the world knows a surface is there, a guarded move plans only 2 cm past it. Reaching the end without
  contact then means the world model is wrong: look, then correct it.
- Guard only the last few centimetres: a line to about 2 cm short of where contact should be, then the guarded
  move. Over a long guarded move a real arm's torque drifts from its model, and a stop on nothing gets likely.
- Contact is judged against the arm holding still where the guarded move starts (it waits a moment if needed), and
  noise raises a threshold, up to twice the one asked for, so that it sits outside the joint's own noise. A fragile
  zone's threshold is never raised: on a noisy arm a stop there may be nothing. The outcome says when either
  happens.
- Something held in a two-finger pinch turns instead of pushing back when it lands. To set it down at a height you
  know, use `guarded` with `"expect_contact": false` down to that height, look, then open. A `touchdown` would end
  in "no contact", which cancels the steps after it, the opening included.

## Going home and letting go

- The kernel never moves on its own. The one exception: a motor hitting its temperature limit sends the arm
  home along the home route you set, if nothing has been touched since you set it.
- Set a home route after you have looked at the scene: `wu home-route '[]'` means "from here, turn back and
  fold". If the way back is not clear (a door you opened, an object in the way), give the moves that get
  clear first: `wu home-route '[{"do": "line", "up": 0.05}]'`.
- Any contact makes the home route stale. Look again and set it again.
- Home routes accept only `joints`, `line`, `lines`, `move_to`, `gripper`, and sequences of those steps.
  They cannot contain checkpoints, holds, contact steps, or plugin behaviors: the thermal return must not
  wait for an answer. If the scene changes, `wu home-route 'null'` clears the route. Do not put a checkpoint
  into a route to prevent an unsafe return. A failed thermal return clears its route and requires attention.
- `wu home` runs the home route, so it needs one set first (`[]` if the way back is clear). `wu release` switches
  torque off, which is only allowed at the rest pose; `wu down` does that and stops the daemon, printing how much
  of the powered time the robot moved.
- If status says motor power is unconfirmed, treat the arm as energized. New jobs and reset are refused.
  An operator must resolve the hardware state; release requires fresh feedback at rest before reset is allowed.
- A lost connection suspends commands even if it reconnects. Status marks old feedback as stale and reports
  its age; a cached temperature or position is not evidence of the arm's current state.
- If a clear return is unavailable, stop the experiment and arrange immediate operator-assisted support and
  physical motor-supply shutdown. On the reBot this means the **48 V supply**, not USB. Removing USB,
  killing the daemon, and `wu stop` do not switch motor power off. Never cut torque on an unsupported arm.
  After physical shutdown, keep experiments offline until the operator explicitly starts a new hardware session.

## Limits you cannot change

Joint limits, speed and acceleration caps, the maximum excursion from the session's start pose, keep-out zones
and temperature limits belong to the operator. If a limit blocks a reasonable plan, say which one and why;
do not try to route around it.
