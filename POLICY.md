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
3. `wu status` is one line: what is running, where the tool is, the gripper, joint torques, the hottest motor.
4. Think with the torque off. With torque on, motors heat even while holding still (on the reBot the elbow
   gains about 8 C per minute, folded or raised), so work out the whole next phase before `wu enable`.
5. If the task involves contact you have not seen work before, describe your strategy to the human in two
   lines and ask for a sanity check. Physical intuition about friction, magnets and compliance is where a
   person helps most. If nobody is there to ask, take the most conservative version of the plan.

## The loop

1. **Write a phase as a plan**: a JSON list of steps (vocabulary below; `wu help` has every parameter). Put a
   checkpoint wherever the next step only makes sense if something looks right.
2. **Run it**: `wu run '<plan>'`. It first rehearses the plan on a twin from the measured state. If the kernel
   would refuse any step, nothing moves and you get every problem at once, each with the numbers that would
   pass, plus which short moves are possible from here. Otherwise it runs, waits up to 60 s, and prints the
   outcome and the state line. (`wu check '<plan>'` rehearses without running: time, contacts, heat. After a
   check, `wu run --checked` runs that same plan without pasting it again.)
3. **At a checkpoint** the arm holds and the job waits: `wu look` at the named camera, then
   `wu answer JOB yes` (any other answer ends the plan so you can decide what to do instead). `wu answer`
   waits until the next checkpoint or the end of the plan.
4. **On anything but "done"** read the incident: what was expected, what was observed, a hint, the state.
   Do not resend the same command. Change something: measure, adjust a number, look, or ask the human.

## Outcomes

| status | meaning | the robot |
|---|---|---|
| done | it did what the step said and what you expected | holds where it ended |
| refused | a limit would have been broken; the message names each one and what would pass | never moved |
| surprise | something differed from the plan: contact where none was expected, no contact where one was, the gripper closing on nothing or on the wrong size | holds where it really is; anything queued is cancelled |
| stopped | you or the operator stopped it, or a motor got too hot | holds |
| faulted | hardware trouble; the operator has to reset it | holds |

A surprise also marks remembered facts as stale (`wu status` shows them): re-check before relying on them.

## Vocabulary

Distances in metres, angles in degrees unless noted. `forward`/`left`/`up` follow the axes of a frame, by
default `work` (see the card). Moves keep the gripper's angle; only `joints` changes it. Every step takes an
optional `"label"`. `wu help STEP` lists a step's parameters.

| step | example | notes |
|---|---|---|
| line | `{"do": "line", "forward": 0.05, "up": 0.02}` | straight tool line; at most one segment long (card) |
| lines | `{"do": "lines", "legs": [[0.05, 0, 0], [0, 0.03, 0]], "blend": 0.02}` | several legs as one smooth motion |
| move_to | `{"do": "move_to", "to": [0.35, -0.05, 0.30]}` | absolute position in a frame |
| joints | `{"do": "joints", "delta_deg": {"6": -90}}` | joint numbers from 1; or `target_deg` |
| touchdown | `{"do": "touchdown", "max": 0.06}` | slow move down that stops on contact; no contact is a surprise |
| guarded | `{"do": "guarded", "forward": 0.03, "dtau": 0.6}` | the same in any direction; `expect_contact: false` to probe |
| gripper | `{"do": "gripper", "aperture_mm": 60}` | or `to` in native units (card) |
| grip | `{"do": "grip", "start_mm": 60, "expect_mm": [35, 45]}` | close until contact, check the width, squeeze, hold |
| hold | `{"do": "hold", "seconds": 2}` | |
| checkpoint | `{"do": "checkpoint", "ask": "is the block between the jaws?", "view": "side"}` | `expect` defaults to "yes" |

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

## Contact

- Intended contact uses `touchdown` or `guarded`: slow, and stopped the moment the joints feel it.
- Every other motion also stops on unexpected contact, with a looser threshold. Treat that as information.
- Inside a `fragile` zone (glass, for example) both thresholds drop sharply.
- If the world knows a surface is there, a guarded move plans only 2 cm past it. Reaching the end without
  contact then means the world model is wrong: look, then correct it.

## Going home and letting go

- The kernel never moves on its own. The one exception: a motor hitting its temperature limit sends the arm
  home along the home route you set, if nothing has been touched since you set it.
- Set a home route after you have looked at the scene: `wu home-route '[]'` means "from here, turn back and
  fold". If the way back is not clear (a door you opened, an object in the way), give the moves that get
  clear first: `wu home-route '[{"do": "line", "up": 0.05}]'`.
- Any contact makes the home route stale. Look again and set it again.
- `wu home` runs the home route, so it needs one set first (`[]` if the way back is clear). `wu release` switches
  torque off, which is only allowed at the rest pose; `wu down` does that and stops the daemon, printing how much
  of the powered time the robot moved.

## Limits you cannot change

Joint limits, speed and acceleration caps, the maximum excursion from the session's start pose, keep-out zones
and temperature limits belong to the operator. If a limit blocks a reasonable plan, say which one and why;
do not try to route around it.
