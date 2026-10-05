# Hardware records

Three sessions on one Seeed reBot Arm B601-RS (six joints, no brakes), driven through `wu`. Each section says what
ran, what the flight records showed and what changed because of it. These are development records on one arm, not
benchmarks.

## 2026-09-27: first runs

world-use v0.2 drove the arm for the first time: Claude as the policy, through the `wu` command line only, on a Mac
(M4 Max) with the arm clamped to a small cart beside a desk. The cameras were a C920 on the wrist, a C920 watching
from the side and an Insta360 X5, all written to files by a capture app. The numbers below come from each run's
flight record. The first four runs were on v0.2 and on patches made during the session; the last three were on the
fixes described below.

### What ran

| run | powered | moving | elbow peak | what |
|---|---|---|---|---|
| 18:41 | 15 s | 87% | 37 C | lift 6 cm, reach 8 cm, fold home, release |
| 18:51 | 339 s | 39% | 52 C | pick a roll of masking tape by its wall, lift it, put it back (v0.2) |
| 19:15 | 127 s | 66% | 43 C | the same pick and place with the first patches |
| 19:20 | 111 s | 45% | 44 C | touch the roll's top with default contact settings |
| 22:13 | 84 s | 63% | 36 C | a 10 cm lift off the rest stops, a reach, home off the stops, gripper back as found |
| 23:12 | 242 s | 7% | 53 C | lift off the stops, rehearsed calibration tours, home |
| 23:21 | 237 s | 18% | 66 C | `wu calibrate side` end to end, `move_to` with `point: "down"`, home |

The roll (124 mm across, 76 mm hole, ~45 mm tall) is wider than the jaws open, so it is held by its wall: nose down
over the far side of the ring, one finger in the hole. The first free-space move landed 1 mm short and 2 mm low of
the plan. Both picks lifted the roll and put it back where it was.

### What world-use got right

- **The rehearsal caught the elbow swinging into the desk.** Turning the base towards the roll from a low lift
  swings the folded elbow, 21 cm behind the base, 20 cm sideways into the desk's keep-out zone, 6 cm below the desk
  top. The whole plan was refused before anything moved; lifting 14 cm first carried the elbow over.
- **Surprises held and said why.** A grip on nothing, a set-down with no contact and five contact stops all left the
  arm holding where it was, with an incident a policy could act on. Nothing moved on its own.
- **Checkpoints did their job.** "Is the roll hanging from the gripper?" and "is it flat where it was?" were answered
  from the side camera before the next step.
- **Moving share.** Planning a phase at a time kept the arm moving for 39-66% of its powered time in the pick and
  place runs.

### What went wrong, and what changed

**The daemon called the CAN driver from two threads.** `wu enable` ran the adapter's one-second engage on the
request thread while the control thread kept reading the body. motorbridge is a ctypes wrapper without locks, and a
start pose misread during an engage is a jump. Once the control loop runs, only its thread calls the body: `enable`
and `release` from other threads are posted to it and awaited, a body exception does not kill the loop, and after a
loop has died enable is refused while release still runs. A second hole: an engage that failed after the motors
were switched on left them on with nothing commanding them. The adapter switches every motor off again before the
error goes up. On the arm, `wu status` answered throughout the engage, and a `kill -TERM` at rest released through
the control thread and wrote the record.

**Contact sensing stopped on nothing.** All four contact stops in the 18:51 run touched nothing:

- The baseline was one reading, and the loaded shoulder and elbow read +-0.5-1 Nm from tick to tick while holding
  still. The baseline became the median of the last 0.3 s, no threshold sits below 3.5 times each joint's measured
  noise, and no job starts until that window has filled after torque-on. On the simulator with the same noise,
  free-air guarded moves stopped on nothing 15 times in 15 before and 0 after.
- The arm folds onto hard stops, which the manifest declares (`Rest.stops`). With torque off it sags into them: the
  elbow read 0.5 deg at rest but meets its stop at 1.2 deg when powered, so home pushed into the stop and every fold
  ended in "unexpected contact". Home folds to 0.02 rad off the stops, and a joint within 0.06 rad of its stop is not
  judged and is re-zeroed there. Afterwards, three lifts off the stops and three folds home: no stop.
- Over a long guarded move the torque strays from the gravity model by 1-3 Nm. That is friction and hysteresis, not
  mass. The 19:15 run's 10 cm guarded descent stopped on nothing after 0.9 cm. So guard only the last 2-3 cm, as the
  reBot's card and the [agent brief](../src/world_use/POLICY.md) say. Short guards found the roll's top twice at
  default settings.

**A held roll pivots instead of pushing back.** Set down with `touchdown`, the roll turned in the pinch and laid
itself flat, so no contact was felt, and the steps after it (opening included) were cancelled. The brief says to
set things down with `guarded` and `expect_contact: false` to the known height.

**`wu run` exited 0 after a surprise.** The policy chained `wu run ... && wu home`, and the arm went home with the
roll still in its jaws. A job command exits 0 only when the job is done: 4 for refused, surprise or stopped, 5
waiting at a checkpoint, 6 still running. `wu status --json` works as well as `wu --json status`.

**The gripper was left open past pi.** Home restored the joints but not the gripper, which was then switched off at
4.39 rad; the reBot's motors come back from a power cycle one turn low beyond pi. Home puts the gripper back as it
was found, unless it holds something. The kernel tracks holding even when the world has no box for the object. A
gripper trip inside a plan also ends the gripper step; before, it did so only for a lone step.

**Grip squeezed past its own watchdog.** The default 0.1 rad squeeze is 5 Nm at the reBot gripper's kp of 50, over
its 4 Nm limit. The squeeze comes from the gripper spec; the reBot's is 0.05 rad.

**Rehearsals stalled the control loop.** Each `wu run` rehearsed on a twin in the daemon's process, competing with
the 100 Hz loop for the interpreter: ticks of 140-300 ms while the arm held a raised pose. Rehearsals and reach
probes moved to a worker process that works from a snapshot of the kernel. In the 23:21 run, with about 20
rehearsals, a calibration job and 10 camera pictures while powered, the tick was 10.0 ms median and 11.9 ms p99.
The longest powered ticks, 30-38 ms, are single ticks where a step starts and plans its motion. Engage and release
block the loop for 1.5-2.5 s by design: nothing is being controlled then. In the 23:12 run the powered p99 was
37 ms, with bursts of 40-136 ms ticks in the first minute after torque-on. They line up with no daemon activity, so
they are not explained yet.

**Cameras.** The side camera and the X5 were moved or knocked over several times in the day. A camera can be a file
a capture app keeps writing, refused when the frame is older than 3 s; that caught the X5's capture having stalled
at the start of one session. The X5 serves pinhole cuts of its 360 aimed by yaw and pitch (28 ms a frame).
`wu calibrate CAMERA` finds a camera's pose from the arm and the policy's answers.

**Records and heat.** The flight record can be written mid-session (`wu record`), and its summary reports the
control period (`tick_ms`). The heat forecast leaves out the first 20 s after torque-on: the elbow's reading rose 28
to 37 C in 12 s and then flattened, so the first forecasts read "1 min to 80 C".

**Install.** motorbridge 0.5.5 has no macOS wheel for Python 3.14, so the reBot extra needs Python 3.11 to 3.13
(see [the setup guide](rebot.md)).

### Checks after the fixes (23:12 and 23:21)

- Lifted 7-10 cm off the rest stops straight after torque-on: exit 0, no contact stop.
- `wu calibrate side` from the arm. A tour only fit in a narrow spot: moving the tool right swings the elbow into the
  desk's keep-out zone, so the box had to sit 42 cm out, where the elbow carries 8-9 Nm. The tour visited six
  corners, and the policy answered each from `wu look side --grid`. Four placed the tool, and the two on the upper
  layer (U+0.40) were above the picture. The fit refused rather than guess ("only 4 answers placed the tool in the
  picture; at least 6 are needed"), and nothing was installed. The flow works on hardware; this camera needs a lower
  or flatter tour.
- `move_to` with `point: "down"` was refused at every target tried: from 42 cm out (the wrist gets 54 of 88 deg), at
  U+0.27 to +0.30 (58-62 deg), and at U+0.16 to +0.18 (81-88 deg, but the last 0.1-1.8 cm of the move out of reach).
  The card's note puts nose-down at U+0.04 to +0.12 and F+0.14 to +0.26; the elbow reached 65 C before a target in
  that range was tried.
- Home folded off the stops and set the gripper back each time; `wu release` after home.

## 2026-09-28: a model fitted from flight records

Goldberg's essay on agentic robotics argues that the simulation a robot program is tuned in can be fitted from
observations of the real robot, and that a real failure then becomes evidence for the next fit. Video cannot show a
mass; joint torque can. So we fitted the reBot's twin from the day's flight records, wrote down what it predicted
for a tour it had never seen, ran the tour, and then ran it again with the fitted model in the loop.

The arm was driven from a Mac (M4 Pro) through `wu` only. Cameras: a C920 in front of the arm, a C920 on the wrist
and an Insta360 X5. The X5's USB webcam mode gives its two lenses stacked in one 1920x1080 picture, not an
equirectangular one, so the 360 cuts (`projection = "equirect"`) do not apply to it; it served as a plain camera
and filmed both runs.

### What ran

| run | powered | moving | elbow | what |
|---|---|---|---|---|
| 19:09 | 116 s | 61% | 25 -> 44 C | the tour, gravity feedforward from the URDF |
| 19:28 | 92 s | 77% | 26 -> 49 C | the same tour with the fitted model in the loop |

The tour: up to a raised pose, then 19 steps at 0.15-0.3 rad/s with 2 s holds. The steps cover the elbow up to
70 deg, the shoulder back to where it nearly balances, the wrist pitched up 10 deg, wrist yaw +-35 deg, a 90 deg
roll and a reach forward, then home to the rest pose. Both runs ended with every step done, no contact stop and no
trip. The second run had no pause for a look at the raised pose, hence the higher moving share.

### The fit

`wu fit` on the six earlier records of the day (about 1,000 s powered, 2,163 joint samples). Each record was
predicted by a fit made without it:

| joint | URDF, Nm rms | fitted |
|---|---|---|
| base | 0.65 | 0.27 |
| shoulder | 0.60 | 0.34 |
| elbow | 0.90 | 0.43 |
| wrist pitch | 0.28 | 0.18 |
| wrist yaw | 0.09 | 0.09 |
| wrist roll | 0.24 | 0.22 |

It found friction the URDF does not have (Coulomb: base 0.60 Nm, elbow 0.37, wrist pitch 0.23). It also found the
forearm lighter than the URDF says (1.25 -> 1.15 kg), with its centre of mass 26 mm further out. The wrist and
gripper came out heavier (0.46 -> 0.58 kg, 0.65 -> 0.69 kg). Together the arm holds more at the elbow than the URDF
says, as the hardware notes already reported. Most of the moving error was friction, not mass.

**On the tour it had not seen**, `wu fit`'s model missed the elbow's torque by 0.45 Nm rms against the URDF's 0.91.
The shoulder went from 0.58 to 0.41 and the wrist pitch from 0.23 to 0.12, much as its own check had forecast.
Before the tour we also wrote down per-step torques from a scratch fit of the same kind, on the torque the position
loop applies. The URDF missed the elbow by 0.62 Nm at holds and 0.71 while moving; that fit missed it by 0.22 and
0.11. Wrist yaw did not improve: no earlier record had turned it. The tour did, and its record informs the next fit.

![Elbow torque over the 19:09 tour, measured against the URDF and the fitted model; tool height at each hold with
each feedforward](assets/hardware-2026-09-28-fit.png)

### The fitted model in the loop

With `fit = "..."` in the workcell, the kernel judges contact against the fitted gravity plus friction at the
commanded velocity. The reBot's feedforward carries the fitted gravity, and rehearsals run on a twin that weighs
the links and feels friction the same way.

- **Sag at holds halved.** With the URDF the tool ended 1.6-4.3 mm low at every hold, 3.1 mm off on average (the
  elbow stopped 2.7-7.5 mrad short of its command). With the fit it ended 0.4-2.8 mm off, 1.6 mm on average, and
  high at nine holds of ten.
- **It errs high because the torque readings are biased.** At holds the elbow reports 0.23 +- 0.03 Nm more than its
  position loop applies, and the shoulder 0.18 Nm less. A fit on the reported torque carries that bias into the
  feedforward.
- **The contact signal steadied at the elbow only.** The largest excursion of its residual during a healthy move
  fell from 1.59 to 1.19 Nm, against a 3 Nm threshold. The other joints did not change: there, the reported
  torque's tick-to-tick noise sets the floor, not the model.
- **Folded at rest, the elbow carried 7.8 Nm instead of 7.1.** The better model lifts load off the rest stop that
  used to take it, and the elbow heats accordingly.

### What went wrong

**Rehearsal forecast half the heat the arm measured.** It forecast the elbow at +8.8 C (URDF twin) and +10.9 C
(fitted twin). The arm measured +19 C and +23 C. Four gaps add up:

- the plan's torques, which the fit only partly fixes;
- powered time outside the plan: switching on, a look with the cameras, going home;
- a step of 3-8 C in the first 10 s after switching on;
- heating that is not a function of torque alone: at similar loads the day's records show the elbow gaining 2 to
  8 C per 30 s.

The wrist pitch rose 10 C against a forecast of 0.2 C. The twin uses one heating constant for every motor, and the
wrist's RS-00 is a much smaller motor than the elbow's RS-06. The heat forecast remains approximate.

**Camera indices shift.** Plugging in the X5 moved the C920s from indices 0 and 1 to 1 and 2. A workcell that names
cameras by index then shows the wrong picture.

## 2026-09-29: a prolonged hold and a blocked thermal return

Three development trials ran on one Seeed reBot Arm B601-RS using runtime commit `68ac0a8`:

| Trial | Observed result |
| --- | --- |
| Free-space lift, reach, and empty-gripper motion | Five steps completed in 6.03 s, followed by home and release. |
| Three small rectangles | Sixteen steps completed in 21.74 s, followed by home and release. Encoder/URDF-FK endpoint spread was 0.241 mm; this was not an external accuracy measurement. |
| Expected missed grasp | Empty jaws produced a surprise at step two. The following reach was cancelled. |

The third trial left the arm raised. Objects were then rearranged beneath it. The assistant requested clearance,
inserted a confirmation checkpoint into the home route, and ended its turn while the arm remained energized.
That was an operating failure: a pending chat reply is not a safe state for a powered arm.

At session t+3197 s, the elbow reported 81 C. The thermal return started but waited at that checkpoint.
It remained powered for over two hours after the missed grasp. At t+8836 s, the CAN connection failed;
the last reported elbow temperature was 88 C. Later status repeated cached measurements and incorrectly
described the faulted arm as holding with confirmed power state.

The operator clarified that USB had been unplugged, which removed computer control but left the motor supply
on. The operator subsequently confirmed physical power-off. No software torque-off was confirmed during this
recovery. Damage was not assessed. These trials do not establish safe unattended operation.

### Corrections

- Home routes reject checkpoints, holds, contact steps, and arbitrary plugin behaviors, including inside
  nested sequences. Routes are copied and checked again before use. `null` explicitly clears a changed route.
- Thermal recovery is tracked by job identity, not its editable label. A successful return releases torque;
  a failed return clears its route and raises an alarm instead of retrying automatically.
- Read or command failures while powered latch unconfirmed power and suspend commands across reconnection.
  Stale feedback is identified with its age, and stale heat forecasts are suppressed.
- The operating instructions require confirmed torque-off before asynchronous handoff. If returning is
  blocked, immediate operator-assisted support and physical motor-supply shutdown are necessary.

Regression tests use simulation and a fake CAN driver. No physical robot was used to validate these changes.
Software cannot remove motor power through a disconnected USB adapter, nor safely drop an unsupported arm.
