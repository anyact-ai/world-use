<p align="center">
  <img src="docs/assets/anyact-arm.svg" width="150" alt="The AnyAct A as a small robot arm lifting its gripper">
</p>

# world-use

Run frontier models as robot policies.

A model like Claude or GPT decides what the robot should do. world-use keeps the robot safe while it does,
wastes as little powered time as possible waiting for the model, and hands the model short, checkable facts
and pictures instead of raw sensor streams. It runs on a laptop with a low-cost arm, or in simulation with no
hardware.

Computer use gave models a screen and a mouse. This gives them an arm.

> Early (v0.2). Tested in simulation, against a faked motor driver, and on the physical reBot. Its first runs
> there, a pick and place with camera checkpoints, and what they changed are in
> [the record](docs/hardware-2026-09-27.md).

## Try it in simulation

```sh
uv tool install "git+https://github.com/anyact-ai/world-use"    # Python 3.14; uv fetches it if needed
wu up --workcell block --enable      # a daemon that owns a simulated arm, a block on a tray in front of it
wu card                              # what this robot is and can do
wu look side                         # a picture with the tool and the known boxes drawn on it; prints its path
wu run '[{"do": "line", "up": 0.06}, {"do": "gripper", "aperture_mm": 60}]'
wu home-route '[]' && wu home        # fold back along a route you have checked
wu down                              # release at rest, write the flight record
```

`wu run` rehearses the plan on a twin first. If the kernel would refuse any step, nothing moves, and the
model gets every problem at once with the numbers that would pass:

```
$ wu run '[{"do": "line", "up": 0.03}, {"do": "line", "left": 0.05}]'
refused in rehearsal, so nothing moved:
check FAILED: 1 limit would be broken, so the kernel would refuse this plan and nothing would move
  step 2/2: line(left=0.05): this turns j1/j5/j6 with the tool at U+0.247; turning needs U+0.267 or higher (5 cm above the start height), or the gripper sweeps across the table (hint: lift at least 2 cm more first)
  rehearsed past them: 4.5 s simulated, 4.5 s of it moving
  ends with the tool at F+0.302 L+0.063 U+0.247 (work)
  heat: j3 +0.6 C, to about 26 C
from here a 3 cm line can go up, forward; not down or back (out of reach with the gripper at this angle); left or right (turning needs the tool at U+0.267).
```

## Let a model drive

Point your agent (Claude Code, Codex, anything with a shell) at [POLICY.md](POLICY.md). It explains the loop:
read the card, look, plan a phase, run it, answer checkpoints, and handle surprises. The daemon holds the
robot between the agent's tool calls, so a slow, interrupted or restarted agent leaves the robot holding still.

Agents that speak MCP get the same verbs as tools, and pictures inline:

```sh
uv tool install "world-use[mcp] @ git+https://github.com/anyact-ai/world-use"
claude mcp add world-use -- wu mcp
```

The same thing from Python:

```python
from world_use import Kernel, Plan, VirtualClock, World, bodies, check

world = World()
k = Kernel(bodies.make("sim", world), world, VirtualClock(100))
k.connect(); k.enable()
world.add_box("table", "surface", center=[0.35, 0, 0.17], size=[1, 1, 0.02])

p = Plan("find the table").line(forward=0.08, up=0.06).touchdown(max=0.12)
print(check(p, k))          # rehearse on a twin
print(k.run(p.spec()))      # then run it
```

## What the kernel does for you

- **Checks the whole plan before it starts**: joint limits, speed, acceleration, reach, how far each joint
  strays from where the session started, gravity load, keep-out zones, known surfaces. A refused plan moves
  nothing and names every limit it would break, with what would pass.
- **Tells the model what it can do**: the card says which way the gripper points and opens, where known things
  are in the frame moves use, and which short moves are possible from the current pose.
- **Shows what it knows**: `wu look` draws the tool point, the known boxes and a planned path onto camera
  images, so a mismatch between the world model and the scene is visible at a glance.
- **Stops on contact**: guarded moves (`touchdown`, `guarded`) stop the moment the joints feel something;
  every other move stops on unexpected contact; `fragile` zones use tighter thresholds.
- **Holds on surprise**: when what happened differs from what a step expected, the arm holds where it
  really is, queued steps are cancelled, and the model gets an incident report.
- **Never moves on its own**: no idle motion. The only automatic move is going home along a route the policy
  set when a motor overheats, and only if nothing has been touched since the route was set.
- **Tracks heat and records everything**: minutes until the hottest motor reaches its limit, in every state
  line; tape at the control rate, events, pictures, world, and how much of the powered time the robot moved.
- **Learns the robot from its records**: `wu fit runs/*` fits the links' masses and the joints' friction from
  flight records and says how well that predicts runs it did not see. A workcell's `fit` line puts the result
  into contact checks, rehearsals and, on the reBot, the gravity feedforward.

It is a helper, not a certified safety system. Keep a person at the power switch.

## Robots

| body | status |
|---|---|
| `sim` | kinematic twin of any manifest: gravity torques, servo stiffness, surfaces that push back, objects that stop the gripper and ride along, motor heating, and cameras that render the scene |
| `rebot` | Seeed reBot Arm B601-RS over CAN (install the `rebot` extra, as with `mcp` above; on a Mac see [the driver note](#the-rebot-driver-on-a-mac)); has run on the arm |

A new arm needs a manifest (joints, limits, gripper, rest pose, what it senses) and an adapter with five
methods. See [body.py](src/world_use/body.py) and [the reBot adapter](src/world_use/bodies/rebot/__init__.py).
Cameras are an HTTP snapshot URL, a command that prints an image, or a file a capture app keeps writing (refused
when stale); a 360 camera serves pinhole cuts. `wu calibrate CAMERA` finds where one is from the arm itself; see
[cameras.py](src/world_use/cameras.py) and [calibrate.py](src/world_use/calibrate.py).

### The reBot driver on a Mac

Seeed's `motorbridge` 0.5.5 publishes no macOS wheel for Python 3.14, and its source build needs a prebuilt Rust
library, so installing the `rebot` extra fails there. The library is loaded through ctypes and does not depend on
the Python version: take it from the 3.13 wheel.

```sh
pip download motorbridge==0.5.5 --no-deps --python-version 3.13 --only-binary :all: -d /tmp/mb
unzip -o -q /tmp/mb/motorbridge-*.whl -d /tmp/mb/x
MOTORBRIDGE_LIB=/tmp/mb/x/motorbridge/lib/libmotor_abi.dylib \
MOTORBRIDGE_WS_GATEWAY_BIN=/tmp/mb/x/motorbridge/bin/ws_gateway pip install motorbridge==0.5.5
```

The CAN adapter also needs the MacCAN PCBUSB runtime (`libPCBUSB.dylib`), which motorbridge looks for in
`/usr/local/lib`, `/opt/homebrew/lib` or `~/.local/lib`.

## Why it is built this way

[DESIGN.md](DESIGN.md) has the measurements behind it. In short: on a real task, the arm spent 86% of its
powered time holding still while the model thought, and holding still overheated it. Planning offline and
running checked batches roughly tripled the share of powered time spent moving. So the model decides in
phases, the kernel runs everything below that, and most thinking happens with the torque off.

## Related work

- [Graph-as-Policy](https://arxiv.org/abs/2607.05369) (Berkeley and NVIDIA) has agents write and improve
  robot programs as graphs of skills, offline in simulation. world-use shares the view that the model does not
  belong in the control loop, and adds the runtime for low-cost hardware and live supervision.
- [Inspect Robots](https://github.com/robocurve/inspect-robots) evaluates LLMs and VLAs on many robots.

## License

Apache-2.0. The reBot URDF is Seeed Studio's, under CERN-OHL-W-2.0 ([notice](src/world_use/bodies/rebot/NOTICE.md)).
