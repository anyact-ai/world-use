<p align="center">
  <img src="docs/assets/anyact-arm.svg" width="150" alt="The AnyAct A as a small robot arm lifting its gripper">
</p>

# world-use

Run frontier models as robot policies.

A model like Claude or GPT decides what the robot should do. world-use keeps the robot safe while it does,
wastes as little powered time as possible waiting for the model, and hands the model short, checkable facts
instead of raw sensor streams. It runs on a laptop with a low-cost arm, or in simulation with no hardware.

Computer use gave models a screen and a mouse. This gives them an arm.

> Early (v0.1). Tested in simulation and against a faked motor driver. The reBot hardware adapter is ported
> from a toolkit that has run on the physical arm, but has not yet run on hardware through world-use.

## Try it in simulation

```sh
pip install "git+https://github.com/anyact-ai/world-use"      # or: uv tool install ...
wu up --body sim --enable            # a daemon that owns the (simulated) robot
wu card                              # what this robot is and can do
wu check '[{"do": "line", "forward": 0.08, "up": 0.06}, {"do": "hold", "seconds": 1}]'
wu run   '[{"do": "line", "forward": 0.08, "up": 0.06}, {"do": "hold", "seconds": 1}]'
wu status
wu home-route '[]' && wu home        # fold back along a route you have checked
wu down                              # release at rest, write the flight record
```

`wu check` rehearses on a twin from the robot's measured state and reports refusals, contacts, time and heat
before anything moves. `wu run` prints the outcome and one state line:

```
job 1 done: line(forward=0.08, up=0.06)
t+8s | idle, holding | tool F+0.382 L+0.000 U+0.278 | grip 0.05rad (0mm) +0.0 | tau +0.0 -1.4 +7.1 +2.0 -0.0 -0.0 | hottest j3 26C (7.5 min to 80C, j3)
```

## Let a model drive

Point your agent (Claude Code, Codex, anything with a shell) at [POLICY.md](POLICY.md). It explains the loop:
read the card, plan a phase, check it, run it, answer checkpoints, and handle surprises. The daemon holds the
robot between the agent's tool calls, so a slow, interrupted or restarted agent leaves the robot holding
still.

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

- **Checks the whole motion before it starts**: joint limits, speed, acceleration, reach, how far each joint
  strays from where the session started, gravity load, keep-out zones, known surfaces. A refused command
  moves nothing and says which limit and what would pass.
- **Stops on contact**: guarded moves (`touchdown`, `guarded`) stop the moment the joints feel something;
  every other move stops on unexpected contact; `fragile` zones use tighter thresholds.
- **Holds on surprise**: when what happened differs from what a step expected, the arm holds where it
  really is, queued steps are cancelled, and the model gets an incident report.
- **Never moves on its own**: no idle motion. The only automatic move is going home along a route the policy
  set when a motor overheats, and only if nothing has been touched since the route was set.
- **Tracks heat**: minutes until the hottest motor reaches its limit, in every state line.
- **Records everything**: tape at the control rate, events, world, and a summary with how much of the
  powered time the robot actually moved.

It is a helper, not a certified safety system. Keep a person at the power switch.

## Robots

| body | status |
|---|---|
| `sim` | kinematic twin of any manifest: gravity torques, servo stiffness, surfaces that push back, objects that stop the gripper, motor heating |
| `rebot` | Seeed reBot Arm B601-RS over CAN (`pip install "world-use[rebot]"`); ported from a toolkit that has run on the arm |

A new arm needs a manifest (joints, limits, gripper, rest pose, what it senses) and an adapter with five
methods. See [body.py](src/world_use/body.py) and [the reBot adapter](src/world_use/bodies/rebot/__init__.py).

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
