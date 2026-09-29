<p align="center">
  <img src="docs/assets/anyact-arm.svg" width="150" alt="AnyAct robot arm">
</p>

# world-use

**Robot control for AI agents.**

Give an agent a view of the scene and a small set of robot actions. It writes a plan;
world-use checks it, runs it, and reports what happened. The control loop stays in a
separate process, with joint limits, contact monitoring, and a record of every run.

Use it from a shell, Python, or MCP. Start in simulation, then use the same interface
with a Seeed reBot arm.

Early development. Tested in simulation and on one physical arm; see the
[hardware records](docs/hardware-2026-09-27.md).

## Try it

Requires Python 3.14+. [uv](https://docs.astral.sh/uv/) can install it for you.

```sh
uv tool install --python 3.14 "git+https://github.com/anyact-ai/world-use"
wu up --body sim --workcell block
wu card
wu look side
wu enable
wu run '[{"do": "line", "up": 0.06}, {"do": "gripper", "aperture_mm": 60}]'
wu home-route '[]' && wu home && wu down
```

This starts a simulated arm with a block on a tray, saves a side view, lifts the tool,
and opens the gripper. The last line follows the demo's clear return path, releases
at rest, and saves the run under `runs/`. `wu look` prints the path to an image you can open.

`wu run` rehearses the plan before submitting it. A limit violation refuses the plan
before its first step. A busy robot or a changed scene requires a fresh check.
Use `wu check '<plan>'` to rehearse without running, and `wu help` for the available actions.

## Give it to an agent

Start the daemon, then give your agent [POLICY.md](POLICY.md). The agent reads the robot's
card, looks at the scene, and submits a phase of work. Checkpoints pause for a visual
check; unexpected outcomes return an incident with what was expected and what happened.

For MCP clients:

```sh
uv tool install --python 3.14 "world-use[mcp] @ git+https://github.com/anyact-ai/world-use"
wu mcp
```

Configure the client to launch `wu mcp` over stdio. It exposes the same actions as the
CLI and returns camera images inline. The daemon continues serving between agent calls.
An accepted plan continues if the client disconnects; use checkpoints where it needs an answer.

## Python

Add the package to a Python 3.14+ project with
`uv add "git+https://github.com/anyact-ai/world-use"`. With an enabled daemon running:

```python
from world_use import Plan
from world_use.client import Client

robot = Client()
plan = Plan("lift and open").line(up=0.06).gripper(aperture_mm=60)

print(robot.check(plan.spec())["text"])
result = robot.run(plan.spec(), wait=60)
print(result["status"])
```

`Client.run` checks again against the current state. Plans are JSON data, so you can
save them, generate them in code, and inspect them before execution. `Kernel` is also
available for embedded use; its `run` method checks each step as it starts. Use `check`
explicitly when working at that level.

## What is here

- **Motion:** Cartesian lines, joint moves, guarded contact, gripping, and checkpoints.
- **Scene context:** named frames, boxes, sourced facts, camera overlays, and calibration.
- **Runtime checks:** motion limits, padded link keep-outs, planned tool-speed limits in
  slow zones, torque and temperature monitoring, and fault handling.
- **Flight records:** commands, measurements, events, images, and elapsed timing. `wu fit`
  estimates link masses and joint friction from those records for the next run.

The included simulator is a kinematic twin with approximate contact and heating.
It does not model slipping, tipping, or general rigid-body dynamics. Keep-out checks
use padded joint-to-joint segments; fingers, payloads, and self-collision need separate
clearance checks. Surface checks use the tool point. A passing rehearsal depends on
what the world model knows.

For physical hardware, read the [reBot setup guide](docs/rebot.md). This is experimental
robot software, not a certified safety system; keep a person at the power switch.

## Develop

```sh
git clone https://github.com/anyact-ai/world-use.git
cd world-use
uv sync --locked
uv run pytest
uv run ruff check .
uv run ty check src
```

A new robot needs a manifest and six I/O methods. Start with the
[body contract](src/world_use/body.py) and [reBot adapter](src/world_use/bodies/rebot/__init__.py).
The [design notes](DESIGN.md) cover the experiments, runtime contracts, and next steps.

The project takes inspiration from [Graph-as-Policy](https://arxiv.org/abs/2607.05369):
write and test robot programs outside the control loop, then improve them from execution evidence.
[Inspect Robots](https://github.com/robocurve/inspect-robots) is related work on evaluating robot policies.

## License

Apache-2.0. The reBot URDF is Seeed Studio's, under CERN-OHL-W-2.0;
see the [notice](src/world_use/bodies/rebot/NOTICE.md).
