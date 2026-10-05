<p align="center">
  <img src="docs/assets/anyact-arm.svg" width="130" alt="AnyAct robot arm">
</p>

# world-use

**Robot control for AI agents.**

An agent looks at the scene and writes a plan. world-use checks the motion, runs it,
and reports what happened. A persistent runtime handles the control loop, contact
monitoring, and recording between agent calls.

Start with one arm and a laptop. Use the CLI, Python, or MCP; the same plan format
works in the included simulator and on a Seeed reBot.

Early development. Tested in simulation and on one physical arm. The
[hardware records](docs/hardware-2026-09-27.md) describe what was tried and measured.

## Try it

Requires Python 3.13+. [uv](https://docs.astral.sh/uv/) can install it for you.

```sh
uv tool install --python 3.13 "git+https://github.com/anyact-ai/world-use"
wu demo --out runs/block-demo
wu inspect runs/block-demo
```

This completes the scripted task locally and saves a GIF, plans, outcomes, and
telemetry. Try `--scenario missing` to see a failed grasp and recovery, or read the
[example](examples/pick-place) to adapt it.

For an interactive session:

```sh
wu up --workcell block
wu card
wu look side
wu enable
wu run '[{"do":"line","up":0.06},{"do":"gripper","aperture_mm":60}]'
wu home-route '[]' && wu home && wu down
```

`look` prints the path to an image. `run` rehearses a plan before submitting it;
`check` rehearses without running. A failed limit check starts nothing. The final
line uses this example's clear return path, releases at rest, and closes the session.

For an interactive 3D viewer with synchronized cameras, joint plots, and events:

```sh
uv tool install --force --python 3.13 "world-use[rerun] @ git+https://github.com/anyact-ai/world-use"
wu view runs/block-demo
wu view                       # follow the local daemon's recording
```

Rerun is optional and runs outside the control loop. The viewer shows measured
joints and the estimated world alongside saved camera observations. Closing it
does not stop a job. See [visualization and headless export](docs/visualization.md).

![MuJoCo block demo in Rerun: measured robot and estimated world, camera observation, joint plots, and events](docs/assets/simulation-rerun.png)

## Give it to an agent

Start the daemon and give your agent `wu policy`, the installed operating brief,
along with a task. The agent reads the robot's card, inspects the scene, and submits
a phase of work. Checkpoints pause for an answer; unexpected contact or grasp width
returns an incident. `wu help` lists the actions and their parameters.

For MCP clients:

```sh
uv tool install --python 3.13 "world-use[mcp] @ git+https://github.com/anyact-ai/world-use"
wu up --workcell block
wu mcp
```

Configure the client to launch `wu mcp` over stdio. It provides the operating brief,
camera images, structured status, job results, and the same actions as the CLI.
The daemon outlives agent calls. An accepted plan continues after a client disconnects;
use checkpoints where it needs an answer.

An agent can also measure what a camera sees, in metres in the work frame, and make a
plan stop once that measurement is too old; `wu mcp --vision` adds EdgeTAM tracking.
Depth comes from simulated cameras for now. See [measuring from pictures](docs/perception.md).

## Python

With an enabled daemon running, save the example below as `task.py`. The CLI
installation has an isolated environment; run the script with its own dependency:

```sh
uv run --python 3.13 --with "world-use @ git+https://github.com/anyact-ai/world-use" python task.py
```

```python
from world_use import Plan
from world_use.client import Client

robot = Client()
phase = Plan("lift and open").line(up=0.06).gripper(aperture_mm=60)
result = robot.run(phase.spec(), wait=60)
print(result["status"])
```

Plans are JSON data: save them, generate them in code, and inspect them before
execution. `Client.run` rehearses against the current state. Embedded `Kernel.run`
checks each step as it starts; call `check` explicitly for a whole-plan rehearsal.

`Client.frame`, `Client.measure` and `Client.run(..., requires=[...])` give Python the same
[measurements](docs/perception.md) as MCP.

## What the runtime provides

- Cartesian lines, joint moves, guarded contact, gripping, and checkpoints.
- Named frames, boxes, sourced facts, camera overlays, and calibration.
- Joint and motion limits, padded link keep-outs, torque and temperature monitoring.
- Records of plans, measurements, outcomes, observations, and elapsed timing.

```sh
wu record --note 'Adjusted the block estimate from the side camera'
wu inspect runs/YOUR-RUN
wu replay runs/YOUR-RUN --speed 3
```

Telemetry is saved in background chunks so an interrupted process still leaves
an inspectable record. [Record format and replay](docs/records.md) explain what is
preserved. `wu fit` can estimate link masses and joint friction from recorded runs.

The simulator uses MuJoCo with the real reBot meshes, rigid-body contacts,
frictional grasps, and rendered cameras. Objects can slip, tip, and fall.
Actuator and thermal models remain approximate. Planning still uses padded link
keep-outs and tool-point surface checks; a passing rehearsal depends on what the
world model knows. See [simulation setup and assumptions](docs/simulation.md).

For physical hardware, use the [reBot setup guide](docs/rebot.md). Keep an operator
at the motor-supply switch. A stopped job still holds with torque; a raised arm
without brakes cannot simply be released.

For another arm, supply a robot description and a small driver in your own package.
The runtime supports rotational arm joints (revolute and continuous); prismatic
joints are rejected. The same configuration feeds execution, simulation and
recorded replay. The [adapter guide](docs/adapters.md) includes a runnable two-joint example.

## Develop

```sh
git clone https://github.com/anyact-ai/world-use.git
cd world-use
uv sync --locked
uv run pytest
uv run ruff check .
uv run ty check src
```

See [adding a body](docs/adapters.md) for the complete adapter path, and the
[design notes](DESIGN.md) for runtime contracts and the experiments behind them.
The [changelog](CHANGELOG.md) covers release and API changes.

## License

Apache-2.0. The reBot URDF and meshes are Seeed Studio's, under CERN-OHL-W-2.0;
see the [notice](src/world_use/bodies/rebot/NOTICE.md).
