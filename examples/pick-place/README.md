# Pick up a block, move it, put it down

A first task for a simulated reBot. A tall block stands on a tray; the target is 10 cm to its right. Success means
the block stands within 1 cm of the target, the jaws have let go of it, and the tool has withdrawn at least 8 cm
above its centre. The session then goes home and switches torque off.

## Run the reference

```sh
wu demo
wu inspect runs/block-demo
wu replay runs/block-demo --out runs/block-replay.gif
```

`wu demo` runs a script in place of an agent, in MuJoCo: no model calls, no daemon and no hardware. Each run
writes a new folder under `./runs` (`runs/block-demo`, then `runs/block-demo-2`, ...), or the empty folder given
with `--out`. Its `demo.gif` shows the simulation's own camera at 3x speed; the same frames are saved under
`views/`, and `wu view` shows them with the robot, the world model and the telemetry. `wu replay` instead redraws
the recorded joints and world model, and labels its frames as a reconstruction.

The [source](../../src/world_use/examples/pick_place.py) builds two plans, pickup and placement, and checks each
before it runs. The grip must close on the expected width, but a plan that ends "done" does not prove the task: the
success check reads the simulator's truth separately.

## Give the task to an agent

```sh
wu up --workcell block
```

Give the agent [this task](TASK.md) and let it read `wu policy`, look at the scene and write its own plans. The
reference shows the API; it is not evidence of what an unaided model can do. Record the model, its inputs and any
help it gets:

```sh
wu record --context '{"model":"your-model","task":"block transfer"}'
wu record --note 'Operator corrected the block position after inspecting the side camera'
```

## Try a failure

```sh
wu demo --scenario shifted --no-video
wu demo --scenario missing --no-video
wu demo --scenario misplaced --no-video
```

`shifted` moves the block 1 cm forward and right, in the simulator and in the robot's world model alike. `missing`
removes it from the simulator only, and `misplaced` moves it 9 cm left in the simulator only. Those two exit with
status 4: the grip closes on nothing. The script then forgets the block, opens, lifts away and goes home; it does
not retry the same guess. That way out is safe only over this clear tray.

Grasps use rigid-body contacts and friction, while servo gains, friction and heating are approximate. These
scenarios check the software, not what will work on hardware; see [simulation](../../docs/simulation.md).
