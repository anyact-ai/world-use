# Pick up a block, move it, put it down

A first task for a simulated reBot. A tall block stands on a tray; the target is 10 cm to its right. Success means
the block stands within 1 cm of the target, preserves its initial upright orientation within 5 degrees, the jaws
have let go of it, and the tool has withdrawn at least 8 cm above its centre. The session then goes home and
switches torque off.

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

## Adapt the task

Keep the scene, procedure and evaluation separate. Start with this working variation: a smaller tray, a shifted
block and a different destination. Run it from an installed environment with `python`, or from a checkout with
`uv run python`:

```python
from world_use import check
from world_use.examples.pick_place import pickup, placement, setup, success

k, truth = setup()
target = [.33, -.06, .20]                    # work-frame metres
try:
    for scene in (k.world, truth):
        scene.add_box("tray", "surface", [.32, 0, .14], [.28, .36, .02], frame="work")
        scene.add_box("block", "object", [.35, .02, .20], [.04, .04, .10], frame="work")
    k.body.reset(k.state.q, k.state.gripper) # apply these initial poses without stepping physics
    k.enable()
    phase = pickup(k)
    assert check(phase, k).ok
    assert k.run(phase).ok
    phase = placement(k, target)
    assert check(phase, k).ok
    assert k.run(phase).ok
    result = success(k, truth, target)       # runner evaluation, after release and withdrawal
    print(result)
    k.set_home_route([], "adapted open tray")
    assert k.run(k.home_plan()).ok
    k.release()
finally:
    k.close()
```

This variation is covered by `tests/test_tasks.py`; it preserves the 4 cm grasp width and 10 cm block height.
Change one initial condition at a time before changing the procedure. `shifted`, `missing` and `misplaced`
already distinguish a changed known scene from a wrong estimate. For repeated trials, restore the intended
box poses, clear the estimated attachment with `k.world.held = None`, and call `SimBody.reset(q, gripper)` while
idle. Reset uses the current truth poses; it does not remember a previous trial's start or reset kernel faults,
measurements, clock or record. Fresh kernels and output folders give independent trials.

For a saved workcell, copy [block.toml](../../src/world_use/workcells/block.toml), keep `body = "sim"`, and edit
its `[[box]]` entries. Surfaces are fixed boxes; objects are free rigid boxes. Set `known = false` on an object
or surface obstruction to put it in physics without giving it to the kernel's world model. A policy's `add_box` changes
only that estimate. The reference pickup assumes a known block; use the [perception example](../perception)
when the procedure must locate an unknown object from images and depth. Its procedure receives MCP replies
and pictures, while its runner keeps truth for independent evaluation.

Compose phases from `line`, `move_to`, `grip` and `gripper`, checking them before execution as `run()` in the
source does. If width, height, orientation or surrounding clearance changes, revise the grasp expectation,
approach, placement and recovery together. Pass a new work-frame destination to `placement` and `success`;
for a different intended rotation, pass its 3×3 matrix and tolerance to `success`. That predicate owns task
criteria, outside the motion kernel. It checks position, full orientation, release and withdrawal; a `done`
job alone establishes none of these outcomes.

Use [the adapter guide](../../docs/adapters.md) to substitute another robot. The planar example demonstrates
rehearsal, execution and records for an external two-joint description; the five-joint jaw fixture in
`tests/test_mujoco.py` demonstrates a revolute gripper and Cartesian moves with free heading. These exercise
software interfaces, not hardware fidelity. [Simulation limits](../../docs/simulation.md#scene-and-model-assumptions)
describe which environment changes the native box workcell can represent.
