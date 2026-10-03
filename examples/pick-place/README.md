# Pick up a block, move it, put it down

A complete first task for a simulated reBot. The block starts on a tray; the target
is 10 cm to its right. Success means the block is within 1 cm of the target, the
jaws have released it, and the tool has withdrawn at least 8 cm above its centre.
The session then returns to rest and switches torque off.

## Run the reference

```sh
wu demo --out runs/block-demo
wu inspect runs/block-demo
wu replay runs/block-demo --out runs/block-replay.gif
```

`wu demo` runs a **scripted policy** in the MuJoCo simulator. It makes
no model API calls and does not connect to a running daemon or physical hardware.
Its GIF contains frames captured during that simulation, at 3× playback. The same
frames are saved under `views/` with timestamps in the event log. With the
optional Rerun extra, `wu view runs/block-demo` shows them alongside the robot,
world estimates, telemetry, and events on a scrubbable timeline.
`wu replay` separately reconstructs the recorded joints and world model; those
frames are labeled as a reconstruction, not original camera observations.
Use a new output directory for each run.

The [source](../../src/world_use/examples/pick_place.py) builds two plans: pickup
and placement. Each is rehearsed before execution. The gripper must meet the
expected width; a successful sequence alone does not establish task success.
The final predicate reads the simulator's separate truth state.

## Give the task to an agent

```sh
wu up --workcell block
wu policy
```

Give the agent the installed brief and [this task](TASK.md). Let it inspect the
scene and write its own plans. The reference is useful for understanding the API,
but is not evidence of what an unaided model can do. Record the model name, inputs
and any help it receives:

```sh
wu record --context '{"model":"your-model","task":"block transfer"}'
wu record --note 'Operator corrected the block position after inspecting the side camera'
```

## Try a failure

```sh
wu demo --scenario shifted --out runs/shifted --no-video
wu demo --scenario missing --out runs/missing --no-video
wu demo --scenario misplaced --out runs/misplaced --no-video
```

`shifted` moves the initial block 1 cm forward and right in both worlds. `missing`
removes it from the simulator while leaving the assumed box in the robot's model.
`misplaced` moves the actual block 9 cm left without updating that model.

The last two should exit with code 4 and `success: false`: the grip finds nothing.
The reference discards the disproved box position, opens, retreats, and returns
home. It does not retry the same guess. This recovery is specific to the example's
clear tray; a different scene needs a different return route.

These are small regression scenarios with rigid-body contacts and frictional
grasps. Servo gains, friction, and heating remain approximate; these results do
not predict hardware success. See the [simulation model](../../docs/simulation.md).
