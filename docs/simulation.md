# MuJoCo simulation

`--body sim`, `--body sim:rebot`, `wu check` and `wu demo` use MuJoCo, a core
dependency. The built-in rehearsal twin uses the same engine, including when
checking a plan for hardware. The kernel owns commands, motion limits, watchdogs,
faults and recovery.

The bundled reBot uses Seeed's URDF, link inertias, colored component meshes,
and convex finger collision segments. Arm collision uses component convex hulls
where a whole-link hull would fill gaps around the wrist. Adjacent links are
excluded from self-contact. MuJoCo handles joint limits, gravity, collisions,
and friction. Grasped objects have free joints: no weld or pose-following rule
attaches them to the tool. They can slip, tip, fall, and collide once let go.

Physics advances one control period per `Body.read`, in substeps of at most
2 ms. A virtual-clock rehearsal runs the same engine without wall-clock waits.
Camera capture does not advance physics.

Disabling removes actuator forces; gravity and collisions continue. With torque
off, each joint resists load up to its gearbox friction, about 0.3 Nm per revolute
joint and 10 N per prismatic one, as unpowered geared motors do. A folded arm and
its gripper stay where they are; a raised arm without brakes falls. An arm without
a rest pose (`self_supporting`) holds on its brakes. Powered, the servos hold every
joint and this friction is off.

## Scene and model assumptions

The simulator's `World` contains physical truth. The kernel keeps a separate
estimated world; passing the same instance to both creates a copy for the kernel.
A rehearsal starts from measured joints and the estimated world, not hidden
simulation truth. An unknown obstruction or misplaced object can therefore pass
a rehearsal and still stop execution. The kernel's object attachment is a
belief used in planning; it never moves the simulated object.

The native workcell supports box geometry: `surface` boxes are static colliders and
`object` boxes are free rigid bodies, defaulting
to 50 g and a sliding friction coefficient of 0.8. Box parameters `mass_kg` and
`friction`, also in workcell `[[box]]` entries, override those values. Place
objects on a surface; unsupported objects fall. The ground plane is at base-frame
z=0 and gravity points along base -z. Keep-out, fragile, and slow boxes remain
planning/monitoring rules.

`known = false` on an `object` or `surface` puts that physical box only in simulation truth. An agent adds
its own estimate after observing it; that estimate does not change physics.
Camera images and depth come from truth, while the card, plans and rehearsal use
the estimate. Task predicates belong in the procedure's evaluator: see the
[block adaptation recipe](../examples/pick-place/README.md#adapt-the-task) for a
changed tray, starting pose and destination with a separate outcome check.

Articulated environment objects such as hinged doors, arbitrary scene meshes,
soft objects and fluids are outside this box workcell. Robot URDF geometry can
include primitive shapes and meshes, but loading a robot does not add support for
those environment types. Their dynamics require an external environment/body
integration; the kernel can still reason about conservative boxes supplied in its
estimated `World`. The [RoboCasa adapter](../examples/robocasa-counter) integrates
an external kitchen scene, including the [articulated oven dial](../examples/robocasa-dial).
It executes in the original scene but disables full-plan rehearsal: the native
twin does not reproduce that kitchen's contact geometry.

Changing the physical scene's topology rebuilds its MuJoCo model. Configure it
before execution. Editing a truth box's pose alone does not teleport the live
body: call `SimBody.reset(q, gripper)` while idle to reset the scene from its
current world poses, without stepping dynamics. This does not reset the kernel's
faults, measurements, clock or record. Recorded replay uses reset without stepping
dynamics; use a fresh kernel and output folder for an independent task trial.

Actuator gains, joint damping, contact friction, and motor heating are approximate
model parameters, not validated hardware measurements. Fitted link masses,
centers of mass, and joint friction apply to dynamics as well as the kernel's
torque prediction. Heating remains a separate approximate thermal model.

MuJoCo contacts improve rehearsal fidelity; they do not replace conservative
planning. Path and measured-state checks use padded link segments, the tool point,
[declared tool/finger bounds](adapters.md#collision-bounds) and the estimated held
object. A passing rehearsal depends on geometry, physical parameters and scene
estimates; it does not cover undeclared geometry or self-collision.

## Cameras and replay

Simulated cameras render the same MuJoCo scene used for dynamics, including the
actual robot meshes. They preserve camera pose and pinhole intrinsics so image
coordinates agree with calibration and overlays. OpenGL runs on a dedicated
thread, using a copied simulation state without holding the physics lock during
rendering.

Headless Linux selects EGL when `DISPLAY` is absent. Install the Mesa EGL runtime
(`libegl1` and `libgl1-mesa-dri` on Ubuntu) or use an EGL-capable GPU driver. An
explicit `MUJOCO_GL` setting takes precedence; `MUJOCO_GL=osmesa` requires OSMesa.
macOS uses MuJoCo's native CGL backend and requires GPU access. GitHub-hosted macOS
VMs cannot provide that context; MuJoCo does not support software rendering there.
Dynamics and rehearsal do not need a display or graphics context. Rendering errors
are reported rather than replaced with a schematic image.

CI runs the full suite, including camera projection and replay, on Linux with Mesa
EGL. Hosted macOS runs `pytest -m 'not rendering'`; it still tests physics, control,
record recovery, Rerun export and image transport through file cameras. On a Mac
with GPU access, run the full suite with `uv run pytest`.

Integration tests exercise longer procedures:

```sh
uv run pytest tests/test_simulation_recovery.py tests/test_simulation_transfer.py
uv run pytest tests/test_simulation_variations.py
```

The recovery test moves an unknown block during a checkpoint, explicitly withdraws its measurement,
and checks that the old grasp is refused. The perception procedure then observes again and completes
the task in the same session, at two changed positions. This tests recovery after invalidation, not
automatic detection of scene changes, and needs camera rendering. The transfer test uses CPU physics:
two friction-held blocks cross a physical divider, low shortcuts fail rehearsal, and both final poses
are checked against simulation truth. Both tests check return, torque release and record completion.
These are scripted integration tests, not model-policy benchmarks.

The variation test keeps the perception procedure's motions unchanged while varying block position,
camera placement, mass and the object's friction setting. Two scenes must complete placement. A heavier
block must pass its lift check but may miss placement; an angled view can defeat the colour selector.
Any claimed placement must pass independent pose evaluation. Every case requires release, withdrawal,
torque off and a complete record; incomplete tasks must report a reason.

MuJoCo normally uses the larger friction coefficient of two contacting geoms at equal priority. The fingers
use 1.0 and the tray 0.8, so lowering only the object's setting to 0.4 does not reduce these contact coefficients.
That variation checks the configuration path, not robustness to a more slippery grip.

Records keep the URDF and copies of a custom robot's meshes. A built-in robot's
meshes ship with world-use; replay finds them by the recorded URDF's exact
contents. Native workcell replay renders measured joints and the recorded **estimated world**, not
a second physics rollout or a reconstruction of unobserved object motion. Saved
camera observations and the demo GIF show the original simulation truth.

## External simulation sessions

[`SimulationSession`](../src/world_use/simulation.py) starts a fresh kernel,
measurement store, clock, daemon and record for a caller-built MuJoCo scene.
The [pushing](../examples/push-letters) and [kitchen packing](../examples/robocasa-lunchbox)
examples share this path. Scene construction, policy and physical success checks
stay with each task. The six-method `Body` contract is unchanged; this MuJoCo
integration additionally needs the body's `model`, `data` and reentrant `lock`.

```python
from pathlib import Path
from world_use.simulation import SimulationSession

with SimulationSession(body, Path("runs/new-episode"), cameras=cameras,
                       sources={"task.py": Path(__file__)}, entrypoint="task.py") as session:
    robot = session.client
    frame = robot.frame("overview", depth=True)
    # Measure, plan and evaluate using current observations.
    robot.enable()
    robot.run(plan, check=False, wait=120)
    # Poll an unfinished job before proceeding; withdraw as the task requires.
    robot.release()
    session.close({"verdict": "inconclusive", "reason": "physical audit pending"})
```

The normal `Client` API records calls and exact RGB-D frame transport. Pass this
client to `Tracking` or `mcp_server.build(client=...)` to include their internal
observations. A source snapshot documents the live caller; it does not execute
that snapshot. Use [`world_use.tasks`](../examples/task-programs) when a task
must execute its frozen source in a fresh Python process.

Sessions use wall-clock control by default. Closing requires idle, confirmed
power-off state. Exiting the context stops work and requests release, including
after a task exception; it never chooses a retreat motion. A refused release
leaves the daemon available for explicit recovery. Use a new output folder for
each trial, even after a simulator reset.

The output keeps the ordinary `flight/` record, `task/` source and observation
trace, a compiled `scene.mjb`, and bounded `physics/` chunks written off the
control thread. Successful finalization also produces `replay.npz` with full
MuJoCo integration state, wall time and simulation time. Recording errors remain
distinct from task outcomes; interrupted records retain their committed chunks.
The compiled model records the MuJoCo version in `physics/format.json` and needs
a compatible MuJoCo runtime to load.

```sh
wu replay runs/new-episode/flight --out scene.gif --speed 3
```

For these sessions, replay restores the original external scene without stepping
physics. It uses the first camera's initial pose as a fixed viewpoint and labels
both clocks. Actual policy observations remain in `task/artifacts/`; replay is
not a substitute for them. Keep the whole episode folder together. `wu view`
still displays measured joints and the estimated world in Rerun.

The card and rehearsal reports expose `rehearsal_coverage`: estimated boxes and
approximate dynamics, with external scene geometry and contact dynamics omitted.
Recording a detailed scene does not extend the planning twin's coverage.

## Reference checks and repeated evaluation

The [scene-validation example](../examples/scene-validation) records reference
provenance, alignment, fixed tolerances and reserved observations against a hashed
scene revision. Appearance/camera, metric geometry and physical response have
separate results. Synthetic examples exercise the checks; the measured MIT pushing
example currently fails the physical-response tolerance and has no camera evidence.
Neither establishes a validated real-to-sim twin.

[`world_use.evaluation`](../src/world_use/evaluation.py) executes named seeded
cases from frozen policy/evaluator sources and declared input snapshots. It keeps
task outcome, scene validity, recording integrity and power cleanup separate.
Every declared case remains in the summary, including invalid or interrupted runs.
`held_for` evaluates sustained conditions on simulation time with a bounded sample
gap; wall time is recorded independently. See the
[pushing evaluation](../examples/push-letters/README.md#compare-policy-revisions-on-fixed-cases) for a
concrete camera-policy case and an independent physical audit.

An optional `setup` program prepares each case's scene in its own process before the
policy starts: seeded initial variations, settling and the start-validity check. Its
revision joins the case identity and its result decides whether the start is valid,
so neither depends on policy code. The policy and evaluator receive the prepared
directory as `params["scene"]`; a changed scene invalidates the attempt.

`rerun` recreates a named case in a fresh directory and refuses changed inputs,
sources or runtime versions. `compare` requires identical case identities,
evaluator and runtime while allowing a different policy revision. Privileged
diagnostics remain visible but are excluded from camera-only comparisons. This
separation is for trusted task code, not a Python sandbox. Identities cover the
dependencies callers declare; compiled MuJoCo models require their recorded runtime.

Pass `policy_context={"skills/grasp.md": skill_path, "prompt.txt": prompt_path}`
to `run_suite` to freeze the declared knowledge behind a policy. These files may
come from outside the task directory. They join the policy source bundle and its
revision; the policy receives saved paths in `params["context"]`. Scene inputs
remain in `inputs`, which determines case identity. This permits comparisons of
different prompts or skills on identical scenes, seeds and evaluator code.
Reruns use the saved context. Changed shared inputs or source/context snapshots
detected before or after execution invalidate the attempt and remain in the report.

The caller must declare the knowledge actually used; copying a document alone
does not establish that an authoring agent read it. Freeze the policy and knowledge
before evaluating held-out cases. Case roles are labels, not an access barrier.
Keep task development and skill promotion outside the kernel. Reproducing a lesson
from its text tests whether the instructions suffice; transfer needs separate
held-out tasks and embodiments with their assumptions and results recorded.

## Other robots

A custom URDF must provide physically valid inertias and collision geometry.
Visual meshes are optional; referenced meshes must be STL or OBJ files available
locally, relative to the URDF or as `package://` paths inside their package. The
planar adapter example includes inertias and primitive geometry.

The simulation drives a gripper's joints outside the arm chain: one joint, revolute
or prismatic, in its URDF coordinate, or two opposed prismatic fingers that share
a calibrated opening (`m_per_unit`), zero at closed. A hinged jaw's servo acts
like a finger's 5 cm from the hinge. An object counts as held when it touches
both fingers, or the jaw and the link it closes against. Other mechanisms need an
explicit simulation model. Fixed child links, such as finger collision pads, are
part of the same MuJoCo rigid group and contribute to those contact labels. The
label never attaches an object to the tool; release and motion remain physics.
Without a gripper description, joints outside the arm
chain stay at their URDF zero. Arm joints are revolute or continuous; see the
[adapter guide](adapters.md).

A workcell's `[simulation]` table accepts `start_deg` (or `q` in radians),
`gripper`, `temp_c`, `ambient_c`, `noise`, and `seed`.
