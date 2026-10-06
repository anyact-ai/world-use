# MuJoCo simulation

`--body sim`, `--body sim:rebot`, `wu check` and `wu demo` use MuJoCo, a core
dependency. The twin that checks and prepares every motion, on hardware as well,
is this simulation. The kernel owns commands, motion limits, watchdogs, faults and
recovery.

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
estimated `World`. The bundled examples demonstrate native boxes and robot
adapters, not such an environment integration.

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
planning. Path checks still use padded link segments and the tool point. A
passing rehearsal depends on geometry, physical parameters, and scene estimates.

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

The variation test keeps the perception procedure unchanged while varying block position, camera placement,
mass and friction. It checks physical placement in three scenes and requires either verified placement or
explicit failure with release, withdrawal and torque off in an angled view that can defeat the colour selector.

Records keep the URDF and copies of a custom robot's meshes. A built-in robot's
meshes ship with world-use; replay finds them by the recorded URDF's exact
contents. Replay renders measured joints and the recorded **estimated world**, not
a second physics rollout or a reconstruction of unobserved object motion. Saved
camera observations and the demo GIF show the original simulation truth.

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
