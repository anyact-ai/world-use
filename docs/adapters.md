# Adding a robot

The [planar example](../examples/adapters/planar.py) is a runnable adapter with its
own two-joint URDF. It emulates a position controller in memory; no hardware is
opened. From a checkout:

```sh
uv run python examples/adapters/planar.py
```

It defines a manifest, implements all six methods, rehearses a joint move on a
`SimBody` twin, executes that move on its own adapter, and releases. Start here to
understand the boundary, then substitute a real driver.

## Describe the robot

Put the adapter and URDF under `src/world_use/bodies/your_arm/`. The module should
export one stable `MANIFEST` object and your body class. Joint order must match the
URDF chain to `tool_link`. The tool origin is the point Cartesian actions control.

Fill in measured joint limits, sensing, the gripper's native-unit/aperture mapping,
and the frame the operator uses. Advertise only sensing the adapter actually
returns. `None` means unavailable; zero means a measured zero.

`rest=None` means the robot supports itself when disabled. For an arm without
brakes, define the supported rest pose and the joints that carry weight. Do not
copy the reBot's folding angles, motor ranges, gripper units or turn-height rule
unless they match the new arm. The planar example has no physical load to support.

## Implement I/O

The [Body contract](../src/world_use/body.py) defines the six methods:

| Method | Responsibility |
| --- | --- |
| `connect` | Open read-only and return a measured state; no motion or release |
| `enable` | Engage at the measured position; report incomplete power transitions |
| `read` | Return fresh measurements or a timestamp-preserving cache; raise on lost feedback |
| `command` | Send one bounded position/velocity/gripper command |
| `disable` | Confirm torque-off, or raise with the unconfirmed motors |
| `close` | Close the connection without changing motor power |

The kernel calls I/O from one thread. Keep `read` and `command` bounded to the
control period. Driver receipt timestamps or sequence counters establish freshness;
changing values do not. Test constant fresh samples, frozen caches, disconnects,
and partial enable/disable failures using a fake transport before connecting hardware.

## Make the daemon and worker find it

There is deliberately no plugin loader. Add the manifest to `bodies.manifests()`
and the constructor to `bodies.make()` in [bodies/__init__.py](../src/world_use/bodies/__init__.py):

```python
# In manifests():
from .your_arm import MANIFEST as YOUR_ARM
return {"rebot": REBOT, "your_arm": YOUR_ARM}

# In make(), before the unknown-body error:
if kind == "your_arm":
    from .your_arm import YourArmBody
    return YourArmBody(**options)
```

`sim:your_arm` then uses the same manifest automatically. A spawned planning
worker imports `bodies.manifests()` afresh, so registration must be in an importable
module, not just a notebook or the launching process. The worker refuses extensions
it cannot reconstruct. Embedded `check` remains available for unregistered adapters.
Custom behaviors likewise need a module imported in both processes.

First run `wu up --body sim:your_arm`, then check the card, frames, joint and
Cartesian moves, gripper expectations, stop, home, release and records. Add fake-driver
failure tests and a real task before trying `--body your_arm` in a separately
authorized physical session. Copy a wheel into a clean environment too: missing
URDFs and import-time registration bugs can hide in a source checkout.
