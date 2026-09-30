# reBot setup

The physical adapter supports the Seeed reBot Arm B601-RS over CAN. Start with the
[simulation demo](../README.md#try-it) and read [POLICY.md](../POLICY.md) before using it.
The arm has no brakes; removing torque away from its supported rest pose can make it fall.
Keep an operator at the power switch.

## Install and connect

```sh
uv tool install "world-use[rebot] @ git+https://github.com/anyact-ai/world-use"
wu up --body rebot
wu status --json
wu card
```

This connects with torque off. Check that status reports `session.mode: "hardware"`,
inspect the scene and camera calibration, and prepare the first plan before `wu enable`.
A workcell TOML can supply body options, camera configuration, boxes, and a fitted model;
see [load_workcell](../src/world_use/daemon.py) and the [example workcell](../src/world_use/workcells/block.toml).

`wu up` reuses a daemon only when its adapter and startup workcell match. To switch from
simulation to hardware, finish the current run, return to rest, and use `wu down` first.

On completion, set a home route based on the scene, run `wu home`, then `wu down`.
An empty route means the arm can turn and fold directly from its current position;
it is not a general escape route. If a power transition fails, status reports unconfirmed
motor power. The operator must resolve it and release at a freshly measured rest pose
before resetting the fault.

Do this before leaving the session unattended or waiting for an open-ended reply, too. Brief supervised
checkpoints remain part of normal operation; there is no automatic homing after each action.
A raised arm holding at a checkpoint continues to heat.
If the scene changes, clear the old route with `wu home-route 'null'` and resolve motor power immediately.
When returning is blocked, the operator must support the arm and switch off its **48 V motor supply**.
USB disconnection, `wu stop`, and terminating the controller do not remove motor power.

On lost or stale feedback, the runtime latches unconfirmed power and suspends commands. Reconnecting USB
does not resume them. Status includes `feedback.age_s`, `feedback.stale`, and the last read error; displayed
values may be cached. Recovery requires fresh rest-pose feedback and a successful release, or physical
shutdown and a new session. Do not reset or restart while the arm may still be energized.

## macOS driver note

This workaround was used with the pinned `motorbridge` 0.5.5 dependency and Python 3.14.

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


## Hardware records

- [First runs and pick-and-place, September 27](hardware-2026-09-27.md)
- [Model fitting and follow-up runs, September 28](hardware-2026-09-28.md)
- [Prolonged hold and thermal-return failure, September 29](hardware-2026-09-29.md)

These are development records on one arm, not reliability benchmarks.
