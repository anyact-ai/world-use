# Working on world-use

Read [DESIGN.md](DESIGN.md) when changing runtime behavior. Use simulation and
fake-driver tests for development. Code-change requests do not authorize hardware
experiments; an ended physical session stays offline until another is requested.

Before operating a robot, read the installed `wu policy` or its
[canonical source](src/world_use/POLICY.md). It owns the operating and power-recovery
rules. Do not leave a physical arm holding while waiting for an open-ended reply.
See the [September 29 incident](docs/hardware-2026-09-29.md) for the failure behind
those rules.
