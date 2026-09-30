# Working on world-use

Read [POLICY.md](POLICY.md) before operating a robot and [DESIGN.md](DESIGN.md) when changing runtime behavior.
Use simulation and fake-driver tests for development. Code-change requests do not authorize hardware experiments.

For physical sessions:

- Plan the phase and a clear return with torque off. Keep the operator at the motor-supply switch.
- Brief observations and supervised checkpoints are normal parts of a live phase. Finish at confirmed
  torque-off before ending a turn, leaving the session unattended, or waiting for an open-ended reply.
  A completed job, stop, checkpoint, disconnected client, or dead controller does not mean torque-off.
- If the return path becomes blocked, clear the home route and immediately arrange operator-assisted support
  and motor-supply shutdown. Do not leave the arm energized while waiting for chat clearance.
- On communication loss, treat power and cached telemetry as unconfirmed. USB power and the reBot's 48 V
  motor supply are separate. Never claim shutdown from USB disconnection or process termination.
- After the operator ends a hardware session, stay offline until another is requested. Ordinary release/enable
  cycles within an already authorized experiment do not need renewed permission.

See the [September 29 incident](docs/hardware-2026-09-29.md) for the failure behind these rules.
