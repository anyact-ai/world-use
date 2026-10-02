# Changelog

## 0.3.0 (unreleased)

- Clear gripper velocity on stop while retaining its position; home load-bearing joints
  to their configured rest pose, even when the session started elsewhere.
- Report camera capture timeouts and correct calibration of resized 360-camera views.
- Describe the configured work frame in the robot card and clarify Python script setup.
- Add unannotated camera frames and optional EdgeTAM tracking for Python procedures.
  Keep forward history bounded and report lost or stale observations.
- Load robot descriptions and external drivers from workcell configuration. Rehearse
  custom robots in the worker; save their models and URDFs for portable replay and fitting.
- Resolve config paths relative to their files and reject unknown fields and invalid models.
- Keep event persistence off the control thread; recording failures cannot block release.
- Reject foreign HTTP hosts/origins and non-JSON commands. Preserve structured CLI JSON,
  report MCP failures as tool errors, and explain malformed local inputs without tracebacks.
- Validate action fields and values before queueing. Refuse incomplete rehearsals.
- Keep collision and tracking checks active during a thermal return. Confirm release
  before reporting that return as complete.
- Restore all home targets after escape moves, and correct exact half-turn interpolation.
- Prepare execution paths in the spawned worker; feedback and stop handling continue
  during preparation. Refuse unsupported worker extensions explicitly.
- Support Python 3.13 and retain 3.14 coverage. Use 3.13 for published macOS reBot wheels.
- Package `wu policy`; complete MCP job, reset, shutdown and structured status tools.
- Persist incremental telemetry, submitted plans, startup state, and structured outcomes.
  Add `wu inspect`, `wu replay`, and optional agent context/intervention annotations.
- Add the complete block task, failure/recovery scenarios, captured simulation preview,
  physical workcell template, adapter example, and installed-wheel CI checks.

**API change:** `run --checked` and `Client.run(checked=True)` are removed. Submit
an explicit plan or plan file; the daemon no longer stores a shared “last checked” plan.
New record readers also accept existing `tape.npz` files. Visual replay needs the
new `session.json` metadata.

Robot loading and the runtime now reject prismatic arm joints: actions, telemetry,
and monitoring support rotational joints only. Offline `Chain` kinematics is unchanged.

## 0.2.0

Previous public release. The September 29 thermal-recovery changes were merged after
that tag; 0.3.0 includes them as well as the changes above. See the hardware records
for the observations behind the recovery contract.
