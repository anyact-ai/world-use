# Oven dial

A Franka Panda must set the oven to **180°C**, choosing among six controls with
burner symbols, timer markings and a Fahrenheit temperature scale.

https://github.com/user-attachments/assets/7dfb024d-f794-41f6-bd02-47260d103639

## What world-use adds

- **Measurement.** Side/wrist RGB-D, EdgeTAM selection and `measure_pixels`
  locate the handle, panel and scale marks.
- **Motion.** Task Python estimates the axis and turn from those points.
  World-use executes the approach, grip, rotation and withdrawal.
- **Verification.** New wrist measurements check the pointer after release.

The LLM selects the control and converts 180°C to 356°F. Codex chose the MCP calls
interactively, with scene knowledge; world-use contains no oven policy.

## Result and scope

The agent planned a **42.7° turn**. After release, the measured pointer read
**about 360°F** against the **356°F target**, within the trial's ±5°F tolerance.
The contact audit found no unintended robot contacts in 8,931 saved states.
One untouched burner knob drifted 0.74° passively; the other non-target controls
changed less than 0.001°.

Finger contact drives the dial. The agent assumes its axis is normal to the
panel; it does not read the simulator's hinge position. This selected trial uses
approximate knob friction and idealized cameras, with no heat, detents or
push-to-turn interlocks. Joint torque sensing and full-scene rehearsal are
unavailable (`rehearse=False`). It is a development example, not a hardware or
repeatability result.

The 40-second video shows static side-camera identification, wrist observations
and final verification. Motion plays at 1× from a labeled presentation view;
decision gaps are cut. Overlays show recorded measurements and calculations.

## Run locally

Complete the [kitchen setup](../robocasa-counter/README.md#build-the-upstream-scene) and
[runtime install](../robocasa-counter/README.md#prepare-the-runtime), then run from the repository root:

```sh
.venv/bin/python examples/robocasa-dial/prepare.py
.venv/bin/python -i examples/robocasa-dial/episode.py --output runs/dial-new
```

This starts an interactive recorder, not an automatic task. Read `wu policy`,
inspect observations and choose calls with `act`; save decisions with `note` and
end with `finish` after releasing the robot. Use a new output directory each time.

The audit and renderer were written for the archived trial 03, whose raw recording
is not bundled. For a new episode, adapt their call indices before running:

```sh
.venv/bin/python examples/robocasa-dial/audit_episode.py runs/dial-new
.venv/bin/python examples/robocasa-dial/render_episode.py runs/dial-new
```

`runs/dial-agent-03/` retains calls, decisions, raw RGB-D, calibration,
replay and audit. Earlier failed trials remain in `dial-agent-01` and
`dial-agent-02`. Raw recordings and downloaded assets are not bundled. See the
[asset attribution](../robocasa-counter/README.md#attribution).
