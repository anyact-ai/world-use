# Oven dial

A simulated Franka Panda sets a RoboCasa oven to 180°C. The stove has six controls, with burner symbols, timer
markings and a Fahrenheit temperature scale. Codex chose each MCP call interactively, with knowledge of the scene:
it picked the temperature control, converted 180°C to 356°F and planned the turn. world-use contains no oven
policy.

https://github.com/user-attachments/assets/7dfb024d-f794-41f6-bd02-47260d103639

The agent planned a 42.7° turn. After release, a fresh wrist image put the pointer at about 360°F against the
356°F target, within the trial's ±5°F tolerance. The contact audit found no unintended robot contacts in 8,931
saved states. One untouched burner knob drifted 0.74° on its own; the other non-target controls changed less than
0.001°. This is one development trial, not a hardware or repeatability result.

## What world-use adds

- Side and wrist RGB-D, EdgeTAM selection and `measure_pixels` locate the handle, the panel and the scale marks.
- Task Python estimates the dial's axis and the turn from those points; world-use executes the approach, grip,
  rotation and withdrawal.
- New wrist measurements read the pointer after release.

## Limits

Finger contact turns the dial. The agent assumes the dial's axis is normal to the panel; it does not read the
simulator's hinge position. Knob friction is approximate and the cameras are ideal, with no heat, detents or
push-to-turn interlocks. There is no joint torque sensing or full-scene rehearsal (`rehearse=False`).

The 40-second video shows the static side-camera identification, the wrist observations and the final check.
Motion plays at 1× from a labelled presentation view, with decision gaps cut. Overlays show recorded measurements
and calculations.

## Run locally

Complete the [kitchen setup](../robocasa-counter/README.md#build-the-upstream-scene) and
[runtime install](../robocasa-counter/README.md#prepare-the-runtime), then run from the repository root:

```sh
.venv/bin/python examples/robocasa-dial/prepare.py
.venv/bin/python -i examples/robocasa-dial/episode.py --output runs/dial-new
```

This starts an interactive recorder, not an automatic task. Read `wu policy`, inspect observations and choose calls
with `act`; save decisions with `note` and end with `finish` after releasing the robot. Use a new output directory
each time.

The audit and renderer index the calls of the archived trial 03, whose raw recording is not bundled. For a new
episode, adapt their call indices before running:

```sh
.venv/bin/python examples/robocasa-dial/audit_episode.py runs/dial-new
.venv/bin/python examples/robocasa-dial/render_episode.py runs/dial-new
```

`runs/dial-agent-03/` keeps the calls, decisions, raw RGB-D, calibration, replay and audit. The earlier failed
trials are in `dial-agent-01` and `dial-agent-02`. Raw recordings and downloaded assets are not bundled. See the
[asset attribution](../robocasa-counter/README.md#attribution).
