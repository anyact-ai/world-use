# reBot setup

The physical adapter supports the Seeed reBot Arm B601-RS over CAN. First complete
the [simulation example](../examples/pick-place). The arm has no brakes: removing
torque while raised can make it fall. Keep an operator at its motor-supply switch.

## Install

Use Python **3.13** for the reBot extra on macOS and Linux:

```sh
uv tool install --python 3.13 "world-use[rebot] @ git+https://github.com/anyact-ai/world-use"
wu policy
```

This keeps `wu` and the driver in the same environment. The pinned motorbridge
0.5.5 has a macOS Python 3.13 wheel; its macOS Python 3.14 source-install workaround
is unnecessary here. The core and MCP interfaces also support Python 3.14.

On **Linux**, configure the CAN adapter as SocketCAN `can0` at the arm's 1 Mbit/s
bitrate before starting world-use. Use your adapter's setup instructions and confirm
the interface with `ip -details link show can0`.

On **macOS**, install the MacCAN PCBUSB runtime using
[motorbridge's setup guide](https://github.com/motorbridge/motorbridge#macos-pcan-runtime-pcbusb).
The driver maps `can0` to `PCAN_USBBUS1` and `can1` to `PCAN_USBBUS2`.
Quit MotorBridge Studio or other programs that own the adapter before connecting.
USB powers the interface; the arm's **48 V supply** powers its motors.

## Configure a camera and workcell

Copy [workcell.toml](../examples/rebot/workcell.toml) to your own folder. Set
`body_options.channel`, then choose one camera source. No guessed scene geometry
or camera calibration is included in the template.

For a Mac webcam, install [ImageSnap](https://github.com/rharder/imagesnap), list
cameras with `imagesnap -l`, and test a still with `imagesnap -d 'DEVICE' side.jpg`.
Set the camera's `command` to `imagesnap -d 'DEVICE' -q -` to return image bytes on
stdout. Grant camera access to the terminal/capture application when macOS asks.

For a Linux webcam with FFmpeg:

```sh
ffmpeg -loglevel error -f video4linux2 -i /dev/video0 -frames:v 1 side.png
```

Then use the matching `command` in the template. The
[FFmpeg device reference](https://ffmpeg.org/ffmpeg-devices.html#video4linux2_002c-v4l2)
explains device selection. Select the actual camera for your system; these commands
do not establish its position relative to the arm.

Alternatively, provide a still-image `url`, or keep `path` and run your existing
capture application. A file source must keep replacing the image, preferably by
atomic rename; frames older than `max_age_s` are refused. Use one source per camera.
An uncalibrated camera returns raw images. Add measured `eye`/`look_at`/`fov_deg`,
or use `wu calibrate side` during an authorized, supervised session, before relying
on world overlays.

## Connect and operate

With the arm supported at rest and the motor supply on:

```sh
wu up --workcell /absolute/path/to/workcell.toml
wu status --json
wu card
wu look side
```

Connecting leaves torque off. Confirm `session.mode: "hardware"`, inspect the scene,
prepare the phase and its return route, then enable. Read the installed `wu policy`
for action parameters and operating guidance.

`wu up` reuses a daemon only when its adapter and startup workcell match. To change
sessions, finish the current run, return to rest, and use `wu down` first.

Before leaving a session or waiting for an open-ended reply, return by the verified
home route and release. Confirm `enabled: false` and `power_uncertain: false`.
Brief supervised observations and checkpoints are part of normal operation;
a stopped or completed job still holds with torque and continues to heat.
An empty home route means the current scene permits turning and folding directly.

If returning is blocked, the operator must support the arm and cut its **48 V motor
supply**. USB disconnection, `stop`, or killing a process does not remove motor power.
On communication loss, status marks feedback stale and power unconfirmed; reconnecting
does not resume commands. Resolve power before reset or restart. See `wu policy` for
the recovery contract.

Feedback freshness currently depends on reported motor values changing. Identical
incoming samples can be mistaken for a stale cache; distinguishing them requires
a receive counter from MotorBridge.
Read-only connection also rejects repeated exact-zero positions because the
current driver cannot reliably distinguish them from failed or misidentified parameter replies.

## Hardware records

- [First runs and pick-and-place, September 27](hardware-2026-09-27.md)
- [Model fitting and follow-up runs, September 28](hardware-2026-09-28.md)
- [Prolonged hold and thermal-return failure, September 29](hardware-2026-09-29.md)

These are development records on one arm, not reliability benchmarks. The current
changes were tested offline; they do not establish the post-incident arm's condition.
