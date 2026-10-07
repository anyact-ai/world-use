# reBot setup

The physical adapter drives the Seeed reBot Arm B601-RS over CAN. Run the
[simulated block task](../examples/pick-place) first. The arm has no brakes: switching torque off while it is
raised can make it fall. Keep an operator at its motor-supply switch.

## Install

Use Python 3.13 for the reBot extra. Its CAN driver, motorbridge 0.5.5, has no wheels for Python 3.14;
on macOS its wheels support Apple Silicon only.

```sh
uv tool install --python 3.13 "world-use[rebot,mcp,rerun] @ git+https://github.com/anyact-ai/world-use"
wu policy
```

This keeps `wu` and the driver in one environment. Drop the extras you do not use.

On **Linux**, configure the CAN adapter as SocketCAN `can0` at the arm's 1 Mbit/s bitrate before starting
world-use. Use your adapter's setup instructions and confirm the interface with `ip -details link show can0`.

On **macOS**, install the MacCAN PCBUSB runtime using
[motorbridge's setup guide](https://github.com/motorbridge/motorbridge#macos-pcan-runtime-pcbusb). The driver maps
`can0` to `PCAN_USBBUS1` and `can1` to `PCAN_USBBUS2`. Quit MotorBridge Studio or any other program that owns the
adapter before connecting. USB powers the interface; the arm's **48 V supply** powers its motors.

## Configure a camera and workcell

Copy [workcell.toml](../examples/rebot/workcell.toml) to your own folder. Set `body_options.channel`, then choose
one camera source. The template holds no scene geometry or camera calibration.

A camera `command` prints one image to stdout. With FFmpeg, test a still first, then use the same input with
`-f image2pipe -c:v png -` in the template:

```sh
ffmpeg -f avfoundation -list_devices true -i ""                                       # macOS: list cameras
ffmpeg -loglevel error -f avfoundation -framerate 30 -i 0 -frames:v 1 side.png        # macOS: camera 0
ffmpeg -loglevel error -f video4linux2 -i /dev/video0 -frames:v 1 side.png            # Linux
```

On macOS, grant the terminal camera access when asked. The
[FFmpeg device reference](https://ffmpeg.org/ffmpeg-devices.html) explains device selection. These commands pick a
camera; they do not tell world-use where it is relative to the arm.

A camera can instead be a still-image `url`, or a `path` that a capture app you already run keeps replacing,
preferably by atomic rename; frames older than `max_age_s` are refused. Use one source per camera. An uncalibrated
camera returns raw pictures. Add a measured `eye`, `look_at` and `fov_deg`, or run `wu calibrate side` during an
authorized, supervised session, before relying on what is drawn on them. A 360 camera that serves an
equirectangular picture can serve pinhole cuts of it instead: `projection = "equirect"`, aimed with `yaw_deg` and
`pitch_deg`.

Camera settings are validated at load time: `max_age_s` must be finite and positive and applies only to
file sources, image dimensions must be positive integers, and `fov_deg` must be between 0 and 180 degrees.
Partial calibrations, unknown projection names and conflicting source or aiming settings are rejected.

## Connect and operate

With the arm supported at rest and the motor supply on:

```sh
wu up --workcell /absolute/path/to/workcell.toml
wu status --json
wu card
wu look side
```

Connecting leaves torque off. Confirm `session.mode: "hardware"`, look at the scene, and prepare the phase and its
way home before `wu enable`. The [agent brief](../src/world_use/POLICY.md) (`wu policy`) holds the operating rules:
how to hand off, when to go home and release, and what to do when the way home is blocked or motor power is
unconfirmed. It applies to people driving the arm as much as to agents.

`wu up` reuses a running daemon only when its adapter and startup workcell match. To change sessions, finish the
run, go home, and `wu down` first.

Feedback counts as fresh only while the reported motor values change, so identical samples can be mistaken for a
stale cache; telling them apart needs a receive counter from motorbridge. A read-only connection also rejects
repeated exact-zero positions, because the driver cannot reliably tell them from failed or misidentified parameter
replies.

The [hardware records](hardware.md) describe three sessions on one arm: first runs, a fitted model, and a prolonged
hold that ended with the motor supply switched off by hand.
