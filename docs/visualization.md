# Rerun visualization

Install the optional viewer with Python 3.13+:

```sh
uv tool install --force --python 3.13 "world-use[rerun] @ git+https://github.com/anyact-ai/world-use"
```

Include any other extras you use in the same command, for example `[mcp,rerun]`.
From a checkout, use `uv run --extra rerun wu view ...`.

## Open a run

```sh
wu demo --out runs/block-demo
wu view runs/block-demo
```

The Rerun window has an orbitable 3D view of the actual robot geometry, a camera
pane, measured/commanded joint plots, torque and temperature plots, gripper and
power state, and execution events. Drag the `elapsed` timeline to inspect a grasp,
refusal, or recovery. All panes use the run's elapsed clock. Robot joint transforms
and plots retain every committed sample; world geometry and the last ten seconds
of tool trajectory update at up to 20 Hz.

The 3D scene shows **measured joints and the estimated world**. A carried box is
what the runtime believes it holds, not recovered physical truth. Saved camera
observations show the captured scene, including any overlays made by `wu look`.
The demo saves its original MuJoCo camera frames unless `--no-video` is supplied.
Runs without camera observations have an empty camera pane.

Registered [RGB-D evidence](perception-design.md) adds source RGB, depth, selected
support and a compact metadata tab for each camera. Green observed surface points
remain separate from the estimated boxes. RGB/depth appear at capture time;
measurements appear when they became available and carry their capture time in
the label. These are recorded observations, not a live scene reconstruction.
Invalid results clear the associated observed surface. Missing source artifacts
produce a warning in the event pane.

The viewer loads the saved URDF and meshes without opening an adapter or running
physics. It moves the runtime's rotational arm joints and a calibrated gripper
with two prismatic fingers; other gripper joints stay where the URDF puts them,
with a warning in the event pane. Missing geometry is reported explicitly. Runs
recorded by world-use 0.2.0 have no saved model to view; `wu inspect` still reads
their telemetry.

## Follow a running session

```sh
wu up --workcell block
wu view
```

With no folder argument, `wu view` discovers the local daemon's run folder and
follows it. To follow a specified recording, use `wu view runs/YOUR-RUN --follow`.
The viewer reads only new committed chunks and events, usually about one second
behind control. Camera images update when another client calls `wu look` or
registers measurement evidence; this is not continuous camera acquisition.

Press Ctrl+C in the viewer command's terminal to stop following. Following stops
automatically once the record's completion marker and all final data are readable.
A record whose process was killed never gets `complete.json`, so following it needs
Ctrl+C; ordinary offline viewing and export still finish immediately. Closing the
Rerun window or stopping the viewer never stops a robot job or changes torque.
Continue to use `wu status`, `wu stop`, and the normal home/release procedure to
operate the session.

The run folder must be readable on the viewer's host. For a remote daemon, copy
its record or mount its recording directory and supply that path explicitly.
A URL alone does not transfer model assets or recordings.

## Headless export

```sh
wu view runs/block-demo --out block-demo.rrd
rerun block-demo.rrd
```

Export needs no display, graphics context, running daemon, or robot driver. The
`.rrd` contains the meshes, measurements, observations, events, and default layout;
copy it to a desktop with Rerun 0.38 installed. `rerun` is installed with the extra.
Use `--follow --out session.rrd` to export a growing run until close or Ctrl+C.

The interactive Rerun viewer needs a supported graphics backend. On cloud hosts,
export the recording and open it locally. MuJoCo's EGL camera requirements are
separate; see [simulation](simulation.md). Rerun is not imported by the daemon or
required for CLI, Python, MCP, simulation, or GIF replay.
