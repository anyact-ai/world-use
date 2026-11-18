# Track an object between agent decisions

The agent selects an object in a picture once. EdgeTAM then follows it through later frames and returns a mask,
a bounding box and a centre in image pixels, so a Python procedure can keep track of the object between phases
without asking a model about every picture.

Tracking is optional and runs in the procedure's process. This example reads pictures and prints what it tracks;
it does not enable or move a robot. Agents on MCP get the same tracking as tools with `wu mcp --vision`; see
[measuring from pictures](../../docs/perception.md#tracking).

## Try a recording

From a checkout of world-use:

```sh
uv run --extra vision --with opencv-python-headless examples/tracking/track.py \
  --video /path/to/recording.mp4 --box 100 120 220 260 --preview latest.png
```

Replace the box with the object's coordinates in the first frame, in native pixels; `--point X Y` selects with a
single foreground point instead. The run processes at most 300 frames (`--frames`) and overwrites `latest.png`
with the newest annotated frame. OpenCV only decodes the video; EdgeTAM segments and tracks.

The first use downloads the pinned public checkpoint and its backbone metadata. By default, Linux with a working
NVIDIA setup uses CUDA, Apple Silicon uses MPS and other machines use the CPU; `--device` overrides this. The CPU
works, with slower updates. Startup and the first inference can take several seconds. The Mac path has been tried
on recorded footage; CUDA has not been tried.

## Use a configured camera

Start the daemon with your camera configuration, and save the exact picture and frame the agent will look at:

```python
import json
from pathlib import Path
from world_use import Client

frame = Client().frame("side")
frame.image.save("selection.png")
Path("selection.json").write_text(json.dumps(frame.to_dict()))
```

After choosing a point in `selection.png`:

```sh
uv run --extra vision examples/tracking/track.py \
  --seed selection.json --point 310 240 --preview latest.png
```

`Client.frame()` returns an upright picture at native resolution, with nothing drawn on it and nothing recorded.
`wu look` may scale and annotate its pictures, so do not take pixel coordinates from it.

In a procedure, the whole tracking interface is:

```python
import json
from pathlib import Path
from world_use import Client
from world_use.cameras import Frame
from world_use.vision import EdgeTAM

robot = Client()
seed = Frame.from_dict(json.loads(Path("selection.json").read_text()))
with EdgeTAM() as tracker:
    tracker.select(seed, point=(310, 240))
    observation = tracker.update(robot.frame(seed.camera))
    if observation.status == "tracked":
        print(observation.center, observation.bbox)
    else:
        print(observation.status)  # lost or stale: no current location to act on
```

Call `update` on new frames while a phase runs, then check the latest observation before submitting the next
phase. Check `status` whenever you use an observation: it turns `stale` once the frame is older than one second
(`max_age_s` changes that). If a file camera has not written a new frame, `update` returns the previous
observation without advancing tracking. Camera errors are raised to the procedure.

To turn a mask into numbers a plan can use, measure it in the frame it came from:
`robot.measure(frame, mask=observation.mask, target="block")` returns the same measurement as MCP's
`measure_pixels`, for frames captured with `depth=True`.

## Limits

An empty mask reports `lost`. `tracked` means the model produced a mask, not that it follows the right object.
Rotation and occlusion can cause loss or drift: look at the scene and `select` again when needed; a new selection
reuses the loaded weights. A point can pull in the object's cast shadow; a box constrains the selection, and it
helps to check the mask before using its geometry. Tracking does not estimate depth, plan a grasp or confirm a
placement by itself. Load the model before powering a physical arm; a camera or tracker failure does not change
the kernel's power rules.

File frames are timestamped with the file's modification time. HTTP and command cameras report no sensor time,
so their frames are timestamped when capture starts; they must serve current pictures, and a slow one-shot
capture command may be too slow for useful tracking. Recorded video is timestamped as it is read.

The tracker only runs forward: it keeps one prompt and a fixed window of recent model outputs (15 for this
checkpoint), and discards each picture after inference. Closing it frees the model, the session and the last
mask, although GPU allocators may keep caches while the process lives. Transformers is pinned because the
tracker relies on its session state.

The [model and weights](https://github.com/facebookresearch/EdgeTAM) are Apache-2.0. The
[checkpoint revision](https://huggingface.co/facebook/EdgeTAM/tree/f3a09791b2343c0733d456d08d73771d9363b69a) is
the official Transformers conversion. Once it is cached, set `HF_HUB_OFFLINE=1` to run offline; `--model-path`
takes a local copy of that revision.
