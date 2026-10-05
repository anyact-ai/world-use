# Track an object between agent decisions

The agent selects an object in a picture once. EdgeTAM follows it through later frames and returns a mask, a
bounding box and a centre in image pixels, so a Python procedure can keep track of the object between checked
phases without asking a model about every picture.

Tracking is optional and runs in the procedure's process. This example reads pictures and prints what it tracks;
it does not enable or move a robot. Agents on MCP get the same tracking as tools with `wu mcp --vision`; see
[measuring from pictures](../../docs/perception.md#tracking).

## Try a recording

From a checkout of world-use:

```sh
uv run --extra vision --with opencv-python-headless examples/tracking/track.py \
  --video /path/to/recording.mp4 --box 100 120 220 260 --preview latest.png
```

Replace the box with the object's coordinates in the **first frame**, in native pixels; `--point X Y` selects
with a single foreground point instead. The run processes at most 300 frames (`--frames`), and overwrites
`latest.png` with the newest annotated frame. OpenCV only decodes the video; EdgeTAM segments and tracks.

The first use downloads the pinned public checkpoint and its backbone metadata. Apple Silicon uses MPS, Linux with
a compatible NVIDIA setup uses CUDA, and other machines use the CPU; `--device` chooses explicitly. The CPU works,
with slower updates. Startup and the first inference can take several seconds. The Mac path has been exercised on
recorded footage; CUDA has not been tried.

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

Call `update` on new frames while a bounded phase runs, then check the latest observation before submitting the
next phase. The tracker keeps no queue of images and no video. Check `status` whenever you use an observation: it
turns `stale` as it ages (after one second by default; `max_age_s` changes that). A file camera that has not
written a new frame returns the same observation without advancing tracking. Camera errors reach the procedure.

To turn a mask into numbers a plan can use, measure it in the frame it came from:
`robot.measure(frame, mask=observation.mask, target="block")` returns the same measurement as MCP's
`measure_pixels`, for frames captured with `depth=True`.

## Limits and memory

An empty mask reports `lost`. `tracked` means the model produced a mask, not that it follows the right object.
Rotation and occlusion can cause loss or drift: look at the scene and `select` again when needed. A point can
pull in the object's cast shadow; a box constrains the selection, and checking the mask before using its geometry
helps. A new selection replaces the old session without reloading the weights. Tracking does not estimate depth,
plan a grasp or confirm a placement by itself. Load the model before powering a physical arm; a camera or tracker
failure does not change the kernel's power rules.

File frames carry the file's modification time and identity. HTTP and command cameras timestamp the start of
capture, because they report no sensor time, so they must serve current pictures; slow one-shot capture commands
may be too slow for useful tracking. Recorded video is timestamped as it is read.

Forward tracking keeps one prompt, a fixed window of recent model outputs (15 in this checkpoint, from the model
configuration) and one feature cache, and discards each picture after inference. Reverse playback and editing
past prompts are not supported. Closing the tracker frees the model, the session and the last mask; GPU
allocators may keep caches while the process lives. The Transformers version is pinned because this retention
relies on its session state.

The [model and weights](https://github.com/facebookresearch/EdgeTAM) are Apache-2.0. The
[checkpoint revision](https://huggingface.co/facebook/EdgeTAM/tree/f3a09791b2343c0733d456d08d73771d9363b69a) is
the official Transformers conversion. Once it is cached, set `HF_HUB_OFFLINE=1` to run offline; `--model-path`
takes a local copy of that revision.
