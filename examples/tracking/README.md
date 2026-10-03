# Track an object between agent decisions

The agent selects an object in an image. EdgeTAM follows it through later frames,
returning a mask, bounding box and centre in image pixels. A Python procedure can
use these observations between checked robot phases without calling a frontier
model for every image.

Tracking is optional and runs in the procedure's process. The example reads images
and prints observations; it does not enable or move a robot.

## Try a recording

From a checkout of world-use:

```sh
uv run --extra vision --with opencv-python-headless examples/tracking/track.py \
  --video /path/to/recording.mp4 --box 100 120 220 260 --preview latest.png
```

Replace the box with the object's coordinates in the **first frame**, in native
pixels. `--point X Y` selects with a single foreground point instead. The default
run processes at most 300 frames. `latest.png` is overwritten, not accumulated.
OpenCV only decodes the example video; EdgeTAM performs segmentation and tracking.

The first use downloads the pinned public checkpoint and its backbone metadata.
Apple Silicon uses MPS, Linux with a compatible NVIDIA setup uses CUDA, and other
machines use CPU. `--device` chooses explicitly. CPU is useful for compatibility,
but expect slower updates. Startup and the first inference can take several seconds.
The Mac path has been exercised on recorded footage; CUDA has not yet been validated here.

## Use a configured camera

Start the daemon with your camera configuration. Save the exact image and frame
metadata the agent will inspect:

```python
import json
from pathlib import Path
from world_use.client import Client

frame = Client().frame("side")
frame.image.save("selection.png")
Path("selection.json").write_text(json.dumps(frame.to_dict()))
```

After choosing a point in `selection.png`:

```sh
uv run --extra vision examples/tracking/track.py \
  --seed selection.json --point 310 240 --preview latest.png
```

`Client.frame()` returns an upright, unannotated image without writing a flight
record. It preserves native resolution. `wu look` may scale and annotate its
image, so do not transfer its pixel coordinates to a native frame.

For a procedure, the complete tracking interface is:

```python
import json
from pathlib import Path
from world_use.cameras import Frame
from world_use.client import Client
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

Call `update` on successive frames in the procedure's loop while a bounded phase
executes, then check the current observation before submitting the next phase.
The helper does not queue images or retain a video. Check `status` when consuming
an observation; it becomes `stale` as it ages (one second by default, configurable
with `max_age_s`). Repeated file frames return the same observation without
advancing tracking. Camera read errors propagate to the procedure.

## Limits and memory

An empty mask reports `lost`. `tracked` means the model produced a mask, not that
the identity is correct. Rotation and occlusion can cause loss or drift; inspect
the scene and call `select` again when needed. A point prompt can include the
object's cast shadow; use a box to constrain the selection and inspect the mask
before using its geometry. A new selection replaces the old
session without reloading the weights. Tracking does not estimate depth, establish
a grasp, or verify a placement by itself. Initialize the model before powering a
physical arm; camera or tracker failure does not change the kernel's power policy.

File frames use the file's modification time and identity. HTTP and command
cameras timestamp the start of acquisition because sensor timestamps are unavailable;
those sources must serve current images. Slow one-shot capture commands may not
support useful tracking rates. Recorded-video timestamps refer to replay acquisition.

Forward tracking retains one prompt, a fixed window of recent model outputs and
one feature cache. Processed images are discarded after each inference. The window
comes from the model configuration (15 outputs in this checkpoint); reverse playback
and historical prompt editing are unsupported. Closing the context releases the
model, session and last mask. GPU allocators may retain caches while the process lives.
The Transformers version is pinned because the retention logic uses its session state.

The [model and weights](https://github.com/facebookresearch/EdgeTAM) are Apache-2.0.
The [checkpoint revision](https://huggingface.co/facebook/EdgeTAM/tree/f3a09791b2343c0733d456d08d73771d9363b69a)
is the official Transformers conversion. Once cached, set `HF_HUB_OFFLINE=1` to
run offline; `--model-path` accepts a local copy of that revision.
