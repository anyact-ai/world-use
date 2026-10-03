"""Optional object tracking in the procedure's process, independent of robot control."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np

from .cameras import Frame

MODEL = "facebook/EdgeTAM"
REVISION = "f3a09791b2343c0733d456d08d73771d9363b69a"
TRANSFORMERS_VERSION = "5.15.1"


@dataclass(frozen=True)
class Observation:
    """Image-space evidence. Check status before using geometry; tracked does not mean correct."""

    camera: str
    frame_id: str
    timestamp: float
    bbox: tuple[int, int, int, int] | None
    center: tuple[float, float] | None
    mask: np.ndarray | None = field(repr=False)
    max_age_s: float = field(repr=False)

    @property
    def age_s(self) -> float:
        return max(0.0, time.monotonic() - self.timestamp)

    @property
    def status(self) -> str:
        if self.age_s > self.max_age_s:
            return "stale"
        return "tracked" if self.bbox is not None else "lost"

    def to_dict(self) -> dict:
        """Compact feedback, without image arrays or stale geometry."""
        status = self.status
        return dict(camera=self.camera, frame_id=self.frame_id, status=status, age_s=self.age_s,
                    bbox=self.bbox if status == "tracked" else None,
                    center=self.center if status == "tracked" else None)


class EdgeTAM:
    """Track one selected object through forward frames. Use as a context manager.

    select(frame, point=(x, y)) or select(frame, box=(x0, y0, x1, y1)) seeds a new
    session from that exact image. Box right/bottom edges are exclusive, as in PIL.crop.
    update(frame) returns a mask and pixel geometry using the same convention.
    Calls are synchronous; keep this object in the procedure, never in a Behavior.tick.
    One instance is used by one thread. Re-select after changing camera or image size.
    """

    def __init__(self, *, device: str = "auto", max_age_s: float = 1.0, model_path: str | Path | None = None):
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s must be finite and positive")
        if device not in {"auto", "mps", "cuda", "cpu"}:
            raise ValueError("device must be auto, mps, cuda or cpu")
        self.max_age_s = float(max_age_s)
        self._closed = False
        self._session = None
        self._last: Observation | None = None
        self._index = 0
        self._size = None
        self._camera = None
        self._load(device, model_path)

    def _load(self, device, model_path):
        try:
            import torch  # ty: ignore[unresolved-import]
            import transformers  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise ImportError("install world-use[vision] in this procedure's environment to use EdgeTAM") from e
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise RuntimeError(f"EdgeTAM history retention requires transformers=={TRANSFORMERS_VERSION}; "
                               "install world-use[vision]")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if ((device == "cuda" and not torch.cuda.is_available())
                or (device == "mps" and not torch.backends.mps.is_available())):
            raise RuntimeError(f"{device} is unavailable; choose an available device explicitly")
        self.device = device
        self._torch = torch
        self._dtype = torch.float32 if device == "cpu" else torch.float16
        source = str(model_path) if model_path is not None else MODEL
        revision = "main" if model_path is not None else REVISION
        self._model = transformers.EdgeTamVideoModel.from_pretrained(
            source, dtype=self._dtype, revision=revision)
        # Transformers' decorated .to loses its bound signature; it implements nn.Module's contract.
        cast(torch.nn.Module, self._model).to(device).eval()
        self._processor = transformers.Sam2VideoProcessor.from_pretrained(source, revision=revision)
        self._recent = max(self._model.config.num_maskmem - 1, self._model.config.max_object_pointers_in_encoder - 1)

    def __enter__(self) -> EdgeTAM:
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        """Release this instance's weights, session and last mask. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._session = self._last = self._model = self._processor = None
        if self.device == "mps":
            self._torch.mps.empty_cache()
        elif self.device == "cuda":
            self._torch.cuda.empty_cache()

    def _check_open(self):
        if self._closed:
            raise RuntimeError("tracker is closed")
        assert self._model is not None and self._processor is not None
        return self._model, self._processor

    @staticmethod
    def _check_frame(frame: Frame):
        if not frame.id or not frame.camera or not math.isfinite(frame.timestamp):
            raise ValueError("frame needs an identity, camera and finite monotonic timestamp")
        if frame.timestamp > time.monotonic():
            raise ValueError("frame timestamp is in the future; use this host's monotonic clock")

    def select(self, frame: Frame, *, point=None, box=None) -> Observation:
        """Select in native upright pixels. Replaces prior tracking state, retaining the loaded model.

        The seed may be an older image the agent inspected. Its result will be stale;
        consume a fresh update before making a decision about the current scene.
        """
        _, processor = self._check_open()
        self._check_frame(frame)
        if (point is None) == (box is None):
            raise ValueError("provide exactly one point or box")
        p = np.asarray(point if point is not None else box, dtype=float)
        if p.shape != ((2,) if point is not None else (4,)) or not np.isfinite(p).all():
            raise ValueError("point needs two finite coordinates; box needs four")
        w, h = frame.image.size
        if (p < 0).any() or (p[:2] >= [w, h]).any() or (box is not None and (p[2:] > [w, h]).any()):
            raise ValueError("selection is outside the frame's native pixel coordinates")
        if box is not None and (p[0] >= p[2] or p[1] >= p[3]):
            raise ValueError("box must have x0 < x1 and y0 < y1")
        self._session = processor.init_video_session(inference_device=self.device, dtype=self._dtype)
        self._index, self._last = 0, None
        self._size, self._camera = frame.image.size, frame.camera
        prompt = (dict(input_points=[[[p.tolist()]]], input_labels=[[[1]]]) if point is not None
                  else dict(input_boxes=[[p.tolist()]]))
        processor.add_inputs_to_inference_session(
            inference_session=self._session, frame_idx=0, obj_ids=1, original_size=(h, w), **prompt)
        return self._infer(frame)

    def update(self, frame: Frame) -> Observation:
        self._check_open()
        self._check_frame(frame)
        if self._session is None or self._last is None:
            raise RuntimeError("select an object before updating")
        if (frame.camera, frame.image.size) != (self._camera, self._size):
            raise ValueError("camera or image size changed; select the object again")
        if frame.id == self._last.frame_id:
            return self._last                       # a repeated file does not advance time or model memory
        if frame.timestamp < self._last.timestamp:
            raise ValueError("frames arrived out of order; select again to start a new sequence")
        if frame.age_s > self.max_age_s:
            return self._observation(frame, None)   # stale input cannot become tracking history
        return self._infer(frame)

    def _observation(self, frame, mask):
        bbox = center = None
        if mask is not None:
            ys, xs = np.nonzero(mask)
            if len(xs):
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                center = (float(xs.mean()), float(ys.mean()))
            mask.flags.writeable = False
        return Observation(frame.camera, frame.id, frame.timestamp, bbox, center, mask, self.max_age_s)

    def _infer(self, frame: Frame) -> Observation:
        model, processor = self._check_open()
        try:
            with self._torch.inference_mode():
                inputs = processor(images=frame.image, return_tensors="pt").to(self.device)
                result = model(inference_session=self._session, frame_idx=self._index,
                               frame=inputs.pixel_values[0].to(self._dtype))
                w, h = frame.image.size
                masks = processor.post_process_masks(
                    [result.pred_masks], original_sizes=[[h, w]], binarize=False)[0]
                mask = (masks[0, 0] > 0).cpu().numpy()
                _discard_history(self._session, self._index, self._recent)
                self._index += 1
                self._last = self._observation(frame, mask)
                return self._last
        except BaseException:
            self._session = self._last = None      # a partially advanced session cannot be reused
            raise


def _discard_history(session, index: int, recent: int):
    """Forward-only retention for the pinned Transformers version; prompt history has one entry.

    Six recent outputs feed mask memory and fifteen feed object pointers in this
    checkpoint. Keep the larger window. Explicit frame indices are essential:
    the upstream default derives the next index from the number of stored images.
    """
    session.processed_frames.clear()
    histories = [out["non_cond_frame_outputs"] for out in session.output_dict_per_obj.values()]
    histories.extend(session.frames_tracked_per_obj.values())
    for history in histories:
        for old in list(history):
            if old <= index - recent:
                del history[old]
