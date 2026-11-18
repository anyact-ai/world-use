"""Optional object tracking in the procedure's process, independent of robot control."""
from __future__ import annotations

import hashlib
import math
import time
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np

from .cameras import Frame

MODEL = "facebook/EdgeTAM"
REVISION = "f3a09791b2343c0733d456d08d73771d9363b69a"
TRANSFORMERS_VERSION = "5.15.1"


def dominant_region(mask: np.ndarray) -> dict:
    """The mask's largest 8-connected region, and how much of the mask lies outside it.

    Small stray predictions can stretch the whole mask's bounding box across the image, so these
    numbers describe only the largest region; other_area_px and components say what was left out.
    width_px is the region's narrowest extent over all directions, length_px its extent across that,
    and axis_deg the direction of its length, from image x toward image y in (-90, 90]: the
    narrowest box around it, not its bounding box.
    """
    pixels = np.asarray(mask, dtype=bool)
    if pixels.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    # Runs of set pixels along each row, end exclusive, in raster order.
    edges = np.diff(np.pad(pixels, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    rows, starts = np.nonzero(edges == 1)
    ends = np.nonzero(edges == -1)[1]
    if not len(rows):
        return dict(bbox=None, center=None, area_px=0, other_area_px=0, components=0,
                    axis_deg=None, length_px=0., width_px=0.)
    parent = np.arange(len(rows))

    def root(i):
        while parent[i] != i:
            parent[i] = i = parent[parent[i]]
        return i

    # A run joins each run in the row above that it touches, diagonally included.
    first = np.searchsorted(rows, np.arange(pixels.shape[0] + 1))
    for j in range(len(rows)):
        y = rows[j]
        if y == 0:
            continue
        above = slice(first[y - 1], first[y])
        lo = first[y - 1] + np.searchsorted(ends[above], starts[j])
        hi = first[y - 1] + np.searchsorted(starts[above], ends[j], side="right")
        for i in range(lo, hi):
            a, b = root(i), root(j)
            parent[max(a, b)] = min(a, b)
    labels = np.array([root(i) for i in range(len(rows))])
    lengths = ends - starts
    sizes = np.bincount(labels, weights=lengths)
    # The largest component; of equal ones, the first in raster order, whose label is its first run.
    best = int(np.flatnonzero(sizes == sizes.max())[0])
    keep = labels == best
    rows, starts, ends, lengths = rows[keep], starts[keep], ends[keep], lengths[keep]
    area = int(lengths.sum())
    center = [float(((starts + ends - 1) * lengths).sum() / (2 * area)), float((rows * lengths).sum() / area)]
    # Extents of the pixel centres along every whole degree, plus the pixel itself. In (-89, 90] degrees
    # the image x weight is not negative, so each row's leftmost and rightmost pixels bound them all.
    index = np.unique(rows, return_index=True)[1]
    line = rows[index]
    points = np.concatenate([np.column_stack([line, np.minimum.reduceat(starts, index)]),
                             np.column_stack([line, np.maximum.reduceat(ends - 1, index)])])
    angles = np.radians(np.arange(-89, 91))
    along = points[:, 1:2] * np.cos(angles) + points[:, 0:1] * np.sin(angles)
    extents = along.max(axis=0) - along.min(axis=0) + 1
    narrow = int(np.argmin(extents))                    # the direction across the narrowest side
    return dict(bbox=[int(starts.min()), int(rows.min()), int(ends.max()), int(rows.max()) + 1],
                center=center, area_px=area, other_area_px=int(pixels.sum()) - area,
                components=int(np.count_nonzero(sizes)),
                axis_deg=float((narrow + 90) % 180 - 89), length_px=float(extents[(narrow + 90) % 180]),
                width_px=float(extents[narrow]))


@dataclass(frozen=True)
class Observation:
    """A tracker's result for one frame, in pixels. Check status before using bbox or center;
    "tracked" does not mean the mask is right."""

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
        # The last capture's image encoding, shared with forks so that several targets encode a frame once.
        self._features: dict = {}
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

    def fork(self) -> EdgeTAM:
        """Independent target history sharing weights and one capture's encoding. Serialize calls."""
        self._check_open()
        child = copy(self)
        child._session = child._last = child._size = child._camera = None
        child._index = 0
        return child

    def __exit__(self, *_):
        self.close()

    def close(self):
        """Release this instance's weights, session and last mask. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._session = self._last = self._model = self._processor = None
        self._features = {}             # other live forks retain their shared entry
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
        assert self._session is not None
        try:
            with self._torch.inference_mode():
                inputs = processor(images=frame.image, return_tensors="pt").to(self.device)
                # Keyed by capture, so identical images from two captures are encoded separately,
                # and by a pixel digest, so a reused frame id cannot serve another image's encoding.
                key = (frame.camera, frame.id, frame.image.mode, frame.image.size,
                       hashlib.sha256(frame.image.tobytes()).digest(), self.device, self._dtype)
                if self._features.get("key") == key:
                    self._session.cache.cache_vision_features(self._index, self._features["value"])
                result = model(inference_session=self._session, frame_idx=self._index,
                               frame=inputs.pixel_values[0].to(self._dtype))
                features = self._session.cache.get_vision_features(self._index)
                self._features.clear()
                if features is not None:
                    self._features.update(key=key, value=features)
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
            self._features.clear()
            raise


def _discard_history(session, index: int, recent: int):
    """Drop session history the model no longer reads, for the pinned Transformers version.

    This checkpoint reads the last six outputs for mask memory and the last fifteen for object
    pointers, so the larger window is kept; the prompt frame's output is never dropped. Stored images
    are cleared too, which is why frames are passed with explicit indices: by default, upstream
    derives the next index from the number of stored images.
    """
    session.processed_frames.clear()
    histories = [out["non_cond_frame_outputs"] for out in session.output_dict_per_obj.values()]
    histories.extend(session.frames_tracked_per_obj.values())
    for history in histories:
        for old in list(history):
            if old <= index - recent:
                del history[old]
