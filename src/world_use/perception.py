"""Measure the visible surface under some pixels of a camera frame, and keep measurements a run can require.

The daemon holds recent frames and the measurements made from them. A run that requires a measurement gets a
guard the kernel calls before each step; it reads no files or images. Nothing here recognizes objects, changes
the world model or moves the robot.
"""
from __future__ import annotations

import math
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .cameras import INK, Frame, unpack
from .errors import Refused
from .recorder import save_arrays

GREEN, RED = (30, 220, 100), (230, 60, 60)


@dataclass(frozen=True)
class Measurement:
    """Surface points under the selected pixels of one frame. An invalid one says why and has no points."""
    frame: Frame = field(repr=False)
    points: np.ndarray = field(repr=False)      # base frame, metres
    pixels: np.ndarray = field(repr=False)      # native pixel cells (x, y) the points come from
    reason: str | None
    diagnostics: dict
    target: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)

    @property
    def valid(self) -> bool:
        return self.reason is None

    @property
    def center(self) -> np.ndarray | None:
        """The median visible surface point (base frame): not an object's centre, which the camera cannot see."""
        return np.median(self.points, axis=0) if self.valid else None

    def summary(self, work: np.ndarray) -> dict:
        """What an agent acts on, in metres in the frame plans use; work is that frame's pose in the base frame."""
        def local(p):
            return (np.asarray(p, float) - work[:3, 3]) @ work[:3, :3]

        out = dict(id=self.id, camera=self.frame.camera, frame=self.frame.id, target=self.target,
                   valid=self.valid, reason=self.reason, surface_center=None, visible_bounds=None, from_tool=None,
                   **self.diagnostics)
        if self.valid:
            center = local(self.center)
            out.update(surface_center=_metres(center),
                       visible_bounds=_metres(np.quantile(local(self.points), [.02, .98], axis=0)))
            if self.frame.tool is not None:
                out["from_tool"] = _metres(center - local(self.frame.tool[:3, 3]))
        return out

    def overlay(self) -> Image.Image:
        """The frame with the measured pixels and the surface centre drawn on it."""
        image = self.frame.image.copy()
        draw = ImageDraw.Draw(image, "RGBA")
        colour = GREEN if self.valid else RED
        for x, y in self.pixels[::max(1, len(self.pixels) // 400)]:
            draw.rectangle((x - 1, y - 1, x + 1, y + 1), fill=colour + (200,))
        if self.valid and self.frame.view is not None:
            (u, v), = self.frame.view.project([self.center])[0]
            for width, ink in ((5, (0, 0, 0, 255)), (2, colour + (255,))):
                draw.line([(u - 9, v), (u + 9, v)], fill=ink, width=width)
                draw.line([(u, v - 9), (u, v + 9)], fill=ink, width=width)
        font = ImageFont.load_default(size=max(12, image.width // 55))
        caption = f"{self.frame.camera} | {self.target or 'surface'}: {self.reason or 'measured'} | {self.id[:8]}"
        h = font.getbbox("Ag")[3] + 10
        draw.rectangle([0, image.height - h, image.width, image.height], fill=(255, 255, 255, 215))
        draw.text((8, image.height - h + 4), caption, fill=INK, font=font)
        return image


def _metres(value) -> list:
    return np.round(value, 4).tolist()


def measure(frame: Frame, *, point=None, mask=None, target: str | None = None) -> Measurement:
    """Measure native pixels with the frame's optical-z depth and calibration.

    A point samples the pixel that contains it. A mask is eroded by one pixel to drop mixed boundary samples, at
    most 2048 of its pixels are sampled, and it needs at least 16 of them and half with valid depth. A depth gap
    over 5 cm inside the middle 80% of the samples means two surfaces were selected, and is refused.
    """
    if (point is None) == (mask is None):
        raise ValueError("provide exactly one point or mask")
    if target is not None and (not isinstance(target, str) or not 0 < len(target) <= 128):
        raise ValueError("target must be a name of 1..128 characters")
    w, h = frame.image.size
    if point is not None:
        p = np.asarray(point, dtype=float)
        if p.shape != (2,) or not np.isfinite(p).all() or (p < 0).any() or (p >= [w, h]).any():
            raise ValueError(f"point must be [x, y] within the frame's {w}x{h} native pixels")
        pixels = np.floor(p).astype(int)[None, :]
    else:
        mask = np.asarray(mask)
        if mask.shape != (h, w) or mask.dtype != bool:
            raise ValueError("mask must be boolean and aligned to the frame's native pixels")
        inside = np.zeros_like(mask)
        inside[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
                              & mask[1:-1, :-2] & mask[1:-1, 2:])
        y, x = np.nonzero(inside)
        pixels = np.column_stack([x, y])
        if len(pixels) > 2048:
            pixels = pixels[np.linspace(0, len(pixels) - 1, 2048).astype(int)]
    least = 1 if point is not None else 16
    points = np.empty((0, 3))
    diagnostics: dict = dict(samples=len(pixels))
    reason = None
    if frame.view is None:
        reason = "missing_calibration"
    elif frame.depth is None:
        reason = "missing_depth"
    elif len(pixels) < least:
        reason = "insufficient_support"
    else:
        z = frame.depth[pixels[:, 1], pixels[:, 0]]
        valid = np.isfinite(z) & (z > 0)
        diagnostics["valid_fraction"] = round(float(valid.mean()), 3)
        if valid.mean() < .5 or valid.sum() < least:
            reason = "invalid_depth"
        else:
            pixels, z = pixels[valid], z[valid]
            if len(z) >= 16:
                gaps = np.diff(np.sort(z))
                if np.max(gaps[max(1, int(len(z) * .1)):int(len(z) * .9)], initial=0) > .05:
                    reason = "mixed_depth_surfaces"
            if reason is None:
                points = frame.view.unproject(pixels + .5, z)
                diagnostics["depth_spread_m"] = round(float(np.quantile(z, .9) - np.quantile(z, .1)), 4)
    points.flags.writeable = pixels.flags.writeable = False
    return Measurement(frame, points, pixels, reason, diagnostics, target)


def rectangle(box, size) -> np.ndarray:
    """The mask of a box [left, top, right, bottom] in whole native pixels; right and bottom are exclusive."""
    w, h = size
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in box)
            or not 0 <= box[0] < box[2] <= w or not 0 <= box[1] < box[3] <= h):
        raise ValueError(f"box must be [left, top, right, bottom] in whole pixels within {w}x{h}, "
                         "right and bottom exclusive")
    mask = np.zeros((h, w), dtype=bool)
    mask[box[1]:box[3], box[0]:box[2]] = True
    return mask


@dataclass(frozen=True)
class Source:
    """Where a measurement came from: all a run's guard needs, so it reads no files or images."""
    camera: str
    calibration: str | None
    timestamp: float
    reason: str | None


class Measurements:
    """The daemon's recent frames, and the measurements made from them in this session.

    Frames stay while they fit (8 frames, 64 MiB together). A measurement keeps only its source, so a run can
    require it after its frame is gone; the 256 most recent stay known.
    """
    FRAMES = 8
    FRAME_BYTES = 64 * 1024 * 1024
    KEPT = 256

    def __init__(self, k, cameras: dict):
        self.k, self.cameras = k, cameras
        self.frames: OrderedDict[str, Frame] = OrderedDict()
        self.sources: OrderedDict[str, Source] = OrderedDict()
        self.withdrawn: set[str] = set()
        self.lock = threading.Lock()
        self.now = time.monotonic                 # frames are stamped on this host's monotonic clock

    def keep(self, frame: Frame) -> Frame:
        if _size(frame) > self.FRAME_BYTES:
            raise Refused("this frame alone is larger than the 64 MiB frame cache", "frame_size")
        with self.lock:
            self.frames.pop(frame.id, None)
            self.frames[frame.id] = frame
            while len(self.frames) > self.FRAMES or sum(map(_size, self.frames.values())) > self.FRAME_BYTES:
                self.frames.popitem(last=False)
        return frame

    def frame(self, identity) -> Frame:
        with self.lock:
            frame = self.frames.get(identity)
        if frame is None:
            raise Refused(f"frame {identity!r} is no longer held (the daemon keeps the last {self.FRAMES})",
                          "missing_frame", "capture a new frame")
        return frame

    def measure(self, frame: str, *, point=None, box=None, mask=None, target: str | None = None) -> dict:
        """Measure, save the overlay and points with the run, and record one event. Returns the summary."""
        source = self.frame(frame)
        if sum(v is not None for v in (point, box, mask)) != 1:
            raise ValueError("give exactly one of point, box or mask")
        w, h = source.image.size
        if box is not None:
            mask = rectangle(box, (w, h))
        elif mask is not None:
            if not isinstance(mask, str):
                raise ValueError("mask must be packed: cameras.pack(np.packbits(mask).tobytes())")
            mask = np.unpackbits(np.frombuffer(unpack(mask, (w * h + 7) // 8), np.uint8))[:w * h]
            mask = mask.reshape(h, w).astype(bool)
        m = measure(source, point=point, mask=mask, target=target)
        k = self.k
        with k.lock:
            work = k.world.frame("work").T.copy()
            capture_t = k.clock.now() - k.t0 - source.age_s
        folder = (k.run_dir or Path(tempfile.gettempdir()) / "world-use") / "perception"
        folder.mkdir(parents=True, exist_ok=True)
        # Written here, before the measurement exists: a full disk fails this request, not a later run.
        m.overlay().save(folder / f"{m.id}.png")
        save_arrays(folder / f"{m.id}.npz", dict(points=m.points, pixels=m.pixels))
        with self.lock:
            self.sources[m.id] = Source(source.camera, source.calibration, source.timestamp, m.reason)
            while len(self.sources) > self.KEPT:
                self.sources.popitem(last=False)
        out: dict[str, Any] = dict(m.summary(work), age_s=round(source.age_s, 2), capture_t=round(capture_t, 3))
        c, name = out["surface_center"], m.target or "surface"
        where = f"{name}: {m.reason}" if c is None else f"{name} at F{c[0]:+.3f} L{c[1]:+.3f} U{c[2]:+.3f}"
        k.emit("measurement", where, measurement=dict(out, image=f"perception/{m.id}.png"))
        return dict(out, image=str(folder / f"{m.id}.png"))

    def withdraw(self, measurements, reason: str) -> list[str]:
        """Runs that require these measurements stop before their next step, e.g. after a tracker lost the target."""
        if not isinstance(measurements, list) or not all(isinstance(m, str) for m in measurements):
            raise ValueError("measurements must be a list of measurement IDs")
        self.withdrawn.update(measurements)
        if measurements:
            self.k.emit("withdrawn", f"{len(measurements)} measurement(s) withdrawn: {reason}",
                        measurements=measurements, reason=reason)
        return measurements

    def guard(self, requires) -> Callable[[], None] | None:
        """For a run requiring [{"evidence": measurement ID, "max_age_s": seconds}, ...]: a check that raises
        Refused once one of them is older than its limit, was withdrawn, or its camera was calibrated again."""
        if not isinstance(requires, list) or len(requires) > 16:
            raise ValueError("requires is a list of at most 16 measurements")
        needs: list[tuple[str, Source, float]] = []
        with self.lock:
            for item in requires:
                if not isinstance(item, dict) or set(item) != {"evidence", "max_age_s"}:
                    raise ValueError('each requirement is {"evidence": a measurement ID, "max_age_s": seconds}')
                evidence, limit = item["evidence"], item["max_age_s"]
                if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not 0 < limit < math.inf:
                    raise ValueError("max_age_s must be a positive number of seconds")
                source = self.sources.get(evidence)
                if source is None:
                    raise Refused(f"no measurement {evidence!r} in this daemon session", "unknown_measurement",
                                  "capture a frame and measure again")
                if source.reason is not None:
                    raise Refused(f"measurement {evidence} found no surface ({source.reason})",
                                  "invalid_measurement", "measure a region with valid depth")
                needs.append((evidence, source, float(limit)))
        if not needs:
            return None

        def guard():
            now = self.now()
            for evidence, source, limit in needs:
                details = dict(hint="capture a frame, measure again and resubmit", evidence=evidence)
                camera = self.cameras.get(source.camera)
                if evidence in self.withdrawn:
                    raise Refused(f"measurement {evidence} was withdrawn", "stale_measurement", **details)
                if camera is None or camera.calibration_id != source.calibration:
                    raise Refused(f"camera {source.camera!r} was calibrated again after measurement {evidence}",
                                  "stale_measurement", **details)
                age = now - source.timestamp
                if not 0 <= age <= limit:
                    raise Refused(f"measurement {evidence} is {age:.1f} s old; this run allows {limit:g} s",
                                  "stale_measurement", **details)

        return guard


def _size(frame: Frame) -> int:
    return frame.image.width * frame.image.height * 3 + (0 if frame.depth is None else frame.depth.nbytes)
