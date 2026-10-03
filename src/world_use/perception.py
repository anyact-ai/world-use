"""Measured visible surfaces. No model inference, world mutation, or robot commands."""
from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, replace
from uuid import uuid4

import numpy as np
from PIL import ImageDraw

from .cameras import Frame, pack, unpack
from .errors import Refused


@dataclass(frozen=True)
class Measurement:
    frame: Frame = field(repr=False)
    selection: dict = field(repr=False)
    points: np.ndarray = field(repr=False)
    pixels: np.ndarray = field(repr=False)
    reason: str | None
    diagnostics: dict
    target: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    available_at: float = field(default_factory=time.monotonic)

    @property
    def valid(self) -> bool:
        return self.reason is None

    def request(self) -> dict:
        """Send support, not claimed coordinates: the daemon recomputes from its captured frame."""
        f = self.frame
        return dict(frame=f.id, session=f.session, calibration=f.calibration,
                    selection=deepcopy(self.selection), target=self.target)

    def to_dict(self) -> dict:
        f = self.frame
        geometry = None
        if self.valid:
            geometry = dict(surface_center=np.median(self.points, axis=0).tolist(),
                            visible_bounds=np.quantile(self.points, [.02, .98], axis=0).tolist())
        return dict(id=self.id, frame=f.id, camera=f.camera, session=f.session, calibration=f.calibration,
                    timestamp=f.timestamp, available_at=self.available_at, timing=f.timing, capture_t=f.elapsed,
                    target=self.target, valid=self.valid, reason=self.reason,
                    method="depth_surface", coordinates="base", units="m", geometry=geometry,
                    diagnostics=self.diagnostics,
                    tool=None if f.tool is None else f.tool.tolist())

    def preview(self):
        image = self.frame.image.copy()
        draw = ImageDraw.Draw(image)
        for x, y in self.pixels[::max(1, len(self.pixels) // 150)]:
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(30, 220, 100))
        draw.rectangle((0, 0, image.width, 24), fill=(20, 25, 30))
        draw.text((8, 6), f"{self.target or 'surface'}: {self.reason or 'measured visible surface'}", fill="white")
        return image


def measure(frame: Frame, *, point=None, mask=None, target: str | None = None) -> Measurement:
    """Measure native pixels with optical-z depth. A region describes only its visible surfaces.

    Region masks are eroded by one pixel to reject mixed boundary samples. At least
    16 supported pixels and 50% valid depth are required. Sample at most 2048 pixels.
    Small fit/spread diagnostics are not an absolute calibration-accuracy guarantee.
    """
    if (point is None) == (mask is None):
        raise ValueError("provide exactly one point or mask")
    if target is not None and (not isinstance(target, str) or not target or len(target) > 128):
        raise ValueError("target must be a nonempty name of at most 128 characters")
    w, h = frame.image.size
    if point is not None:
        p = np.asarray(point, dtype=float)
        if p.shape != (2,) or not np.isfinite(p).all() or (p < 0).any() or (p >= [w, h]).any():
            raise ValueError("point must be within the native upright image")
        # Coordinates index pixel cells; depth is sampled and unprojected at their centres.
        pixels = np.floor(p).astype(int)[None, :]
        selection = dict(point=p.tolist())
    else:
        mask = np.asarray(mask)
        if mask.shape != (h, w) or mask.dtype != bool:
            raise ValueError("mask must be boolean and aligned to the native upright image")
        selection = dict(mask=pack(np.packbits(mask).tobytes()))
        inside = np.zeros_like(mask)
        inside[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
                              & mask[1:-1, :-2] & mask[1:-1, 2:])
        y, x = np.nonzero(inside)
        pixels = np.column_stack([x, y])
        if len(pixels) > 2048:
            pixels = pixels[np.linspace(0, len(pixels) - 1, 2048).astype(int)]
    points = np.empty((0, 3))
    diagnostics = dict(samples=len(pixels), valid_samples=0, valid_fraction=0.0)
    reason = None
    if frame.view is None:
        reason = "missing_calibration"
    elif frame.depth is None:
        reason = "missing_depth"
    elif len(pixels) < (1 if point is not None else 16):
        reason = "insufficient_support"
    else:
        z = frame.depth[pixels[:, 1], pixels[:, 0]]
        valid = np.isfinite(z) & (z > 0)
        diagnostics.update(valid_samples=int(valid.sum()), valid_fraction=float(valid.mean()))
        if valid.mean() < .5 or valid.sum() < (1 if point is not None else 16):
            reason = "invalid_depth"
        else:
            pixels, z = pixels[valid], z[valid]
            ordered = np.sort(z)
            differences = np.diff(ordered)
            if len(z) >= 16:
                lo, hi = max(1, int(len(z) * .1)), int(len(z) * .9)
                if np.max(differences[lo:hi], initial=0) > .05:
                    reason = "mixed_depth_surfaces"
            if reason is None:
                points = frame.view.unproject(pixels + .5, z)
                diagnostics["depth_spread_m"] = float(np.quantile(z, .9) - np.quantile(z, .1))
    points.flags.writeable = pixels.flags.writeable = False
    return Measurement(frame, selection, points, pixels, reason, diagnostics, target)


def from_selection(frame: Frame, selection: dict, *, target=None) -> Measurement:
    if not isinstance(selection, dict) or set(selection) not in ({"point"}, {"mask"}):
        raise ValueError("selection needs a point or a packed mask")
    if "point" in selection:
        return measure(frame, point=selection["point"], target=target)
    h, w = frame.image.height, frame.image.width
    packed = unpack(selection["mask"], (h * w + 7) // 8)
    mask = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[:h * w].reshape(h, w).astype(bool)
    return measure(frame, mask=mask, target=target)


@dataclass(frozen=True)
class Requirement:
    """Copied live metadata; check performs no I/O or inference."""
    evidence: str
    session: str
    camera: str
    calibration: str
    timestamp: float
    max_age_s: float

    def check(self, session, calibrations, now):
        if self.session != session or calibrations.get(self.camera) != self.calibration:
            raise Refused("evidence belongs to a different session or calibration", "stale_evidence",
                          "capture and measure again", evidence=self.evidence)
        age = now - self.timestamp
        if not math.isfinite(age) or age < 0 or age > self.max_age_s:
            raise Refused(f"evidence is {age:.2f} s old; limit {self.max_age_s:g} s", "stale_evidence",
                          "capture and measure again", evidence=self.evidence)


class EvidenceStore:
    """Bounded daemon-owned captures and small registered receipts. Geometry is recomputed on registration."""
    MAX_BYTES = 64 * 1024 * 1024
    MAX_FRAMES = 8
    MAX_RECEIPTS = 256

    def __init__(self, kernel):
        self.k = kernel
        self.session = kernel.evidence_session
        self.frames: OrderedDict[tuple, Frame] = OrderedDict()
        self.receipts: OrderedDict[str, dict] = OrderedDict()
        self.bytes = 0
        self.lock = threading.Lock()

    @staticmethod
    def size(frame):
        return frame.image.width * frame.image.height * 3 + (0 if frame.depth is None else frame.depth.nbytes)

    def remember(self, frame: Frame) -> Frame:
        if self.k.evidence_closed:
            raise Refused("session has closed", "closed")
        frame = replace(frame, session=self.session)
        size = self.size(frame)
        if size > self.MAX_BYTES:
            raise Refused("frame exceeds acquisition cache capacity", "frame_size")
        key = (frame.id, frame.calibration)
        with self.lock:
            old = self.frames.pop(key, None)
            if old:
                self.bytes -= self.size(old)
            self.frames[key] = replace(frame, image=frame.image.copy())
            self.bytes += size
            while len(self.frames) > self.MAX_FRAMES or self.bytes > self.MAX_BYTES:
                _, old = self.frames.popitem(last=False)
                self.bytes -= self.size(old)
        return frame

    def register(self, request: dict) -> dict:
        if self.k.evidence_closed:
            raise Refused("session has closed", "closed")
        if not isinstance(request, dict):
            raise ValueError("evidence must be a measurement request")
        if request.get("session") != self.session:
            raise Refused("frame belongs to another session", "stale_evidence", "capture a new frame")
        with self.lock:
            frame = self.frames.get((request.get("frame"), request.get("calibration")))
        if frame is None:
            raise Refused("source frame is no longer in the acquisition cache", "missing_frame", "capture again")
        result = from_selection(frame, request["selection"], target=request.get("target"))
        receipt = result.to_dict()
        receipt["available_t"] = self.k.clock.now() - self.k.t0
        # Only the request thread touches arrays. Control receives a copy of the small receipt.
        with self.lock:
            self.receipts[result.id] = receipt
            while len(self.receipts) > self.MAX_RECEIPTS:
                self.receipts.popitem(last=False)
        self.k.emit("evidence", f"{result.target or 'surface'}: {result.reason or 'measured'}", measurement=receipt)
        if self.k.journal is not None:
            metadata = dict(receipt, format_version=1, selection=result.selection,
                            view=None if frame.view is None else frame.view.to_dict())
            folder = self.k.run_dir / "perception" / result.id

            def save():
                from .recorder import _atomic, save_arrays, save_summary
                folder.mkdir(parents=True, exist_ok=True)
                _atomic(folder / "rgb.png", lambda f: frame.image.save(f, format="PNG"))
                _atomic(folder / "overlay.png", lambda f: result.preview().save(f, format="PNG"))
                arrays = dict(points=result.points, pixels=result.pixels)
                if frame.depth is not None:
                    arrays["depth"] = frame.depth
                save_arrays(folder / "surfaces.npz", arrays)
                save_summary(folder / "measurement.json", metadata)

            self.k.journal.artifact(result.id, self.size(frame) + result.points.nbytes + result.pixels.nbytes,
                                    save, dict(measurement=receipt, path=f"perception/{result.id}"))
        return deepcopy(receipt)

    def resolve(self, requires) -> tuple[Requirement, ...]:
        if not isinstance(requires, list) or len(requires) > 16:
            raise ValueError("requires must be a list of at most 16 evidence prerequisites")
        out = []
        with self.lock:
            for item in requires:
                if not isinstance(item, dict) or set(item) != {"evidence", "max_age_s"}:
                    raise ValueError("each prerequisite needs evidence and max_age_s")
                age = item["max_age_s"]
                if isinstance(age, bool) or not isinstance(age, (int, float)) or not math.isfinite(age) or age <= 0:
                    raise ValueError("max_age_s must be finite and positive")
                r = self.receipts.get(item["evidence"])
                if r is None or not r["valid"]:
                    raise Refused("evidence is missing or has no valid geometry", "invalid_evidence", "remeasure")
                out.append(Requirement(r["id"], r["session"], r["camera"], r["calibration"], r["timestamp"], age))
        return tuple(out)
