"""Client-side target selection and inspection, shared by Python procedures and MCP.

An optional tracker supplies masks. Geometry remains registered by the daemon from
the exact source capture. No hidden background capture or motion occurs here.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from contextlib import suppress
from uuid import uuid4

import numpy as np
from PIL import ImageDraw

from .errors import Refused
from .perception import EvidenceStore, measure


def rectangle(frame, box):
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in box)
            or not 0 <= box[0] < box[2] <= frame.image.width
            or not 0 <= box[1] < box[3] <= frame.image.height):
        raise ValueError("box must be [left, top, right, bottom] within native pixels; right/bottom exclusive")
    mask = np.zeros((frame.image.height, frame.image.width), dtype=bool)
    mask[box[1]:box[3], box[0]:box[2]] = True
    return mask


def crop_image(image, *, box=None, max_side=1024):
    if isinstance(max_side, bool) or not isinstance(max_side, int) or not 64 <= max_side <= 2048:
        raise ValueError("max_side must be in 64..2048")
    bounds = [0, 0, image.width, image.height] if box is None else box
    from .cameras import Frame
    rectangle(Frame(image, "inspection"), bounds)
    result = image.crop(bounds)
    width, height = result.size
    scale = max_side / max(width, height)
    result = result.resize((max(1, round(width * scale)), max(1, round(height * scale))))
    return result, dict(size=list(result.size), native_size=list(image.size), crop=bounds,
                        native_from_image=[[width / result.width, 0, bounds[0]],
                                           [0, height / result.height, bounds[1]], [0, 0, 1]],
                        pixel_convention="pixel edges; point measurements sample the containing pixel centre")


class Perception:
    MAX_TARGETS = 4

    def __init__(self, client, tracker=None):
        self.client, self.tracker = client, tracker
        self.frames = OrderedDict()
        self.targets = {}
        self.lock = threading.RLock()
        self.inference = threading.Lock()
        self.closed = False

    def remember(self, frame):
        if EvidenceStore.size(frame) > EvidenceStore.MAX_BYTES:
            raise ValueError("frame exceeds capture capacity")
        with self.lock:
            self.frames[frame.id] = frame
            while (len(self.frames) > EvidenceStore.MAX_FRAMES
                   or sum(EvidenceStore.size(f) for f in self.frames.values()) > EvidenceStore.MAX_BYTES):
                self.frames.popitem(last=False)
        return frame

    def capture(self, camera=None, *, depth=False):
        return self.remember(self.client.frame(camera, depth=depth))

    def frame(self, identity):
        with self.lock:
            frame = self.frames.get(identity)
        if frame is None:
            raise Refused("frame expired; capture again", "data_unavailable")
        return frame

    def measure_pixels(self, frame, *, point=None, box=None, target=None):
        source = self.frame(frame)
        if (point is None) == (box is None):
            raise ValueError("provide exactly one point or box")
        mask = None if box is None else rectangle(source, box)
        return self.client.record(evidence=measure(source, point=point, mask=mask, target=target))

    def _begin(self):
        if self.closed:
            raise Refused("perception session is closed", "provider_unavailable")
        if self.tracker is None:
            raise Refused("selection needs a configured tracker; use explicit measure_pixels otherwise", "unsupported")
        if not self.inference.acquire(blocking=False):
            raise Refused("another target update is running", "busy")

    def _lost(self, target):
        with self.lock:
            self.targets[target]["lost"] = True
            self.targets[target]["observation"] = None
        self.client.invalidate_target(target)

    def _failed(self, target):
        affected = list(self.targets) if not getattr(self.tracker, "ready", True) else [target]
        for identity in affected:
            self._lost(identity)

    def _result(self, target, source, observation):
        if self.closed:
            raise Refused("perception session closed before inference completed", "provider_unavailable")
        if observation.frame_id != source.id or observation.camera != source.camera:
            raise ValueError("provider returned an observation from another capture")
        if observation.status != "tracked":
            self._lost(target)
            return dict(target=target, frame=source.id, status=observation.status, evidence=None,
                        age_s=source.age_s, reason="reselect after lost or stale correspondence")
        provider = getattr(self.tracker, "info", dict(provider=type(self.tracker).__name__))
        receipt = self.client.record(evidence=measure(source, mask=observation.mask, target=target),
                                     context=dict(provider=provider, label=self.targets[target]["label"]))
        with self.lock:
            self.targets[target]["observation"] = observation
        return dict(target=target, label=self.targets[target]["label"], status="tracked",
                    observation=observation.to_dict(),
                    evidence=receipt, identity="selection_and_correspondence_claim", provider=provider,
                    age_s=source.age_s)

    def select_target(self, frame, *, point=None, box=None, label=None, replace_target=None):
        source = self.frame(frame)
        if (point is None) == (box is None):
            raise ValueError("provide exactly one point or box")
        if box is not None:
            rectangle(source, box)
        else:
            # Validate native pixel coordinates without using metric geometry.
            p = np.asarray(point, float)
            if p.shape != (2,) or not np.isfinite(p).all() or (p < 0).any() or (p >= source.image.size).any():
                raise ValueError("point must lie within native pixels")
        if label is not None and (not isinstance(label, str) or not 0 < len(label) <= 128):
            raise ValueError("label must contain 1..128 characters")
        self._begin()
        assert self.tracker is not None
        identity = uuid4().hex
        try:
            if replace_target is not None:
                if replace_target not in self.targets:
                    raise ValueError("replace_target must be an existing selection")
                self._lost(replace_target)
                self.tracker.forget(replace_target)
                del self.targets[replace_target]
            if len(self.targets) >= self.MAX_TARGETS:
                raise Refused("four targets are already selected; replace an existing target", "target_capacity")
            self.targets[identity] = dict(camera=source.camera, session=source.session, calibration=source.calibration,
                                          label=label, lost=False, observation=None)
            try:
                observation = self.tracker.select(identity, source, **({"point": point} if point is not None
                                                                       else {"box": box}))
                return self._result(identity, source, observation)
            except Exception:
                self._failed(identity)
                raise
        finally:
            self.inference.release()

    def observe_targets(self, targets, *, depth=True, frames=None):
        if (not isinstance(targets, list) or not 1 <= len(targets) <= self.MAX_TARGETS
                or len(set(targets)) != len(targets) or any(t not in self.targets for t in targets)):
            raise ValueError("targets must contain 1..4 distinct existing target IDs")
        self._begin()
        assert self.tracker is not None
        try:
            sources = {}
            for target in targets:
                name = self.targets[target]["camera"]
                if name not in sources:
                    source = self.capture(name, depth=depth) if frames is None else self.frame(frames[name])
                    if source.camera != name:
                        raise ValueError("capture camera does not match target camera")
                    sources[name] = source
            results = []
            for target in targets:
                state = self.targets[target]
                source = sources[state["camera"]]
                if state["lost"] or (state["session"], state["calibration"]) != (source.session, source.calibration):
                    self._lost(target)
                    results.append(dict(target=target, status="lost", evidence=None, reason="reselect_target"))
                    continue
                try:
                    observation = self.tracker.update(target, source)
                    results.append(self._result(target, source, observation))
                except (Refused, ValueError) as e:
                    self._failed(target)
                    results.append(dict(target=target, status="lost", evidence=None,
                                        reason=getattr(e, "rule", "provider_error"), message=str(e)))
            times = [source.timestamp for source in sources.values()]
            return dict(observations=results, frames={name: f.id for name, f in sources.items()},
                        capture_skew_s=max(times) - min(times))
        finally:
            self.inference.release()

    def inspect_image(self, frame, *, box=None, target=None, max_side=1024):
        source = self.frame(frame)
        image = source.image.copy()
        if target is not None:
            with self.lock:
                observation = self.targets.get(target, {}).get("observation")
            if observation is None or observation.frame_id != source.id:
                raise ValueError("target overlay must come from this exact frame")
            draw = ImageDraw.Draw(image)
            ys, xs = np.nonzero(observation.mask)
            stride = max(1, len(xs) // 1000)
            for x, y in zip(xs[::stride], ys[::stride], strict=True):
                draw.point((int(x), int(y)), fill=(20, 240, 90))
        image, metadata = crop_image(image, box=box, max_side=max_side)
        return image, dict(metadata, frame=source.id, camera=source.camera, timestamp=source.timestamp,
                           age_s=source.age_s, historical=False, target=target)

    def close(self):
        self.closed = True
        for target in list(self.targets):
            with suppress(OSError):
                self._lost(target)
        if self.tracker is not None:
            self.tracker.close()
        self.frames.clear()
        self.targets.clear()
