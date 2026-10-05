"""Optional tracking: select an object once, then measure it again in later frames without selecting it again.

A tracker (EdgeTAM in its own process, `wu mcp --vision`) turns the selection into a mask in each new frame, and
the daemon measures that mask the way measure_pixels measures a box. Tracking runs only when asked: never in the
background and never in the control loop.
"""
from __future__ import annotations

from collections import deque
from contextlib import suppress

from .client import Client


class Tracking:
    """Selected targets by name, each with its camera and the measurements made of it since it was selected."""

    def __init__(self, client: Client, tracker):
        self.client, self.tracker = client, tracker
        self.targets: dict[str, dict] = {}

    def select(self, frame: str, target: str, *, point=None, box=None) -> dict:
        """Select target by a point or box in a frame the agent looked at, and measure it. A seed older than the
        tracker accepts (an agent's thinking time can do that) is followed into a new frame before measuring."""
        if not isinstance(target, str) or not 0 < len(target) <= 128:
            raise ValueError("target must be a name of 1..128 characters")
        if (point is None) == (box is None):
            raise ValueError("select with exactly one point or box")
        source = self.client.frame(id=frame)
        if target in self.targets:
            self._lose(target, "reselected")
        try:
            observation = self.tracker.select(target, source, **(dict(point=point) if box is None else dict(box=box)))
            if observation.status == "stale":
                source = self.client.frame(source.camera, depth=source.depth is not None)
                observation = self.tracker.update(target, source)
        except BaseException:
            with suppress(Exception):
                self.tracker.forget(target)
            if not getattr(self.tracker, "ready", True):
                for name in list(self.targets):
                    self._lose(name, "the tracker stopped during selection")
            raise
        self.targets[target] = dict(camera=source.camera, depth=source.depth is not None, measured=deque(maxlen=256))
        return self._measure(target, source, observation)

    def observe(self, targets: list[str]) -> list[dict]:
        """Measure selected targets again, in one new frame per camera."""
        if not isinstance(targets, list) or not targets or len(set(targets)) != len(targets):
            raise ValueError("targets must be a nonempty list of distinct names")
        for name in targets:
            if name not in self.targets:
                raise ValueError(f"no selected target {name!r}: select it first")
        frames, results = {}, []
        for name in targets:
            if name not in self.targets:          # a tracker that stopped took every target with it
                continue
            key = self.targets[name]["camera"], self.targets[name]["depth"]
            if key not in frames:
                frames[key] = self.client.frame(key[0], depth=key[1])
            try:
                observation = self.tracker.update(name, frames[key])
            except ValueError as e:               # Refused included; busy is not a loss
                if getattr(e, "rule", "") == "busy":
                    raise
                lost = list(self.targets) if not getattr(self.tracker, "ready", True) else [name]
                results += [self._lose(target, str(e)) for target in lost]
                continue
            results.append(self._measure(name, frames[key], observation))
        return results

    def _measure(self, name, frame, observation) -> dict:
        if observation.status != "tracked":
            return dict(self._lose(name, f"the tracker reports {observation.status}; select it again"),
                        frame=frame.id)
        measurement = self.client.measure(frame, mask=observation.mask, target=name)
        history = self.targets[name]["measured"]
        if len(history) == history.maxlen:
            # An accepted job can outlive the daemon's measurement cache. Retire the
            # oldest source before forgetting it so later loss cannot leave a usable guard.
            self.client.withdraw([history[0]], "tracking history expired; use a newer measurement")
        history.append(measurement["id"])
        return dict(measurement, tracking="tracked")

    def _lose(self, name, why) -> dict:
        """Drop a target the tracker no longer follows and withdraw what was measured of it."""
        state = self.targets.pop(name, None)
        with suppress(Exception):
            self.tracker.forget(name)
        withdrawn = list(state["measured"]) if state else []
        if withdrawn:
            self.client.withdraw(withdrawn, f"tracking lost {name!r}")
        return dict(target=name, tracking="lost", valid=False, reason=why, withdrawn=withdrawn)

    def close(self):
        self.targets.clear()
        self.tracker.close()
