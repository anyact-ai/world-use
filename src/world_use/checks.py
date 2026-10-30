"""Measured convergence for native controllers; this module never commands a robot."""
from __future__ import annotations

from collections import deque

import numpy as np


class Settling:
    """Bound a final-target wait using fresh, nonnegative residuals in the caller's units.

    Tolerances may mix units (e.g. metres and radians). The caller must set command_active=False while
    an approver/controller is still changing the applied target. Missing feedback is not convergence.
    A stall is little progress over a full window, not a collision or unreachability diagnosis.
    """

    def __init__(self, tolerances, timeout_s: float, window_s: float, progress_fraction: float = .05,
                 stable_s: float = 0.):
        self.tolerances = np.atleast_1d(np.asarray(tolerances, float)).copy()
        if (self.tolerances.ndim != 1 or not self.tolerances.size
                or not np.isfinite(self.tolerances).all() or np.any(self.tolerances <= 0)):
            raise ValueError("settling tolerances must be finite and positive")
        if (not np.isfinite([timeout_s, window_s, progress_fraction, stable_s]).all()
                or not 0 < window_s <= timeout_s or not 0 < progress_fraction < 1
                or not 0 <= stable_s < timeout_s):
            raise ValueError("settling needs 0 < window <= timeout and 0 < progress_fraction < 1")
        self.timeout_s, self.window_s, self.progress_fraction = timeout_s, window_s, progress_fraction
        self.started: float | None = None
        self.stamp: float | None = None
        self.samples: deque[tuple[float, np.ndarray]] = deque()
        self.elapsed_s = 0.
        self.stable_s, self.arrived_since = stable_s, None

    def update(self, stamp: float, errors, *, command_active: bool = True) -> str | None:
        errors = np.atleast_1d(np.asarray(errors, float))
        if (not np.isfinite(stamp) or errors.shape != self.tolerances.shape
                or not np.isfinite(errors).all() or np.any(errors < 0)):
            raise ValueError("settling needs a finite timestamp and matching nonnegative residuals")
        if self.stamp is not None and stamp < self.stamp:
            raise ValueError("settling feedback time moved backwards")
        if not command_active:
            self.samples.clear()
            self.arrived_since = None
        if self.stamp == stamp:
            return None                       # a cached sample cannot complete a wait
        self.stamp = stamp
        if self.started is None:
            self.started = stamp
        self.elapsed_s = stamp - self.started
        if self.elapsed_s > self.timeout_s:
            return "timeout"
        scaled = errors / self.tolerances
        if command_active and np.all(scaled <= 1):
            if self.arrived_since is None:
                self.arrived_since = stamp
            if stamp - self.arrived_since + 1e-9 >= self.stable_s:
                return "arrived"
        else:
            self.arrived_since = None
        if self.elapsed_s >= self.timeout_s:
            return "timeout"
        if not command_active:
            return None
        self.samples.append((stamp, scaled.copy()))
        while len(self.samples) > 3 and stamp - self.samples[1][0] >= self.window_s:
            self.samples.popleft()
        span = stamp - self.samples[0][0]
        if len(self.samples) < 3 or span < self.window_s:
            return None
        unmet = scaled > 1
        progress = np.maximum(0, self.samples[0][1] - np.min([v for _, v in self.samples], axis=0))
        projected = scaled - progress / span * (self.timeout_s - self.elapsed_s)
        if np.any(unmet) and np.all(progress[unmet] < self.progress_fraction) and np.all(projected[unmet] > 1):
            return "stalled"
        return None
