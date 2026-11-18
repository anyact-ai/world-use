"""Measured convergence and contact for native controllers; this module never commands a robot."""
from __future__ import annotations

from collections import deque

import numpy as np


class Settling:
    """Bound a final-target wait using fresh, nonnegative residuals in the caller's units.

    Tolerances may mix units (e.g. metres and radians). The caller must set command_active=False while
    an approver/controller is still changing the applied target. Missing feedback is not convergence.
    A stall is little progress over a full window, not a collision or unreachability diagnosis. An arm
    that swings past its target also makes no progress while it turns round: stall_s is how long a stall
    must last before it is reported.
    """

    def __init__(self, tolerances, timeout_s: float, window_s: float, progress_fraction: float = .05,
                 stable_s: float = 0., stall_s: float = 0.):
        self.tolerances = np.atleast_1d(np.asarray(tolerances, float)).copy()
        if (self.tolerances.ndim != 1 or not self.tolerances.size
                or not np.isfinite(self.tolerances).all() or np.any(self.tolerances <= 0)):
            raise ValueError("settling tolerances must be finite and positive")
        if (not np.isfinite([timeout_s, window_s, progress_fraction, stable_s, stall_s]).all()
                or not 0 < window_s <= timeout_s or not 0 < progress_fraction < 1
                or not 0 <= stable_s < timeout_s or stall_s < 0):
            raise ValueError("settling needs 0 < window <= timeout, 0 < progress_fraction < 1, "
                             "0 <= stable_s < timeout and stall_s >= 0")
        self.timeout_s, self.window_s, self.progress_fraction = timeout_s, window_s, progress_fraction
        self.started: float | None = None
        self.stamp: float | None = None
        self.samples: deque[tuple[float, np.ndarray]] = deque()
        self.elapsed_s = 0.
        self.stable_s, self.arrived_since = stable_s, None
        self.stall_s, self.stalled_since = stall_s, None

    def update(self, stamp: float, errors, *, command_active: bool = True) -> str | None:
        repeated = self.stamp == stamp
        result = self._judge(stamp, errors, command_active)
        if result != "stalled":
            # A repeated sample says nothing new: it neither ends a stall nor extends it.
            if not (repeated and command_active):
                self.stalled_since = None
            return result
        if self.stalled_since is None:
            self.stalled_since = stamp
        return "stalled" if stamp - self.stalled_since + 1e-9 >= self.stall_s else None

    def _judge(self, stamp: float, errors, command_active: bool) -> str | None:
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


class Following:
    """Tell a position-controlled hand that something holds from one that only trails its command.

    A free hand follows a moving command at a lag and gains on it every step. Call update after each
    control step with the measured position and the command that step applied: it returns True once the
    hand has gained less than progress_m toward a command more than lead_m ahead of it, twice
    running. Nearer than lead_m nothing is concluded, because a hand still carrying momentum can drift
    from its command for a few steps, which settling waits out. This needs no torque sensing; it does
    not say what holds the hand. travel_m is the distance the hand moved over the last step.
    """

    def __init__(self, position, *, lead_m: float, progress_m: float):
        if not (np.isfinite([lead_m, progress_m]).all() and lead_m > 0 and progress_m > 0):
            raise ValueError("following needs a positive lead and progress")
        self.lead_m, self.progress_m = lead_m, progress_m
        self.before = np.array(position, float)
        self.held, self.travel_m = 0, 0.

    def update(self, position, command) -> bool:
        here = np.array(position, float)
        lead = np.asarray(command, float) - self.before
        distance = float(np.linalg.norm(lead))
        gained = float((here - self.before) @ lead) / distance if distance > self.lead_m else np.inf
        self.travel_m = float(np.linalg.norm(here - self.before))
        self.held = self.held + 1 if gained < self.progress_m else 0
        self.before = here
        return self.held >= 2
