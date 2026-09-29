"""Flight recorder: every control tick while connected, plus the numbers that say how a run went.

The summary answers the questions that decide whether a policy is worth running on hardware: how long were
the motors on, how much of that time did the robot actually move, how hot did it get.
"""
import json
import threading
from pathlib import Path

import numpy as np


class Tape:
    """Every control tick, kept in preallocated arrays a minute at a time: a tick writes one row and allocates
    nothing, and a copy of the record so far is a few array copies. So it can be saved while the robot runs,
    without churning Python on the interpreter the control loop shares."""
    CHUNK = 6000                      # rows per block: a minute at 100 Hz

    def __init__(self, n: int):
        self.n = n
        self._blocks: list[dict[str, np.ndarray]] = []
        self._used = self.CHUNK       # rows written in the last block
        self._lock = threading.Lock() # the control thread writes while a request thread may copy
        self._power: list[tuple[float, bool]] = []

    def mark_power(self, t: float, on: bool):
        """Power budget: from an enable attempt through confirmed disable, including incomplete transitions."""
        with self._lock:
            if not self._power or self._power[-1][1] != on:
                self._power.append((t, on))

    def _block(self, rows: int | None = None) -> dict[str, np.ndarray]:
        c, n = self.CHUNK if rows is None else rows, self.n
        return dict(t=np.zeros(c), enabled=np.zeros(c, bool), moving=np.zeros(c, bool), job=np.zeros(c, int),
                    q_cmd=np.zeros((c, n)), q=np.zeros((c, n)), tau=np.full((c, n), np.nan),
                    temp=np.full((c, n), np.nan), grip_cmd=np.full(c, np.nan), grip=np.full(c, np.nan),
                    grip_tau=np.full(c, np.nan))

    def add(self, t, enabled, moving, job, q_cmd, q, tau, temp, grip_cmd, grip, grip_tau):
        with self._lock:
            if self._used == self.CHUNK:
                self._blocks.append(self._block())
                self._used = 0
            b, i = self._blocks[-1], self._used
            b["t"][i], b["enabled"][i], b["moving"][i], b["job"][i] = t, enabled, moving, job
            b["q_cmd"][i], b["q"][i] = q_cmd, q
            if tau is not None:
                b["tau"][i] = tau
            if temp is not None:
                b["temp"][i] = temp
            for key, v in (("grip_cmd", grip_cmd), ("grip", grip), ("grip_tau", grip_tau)):
                if v is not None:
                    b[key][i] = v
            self._used = i + 1

    def __len__(self) -> int:
        return (len(self._blocks) - 1) * self.CHUNK + self._used if self._blocks else 0

    def arrays(self) -> dict:
        with self._lock:
            power = dict(power_t=np.array([t for t, _ in self._power]),
                         power_on=np.array([on for _, on in self._power], bool)) if self._power else {}
            if not self._blocks:
                return self._block(0) | power if power else {}
            blocks, used = list(self._blocks), self._used
            last = {key: v[:used].copy() for key, v in blocks[-1].items()}
        return {key: np.concatenate([b[key] for b in blocks[:-1]] + [last[key]]) for key in last} | power

    def summary(self, rate_hz: float, *, until: float | None = None) -> dict:
        """Elapsed durations, not tick counts divided by the nominal rate (kept for API compatibility)."""
        return self._summary(self.arrays(), until)

    @staticmethod
    def _summary(a: dict, until: float | None) -> dict:
        if not a:
            return dict(ticks=0)
        t = a.get("t", np.array([]))
        power_t, power_on = a.get("power_t", t), a.get("power_on", a.get("enabled", np.array([], bool)))
        end = max(until or 0.0, float(t[-1]) if len(t) else 0.0, float(power_t[-1]) if len(power_t) else 0.0)
        powered = float(np.maximum(0, np.diff(np.append(power_t, end))) @ power_on)
        timing = dict(time_basis="elapsed_monotonic", powered_s=round(powered, 1),
                      power_basis="enable_attempt_to_confirmed_disable" if "power_t" in a else "sampled_enabled")
        if not len(t):
            return dict(ticks=0, wall_s=round(end - power_t[0], 1), moving_s=0.0, moving_share=0.0, **timing)
        on = a["enabled"]
        moving = np.maximum(0, np.diff(np.append(t, end))) @ (on & a["moving"])
        with np.errstate(all="ignore"):
            err = np.abs(a["q"] - a["q_cmd"])[on & a["moving"]]
            temp = a["temp"][on] if on.any() else a["temp"]
            # how late the control loop ran while the motors were on: a busy host shows up here, not in the motion
            gaps = np.diff(a["t"])[on[1:] & on[:-1]] * 1000
            return dict(
                tick_ms=dict(median=round(float(np.median(gaps)), 1), p99=round(float(np.percentile(gaps, 99)), 1),
                             max=round(float(gaps.max()), 1)) if len(gaps) else None,
                ticks=len(t), wall_s=round(end - min(float(t[0]), float(power_t[0])), 1),
                **timing, moving_s=round(float(moving), 1),
                moving_share=round(float(moving / powered), 3) if powered > 0 else None,
                max_tracking_error_deg=np.round(np.degrees(np.nanmax(err, axis=0)), 2).tolist() if len(err) else None,
                max_abs_torque=(np.round(np.nanmax(np.abs(a["tau"]), axis=0), 2).tolist()
                                if np.isfinite(a["tau"]).any() else None),
                max_temp_c=np.round(np.nanmax(temp, axis=0), 1).tolist() if np.isfinite(temp).any() else None)

    def save(self, path: Path, rate_hz: float, *, until: float | None = None):
        a = self.arrays()
        if a:
            np.savez_compressed(path, **a)
        return self._summary(a, until)


def save_summary(path: Path, summary: dict):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
