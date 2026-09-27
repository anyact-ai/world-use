"""Flight recorder: every control tick while connected, plus the numbers that say how a run went.

The summary answers the questions that decide whether a policy is worth running on hardware: how long were
the motors on, how much of that time did the robot actually move, how hot did it get.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Tape:
    def __init__(self, n: int):
        self.n = n
        self.rows: list[tuple] = []

    def add(self, t, enabled, moving, job, q_cmd, q, tau, temp, grip_cmd, grip, grip_tau):
        nan = np.full(self.n, np.nan)
        self.rows.append((t, enabled, moving, job, np.asarray(q_cmd, float), np.asarray(q, float),
                          nan if tau is None else np.asarray(tau, float), nan if temp is None else np.asarray(temp, float),
                          np.nan if grip_cmd is None else grip_cmd, np.nan if grip is None else grip,
                          np.nan if grip_tau is None else grip_tau))

    def arrays(self) -> dict:
        if not self.rows:
            return {}
        cols = list(zip(*self.rows))
        return dict(t=np.array(cols[0]), enabled=np.array(cols[1], bool), moving=np.array(cols[2], bool),
                    job=np.array(cols[3], int), q_cmd=np.array(cols[4]), q=np.array(cols[5]), tau=np.array(cols[6]),
                    temp=np.array(cols[7]), grip_cmd=np.array(cols[8]), grip=np.array(cols[9]), grip_tau=np.array(cols[10]))

    def summary(self, rate_hz: float) -> dict:
        a = self.arrays()
        if not a:
            return dict(ticks=0)
        on = a["enabled"]
        powered = on.sum() / rate_hz
        moving = (on & a["moving"]).sum() / rate_hz
        with np.errstate(all="ignore"):
            err = np.abs(a["q"] - a["q_cmd"])[on & a["moving"]]
            temp = a["temp"][on] if on.any() else a["temp"]
            return dict(
                ticks=len(a["t"]), wall_s=round(float(a["t"][-1] - a["t"][0]), 1),
                powered_s=round(float(powered), 1), moving_s=round(float(moving), 1),
                moving_share=round(float(moving / powered), 3) if powered > 0 else None,
                max_tracking_error_deg=np.round(np.degrees(np.nanmax(err, axis=0)), 2).tolist() if len(err) else None,
                max_abs_torque=np.round(np.nanmax(np.abs(a["tau"]), axis=0), 2).tolist() if np.isfinite(a["tau"]).any() else None,
                max_temp_c=np.round(np.nanmax(temp, axis=0), 1).tolist() if np.isfinite(temp).any() else None)

    def save(self, path: Path, rate_hz: float):
        a = self.arrays()
        if a:
            np.savez_compressed(path, **a)
        return self.summary(rate_hz)


def save_summary(path: Path, summary: dict):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
