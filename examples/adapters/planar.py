"""Six-method adapter example. An in-memory position controller, with no hardware I/O or dynamics."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from world_use import JointState, Kernel, VirtualClock, World, check
from world_use.config import load_robot


class PlanarBody:
    simulated = True

    def __init__(self, manifest=None, start=(0.0, 0.5)):
        self.manifest = manifest or load_robot(Path(__file__).with_suffix(".toml"))
        self.q = np.asarray(start, float).copy()
        self.target = self.q.copy()
        self.on = False
        self.t = 0.0

    def connect(self):
        return JointState(self.t, self.q.copy())

    def enable(self):
        self.target, self.on = self.q.copy(), True

    def read(self):
        self.t += 1 / self.manifest.rate_hz
        if self.on:
            self.q = self.target.copy()
        return JointState(self.t, self.q.copy())

    def command(self, q, dq, gripper, gripper_v=0.0):
        self.target = np.asarray(q, float).copy()

    def disable(self):
        self.on = False

    def close(self):
        pass                         # closing transport must not silently release a physical robot


def main():
    body = PlanarBody()
    k = Kernel(body, World(), VirtualClock(body.manifest.rate_hz))
    k.connect()
    k.enable()
    phase = {"do": "joints", "delta_deg": {"1": 10, "2": -5}}
    report = check(phase, k)
    assert report.ok, report
    assert k.run(phase).ok
    k.release()
    k.close()
    print("Planar adapter: rehearsal, execution and release passed")


if __name__ == "__main__":
    main()
