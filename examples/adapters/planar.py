"""Six-method adapter example. An in-memory position controller, with no hardware I/O or dynamics."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from world_use import JointSpec, JointState, Kernel, Manifest, VirtualClock, World, check

MANIFEST = Manifest("Planar example", Path(__file__).with_suffix(".urdf"), "tool",
                    (JointSpec("shoulder", -2.5, 2.5), JointSpec("elbow", -2.5, 2.5)))


class PlanarBody:
    manifest = MANIFEST
    simulated = True

    def __init__(self):
        self.q = np.array([0.0, 0.5])
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
    k = Kernel(PlanarBody(), World(), VirtualClock(MANIFEST.rate_hz))
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
