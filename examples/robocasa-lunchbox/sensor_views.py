"""A fixed side camera and the Panda's upstream eye-in-hand camera.

The moving camera snapshots RGB-D, optical pose and TCP from one physics state.
Its calibration identity represents the unchanged mounting/intrinsics; each Frame
contains the current base-to-camera transform. No object truth is exposed.
"""
import time
from dataclasses import replace

import mujoco as mj
import numpy as np

from world_use.cameras import Frame, SimCamera, View


def wrist_view(model, data, base, size=(1280, 960)):
    camera = data.camera('robot0_eye_in_hand')
    pose = np.eye(4)
    pose[:3, :3] = camera.xmat.reshape(3, 3) @ np.diag([1, -1, -1])
    pose[:3, 3] = camera.xpos
    w, h = size
    f = h / 2 / np.tan(np.radians(model.camera('robot0_eye_in_hand').fovy[0]) / 2)
    return View(np.linalg.inv(base) @ pose, f, f, w / 2, h / 2, w, h)


class WristCamera(SimCamera):
    def __init__(self, body):
        super().__init__('wrist', wrist_view(body.model, body.data, body.base), body)

    def capture(self, k, *, depth=False):
        body = self.body
        with body.lock:
            state = mj.MjData(body.model)
            mj.mj_copyData(state, body.model, body.data)
            timestamp, tool = time.monotonic(), body.tool_pose()
            view = wrist_view(body.model, state, body.base)
        image, distance = body.renderer.render(body.model, state, replace(view, T=body.base @ view.T), depth=True)
        self._calibration = (view, self.calibration_id)
        return Frame(image, self.name, timestamp=timestamp, view=view,
                     depth=distance if depth else None, calibration=self.calibration_id, tool=tool)

    def snap(self, k):
        return self.capture(k).image


def cameras(body):
    return {
        'side': SimCamera('side', View.look_at([.05, .80, .95], [.48, -.01, .24],
                                             fov_deg=49, size=(1280, 960)), body),
        'wrist': WristCamera(body),
    }
