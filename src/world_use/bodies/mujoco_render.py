"""Offscreen cameras. OpenGL lives on one thread; rendering never holds the physics lock."""
import sys
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from .mujoco_scene import mj

THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mujoco-camera")


class CameraRenderer:
    def __init__(self):
        self.renderer = None
        self.model = None
        self.size = None
        self.closed = False

    def close(self):
        THREAD.submit(self._finish).result()

    def _finish(self):
        self.closed = True
        self._close()

    def _close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
            self.model = None

    def render(self, model, data, view):
        return THREAD.submit(self._render, model, data, view).result()

    def _render(self, model, data, view):
        if self.closed:
            raise RuntimeError("simulation camera is closed")
        size = (view.width, view.height)
        if self.model is not model or self.size != size:
            self._close()
            try:
                self.renderer = mj.Renderer(model, height=view.height, width=view.width)
            except Exception as e:
                hint = ("macOS needs GPU access for native CGL; hosted macOS VMs cannot render"
                        if sys.platform == "darwin" else
                        "on headless Linux install Mesa EGL or select MUJOCO_GL=osmesa with OSMesa installed")
                raise RuntimeError(f"MuJoCo camera needs OpenGL: {hint}") from e
            self.model, self.size = model, size
        assert self.renderer is not None
        option = mj.MjvOption()
        option.geomgroup[3] = False
        self.renderer.update_scene(data, scene_option=option)
        for camera in self.renderer.scene.camera:
            camera.pos = view.T[:3, 3]
            camera.forward = view.T[:3, 2]
            camera.up = -view.T[:3, 1]
            near = .01
            camera.frustum_near, camera.frustum_far = near, 20
            camera.frustum_bottom = -(view.height - view.cy) * near / view.fy
            camera.frustum_top = view.cy * near / view.fy
            camera.frustum_center = (view.width / 2 - view.cx) * near / view.fx
            camera.frustum_width = view.width * near / (2 * view.fx)
        return Image.fromarray(self.renderer.render())
