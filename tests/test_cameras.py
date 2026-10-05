"""360 cameras: pinhole cuts of an equirectangular picture, and where the world lands in them."""
import subprocess

import numpy as np
import pytest
from PIL import Image

from world_use import World
from world_use.cameras import Camera, CommandCamera, EquirectCut, equirect_dirs, equirect_uv, from_config
from world_use.client import DaemonError


class Still(Camera):
    def __init__(self, img):
        super().__init__("still")
        self.img = img

    def snap(self, k):
        return self.img


def _marked(w, h, uvs):
    """A black 360 picture with a small white dot at each (u, v)."""
    pano = np.zeros((h, w, 3), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    for u, v in uvs:
        pano[(xx - (u * w - 0.5)) ** 2 + (yy - (v * h - 0.5)) ** 2 <= 9] = 255
    return Image.fromarray(pano)


def _dot(img) -> np.ndarray:
    a = np.asarray(img, float).sum(axis=2)
    ys, xs = np.nonzero(a > 0.5 * a.max())
    return np.array([xs.mean() + 0.5, ys.mean() + 0.5])


def test_the_convention_of_a_360_picture():
    assert np.allclose(equirect_dirs(0.5, 0.5), [-1, 0, 0], atol=1e-12)     # the middle looks along -x
    assert np.allclose(equirect_dirs(0.3, 0.0), [0, 0, 1], atol=1e-12)      # the top edge looks straight up
    assert np.allclose(equirect_dirs(0.75, 0.5), [0, 1, 0], atol=1e-12)     # three quarters across: +y
    d = np.random.default_rng(0).normal(size=(50, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    assert np.allclose(equirect_dirs(*equirect_uv(d)), d, atol=1e-9)


def test_uncalibrated_the_cut_is_aimed_by_yaw_and_pitch():
    cut = EquirectCut("x5", Still(_marked(1440, 720, [(0.5 + 30 / 360, 0.5 - 20 / 180)])), 60.0, (400, 300),
                      yaw_deg=30, pitch_deg=20)
    assert cut.view is None and np.allclose(_dot(cut.picture(None)), [200, 150], atol=1.0)


def test_a_known_point_lands_in_a_calibrated_cut_where_its_view_says():
    """The cut's pixels and its drawings agree, so what the kernel knows is drawn over what the 360 sees."""
    world = World()
    cfg = dict(name="x5", path="unused", projection="equirect", frame="base", eye=[0.6, -0.9, 0.7],
               facing=[-0.3, 0.8, -0.4], up=[0, 0, 1], look_at=[0.3, 0.0, 0.1], fov_deg=70, size=[800, 500])
    cut = from_config(cfg, world)
    assert isinstance(cut, EquirectCut) and cut.view is not None and cut.pose is not None
    for P in ([0.3, 0.0, 0.1], [0.36, 0.07, 0.13], [0.22, -0.05, 0.02]):
        d = cut.pose[:3, :3].T @ (np.asarray(P) - cut.pose[:3, 3])
        cut.source = Still(_marked(2880, 1440, [equirect_uv(d)]))
        (want,), _ = cut.view.project([P])
        assert np.linalg.norm(_dot(cut.picture(None)) - want) < 1.5
    assert len(cut._tables) == 1                                         # resampled through one table


def test_file_frames_preserve_identity_age_and_rotation(tmp_path):
    """A capture app that died must not hand the policy an old picture as if it were now."""
    import os
    import time

    path = tmp_path / "camera.png"
    image = Image.new("RGB", (12, 8))
    image.putpixel((0, 0), (255, 0, 0))
    image.save(path)
    os.utime(path, (time.time() - 2, time.time() - 2))
    camera = from_config({"name": "side", "path": str(path), "max_age_s": 5, "rotate": 90}, World())
    first, again = camera.capture(None), camera.capture(None)
    assert first.id == again.id and 1.9 < first.age_s < 3
    assert np.array_equal(first.image, camera.picture(None))
    assert first.image.size == (8, 12)                                  # turned a quarter clockwise
    # Atomic replacement with the same modification time is still a different frame.
    replacement = tmp_path / "new.png"
    image.save(replacement)
    stat = path.stat()
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    replacement.replace(path)
    assert camera.capture(None).id != first.id
    os.utime(path, (time.time() - 10, time.time() - 10))
    with pytest.raises(RuntimeError, match="newest frame"):
        camera.capture(None)
    path.unlink()
    with pytest.raises(RuntimeError, match="no frame at"):
        camera.capture(None)
    with pytest.raises(ValueError, match="rotate"):
        from_config({"name": "side", "path": str(path), "rotate": 45}, World())


def test_360_cut_preserves_source_frame_identity_and_age(tmp_path):
    from world_use.cameras import FileCamera

    path = tmp_path / "pano.png"
    Image.new("RGB", (200, 100), "red").save(path)
    source = FileCamera("pano", path)
    cut = EquirectCut("front", source, 60, (80, 60))
    one, two = cut.capture(None), cut.capture(None)
    assert one.id == two.id and one.camera == "front"
    assert one.timestamp == pytest.approx(two.timestamp, abs=.01)
    assert np.array_equal(one.image, cut.picture(None))


def test_command_camera_timeout_is_reported_to_the_client(monkeypatch, daemon):
    d, c = daemon
    camera = CommandCamera("side", "capture", timeout=0.25)
    d.cameras["side"] = camera

    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(camera.command, camera.timeout)

    monkeypatch.setattr(subprocess, "run", timed_out)
    with pytest.raises(RuntimeError, match=r"camera 'side'.*0\.25"):
        camera.capture(None)
    with pytest.raises(DaemonError) as error:
        c.look("side")
    assert error.value.code == 502
    assert "side" in error.value.body["error"] and "0.25" in error.value.body["error"]
