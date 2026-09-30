"""Cameras: what a policy looks at. `wu look` saves one picture per call, with what the kernel knows drawn on it.

A camera is a name, a way to get a picture and, optionally, a calibration: where it is and how it projects. With a
calibration the kernel draws the tool point, the work axes, the boxes it knows about and a planned path onto the
picture, so a model can see at a glance whether its world model matches the scene. A simulated robot gets
simulated cameras that render the scene the simulator is running (its truth, which the kernel may not fully know).

Workcell entry (positions in the work frame, metres):

    [[camera]]
    name = "side"
    path = "~/frames/side.jpg"                     # the newest frame a capture app keeps writing (refused when
    max_age_s = 3                                  # older than this); or url = "http://.../snapshot.jpg", or
                                                   # command = "imagesnap -q -" (prints an image to stdout)
    rotate = 180                                   # optional: 90, 180 or 270 clockwise, for a camera mounted turned
    eye = [0.35, -0.60, 0.40]                      # calibration, optional: where the camera is,
    look_at = [0.30, 0.0, 0.15]                    # what the image centre shows,
    fov_deg = 55                                   # and its horizontal field of view
"""
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MAX_SIDE = 1024                        # pictures are scaled down to this: plenty for a model, cheap in context

INK = (32, 36, 42)
BG = (243, 244, 246)
GRID = (214, 218, 224)
KIND = {"surface": (201, 178, 143), "object": (226, 128, 60), "keep_out": (220, 60, 60), "fragile": (60, 170, 210),
        "slow": (230, 190, 40)}
MODEL = (22, 140, 80)                  # what the kernel believes: outlines
TOOL = (200, 30, 160)
PLAN = (40, 90, 230)
AXES = {"F": (215, 50, 50), "L": (40, 160, 70), "U": (50, 90, 220)}


@dataclass
class View:
    """A pinhole camera: its pose in the base frame (x right, y down, z along the view) and its intrinsics."""
    T: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def look_at(cls, eye, target, fov_deg: float = 55.0, size=(800, 600), up=(0.0, 0.0, 1.0)) -> View:
        eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
        z = (target - eye) / np.linalg.norm(target - eye)
        x = np.cross(z, up)
        if np.linalg.norm(x) < 1e-6:                   # looking straight along `up`: image top = forward instead
            x = np.cross(z, [1.0, 0.0, 0.0])
        x /= np.linalg.norm(x)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(z, x), z, eye
        w, h = size
        f = (w / 2) / np.tan(np.radians(fov_deg) / 2)
        return cls(T, f, f, w / 2, h / 2, int(w), int(h))

    def scaled(self, width: int, height: int) -> View:
        """The same camera on a picture of another size. Pixels stay square and the horizontal field of view (what
        fov_deg means) is kept, with the optical centre as far from the middle as it was: a 16:9 camera described
        at the default 800x600 used to come out squashed, with fx and fy different."""
        s = width / self.width
        return View(self.T, self.fx * s, self.fy * s, width / 2 + (self.cx - self.width / 2) * s,
                    height / 2 + (self.cy - self.height / 2) * s, width, height)

    def project(self, pts) -> tuple[np.ndarray, np.ndarray]:
        """Pixels of base-frame points, and each point's depth along the view (<= 0 means behind the camera)."""
        c = (np.atleast_2d(np.asarray(pts, float)) - self.T[:3, 3]) @ self.T[:3, :3]
        z = c[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.stack([self.fx * c[:, 0] / z + self.cx, self.fy * c[:, 1] / z + self.cy], axis=1)
        return uv, z


def view_from_config(cfg: dict, world, size=(800, 600)) -> View | None:
    """A calibration from a workcell camera entry: eye/look_at/fov_deg in the work frame, if given."""
    if "eye" not in cfg or "look_at" not in cfg:
        return None
    frame = cfg.get("frame", "work")
    up = world.frame(frame).T[:3, :3] @ np.asarray(cfg.get("up", [0.0, 0.0, 1.0]), float)
    return View.look_at(world.to_base(frame, cfg["eye"]), world.to_base(frame, cfg["look_at"]),
                        float(cfg.get("fov_deg", 55.0)), tuple(cfg.get("size", size)), up)


# -- cameras ------------------------------------------------------------------------------------------

ROTATE = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}


class Camera:
    fov_deg: float | None = None      # the field of view it is said to have: a calibration's prior

    def __init__(self, name: str, view: View | None = None, rotate: int = 0):
        if rotate not in (0, *ROTATE):
            raise ValueError(f"camera {name!r}: rotate is 0, 90, 180 or 270 (degrees clockwise), not {rotate!r}")
        self.name, self.view, self.rotate = name, view, rotate

    def snap(self, k) -> Image.Image:
        raise NotImplementedError

    def picture(self, k) -> Image.Image:
        """The picture as the camera serves it: turned upright if it is mounted turned."""
        img = self.snap(k)
        return img.transpose(ROTATE[self.rotate]) if self.rotate else img


class HttpCamera(Camera):
    """Any camera that serves a still image over HTTP (an IP camera, a phone app, a small snapshot server)."""

    def __init__(self, name: str, url: str, view: View | None = None, timeout: float = 5.0, rotate: int = 0):
        super().__init__(name, view, rotate)
        self.url, self.timeout = url, timeout

    def snap(self, k) -> Image.Image:
        with urllib.request.urlopen(self.url, timeout=self.timeout) as r:
            return Image.open(BytesIO(r.read())).convert("RGB")


class CommandCamera(Camera):
    """A command that prints one image to stdout, e.g. `imagesnap -q -` or an ffmpeg one-frame grab."""

    def __init__(self, name: str, command: str, view: View | None = None, timeout: float = 15.0, rotate: int = 0):
        super().__init__(name, view, rotate)
        self.command, self.timeout = command, timeout

    def snap(self, k) -> Image.Image:
        out = subprocess.run(self.command, shell=True, capture_output=True, timeout=self.timeout)
        if out.returncode != 0 or not out.stdout:
            raise RuntimeError(f"camera {self.name!r}: {self.command!r} failed: {out.stderr.decode()[-300:].strip()}")
        return Image.open(BytesIO(out.stdout)).convert("RGB")


class FileCamera(Camera):
    """The newest frame a capture app keeps writing to a file (how a Mac's cameras reach a process that may not open
    them itself). A frame older than max_age_s is refused: a capture that died must not hand the policy an old
    picture as if it were now."""

    def __init__(self, name: str, path: str | Path, view: View | None = None, max_age_s: float = 3.0,
                 rotate: int = 0):
        super().__init__(name, view, rotate)
        self.path, self.max_age_s = Path(path).expanduser(), float(max_age_s)

    def snap(self, k) -> Image.Image:
        try:
            age = time.time() - self.path.stat().st_mtime
        except FileNotFoundError:
            raise RuntimeError(f"camera {self.name!r}: no frame at {self.path}") from None
        if age > self.max_age_s:
            raise RuntimeError(f"camera {self.name!r}: the newest frame is {age:.0f} s old (max_age_s "
                               f"{self.max_age_s:g}): is the capture running?")
        try:                          # read once: a writer that renames frames into place cannot mix two of them
            return Image.open(BytesIO(self.path.read_bytes())).convert("RGB")
        except OSError:               # caught mid-write by a writer that does not rename: one more try
            return Image.open(BytesIO(self.path.read_bytes())).convert("RGB")


class SimCamera(Camera):
    """Renders the simulator's scene (its truth) through `lens`. Its `view` starts as the lens, and is what the
    kernel believes: a calibration replaces the view, never the lens."""

    def __init__(self, name: str, lens: View, body):
        super().__init__(name, lens)
        self.lens, self.body = lens, body

    def snap(self, k) -> Image.Image:
        b = self.body
        g = b.manifest.gripper
        opening = None if g is None or b.grip is None else g.aperture(b.grip)
        return render(self.lens, b.world, b.chain, b.q, g, opening)


def equirect_dirs(u, v) -> np.ndarray:
    """Directions, in a 360 camera's own frame, of the points u, v (0..1 across and down) of its equirectangular
    picture. The capture's convention: +z is the top of the picture, its middle looks along -x, and three quarters
    of the way across looks along +y."""
    theta, phi = (1.0 - np.asarray(u, float)) * 2 * np.pi, np.asarray(v, float) * np.pi
    return np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1)


def equirect_uv(d) -> tuple[np.ndarray, np.ndarray]:
    """Where directions d (the 360's own frame) land in its picture: u, v in 0..1 across and down."""
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    u = (1.0 - np.arctan2(d[..., 1], d[..., 0]) / (2 * np.pi)) % 1.0
    return u, np.arccos(np.clip(d[..., 2], -1.0, 1.0)) / np.pi


def _axes(forward, up) -> np.ndarray:
    """Columns: a pinhole camera's x (right), y (down) and z (forward), for a forward direction and an up."""
    z = np.asarray(forward, float) / np.linalg.norm(forward)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:                       # looking straight up or down: image top = +x instead
        x = np.cross(z, [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


class EquirectCut(Camera):
    """A pinhole view cut out of a 360 camera's equirectangular picture (see equirect_dirs for its convention).

    Uncalibrated, it is aimed with yaw_deg (right of the picture's middle) and pitch_deg (up), and nothing can be
    drawn on it. Once the 360's pose is known (`pose`, its own frame in the base frame, from `wu calibrate` or the
    workcell), look_at aims it at a point and the cut has a view: the kernel's tool, axes and boxes are drawn on it.
    Each frame is resampled through a table built once for its size, a few milliseconds of numpy.
    """

    def __init__(self, name: str, source: Camera, fov_deg: float = 70.0, size=(800, 500), yaw_deg: float = 0.0,
                 pitch_deg: float = 0.0, pose=None, look_at=None):
        super().__init__(name, None)
        if abs(pitch_deg) >= 85:
            raise ValueError(f"camera {name!r}: pitch_deg must be within 85 deg of level")
        self.source, self.size = source, (int(size[0]), int(size[1]))
        self.f = (self.size[0] / 2) / np.tan(np.radians(fov_deg) / 2)
        self.pose = None if pose is None else np.asarray(pose, float)
        self._tables: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        if look_at is not None:
            if self.pose is None:
                raise ValueError(f"camera {name!r}: look_at needs the 360's pose; aim it with yaw_deg and pitch_deg")
            forward = self.pose[:3, :3].T @ (np.asarray(look_at, float) - self.pose[:3, 3])
        else:
            y, p = np.radians(yaw_deg), np.radians(pitch_deg)
            forward = np.array([-np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])
        self.R = _axes(forward, [0.0, 0.0, 1.0])       # the cut's axes in the 360's frame: its up is the 360's up
        if self.pose is not None:
            self.install(self.pose)

    def install(self, pose):
        """The 360's pose (its own frame in the base frame): from then on the cut has a view to draw on."""
        self.pose = np.asarray(pose, float)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.pose[:3, :3] @ self.R, self.pose[:3, 3]
        w, h = self.size
        self.view = View(T, self.f, self.f, w / 2, h / 2, w, h)

    def _table(self, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        if (h, w) not in self._tables:
            cw, ch = self.size
            xs, ys = np.meshgrid(np.arange(cw) + 0.5 - cw / 2, np.arange(ch) + 0.5 - ch / 2)
            u, v = equirect_uv(np.stack([xs / self.f, ys / self.f, np.ones_like(xs)], -1) @ self.R.T)
            x, y = u * w - 0.5, np.clip(v * h - 0.5, 0.0, h - 1.0)
            x0, y0 = np.floor(x).astype(int), np.minimum(np.floor(y).astype(int), h - 2)
            ax, ay = (x - x0).ravel(), (y - y0).ravel()
            x0, x1, y0 = (x0 % w).ravel(), ((x0 + 1) % w).ravel(), y0.ravel()     # wraps around the seam
            idx = np.stack([y0 * w + x0, y0 * w + x1, (y0 + 1) * w + x0, (y0 + 1) * w + x1])
            wts = np.stack([(1 - ax) * (1 - ay), ax * (1 - ay), (1 - ax) * ay, ax * ay]).astype(np.float32)
            self._tables[(h, w)] = (idx, wts)
        return self._tables[(h, w)]

    def snap(self, k) -> Image.Image:
        pano = np.asarray(self.source.picture(k), dtype=np.uint8)
        h, w = pano.shape[:2]
        idx, wts = self._table(h, w)
        flat = pano.reshape(-1, 3).astype(np.float32)
        out = np.einsum("ki,kic->ic", wts, flat[idx])
        return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8).reshape(self.size[1], self.size[0], 3))


SIM_VIEWS = {                           # eye, look_at, up (work frame): three views that together fix a position
    "side": ([0.12, -0.82, 0.46], [0.12, 0.0, 0.13], [0, 0, 1]),
    "front": ([0.95, 0.38, 0.50], [0.16, 0.0, 0.14], [0, 0, 1]),
    "top": ([0.12, 0.0, 1.15], [0.12, 0.0, 0.10], [1, 0, 0]),
}


def sim_cameras(body, world) -> dict[str, Camera]:
    return {name: SimCamera(name, View.look_at(world.to_base("work", eye), world.to_base("work", at), 55.0, (800, 600),
                                               world.frame("work").T[:3, :3] @ np.asarray(up, float)), body)
            for name, (eye, at, up) in SIM_VIEWS.items()}


def from_config(cfg: dict, world) -> Camera:
    if cfg.get("projection") == "equirect":
        return equirect_from_config(cfg, world)
    view, rotate = view_from_config(cfg, world), int(cfg.get("rotate", 0))
    if "path" in cfg:
        cam: Camera = FileCamera(cfg["name"], cfg["path"], view, float(cfg.get("max_age_s", 3.0)), rotate)
    elif "url" in cfg:
        cam = HttpCamera(cfg["name"], cfg["url"], view, rotate=rotate)
    elif "command" in cfg:
        cam = CommandCamera(cfg["name"], cfg["command"], view, rotate=rotate)
    else:
        raise ValueError(f"camera {cfg.get('name')!r} needs a path, a url or a command")
    cam.fov_deg = float(cfg["fov_deg"]) if "fov_deg" in cfg else None
    return cam


def equirect_from_config(cfg: dict, world) -> EquirectCut:
    """A 360 cut from a workcell entry: the source as for any camera, then fov_deg, size and either yaw_deg and
    pitch_deg, or the 360's pose (eye, facing = where the middle of its picture looks, up; in `frame`) and
    look_at."""
    name = cfg["name"]
    if cfg.get("rotate") or any(key in cfg for key in ("crop",)):
        raise ValueError(f"camera {name!r}: a 360 picture is not rotated or cropped; aim the cut instead")
    source = from_config({key: cfg[key] for key in ("name", "path", "url", "command", "max_age_s") if key in cfg},
                         world)
    frame = cfg.get("frame", "work")
    pose = None
    if "eye" in cfg and "facing" in cfg:
        F = world.frame(frame).T[:3, :3]
        facing, up = F @ np.asarray(cfg["facing"], float), F @ np.asarray(cfg.get("up", [0.0, 0.0, 1.0]), float)
        x = -facing / np.linalg.norm(facing)                  # the 360's -x is where the middle of its picture looks
        z = up - (up @ x) * x
        z /= np.linalg.norm(z)
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = np.column_stack([x, np.cross(z, x), z]), world.to_base(frame, cfg["eye"])
    look_at = world.to_base(frame, cfg["look_at"]) if "look_at" in cfg else None
    return EquirectCut(name, source, float(cfg.get("fov_deg", 70.0)), tuple(cfg.get("size", (800, 500))),
                       float(cfg.get("yaw_deg", 0.0)), float(cfg.get("pitch_deg", 0.0)), pose, look_at)


# -- drawing ------------------------------------------------------------------------------------------

def _font(size: int):
    return ImageFont.load_default(size=size)


def _corners(box) -> np.ndarray:
    s = box.size / 2
    local = np.array([[x, y, z] for x in (-s[0], s[0]) for y in (-s[1], s[1]) for z in (-s[2], s[2])])
    return local @ box.pose[:3, :3].T + box.pose[:3, 3]


FACES = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
EDGES = ((0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def render(view: View, world, chain, q, gripper=None, opening=None, ss: int = 2) -> Image.Image:
    """A plain, legible picture of the scene: the floor grid, every box shaded by kind, the arm as a stick figure."""
    v = view.scaled(view.width * ss, view.height * ss)
    img = Image.new("RGB", (v.width, v.height), BG)
    d = ImageDraw.Draw(img, "RGBA")
    work = world.frame("work").T if "work" in world.frames else np.eye(4)
    for i in np.arange(-0.2, 0.81, 0.1):              # a 10 cm grid on the floor of the work frame
        for a, b in (([i, -0.5, 0], [i, 0.5, 0]), ([-0.2, i - 0.3, 0], [0.8, i - 0.3, 0])):
            _line(d, v, work[:3, :3] @ a + work[:3, 3], work[:3, :3] @ b + work[:3, 3], GRID, 1 * ss)
    items: list[tuple[float, str, tuple]] = []         # (depth, kind, payload): drawn far to near
    light = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    for box in world.boxes.values():
        pts = _corners(box)
        uv, z = v.project(pts)
        if (z <= 0.01).any():
            continue
        base = KIND.get(box.kind, (150, 150, 150))
        zone = box.kind in ("keep_out", "fragile", "slow")
        for face in FACES:
            p = pts[list(face)]
            normal = np.cross(p[1] - p[0], p[3] - p[0])
            normal /= np.linalg.norm(normal) + 1e-12
            if normal @ (p.mean(0) - v.T[:3, 3]) >= 0:
                continue                                # facing away
            shade = 0.62 + 0.38 * max(0.0, float(normal @ light))
            fill = tuple(int(c * shade) for c in base) + ((70,) if zone else (255,))
            items.append((float(z[list(face)].mean()), "poly", ([tuple(uv[j]) for j in face], fill)))
    pts = chain.points(q)
    uv, z = v.project(pts)
    for a in range(1, len(pts) - 1):
        if min(z[a], z[a + 1]) > 0.01 and np.linalg.norm(pts[a + 1] - pts[a]) > 1e-4:
            width = max(2, int(v.fx * 0.028 / max(0.05, (z[a] + z[a + 1]) / 2)))
            items.append((float((z[a] + z[a + 1]) / 2), "seg", (tuple(uv[a]), tuple(uv[a + 1]), width, (70, 76, 86))))
    if gripper is not None:
        T = chain.fk(q)
        ahead = T[:3, :3] @ np.asarray(gripper.approach, float)
        side = T[:3, :3] @ np.asarray(gripper.opens_along, float)
        half = 0.5 * (opening if opening is not None else 0.04) + 0.006
        tip = T[:3, 3]
        for sgn in (-1, 1):
            p0, p1 = tip - 0.045 * ahead + sgn * half * side, tip + 0.008 * ahead + sgn * half * side
            (u0, u1), zz = v.project([p0, p1])
            if zz.min() > 0.01:
                width = max(2, int(v.fx * 0.012 / max(0.05, zz.mean())))
                items.append((float(zz.mean()) - 0.001, "seg", (tuple(u0), tuple(u1), width, INK)))
    items.sort(key=lambda it: -it[0])
    for _, kind, payload in items:
        if kind == "poly":
            corners, fill = payload
            d.polygon(corners, fill=fill, outline=tuple(int(c * 0.55) for c in fill[:3]) + (fill[3],))
        else:
            a, b, width, colour = payload
            d.line([a, b], fill=colour, width=width)
            for p in (a, b):
                r = width / 2
                d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=colour)
    return img.resize((view.width, view.height), Image.Resampling.LANCZOS)


def overlay(img: Image.Image, view: View | None, k, path=None, caption: str = "") -> Image.Image:
    """Draw what the kernel knows onto a picture: known boxes (green outlines, named), the work axes, the tool point,
    and a planned path (blue). Without a calibration only the caption is added."""
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        s = MAX_SIDE / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(img, "RGBA")
    small = _font(max(11, img.width // 60))
    if view is not None:
        v = view.scaled(img.width, img.height)
        w = k.world
        for box in w.boxes.values():
            pts = _corners(box)
            uv, z = v.project(pts)
            if (z <= 0.01).any():
                continue
            for a, b in EDGES:
                d.line([tuple(uv[a]), tuple(uv[b])], fill=MODEL + (230,), width=2)
            top = [1, 3, 5, 7]                            # corners on the box's top face
            if box.kind == "surface":                     # a table: named at its nearest corner, clear of what is on it
                u = uv[top[int(np.argmax(uv[top, 1]))]]
            else:
                (u,), _ = v.project([pts[top].mean(0) + box.pose[:3, 2] * 0.012])
            _label(d, u, box.name, MODEL, small)
        work = w.frame("work").T if "work" in w.frames else np.eye(4)
        o = work[:3, 3]
        for i, name in enumerate("FLU"):
            end = o + work[:3, i] * 0.06
            if _line(d, v, o, end, AXES[name] + (255,), 3):
                (u,), _ = v.project([end + work[:3, i] * 0.012])
                _label(d, u, name, AXES[name], small, box=False)
        if path is not None and len(path) > 1:
            uv, z = v.project(path)
            keep = z > 0.01
            pts = [tuple(p) for p in uv[keep]]
            if len(pts) > 1:
                d.line(pts, fill=PLAN + (235,), width=3, joint="curve")
                _dot(d, pts[-1], 6, PLAN)
                _label(d, pts[-1], "plan end", PLAN, small)
        tool = k.chain.fk(k.state.q)[:3, 3]
        (u,), z = v.project([tool])
        if z[0] > 0.01:
            r = 11
            d.line([(u[0] - r, u[1]), (u[0] + r, u[1])], fill=TOOL + (255,), width=3)
            d.line([(u[0], u[1] - r), (u[0], u[1] + r)], fill=TOOL + (255,), width=3)
    if caption:
        font = _font(max(12, img.width // 55))
        h = font.getbbox("Ag")[3] + 10
        d.rectangle([0, img.height - h, img.width, img.height], fill=(255, 255, 255, 215))
        d.text((8, img.height - h + 4), caption, fill=INK, font=font)
    return img


def ruler(img: Image.Image, step: int = 100) -> Image.Image:
    """The picture, scaled as `wu look` saves it, with a labelled pixel grid every `step` px and nothing else: for
    reading off where something is. Coordinates count from the top left, in this picture's own pixels."""
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        s = MAX_SIDE / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(img, "RGBA")
    font = _font(max(11, img.width // 70))
    for x in range(step, img.width, step):
        d.line([(x, 0), (x, img.height)], fill=(255, 235, 60, 110), width=1)
        _label(d, (x - 6, 14), str(x), (20, 20, 20), font)
    for y in range(step, img.height, step):
        d.line([(0, y), (img.width, y)], fill=(255, 235, 60, 110), width=1)
        _label(d, (0, y + 6), str(y), (20, 20, 20), font)
    return img


def _line(d, v: View, a, b, colour, width) -> bool:
    uv, z = v.project([a, b])
    if z.min() <= 0.01:
        return False
    d.line([tuple(uv[0]), tuple(uv[1])], fill=colour, width=width)
    return True


def _dot(d, p, r, colour):
    d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=colour + (255,), outline=(255, 255, 255, 255), width=2)


def _label(d, p, text, colour, font, box: bool = True):
    x, y = float(p[0]) + 6, float(p[1]) - 6
    if box:
        x0, y0, x1, y1 = d.textbbox((x, y), text, font=font)
        d.rectangle([x0 - 3, y0 - 2, x1 + 3, y1 + 2], fill=(255, 255, 255, 200))
    d.text((x, y), text, fill=colour + (255,), font=font)
