"""Cameras: what a policy looks at. `wu look` saves one picture per call, with what the kernel knows drawn on it.

A camera is a name, a way to get a picture and, optionally, a calibration: where it is and how it projects. With a
calibration the kernel draws the tool point, the work axes, the boxes it knows about and a planned path onto the
picture, so a model can see at a glance whether its world model matches the scene. A simulated robot gets
simulated cameras that render the scene the simulator is running (its truth, which the kernel may not fully know).

Workcell entry (positions in the work frame, metres):

    [[camera]]
    name = "side"
    url = "http://127.0.0.1:8081/snapshot.jpg"    # or: command = "imagesnap -q -"  (prints an image to stdout)
    eye = [0.35, -0.60, 0.40]                      # calibration, optional: where the camera is,
    look_at = [0.30, 0.0, 0.15]                    # what the image centre shows,
    fov_deg = 55                                   # and its horizontal field of view
"""
from __future__ import annotations

import subprocess
import urllib.request
from dataclasses import dataclass
from io import BytesIO

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
        sx, sy = width / self.width, height / self.height
        return View(self.T, self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, width, height)

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

class Camera:
    def __init__(self, name: str, view: View | None = None):
        self.name, self.view = name, view

    def snap(self, k) -> Image.Image:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class HttpCamera(Camera):
    """Any camera that serves a still image over HTTP (an IP camera, a phone app, a small snapshot server)."""

    def __init__(self, name: str, url: str, view: View | None = None, timeout: float = 5.0):
        super().__init__(name, view)
        self.url, self.timeout = url, timeout

    def snap(self, k) -> Image.Image:
        with urllib.request.urlopen(self.url, timeout=self.timeout) as r:
            return Image.open(BytesIO(r.read())).convert("RGB")


class CommandCamera(Camera):
    """A command that prints one image to stdout, e.g. `imagesnap -q -` or an ffmpeg one-frame grab."""

    def __init__(self, name: str, command: str, view: View | None = None, timeout: float = 15.0):
        super().__init__(name, view)
        self.command, self.timeout = command, timeout

    def snap(self, k) -> Image.Image:
        out = subprocess.run(self.command, shell=True, capture_output=True, timeout=self.timeout)
        if out.returncode != 0 or not out.stdout:
            raise RuntimeError(f"camera {self.name!r}: {self.command!r} failed: {out.stderr.decode()[-300:].strip()}")
        return Image.open(BytesIO(out.stdout)).convert("RGB")


class SimCamera(Camera):
    """Renders the simulator's scene (its truth) from a fixed viewpoint."""
    view: View

    def __init__(self, name: str, view: View, body):
        super().__init__(name, view)
        self.body = body

    def snap(self, k) -> Image.Image:
        b = self.body
        g = b.manifest.gripper
        opening = None if g is None or b.grip is None else g.aperture(b.grip)
        return render(self.view, b.world, b.chain, b.q, g, opening)


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
    view = view_from_config(cfg, world)
    if "url" in cfg:
        return HttpCamera(cfg["name"], cfg["url"], view)
    if "command" in cfg:
        return CommandCamera(cfg["name"], cfg["command"], view)
    raise ValueError(f"camera {cfg.get('name')!r} needs a url or a command")


# -- drawing ------------------------------------------------------------------------------------------

def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                  # Pillow < 10.1
        return ImageFont.load_default()


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
