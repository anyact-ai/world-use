from contextlib import contextmanager

import numpy as np
import pytest
from PIL import Image

from world_use import Client, Kernel, VirtualClock, World, bodies, cameras
from world_use.daemon import Daemon, apply_workcell, make_body, make_cameras, session_identity

# Measured on the physical reBot (2026-09-16): its folded rest pose.
Q_REST = np.array([0.3782, 0.0015, -0.0005, -0.0011, 0.0625, 0.0051])


def make_kernel(world=None, sim_world=None, q=Q_REST, gripper=1.0, **sim_options):
    """A kernel on a simulated reBot. sim_world lets the simulator know things the kernel does not."""
    world = world if world is not None else World()
    body = bodies.make("sim", world if sim_world is None else sim_world, q=q, gripper=gripper, **sim_options)
    k = Kernel(body, world, VirtualClock(100.0))
    k.connect()
    k.enable()
    return k


@pytest.fixture
def k():
    robot = make_kernel()
    yield robot
    robot.close()


@pytest.fixture
def lifted():
    """Kernel with the arm 8 cm forward and 6 cm up from rest (the pose used on hardware)."""
    k = make_kernel()
    assert k.run({"do": "line", "forward": 0.08, "up": 0.06}).ok
    yield k
    k.close()


@pytest.fixture(scope="session")
def rehearser():
    """One rehearsal worker for the whole test session: each spawns a Python process."""
    from world_use.worker import Rehearser
    r = Rehearser()
    yield r
    r.close()


@contextmanager
def serving(tmp_path, cell=None, rehearser=None):
    """A daemon as `wu up` makes it from a workcell, with torque on. Unless the workcell names another body, a
    simulated reBot at its measured rest: the simulator's truth and the kernel's model apart.
    Leaving stops its threads and closes its kernel and server, whatever state the arm is in."""
    cell = cell or {}
    name = cell.get("body", "sim")
    truth = World() if name == "sim" else None
    body = make_body(name, cell) if truth is None else bodies.make("sim", truth, q=Q_REST, gripper=1.0)
    k = Kernel(body, World(), run_dir=tmp_path / "run")
    k.connect()
    apply_workcell(cell, k, truth)
    k.enable()
    d = Daemon(k, port=0, cams=make_cameras(cell, k, body, truth), rehearser=rehearser,   # port 0: any free port
               session=session_identity(name, cell), config=cell)
    d.start()
    try:
        yield d, Client(f"http://127.0.0.1:{d.http.server_address[1]}")
    finally:
        d.stop_loop.set()
        d.http.shutdown()
        d.control.join(timeout=5)
        d.http.server_close()
        k.close()


@pytest.fixture
def daemon(tmp_path, rehearser):
    with serving(tmp_path, rehearser=rehearser) as served:
        yield served


@pytest.fixture
def client(daemon):
    return daemon[1]


@pytest.fixture
def file_camera(daemon, tmp_path):
    """Exercise image transport and overlays without requiring a graphics context."""
    d, _ = daemon
    path = tmp_path / "camera.png"
    Image.new("RGB", (800, 600), "gray").save(path)
    d.cameras["side"] = cameras.FileCamera("side", path, d.cameras["side"].view, max_age_s=300)


def supported_object(k, name="block", size=(.04, .04, .06), center=None, *, known=True):
    """Place an upright object inside the finger pads, on a pedestal clear of the arm."""
    ahead = k.tool[:3, :3] @ np.asarray(k.manifest.gripper.approach)
    point = k.tool[:3, 3] - .018 * ahead if center is None else np.asarray(center, float)
    yaw = np.degrees(np.arctan2(ahead[1], ahead[0]))
    for w in (k.world, k.body.world) if known else (k.body.world,):
        w.add_box(name, "object", center=point, size=size, yaw_deg=yaw)
        bottom = point[2] - size[2] / 2
        w.add_box(name + " support", "surface", center=[point[0], point[1], bottom - .005],
                  size=[max(.02, size[0]), max(.02, size[1]), .01], yaw_deg=yaw)


def table_below(k, distance=.04, *, name="table", known=True):
    point = k.tool[:3, 3].copy()
    point[2] -= distance + .01
    for w in (k.world, k.body.world) if known else (k.body.world,):
        w.add_box(name, "surface", center=point, size=[.07, .10, .02])
    return point[2] + .01
