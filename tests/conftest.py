import numpy as np
import pytest

from world_use import Kernel, RealClock, VirtualClock, World, bodies, cameras
from world_use.client import Client
from world_use.daemon import Daemon, apply_workcell

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
    return make_kernel()


@pytest.fixture
def lifted():
    """Kernel with the arm 8 cm forward and 6 cm up from rest (the pose used on hardware)."""
    k = make_kernel()
    assert k.run({"do": "line", "forward": 0.08, "up": 0.06}).ok
    return k


@pytest.fixture(scope="session")
def rehearser():
    """One rehearsal worker for the whole test session: each spawns a Python process."""
    from world_use.worker import Rehearser
    r = Rehearser()
    yield r
    r.close()


def serve(tmp_path, cell=None, rehearser=None):
    """A daemon on a simulated reBot as `wu up` makes it: the simulator's truth and the kernel's model apart."""
    world, truth = World(), World()
    body = bodies.make("sim", truth, q=Q_REST, gripper=1.0)
    k = Kernel(body, world, RealClock(100.0), run_dir=tmp_path / "run")
    k.connect()
    truth.frames.update(world.frames)
    apply_workcell(cell or {}, k, truth)
    k.enable()
    d = Daemon(k, port=0, cams=cameras.sim_cameras(body, truth), rehearser=rehearser)      # port 0: any free port
    d.start()
    return d, Client(f"http://127.0.0.1:{d.http.server_address[1]}")


@pytest.fixture
def daemon(tmp_path, rehearser):
    d, c = serve(tmp_path, rehearser=rehearser)
    yield d, c
    d.stop_loop.set()
    d.http.shutdown()


@pytest.fixture
def client(daemon):
    return daemon[1]
