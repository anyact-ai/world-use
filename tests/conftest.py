import numpy as np
import pytest

from world_use import Kernel, VirtualClock, World, bodies

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
