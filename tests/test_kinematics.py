"""Kinematics against numbers measured on the physical reBot, and against finite differences."""
import numpy as np
import pytest
from conftest import Q_REST

from world_use import motion
from world_use.bodies.rebot import MANIFEST
from world_use.geometry import axis_angle, interpolate_rotation, pose_error, rotation_log
from world_use.kinematics import Chain

CHAIN = Chain(MANIFEST.urdf, MANIFEST.tool_link)


def test_fk_matches_the_rest_pose_measured_on_hardware():
    assert np.allclose(CHAIN.fk(Q_REST)[:3, 3], [0.2852, -0.0997, 0.2167], atol=2e-4)


def test_urdf_limits_and_mass():
    assert CHAIN.joint_names == [f"joint{i}" for i in range(1, 7)]
    assert np.allclose(CHAIN.lower, [-2.8, 0, 0, -1.57, -1.57, -3.14])
    assert np.allclose(CHAIN.upper, [2.8, 3.14, 3.14, 1.57, 1.57, 3.14])
    assert abs(CHAIN.mass - 6.01) < 0.02


def test_jacobian_equals_finite_differences():
    rng = np.random.default_rng(2)
    for _ in range(20):
        q = rng.uniform(CHAIN.lower, CHAIN.upper)
        T0 = CHAIN.fk(q)
        J = np.zeros((6, 6))
        for i in range(6):
            dq = q.copy()
            dq[i] += 1e-6
            J[:, i] = pose_error(T0, CHAIN.fk(dq)) / 1e-6
        assert np.allclose(CHAIN.jacobian(q), J, atol=1e-5)


def test_gravity_equals_the_derivative_of_potential_energy_and_matches_hardware():
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.uniform(CHAIN.lower, CHAIN.upper)
        num = np.array([(CHAIN.potential(q + e) - CHAIN.potential(q - e)) / 2e-5 for e in np.eye(6) * 1e-5])
        assert np.allclose(CHAIN.gravity(q), num, atol=1e-6)
    # Holding [21.7, 31.1, 0.1, 26.3, 3.6, 0.1] deg the motors reported +3.8 Nm (elbow, partly on its stop),
    # +2.6 Nm (wrist) and ~0 at the shoulder. The model must agree in sign and rough size.
    g = CHAIN.gravity(np.radians([21.664, 31.088, 0.149, 26.348, 3.571, 0.122]))
    assert abs(g[1]) < 0.1 and 6.5 < g[2] < 7.6 and 1.7 < g[3] < 2.3


@pytest.mark.parametrize("axis", [[0, 1, -1], [1, 0, -1], [1, -1, 0], [-1, 2, 3]])
def test_half_turn_interpolation_reaches_the_requested_rotation(axis):
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    R = axis_angle(a, np.pi)
    w = rotation_log(R)
    assert np.allclose(axis_angle(w / np.linalg.norm(w), np.linalg.norm(w)), R, atol=1e-7)
    assert np.allclose(interpolate_rotation(np.eye(3), R, 1), R, atol=1e-7)


def test_ik_round_trip():
    rng = np.random.default_rng(3)
    for _ in range(10):
        q = rng.uniform(CHAIN.lower * 0.5, CHAIN.upper * 0.5)
        seed = q + rng.normal(0, 0.05, 6)
        _, res = CHAIN.ik(CHAIN.fk(q), seed)
        assert res < 1e-6


def test_reach_reproduces_the_joint_motion_validated_on_hardware():
    path, _, res = motion.line(CHAIN, Q_REST, _work(Q_REST, 0.08, 0.06), motion.Timing(), duration=6.0)
    assert res < 1e-5 and len(path) == 600
    assert np.allclose(np.degrees(path[-1] - Q_REST), [0, 45.8, 19.4, 26.4, 0, 0], atol=0.15)
    assert np.abs(np.diff(path, axis=0)).max() * 100 < 0.27                   # hardware peak: 0.262 rad/s
    tool = np.array([CHAIN.fk(q)[:3, 3] for q in path[::20]]) - CHAIN.fk(Q_REST)[:3, 3]
    d = tool[-1] / np.linalg.norm(tool[-1])
    assert np.linalg.norm(tool - np.outer(tool @ d, d), axis=1).max() < 1e-5   # straight
    assert np.allclose(CHAIN.fk(path[-1])[:3, :3], CHAIN.fk(Q_REST)[:3, :3], atol=1e-5)   # orientation held


def _work(q, forward, up):
    T = MANIFEST.frames(CHAIN, q)["work"]
    return forward * T[:3, 0] + up * T[:3, 2]


def test_gravity_is_linear_in_each_links_mass_and_first_moment():
    """So recorded torques can fit them (fit.py)."""
    chain = Chain(MANIFEST.urdf, MANIFEST.tool_link)
    links = list(chain.links)
    phi = np.concatenate([[m, *(m * c)] for m, c in (chain.links[name] for name in links)])
    for q in np.random.default_rng(3).uniform(-1.5, 1.5, (5, chain.n)):
        assert np.allclose(chain.gravity_regressor(q, links) @ phi, chain.gravity(q), atol=1e-9)
