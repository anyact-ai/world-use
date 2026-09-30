"""Combined failure cases and boundary inputs that normal motion tests do not cover."""
import numpy as np
import pytest

from world_use.body import JointState
from world_use.errors import Refused
from world_use.geometry import axis_angle, interpolate_rotation, rotation_log


@pytest.mark.parametrize('spec', [
    {'do': 'hold', 'second': 1}, {'do': 'line', 'up': .01, 'duraton': 2},
    {'do': 'line', 'up': .01, 'speed': 0}, {'do': 'hold', 'seconds': float('nan')},
    {'do': 'lines', 'legs': []}, {'do': 'checkpoint'},
    [{'do': 'hold', 'seconds': .1}, {'do': 'gripper', 'apeture_mm': 60}],
])
def test_bad_specs_are_rejected_before_queueing(k, spec):
    with pytest.raises(Refused):
        k.submit(spec)
    assert not k.jobs and not k.queue and not k.faulted


def test_hot_return_still_stops_for_a_jammed_arm(lifted, monkeypatch):
    k = lifted
    k.set_home_route([])
    q, g = k.state.q.copy(), k.state.gripper
    monkeypatch.setattr(k.body, 'read', lambda: JointState(k.clock.now(), q.copy(), tau=k.chain.gravity(q),
                                                        temp=np.full(6, 81.), gripper=g, gripper_tau=0.))
    for _ in range(1000):
        k.tick()
        k.clock.wait()
        if k.home_route is None:
            break
    job = list(k.jobs.values())[-1]
    assert job.status == 'surprise' and job.outcome.data['trip'] == 'blocked'
    assert k.home_route is None and k.enabled
    assert np.allclose(k.cmd.q, q)


def test_home_returns_joints_and_gripper_changed_by_the_prefix(lifted):
    k = lifted
    k.set_home_route([{'do': 'joints', 'delta_deg': {'1': 5}}, {'do': 'gripper', 'to': 2}])
    assert k.run(k.home_plan()).ok
    assert abs(k.cmd.q[0] - k.q_start[0]) < 1e-6
    assert k.cmd.gripper == k.grip_start
    k.release()
    assert not k.enabled


@pytest.mark.parametrize('axis', [[0, 1, -1], [1, 0, -1], [1, -1, 0], [-1, 2, 3]])
def test_half_turn_interpolation_reaches_the_requested_rotation(axis):
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    R = axis_angle(a, np.pi)
    w = rotation_log(R)
    assert np.allclose(axis_angle(w / np.linalg.norm(w), np.linalg.norm(w)), R, atol=1e-7)
    assert np.allclose(interpolate_rotation(np.eye(3), R, 1), R, atol=1e-7)


def test_a_timed_out_rehearsal_is_not_submitted(daemon, monkeypatch):
    from world_use import plan
    d, c = daemon
    spec = {'do': 'hold', 'seconds': 20}
    report = plan.check(spec, d.k, timeout_s=.1)
    assert report.outcome.status == 'stopped'
    monkeypatch.setattr(d.rehearser, 'check', lambda *a, **kw: report)
    result = c.run(spec)
    assert result['status'] == 'refused' and not d.k.jobs
