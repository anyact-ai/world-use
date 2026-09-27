"""Rehearsal: a plan checked on a twin reports what will happen, and nothing real moves."""
import numpy as np
import pytest

from world_use import Plan, check


def test_plan_builds_a_spec_from_any_registered_behavior():
    p = Plan("put down").line(up=0.05).touchdown(max=0.07).gripper(to=3.0)
    spec = p.spec()
    assert spec["do"] == "seq" and spec["label"] == "put down"
    assert [s["do"] for s in spec["steps"]] == ["line", "touchdown", "gripper"]
    with pytest.raises(AttributeError):
        p.teleport(to=[0, 0, 0])


def test_check_rehearses_on_a_twin_without_moving_the_robot(lifted):
    k = lifted
    top = k.chain.fk(k.state.q)[2, 3] - 0.04
    k.world.add_box("table", "surface", center=[0.3, 0, top - 0.01], size=[1, 1, 0.02])
    q_before, sim_q_before = k.cmd.q.copy(), k.body.q.copy()
    p = Plan().touchdown(max=0.08).checkpoint(ask="is it on the table?").line(up=0.05)
    report = check(p, k)
    assert report.ok, str(report)
    assert any("contact after" in c for c in report.contacts)
    assert report.assumed and "is it on the table?" in report.assumed[0]
    assert report.moving_s > 1.0
    assert np.allclose(k.cmd.q, q_before) and np.allclose(k.body.q, sim_q_before)
    assert "ASSUMED" in str(report)


def test_check_reports_a_refusal_with_the_step_that_failed(lifted):
    report = check(Plan().line(up=0.02).line(forward=0.40), lifted)
    assert not report.ok and report.outcome.status == "refused"
    assert "step 2" in report.outcome.message


def test_check_forecasts_heat(lifted):
    report = check(Plan().hold(seconds=60), lifted)
    assert report.ok and report.temp_rise["joint"] == 3 and report.temp_rise["rise_c"] > 3
