"""Execution evidence on generic fake feedback and the existing MuJoCo gripper fixture."""
import json
from dataclasses import replace

import numpy as np
import pytest
from conftest import supported_object

from world_use.behaviors import CheckGrip, Outcome, build
from world_use.checks import Settling
from world_use.errors import Refused
from world_use.views import incident, job_text


@pytest.mark.parametrize("rate", [20, 100])
def test_settling_returns_on_a_plateau_at_different_control_rates(rate):
    check = Settling([.008, .1], 1, .2)
    for i in range(rate):
        out = check.update(i / rate, [.03, .05])
        if out:
            break
    assert out == "stalled" and .2 <= check.elapsed_s < .3


def test_slow_progress_that_projects_arrival_keeps_its_time():
    check = Settling([.008], 1, .2)
    for i in range(21):
        out = check.update(i / 20, [.0088 - i * .00005])
        if out:
            break
    assert out == "arrived" and check.elapsed_s >= .75


def test_pending_applied_commands_and_cached_feedback_cannot_confirm_a_stall():
    check = Settling([.008], 1, .2)
    for i in range(11):
        assert check.update(i / 20, [.03], command_active=False) is None
    assert check.update(.5, [0.]) is None
    assert check.update(.55, [.03]) is None
    assert check.update(.6, [.03]) is None
    assert check.update(.8, [.03]) == "stalled"


def test_arrival_window_requires_fresh_samples_inside_tolerance():
    check = Settling([.01], 1, .2, stable_s=.1)
    assert check.update(0., [.005]) is None
    for _ in range(30):
        assert check.update(0., [.005]) is None
    assert check.update(.05, [.02]) is None
    assert check.update(.1, [.005]) is None
    assert check.update(.2, [.005]) == "arrived"


def test_progressing_motion_times_out_and_bad_measurements_are_rejected():
    check = Settling([.01], .4, .2)
    for i in range(5):
        out = check.update(i / 10, [.03 - i * .001])
    assert out == "timeout"
    for stamp, errors in [(np.nan, [.01]), (.3, [.01]), (.5, [np.inf]), (.5, [.01, .01])]:
        with pytest.raises(ValueError):
            check.update(stamp, errors)


def test_a_measurement_after_the_deadline_cannot_confirm_arrival():
    check = Settling([.01], .4, .2)
    assert check.update(0., [.03]) is None
    assert check.update(.5, [0.]) == "timeout"


def measured_gripper(k, monkeypatch, position):
    read = k.body.read
    monkeypatch.setattr(k.body, "read", lambda: replace(read(), gripper=position(), gripper_tau=0.))


def test_joint_settling_stops_a_blocked_plan_and_cancels_queued_work(k, monkeypatch):
    read = k.body.read
    stuck = k.state.q.copy()
    stuck[0] += .03                       # below the watchdog limit, above the declared arrival tolerance
    monkeypatch.setattr(k.body, "read", lambda: replace(read(), q=stuck.copy(), tau=None))
    before = k.cmd.gripper
    job = k.submit([{"do": "settle", "seconds": 1, "tolerance_deg": .2}, {"do": "gripper", "to": 2}])
    queued = k.submit({"do": "hold", "seconds": 1})
    while not job.finished:
        k.tick()
        k.clock.wait()
    assert job.status == "surprise" and job.outcome.data["reason"] == "stalled"
    assert queued.status == "cancelled" and k.cmd.gripper == before
    np.testing.assert_allclose(k.cmd.q, stuck)
    assert not np.any(k.cmd.dq)
    report = incident(k, job)
    assert "reason=stalled" in report and "joint_error_deg=" in report and "hint:" in report


def test_gripper_returns_early_only_after_measured_arrival_window(k, monkeypatch):
    measured_gripper(k, monkeypatch, lambda: k.cmd.gripper)
    out = k.run({"do": "gripper", "to": 1.5, "check_arrival": True})
    assert out.ok and out.data["reason"] == "arrived"
    assert .1 <= out.data["settled_s"] < .3
    assert not out.data["object_retention_verified"]


def test_legacy_gripper_completion_is_distinct_from_measured_arrival(k, monkeypatch):
    measured_gripper(k, monkeypatch, lambda: 1.)
    out = k.run({"do": "gripper", "to": 1.3})
    assert out.ok and out.data["reason"] == "command_complete"
    assert out.data["target"] == 1.3 and out.data["position"] == 1.


def test_failed_preopening_stops_grip_before_it_starts_closing(k, monkeypatch):
    measured_gripper(k, monkeypatch, lambda: 1.)
    out = k.run({"do": "grip", "start": 1.3})
    assert out.status == "surprise" and out.data["reason"] == "stalled"
    assert k.cmd.gripper == 1. and k.cmd.gripper_v == 0
    assert not any(e["kind"] == "grip" for e in k.events.since())


def test_grip_check_accepts_bounded_sensor_noise_without_claiming_retention(k, monkeypatch):
    measured_gripper(k, monkeypatch, lambda: 1. + .01 * np.sin(k.clock.now() * 80))
    out = k.run({"do": "check_grip", "expect": [.9, 1.1], "seconds": .2, "max_drift": .03})
    assert out.ok and out.data["observed_s"] >= .2
    assert not out.data["object_retention_verified"]


def test_grip_check_reports_drift_and_preserves_the_closing_command(k, monkeypatch):
    start, command = k.clock.now(), k.cmd.gripper
    measured_gripper(k, monkeypatch, lambda: 1. - .5 * (k.clock.now() - start))
    out = k.run({"do": "check_grip", "expect": [.5, 1.5], "seconds": .3, "max_drift": .03})
    assert out.status == "surprise" and out.data["reason"] == "grip_drift"
    assert out.observed > out.expected and out.data["observed_s"] < .3
    assert k.cmd.gripper == command and k.cmd.gripper_v == 0


@pytest.mark.parametrize("step", [{"do": "hold", "seconds": 1}, {"do": "checkpoint", "ask": "object followed?"}])
def test_scoped_grip_expectation_watches_motion_and_checkpoint_waits(k, monkeypatch, step):
    start, command = k.clock.now(), k.cmd.gripper
    measured_gripper(k, monkeypatch, lambda: 1. if k.clock.now() - start < .1 else .8)
    job = k.submit({"do": "seq", "expect_grip": [.9, 1.1], "steps": [step, {"do": "gripper", "to": 2.}]})
    queued = k.submit({"do": "hold", "seconds": 1})
    for _ in range(100):
        k.tick()
        k.clock.wait()
        if job.finished:
            break
    assert job.status == "surprise" and job.outcome.data["reason"] == "grip_range"
    assert queued.status == "cancelled" and k.cmd.gripper == command
    assert not np.any(k.cmd.dq)


def test_intentional_release_outside_the_expectation_is_allowed(k, monkeypatch):
    measured_gripper(k, monkeypatch, lambda: k.cmd.gripper)
    out = k.run([{"do": "seq", "expect_grip": [.9, 1.1], "steps": [{"do": "hold", "seconds": .05}]},
                 {"do": "gripper", "to": 2.}])
    assert out.ok and k.cmd.gripper == 2.


def test_unavailable_gripper_measurement_refuses_without_motion(k, monkeypatch):
    measured_gripper(k, monkeypatch, lambda: None)
    before = k.cmd.gripper
    out = k.run({"do": "check_grip", "expect": [.5, 1.5]})
    assert out.status == "refused" and out.data["rule"] == "sensing"
    assert k.cmd.gripper == before


def test_losing_feedback_during_a_watched_sequence_holds_and_cancels(k, monkeypatch):
    start, command = k.clock.now(), k.cmd.gripper
    measured_gripper(k, monkeypatch, lambda: 1. if k.clock.now() - start < .1 else None)
    job = k.submit({"do": "seq", "expect_grip": [.9, 1.1], "steps": [{"do": "hold", "seconds": 1}]})
    queued = k.submit({"do": "gripper", "to": 2.})
    while not job.finished:
        k.tick()
        k.clock.wait()
    assert job.status == "refused" and job.outcome.data["rule"] == "sensing"
    assert queued.status == "cancelled" and k.cmd.gripper == command
    np.testing.assert_allclose(k.cmd.q, k.state.q)


def test_cached_grip_feedback_cannot_complete_the_observation(k):
    check = build({"do": "check_grip", "expect": [.9, 1.1], "seconds": .1})
    check.start(k)
    for _ in range(50):
        assert check.tick(k) is None
    k.state = replace(k.state, t=k.state.t + .1)
    assert check.tick(k).ok


@pytest.mark.parametrize("scale", [1., -1.])
def test_grip_check_converts_width_and_drift_for_native_metres_and_reversed_travel(k, scale):
    g = replace(k.manifest.gripper, closed=0. if scale > 0 else .08, open=.08 if scale > 0 else 0.,
                unit="m", m_per_unit=scale)
    k.manifest = replace(k.manifest, gripper=g)
    k.state = replace(k.state, gripper=g.position(.04))
    check = build({"do": "check_grip", "expect_mm": [35, 45], "seconds": .2, "max_drift_mm": 1})
    check.start(k)
    assert check.tick(k) is None
    k.state = replace(k.state, t=k.state.t + .1, gripper=g.position(.038))
    out = check.tick(k)
    assert out.status == "surprise" and out.data["reason"] == "grip_drift"
    assert out.expected == pytest.approx(.001) and out.observed == pytest.approx(.002)
    assert out.data["unit"] == "m"


def test_grip_refuses_uncalibrated_drift_units_before_opening(k):
    k.manifest = replace(k.manifest, gripper=replace(k.manifest.gripper, m_per_unit=None))
    before = k.cmd.gripper
    out = k.run({"do": "grip", "start": 2., "verify_s": .2, "max_drift_mm": 1})
    assert out.status == "refused" and k.cmd.gripper == before


def test_serialized_sequences_preserve_the_grip_expectation(k):
    command = k.cmd.gripper
    spec = {"do": "seq", "label": "carry", "expect_grip": [1.1, 1.5],
            "steps": [{"do": "gripper", "to": 2.}]}
    restored = build(json.loads(json.dumps(build(spec).spec())))
    out = k.run(restored)
    assert out.status == "surprise" and out.data["reason"] == "grip_range"
    assert k.cmd.gripper == command


def test_grip_verification_on_a_physical_simulated_object(lifted):
    k = lifted
    assert k.run({"do": "gripper", "aperture_mm": 60}).ok
    supported_object(k, size=[.04, .04, .04])
    out = k.run({"do": "grip", "start_mm": 60, "expect_mm": [35, 45], "verify_s": .2, "max_drift_mm": 1})
    assert out.ok, out.message
    assert out.data["verification"]["data"]["observed_s"] >= .2
    assert not out.data["object_retention_verified"]


def test_interrupted_grip_verification_preserves_payload_and_invalidates_old_home(lifted):
    k = lifted
    assert k.run({"do": "gripper", "aperture_mm": 60}).ok
    supported_object(k, size=[.04, .04, .04])
    k.set_home_route([])
    job = k.submit({"do": "grip", "expect_mm": [35, 45], "verify_s": 1})
    for _ in range(1000):
        k.tick()
        k.clock.wait()
        if job.finished or job.behavior.verifier is not None:
            break
    assert not job.finished and k.held_at is not None and k.world.held is None
    with pytest.raises(Refused):
        k.home_plan()
    k.stop("inspect grip")
    k.tick()
    assert job.status == "stopped" and k.held_at is not None
    k.set_home_route([])
    assert all(step["do"] != "gripper" for step in k.home_plan())


def test_grasp_uses_its_bounded_search_after_failed_verification(lifted, monkeypatch):
    k = lifted
    assert k.run({"do": "gripper", "aperture_mm": 60}).ok
    supported_object(k, size=[.04, .04, .04])
    monkeypatch.setattr(CheckGrip, "tick", lambda self, k: self.surprise("unstable measured grip"))
    out = k.run({"do": "grasp", "start_mm": 60, "expect_mm": [35, 45], "verify_s": .2,
                 "search_mm": [[6, 0]]})
    assert out.status == "surprise" and out.data["tries"] == 2
    assert any(e["kind"] == "grasp_retry" for e in k.events.since())


@pytest.mark.parametrize("spec", [
    {"do": "check_grip"}, {"do": "check_grip", "expect": [.5, 1.5], "seconds": 0},
    {"do": "check_grip", "expect": [.5, 1.5], "max_drift": float("nan")},
    {"do": "grip", "max_drift": .1},
    {"do": "grip", "verify_s": .2, "max_drift": .1, "max_drift_mm": 1},
    {"do": "seq", "steps": [], "expect_grip": [.5, 1.5], "expect_grip_mm": [10, 20]},
    {"do": "settle", "seconds": float("inf")},
])
def test_invalid_contracts_are_refused_before_queueing(spec):
    with pytest.raises(Refused):
        build(spec)


def test_compact_feedback_retains_final_measurements_and_full_step_records(k):
    out = k.run([{"do": "hold", "seconds": .01}] * 12 + [
        {"do": "check_grip", "expect": [.5, 1.5], "seconds": .02}])
    assert out.ok
    compact = out.feedback()
    assert "steps" not in compact.get("data", {})
    assert compact["last_step"]["data"]["reason"] == "grip_consistent"
    assert len(json.dumps(compact)) < len(json.dumps(out.to_dict()))
    completed = [e for e in k.events.since() if e["kind"] == "step_done"]
    assert len(completed) == 13 and all("outcome" in e["data"] for e in completed)
    text = job_text(dict(id=1, outcome=out.to_dict(), line="state"), answer="{id}", wait="{id}")
    assert "grip_consistent" in text and "object_retention_verified=False" in text


def test_failure_feedback_preserves_expected_observed_and_recovery_hint():
    out = Outcome("surprise", "check_grip", "unexpected width", dict(reason="grip_range"),
                  expected=[35, 45], observed=20, hint="look at the object")
    assert out.feedback() == out.to_dict()
