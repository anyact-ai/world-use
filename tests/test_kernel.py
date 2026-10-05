"""Kernel semantics on a simulated reBot: refusals, surprises, contact, grip, checkpoints, stop, heat, home."""
import re
import threading
import time

import numpy as np
import pytest
from conftest import Q_REST, make_kernel, supported_object, table_below

from world_use import Kernel, RealClock, Refused, World, bodies
from world_use.body import JointState

HOME = Q_REST.copy()
HOME[1:4] = [0.02, 0.02, 0.0]   # declared rest, just off the shoulder and elbow stops


@pytest.mark.parametrize("trip", ["fault", "hot", "blocked"])
def test_an_idle_watchdog_trip_cancels_motion_before_it_starts(k, monkeypatch, trip):
    from world_use.envelope import Trip

    k.run({"do": "hold", "seconds": 0.2})
    before = k.cmd.q.copy()
    job = k.submit({"do": "line", "up": 0.02})
    monkeypatch.setattr(k.envelope, "watch", lambda *args: Trip(trip, "injected watchdog finding"))
    k.tick()
    assert job.status == "cancelled" and k.active is None
    assert np.array_equal(k.cmd.q, before if trip == "hot" else k.state.q)
    assert not np.any(k.cmd.dq)
    assert k.faulted == (trip == "fault")


@pytest.mark.parametrize("finding", ["keep_out", "overload"])
def test_a_persistent_idle_finding_is_reported_once_and_still_cancels_work(lifted, monkeypatch, finding):
    """A box over the idle arm, or a steady overload, raised 100 events a second: the event buffer then held 25 s
    of repeats instead of the cause."""
    from dataclasses import replace

    k = lifted
    if finding == "keep_out":
        points = k.chain.points(k.state.q)
        k.world.add_box("cart", "keep_out", (points[2] + points[3]) / 2, [0.05] * 3)
    else:
        read = k.body.read

        def leaning():
            st = read()
            return replace(st, tau=st.tau + [0, 0, 0, 0, 10, 0])            # j5 past its 4 Nm limit
        monkeypatch.setattr(k.body, "read", leaning)
    seq = k.events.seq
    for _ in range(100):
        k.tick()
        k.clock.wait()
    job = k.submit({"do": "line", "up": 0.02})
    k.tick()
    assert job.status == "cancelled"
    kinds = [e["kind"] for e in k.events.since(seq)]
    assert kinds.count("trip") == 1 and kinds.count("contact") == (finding == "overload")


def test_a_fault_that_outlasts_a_reset_or_changes_is_reported_again(lifted, monkeypatch):
    """The latch must not swallow a fault the operator's reset did not fix, or a second fault on top of the first."""
    from dataclasses import replace

    k = lifted
    read, faults = k.body.read, ["joint3 overcurrent"]
    monkeypatch.setattr(k.body, "read", lambda: replace(read(), faults=list(faults)))

    def alarms():
        seq = k.events.seq
        for _ in range(20):
            k.tick()
            k.clock.wait()
        return [e["message"] for e in k.events.since(seq) if e["kind"] == "trip"]

    assert alarms() == ["joint3 overcurrent"] and k.faulted
    k.reset()
    assert alarms() == ["joint3 overcurrent"] and k.faulted
    faults.append("joint5 encoder lost")
    assert alarms() == ["joint3 overcurrent; joint5 encoder lost"]


@pytest.mark.parametrize("phase", ["start", "tick"])
def test_a_behavior_exception_latches_the_fault(k, phase):
    from world_use.behaviors import Behavior

    class Broken(Behavior):
        kind = "broken"

        def start(self, k):
            if phase == "start":
                raise RuntimeError("behavior failed")

        def tick(self, k):
            raise RuntimeError("behavior failed")

    assert k.run(Broken()).status == "faulted"
    assert k.faulted and k.submit({"do": "line", "up": 0.02}).status == "refused"
    k.reset()
    assert k.run({"do": "hold", "seconds": 0.1}).ok


@pytest.mark.parametrize("phase", ["enable", "read"])
def test_incomplete_enable_stays_visible_and_release_retries_torque_off(k, monkeypatch, phase):
    from world_use import views

    k.release()
    original_enable, original_read = k.body.enable, k.body.read

    def fail():
        if phase == "enable":
            original_enable()                   # a partial enable can leave a motor powered
        raise OSError("lost feedback")

    monkeypatch.setattr(k.body, phase, fail)
    with pytest.raises(OSError, match="lost feedback"):
        k.enable()
    assert k.body.enabled and k.faulted and views.status(k)["power_uncertain"]
    with pytest.raises(Refused, match="unconfirmed"):
        k.reset()
    monkeypatch.setattr(k.body, "read", original_read)
    k.release()
    assert not k.body.enabled and not k.enabled and not k.power_uncertain


def test_line_moves_the_tool_by_the_request_in_the_work_frame(k):
    p0 = k.world.from_base("work", k.chain.fk(k.state.q)[:3, 3])
    assert k.run({"do": "line", "forward": 0.08, "up": 0.06}).ok
    assert k.run({"do": "line", "back": 0.02, "down": 0.01}).ok                # the words point takes, negated
    k.run({"do": "hold", "seconds": 0.2})
    p1 = k.world.from_base("work", k.chain.fk(k.state.q)[:3, 3])
    assert np.allclose(p1 - p0, [0.06, 0, 0.05], atol=1e-3)


def test_refused_command_moves_nothing_and_cancels_what_was_queued(lifted):
    k = lifted
    q0 = k.cmd.q.copy()
    bad = k.submit({"do": "line", "forward": 0.40})              # longer than one segment may be
    queued = [k.submit({"do": "line", "up": 0.01}) for _ in range(500)]
    while not queued[-1].finished:
        k.tick()
        k.clock.wait()
    assert bad.status == "refused" and "segment" in bad.outcome.message
    assert all(j.status == "cancelled" for j in queued) and not k.faulted
    assert np.allclose(k.cmd.q, q0)


def test_too_fast_is_refused_with_a_hint(lifted):
    out = lifted.run({"do": "line", "forward": 0.05, "duration": 0.2})
    assert out.status == "refused" and out.hint


def test_turning_the_base_at_table_height_is_refused(k):
    out = k.run({"do": "joints", "delta_deg": {"1": 10}})
    assert out.status == "refused" and out.data["rule"] == "turn_clearance"


@pytest.mark.parametrize("spec", [
    {"do": "teleport", "to": [0, 0, 1]}, {"do": []}, {"do": "hold", "label": 5},
    {"do": "hold", "second": 1}, {"do": "line", "up": .01, "duraton": 2},
    {"do": "line", "up": .01, "speed": 0}, {"do": "hold", "seconds": float("nan")},
    {"do": "line"}, {"do": "line", "up": 0}, {"do": "line", "up": .01, "down": .02},
    {"do": "line", "up": .01, "frame": "tabel"},
    [{"do": "hold", "seconds": .1}, {"do": "guarded", "frame": "tabel", "down": .01}],
    {"do": "lines", "legs": []}, {"do": "lines", "legs": [[.01, 0, 0], [0, 0, 0]]},
    {"do": "move_to"}, {"do": "move_to", "point": "sideways"}, {"do": "move_to", "to": [.2, 0, .1], "jaws": [0, 0, 0]},
    {"do": "joints", "delta_deg": {"9": 3}}, {"do": "touchdown", "joints": [9]},
    {"do": "checkpoint"}, {"do": "checkpoint", "ask": "ready?", "expect": 1},
    {"do": "gripper"}, [{"do": "hold", "seconds": .1}, {"do": "gripper", "apeture_mm": 60}],
    {"do": "grip", "effort": .6}, {"do": "grip", "squeeze": -.1}, {"do": "grip", "squeeze": float("nan")},
    {"do": "grip", "start": float("nan")}, {"do": "grip", "expect": 2}, {"do": "grip", "expect": [.5, float("nan")]},
    {"do": "grip", "hold_effort": "oops"},
    {"do": "grasp", "expect": [.4, 1.]},
    {"do": "grasp", "start": 3., "search_mm": None},
    {"do": "grasp", "start": 3., "search_mm": [["oops", 0]]},
    {"do": "grasp", "start": 3., "search_mm": [[float("inf"), 0]]},
    {"do": "grasp", "start": 3., "lift_mm": "oops"},
    {"do": "grasp", "start": 3., "lift_mm": 51},
    {"do": "grasp", "start": 3., "expect_mm": [10]},
    {"do": "grasp", "start": 3., "hold_effort": -1},
])
def test_bad_specs_are_rejected_before_queueing(k, spec):
    with pytest.raises(Refused):
        k.submit(spec)
    assert not k.jobs and not k.queue and not k.faulted


def test_every_step_documents_each_parameter_it_accepts():
    """`wu help STEP` is the docstring: a parameter it leaves out is one a policy cannot know about."""
    from world_use.behaviors import REGISTRY
    from world_use.validation import FIELDS

    for kind, fields in FIELDS.items():
        missing = [name for name in fields if not re.search(rf"\b{name}\b", REGISTRY[kind].__doc__ or "")]
        assert not missing, f"{kind} accepts {missing} but its help does not say so"


def test_touchdown_finds_a_table_and_stops_on_it(lifted):
    k = lifted
    top = k.chain.fk(k.state.q)[2, 3] - 0.04
    table_below(k)
    out = k.run({"do": "touchdown", "max": 0.08})
    assert out.ok, out.message
    # Mesh fingers touch before the empty tool point between their tips reaches the table.
    assert .02 < out.data["moved_m"] < .04
    assert k.body.data.ncon > 0
    assert top < k.chain.fk(k.state.q)[2, 3] < top + .02
    assert k.last_touch > 0


def test_touchdown_where_the_world_says_a_table_is_but_none_is_there_is_a_surprise():
    k = make_kernel(world=World(), sim_world=World())            # the simulator has no table
    assert k.run({"do": "line", "forward": 0.08, "up": 0.06}).ok
    top = k.chain.fk(k.state.q)[2, 3] - 0.04
    k.world.add_box("table", "surface", center=[0.3, 0, top - 0.01], size=[1, 1, 0.02])
    out = k.run({"do": "touchdown", "max": 0.08})
    assert out.status == "surprise" and "table" in out.message and "world model" in out.hint


def test_moving_into_something_unknown_trips_the_watchdog_and_holds():
    sim_world = World()
    k = make_kernel(sim_world=sim_world)
    assert k.run({"do": "line", "forward": 0.08, "up": 0.06}).ok
    table_below(k, .03, name="box nobody mentioned", known=False)
    out = k.run({"do": "line", "up": -0.06})
    assert out.status == "surprise"
    assert np.abs(k.cmd.q - k.state.q).max() < 0.01                 # holding where it is, not pressing on
    assert k.last_touch > 0


def test_grip_on_an_object_reports_where_the_fingers_met_it(lifted):
    k = lifted
    supported_object(k, size=[.03, .012, .04])
    out = k.run({"do": "grip", "start": 3.0, "expect": [0.4, 1.0], "squeeze": 0.05})
    assert out.ok, out.message
    assert abs(out.data["contact_at"] - (0.05 + 0.012 / 0.020)) < 0.08
    assert out.data["holding_effort"] < 0  # measured closing effort from the physical actuator


def test_grip_on_nothing_is_a_surprise(lifted):
    out = lifted.run({"do": "grip", "start": 3.0})
    assert out.status == "surprise" and "nothing" in out.message


def test_grip_outside_the_expected_width_is_a_surprise(lifted):
    k = lifted
    assert k.run({"do": "gripper", "to": 3.0}).ok                  # open wider than the object before it appears
    supported_object(k, name="big", size=[.04, .04, .04])
    out = k.run({"do": "grip", "start": 3.0, "expect": [0.4, 1.0]})
    assert out.status == "surprise" and out.observed > 1.0


def test_checkpoint_waits_for_an_answer_and_a_different_answer_ends_the_plan(lifted):
    k = lifted
    job = k.submit([{"do": "checkpoint", "ask": "is the loop between the jaws?", "view": "side"},
                    {"do": "line", "up": 0.02}])
    for _ in range(20):
        k.tick()
        k.clock.wait()
    assert job.status == "waiting" and job.question["view"] == "side"
    q_hold = k.cmd.q.copy()
    for _ in range(200):                                        # the arm holds while the question waits
        k.tick()
        k.clock.wait()
    assert np.allclose(k.cmd.q, q_hold)
    k.answer(job.id, "no")
    while not job.finished:
        k.tick()
        k.clock.wait()
    assert job.status == "surprise" and job.outcome.observed == "no"


def test_stop_holds_where_the_arm_is(lifted):
    k = lifted
    job = k.submit({"do": "line", "forward": 0.05, "duration": 4.0})
    for _ in range(100):
        k.tick()
        k.clock.wait()
    k.stop("operator said stop")
    for _ in range(3):
        k.tick()
        k.clock.wait()
    assert job.status == "stopped"
    q = k.cmd.q.copy()
    for _ in range(50):
        k.tick()
        k.clock.wait()
    assert np.allclose(k.cmd.q, q)


def test_stop_clears_gripper_velocity_without_releasing_its_position(k, monkeypatch):
    sent = []
    command = k.body.command

    def capture(q, dq, gripper, gripper_v=0.0):
        sent.append((gripper, gripper_v))
        command(q, dq, gripper, gripper_v)

    monkeypatch.setattr(k.body, "command", capture)
    job = k.submit({"do": "gripper", "to": 4.0, "seconds": 3.0})
    for _ in range(100):
        k.tick()
        k.clock.wait()
    position, velocity = sent[-1]
    assert velocity > 0
    sent.clear()
    k.stop("operator stopped the gripper")
    for _ in range(50):
        k.tick()
        k.clock.wait()
    assert job.status == "stopped"
    assert sent and all(target == position and speed == 0 for target, speed in sent)


def test_a_surprise_marks_facts_stale(lifted):
    k = lifted
    k.world.assert_fact("door.angle_deg", 22, "side camera")
    k.run({"do": "grip", "start": 3.0})                         # closes on nothing
    assert k.world.facts["door.angle_deg"].stale


def test_home_needs_a_route_and_a_contact_makes_it_stale(lifted):
    k = lifted
    with pytest.raises(Refused):
        k.home_plan()
    k.set_home_route([])
    assert len(k.home_plan()) >= 1
    k.touched("contact", "test contact")
    with pytest.raises(Refused):
        k.home_plan()


def test_home_folds_to_rest_and_release_is_then_allowed(lifted):
    k = lifted
    with pytest.raises(Refused):
        k.release()                                              # raised: releasing would drop the arm
    k.set_home_route([])
    out = k.run({"do": "seq", "steps": k.home_plan()})
    assert out.ok, out.message
    assert np.allclose(k.cmd.q, HOME, atol=1e-4)
    k.release()
    assert not k.enabled


def test_home_returns_joints_and_gripper_changed_by_the_prefix(lifted):
    k = lifted
    k.set_home_route([{"do": "joints", "delta_deg": {"1": 5}}, {"do": "gripper", "to": 2}])
    assert k.run(k.home_plan()).ok
    assert abs(k.cmd.q[0] - k.q_start[0]) < 1e-6
    assert k.cmd.gripper == k.grip_start
    k.release()
    assert not k.enabled


@pytest.mark.parametrize("stops", [(), (1,)])
def test_home_uses_declared_rest_when_the_session_started_elsewhere(stops):
    from dataclasses import replace
    from pathlib import Path

    from world_use import VirtualClock
    from world_use.bodies.sim import SimBody
    from world_use.body import Rest
    from world_use.config import load_robot

    model = load_robot(Path(__file__).resolve().parents[1] / "examples/adapters/planar.toml")
    model = replace(model, rest=Rest(q=(0.0, 0.5), joints=(1,), tol=0.1, stops=stops))
    k = Kernel(SimBody(model, q=[0.2, 0.0]), clock=VirtualClock(model.rate_hz))
    k.connect()
    k.enable()
    k.set_home_route([{"do": "joints", "target_deg": {"1": 30, "2": float(np.degrees(0.5))}}])
    out = k.run(k.home_plan())
    assert out.ok, out.message
    assert model.rest.holds(k.state.q)
    assert k.state.q[0] == pytest.approx(0.2, abs=0.01)
    k.release()
    assert not k.enabled and not k.power_uncertain
    k.close()


@pytest.mark.parametrize("sign", [-1, 1])
def test_home_reaches_supported_rest_near_a_joint_limit(sign):
    from dataclasses import replace
    from pathlib import Path

    from world_use import VirtualClock
    from world_use.bodies.sim import SimBody
    from world_use.body import Rest
    from world_use.config import load_robot

    model = load_robot(Path(__file__).resolve().parents[1] / "examples/adapters/planar.toml")
    model = replace(model, rest=Rest(q=(0.0, sign * 2.49), joints=(1,), tol=0.1))
    k = Kernel(SimBody(model, q=[0.2, sign * 2.4]), clock=VirtualClock(model.rate_hz))
    k.connect()
    k.enable()
    assert k.run({"do": "joints", "target_deg": {"2": float(np.degrees(sign * 2.0))}}).ok
    assert not model.rest.holds(k.state.q)
    k.set_home_route([])
    out = k.run(k.home_plan())
    assert out.ok, out.message
    assert model.rest.holds(k.state.q)
    assert k.state.q[0] == pytest.approx(0.2, abs=0.01)
    k.release()
    assert not k.enabled and not k.power_uncertain
    k.close()


@pytest.mark.parametrize("stops, rest_q", [((), -2.49), ((1,), 0.5)])
def test_home_refuses_when_rest_tolerance_cannot_accommodate_clearance(stops, rest_q):
    from dataclasses import replace
    from pathlib import Path

    from world_use import VirtualClock
    from world_use.bodies.sim import SimBody
    from world_use.body import Rest
    from world_use.config import load_robot

    model = load_robot(Path(__file__).resolve().parents[1] / "examples/adapters/planar.toml")
    model = replace(model, rest=Rest(q=(0.0, rest_q), joints=(1,), tol=0.005, stops=stops))
    k = Kernel(SimBody(model, q=[0.2, 0.0]), clock=VirtualClock(model.rate_hz))
    k.connect()
    k.set_home_route([])
    with pytest.raises(Refused, match="no supported rest target"):
        k.home_plan()
    assert k.active is None and not k.queue and not k.enabled
    k.close()


def test_hot_motor_goes_home_along_a_valid_route_and_otherwise_holds_and_alarms():
    k = make_kernel(temp_c=[30, 30, 79.0, 30, 30, 30])
    assert k.run({"do": "line", "forward": 0.08, "up": 0.06}).ok
    for _ in range(3000):                                        # holding a raised pose: the elbow heats
        k.tick()
        k.clock.wait()
        if any(e["kind"] == "hot" for e in k.events.since(0)):
            break
    alarms = [e for e in k.events.since(0) if e["kind"] == "hot"]
    assert alarms and "holding" in alarms[-1]["message"]
    q = k.cmd.q.copy()
    for _ in range(100):
        k.tick()
        k.clock.wait()
    assert np.allclose(k.cmd.q, q, atol=1e-6)                    # no route: it did not move on its own

    k2 = make_kernel(temp_c=[30, 30, 79.0, 30, 30, 30])
    assert k2.run({"do": "line", "forward": 0.08, "up": 0.06}).ok
    k2.set_home_route([])
    for _ in range(6000):
        k2.tick()
        k2.clock.wait()
        if not k2.enabled:
            break
    assert np.allclose(k2.cmd.q, HOME, atol=1e-4)
    assert not k2.enabled and not k2.power_uncertain and not k2.body.enabled


@pytest.mark.parametrize("step", [
    {"do": "checkpoint", "ask": "is the path clear?"},
    {"do": "hold"},
    {"do": "hold", "seconds": 3600},
    {"do": "guarded", "up": 0.02},
])
@pytest.mark.parametrize("nested", [False, True])
def test_home_routes_refuse_waits_and_contact_steps_even_when_nested(k, step, nested):
    route = [{"do": "seq", "steps": [[step]]}] if nested else [step]
    with pytest.raises(Refused, match="not allowed in a home route"):
        k.set_home_route(route)
    assert k.home_route is None


def test_home_route_is_a_snapshot_and_can_be_cleared(lifted):
    route = [{"do": "seq", "steps": [{"do": "line", "up": 0.01}]}]
    lifted.set_home_route(route)
    route[0]["steps"][0] = {"do": "checkpoint", "ask": "wait forever"}
    plan = lifted.home_plan()
    assert plan[0]["steps"][0]["do"] == "line"
    plan[0]["steps"].clear()
    assert lifted.home_plan()[0]["steps"]
    lifted.set_home_route(None, "operator moved objects into the return path")
    with pytest.raises(Refused, match="no home route"):
        lifted.home_plan()


def test_a_legacy_blocking_home_route_cannot_start_a_thermal_return(lifted, monkeypatch):
    from world_use.envelope import Trip

    lifted.home_route = ([{"do": "checkpoint", "ask": "clear?"}], lifted.events.seq + 1)
    monkeypatch.setattr(lifted.envelope, "watch", lambda *args: Trip("hot", "motor too hot"))
    q = lifted.cmd.q.copy()
    lifted.tick()
    assert lifted.active is None and lifted.enabled
    assert np.allclose(lifted.cmd.q, q, atol=1e-6)
    assert any("Operator must resolve power now" in e["message"] for e in lifted.events.since(0))


def test_a_user_label_cannot_bypass_thermal_return_and_cooling_does_not_cancel_release(lifted, monkeypatch):
    from world_use.envelope import Trip

    lifted.set_home_route([])
    job = lifted.submit({"do": "hold", "label": "home: motor hot"})
    for _ in range(20):
        lifted.tick()
        lifted.clock.wait()
    assert job.status == "running"
    monkeypatch.setattr(lifted.envelope, "watch", lambda *args: Trip("hot", "motor too hot"))
    lifted.tick()
    assert job.status == "stopped" and lifted.active is not None
    monkeypatch.setattr(lifted.envelope, "watch", lambda *args: None)
    for _ in range(3000):
        lifted.tick()
        lifted.clock.wait()
        if not lifted.enabled:
            break
    assert not lifted.enabled and not lifted.power_uncertain and not lifted.body.enabled


def test_a_job_submitted_during_the_thermal_return_is_refused_and_the_return_still_releases(lifted):
    """A hold, or `wu home` after the alarm, queued behind the return blocked its release at rest: the kernel
    faulted with the hot motor powered."""
    k = lifted
    k.set_home_route([])
    k.body.temp[:] = 81
    k.tick()
    thermal = k.active
    assert thermal is not None and thermal.behavior.label == "home: motor hot"
    assert k.submit({"do": "hold", "seconds": 1}).status == "refused"
    while not thermal.finished:
        k.tick()
        k.clock.wait()
    assert thermal.status == "done" and not k.enabled and not k.faulted and not k.power_uncertain


def test_a_failed_thermal_return_is_not_retried_automatically(lifted, monkeypatch):
    from world_use.envelope import Trip

    lifted.set_home_route([{"do": "line", "up": 100}])  # motion admission will refuse this
    monkeypatch.setattr(lifted.envelope, "watch", lambda *args: Trip("hot", "motor too hot"))
    for _ in range(20):
        lifted.tick()
        lifted.clock.wait()
    assert lifted.active is None and lifted.home_route is None and lifted.enabled
    assert len([j for j in lifted.jobs.values() if j.behavior.label == "home: motor hot"]) == 1


def test_hot_return_still_stops_for_a_jammed_arm(lifted, monkeypatch):
    k = lifted
    k.set_home_route([])
    q, g = k.state.q.copy(), k.state.gripper
    monkeypatch.setattr(k.body, "read", lambda: JointState(k.clock.now(), q.copy(), tau=k.chain.gravity(q),
                                                        temp=np.full(6, 81.), gripper=g, gripper_tau=0.))
    for _ in range(1000):
        k.tick()
        k.clock.wait()
        if k.home_route is None:
            break
    job = list(k.jobs.values())[-1]
    assert job.status == "surprise" and job.outcome.data["trip"] == "blocked"
    assert k.home_route is None and k.enabled
    assert np.allclose(k.cmd.q, q)


def test_failed_thermal_release_is_not_reported_done(lifted, monkeypatch):
    k = lifted
    k.set_home_route([])
    k.body.temp[:] = 81

    def failed_disable():
        raise OSError("one motor did not acknowledge disable")

    monkeypatch.setattr(k.body, "disable", failed_disable)
    for _ in range(3000):
        k.tick()
        k.clock.wait()
        if k.faulted:
            break
    job = list(k.jobs.values())[-1]
    assert job.status == "faulted" and "could not release" in job.outcome.message
    assert k.power_uncertain and k.home_route is None
    assert not any("thermal return complete" in e["message"] for e in k.events.since(0))


def test_no_heat_forecast_until_the_switch_on_transient_has_passed(lifted):
    k = lifted
    for _ in range(1500):
        k.tick()
        k.clock.wait()
    assert k.heat.minutes_left(k.manifest.temp_limit_c) is None             # 15-18 s after switching on


def test_heat_budget_is_reported(lifted):
    k = lifted
    for _ in range(3000):
        k.tick()
        k.clock.wait()
    left = k.heat.minutes_left(k.manifest.temp_limit_c)
    assert left is not None and left[0] == 2 and 1 < left[2] < 30   # the elbow, several minutes from its limit


def test_motion_time_is_measured(k):
    k.run({"do": "line", "forward": 0.08, "up": 0.06, "duration": 3.0})
    k.run({"do": "hold", "seconds": 3.0})
    s = k.tape.summary()
    assert abs(s["moving_s"] - 3.0) < 0.1 and abs(s["moving_share"] - 0.5) < 0.05
    assert s["tick_ms"] == dict(median=10.0, p99=10.0, max=10.0)


def test_after_touching_down_the_arm_can_lift_off_even_if_it_rests_a_hair_inside_the_modelled_table(lifted):
    k = lifted
    table_below(k)
    assert k.run({"do": "touchdown", "max": 0.08}).ok
    k.world.boxes["table"].pose[2, 3] += 0.002                   # the model says the table is 2 mm higher
    assert k.run({"do": "line", "up": 0.03}).ok
    out = k.run({"do": "line", "up": -0.05})                     # but going back down into it is refused
    assert out.status == "refused" and "table" in out.message


def test_an_intended_touchdown_inside_a_fragile_zone_ends_done_not_surprise(lifted):
    k = lifted
    tool = k.chain.fk(k.state.q)[:3, 3]
    table_below(k, .03, name="glass shelf")
    k.world.add_box("near glass", "fragile", center=tool - [0, 0, 0.03], size=[0.3, 0.3, 0.1], dtau=0.4)
    out = k.run({"do": "touchdown", "max": 0.06})
    assert out.ok, out.message


def test_grip_takes_millimetres(lifted):
    k = lifted
    assert k.run({"do": "gripper", "aperture_mm": 60}).ok          # open before the object appears between the jaws
    supported_object(k, size=[.04, .04, .04])
    out = k.run({"do": "grip", "start_mm": 60, "expect_mm": [35, 45]})
    assert out.ok and "mm" in out.message and "'block'" in out.message
    assert 35 <= k.manifest.gripper.aperture(out.data["contact_at"]) * 1000 <= 45


def test_a_gripper_described_in_metres_grips_and_lets_go():
    """Grip defaults and the let-go margin were reBot radians: a gripper in metres was refused, then never let go."""
    from dataclasses import replace

    from world_use.bodies.rebot import MANIFEST

    g, s = MANIFEST.gripper, MANIFEST.gripper.m_per_unit                 # the reBot's own gripper, in metres
    metres = replace(g, closed=0.0, open=(g.open - g.closed) * s, unit="m", m_per_unit=1.0, v_max=g.v_max * s,
                     track_tol=g.track_tol * s, tau_max=g.tau_max / s, squeeze=g.squeeze * s)
    k = make_kernel(manifest=replace(MANIFEST, gripper=metres), gripper=0.02)
    assert k.run([{"do": "line", "forward": 0.08, "up": 0.06}, {"do": "gripper", "aperture_mm": 60}]).ok
    supported_object(k, size=[.04, .04, .04])
    out = k.run({"do": "grip", "start_mm": 60, "expect_mm": [35, 45]})
    assert out.ok and k.world.held is not None, out.message
    assert k.run({"do": "gripper", "aperture_mm": 60}).ok and k.world.held is None and k.held_at is None
    k.close()


@pytest.mark.parametrize("params", [{"squeeze": 10}, {"start": 5}, {"expect": [0.5, 9]}, {"hold_effort": 9}])
def test_grip_values_beyond_the_grippers_own_limits_never_move_it(k, params):
    before = k.cmd.gripper
    out = k.run({"do": "grip", **params})
    assert out.status == "refused" and not k.faulted
    assert np.all(k.tape.arrays()["grip_cmd"] == before)


def test_contact_squeeze_obeys_position_and_speed_limits(lifted, monkeypatch):
    from dataclasses import replace

    k = lifted
    k.run({"do": "gripper", "to": 0.4})
    g = k.manifest.gripper
    read = k.body.read
    # Contact this close to fully closed puts the squeeze target past the limit, isolating command bounding.
    monkeypatch.setattr(k.body, "read", lambda: replace(read(), gripper_tau=.7 if k.cmd.gripper < .08 else 0))
    first = len(k.tape)
    assert k.run({"do": "grip"}).ok
    commands = k.tape.arrays()["grip_cmd"][first - 1:]
    assert commands.min() >= g.closed and commands.max() <= g.open
    assert np.abs(np.diff(commands)).max() * k.manifest.rate_hz <= g.v_max + 1e-9


def test_grip_reads_contact_after_issuing_the_final_close_command(lifted, monkeypatch):
    from dataclasses import replace

    k = lifted
    assert k.run({"do": "gripper", "to": .15}).ok
    read = k.body.read
    closed = k.manifest.gripper.closed
    monkeypatch.setattr(k.body, "read", lambda: replace(read(), gripper_tau=.7 if k.cmd.gripper == closed else 0))
    assert k.run({"do": "grip"}).ok
    assert any(e["kind"] == "grip" for e in k.events.since(0))


@pytest.mark.parametrize("nested", [False, True])
def test_a_step_refused_midway_leaves_no_feedforward_alone_or_in_a_plan(lifted, monkeypatch, nested):
    """Inside a plan, a grip refused mid-close left the closing speed commanded on every idle tick after it."""
    from dataclasses import replace

    k = lifted
    assert k.run({"do": "gripper", "to": 3.0}).ok
    read = k.body.read
    monkeypatch.setattr(k.body, "read",
                        lambda: replace(read(), gripper_tau=float("nan") if k.cmd.gripper < 2.0 else 0.0))
    spec = {"do": "grip", "hold_effort": 0.4}
    out = k.run([spec] if nested else spec)
    assert out.status == "refused" and "sensing" in out.message and out.message.startswith("step 1/1") == nested
    assert k.cmd.gripper_v == 0 and not np.any(k.cmd.dq) and not k.faulted


def test_grip_outside_the_expected_millimetres_says_so_in_millimetres(lifted):
    k = lifted
    assert k.run({"do": "gripper", "aperture_mm": 60}).ok
    supported_object(k, size=[.04, .04, .04])
    out = k.run({"do": "grip", "start_mm": 60, "expect_mm": [10, 20]})
    assert out.status == "surprise" and "10..20 mm" in out.message


def test_a_gripped_object_moves_with_the_tool_in_the_model_and_lands_where_it_is_let_go(lifted):
    """The simulator and the kernel keep separate worlds here: the kernel tracks the object on its own."""
    truth = World.from_dict(lifted.world.to_dict())
    k = make_kernel(sim_world=truth)
    assert k.run({"do": "line", "forward": 0.08, "up": 0.06}).ok
    assert k.run({"do": "gripper", "aperture_mm": 60}).ok
    supported_object(k, size=[.04, .04, .08])
    top = k.world.boxes["block support"].top
    assert k.run({"do": "grip", "start_mm": 60, "expect_mm": [35, 45]}).ok
    assert k.world.held is not None and k.world.held[0] == "block"
    before = k.world.boxes["block"].pose[:3, 3].copy()
    assert k.run({"do": "line", "up": 0.03}).ok
    assert np.allclose(k.world.boxes["block"].pose[:3, 3] - before, [0, 0, 0.03], atol=2e-3)
    assert k.run({"do": "gripper", "aperture_mm": 60}).ok
    assert k.world.held is None
    assert abs(k.world.boxes["block"].pose[2, 3] - (top + 0.04)) < 1e-6           # standing on the tray
    assert any(e["kind"] == "let_go" for e in k.events.since(0))


def _looping(body_hooks=None):
    """A kernel on a simulated reBot with its control loop running in a thread, as in the daemon."""
    world = World()
    body = bodies.make("sim", world, q=Q_REST, gripper=1.0)
    for name, hook in (body_hooks or {}).items():
        setattr(body, name, hook(getattr(body, name)))
    k = Kernel(body, world, RealClock(100.0))
    k.connect()
    stop = threading.Event()
    loop = threading.Thread(target=k.loop, args=(stop,), daemon=True)
    loop.start()
    return k, stop, loop


def _until(condition, timeout=10.0):
    end = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.01)


def test_once_the_loop_runs_only_its_thread_calls_the_body():
    """The daemon switches torque on and off from request threads; on the real reBot that meant a lock-free CAN
    driver called from two threads at once. Now the control thread does it, and the caller waits."""
    callers = []

    def spy(fn):
        def call(*a, **kw):
            callers.append(threading.current_thread())
            return fn(*a, **kw)
        return call
    k, stop, loop = _looping({name: spy for name in ("enable", "read", "command", "disable")})
    k.enable()
    job = k.submit({"do": "hold", "seconds": 30})
    _until(lambda: job.status == "running")
    with pytest.raises(Refused, match="job is running"):
        k.release()                                    # refused on the control thread, raised here
    with pytest.raises(RuntimeError, match="control loop is running"):
        k.run({"do": "hold", "seconds": 0.1})
    k.stop("test")
    _until(lambda: job.finished)
    stop.set()
    loop.join(2.0)
    assert k.enabled and callers and all(t is loop for t in callers)
    with pytest.raises(Refused, match="control loop has stopped"):
        k.enable()                                     # nothing would be left to command the motors
    k.release()                                        # at rest, torque off from here, inline
    assert not k.enabled and callers[-1] is threading.current_thread()


@pytest.mark.parametrize("failed_call", ["read", "command"])
def test_a_body_that_raises_suspends_commands_even_after_reconnection(failed_call):
    fail = threading.Event()
    commands = []

    def flaky(fn):
        def call(*args):
            if fail.is_set():
                fail.clear()
                raise OSError("the CAN adapter went away")
            return fn(*args)
        return call

    def spy(fn):
        def command(*args):
            commands.append(args)
            return fn(*args)
        return command

    hooks = {"command": spy}
    hooks[failed_call] = (lambda fn: flaky(spy(fn))) if failed_call == "command" else flaky
    k, stop, loop = _looping(hooks)
    k.enable()
    job = k.submit({"do": "hold", "seconds": 30})
    _until(lambda: job.status == "running")
    fail.set()
    _until(lambda: job.finished)
    assert job.outcome.status == "faulted" and "adapter went away" in job.outcome.message
    assert loop.is_alive() and k.faulted and k.power_uncertain
    assert any(e["kind"] == "fault" and e["level"] == "alarm" for e in k.events.since(0))
    count, stamp = len(commands), k.state.t
    _until(lambda: k.state.t > stamp + 0.1)            # reads recovered; motion commands must stay suspended
    assert len(commands) == count
    with pytest.raises(Refused, match="motor power is unconfirmed"):
        k.reset()
    assert k.submit({"do": "line", "up": 0.02}).status == "refused"
    k.release()                                     # fresh feedback at rest and successful disable are required
    assert not k.enabled and not k.power_uncertain
    k.reset()
    stop.set()
    loop.join(2.0)


def test_cached_feedback_cannot_authorize_a_release(k, monkeypatch):
    cached = k.state
    monkeypatch.setattr(k.body, "read", lambda: cached)
    for _ in range(101):
        k.clock.wait()
    with pytest.raises(ConnectionError, match="no new feedback"):
        k.release()
    assert k.faulted and k.power_uncertain and k.body.enabled
    assert k.feedback_status()["stale"]


def test_an_embedded_kernel_also_latches_command_failure(k, monkeypatch):
    commands = []

    def lost(*args):
        commands.append(args)
        raise ConnectionError("CAN command failed")

    monkeypatch.setattr(k.body, "command", lost)
    with pytest.raises(ConnectionError, match="CAN command failed"):
        k.tick()
    assert k.faulted and k.power_uncertain
    k.tick()
    assert len(commands) == 1


def test_a_fault_with_torque_off_keeps_it_off_until_an_operator_resets():
    """On the reBot a parameter read timed out with torque off (2026-09-28) and the kernel faulted. A later enable
    switched the motors on all the same, under a kernel that then refused every job. Enable waits for the reset."""
    fail = threading.Event()

    def flaky(fn):
        def read():
            if fail.is_set():
                fail.clear()
                raise OSError("parameter 0x7019 not received within 300ms")
            return fn()
        return read
    k, stop, loop = _looping({"read": flaky})
    fail.set()
    _until(lambda: k.faulted)
    with pytest.raises(Refused, match="faulted"):
        k.enable()
    assert not k.enabled and not k.body.enabled
    k.reset()
    k.enable()
    assert k.enabled
    stop.set()
    loop.join(2.0)



def test_torque_noise_does_not_read_as_contact_and_real_contact_still_does():
    """A real reBot's loaded joints read +-0.5-1 Nm from one tick to the next while holding still (2026-09-27): a
    baseline taken from one reading was enough to end a guarded move 2 mm into free air."""
    k = make_kernel(noise=0.5)
    assert k.run({"do": "line", "forward": 0.08, "up": 0.06}).ok
    for _ in range(3):
        out = k.run({"do": "guarded", "up": -0.02, "dtau": 0.6, "expect_contact": False})
        assert out.ok and "no contact" in out.message, out.message
    table_below(k, .02)
    out = k.run({"do": "touchdown", "max": 0.05, "dtau": 0.6})
    assert out.ok and "contact after" in out.message and "noise raised the threshold" in out.message, out.message


def test_noise_never_raises_a_fragile_zones_threshold():
    """A glass zone asking for 0.3 Nm was judged at 1.4-2.5 Nm on an arm this noisy, and nothing said so. It keeps
    its 0.3: a noisy arm may stop on nothing there, and says so, rather than press harder than the zone allows."""
    k = make_kernel(noise=0.5)
    assert k.run({"do": "line", "forward": 0.08, "up": 0.06}).ok
    p = k.chain.fk(k.state.q)[:3, 3]
    k.world.add_box("glass", "fragile", center=p, size=[0.3, 0.3, 0.3], frame="base", dtau=0.3)
    assert k.run({"do": "hold", "seconds": 0.5}).ok
    out = k.run({"do": "line", "forward": 0.03, "duration": 3.0})
    assert out.status == "surprise" and "(limit 0.3; noise alone can cross the limit" in out.message, out.message


def test_noise_raises_a_threshold_at_most_twofold():
    from world_use.behaviors import ContactSense

    k = make_kernel(noise=3.0)                                   # a joint gone this noisy must not go numb
    for _ in range(k.residuals.window):
        k.tick()
        k.clock.wait()
    sense = ContactSense(k)
    asked = np.array([j.contact_dtau for j in k.manifest.joints])
    assert np.allclose(sense.limits(asked), 2 * asked) and np.allclose(sense.limits(asked, 0.3), 0.3)


def test_a_job_waits_for_a_torque_baseline_after_switching_on():
    k = make_kernel(noise=0.5)                                  # just switched on: no readings yet
    job = k.submit({"do": "line", "up": 0.02})
    for _ in range(k.residuals.need - 1):
        k.tick()
        k.clock.wait()
    assert job.status == "queued"
    k.tick()
    k.clock.wait()
    assert job.status == "running" and (k._sense.floor > 0).all()   # judged against a baseline, noise included


def test_a_joint_on_its_rest_stop_is_not_judged_and_is_re_zeroed_until_it_leaves(k):
    """Folded, the reBot's shoulder and elbow rest on hard stops that carry part of their load: arriving there or
    lifting off moved ~2 Nm between motor and stop with nothing touched, and stopped folds home on hardware."""
    from world_use.behaviors import ContactSense

    assert k.manifest.rest.stops == (1, 2)
    for _ in range(k.residuals.need):
        k.tick()
        k.clock.wait()
    sense = ContactSense(k)

    def feel(q, extra):
        k.state = JointState(k.state.t, np.asarray(q, float), None, k.chain.gravity(q) + extra, k.state.temp)
        for _ in range(5):
            dev = sense.deviation(k)
        return dev
    stop_load = np.array([0, 0, -2.0, 0, 0, 0])
    assert abs(feel(Q_REST, stop_load)[2]) < 1e-9                # on the stop: not judged
    lifted = Q_REST + [0, 0.1, 0.1, 0, 0, 0]
    assert abs(feel(lifted, stop_load)[2]) < 1e-9                # off it: judged from where it let go
    assert feel(lifted, stop_load + [0, 0, 3.5, 0, 0, 0])[2] > 3.0



def test_a_gripper_trip_ends_the_gripper_step_inside_a_plan_too(lifted, monkeypatch):
    """Alone, a gripper step closing on something too wide ended at the first trip; inside a plan it tripped on
    every tick and still ended "done"."""
    k = lifted
    assert k.run({"do": "gripper", "to": 3.0}).ok
    supported_object(k)
    from dataclasses import replace
    read = k.body.read
    monkeypatch.setattr(k.body, "read", lambda: replace(read(), gripper_tau=5 if k.cmd.gripper < 2.3 else 0))
    out = k.run([{"do": "hold", "seconds": 0.1}, {"do": "gripper", "to": 1.0}])
    assert out.status == "surprise" and out.message.startswith("step 2/2: gripper")
    assert sum(e["kind"] == "gripper_trip" for e in k.events.since(0)) == 1


def test_home_puts_the_gripper_back_as_it_was_found(lifted):
    """The session ended with the reBot's gripper open at 4.39 rad: past pi, it comes back a turn low after a power
    cycle. Home now closes it to where the session found it."""
    k = lifted
    assert k.run({"do": "gripper", "to": 3.0}).ok
    k.set_home_route([])
    plan = k.home_plan()
    assert plan[-1] == {"do": "gripper", "to": 1.0, "label": "gripper as it was found"}
    assert k.run({"do": "seq", "steps": plan}).ok and abs(k.cmd.gripper - 1.0) < 1e-9


def test_a_grip_of_the_wrong_width_still_holds_and_home_does_not_let_go():
    """A grip that found an unexpected width ended in a surprise without counting as holding, so home would have
    closed the gripper on the object, or opened it at rest."""
    from world_use import views

    truth = World()                                           # the simulator knows the block, the kernel does not
    k = make_kernel(sim_world=truth)
    assert k.run([{"do": "line", "forward": 0.08, "up": 0.06}, {"do": "gripper", "to": 3.0}]).ok
    supported_object(k, known=False)
    out = k.run({"do": "grip", "expect_mm": [10, 20]})
    assert out.status == "surprise" and k.held_at is not None and k.world.held is None
    k.set_home_route([])
    assert all(step["do"] != "gripper" for step in k.home_plan())
    assert views.status(k)["holding"] == "something the world has no box for"
    assert k.run({"do": "gripper", "to": 3.0}).ok and k.held_at is None     # opened past it: let go


def test_grip_squeezes_by_the_grippers_own_amount(lifted):
    """0.1 rad at the reBot gripper's kp of 50 is 5 Nm on anything rigid, past its 4 Nm watchdog."""
    k = lifted
    assert k.manifest.gripper.squeeze == 0.05
    assert k.run({"do": "gripper", "to": 3.0}).ok
    supported_object(k)
    out = k.run({"do": "grip"})
    assert out.ok and abs(k.cmd.gripper - (out.data["contact_at"] - 0.05)) < 2e-3


def test_the_tape_keeps_every_tick_across_its_blocks(monkeypatch):
    from world_use.recorder import Tape

    monkeypatch.setattr(Tape, "CHUNK", 5)
    tape = Tape(2)
    for i in range(12):
        tape.add(i / 100, True, i % 2 == 0, 1, [i, i], [i, -i], None, [30.0, 31.0], None, 1.0, None)
    a = tape.arrays()
    assert len(tape) == 12 and a["t"].tolist() == [i / 100 for i in range(12)]
    assert a["q"][:, 1].tolist() == [-i for i in range(12)] and np.isnan(a["tau"]).all() and a["grip"].sum() == 12


def _pointing(k):
    from world_use.world import along
    W = k.world.frame("work").T[:3, :3]
    R = k.chain.fk(k.state.q)[:3, :3]
    g = k.manifest.gripper
    return W.T @ R @ np.asarray(g.approach), along(W.T @ R @ np.asarray(g.opens_along))


def test_move_to_can_point_the_gripper_down(k):
    """Near its base a real reBot can point down only by tilting: turning the wrist or base that low is refused."""
    L = float(k.world.from_base("work", k.chain.fk(k.state.q)[:3, 3])[1])
    q0 = k.state.q.copy()
    out = k.run({"do": "move_to", "to": [0.22, L, 0.10], "point": "down"})
    assert out.ok and 0.5 < out.data["off_deg"] < 5 and "now pointing straight down" in out.message, out.message
    down, jaws = _pointing(k)
    assert np.degrees(np.arccos(-down[2])) < 5 and jaws == "left and right"
    assert np.degrees(np.abs(k.state.q - q0)[[0, 4, 5]]).max() < 1.0              # base and wrist held still
    refused = make_kernel().run({"do": "move_to", "to": [0.22, L, 0.10], "point": "down", "within_deg": 1})
    assert refused.status == "refused" and refused.data["rule"] == "turn_clearance" and "within_deg" in refused.hint


def test_move_to_turns_in_place_and_says_when_it_cannot(k):
    assert k.run([{"do": "line", "forward": 0.08, "up": 0.14}]).ok
    p0 = k.chain.fk(k.state.q)[:3, 3].copy()
    assert k.run({"do": "move_to", "jaws": "up"}).ok                               # a wrist roll, high up
    assert np.linalg.norm(k.chain.fk(k.state.q)[:3, 3] - p0) < 1e-3 and _pointing(k)[1] == "up and down"
    out = k.run({"do": "move_to", "point": "down"})
    assert out.status == "refused" and "cannot turn to point straight down here" in out.message, out.message
    for bad in ({"point": "sideways"}, {"point": "down", "jaws": "up"}):
        assert k.run({"do": "move_to", **bad}).status == "refused"


def test_a_check_says_where_the_gripper_ends_pointing(k):
    from world_use import check
    L = float(k.world.from_base("work", k.chain.fk(k.state.q)[:3, 3])[1])
    report = check({"do": "move_to", "to": [0.22, L, 0.10], "point": "down"}, k)
    end = report.gripper_end
    assert end["points"] == "straight down" and end["jaws"] == "left and right" and 80 < end["turned_deg"] < 100
    assert end["points"] in str(report)


def test_a_free_checkpoint_takes_any_answer_and_keeps_it_with_where_the_tool_was(k):
    job = k.submit([{"do": "checkpoint", "ask": "where is the tool?", "expect": None}, {"do": "line", "up": 0.02},
                    [{"do": "checkpoint", "ask": "and now?", "expect": None}]])
    for answer in ("512,300", "unseen"):
        while job.status != "waiting":
            k.tick()
            k.clock.wait()
        k.answer(job.id, answer)
    while not job.finished:
        k.tick()
        k.clock.wait()
    first, second = job.outcome.data["answers"]
    assert first["answer"] == "512,300" and second["answer"] == "unseen" and second["step"] == 3
    assert abs(second["tool"][2] - first["tool"][2] - 0.02) < 2e-3                  # measured, 2 cm apart


def test_a_rehearsal_assumes_any_answer_at_a_free_checkpoint(k):
    from world_use import check
    report = check([{"do": "checkpoint", "ask": "where is the tool?", "expect": None}], k)
    assert report.ok and "(any answer)" in report.assumed[0]


def _across(k):
    along = k.tool[:3, :3] @ np.asarray(k.manifest.gripper.opens_along, float)
    along = np.array([along[0], along[1], 0.0]) / np.linalg.norm(along[:2])
    return np.cross([0.0, 0.0, 1.0], along)


def test_grasp_searches_on_the_spot_after_a_miss_and_holds_what_it_finds(lifted):
    k = lifted
    tool = k.chain.fk(k.state.q)[:3, 3]
    supported_object(k, size=[.004, .012, .04], center=tool - .046 * _across(k))
    assert k.run({"do": "grip", "start": 3.0}).status == "surprise"          # a plain grip misses it
    out = k.run({"do": "grasp", "start": 3.0, "expect": [0.4, 1.0], "search_mm": [[12, 0], [-64, 0]]})
    assert out.ok and out.data["tries"] == 3 and out.data["offset_mm"] == [-64.0, 0.0], out.message
    assert np.linalg.norm(k.chain.fk(k.state.q)[:3, 3] - (tool - 0.064 * _across(k))) < 0.003
    assert any(e["kind"] == "grasp_retry" for e in k.events.since(0))


def test_grasp_that_finds_nothing_is_a_surprise_after_its_last_try(lifted):
    out = lifted.run({"do": "grasp", "start": 3.0, "search_mm": [[5, 0]]})
    assert out.status == "surprise" and out.data["tries"] == 2 and "after 2 tries" in out.message


def test_a_weak_grip_is_a_surprise_and_grasp_retries_it(lifted):
    k = lifted
    supported_object(k, size=[.012, .012, .04])
    firm = k.run({"do": "grip", "start": 3.0, "expect": [0.4, 1.0]})
    assert firm.ok and firm.data["holding_effort"] is not None
    k.run({"do": "gripper", "to": 3.0})
    weak = k.run({"do": "grip", "start": 3.0, "hold_effort": max(.2, 2 * abs(firm.data["holding_effort"]))})
    assert weak.status == "surprise" and "weak grip" in weak.message, weak.message
    k.run({"do": "gripper", "to": 3.0})
    strict = max(.2, 2 * abs(firm.data["holding_effort"]))
    out = k.run({"do": "grasp", "start": 3.0, "hold_effort": strict, "search_mm": [[4, 0]]})
    assert out.status == "surprise" and out.data["tries"] == 2 and "weak grip" in out.message
    assert any(e["kind"] == "grasp_retry" and "weak grip" in e["message"] for e in k.events.since(0))


def test_grasp_lifts_past_the_turn_height_before_it_shifts(lifted):
    k = lifted
    tool = k.chain.fk(k.state.q)[:3, 3]
    up = float(k.world.from_base("work", tool)[2])
    k.envelope.override("turn_height", up + 0.03, "the base may only turn 3 cm higher")    # 8 mm would not do
    along = np.cross(_across(k), [0.0, 0.0, 1.0])                  # sideways: a shift this way turns the base
    supported_object(k, size=[.012, .012, .04], center=tool + .065 * along + .018 * _across(k))
    out = k.run({"do": "grasp", "start": 3.0, "expect": [0.4, 1.0], "search_mm": [[0, 65], [0, -65]]})
    assert out.ok and out.data["tries"] in (2, 3), out.message


def test_the_flight_record_is_written_when_the_adapter_fails_to_close(tmp_path):
    from world_use import Kernel, VirtualClock, World, bodies
    world = World()
    body = bodies.make("sim", world)
    k = Kernel(body, world, VirtualClock(100), run_dir=tmp_path)
    k.connect()

    def unplugged():
        raise OSError("pcan uninitialize failed: PCAN_ERROR_ILLHW")
    body.close = unplugged
    k.close()
    assert (tmp_path / "summary.json").exists() and list((tmp_path / "tape").glob("[0-9]*.npz"))
    assert any(e["kind"] == "adapter" and "ILLHW" in e["message"] for e in k.events.since(0))


@pytest.mark.parametrize(("phase", "nested"), [("grip", False), ("gripper", True), ("lines", True)])
def test_a_gripper_trip_ends_the_grasp_without_retrying(lifted, monkeypatch, phase, nested):
    from dataclasses import replace

    k = lifted
    spec = {"do": "grasp", "start": 1.0, "search_mm": [[6, 0]]}
    job = k.submit([[spec], {"do": "gripper", "to": 3.0}] if nested else spec)
    grasp = job.behavior.steps[0].steps[0] if nested else job.behavior
    for _ in range(3000):
        k.tick()
        k.clock.wait()
        if job.status == "running" and grasp.current is not None and grasp.current.kind == phase:
            break
    else:
        pytest.fail(f"grasp did not reach {phase}")
    retries = sum(e["kind"] == "grasp_retry" for e in k.events.since(0))
    read = k.body.read
    monkeypatch.setattr(k.body, "read", lambda: replace(read(), gripper_tau=k.manifest.gripper.tau_max + 1))
    k.tick()
    assert job.status == "surprise" and k.active is None
    assert k.cmd.gripper == k.state.gripper and k.cmd.gripper_v == 0
    for _ in range(5):
        k.clock.wait()
        k.tick()
    assert k.cmd.gripper == k.state.gripper and k.cmd.gripper_v == 0
    assert sum(e["kind"] == "grasp_retry" for e in k.events.since(0)) == retries


@pytest.mark.parametrize(("sensed", "reading"), [(False, None), (True, None), (True, float("nan"))])
def test_hold_effort_requires_sensing_and_a_finite_reading(lifted, monkeypatch, sensed, reading):
    from dataclasses import replace

    k = lifted
    if not sensed:
        k.manifest = replace(k.manifest, sensing=k.manifest.sensing - {"gripper_effort"})
    read = k.body.read
    monkeypatch.setattr(k.body, "read", lambda: replace(read(), gripper_tau=reading))
    before = k.cmd.gripper
    out = k.run({"do": "grasp", "start": 3.0, "hold_effort": 0.4})
    assert out.status == "refused" and "sensing" in out.message
    assert k.cmd.gripper == before and not k.faulted
    assert not any(e["kind"] == "grasp_retry" for e in k.events.since(0))


def test_grasp_on_a_robot_without_a_gripper_is_refused(k):
    from dataclasses import replace

    k.manifest = replace(k.manifest, gripper=None)
    out = k.run({"do": "grasp", "start": 1.0})
    assert out.status == "refused" and "no gripper" in out.message and not k.faulted
