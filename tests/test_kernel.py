"""Kernel semantics on a simulated reBot: refusals, surprises, contact, grip, checkpoints, stop, heat, home."""
import numpy as np
import pytest

from conftest import Q_REST, make_kernel
from world_use import Refused, World


def test_line_moves_the_tool_by_the_request_in_the_work_frame(k):
    p0 = k.world.from_base("work", k.chain.fk(k.state.q)[:3, 3])
    out = k.run({"do": "line", "forward": 0.08, "up": 0.06})
    assert out.ok
    k.run({"do": "hold", "seconds": 0.2})
    p1 = k.world.from_base("work", k.chain.fk(k.state.q)[:3, 3])
    assert np.allclose(p1 - p0, [0.08, 0, 0.06], atol=1e-3)


def test_refused_command_moves_nothing_and_cancels_what_was_queued(lifted):
    k = lifted
    q0 = k.cmd.q.copy()
    bad = k.submit({"do": "line", "forward": 0.40})              # longer than one segment may be
    queued = k.submit({"do": "line", "up": 0.01})
    while not queued.finished:
        k.tick(); k.clock.wait()
    assert bad.status == "refused" and "segment" in bad.outcome.message
    assert queued.status == "cancelled"
    assert np.allclose(k.cmd.q, q0)


def test_unknown_behavior_is_refused_at_submit(k):
    with pytest.raises(Refused):
        k.submit({"do": "teleport", "to": [0, 0, 1]})


def test_too_fast_is_refused_with_a_hint(lifted):
    out = lifted.run({"do": "line", "forward": 0.05, "duration": 0.2})
    assert out.status == "refused" and out.hint


def test_turning_the_base_at_table_height_is_refused(k):
    out = k.run({"do": "joints", "delta_deg": {"1": 10}})
    assert out.status == "refused" and "table" in out.message


def test_touchdown_finds_a_table_and_stops_on_it(lifted):
    k = lifted
    top = k.chain.fk(k.state.q)[2, 3] - 0.04
    k.world.add_box("table", "surface", center=[0.3, 0, top - 0.01], size=[1, 1, 0.02])   # same world as the sim
    out = k.run({"do": "touchdown", "max": 0.08})
    assert out.ok, out.message
    assert abs(out.data["moved_m"] - 0.04) < 0.006
    assert abs(k.chain.fk(k.state.q)[2, 3] - top) < 0.003
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
    top = k.chain.fk(k.state.q)[2, 3] - 0.03
    sim_world.add_box("box nobody mentioned", "surface", center=[0.3, 0, top - 0.05], size=[1, 1, 0.1])
    out = k.run({"do": "line", "up": -0.06})
    assert out.status == "surprise"
    assert np.abs(k.cmd.q - k.state.q).max() < 0.01                 # holding where it is, not pressing on
    assert k.last_touch > 0


def test_grip_on_an_object_reports_where_the_fingers_met_it(lifted):
    k = lifted
    tool = k.chain.fk(k.state.q)[:3, 3]
    k.body.world.add_box("block", "object", center=tool, size=[0.03, 0.012, 0.03], grip_width=0.012)
    out = k.run({"do": "grip", "start": 3.0, "expect": [0.4, 1.0], "squeeze": 0.1})
    assert out.ok, out.message
    assert abs(out.data["contact_at"] - (0.05 + 0.012 / 0.020)) < 0.08
    assert out.data["holding_effort"] < -0.5


def test_grip_on_nothing_is_a_surprise(lifted):
    out = lifted.run({"do": "grip", "start": 3.0})
    assert out.status == "surprise" and "nothing" in out.message


def test_grip_outside_the_expected_width_is_a_surprise(lifted):
    k = lifted
    assert k.run({"do": "gripper", "to": 3.0}).ok                  # open wider than the object before it appears
    tool = k.chain.fk(k.state.q)[:3, 3]
    k.body.world.add_box("big", "object", center=tool, size=[0.04, 0.04, 0.03], grip_width=0.04)
    out = k.run({"do": "grip", "start": 3.0, "expect": [0.4, 1.0]})
    assert out.status == "surprise" and out.observed > 1.0


def test_checkpoint_waits_for_an_answer_and_a_different_answer_ends_the_plan(lifted):
    k = lifted
    job = k.submit([{"do": "checkpoint", "ask": "is the loop between the jaws?", "view": "side"},
                    {"do": "line", "up": 0.02}])
    for _ in range(20):
        k.tick(); k.clock.wait()
    assert job.status == "waiting" and job.question["view"] == "side"
    q_hold = k.cmd.q.copy()
    for _ in range(200):                                        # the arm holds while the question waits
        k.tick(); k.clock.wait()
    assert np.allclose(k.cmd.q, q_hold)
    k.answer(job.id, "no")
    while not job.finished:
        k.tick(); k.clock.wait()
    assert job.status == "surprise" and job.outcome.observed == "no"


def test_stop_holds_where_the_arm_is(lifted):
    k = lifted
    job = k.submit({"do": "line", "forward": 0.05, "duration": 4.0})
    for _ in range(100):
        k.tick(); k.clock.wait()
    k.stop("operator said stop")
    for _ in range(3):
        k.tick(); k.clock.wait()
    assert job.status == "stopped"
    q = k.cmd.q.copy()
    for _ in range(50):
        k.tick(); k.clock.wait()
    assert np.allclose(k.cmd.q, q)


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


def test_home_folds_back_to_the_session_start_and_release_is_then_allowed(lifted):
    k = lifted
    with pytest.raises(Refused):
        k.release()                                              # raised: releasing would drop the arm
    k.set_home_route([])
    out = k.run({"do": "seq", "steps": k.home_plan()})
    assert out.ok, out.message
    assert np.allclose(k.cmd.q, Q_REST, atol=1e-4)
    k.release()
    assert not k.enabled


def test_hot_motor_goes_home_along_a_valid_route_and_otherwise_holds_and_alarms():
    k = make_kernel(temp_c=[30, 30, 79.0, 30, 30, 30])
    assert k.run({"do": "line", "forward": 0.08, "up": 0.06}).ok
    for _ in range(3000):                                        # holding a raised pose: the elbow heats
        k.tick(); k.clock.wait()
        if any(e["kind"] == "hot" for e in k.events.since(0)):
            break
    alarms = [e for e in k.events.since(0) if e["kind"] == "hot"]
    assert alarms and "holding" in alarms[-1]["message"]
    q = k.cmd.q.copy()
    for _ in range(100):
        k.tick(); k.clock.wait()
    assert np.allclose(k.cmd.q, q, atol=1e-6)                    # no route: it did not move on its own

    k2 = make_kernel(temp_c=[30, 30, 79.0, 30, 30, 30])
    assert k2.run({"do": "line", "forward": 0.08, "up": 0.06}).ok
    k2.set_home_route([])
    for _ in range(6000):
        k2.tick(); k2.clock.wait()
        if k2.active is None and np.allclose(k2.cmd.q, Q_REST, atol=1e-4):
            break
    assert np.allclose(k2.cmd.q, Q_REST, atol=1e-4)


def test_heat_budget_is_reported(lifted):
    k = lifted
    for _ in range(1500):
        k.tick(); k.clock.wait()
    left = k.heat.minutes_left(k.manifest.temp_limit_c)
    assert left is not None and left[0] == 2 and 1 < left[2] < 30   # the elbow, several minutes from its limit


def test_motion_time_is_measured(k):
    k.run({"do": "line", "forward": 0.08, "up": 0.06, "duration": 3.0})
    k.run({"do": "hold", "seconds": 3.0})
    s = k.tape.summary(k.manifest.rate_hz)
    assert abs(s["moving_s"] - 3.0) < 0.1 and abs(s["moving_share"] - 0.5) < 0.05


def test_after_touching_down_the_arm_can_lift_off_even_if_it_rests_a_hair_inside_the_modelled_table(lifted):
    k = lifted
    top = k.chain.fk(k.state.q)[2, 3] - 0.04
    k.world.add_box("table", "surface", center=[0.3, 0, top - 0.01], size=[1, 1, 0.02])
    assert k.run({"do": "touchdown", "max": 0.08}).ok
    k.world.boxes["table"].pose[2, 3] += 0.002                   # the model says the table is 2 mm higher
    assert k.run({"do": "line", "up": 0.03}).ok
    out = k.run({"do": "line", "up": -0.05})                     # but going back down into it is refused
    assert out.status == "refused" and "table" in out.message


def test_an_intended_touchdown_inside_a_fragile_zone_ends_done_not_surprise(lifted):
    k = lifted
    tool = k.chain.fk(k.state.q)[:3, 3]
    top = tool[2] - 0.03
    k.world.add_box("glass shelf", "surface", center=[0.3, 0, top - 0.01], size=[1, 1, 0.02])
    k.world.add_box("near glass", "fragile", center=tool - [0, 0, 0.03], size=[0.3, 0.3, 0.1], dtau=0.4)
    out = k.run({"do": "touchdown", "max": 0.06})
    assert out.ok, out.message
