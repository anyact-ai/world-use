"""Grasp and place run the caller's commands in order, stop at the first failure and never certify the result."""
import asyncio

import pytest
from conftest import supported_object

from world_use.behaviors import Outcome
from world_use.procedures import async_place, feedback, grasp, place

GRASP = dict(approach={"approach": 1}, close={"close": 1}, check={"check": 1}, lift={"lift": 1})
PLACE = dict(move={"move": 1}, release={"release": 1}, check={"check": 1}, withdraw={"withdraw": 1})


def recorder(fail_at=None):
    """execute and observe callbacks: the command numbered fail_at fails, and each observation counts the commands
    run so far."""
    calls = []

    def execute(command):
        calls.append(command)
        if len(calls) == fail_at:
            return Outcome("surprise", "fixture", "measured mismatch", expected=20, observed=4, hint="inspect first")
        return Outcome("done", "fixture", "command completed")

    return calls, execute, lambda: dict(executed=len(calls))


def test_grasp_checks_twice_and_observes_around_the_lift():
    calls, execute, observe = recorder()
    out = grasp(execute, observe, **GRASP)
    assert calls == [{"approach": 1}, {"close": 1}, {"check": 1}, {"lift": 1}, {"check": 1}]
    assert out.ok and out.data["assessment"] == "unknown" and not out.data["completion_verified"]
    assert feedback(out)["observations"] == {"before_lift": {"executed": 3}, "after": {"executed": 5}}


@pytest.mark.parametrize("procedure,stage,failed_at", [
    *[(grasp, s, i) for i, s in enumerate(["approach", "close", "check_before_lift", "test_lift",
                                           "check_after_lift"], 1)],
    *[(place, s, i) for i, s in enumerate(["arrive", "release", "check_release", "withdraw"], 1)]])
def test_a_failed_command_ends_the_procedure_and_reports_it(procedure, stage, failed_at):
    calls, execute, observe = recorder(fail_at=failed_at)
    result = procedure(execute, observe, **(GRASP if procedure is grasp else PLACE))
    assert len(calls) == failed_at
    assert result.status == "surprise" and result.message.startswith(stage)
    assert not result.data["completion_verified"]
    report = feedback(result)
    assert report["expected"] == 20 and report["observed"] == 4 and report["hint"] == "inspect first"
    assert report["observations"]["after"] == {"executed": failed_at}
    assert report["stages"][-1] == dict(stage=stage, status="surprise")
    assert report["last_command"]["command"] == calls[-1]


def test_exceptions_propagate_without_another_command():
    calls = []

    def execute(command):
        calls.append(command)
        raise RuntimeError("native episode ended")

    with pytest.raises(RuntimeError, match="native episode ended"):
        place(execute, dict, **PLACE)
    assert len(calls) == 1

    def unavailable():
        raise RuntimeError("camera unavailable")

    with pytest.raises(RuntimeError, match="camera unavailable"):
        grasp(execute, unavailable, **GRASP)
    assert len(calls) == 1


def test_bound_targets_must_stay_valid_at_every_checkpoint():
    calls, execute, _ = recorder()

    def bound(shelf_valid):
        def observe():
            return dict(targets=[dict(target="cup", valid=True), dict(target="shelf", valid=shelf_valid())])
        return place(execute, observe, **PLACE, object_target="cup", destination_target="shelf")

    refused = bound(lambda: False)
    assert refused.status == "refused" and not calls
    assert refused.data["intent"] == {"object": "cup", "destination": "shelf"}
    # Valid selections and completed commands cannot certify the right cup rests on the shelf.
    done = bound(lambda: True)
    assert done.ok and len(calls) == 4 and not done.data["completion_verified"]
    calls.clear()
    stopped = bound(lambda: not calls)              # lost once the arm has arrived
    assert stopped.status == "stopped" and len(calls) == 1 and "shelf" in stopped.message
    assert stopped.data["checkpoint"] == "before_release" and stopped.data["next_stage"] == "release"


@pytest.mark.parametrize("checkpoint,commands,next_stage",
                         [("before_close", 1, "close"), ("before_lift", 3, "test_lift"), ("after", 5, None)])
def test_grasp_assessment_stops_at_the_decision_point(checkpoint, commands, next_stage):
    calls, execute, observe = recorder()

    def assess(name, observations):
        assert observations[name]["executed"] == len(calls)
        if name == checkpoint:
            return Outcome("stopped", "visual_check", "identity or retention is uncertain", observed="occluded")
        return Outcome("done", "visual_check", "caller checked this stage")

    out = grasp(execute, observe, **GRASP, assess=assess)
    assert out.status == "stopped" and len(calls) == commands
    report = feedback(out)
    assert report["checkpoint"] == checkpoint and report["next_stage"] == next_stage
    assert report["observed"] == "occluded" and not report["completion_verified"]


def test_place_assessor_sees_new_geometry_and_cannot_rewrite_recorded_evidence():
    calls = []
    observation = dict(error_m=.001)

    def execute(command):
        calls.append(command)
        observation["error_m"] = .04  # the grasp changed during transport despite a successful arrival
        return Outcome("done", "fixture", "arrived")

    def assess(name, observations):
        assert name == "before_release" and observations["before"]["error_m"] == .001
        measured = observations[name]["error_m"]
        observations[name]["error_m"] = 0
        return Outcome("surprise", "placement", "object missed its destination", expected=.005, observed=measured)

    out = place(execute, lambda: observation, **PLACE, assess=assess)
    assert len(calls) == 1 and out.observed == .04
    assert out.data["observations"]["before_release"]["error_m"] == .04


def test_an_assessor_may_abstain_only_where_no_stage_follows():
    calls, execute, observe = recorder()
    out = place(execute, observe, **PLACE, assess=lambda checkpoint, _: None if checkpoint == "after" else
                Outcome("done", "fixture", "continue"))
    assert out.ok and out.data["assessment"] == "unknown"
    calls.clear()
    with pytest.raises(TypeError, match="Outcome"):
        place(execute, observe, **PLACE, assess=lambda *_: None)
    assert calls == [PLACE["move"]]                  # never released


@pytest.mark.parametrize("asynchronous", [False, True], ids=["place", "async_place"])
def test_passed_assessments_report_caller_checks_not_physical_support(asynchronous):
    def assess(checkpoint, observations):
        return Outcome("done", "camera_check", "within the caller's visual tolerance")

    _, execute, observe = recorder()
    if asynchronous:
        async def awaited(value):
            return value
        out = asyncio.run(async_place(lambda c: awaited(execute(c)), lambda: awaited(observe()), **PLACE,
                                      assess=lambda *a: awaited(assess(*a))))
    else:
        out = place(execute, observe, **PLACE, assess=assess)
    assert out.ok and out.data["assessment"] == "passed_caller_checks" and not out.data["completion_verified"]
    assert [a["checkpoint"] for a in out.data["assessments"]] == ["before_release", "after"]
    assert feedback(out)["observations"] == {"before": {"executed": 0}, "after": {"executed": 4}}


def test_async_cancellation_at_assessment_never_opens_or_withdraws():
    async def scenario():
        calls = []
        assessing = asyncio.Event()

        async def execute(command):
            calls.append(command)
            return Outcome("done", "fixture", "arrived")

        async def observe():
            return dict(executed=len(calls))

        async def assess(name, observations):
            assert name == "before_release" and observations[name]["executed"] == 1
            assessing.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(async_place(execute, observe, **PLACE, assess=assess))
        await asyncio.wait_for(assessing.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        assert calls == [PLACE["move"]]

    asyncio.run(scenario())


@pytest.fixture
def block(lifted):
    """The lifted kernel with its gripper at 60 mm around a 4 cm block on a pedestal."""
    assert lifted.run({"do": "gripper", "aperture_mm": 60}).ok
    supported_object(lifted, size=[.04, .04, .04])
    return lifted


def test_grasp_uses_actual_mujoco_feedback_on_a_second_adapter(block):
    k = block
    initial = k.body.world.boxes["block"].pose[2, 3]
    out = grasp(k.run, lambda: dict(opening=k.state.gripper, tool=k.tool[:3, 3].tolist()),
                approach={"do": "settle"}, close={"do": "grip", "start_mm": 60, "expect_mm": [35, 45]},
                check={"do": "check_grip", "expect_mm": [35, 45]},
                lift={"do": "seq", "expect_grip_mm": [35, 45],
                      "steps": [{"do": "line", "up": .02}, {"do": "settle"}]})
    assert out.ok, out.to_dict()
    assert k.body.world.boxes["block"].pose[2, 3] - initial > .015


def test_uncertain_release_keeps_actual_mujoco_grip_closed(block):
    k = block
    assert k.run({"do": "grip", "start_mm": 60, "expect_mm": [35, 45]}).ok
    out = place(k.run, lambda: dict(opening=k.state.gripper), move={"do": "settle"},
                release={"do": "gripper", "aperture_mm": 60},
                check={"do": "check_grip", "expect_mm": [55, 65]}, withdraw={"do": "line", "up": .02},
                assess=lambda *_: Outcome("stopped", "placement_check", "support cannot be established"))
    assert out.status == "stopped" and [s["stage"] for s in out.data["stages"]] == ["arrive"]
    assert k.manifest.gripper.aperture(k.state.gripper) < .05
