import asyncio

import pytest

from world_use.behaviors import Outcome
from world_use.procedures import async_place, feedback, grasp, place


def test_bound_place_requires_both_regions_before_any_motion():
    calls = []

    def execute(command):
        calls.append(command)
        return Outcome("done", "fixture", "command completed")

    observation = {"targets": [{"target": "cup", "valid": True},
                               {"target": "shelf", "valid": False, "tracking": "lost"}]}
    result = place(execute, lambda: observation, move={}, release={}, check={}, withdraw={},
                   object_target="cup", destination_target="shelf")
    assert not result.ok and not calls
    assert result.data["intent"] == {"object": "cup", "destination": "shelf"}
    assert result.data["assessment"] == "unknown" and "reselect" in result.hint

    observation["targets"][1].update(valid=True, tracking="tracked")
    result = place(execute, lambda: observation, move={}, release={}, check={}, withdraw={},
                   object_target="cup", destination_target="shelf")
    assert result.ok and len(calls) == 4
    # Selected masks, successful commands and open fingers cannot certify that
    # the policy named the right cup or that it rests on the shelf.
    assert result.data["assessment"] == "unknown" and not result.data["completion_verified"]


@pytest.mark.parametrize("procedure,failed_at", [(grasp, i) for i in range(5)] + [(place, i) for i in range(4)])
def test_failure_cancels_remaining_manipulation_and_preserves_evidence(procedure, failed_at):
    calls, pictures = [], []

    def execute(command):
        calls.append(command)
        if len(calls) == failed_at + 1:
            return Outcome("surprise", "fixture", "measured mismatch", expected=20, observed=4,
                           hint="inspect before deciding recovery")
        return Outcome("done", "fixture", "measured completion")

    def observe():
        pictures.append(len(calls))
        return dict(frame=str(len(pictures)), executed=len(calls))

    args = (dict(approach={"approach": 1}, close={"close": 1}, check={"check": 1}, lift={"lift": 1})
            if procedure is grasp else
            dict(move={"move": 1}, release={"release": 1}, withdraw={"withdraw": 1}, check={"check": 1}))
    result = procedure(execute, observe, **args)
    assert len(calls) == failed_at + 1 and not result.ok
    assert result.expected == 20 and result.observed == 4
    assert result.data["observations"]["after"]["executed"] == failed_at + 1
    assert not result.data["completion_verified"]
    compact = feedback(result)
    assert compact["expected"] == 20 and compact["observed"] == 4
    assert compact["observations"]["after"]["executed"] == failed_at + 1
    assert "inspect" in compact["hint"]
    assert all(set(stage) == {"stage", "status"} for stage in compact["stages"])


def test_grasp_observes_the_lift_and_never_claims_retention():
    calls = []

    def execute(command):
        calls.append(command)
        return Outcome("done", "fixture", "checked")

    out = grasp(execute, lambda: dict(executed=len(calls)), approach={"approach": 1}, close={"close": 1},
                check={"check": 1}, lift={"lift": 1})
    assert calls == [{"approach": 1}, {"close": 1}, {"check": 1}, {"lift": 1}, {"check": 1}]
    assert out.data["observations"]["before_lift"] == {"executed": 3}
    assert out.data["observations"]["after"] == {"executed": 5}
    assert out.ok and not out.data["completion_verified"]


def test_native_termination_and_observation_failure_cannot_continue():
    calls = []

    def execute(command):
        calls.append(command)
        raise RuntimeError("native episode ended")

    with pytest.raises(RuntimeError, match="native episode ended"):
        place(execute, lambda: {}, move={}, release={}, withdraw={}, check={})
    assert len(calls) == 1
    calls.clear()

    def unavailable():
        raise RuntimeError("camera unavailable")

    with pytest.raises(RuntimeError, match="camera unavailable"):
        grasp(execute, unavailable, approach={}, close={}, check={}, lift={})
    assert calls == []


def test_grasp_uses_actual_mujoco_feedback_on_a_second_adapter(lifted):
    from conftest import supported_object

    k = lifted
    assert k.run({"do": "gripper", "aperture_mm": 60}).ok
    supported_object(k, size=[.04, .04, .04])
    initial = k.body.world.boxes["block"].pose[:3, 3].copy()
    out = grasp(k.run, lambda: dict(opening=k.state.gripper, tool=k.tool[:3, 3].tolist()),
                approach={"do": "settle"}, close={"do": "grip", "start_mm": 60, "expect_mm": [35, 45]},
                check={"do": "check_grip", "expect_mm": [35, 45]},
                lift={"do": "seq", "expect_grip_mm": [35, 45],
                      "steps": [{"do": "line", "up": .02}, {"do": "settle"}]})
    assert out.ok, out.to_dict()
    assert k.body.world.boxes["block"].pose[2, 3] - initial[2] > .015
    assert not out.data["completion_verified"]


@pytest.mark.parametrize("checkpoint,commands", [("before_close", 1), ("before_lift", 3), ("after", 5)])
def test_grasp_assessment_stops_at_the_decision_point(checkpoint, commands):
    calls = []

    def execute(command):
        calls.append(command)
        return Outcome("done", "fixture", "arrived")

    def assess(name, observations):
        assert observations[name]["executed"] == len(calls)
        if name == checkpoint:
            return Outcome("stopped", "visual_check", "identity or retention is uncertain",
                           expected="the selected object follows the gripper", observed="occluded",
                           hint="inspect from another view before deciding the next action")
        return Outcome("done", "visual_check", "caller checked this stage")

    out = grasp(execute, lambda: dict(executed=len(calls)), approach={}, close={}, check={}, lift={}, assess=assess)
    assert out.status == "stopped" and len(calls) == commands
    report = feedback(out)
    assert report["checkpoint"] == checkpoint and report["observed"] == "occluded"
    assert report["next_stage"] == {"before_close": "close", "before_lift": "test_lift", "after": None}[checkpoint]
    assert not report["completion_verified"]


def test_place_rechecks_bound_targets_after_arrival_before_opening():
    calls = []

    def execute(command):
        calls.append(command)
        return Outcome("done", "fixture", "arrived")

    def observe():
        return dict(targets=[dict(target="cup", valid=True), dict(target="shelf", valid=not calls)])

    out = place(execute, observe, move={}, release={}, check={}, withdraw={},
                object_target="cup", destination_target="shelf")
    assert out.status == "stopped" and len(calls) == 1
    assert out.data["checkpoint"] == "before_release" and out.data["next_stage"] == "release"
    assert "shelf" in out.message


def test_place_assessor_sees_new_geometry_and_cannot_rewrite_recorded_evidence():
    calls = []
    observation = dict(error_m=.001)

    def execute(command):
        calls.append(command)
        observation["error_m"] = .04  # grasp changed during transport despite successful arm arrival
        return Outcome("done", "fixture", "arrived")

    def assess(name, observations):
        assert name == "before_release"
        assert observations["before"]["error_m"] == .001
        measured = observations[name]["error_m"]
        observations[name]["error_m"] = 0  # callback mutation cannot change the evidence record
        return Outcome("surprise", "placement", "object missed its destination", expected=.005, observed=measured)

    out = place(execute, lambda: observation, move={}, release={}, check={}, withdraw={}, assess=assess)
    assert len(calls) == 1 and out.observed == .04
    assert out.data["observations"]["before_release"]["error_m"] == .04


def test_assessor_failure_never_releases_or_withdraws():
    calls = []

    def execute(command):
        calls.append(command)
        return Outcome("done", "fixture", "arrived")

    def invalid(name, observations):
        return None

    with pytest.raises(TypeError, match="Outcome"):
        place(execute, lambda: {}, move={}, release={}, check={}, withdraw={}, assess=invalid)
    assert len(calls) == 1


def test_success_reports_caller_checks_without_certifying_physical_support():
    out = place(lambda _: Outcome("done", "fixture", "executed"), lambda: {},
                move={}, release={}, check={}, withdraw={},
                assess=lambda *_: Outcome("done", "camera_check", "within the caller's visual tolerance"))
    assert out.ok and out.data["assessment"] == "passed_caller_checks"
    assert not out.data["completion_verified"]
    assert [a["checkpoint"] for a in out.data["assessments"]] == ["before_release", "after"]


def test_uncertain_release_keeps_actual_mujoco_grip_closed(lifted):
    from conftest import supported_object

    k = lifted
    assert k.run({"do": "gripper", "aperture_mm": 60}).ok
    supported_object(k, size=[.04, .04, .04])
    assert k.run({"do": "grip", "start_mm": 60, "expect_mm": [35, 45]}).ok
    out = place(k.run, lambda: dict(opening=k.state.gripper), move={"do": "settle"},
                release={"do": "gripper", "aperture_mm": 60},
                check={"do": "check_grip", "expect_mm": [55, 65]}, withdraw={"do": "line", "up": .02},
                assess=lambda *_: Outcome("stopped", "placement_check", "support cannot be established"))
    assert out.status == "stopped" and out.data["next_stage"] == "release"
    assert [s["stage"] for s in out.data["stages"]] == ["arrive"]
    assert k.manifest.gripper.aperture(k.state.gripper) < .05


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

        task = asyncio.create_task(async_place(execute, observe, move={"move": 1}, release={"open": 1},
                                              check={}, withdraw={}, assess=assess))
        await asyncio.wait_for(assessing.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        assert calls == [{"move": 1}]

    asyncio.run(scenario())


def test_async_place_applies_the_same_pre_release_and_final_checks():
    async def execute(command):
        return Outcome("done", "fixture", "executed")

    async def observe():
        return {}

    async def assess(name, observations):
        return Outcome("done", "camera_check", "observed", observed=list(observations))

    out = asyncio.run(async_place(execute, observe, move={}, release={}, check={}, withdraw={}, assess=assess))
    assert out.ok and out.data["assessment"] == "passed_caller_checks"
    assert len(feedback(out)["observations"]) == 2
    assert [a["checkpoint"] for a in out.data["assessments"]] == ["before_release", "after"]
