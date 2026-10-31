import pytest

from world_use.behaviors import Outcome
from world_use.procedures import feedback, grasp, place


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
