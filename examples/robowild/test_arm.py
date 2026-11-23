"""The hand tools on a fake mobile Franka whose move_robot results follow the harness's result schemas.

    uv run pytest examples/robowild/test_arm.py
"""

import json

import numpy as np
import pytest
from arm import HAND, NoTracking, RoboWildArm, quaternion, rotation

from world_use.hand_tools import DOWN, HandTools

DOWNWARD = quaternion(np.column_stack([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]))  # +Z down, fingers along -X


class RobotError(RuntimeError):
    def __init__(self, message, code):
        super().__init__(message)
        self.code, self.details = code, {}


class FakeFranka:
    """Goes where it is told, except that a block (centre and half sizes, robot_base) wider than the opening stops
    the fingertips on its top face, and fingers closing round it stop at its width. `refuse` names a planner
    failure category for every move."""

    def __init__(self, block=None, refuse=None, error=None, misses=()):
        self.at, self.q, self.fingers = np.array([0.5, 0.0, 0.4]), DOWNWARD, 0.04
        self.block, self.refuse, self.error, self.calls, self.sequence = block, refuse, error, [], 0
        self.misses = list(misses)  # how far each move in turn ends off its goal, as a controller lagging behind

    def scene(self):
        arm = dict(eef_position=self.at.tolist(), eef_quaternion_wxyz=list(self.q),
                   gripper_joint_names=["panda_finger_joint1", "panda_finger_joint2"],
                   gripper_positions=[self.fingers, self.fingers])
        return dict(sequence=self.sequence, images=[], robot_state=dict(reference_frame="robot_base",
                                                                        arms=dict(left=arm)))

    def move_robot(self, goals):
        (goal,) = goals
        self.calls.append(goal)
        if self.error:
            raise RobotError(f"{self.error}: no", self.error)
        if self.refuse:
            return dict(moved=False, executed_steps=0, failure=dict(category=self.refuse, message="no plan"),
                        move=dict(goals=[goal]))
        start = self.at.copy()
        entry = dict(target="left", start_position=start.tolist())
        if "position" in goal:
            target = np.asarray(goal["position"], float)
            reached = target.copy()
            if self.block is not None:
                centre, half = self.block
                inside = np.all(np.abs(target[:2] - centre[:2]) <= half[:2]) and 2 * self.fingers < 2 * half[1]
                top = centre[2] + half[2]
                if inside and target[2] < top <= start[2]:
                    # Blocked on the top face, and pushed a little aside as the controller keeps driving.
                    reached = start + (target - start) * (start[2] - top) / (start[2] - target[2]) + [0.025, 0, 0]
            reached = reached + (self.misses.pop(0) if self.misses else 0)
            self.at = reached
            error = float(np.linalg.norm(target - reached))
            entry.update(target_position=target.tolist(), reached_position=reached.tolist(), position_error_m=error,
                         tracked=error <= 0.02)
        if "quaternion_wxyz" in goal:
            self.q = goal["quaternion_wxyz"]
        if "gripper" in goal:
            held = self.block is not None and np.all(np.abs(self.at - self.block[0]) <= self.block[1])
            closed = self.block[1][1] if held else 0.0  # the block's half width per finger
            self.fingers = 0.04 * (1 - goal["gripper"]) + closed * goal["gripper"]
            if held and goal["gripper"]:
                self.carried = True
        if getattr(self, "carried", False) and self.fingers > 0 and self.block is not None:
            self.block = (self.at.copy(), self.block[1])
        self.sequence += 1
        scene = self.scene()
        return dict(moved=True, executed_steps=10, execution_status="completed", finished=False,
                    robot_state=scene["robot_state"], scene_state=scene, move=dict(goals=[entry]),
                    _harness=dict(robot_api_calls_used=len(self.calls), robot_api_call_limit=100))


def tools(robot):
    return HandTools(RoboWildArm(robot, robot.scene(), failures=(RobotError,)), HAND, NoTracking())


BLOCK = (np.array([0.55, 0.05, 0.05]), np.array([0.02, 0.02, 0.05]))   # a 4 cm block, 10 cm tall, top at z 0.10
GRASP = dict(object_target="block", approach=dict(x=0.55, y=0.05, z=0.07), lift=dict(z=0.11),
             expect_grip_mm=[35, 45])


@pytest.mark.parametrize("yaw_deg", range(-180, 180, 15))
def test_a_hand_turned_down_at_any_yaw_reaches_the_robot_as_that_rotation(yaw_deg):
    """Pointing down is a half turn, whose quaternion has w = 0: the case a sign rule gets wrong."""
    turn = np.radians(yaw_deg)
    jaws = np.array([np.sin(turn), np.cos(turn), 0.0])
    R = np.column_stack([np.cross(jaws, DOWN), jaws, DOWN])  # panda_tcp: fingers open along y, approach along z
    assert np.allclose(rotation(quaternion(R)), R)


def test_quaternions_round_trip_and_an_upright_hand_points_down():
    for q in np.random.default_rng(0).normal(size=(200, 4)):
        assert np.allclose(rotation(quaternion(rotation(q))), rotation(q))
    arm = RoboWildArm(FakeFranka(), FakeFranka().scene())
    targets, jaws = arm.orientation(DOWN, [1, 0, 0])
    approach = rotation(targets["quaternion_wxyz"]) @ [0, 0, 1]
    assert approach == pytest.approx(DOWN)
    assert abs(jaws @ [1, 0, 0]) == pytest.approx(1)


def test_a_grasp_opens_above_comes_down_closing_as_it_arrives_and_lifts_in_three_moves():
    robot = FakeFranka(block=BLOCK)
    record, text = tools(robot).call("grasp", GRASP)
    assert record["status"] == "done", text
    level, down, lift = robot.calls
    assert level["position"] == pytest.approx([0.55, 0.05, 0.4]) and level["gripper"] == 0
    assert down["position"] == pytest.approx([0.55, 0.05, 0.07]) and down["gripper"] == 1.0
    assert lift["position"][2] == pytest.approx(robot.calls[1]["position"][2] + 0.04)


def test_a_reading_an_empty_close_could_give_is_never_taken_for_a_hold():
    """3 mm is within the 1 mm slack of a 4-20 mm expectation, but empty fingers read as much."""
    robot = FakeFranka(block=(BLOCK[0], np.array([0.02, 0.0015, 0.05])))
    record, text = tools(robot).call("grasp", dict(GRASP, expect_grip_mm=[4, 20]))
    assert record["status"] == "surprise" and json.loads(text)["last_command"]["gripper"] == "reached_end", text


def test_an_approach_the_controller_ends_above_its_goal_says_how_far_above():
    """move_robot calls an approach within 2 cm arrived; fingers 8 mm above a thin part close over it."""
    robot = FakeFranka(block=BLOCK, misses=[np.zeros(3), np.array([0, 0, 0.008])])
    _, text = tools(robot).call("grasp", GRASP)
    assert json.loads(text)["approach_ended_short_m"] == pytest.approx(0.008, abs=1e-4), text
    _, text = tools(FakeFranka(block=BLOCK)).call("grasp", GRASP)
    assert "approach_ended_short_m" not in json.loads(text)


def test_a_tilt_turns_the_hand_about_the_pivot_in_steps_of_at_most_30_degrees():
    """A tube held at its mouth, 2 cm below the tool point, tipped to horizontal across the fingers: the mouth stays
    put. The fingers lie along x, so a lean toward +y is a pure tilt."""
    robot = FakeFranka()
    robot.fingers = 0.01  # a 20 mm grip
    arm = RoboWildArm(robot, robot.scene(), failures=(RobotError,))
    start = robot.at.copy()
    pivot = start + [0, 0, -0.02]
    toward = [pivot[0], pivot[1] + 1.0]
    told = arm.tilt(pivot.tolist(), toward, 90, [15, 25])
    assert told["status"] == "done" and told["moves"] == 3 and told["pivot_moved_m"] < 1e-6, told
    assert rotation(robot.q) @ [0, 0, 1] == pytest.approx([0, 1, 0], abs=1e-6)  # the fingertips point along +y
    for goal in robot.calls:  # every step keeps the tool point 2 cm from the pivot
        assert np.linalg.norm(np.asarray(goal["position"]) - pivot) == pytest.approx(0.02)
    told = arm.tilt(told["pivot"], toward, 0, [15, 25])
    assert told["moves"] == 3 and rotation(robot.q) @ [0, 0, 1] == pytest.approx([0, 0, -1], abs=1e-6)
    assert robot.at == pytest.approx(start)


def test_a_tilt_carries_the_pivot_over_its_target_first_and_refuses_a_pivot_off_the_hand():
    robot = FakeFranka()
    robot.fingers = 0.01
    arm = RoboWildArm(robot, robot.scene(), failures=(RobotError,))
    pivot = (robot.at + [0, 0, -0.02]).tolist()
    told = arm.tilt(pivot, [0.6, 1.1], 60, [15, 25], over=[0.6, 0.1, 0.5])
    assert told["moves"] == 2 and told["pivot"] == pytest.approx([0.6, 0.1, 0.5], abs=1e-6)
    with pytest.raises(ValueError, match=r"within 0\.3 m"):
        arm.tilt([2.0, 0.0, 0.4], [3.0, 0.0], 60, [15, 25])


def test_a_tools_result_ends_with_the_call_budget_its_moves_left():
    robot = FakeFranka(block=BLOCK)
    hand = tools(robot)
    _, text = hand.call("grasp", GRASP)
    assert json.loads(hand.arm.with_budget(text))["_harness"] == dict(robot_api_calls_used=3,
                                                                       robot_api_call_limit=100)


def test_a_grasp_can_open_the_fingers_only_as_wide_as_the_crowd_around_the_object_allows():
    robot = FakeFranka(block=BLOCK)
    record, text = tools(robot).call("grasp", dict(GRASP, open_mm=50))
    assert record["status"] == "done", text
    assert robot.calls[0]["gripper"] == pytest.approx(0.375)  # 50 of the Panda's 80 mm
    _, text = tools(FakeFranka(block=BLOCK)).call("grasp", dict(GRASP, open_mm=40))
    assert json.loads(text)["status"] == "refused"  # no wider than the 35-45 mm expected


def test_an_approach_that_meets_the_object_above_its_pose_reports_how_far_short():
    robot = FakeFranka(block=(BLOCK[0], np.array([0.06, 0.06, 0.07])))   # 12 cm wide, top 5 cm above the pose
    record, _ = tools(robot).call("grasp", GRASP)
    assert record["approach_ended_short_m"] == pytest.approx(0.05, abs=1e-3)


@pytest.mark.parametrize("misses,status", [([(0.03, 0, 0)], "done"), ([(0, 0, 0), (0.03, 0, 0)], "surprise")])
def test_a_move_on_the_way_may_end_off_its_goal_but_the_last_one_may_not(misses, status):
    """The fingers open above the block and come straight down: 3 cm off above it is corrected on the way down,
    3 cm aside at the bottom is not."""
    robot = FakeFranka(block=BLOCK, misses=[np.asarray(m, float) for m in misses])
    record, text = tools(robot).call("grasp", GRASP)
    assert {s["stage"]: s["status"] for s in record["stages"]}["approach"] == status, text


@pytest.mark.parametrize("block,gripper", [(None, "reached_end"),
                                           ((BLOCK[0], np.array([0.02, 0.003, 0.05])), "stopped_apart")])
def test_a_close_outside_the_expected_opening_says_whether_the_fingers_met_anything(block, gripper):
    """Empty fingers close; on a 6 mm part they stop apart, though narrower than the 35-45 mm expected."""
    robot = FakeFranka(block=block)
    record, text = tools(robot).call("grasp", GRASP)
    brief = json.loads(text)
    assert record["status"] == "surprise" and brief["last_command"]["gripper"] == gripper, text
    assert brief["stages"] == {"approach": "done", "close": "surprise"}  # the close, not the approach, failed
    assert len(robot.calls) == 2   # no lift after a failed check


@pytest.mark.parametrize("error,status", [("call_budget_exceeded", "stopped"), ("invalid_argument", "refused"),
                                          ("connection_closed", "stopped")])
def test_robot_errors_end_the_call_with_a_status_instead_of_an_exception(error, status):
    record, text = tools(FakeFranka(error=error)).call("grasp", GRASP)
    assert record["status"] == status, text
    assert (error == "connection_closed") == ("controller failed" in text)


def test_a_pose_the_planner_cannot_reach_is_refused_with_its_reason():
    record, text = tools(FakeFranka(refuse="out_of_reach")).call("grasp", GRASP)
    assert record["status"] == "refused" and "out_of_reach" in text


def test_a_release_checkpoint_is_void_once_the_arm_has_moved():
    robot = FakeFranka(block=BLOCK)
    hand = tools(robot)
    assert hand.call("grasp", GRASP)[0]["status"] == "done"
    place = dict(object_target="block", move=dict(x=0.5, y=-0.1, z=0.11), withdraw=dict(x=0.5, y=-0.1, z=0.25),
                 expect_grip_mm=[35, 45])
    record, text = hand.call("place", place)
    (decision,) = (a["outcome"] for a in record["assessments"])
    assert decision["kind"] == "placement_decision", text
    release = dict(checkpoint=decision["data"]["checkpoint"], note="on the table")
    hand.arm.robot.move_robot(goals=[dict(target="left", position=[0.5, -0.1, 0.12])])
    hand.arm.absorb(dict(scene_state=robot.scene()))
    assert hand.call("release_placement", release)[0]["status"] == "refused"
