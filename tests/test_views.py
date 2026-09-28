"""What a policy reads: the card, the world, the state line."""
import numpy as np
from conftest import make_kernel

from world_use import card, state_line
from world_use.views import world_text


def test_the_card_gives_positions_in_the_work_frame_moves_use():
    k = make_kernel()                                    # the real rest pose: the work frame is turned 22 deg
    assert abs(np.degrees(np.arctan2(*k.world.frame("work").T[1::-1, 0]))) > 20
    k.world.add_box("block", "object", center=[0.34, 0.03, 0.20], size=[0.04, 0.04, 0.10], frame="work")
    text = card(k)
    assert "object 'block': centre F+0.340 L+0.030 U+0.200" in text and "40 mm across the jaws" in text
    assert "block" in world_text(k)


def test_the_card_says_which_way_the_gripper_points_and_opens():
    text = card(make_kernel())
    assert "the gripper points forward, level; its jaws open left and right" in text


def test_the_card_states_the_turn_height_and_what_is_possible_from_here():
    text = card(make_kernel())
    assert "needs the tool at U+0.267 or higher" in text
    assert "from here a 3 cm line can go up, forward" in text and "turning needs the tool at U+0.267" in text


def test_the_card_says_when_the_robot_is_simulated_and_marks_hardware_notes():
    text = card(make_kernel())
    assert text.startswith("# reBot Arm B601-RS (simulated)") and "hardware: " in text


def test_no_heat_forecast_with_torque_off():
    k = make_kernel()
    k.heat.minutes_left = lambda limit: (2, 40.0, 5.0)   # as if the elbow were heating fast
    assert "min to 80C" in state_line(k)
    k.release()
    assert "min to 80C" not in state_line(k)
