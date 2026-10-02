"""What a policy reads: the card, the world, the state line."""
import numpy as np
from conftest import make_kernel

from world_use import card, state_line
from world_use.daemon import apply_workcell
from world_use.views import status, world_text


def test_the_card_gives_positions_in_the_work_frame_moves_use():
    k = make_kernel()                                    # the real rest pose: the work frame is turned 22 deg
    assert abs(np.degrees(np.arctan2(*k.world.frame("work").T[1::-1, 0]))) > 20
    k.world.add_box("block", "object", center=[0.34, 0.03, 0.20], size=[0.04, 0.04, 0.10], frame="work")
    text = card(k)
    assert "object 'block': centre F+0.340 L+0.030 U+0.200" in text and "40 mm across the jaws" in text
    assert "block" in world_text(k)


def test_the_card_describes_the_resolved_workcell_frame():
    k = make_kernel()
    apply_workcell({"frame": [dict(name="work", origin=[.01, .02, .03], rpy_deg=[0, 0, 90])]}, k)
    text = card(k, reach=lambda _: "")
    assert "origin in base F+0.010 L+0.020 U+0.030 m" in text and "from workcell" in text
    assert "axes in base: forward points left, level; left points back, level; up points straight up" in text


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
    assert "heat" not in status(k)


def test_lost_feedback_is_cached_not_a_claim_of_holding(k, monkeypatch):
    import pytest

    def lost():
        raise ConnectionError("USB unplugged")

    k.heat.minutes_left = lambda limit: (2, 88.0, -1.0)
    monkeypatch.setattr(k.body, "read", lost)
    with pytest.raises(ConnectionError):
        k.tick()
    for _ in range(500):
        k.clock.wait()
    s = status(k)
    assert s["power_uncertain"] and s["feedback"]["stale"]
    assert s["feedback"]["age_s"] == 5.0 and "USB unplugged" in s["feedback"]["error"]
    assert "motor power unconfirmed" in s["line"] and "STALE feedback" in s["line"]
    assert "holding" not in s["line"] and "heat" not in s and "min to" not in s["line"]


def test_a_stopped_loop_does_not_keep_telemetry_fresh(k):
    for _ in range(200):
        k.clock.wait()
    assert status(k)["feedback"]["stale"]
    assert "STALE feedback" in state_line(k)


def test_a_view_drawn_on_a_picture_of_another_shape_keeps_square_pixels():
    """A 1920x1080 webcam described with the default 800x600 came out with fx != fy: boxes drawn squashed."""
    from world_use.cameras import View
    eye, at = [0.9, -0.4, 0.5], [0.3, 0.0, 0.1]
    moved = View.look_at(eye, at, 70.0, (800, 600)).scaled(1024, 576)
    native = View.look_at(eye, at, 70.0, (1024, 576))
    pts = np.array([[0.3, 0.0, 0.1], [0.25, 0.1, 0.0], [0.4, -0.1, 0.2]])
    assert moved.fx == moved.fy and np.allclose(moved.project(pts)[0], native.project(pts)[0], atol=1e-6)


def test_a_file_camera_serves_the_newest_frame_and_refuses_a_stale_one(tmp_path):
    """A capture app that died must not hand the policy an old picture as if it were now."""
    import os
    import time

    import pytest
    from PIL import Image

    from world_use import World
    from world_use.cameras import from_config
    frame = tmp_path / "side.jpg"
    Image.new("RGB", (64, 32), (200, 30, 30)).save(frame)
    cam = from_config({"name": "side", "path": str(frame), "max_age_s": 2, "rotate": 90}, World())
    assert cam.picture(None).size == (32, 64)                  # turned a quarter clockwise
    old = time.time() - 60
    os.utime(frame, (old, old))
    with pytest.raises(RuntimeError, match="60 s old"):
        cam.picture(None)
    frame.unlink()
    with pytest.raises(RuntimeError, match="no frame at"):
        cam.picture(None)
    with pytest.raises(ValueError, match="rotate"):
        from_config({"name": "side", "path": str(frame), "rotate": 45}, World())
