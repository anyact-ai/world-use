"""Task predicates, not just command completion, decide whether placement worked."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from world_use import World, check
from world_use.examples.pick_place import TARGET, pickup, placement, run, setup, success
from world_use.geometry import axis_angle


@pytest.mark.parametrize("scenario,expected", [("nominal", True), ("shifted", True),
                                                ("missing", False), ("misplaced", False)])
def test_block_task_and_recovery(tmp_path, scenario, expected):
    result = run(tmp_path / scenario, scenario, video=False)
    assert result["success"] is expected
    assert result["torque_off"] and result["return_outcome"]["status"] == "done"
    if not expected:
        assert result["outcomes"][0]["status"] == "surprise"
        assert result["outcomes"][1]["status"] == "done"      # explicit open-and-retreat recovery


def test_a_done_pickup_is_not_a_successful_placement():
    k, truth = setup()
    k.enable()
    assert k.run(pickup(k)).ok
    assert truth.held is not None
    assert not success(k, truth)["success"]
    k.close()


@pytest.mark.parametrize("axis,degrees,expected", [((1, 0, 0), 0, True), ((1, 0, 0), 4, True),
                                                 ((1, 0, 0), 6, False), ((1, 0, 0), 180, False),
                                                 ((0, 0, 1), 90, False)])
def test_task_requires_the_intended_orientation_in_the_work_frame(axis, degrees, expected):
    truth = World()
    frame = np.eye(4)
    frame[:3, :3] = axis_angle((0, 0, 1), .4)
    frame[:3, 3] = [.1, .2, .3]
    truth.add_frame("work", frame)
    block = truth.add_box("block", "object", TARGET, [.04, .04, .10], frame="work")
    block.pose[:3, :3] = frame[:3, :3] @ axis_angle(axis, np.radians(degrees))
    tool = np.eye(4)
    tool[:3, 3] = truth.to_base("work", TARGET + [0, 0, .10])
    result = success(SimpleNamespace(tool=tool), truth)
    assert result["placed"] and result["released"] and result["withdrawn"]
    assert result["oriented"] is expected and result["success"] is expected
    assert result["orientation_error_deg"] == pytest.approx(degrees, abs=1e-6)


def test_adapting_the_block_workcell_and_destination():
    k, truth = setup()
    target = [.33, -.06, .20]
    try:
        for world in (k.world, truth):
            world.add_box("tray", "surface", [.32, 0, .14], [.28, .36, .02], frame="work")
            world.add_box("block", "object", [.35, .02, .20], [.04, .04, .10], frame="work")
        k.body.reset(k.state.q, k.state.gripper)
        k.enable()
        phase = pickup(k)
        assert check(phase, k).ok
        assert k.run(phase).ok
        phase = placement(k, target)
        assert check(phase, k).ok
        assert k.run(phase).ok
        result = success(k, truth, target)
        assert result["success"], result
        k.set_home_route([], "adapted open tray")
        assert k.run(k.home_plan()).ok
        k.release()
        assert not k.enabled
    finally:
        k.close()


def test_a_new_obstacle_blocks_the_return():
    k, _ = setup()
    k.enable()
    assert k.run({"do": "line", "up": .08}).ok
    k.set_home_route([])
    rest_tool = k.chain.fk(k.q_start)[:3, 3]
    k.world.add_box("obstacle", "keep_out", rest_tool, [.03, .03, .03], frame="base")
    before = k.cmd.q.copy()
    out = k.run(k.home_plan())
    assert out.status == "refused"
    assert np.allclose(k.cmd.q, before, atol=.03)
    assert k.enabled                       # a refused return is never described as torque-off
    k.close()


def test_wu_demo_prints_a_summary_into_a_fresh_record_folder(tmp_path, monkeypatch, capsys):
    from world_use import cli

    monkeypatch.setenv("WORLD_USE_RUNS", str(tmp_path))
    (tmp_path / "block-demo").mkdir()
    (tmp_path / "block-demo" / "earlier run").write_text("")
    assert cli.main(["demo", "--no-video"]) == 0
    out = capsys.readouterr().out
    assert len(out.splitlines()) <= 3 and str(tmp_path / "block-demo-2") in out
    assert (tmp_path / "block-demo-2" / "result.json").exists()
