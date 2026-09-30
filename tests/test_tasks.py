"""Task predicates, not just command completion, decide whether placement worked."""
from __future__ import annotations

import numpy as np
import pytest

from world_use.examples.pick_place import pickup, run, setup, success


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
