"""The same camera-driven procedure faces changed geometry and physical parameters, without tuning."""
import asyncio
import json

import pytest
from conftest import FastClock, serving

from world_use.config import load_workcell
from world_use.examples.perception import procedure
from world_use.examples.pick_place import success
from world_use.mcp_server import build
from world_use.recorder import save_summary
from world_use.records import inspect


@pytest.mark.rendering
@pytest.mark.parametrize("center,mass,friction,eye,required_phase", [
    pytest.param([.30, .06, .20], .03, .6, [.30, -.08, .85], "placement", id="near-light"),
    # This heavier grasp settles during transport. Its lift must work, but final placement may miss the
    # unchanged 1 cm tolerance: the procedure must report that failure and still finish with torque off.
    pytest.param([.38, .10, .20], .10, .8, [.42, .06, 1.0], "lift", id="far-heavy"),
    pytest.param([.33, -.01, .20], .05, .4, [.34, 0, .90], "placement", id="lower-object-friction"),
    # This angle can move the lit top face outside the example's colour threshold. A clear abort is valid;
    # improved perception may also complete it, but must never claim a placement that truth does not confirm.
    pytest.param([.33, -.01, .20], .05, .4, [.28, .10, .95], None, id="angled-view"),
])
def test_unchanged_perception_procedure_in_varied_scenes(tmp_path, rehearser, center, mass, friction, eye,
                                                      required_phase):
    cell = load_workcell("block")
    cell["box"][1].update(center=center, known=False, mass_kg=mass, friction=friction)
    cell["camera"] = [dict(name="overhead", eye=eye, look_at=[.34, 0, .15],
                           size=[512, 512], fov_deg=40)]
    evaluated, result = {}, {}
    with serving(tmp_path, cell, rehearser, FastClock(100, 4)) as (d, c):
        c.release()
        assert "block" not in c.world()["boxes"]

        def evaluate():
            with d.k.lock, d.k.body.lock:
                evaluated.update(success(d.k, d.k.body.world))
            d.k.emit("task_result", "independent simulator evaluation", **evaluated)

        try:
            result = asyncio.run(procedure(build(c.url), evaluate))
        finally:
            save_summary(tmp_path / "result.json", dict(config=cell, result=result, evaluation=evaluated))
            # Keep even pictures the colour selector could not measure, so a failed observation is inspectable.
            for index, frame in enumerate(d.measurements.frames.values()):
                frame.image.save(tmp_path / f"camera-{index}.png")
        details = json.dumps(dict(result=result, evaluation=evaluated), indent=2)
        if required_phase is not None:
            assert result[required_phase] == "pass", details
        if result["placement"] == "pass":
            assert result["lift"] == "pass", details
            assert evaluated["success"], details
        else:
            assert result.get("reason"), details
        assert evaluated["released"] and evaluated["withdrawn"], details
        assert result["return_outcome"]["status"] == "done" and result["torque_off"], result
        status = c.status()
        assert not status["enabled"] and not status["faulted"] and not status["power_uncertain"]
    record = inspect(tmp_path / "run")
    assert record["closed"] and "recording_lost" not in record["summary"]
