"""A scene change interrupts a measured grasp; a new attempt must observe it and finish the physical task."""
import asyncio
import json

import numpy as np
import pytest
from conftest import FastClock, serving

from world_use.bodies.mujoco_scene import mj
from world_use.config import load_workcell
from world_use.examples.perception import procedure
from world_use.examples.pick_place import success
from world_use.mcp_server import build
from world_use.recorder import save_summary
from world_use.records import events, inspect


@pytest.mark.rendering
@pytest.mark.parametrize("destination", [[.33, .09, .20], [.35, .08, .20]])
def test_changed_scene_refuses_old_grasp_then_reobserves_and_recovers(tmp_path, rehearser, destination):
    cell = load_workcell("block")
    cell["box"][1]["known"] = False
    cell["camera"] = [dict(name="overhead", eye=[.34, 0, .90], look_at=[.34, 0, .15],
                           size=[512, 512], fov_deg=40)]
    evaluations, attempts = [], []
    with serving(tmp_path, cell, rehearser, FastClock(100, 4)) as (d, c):
        c.release()
        k, truth = d.k, d.k.body.world
        assert "block" not in c.world()["boxes"]
        server = build(c.url)
        interrupted = False

        def evaluate():
            with k.lock, k.body.lock:
                result = success(k, truth)
            evaluations.append(result)
            k.emit("task_result", "independent simulator evaluation", **result)

        class ChangedScene:
            """Test runner only: the unmodified procedure still receives just MCP replies and images."""

            async def call_tool(self, name, arguments):
                nonlocal interrupted
                plan = arguments.get("plan")
                if (interrupted or name != "run" or not isinstance(plan, list)
                        or not any(step.get("do") == "grip" for step in plan)):
                    return await server.call_tool(name, arguments)
                interrupted = True
                # Pause an accepted, rehearsed phase before descent, as a procedure might await an operator.
                paused = await server.call_tool("run", dict(arguments, plan=[
                    dict(do="checkpoint", ask="continue with the measured grasp?"), *plan]))
                job = paused.structured_content
                assert job["status"] == "waiting", job
                before = c.status()["tool"]["work"]
                evidence = [item["evidence"] for item in arguments["requires"]]
                with k.lock, k.body.lock:
                    estimated = k.world.boxes["block"].pose.copy()
                    joint = k.body.model.joint("free/block")
                    q, v = int(joint.qposadr[0]), int(joint.dofadr[0])
                    k.body.data.qpos[q:q + 3] = truth.to_base("work", destination)
                    k.body.data.qvel[v:v + 6] = 0
                    mj.mj_forward(k.body.model, k.body.data)
                    np.testing.assert_array_equal(k.world.boxes["block"].pose, estimated)
                # Explicit tracker/operator invalidation; freshness alone cannot detect an unseen scene change.
                c.withdraw(evidence, "test intervention: block moved during the checkpoint")
                c.record(note="Runner moved the physical block; the procedure receives no new pose.")
                refused = await server.call_tool("answer", dict(job=job["id"], answer="yes", wait_s=30))
                outcome = refused.structured_content
                assert outcome["status"] == "refused", outcome
                assert outcome["outcome"]["data"]["rule"] == "stale_measurement"
                after = c.status()
                assert np.linalg.norm(np.subtract(after["tool"]["work"], before)) < .002
                assert after["gripper"]["aperture_mm"] > 60
                return refused

        async def exercise():
            attempts.append(await procedure(ChangedScene(), evaluate))
            assert interrupted and attempts[0]["torque_off"]
            assert attempts[0]["return_outcome"]["status"] == "done"
            assert "reason" in attempts[0] and not evaluations[0]["success"]
            assert not c.status()["faulted"]
            # Keep the same physics, world estimate, measurement registry and daemon. No reset between attempts.
            attempts.append(await procedure(server, evaluate))

        try:
            asyncio.run(exercise())
        finally:
            save_summary(tmp_path / "result.json", dict(destination=destination, attempts=attempts,
                                                        evaluations=evaluations))
        result = attempts[1]
        details = json.dumps(dict(result=result, evaluation=evaluations[1]), indent=2)
        assert result["lift"] == "pass", details
        assert evaluations[1]["success"], details
        # Recovery must physically succeed; a conservative camera rejection is not a physical failure.
        assert result["placement"] in ("pass", "fail"), details
        if result["placement"] == "fail":
            assert result.get("reason"), details
        assert result["torque_off"] and result["return_outcome"]["status"] == "done"
        measured = result["measurements"][0]
        assert measured["surface_center"] == pytest.approx(np.array(destination) + [0, 0, .05], abs=.004)
        assert measured["id"] not in {m["id"] for m in attempts[0]["measurements"]}
        assert not c.status()["power_uncertain"]

    record = inspect(tmp_path / "run")
    assert record["closed"] and "recording_lost" not in record["summary"]
    initial = json.loads((tmp_path / "run/session.json").read_text())["initial"]["world"]
    assert "block" not in initial["boxes"]
    log = events(tmp_path / "run")
    assert any(e["kind"] == "withdrawn" for e in log)
    assert [e["data"]["success"] for e in log if e["kind"] == "task_result"] == [False, True]
    for attempt in attempts:
        for m in attempt["measurements"]:
            assert all((tmp_path / "run/perception" / f"{m['id']}{ext}").is_file() for ext in (".png", ".npz"))
