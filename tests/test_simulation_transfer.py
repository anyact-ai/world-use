"""Two friction-held objects cross a physical divider, with unsafe shortcuts refused before execution."""
import numpy as np

from world_use import check
from world_use.examples.pick_place import setup
from world_use.recorder import save_summary
from world_use.records import inspect


def test_two_blocks_cross_a_divider_and_remain_placed(tmp_path):
    k, truth = setup(output=tmp_path)
    # Fill the far slot first so the housing does not descend onto an already placed near block.
    targets = {"far": [.38, -.20, .20], "near": [.30, -.20, .20]}
    starts = {"far": [.38, .06, .20], "near": [.30, .20, .20]}
    outcomes, evaluated = [], {}
    try:
        for world in (k.world, truth):
            world.boxes.pop("block")
            world.add_box("tray", "surface", [.32, 0, .14], [.30, .50, .02], frame="work")
            world.add_box("divider", "surface", [.37, -.04, .21], [.14, .015, .12], frame="work")
            world.add_box("divider clearance", "keep_out", [.37, -.04, .21], [.14, .015, .12], frame="work")
            for name, source in starts.items():
                world.add_box(name, "object", source, [.04, .04, .10], frame="work")
        k.body.reset(k.state.q, k.state.gripper)
        k.record_session(task="two blocks across a divider", policy="scripted known-scene procedure",
                         simulation_truth=truth.to_dict())
        k.enable()

        def execute(plan):
            report = check(plan, k)
            assert report.ok, str(report)
            out = k.run(plan)
            outcomes.append(out.to_dict())
            assert out.ok, out

        execute([dict(do="line", up=.14), dict(do="gripper", aperture_mm=65),
                 dict(do="move_to", point="forward", jaws="left", within_deg=0)])
        for name, target in targets.items():
            # The procedure uses the declared scene; only the evaluator below reads physical truth.
            source = k.world.from_base("work", k.world.boxes[name].pose[:3, 3])
            grasp = source + [.02, 0, .02]
            execute([dict(do="move_to", to=[grasp[0], -.04, .36]),
                     dict(do="move_to", to=[*grasp[:2], .36]),
                     dict(do="move_to", to=grasp.tolist()), dict(do="grip", expect_mm=[35, 45])])
            assert truth.held is not None and truth.held[0] == name
            tool = k.world.from_base("work", k.tool[:3, 3])
            block = k.world.from_base("work", k.world.boxes[name].pose[:3, 3])
            to = np.asarray(target) + tool - block
            before = k.cmd.q.copy()
            shortcut = check([dict(do="move_to", to=[to[0], -.04, to[2]]),
                              dict(do="move_to", to=to.tolist())], k)
            assert not shortcut.ok and any(r["rule"] == "keep_out" for r in shortcut.problems)
            np.testing.assert_array_equal(k.cmd.q, before)
            # Clear the divider with the bottom of the payload too; the kernel does not model payload collision.
            execute([dict(do="move_to", to=[tool[0], tool[1], .36]),
                     dict(do="move_to", to=[to[0], -.04, .36]),
                     dict(do="move_to", to=[to[0], to[1], .36]),
                     dict(do="move_to", to=to.tolist()), dict(do="gripper", aperture_mm=65),
                     dict(do="line", up=.14)])

        for name, target in targets.items():
            box = truth.boxes[name]
            center = truth.from_base("work", box.pose[:3, 3])
            rotation = truth.frame("work").T[:3, :3].T @ box.pose[:3, :3]
            error = float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
            evaluated[name] = dict(position_error_m=float(np.linalg.norm(center - target)),
                                   orientation_error_deg=error, center=center.tolist())
            assert evaluated[name]["position_error_m"] < .01, evaluated
            assert error < 5, evaluated
        assert truth.held is None
        assert truth.from_base("work", k.tool[:3, 3])[2] > .30
        k.emit("task_result", "both blocks independently verified", objects=evaluated)
        k.set_home_route([dict(do="move_to", to=[.32, .02, .36])], "above the divider, then back to the clear side")
        execute(k.home_plan())
        k.release()
        assert not k.enabled and not k.power_uncertain
    finally:
        summary = k.close()
        save_summary(tmp_path / "result.json", dict(objects=evaluated, outcomes=outcomes, recording=summary))
    record = inspect(tmp_path)
    assert record["closed"] and "recording_lost" not in record["summary"]
