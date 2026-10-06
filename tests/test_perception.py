"""Measurements describe the daemon's own pixels, and a run that requires one stops before any step once it is stale."""
import json
import tempfile
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from world_use import Refused
from world_use.behaviors import Behavior, Joints
from world_use.cameras import Camera, Frame, SimCamera, View, pack, unpack
from world_use.client import DaemonError
from world_use.perception import Measurements, measure, rectangle
from world_use.plan import snapshot


def source(**kwargs):
    defaults = dict(view=View(np.eye(4), 50, 50, 30, 20, 60, 40), depth=np.full((40, 60), .5))
    return Frame(Image.new("RGB", (60, 40), "orange"), "test", **(defaults | kwargs))


def held(cameras):
    """A frame from a calibrated test camera, with depth and a tool pose, as the daemon keeps it."""
    camera = cameras.setdefault("test", Camera("test", View(np.eye(4), 50, 50, 30, 20, 60, 40)))
    tool = np.eye(4)
    tool[:3, 3] = [.02, 0, .45]
    return source(view=camera.view, calibration=camera.calibration_id, tool=tool)


@pytest.fixture
def required(k, monkeypatch, tmp_path):
    """A guard requiring a 5 s fresh measurement, and the clock its age is read from."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    registry = Measurements(k, k.cameras)
    frame = registry.keep(held(k.cameras))
    m = registry.measure(frame.id, point=[30, 20], target="block")
    now = [frame.timestamp + 1]
    registry.now = lambda: now[0]
    return registry.guard([dict(evidence=m["id"], max_age_s=5)]), now


def advance(k, predicate, limit=300):
    for _ in range(limit):
        k.tick()
        k.clock.wait()
        if predicate():
            return
    pytest.fail("kernel did not reach the expected state")


def test_rgbd_roundtrip_and_pixel_centres():
    frame = source(calibration="revision", tool=np.eye(4))
    decoded = Frame.from_dict(frame.to_dict())
    assert decoded.calibration == "revision"
    np.testing.assert_array_equal(decoded.tool, np.eye(4))
    np.testing.assert_array_equal(decoded.depth, frame.depth)
    result = measure(decoded, point=[30, 20])
    np.testing.assert_allclose(result.points, [[.005, .005, .5]])
    assert not decoded.depth.flags.writeable and not decoded.view.T.flags.writeable
    with pytest.raises(ValueError, match="array"):
        unpack(pack(b"x" * 10000), 5)


def test_regions_reject_missing_mixed_or_insufficient_depth():
    frame = source()
    mask = np.ones(frame.depth.shape, dtype=bool)
    measured = measure(frame, mask=mask)
    assert measured.valid and len(measured.points) <= 2048
    assert measure(replace(frame, depth=None), mask=mask).reason == "missing_depth"
    assert measure(replace(frame, view=None), mask=mask).reason == "missing_calibration"
    depth = frame.depth.copy()
    depth[:, :30] = .8
    assert measure(replace(frame, depth=depth), mask=mask).reason == "mixed_depth_surfaces"
    depth[:, :] = np.nan
    assert measure(replace(frame, depth=depth), mask=mask).reason == "invalid_depth"
    assert measure(frame, mask=np.zeros_like(mask)).reason == "insufficient_support"
    assert measure(frame, point=[30, 20]).summary(np.eye(4))["in_tool"] is None
    assert measure(replace(frame, depth=None, tool=np.eye(4)), point=[30, 20]).summary(np.eye(4))["in_tool"] is None


def test_projection_uses_measured_plane_instead_of_background_depth():
    # A tilted surface and a translated, rotated camera; no world-Z assumption.
    camera = np.eye(4)
    camera[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    camera[:3, 3] = [.3, -.2, .1]
    view = View(camera, 50, 50, 30, 20, 60, 40)
    y, x = np.indices((40, 60))
    depth = .5 / (1 + .2 * (x + .5 - 30) / 50 - .1 * (y + .5 - 20) / 50)
    expected = view.unproject([[42.5, 22.5]], [depth[22, 42]])
    depth[22, 42] = .8                         # The edge pixel sees a background surface.
    frame = source(view=view, depth=depth)
    plane = dict(box=[8, 8, 27, 32], max_error_m=.0001)
    raw = measure(frame, point=[42, 22])
    projected = measure(frame, point=[42, 22], plane=plane)
    assert np.linalg.norm(raw.points - expected) > .2
    assert projected.valid
    np.testing.assert_allclose(projected.points, expected, atol=1e-7)
    assert projected.frame is frame and projected.diagnostics["method"] == "plane_projection"
    assert projected.support_points is not None and not projected.support_points.flags.writeable
    # The selected feature's own depth is not needed, but the support patch is.
    depth[22, 42] = np.nan
    np.testing.assert_allclose(measure(replace(frame, depth=depth), point=[42, 22], plane=plane).points,
                               expected, atol=1e-7)


def test_plane_projection_rejects_bad_support_and_unstable_rays():
    plane = dict(box=[3, 3, 27, 35], max_error_m=.0001)
    frame = source()
    assert measure(replace(frame, depth=None), point=[35, 20], plane=plane).reason == "missing_depth"
    depth = frame.depth.copy()
    depth[10:25, 10:20] += .02                 # Two nearby surfaces used to pass the 5 cm gap check.
    assert measure(replace(frame, depth=depth), point=[35, 20], plane=plane).reason == "nonplanar_support"
    tiny = dict(plane, box=[0, 0, 3, 3])
    assert measure(frame, point=[35, 20], plane=tiny).reason == "insufficient_support"
    depth[:] = np.nan
    depth[4:34, 15] = .5                      # A line cannot establish a plane.
    assert measure(replace(frame, depth=depth), point=[35, 20], plane=plane).reason == "invalid_depth"
    narrow = dict(plane, box=[14, 2, 17, 37])
    assert measure(replace(frame, depth=depth), point=[35, 20], plane=narrow).reason == "degenerate_plane_support"
    view = replace(frame.view, cx=30.5)
    _, x = np.indices(frame.depth.shape)
    depth = np.full_like(frame.depth, np.nan)
    depth[:, 3:27] = -.1 / ((x[:, 3:27] + .5 - view.cx) / view.fx)  # Plane at camera X=-.1.
    vertical = replace(frame, view=view, depth=depth)
    side = dict(plane, box=[3, 3, 12, 35])
    assert measure(vertical, point=[30, 20], plane=side).reason == "grazing_plane_ray"
    assert measure(vertical, point=[45, 20], plane=side).reason == "plane_behind_camera"
    with pytest.raises(ValueError, match="requires a point"):
        measure(frame, mask=rectangle([5, 5, 25, 25], frame.image.size), plane=plane)


def test_tool_coordinates_distinguish_rotation_from_landmark_slip():
    tool = np.eye(4)
    tool[:3, 3] = [-.015, .035, .55]
    first = measure(source(tool=tool), point=[30, 20]).summary(np.eye(4))
    assert first["in_tool"] == pytest.approx([.02, -.03, -.05])
    # The same feature after a 90-degree tool turn and translation: [0.16, -0.08, 0.2] in base.
    tool[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    tool[:3, 3] = [.13, -.10, .25]
    camera = np.eye(4)
    camera[:3, 3] = [.155, -.085, -.30]
    frame = source(tool=tool, view=View(camera, 50, 50, 30, 20, 60, 40))
    work = np.eye(4)                         # An independently rotated and translated work frame.
    work[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    work[:3, 3] = [-.1, .2, .35]
    second = measure(frame, point=[30, 20]).summary(work)
    assert second["in_tool"] == pytest.approx(first["in_tool"])
    assert second["from_tool"] != first["from_tool"]
    assert second["surface_center"] != first["surface_center"]
    # An 8 mm shift along base X is -8 mm along the turned tool's Y axis.
    camera[0, 3] += .008
    slipped = measure(replace(frame, view=View(camera, 50, 50, 30, 20, 60, 40)), point=[30, 20]).summary(work)
    assert slipped["in_tool"] == pytest.approx([.02, -.038, -.05])


@pytest.mark.rendering
def test_mujoco_depth_unprojects_to_the_actual_surface(k):
    k.body.world.add_box("plane", "surface", [.6, 0, .05], [.5, .5, .1], frame="base")
    lens = View.look_at([.6, 0, 1], [.6, 0, 0], size=(160, 120))
    lens = replace(lens, fx=135, fy=105, cx=67, cy=51)
    frame = SimCamera("depth", lens, k.body).capture(k, depth=True)
    result = measure(frame, point=[lens.cx, lens.cy])
    assert result.valid
    assert result.points[0, 2] == pytest.approx(.1, abs=.001)
    np.testing.assert_allclose(frame.tool, k.tool)
    # Changing the estimated camera pose must change measurements, not simulator truth.
    wrong = lens.T.copy()
    wrong[0, 3] += .1
    camera = SimCamera("wrong", lens, k.body)
    camera.view = replace(lens, T=wrong)
    other = measure(camera.capture(k, depth=True), point=[lens.cx, lens.cy])
    assert other.points[0, 0] - result.points[0, 0] == pytest.approx(.1, abs=1e-6)


def test_the_daemon_measures_its_own_frame_in_the_work_frame_and_records_it(daemon):
    d, c = daemon
    frame = d.measurements.keep(held(d.cameras))
    seen = d.k.events.seq
    m = c.measure(frame.id, point=[30, 20], target="block")
    work = d.k.world
    center, tool = work.from_base("work", [.005, .005, .5]), work.from_base("work", [.02, 0, .45])
    assert m["valid"] and m["surface_center"] == pytest.approx(center, abs=1e-4)
    assert m["from_tool"] == pytest.approx(center - tool, abs=1e-4)
    assert Image.open(m["image"]).size == frame.image.size
    events = [e for e in d.k.events.since(seen) if e["kind"] == "measurement"]
    assert len(events) == 1 and events[0]["data"]["measurement"]["image"] == f"perception/{m['id']}.png"
    with np.load(d.k.run_dir / "perception" / f"{m['id']}.npz") as arrays:
        np.testing.assert_allclose(arrays["points"], [[.005, .005, .5]])
    with pytest.raises(DaemonError, match="no longer held"):
        c.measure("an old frame", point=[30, 20])
    projected = c.measure(frame.id, point=[30, 20], plane=dict(box=[5, 5, 25, 30], max_error_m=.001))
    assert projected["surface_center"] == m["surface_center"]
    with np.load(Path(projected["image"]).with_suffix(".npz")) as arrays:
        assert len(arrays["support_points"]) > 16
    guard = d.measurements.guard([dict(evidence=projected["id"], max_age_s=5)])
    d.measurements.now = lambda: frame.timestamp + 6
    with pytest.raises(Refused, match="old"):
        guard()


def test_stale_or_recalibrated_measurements_refuse_admission_and_later_starts(k, required):
    guard, now = required
    now[0] += 5
    with pytest.raises(Refused, match="old"):
        k.submit({"do": "gripper", "aperture_mm": 65}, guard=guard)
    now[0] -= 5
    job = k.submit({"do": "gripper", "aperture_mm": 65}, guard=guard)
    before = k.cmd.gripper
    camera = k.cameras["test"]
    camera.view = camera.view            # installing even the same calibration again makes measurements stale
    advance(k, lambda: job.finished)
    assert job.status == "refused" and k.cmd.gripper == before
    assert job.outcome.data["rule"] == "stale_measurement"


def test_expiry_at_a_nested_step_does_not_open_the_gripper(k, required):
    guard, now = required
    job = k.submit([[{"do": "checkpoint", "ask": "continue?"}], [{"do": "gripper", "aperture_mm": 65}]],
                   guard=guard)
    advance(k, lambda: job.status == "waiting")
    before = k.cmd.gripper
    now[0] += 6
    k.answer(job.id, "yes")
    advance(k, lambda: job.finished)
    assert job.status == "refused" and k.cmd.gripper == before


def test_a_grip_checks_again_between_opening_and_closing(k, required):
    guard, now = required
    job = k.submit({"do": "grip", "start_mm": 65, "expect_mm": [35, 45]}, guard=guard)
    advance(k, lambda: job.status == "running")
    now[0] += 6
    advance(k, lambda: job.finished, limit=1000)
    assert job.status == "refused" and job.outcome.data["rule"] == "stale_measurement"
    assert k.manifest.gripper.aperture(k.cmd.gripper) == pytest.approx(.065)


def test_expiry_during_background_preparation_is_checked_before_motion(k, required):
    guard, now = required
    prepared = Joints(delta_deg={"1": 1})
    prepared.prepare(k)
    pending = Future()
    k.planner = SimpleNamespace(prepare=lambda *args: (pending, snapshot(k)))
    before = k.cmd.q.copy()
    job = k.submit(prepared.spec(), guard=guard)
    advance(k, lambda: job.status == "running")
    now[0] += 6
    pending.set_result(prepared.__dict__)
    advance(k, lambda: job.finished)
    assert job.status == "refused" and job.outcome.data["rule"] == "stale_measurement"
    assert k.cmd.q[0] == pytest.approx(before[0], abs=.001)
    np.testing.assert_array_equal(k.cmd.q, k.state.q)  # hold measured feedback, including idle drift


def test_a_guarded_plan_with_a_custom_step_is_refused_before_any_step_starts(k, required):
    guard, _ = required

    class Custom(Behavior):
        kind = "custom"

    with pytest.raises(Refused, match="built-in"):
        k.submit([{"do": "gripper", "aperture_mm": 65}, Custom()], guard=guard)
    assert not k.jobs
    k.guarded_steps = k.guarded_steps | {Custom}
    job = k.submit(Custom(), guard=guard)
    advance(k, lambda: job.finished)
    assert job.status == "done"

    class Unreviewed(Custom):
        pass

    with pytest.raises(Refused, match="unchecked"):
        k.submit(Unreviewed(), guard=guard)


def test_expiry_during_rehearsal_refuses_before_submission(daemon, monkeypatch):
    d, c = daemon
    frame = d.measurements.keep(held(d.cameras))
    m = c.measure(frame.id, point=[30, 20])
    original = d.rehearser.check

    def delayed(*args, **kwargs):
        report = original(*args, **kwargs)
        d.measurements.now = lambda: frame.timestamp + 6
        return report

    monkeypatch.setattr(d.rehearser, "check", delayed)
    with pytest.raises(DaemonError, match="old"):
        c.run({"do": "gripper", "aperture_mm": 65}, requires=[dict(evidence=m["id"], max_age_s=5)])
    assert not d.k.jobs


def test_a_withdrawn_measurement_stops_its_run_and_leaves_new_ones_of_the_target_usable(daemon):
    d, c = daemon
    first = c.measure(d.measurements.keep(held(d.cameras)).id, point=[30, 20], target="block")
    job = c.run([{"do": "checkpoint", "ask": "continue?"}, {"do": "gripper", "aperture_mm": 65}], check=False,
                wait=30, requires=[dict(evidence=first["id"], max_age_s=30)])
    assert job["status"] == "waiting"
    c.withdraw([first["id"]], "tracking lost 'block'")
    result = c.answer(job["id"], "yes", wait=30)
    assert result["status"] == "refused" and result["outcome"]["data"]["rule"] == "stale_measurement"
    second = c.measure(d.measurements.keep(held(d.cameras)).id, point=[30, 20], target="block")
    requires = [dict(evidence=second["id"], max_age_s=30)]
    assert c.run({"do": "hold", "seconds": .01}, check=False, wait=30, requires=requires)["status"] == "done"
    with pytest.raises(DaemonError, match="no measurement"):
        c.run({"do": "hold", "seconds": .01}, requires=[dict(evidence="from another session", max_age_s=30)])


@pytest.mark.rendering
def test_simulated_depth_measurement_guards_runs_without_rehearsal(daemon):
    d, c = daemon
    frame = c.frame("top", depth=True)
    valid = np.argwhere(np.isfinite(frame.depth))
    y, x = valid[len(valid) // 2]
    m = c.measure(frame, point=[int(x), int(y)], target="surface")
    assert m["valid"] and m["capture_t"] is not None
    requires = [dict(evidence=m["id"], max_age_s=30)]
    assert c.run({"do": "hold", "seconds": .01}, requires=requires, wait=30)["status"] == "done"
    d.measurements.now = lambda: frame.timestamp + 31
    with pytest.raises(DaemonError, match="old"):
        c.run({"do": "gripper", "aperture_mm": 65}, requires=requires, check=False)


def test_the_example_checks_lift_and_placement_from_measurements():
    from world_use.examples.perception import TARGET, lifted, placed

    before = dict(surface_center=[.34, .03, .25], from_tool=[-.02, 0, .03])
    assert lifted(before, dict(surface_center=[.34, .03, .31], from_tool=[-.02, 0, .03]))
    assert not lifted(before, dict(surface_center=[.34, .03, .25], from_tool=[-.02, 0, -.03]))   # stayed down
    top = (TARGET + [0, 0, .05]).tolist()
    assert placed(dict(surface_center=top, from_tool=[0, 0, -.06]), 65)
    assert not placed(dict(surface_center=top, from_tool=[0, 0, -.06]), 40)                # still closed
    assert not placed(dict(surface_center=top, from_tool=[0, 0, -.02]), 65)                # not withdrawn
    assert not placed(dict(surface_center=[top[0] + .02, *top[1:]], from_tool=[0, 0, -.06]), 65)


@pytest.mark.rendering
def test_the_example_measures_moves_and_measures_again(tmp_path):
    from world_use.examples.perception import run

    result = run(tmp_path, speed=4)           # physics at four times real time; a faster clock is not modest
    assert result["lift"] == result["placement"] == "pass", result
    assert result["evaluation"]["success"] and result["torque_off"]
    assert "block" not in json.loads((tmp_path / "session.json").read_text())["initial"]["world"]["boxes"]
    files = [Path(tmp_path / "perception" / m["id"]) for m in result["measurements"]]
    assert len(files) == 5 and all(f.with_suffix(".png").is_file() and f.with_suffix(".npz").is_file() for f in files)
