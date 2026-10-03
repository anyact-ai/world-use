"""Evidence must describe captured pixels, and expire before subsequent actuation."""
import time
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from world_use import Refused
from world_use.behaviors import Behavior, Joints
from world_use.cameras import Camera, Frame, SimCamera, View, pack, unpack
from world_use.perception import EvidenceStore, measure
from world_use.plan import snapshot


def source(**kwargs):
    return Frame(Image.new("RGB", (60, 40), "orange"), "test",
                 view=View(np.eye(4), 50, 50, 30, 20, 60, 40),
                 depth=np.full((40, 60), .5), **kwargs)


def register(k):
    frame = source(elapsed=0)
    cam = Camera(frame.camera, frame.view)
    k.cameras[cam.name] = cam
    store = EvidenceStore(k)
    frame = store.remember(replace(frame, calibration=cam.calibration_id))
    receipt = store.register(measure(frame, point=[30, 20], target="block").request())
    return store, frame, receipt, [dict(evidence=receipt["id"], max_age_s=5)]


def advance(k, predicate, limit=300):
    for _ in range(limit):
        k.tick()
        k.clock.wait()
        if predicate():
            return
    pytest.fail("kernel did not reach the expected state")


def test_rgbd_roundtrip_and_pixel_centres():
    frame = source(session="session", calibration="revision", tool=np.eye(4), elapsed=1.5)
    decoded = Frame.from_dict(frame.to_dict())
    assert decoded.elapsed == 1.5 and decoded.session == "session"
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


def test_task_verification_needs_visible_geometry_and_consistent_displacement():
    from world_use.examples.perception import TARGET, lift_result, placement_result

    before = np.array([.34, .03, .20])
    assert lift_result(before, before + [0, 0, .06], np.array([0, 0, .06])) == "pass"
    assert lift_result(before, before, np.array([0, 0, .06])) == "fail"
    assert lift_result(before, None, np.array([0, 0, .06])) == "unknown"
    assert placement_result(TARGET, TARGET + [0, 0, .1], 65) == "pass"
    assert placement_result(TARGET, TARGET + [0, 0, .1], 40) == "fail"
    assert placement_result(TARGET + [0, 0, .05], TARGET + [0, 0, .15], 65) == "fail"
    assert placement_result(None, TARGET + [0, 0, .1], 65) == "unknown"


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


def test_registration_owns_capture_and_recomputes_geometry(k):
    store, frame, receipt, requires = register(k)
    frame.image.paste("blue", (0, 0, 60, 40))
    request = measure(frame, point=[30, 20]).request()
    request.update(timestamp=time.monotonic() + 1000, geometry={"surface_center": [99, 99, 99]})
    registered = store.register(request)
    assert registered["timestamp"] == frame.timestamp
    assert registered["geometry"]["surface_center"] == pytest.approx([.005, .005, .5])
    assert store.frames[(frame.id, frame.calibration)].image.getpixel((0, 0)) != (0, 0, 255)
    assert not k.world.boxes and store.resolve(requires)
    with pytest.raises(Refused, match="another session"):
        store.register(dict(request, session="previous"))
    store.MAX_FRAMES = 1
    store.remember(source())
    with pytest.raises(Refused, match="acquisition cache"):
        store.register(request)
    # Cache eviction removes raw images, not already accepted guard metadata.
    assert store.resolve(requires)[0].evidence == receipt["id"]


def test_expired_evidence_refuses_admission_and_changed_calibration_refuses_start(k):
    store, frame, _, requires = register(k)
    rules = store.resolve(requires)
    k.evidence_now = lambda: frame.timestamp + 6
    with pytest.raises(Refused, match="old"):
        k.submit({"do": "gripper", "aperture_mm": 65}, requires=rules)
    k.evidence_now = lambda: frame.timestamp + 1
    job = k.submit({"do": "gripper", "aperture_mm": 65}, requires=rules)
    before = k.cmd.gripper
    cam = k.cameras[frame.camera]
    cam.view = cam.view                  # reinstalling even identical calibration invalidates old evidence
    advance(k, lambda: job.finished)
    assert job.status == "refused" and k.cmd.gripper == before
    assert job.outcome.data["rule"] == "stale_evidence"


def test_expiry_at_nested_step_does_not_open_the_gripper(k):
    store, frame, _, requires = register(k)
    now = [frame.timestamp + 1]
    k.evidence_now = lambda: now[0]
    job = k.submit([[{"do": "checkpoint", "ask": "continue?"}],
                    [{"do": "gripper", "aperture_mm": 65}]], requires=store.resolve(requires))
    advance(k, lambda: job.status == "waiting")
    before = k.cmd.gripper
    now[0] += 6
    k.answer(job.id, "yes")
    advance(k, lambda: job.finished)
    assert job.status == "refused" and k.cmd.gripper == before


def test_grip_rechecks_evidence_between_its_preopen_and_closing_phases(k):
    store, frame, _, requires = register(k)
    now = [frame.timestamp + 1]
    k.evidence_now = lambda: now[0]
    job = k.submit({"do": "grip", "start_mm": 65, "expect_mm": [35, 45]}, requires=store.resolve(requires))
    advance(k, lambda: job.status == "running")
    now[0] += 6
    advance(k, lambda: job.finished, limit=1000)
    assert job.status == "refused" and job.outcome.data["rule"] == "stale_evidence"
    assert k.manifest.gripper.aperture(k.cmd.gripper) == pytest.approx(.065)


def test_expiry_during_background_preparation_is_checked_before_motion(k):
    store, frame, _, requires = register(k)
    now = [frame.timestamp + 1]
    k.evidence_now = lambda: now[0]
    prepared = Joints(delta_deg={"1": 1})
    prepared.prepare(k)
    pending = Future()
    k.planner = SimpleNamespace(prepare=lambda *args: (pending, snapshot(k)))
    before = k.cmd.q.copy()
    job = k.submit(prepared.spec(), requires=store.resolve(requires))
    advance(k, lambda: job.status == "running")
    now[0] += 6
    pending.set_result(prepared.__dict__)
    advance(k, lambda: job.finished)
    assert job.status == "refused"
    assert job.outcome.data["rule"] == "stale_evidence"
    assert k.cmd.q[0] == pytest.approx(before[0], abs=.001)
    np.testing.assert_array_equal(k.cmd.q, k.state.q)  # hold measured feedback, including idle drift


def test_custom_behavior_is_rejected_before_any_step_starts(k):
    store, _, _, requires = register(k)

    class Custom(Behavior):
        kind = "custom"

    with pytest.raises(Refused, match="built-in"):
        k.submit([{"do": "gripper", "aperture_mm": 65}, Custom()], requires=store.resolve(requires))
    assert not k.jobs


def test_expiry_during_rehearsal_refuses_before_submission(daemon, monkeypatch):
    from world_use.client import DaemonError

    d, c = daemon
    store, frame, _, requires = register(d.k)
    d.evidence = store
    original = d.rehearser.check

    def delayed(*args, **kwargs):
        report = original(*args, **kwargs)
        d.k.evidence_now = lambda: frame.timestamp + 6
        return report

    monkeypatch.setattr(d.rehearser, "check", delayed)
    with pytest.raises(DaemonError, match="old"):
        c.run({"do": "gripper", "aperture_mm": 65}, requires=requires)
    assert not d.k.jobs


@pytest.mark.rendering
def test_daemon_rgbd_receipt_and_no_check_prerequisite(daemon):
    from world_use.client import DaemonError

    d, c = daemon
    frame = c.frame("top", depth=True)
    valid = np.argwhere(np.isfinite(frame.depth))
    y, x = valid[len(valid) // 2]
    receipt = c.record(evidence=measure(frame, point=[int(x), int(y)], target="surface"))
    assert receipt["valid"] and receipt["capture_t"] is not None
    requires = [dict(evidence=receipt["id"], max_age_s=30)]
    assert c.run({"do": "hold", "seconds": .01}, requires=requires, wait=5)["status"] == "done"
    d.k.evidence_now = lambda: frame.timestamp + 31
    with pytest.raises(DaemonError, match="old"):
        c.run({"do": "gripper", "aperture_mm": 65}, requires=requires, check=False)
    c.record()
    folder = d.k.run_dir / "perception" / receipt["id"]
    assert (folder / "rgb.png").is_file() and (folder / "surfaces.npz").is_file()


@pytest.mark.rendering
def test_live_procedure_transfers_an_unknown_block_and_verifies_from_pixels(tmp_path):
    import json

    from world_use.examples.perception import run

    result = run(tmp_path, scenario="shifted")
    assert result["lift"] == result["placement"] == "pass"
    assert result["evaluation"]["success"] and result["torque_off"]
    initial = json.loads((tmp_path / "session.json").read_text())["initial"]["world"]
    assert "block" not in initial["boxes"]
    assert len(list((tmp_path / "perception").glob("*/measurement.json"))) >= 4
