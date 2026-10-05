"""Procedure contracts on synthetic captures; real model/task checks are separate."""
import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from world_use.cameras import Camera, Frame, View
from world_use.client import DaemonError
from world_use.errors import Refused
from world_use.mcp_server import build
from world_use.observations import Perception, crop_image, rectangle
from world_use.perception import measure
from world_use.vision import Observation


def capture(daemon, *, height=.5, target="object", aperture=65, timestamp=None, tool=None):
    d, c = daemon
    if "fixture" not in d.cameras:
        d.cameras["fixture"] = Camera("fixture", View(np.eye(4), 1000, 1000, 50, 50, 100, 100))
    cam = d.cameras["fixture"]
    if tool is None:
        tool = np.eye(4)
        tool[:3, 3] = [.02, 0, height]
    frame = Frame(Image.new("RGB", (100, 100), "orange"), "fixture", view=cam.view,
                  depth=np.full((100, 100), height), tool=tool, aperture_mm=aperture,
                  calibration=cam.calibration_id, elapsed=d.k.clock.now() - d.k.t0)
    if timestamp is not None:
        frame = replace(frame, timestamp=timestamp)
    frame = d.evidence.remember(frame)
    extent = round(.04 * 1000 / height / 2)
    receipt = c.record(evidence=measure(frame, mask=rectangle(frame, [50-extent, 50-extent, 50+extent, 50+extent]),
                                       target=target))
    return frame, receipt


def box(daemon, **kwargs):
    frame, receipt = capture(daemon, **kwargs)
    geometry = daemon[1].fit_geometry(receipt["id"], size_m=[.04, .04, .1], frame="base")
    assert geometry["valid"], geometry
    return frame, receipt, geometry


def lift_spec(geometry):
    return dict(kind="lift", before=geometry["id"], frame="base", min_up_m=.035, max_error_m=.015, max_age_s=30)


def test_geometry_reference_resolves_frames_and_carries_support(daemon):
    d, _ = daemon
    _, receipt, geometry = box(daemon)
    spec = dict(do="move_to", frame="work", to=dict(geometry=geometry["id"], component="center",
                                                   offset_m=[.01, .02, .03], offset_frame="work"))
    before = d.k.cmd.q.copy()
    prepared = d.phases.prepare(spec, max_age_s=10)
    expected = d.k.world.from_base("work", geometry["components"]["center"]["value"]) + [.01, .02, .03]
    assert prepared["plan"]["to"] == pytest.approx(expected)
    assert prepared["requires"] == [dict(evidence=receipt["id"], max_age_s=10)]
    np.testing.assert_array_equal(d.k.cmd.q, before)
    spec["to"]["offset_m"][0] = 99
    assert prepared["derivations"][0]["reference"]["offset_m"][0] == .01
    with pytest.raises((ValueError, Refused), match="max_age_s"):
        d.phases.prepare(spec)
    with pytest.raises(ValueError, match="direction"):
        d.phases.prepare(dict(do="move_to", point=dict(geometry=geometry["id"], component="center")), max_age_s=10)


def test_prepared_plan_retries_and_payload_conflicts_cannot_duplicate_motion(daemon):
    d, c = daemon
    prepared = c.check(dict(do="hold", seconds=.1), prepare=True)["prepared"]
    count = len(d.k.jobs)
    with ThreadPoolExecutor(2) as pool:
        calls = [pool.submit(c.run, plan_id=prepared["id"], wait=5) for _ in range(2)]
        one, two = [f.result() for f in calls]
    assert one["id"] == two["id"] and one["status"] == two["status"] == "done"
    assert len(d.k.jobs) == count + 1
    with pytest.raises(DaemonError, match="one request_id"):
        c.run(plan_id=prepared["id"], request_id=c.status()["session_id"] + ":different")
    # The original job remains retrievable after the prepared-plan cache expires.
    d.phases.plans.clear()
    assert c.run(plan_id=prepared["id"])["id"] == one["id"]
    with pytest.raises(DaemonError, match="different submission"):
        c.run(dict(do="hold", seconds=.2), request_id=prepared["request_id"])
    with pytest.raises(DaemonError, match="session_id"):
        c.run(dict(do="hold", seconds=.1), request_id="previous-session:request")
    assert len(d.k.jobs) == count + 1


def test_literal_key_retries_keep_refusals_and_accept_new_keys(daemon):
    d, c = daemon
    key = c.status()["session_id"] + ":intent"
    with pytest.raises(DaemonError, match="unknown behavior"):
        c.run(dict(do="unknown"), request_id=key)
    with pytest.raises(DaemonError, match="different submission"):
        c.run(dict(do="hold", seconds=.1), request_id=key)
    assert c.run(dict(do="hold", seconds=.1), request_id=key + "2", wait=5)["status"] == "done"
    assert len(d.k.jobs) == 1


def test_target_loss_refuses_pending_step_but_preserves_historical_receipt(daemon):
    d, c = daemon
    _, receipt, _ = box(daemon)
    job = c.run([dict(do="checkpoint", ask="continue?"), dict(do="gripper", aperture_mm=65)],
                check=False, wait=5, requires=[dict(evidence=receipt["id"], max_age_s=30)])
    assert job["status"] == "waiting"
    c.invalidate_target("object")
    result = c.answer(job["id"], "yes", wait=5)
    assert result["status"] == "refused" and result["outcome"]["data"]["rule"] == "plan_invalidated"
    assert d.evidence.receipts[receipt["id"]]["valid"]


def test_prepared_plan_refuses_changed_frames_and_cannot_skip_checks(daemon):
    _, c = daemon
    prepared = c.check(dict(do="hold", seconds=.1), prepare=True)["prepared"]
    with pytest.raises(DaemonError, match="cannot override"):
        c.run(plan_id=prepared["id"], check=False)
    fresh = c.check(dict(do="hold", seconds=.1), prepare=True)["prepared"]
    c.world(frame=dict(name="work", origin=[.02, 0, 0]))
    with pytest.raises(DaemonError, match="reference frame changed"):
        c.run(plan_id=fresh["id"])


def test_verification_uses_frozen_criteria_and_observations_on_both_sides_of_job(daemon):
    d, c = daemon
    _, _, before = box(daemon, tool=d.k.tool.copy())
    criterion = lift_spec(before)
    prepared = c.check(dict(do="line", up=.06), prepare=True, effects=[criterion])["prepared"]
    criterion["min_up_m"] = .9
    job = c.run(plan_id=prepared["id"], wait=5)
    assert c.verify_effect(job["id"], before["id"])["reason"] == "observation_must_follow_job_completion"
    _, _, after = box(daemon, height=.56, tool=d.k.tool.copy())
    result = c.verify_effect(job["id"], after["id"])
    assert result["status"] == "pass" and result["criteria"]["min_up_m"] == .035
    _, _, stationary = box(daemon, tool=d.k.tool.copy())
    assert c.verify_effect(job["id"], stationary["id"])["status"] == "fail"
    _, _, other = box(daemon, height=.56, target="another object")
    assert c.verify_effect(job["id"], other["id"])["reason"] == "incompatible_observations"
    assert c.verify_effect(job["id"], "expired")["status"] == "unknown"


def test_lift_verification_cannot_credit_motion_outside_the_job(daemon):
    _, c = daemon
    _, _, before = box(daemon)
    prepared = c.check(dict(do="hold", seconds=.1), prepare=True, effects=[lift_spec(before)])["prepared"]
    job = c.run(plan_id=prepared["id"], wait=5)
    _, _, after = box(daemon, height=.56)  # Synthetic object/tool displacement did not occur in this hold job.
    result = c.verify_effect(job["id"], after["id"])
    assert result["status"] == "unknown" and result["reason"] == "observation_does_not_match_job_boundary"


def test_placement_needs_captured_release_withdrawal_and_shape_height(daemon):
    _, c = daemon
    _, _, before = box(daemon)
    effects = [dict(kind="placement", before=before["id"], frame="base", target_m=[0, 0, .45],
                    position_tolerance_m=.01, support_z_m=.4, height_tolerance_m=.008,
                    min_clearance_m=.04, min_aperture_mm=60, max_age_s=30)]
    prepared = c.check(dict(do="hold", seconds=.1), prepare=True, effects=effects)["prepared"]
    job = c.run(plan_id=prepared["id"], wait=5)
    _, _, released = box(daemon)
    assert c.verify_effect(job["id"], released["id"])["status"] == "pass"
    _, _, closed = box(daemon, aperture=30)
    assert c.verify_effect(job["id"], closed["id"])["status"] == "fail"
    _, _, unsupported = box(daemon, aperture=None)
    assert c.verify_effect(job["id"], unsupported["id"])["status"] == "unknown"


def test_unknown_effect_fields_and_degenerate_geometry_are_not_ignored(daemon):
    _, c = daemon
    _, receipt, before = box(daemon)
    with pytest.raises(DaemonError, match="requires"):
        c.check(dict(do="hold", seconds=.1), prepare=True, effects=[dict(lift_spec(before), typo=1)])
    plane = c.fit_geometry(receipt["id"], kind="plane", frame="base")
    assert plane["valid"] and plane["components"]["normal"]["unsigned"]
    axis = c.fit_geometry(receipt["id"], kind="axis", frame="base")
    assert not axis["valid"] and axis["reason"] == "ambiguous_axis" and not axis["components"]
    wrong = c.fit_geometry(receipt["id"], size_m=[1, 1, 1], frame="base")
    assert not wrong["valid"] and not wrong["components"]


def test_source_support_bounds_and_calibration_invalidation(daemon):
    d, c = daemon
    _, receipt, geometry = box(daemon)
    d.evidence.MAX_SUPPORT_BYTES = 1
    capture(daemon)
    with pytest.raises(DaemonError, match="support expired"):
        c.fit_geometry(receipt["id"], size_m=[.04, .04, .1])
    assert d.phases.geometry(geometry["id"])["valid"]  # derived small geometry survives raw-support eviction
    d.cameras["fixture"].view = d.cameras["fixture"].view
    invalid = d.phases.geometry(geometry["id"])
    assert not invalid["valid"] and not invalid["components"]


def test_reusing_evicted_target_name_does_not_revive_old_receipts(daemon):
    d, c = daemon
    d.evidence.MAX_RECEIPTS = 3
    frame, _ = capture(daemon, target="A")
    for target in ("B", "C", "A", "A"):
        old = c.record(evidence=measure(frame, point=[50, 50], target=target))
    c.record(evidence=measure(frame, point=[50, 50], target="D"))
    new = c.record(evidence=measure(frame, point=[50, 50], target="A"))
    with pytest.raises(Refused, match="continuity"):
        d.k.check_requirements(d.evidence.resolve([dict(evidence=old["id"], max_age_s=30)]))
    d.k.check_requirements(d.evidence.resolve([dict(evidence=new["id"], max_age_s=30)]))


def test_crop_mapping_and_archived_evidence_keep_original_identity(daemon):
    d, c = daemon
    frame, receipt = capture(daemon)
    image, data = crop_image(frame.image, box=[10, 20, 50, 80], max_side=600)
    assert image.size == (400, 600)
    assert np.asarray(data["native_from_image"]) @ [200, 300, 1] == pytest.approx([30, 50, 1])
    c.record()
    deadline = time.monotonic() + 5
    while not (d.k.run_dir / "perception" / receipt["id"] / "measurement.json").exists():
        assert time.monotonic() < deadline
        threading.Event().wait(.01)
    saved = c.evidence_image(receipt["id"])
    assert saved["historical"] and saved["metadata"]["timestamp"] == frame.timestamp


def test_inspection_paginates_persisted_events_and_reports_gaps(daemon):
    d, c = daemon
    for index in range(8):
        d.k.emit("note", f"entry {index}")
    first = c.inspect_run(limit=3)
    second = c.inspect_run(since=first["next_cursor"], limit=3)
    assert len(first["events"]) == 3 and first["more"]
    assert second["events"][0]["seq"] == first["next_cursor"] + 1
    assert first["historical"] and first["record_complete"]
    from world_use.records import page
    assert page(None, live=[dict(seq=5, data={})])["missed"] == 4
    d.k.journal._error = "fixture disk failure"
    damaged = c.inspect_run()
    assert not damaged["record_complete"] and "fixture disk failure" in damaged["problems"][0]


def _hung_provider(conn, options):
    conn.send((True, dict(provider="deadline fixture")))
    conn.recv()
    threading.Event().wait(10)


def _unresponsive_reader(conn, options):
    conn.send((True, dict(provider="blocked input fixture")))
    threading.Event().wait(10)


@pytest.mark.parametrize("provider", [_hung_provider, _unresponsive_reader])
def test_provider_deadline_terminates_process_and_never_replays_request(monkeypatch, provider):
    from world_use import vision_worker
    monkeypatch.setattr(vision_worker, "_serve", provider)
    worker = vision_worker.TrackerProcess(timeout_s=.05, startup_timeout_s=5)
    try:
        started = time.monotonic()
        with pytest.raises(Refused, match="deadline"):
            worker.select("target", Frame(Image.new("RGB", (1024, 1024)), "fixture"), box=[0, 0, 8, 8])
        assert time.monotonic() - started < 3
        assert not worker.ready and not worker.process.is_alive()
        with pytest.raises(Refused, match="closed"):
            worker.update("target", Frame(Image.new("RGB", (8, 8)), "fixture"))
    finally:
        worker.close()


class StaticTracker:
    """Contract fixture, not a learned tracker: retains each explicitly selected rectangle."""
    ready = True

    def __init__(self):
        self.masks = {}
        self.entered = threading.Event()
        self.proceed = threading.Event()
        self.proceed.set()
        self.lost = False

    def select(self, target, frame, **prompt):
        self.masks[target] = rectangle(frame, prompt["box"])
        return self.update(target, frame)

    def update(self, target, frame):
        self.entered.set()
        assert self.proceed.wait(5)
        mask = self.masks[target]
        ys, xs = np.nonzero(mask)
        bounds = None if self.lost else (int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1))
        return Observation(frame.camera, frame.id, frame.timestamp, bounds, None,
                           None if self.lost else mask, 30)

    def forget(self, target):
        self.masks.pop(target, None)

    def close(self):
        self.masks.clear()


def test_mcp_selection_and_fits_share_python_contract_and_slow_inference_does_not_block_status(daemon):
    _, c = daemon
    frame, _ = capture(daemon)
    tracker = StaticTracker()
    p = Perception(c, tracker)
    p.remember(frame)
    server = build(c.url, perception=p)

    async def session():
        tracker.proceed.clear()
        selecting = asyncio.create_task(server.call_tool("select_target", dict(frame=frame.id, box=[10, 10, 90, 90])))
        await asyncio.to_thread(tracker.entered.wait, 3)
        status = await asyncio.wait_for(server.call_tool("status", {}), timeout=1)
        assert status.structured_content["enabled"]
        await asyncio.wait_for(server.call_tool("stop", {}), timeout=1)
        tracker.proceed.set()
        selected = await selecting
        data = selected.structured_content
        assert any(item.type == "image" for item in selected.content)
        geometry = await server.call_tool("fit_geometry", dict(evidence=data["evidence"]["id"],
                                                                size_m=[.04, .04, .1], frame="base"))
        assert geometry.structured_content["valid"]
        updated = await server.call_tool("observe_targets", dict(targets=[data["target"]],
                                                                  frames={"fixture": frame.id}))
        assert updated.structured_content["capture_skew_s"] == 0
        tracker.lost = True
        lost = await server.call_tool("observe_targets", dict(targets=[data["target"]],
                                                               frames={"fixture": frame.id}))
        assert lost.structured_content["observations"][0]["status"] == "lost"
        assert not daemon[0].phases.geometry(geometry.structured_content["id"])["valid"]
        history = await server.call_tool("inspect_run", {})
        assert history.structured_content["historical"]
        assert json.loads(history.content[0].text)["session_id"] == c.status()["session_id"]
    try:
        asyncio.run(session())
    finally:
        tracker.proceed.set()
        p.close()


@pytest.mark.rendering
@pytest.mark.parametrize("scenario", ["displaced", "missing"])
def test_complete_mcp_procedure_and_independent_after_release_evaluation(tmp_path, scenario):
    from world_use.examples.procedures import run
    result = run(tmp_path / scenario, scenario=scenario)
    assert result["torque_off"] and result["interface"] == "MCP"
    if scenario == "missing":
        assert not result["outcomes"] and not result["evaluation"]["success"]
    else:
        assert result["lift"] == result["placement"] == "pass", result.get("reason")
        assert result["evaluation"]["success"]
        assert all("phase" in job for job in result["outcomes"])
