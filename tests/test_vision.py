"""Tracking contracts without downloading weights. The real model runs in CI's vision job."""
import asyncio
import os
import threading
import time
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from world_use.cameras import FileCamera, Frame, SimCamera, View
from world_use.client import DaemonError
from world_use.errors import Refused
from world_use.perception import rectangle
from world_use.tracking import Tracking
from world_use.vision import EdgeTAM, Observation, _discard_history


@pytest.fixture
def tracker(monkeypatch):
    def load(self, *_):
        self.device, self._dtype, self._model = "cpu", None, object()
        self._torch = SimpleNamespace(inference_mode=nullcontext)
        self._processor = SimpleNamespace(init_video_session=lambda **_: object(),
                                          add_inputs_to_inference_session=lambda **_: None)

    def infer(self, frame):
        self._index += 1
        mask = np.asarray(frame.image)[:, :, 0] > 0
        self._last = self._observation(frame, mask)
        return self._last

    monkeypatch.setattr(EdgeTAM, "_load", load)
    monkeypatch.setattr(EdgeTAM, "_infer", infer)
    with EdgeTAM() as instance:
        yield instance


def test_retention_keeps_the_prompt_and_required_forward_memory():
    prompt = object()
    outputs = dict(cond_frame_outputs={0: prompt}, non_cond_frame_outputs={})
    session = SimpleNamespace(processed_frames={}, output_dict_per_obj={0: outputs}, frames_tracked_per_obj={0: {}})
    for index in range(250):
        session.processed_frames[index] = object()
        if index:
            outputs["non_cond_frame_outputs"][index] = object()
            session.frames_tracked_per_obj[0][index] = object()
        _discard_history(session, index, 15)
        assert not session.processed_frames
        assert outputs["cond_frame_outputs"] == {0: prompt}
        expected = set(range(max(1, index - 14), index + 1))
        assert set(outputs["non_cond_frame_outputs"]) == expected
        assert set(session.frames_tracked_per_obj[0]) == expected


def test_duplicate_and_stale_frames_do_not_advance_tracking(tracker, monkeypatch):
    now = [100.0]
    monkeypatch.setattr("world_use.vision.time.monotonic", lambda: now[0])
    image = Image.new("RGB", (10, 10), "red")
    # Selection uses the exact old image the agent inspected.
    seed = tracker.select(Frame(image, "side", timestamp=90), point=(4, 4))
    assert seed.status == "stale" and seed.to_dict()["bbox"] is None
    fresh = Frame(image, "side", timestamp=100)
    result = tracker.update(fresh)
    assert result.status == "tracked" and result.center == (4.5, 4.5)
    assert tracker.update(fresh) is result and tracker._index == 2
    now[0] = 103
    assert result.status == "stale" and result.to_dict()["center"] is None
    stale = tracker.update(Frame(image, "side", timestamp=101))
    assert stale.status == "stale" and stale.mask is None and tracker._index == 2
    missing = tracker.update(Frame(Image.new("RGB", (10, 10)), "side", timestamp=103))
    assert missing.status == "lost" and missing.bbox is None


def test_file_replacement_with_equal_mtime_advances_tracking(tracker, tmp_path, monkeypatch):
    wall, monotonic = [1000.0], [100.0]
    monkeypatch.setattr("world_use.cameras.time", SimpleNamespace(
        time=lambda: wall[0], monotonic=lambda: monotonic[0]))
    monkeypatch.setattr("world_use.vision.time.monotonic", lambda: monotonic[0])
    path = tmp_path / "camera.png"
    Image.new("RGB", (10, 10), "red").save(path)
    os.utime(path, (999.8, 999.8))
    camera = FileCamera("side", path)
    first = camera.capture(None)
    seed = tracker.select(first, point=(4, 4))

    replacement = tmp_path / "replacement.png"
    Image.new("RGB", (10, 10)).save(replacement)
    stat = path.stat()
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    replacement.replace(path)
    # Separate wall/monotonic reads can differ by a microsecond between captures.
    wall[0] += 0.01
    monotonic[0] += 0.009999
    fresh = camera.capture(None)
    result = tracker.update(fresh)
    assert result is not seed and result.frame_id == fresh.id != first.id
    assert result.status == "lost" and fresh.timestamp == first.timestamp
    assert tracker.update(camera.capture(None)) is result
    monotonic[0] += 2
    assert result.status == "stale"


def test_reselection_replaces_history_and_close_ends_the_session(tracker):
    frame = Frame(Image.new("RGB", (10, 10), "red"), "side")
    assert tracker.select(frame, box=(0, 0, 10, 10)).bbox == (0, 0, 10, 10)
    old_session = tracker._session
    for wrong in [Frame(frame.image, "other"), Frame(Image.new("RGB", (20, 10)), "side"),
                  Frame(frame.image, "side", timestamp=frame.timestamp - 1)]:
        with pytest.raises(ValueError, match=r"changed|order"):
            tracker.update(wrong)
    other = Frame(Image.new("RGB", (20, 10), "red"), "other")
    assert tracker.select(other, point=(5, 5)).camera == "other"
    assert tracker._session is not old_session and tracker._index == 1
    tracker.close()
    tracker.close()
    assert tracker._session is None and tracker._last is None and tracker._model is None
    with pytest.raises(RuntimeError, match="closed"):
        tracker.update(other)


@pytest.mark.parametrize("selection", [{}, {"point": (1, 1), "box": (1, 1, 2, 2)},
                                       {"point": (float("nan"), 1)}, {"point": (10, 1)}, {"box": (5, 1, 2, 5)}])
def test_invalid_selection_does_not_replace_a_valid_session(tracker, selection):
    frame = Frame(Image.new("RGB", (10, 10), "red"), "side")
    tracker.select(frame, point=(4, 4))
    session = tracker._session
    with pytest.raises(ValueError):
        tracker.select(frame, **selection)
    assert tracker._session is session


class Synthetic(SimCamera):
    """Serves one synthetic RGB-D picture: a flat orange surface 0.5 m from the camera."""

    def __init__(self):
        super().__init__("synthetic", View(np.eye(4), 1000, 1000, 50, 50, 100, 100), None)

    def capture(self, k, *, depth=False):
        return Frame(Image.new("RGB", (100, 100), "orange"), self.name, view=self.view,
                     calibration=self.calibration_id, depth=np.full((100, 100), .5) if depth else None,
                     tool=np.eye(4))


class Tracker:
    """Contract fixture, not a learned tracker: follows each selected box until told it is lost."""
    ready = True

    def __init__(self):
        self.boxes, self.lost = {}, False
        self.entered, self.proceed = threading.Event(), threading.Event()
        self.proceed.set()

    def select(self, target, frame, **prompt):
        self.boxes[target] = prompt["box"]
        return self.update(target, frame)

    def update(self, target, frame):
        self.entered.set()
        assert self.proceed.wait(5)
        box = None if self.lost else tuple(self.boxes[target])
        mask = None if box is None else rectangle(list(box), frame.image.size)
        return Observation(frame.camera, frame.id, frame.timestamp, box, None, mask, 15)

    def forget(self, target):
        self.boxes.pop(target, None)

    def close(self):
        self.boxes.clear()


def test_a_seed_older_than_the_tracker_accepts_is_measured_in_a_new_frame(daemon):
    d, c = daemon
    d.cameras["synthetic"] = camera = Synthetic()
    old = d.measurements.keep(replace(camera.capture(None, depth=True), timestamp=time.monotonic() - 20))
    m = Tracking(c, Tracker()).select(old.id, "block", box=[30, 30, 70, 70])
    assert m["tracking"] == "tracked" and m["valid"] and m["frame"] != old.id and m["age_s"] < 5


def test_failed_or_lost_targets_are_dropped_and_their_measurements_withdrawn(daemon, monkeypatch):
    d, c = daemon
    d.cameras["synthetic"] = Synthetic()
    tracker = Tracker()
    tracking = Tracking(c, tracker)
    frame = c.frame("synthetic", depth=True)

    def fail(*args, **kwargs):
        raise Refused("fixture inference failed", "provider_error")

    with monkeypatch.context() as patch:
        patch.setattr(tracker, "update", fail)
        with pytest.raises(Refused, match="fixture"):
            tracking.select(frame.id, "block", box=[30, 30, 70, 70])
    assert not tracking.targets and not tracker.boxes
    selected = tracking.select(frame.id, "block", box=[30, 30, 70, 70])
    tracker.lost = True
    (lost,) = tracking.observe(["block"])
    assert lost["tracking"] == "lost" and lost["withdrawn"] == [selected["id"]] and not tracking.targets
    with pytest.raises(DaemonError, match="withdrawn"):
        c.run({"do": "hold", "seconds": .01}, requires=[dict(evidence=selected["id"], max_age_s=30)])
    with pytest.raises(ValueError, match="select it first"):
        tracking.observe(["block"])


def test_mcp_tracking_replies_are_measurements_and_inference_does_not_block_status(daemon):
    from world_use.mcp_server import build

    d, c = daemon
    d.cameras["synthetic"] = Synthetic()
    tracker = Tracker()
    server = build(c.url, tracker=tracker)
    frame = c.frame("synthetic", depth=True)

    async def session():
        tracker.proceed.clear()
        selecting = asyncio.create_task(server.call_tool("select_target", dict(frame=frame.id, target="block",
                                                                               box=[30, 30, 70, 70])))
        await asyncio.to_thread(tracker.entered.wait, 3)
        status = await asyncio.wait_for(server.call_tool("status", {}), timeout=1)
        assert status.structured_content["enabled"]
        await asyncio.wait_for(server.call_tool("stop", {}), timeout=1)
        tracker.proceed.set()
        selected = await selecting
        assert selected.structured_content["valid"] and any(item.type == "image" for item in selected.content)
        observed = await server.call_tool("observe_targets", dict(targets=["block"]))
        (again,) = observed.structured_content["measurements"]
        assert again["target"] == "block" and again["valid"] and again["frame"] != frame.id

    try:
        asyncio.run(session())
    finally:
        tracker.proceed.set()


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


def test_tracking_history_retirement_revokes_an_active_jobs_old_measurement(daemon):
    d, c = daemon
    d.cameras["synthetic"] = Synthetic()
    tracker = Tracker()
    tracking = Tracking(c, tracker)
    first = tracking.select(c.frame("synthetic", depth=True).id, "block", box=[30, 30, 70, 70])
    job = c.run([{"do": "checkpoint", "ask": "Continue?"}, {"do": "hold", "seconds": .01}], check=False,
                wait=30, requires=[dict(evidence=first["id"], max_age_s=3600)])
    for _ in range(256):
        tracking.observe(["block"])
    tracker.lost = True
    tracking.observe(["block"])
    result = c.answer(job["id"], "yes", wait=30)
    assert result["status"] == "refused" and "withdrawn" in result["outcome"]["message"]


def test_selection_provider_failure_withdraws_other_targets(daemon, monkeypatch):
    d, c = daemon
    d.cameras["synthetic"] = Synthetic()
    tracker = Tracker()
    tracking = Tracking(c, tracker)
    first = tracking.select(c.frame("synthetic", depth=True).id, "first", box=[30, 30, 70, 70])
    job = c.run([{"do": "checkpoint", "ask": "Continue?"}, {"do": "hold", "seconds": .01}], check=False,
                wait=30, requires=[dict(evidence=first["id"], max_age_s=60)])

    def timeout(*args, **kwargs):
        tracker.ready = False
        raise Refused("the provider stopped", "provider_timeout")

    monkeypatch.setattr(tracker, "select", timeout)
    with pytest.raises(Refused, match="provider stopped"):
        tracking.select(c.frame("synthetic", depth=True).id, "second", point=[40, 40])
    result = c.answer(job["id"], "yes", wait=30)
    assert result["status"] == "refused" and "withdrawn" in result["outcome"]["message"]
    assert not tracking.targets
