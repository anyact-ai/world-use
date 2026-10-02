"""Tracking contracts without downloading weights. Real-model replay is an explicit example check."""
import os
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from world_use.cameras import FileCamera, Frame
from world_use.vision import EdgeTAM, _discard_history


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
