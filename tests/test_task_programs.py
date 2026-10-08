"""Task recording with saved inputs and a fake transport; no daemon, model or simulator is started."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
from PIL import Image

from world_use.cameras import Frame, View
from world_use.client import Client

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/task-programs"
spec = importlib.util.spec_from_file_location("task_record", EXAMPLE / "task_record.py")
assert spec is not None and spec.loader is not None
recording = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = recording
spec.loader.exec_module(recording)


def bundle(tmp_path, code="def run(robot, params, record):\n    return {'verdict': 'pass'}\n", **options):
    task = tmp_path / "task.py"
    task.write_text(code)
    sources = {"task.py": task, **{f"__runner__/{name}": EXAMPLE / name
                                  for name in ("run_task.py", "task_record.py")}}
    return recording.prepare(tmp_path / "invocation", sources, entrypoint="task.py", parameters={}, **options)


def events(root):
    return [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]


def execute(root):
    return subprocess.run([sys.executable, "-I", str(root / "source/__runner__/run_task.py"),
                           "--execute", str(root)], cwd=root / "source", text=True, capture_output=True, check=False)


def test_frozen_entrypoint_and_helper_are_executed_once(tmp_path):
    task = tmp_path / "task.py"
    task.write_text("from helper import VALUE\ndef run(robot, params, record):\n"
                    "    return {'verdict': 'pass', 'value': VALUE, 'parameter': params['n']}\n")
    helper = tmp_path / "helper.py"
    helper.write_text("VALUE = 7\n")
    sources = {"task.py": task, "helper.py": helper,
               **{f"__runner__/{name}": EXAMPLE / name for name in ("run_task.py", "task_record.py")}}
    root = recording.prepare(tmp_path / "run", sources, entrypoint="task.py", parameters={"n": 3})
    task.write_text("raise RuntimeError('mutable original was imported')\n")
    helper.write_text("VALUE = 900\n")
    first = execute(root)
    assert first.returncode == 0, first.stderr
    result_bytes = (root / "result.json").read_bytes()
    assert json.loads(result_bytes)["result"] == {"verdict": "pass", "value": 7, "parameter": 3}
    assert execute(root).returncode != 0
    assert (root / "result.json").read_bytes() == result_bytes


def test_edited_snapshot_is_refused_before_task_import(tmp_path):
    root = bundle(tmp_path)
    (root / "source/task.py").write_text("raise RuntimeError('must not import')\n")
    result = execute(root)
    assert result.returncode != 0 and "source bundle changed" in result.stderr
    assert not (root / "started.json").exists()


def test_failed_cell_keeps_source_inputs_output_and_exception(tmp_path):
    root = bundle(tmp_path)
    record = recording.Record(root)
    cell = tmp_path / "cell.py"
    source = "scaled = points * 2\nprint('calculated')\nraise ValueError('bad fit')\n"
    cell.write_text(source)
    namespace = {}
    with pytest.raises(ValueError, match="bad fit"):
        record.cell(cell, namespace, inputs={"points": np.array([1., 2.])}, outputs=["scaled"])
    cell.write_text("# later revision\n")
    intent, error = events(root)
    assert (root / intent["inputs"]["source"]["path"]).read_text() == source
    assert (root / intent["inputs"]["stdout"]).read_text() == "calculated\n"
    np.testing.assert_array_equal(np.load(root / intent["inputs"]["inputs"]["points"]["array"]["path"]), [1, 2])
    assert error["exception"] == "ValueError" and error["call"] == intent["seq"]
    assert not (root / "result.json").exists()
    cell.write_text("answer = scaled.sum()\n")
    assert record.cell(cell, namespace, outputs=["answer"]) == {"answer": 6.0}


def test_killed_task_leaves_unanswered_intent_not_a_success(tmp_path):
    root = bundle(tmp_path, "import os\ndef run(robot, params, record):\n    record.call(os._exit, 9)\n")
    result = execute(root)
    assert result.returncode == 9
    assert events(root)[-1]["kind"] == "intent" and events(root)[-1]["inputs"]["args"] == [9]
    assert not (root / "result.json").exists()


def test_truncated_trace_exposes_only_the_committed_prefix(tmp_path):
    root = bundle(tmp_path)
    record = recording.Record(root)
    record.append("intent", operation="example", inputs={})
    with (root / "trace.jsonl").open("ab") as stream:
        stream.write(b'{"seq": 2, "kind": "rep')
    assert recording.read_trace(root)[0]["kind"] == "intent"
    assert len(recording.read_trace(root)) == 1
    with (root / "trace.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(json.JSONDecodeError):
        recording.read_trace(root)


@pytest.mark.parametrize("code", ["raise ValueError('broken')", "return None", "return {'verdict': 'maybe'}"])
def test_invalid_or_failed_task_has_a_failure_record(tmp_path, code):
    root = bundle(tmp_path, f"def run(robot, params, record):\n    {code}\n")
    result = execute(root)
    assert result.returncode != 0
    saved = json.loads((root / "result.json").read_text())
    assert saved["result"] is None and saved["exception"]


class Transport:
    def __init__(self):
        self.path = "/fake/flight-1"
        self.requests = []
        self.frames = {}
        self.counter = 0
        self.lose_reply = False
        self.enabled = False
        self.mode = "simulation"
        self.extra_status = {}
        self.after_run = lambda: None

    def call(self, client, method, path, body=None):
        self.requests.append((method, path, body))
        if path == "/status":
            return dict(recording=dict(path=self.path, error=None), session=dict(mode=self.mode),
                        enabled=self.enabled, power_uncertain=False, faulted=False, events=len(self.requests)
                        ) | self.extra_status
        if path == "/world":
            return {"frames": {"work": {"T": np.eye(4).tolist()}}}
        if path == "/card":
            return {"card": "fake simulation"}
        if path.startswith("/frame"):
            frame_id = parse_qs(urlsplit(path).query).get("id", [None])[0]
            if frame_id is not None:
                return self.frames[frame_id].to_dict()
            self.counter += 1
            frame = Frame(Image.fromarray(np.arange(18, dtype=np.uint8).reshape(2, 3, 3)), "top",
                          id=f"frame-{self.counter}", timestamp=time.monotonic(), calibration="calibration-2",
                          view=View.look_at([0, 0, 1], [0, 0, 0], size=(3, 2)),
                          depth=np.array([[1., np.nan, 2.], [3., 4., 5.]]), tool=np.eye(4))
            self.frames[frame.id] = frame
            return frame.to_dict()
        if path == "/measure":
            return dict(id=f"measurement-{self.counter}", frame=body["frame"], valid=True, surface_center=[0., 0., 0.])
        if path == "/run":
            if self.lose_reply:
                raise TimeoutError("reply lost after submission")
            self.after_run()
            return dict(id=3, status="running")
        if path == "/release":
            self.enabled = False
        return {}


@pytest.fixture
def connected(tmp_path, monkeypatch):
    root = bundle(tmp_path, url="http://fake.invalid")
    record = recording.Record(root)
    transport = Transport()
    monkeypatch.setattr(Client, "_call", lambda *args, **kwargs: transport.call(*args, **kwargs))
    client = recording.RecordedClient("http://fake.invalid", record)
    client.begin()
    return client, record, transport


def test_raw_frame_survives_cache_eviction_with_depth_calibration_and_tool(connected):
    client, record, transport = connected
    frame = client.frame("top", depth=True)
    transport.frames.clear()
    reply = events(record.root)[-1]
    saved = Frame.from_dict(json.loads((record.root / reply["result"]["frame"]["path"]).read_text()))
    np.testing.assert_array_equal(saved.image, frame.image)
    np.testing.assert_array_equal(saved.depth, frame.depth)
    np.testing.assert_array_equal(saved.tool, frame.tool)
    assert saved.view.to_dict() == frame.view.to_dict()
    assert saved.timestamp == frame.timestamp and saved.calibration == "calibration-2" and saved.id == frame.id


def test_hardware_is_refused_before_any_mutating_request(tmp_path, monkeypatch):
    root = bundle(tmp_path, url="http://fake.invalid")
    transport = Transport()
    transport.mode = "hardware"
    monkeypatch.setattr(Client, "_call", lambda *args, **kwargs: transport.call(*args, **kwargs))
    client = recording.RecordedClient("http://fake.invalid", recording.Record(root))
    with pytest.raises(ValueError, match=r"session\.mode=simulation"):
        client.begin()
    assert all(method == "GET" for method, _, _ in transport.requests)


def test_another_invocation_cannot_reuse_frames_or_measurements(connected, tmp_path):
    first, _, transport = connected
    frame = first.frame("top", depth=True)
    measurement = first.measure(frame, point=[0, 0])
    folder = tmp_path / "second"
    folder.mkdir()
    root = bundle(folder, url="http://fake.invalid")
    second = recording.RecordedClient("http://fake.invalid", recording.Record(root))
    second.begin()
    before = len(transport.requests)
    with pytest.raises(ValueError, match="another invocation"):
        second.frame(id=frame.id)
    with pytest.raises(ValueError, match="this invocation"):
        second.measure(frame, point=[0, 0])
    with pytest.raises(ValueError, match="another invocation"):
        second.run([], requires=[dict(evidence=measurement["id"], max_age_s=30)])
    assert len(transport.requests) == before
    fresh = second.measure(second.frame("top", depth=True), point=[0, 0])
    assert second.run([], requires=[dict(evidence=fresh["id"], max_age_s=30)])["id"] == 3


def test_tracker_internal_frames_are_captured(connected):
    from world_use.tracking import Tracking

    client, record, _ = connected

    class Tracker:
        def select(self, name, frame, **selection):
            return self.update(name, frame)

        def update(self, name, frame):
            return SimpleNamespace(status="tracked", mask=np.ones((frame.image.height, frame.image.width), bool))

    tracking = Tracking(client, Tracker())
    seed = client.frame("top", depth=True)
    tracking.select(seed.id, "item", point=[0, 0])
    observed = tracking.observe(["item"])[0]
    assert observed["frame"] != seed.id
    frame_refs = [e["result"]["frame"] for e in events(record.root)
                  if e["kind"] == "reply" and isinstance(e["result"], dict)
                  and isinstance(e["result"].get("frame"), dict)]
    saved_ids = {json.loads((record.root / ref["path"]).read_text())["id"] for ref in frame_refs}
    assert saved_ids == {seed.id, observed["frame"]}


def test_daemon_restart_blocks_motion_even_with_current_invocation_evidence(connected):
    client, _, transport = connected
    measurement = client.measure(client.frame("top"), point=[0, 0])
    transport.path = "/fake/flight-2"
    with pytest.raises(ValueError, match="daemon record changed"):
        client.run([], requires=[dict(evidence=measurement["id"], max_age_s=30)])
    assert not any(path == "/run" for _, path, _ in transport.requests)


def test_submission_timeout_is_recorded_and_never_retried(connected):
    client, record, transport = connected
    transport.lose_reply = True
    with pytest.raises(TimeoutError):
        client.run([dict(do="line", up=.01)])
    assert len([1 for _, path, _ in transport.requests if path == "/run"]) == 1
    last = events(record.root)[-1]
    assert last["kind"] == "exception" and last["exception"] == "TimeoutError"


@pytest.mark.parametrize("state", ["powered", "queued", "restarted", "recording_failed"])
def test_launcher_reports_exit_state_without_issuing_robot_actions(tmp_path, monkeypatch, state):
    # Execute with the real recorded client and a fake transport, without spawning a daemon.
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(sys.modules, "task_program", None)
    spec = importlib.util.spec_from_file_location("task_launcher", EXAMPLE / "run_task.py")
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    code = ("import numpy as np\ndef run(robot, params, record):\n"
            "    robot.run([])\n    return {'verdict': 'pass', 'values': np.array([1, 2])}\n")
    root = bundle(tmp_path, code, url="http://fake.invalid")
    transport = Transport()

    def change_state():
        if state == "powered":
            transport.enabled = True
        elif state == "queued":
            transport.extra_status["queued"] = [3]
        elif state == "restarted":
            transport.path = "/fake/flight-2"
        else:
            transport.extra_status["recording"] = dict(path=transport.path, error="disk full")

    transport.after_run = change_state
    monkeypatch.setattr(Client, "_call", lambda *args, **kwargs: transport.call(*args, **kwargs))
    assert launcher.execute(root) == 1
    saved = json.loads((root / "result.json").read_text())
    assert saved["result"]["verdict"] == "pass"  # Preserve what the task actually returned.
    assert all(path in ("/record", "/run") for method, path, _ in transport.requests if method == "POST")
    if state == "restarted":
        assert "daemon record changed" in saved["exception"]
        assert len([1 for _, path, _ in transport.requests if path == "/record"]) == 1  # Begin only.
    else:
        assert saved["exception"] is None


def test_storage_failure_blocks_dispatch_but_not_stop_or_release(connected, monkeypatch):
    client, record, transport = connected
    original = Path.open

    def disk_full(path, *args, **kwargs):
        if path == record.root / "trace.jsonl":
            raise OSError("disk full")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", disk_full)
    with pytest.raises(recording.RecordingError, match="disk full"):
        client.run([dict(do="line", up=.01)])
    assert not any(path == "/run" for _, path, _ in transport.requests)
    transport.enabled = True
    client.stop()
    client.release()
    assert client.status()["enabled"] is False
    assert record.error == "disk full"


def test_frame_storage_failure_does_not_return_an_unrecorded_observation(connected, monkeypatch):
    client, record, transport = connected
    original = recording.write_new

    def fail_frame(path, data):
        if path.name.endswith(".frame.json"):
            raise OSError("frame storage failed")
        return original(path, data)

    monkeypatch.setattr(recording, "write_new", fail_frame)
    with pytest.raises(recording.RecordingError, match="frame storage failed"):
        client.frame("top", depth=True)
    assert not client.frames
    with pytest.raises(recording.RecordingError):
        client.run([])
    assert not any(path == "/run" for _, path, _ in transport.requests)
    assert record.error == "frame storage failed"


def test_analysis_compares_saved_geometry_without_constructing_a_client(tmp_path):
    geometry = tmp_path / "geometry.json"
    data = (EXAMPLE / "alignment.json").read_text()
    geometry.write_text(data)
    output = tmp_path / "analysis"
    result = subprocess.run([sys.executable, str(EXAMPLE / "analyze_alignment.py"), str(geometry),
                             "--output", str(output)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    saved = json.loads((output / "result.json").read_text())
    assert saved["result"]["comparisons"][0]["matches"]
    assert not any(e.get("operation") == "http" for e in events(output))
    assert geometry.read_text() == data
