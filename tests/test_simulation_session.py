"""External scene lifecycle and evidence through the same client used by agents."""
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from world_use import Refused, World, bodies
from world_use.body import ToolBox
from world_use.cameras import SimCamera, View
from world_use.client import Client
from world_use.records import replay
from world_use.simulation import MujocoRecording, SimulationSession
from world_use.tasks import RecordingError, read_trace


def session_at(path, *, camera=False, tool_boxes=()):
    truth = World()
    truth.add_box("hidden", "object", [.5, 0, .02], [.03, .03, .04])
    body = bodies.make("sim", truth, q=[.3782, .0015, -.0005, -.0011, .0625, .0051], gripper=1.)
    body.manifest = replace(body.manifest, tool_boxes=tool_boxes)
    cams = {"overview": SimCamera("overview", View.look_at([.8, -.8, .7], [.3, 0, .1], size=(160, 120)), body)}
    return SimulationSession(body, path, cameras=cams if camera else {},
                             sources={"test.py": Path(__file__)}, entrypoint="test.py")


def test_client_execution_keeps_physics_and_rehearsal_scope(tmp_path):
    with session_at(tmp_path / "run") as session:
        client = session.client
        assert client.status()["session"]["mode"] == "simulation"
        assert not client.status()["enabled"]
        assert "hidden" not in client.world()["boxes"]
        coverage = client.capabilities()["rehearsal_coverage"]
        assert coverage["simulation_truth"] is False
        assert "external_scene_geometry_and_dynamics" in coverage["omitted"]
        assert client.check([])["coverage"] == coverage
        client.enable()
        outcome = client.run([dict(do="hold", seconds=.15)], wait=10, check=False)
        assert outcome["status"] == "done"
        with pytest.raises(RuntimeError, match="release power"):
            session.close()
        assert client.status()["enabled"]
        client.release()
        saved = session.close(dict(verdict="pass"))
        assert saved["recording_error"] is None
    root = tmp_path / "run"
    assert not session.daemon.control.is_alive()
    with np.load(root / "replay.npz") as replay:
        assert len(replay["qpos"]) > 2
        assert np.all(np.diff(replay["simulation_time"]) > 0)
        assert replay["state"].shape[0] == replay["time"].shape[0]
    assert json.loads((root / "physics/complete.json").read_text())["error"] is None
    assert any(e["kind"] == "intent" and e["inputs"].get("path") == "/run"
               for e in read_trace(root / "task"))
    assert saved["final_status"]["physics_recording"]["error"] is None


def test_capabilities_identify_declared_collision_geometry(tmp_path):
    bounds = ToolBox("pin", (0, 0, .02), (.008, .008, .04), "measured pin dimensions")
    with session_at(tmp_path / "run", tool_boxes=(bounds,)) as session:
        card = session.client.capabilities()
        coverage = card["rehearsal_coverage"]
        assert coverage["collision"]["tool_boxes"] == [
            dict(name="pin", role="tool", source="measured pin dimensions")]
        assert session.client.check([])["coverage"] == coverage
        assert "declared_tool_and_finger_boxes" in card["collision_coverage"]["checked"]
        assert "self_collision" in card["collision_coverage"]["unchecked"]


@pytest.mark.parametrize("failure", [ValueError("policy failed"), KeyboardInterrupt("task interrupted")])
def test_exception_preserves_trace_and_shuts_down_simulation(tmp_path, failure):
    with pytest.raises(type(failure)), session_at(tmp_path / "run") as session:
        session.client.enable()
        raise failure
    saved = json.loads((tmp_path / "run/task/result.json").read_text())
    assert repr(failure) == saved["exception"]
    assert not saved["final_status"]["enabled"]
    assert not session.daemon.control.is_alive()
    assert not session.physics.thread.is_alive()


def test_shared_client_cannot_stop_the_session_before_its_record_is_saved(tmp_path):
    # MCP servers and trackers built on this client must not end the episode without its task record.
    with session_at(tmp_path / "run") as session:
        with pytest.raises(Refused, match="saves its task record"):
            session.client.shutdown()
        assert session.daemon.control.is_alive() and not session.client.status()["enabled"]
    assert json.loads((tmp_path / "run/task/result.json").read_text())["exception"] is None


def test_failed_cleanup_keeps_the_task_exception_and_the_session_open(tmp_path, monkeypatch):
    def refuse(self, *args, **kwargs):
        raise RuntimeError("release refused")

    monkeypatch.setattr(Client, "release", refuse)
    with pytest.raises(ValueError, match="policy failed") as raised, session_at(tmp_path / "run") as session:
        raise ValueError("policy failed")
    assert "release refused" in raised.value.__notes__[0]
    assert not session.closed
    monkeypatch.undo()
    session.close(exception="recovered after the test")


def test_episodes_do_not_reuse_kernel_or_record(tmp_path):
    with session_at(tmp_path / "one") as first:
        first_id = first.record.manifest["invocation"]
        first.client.world(fact=dict(key="prior", value="old", source="test"))
    with session_at(tmp_path / "two") as second:
        assert second.record.manifest["invocation"] != first_id
        assert "prior" not in second.client.world()["facts"]
        assert second.kernel is not first.kernel
    body = bodies.make("sim", World())
    try:
        with pytest.raises(FileExistsError):
            SimulationSession(body, tmp_path / "one", cameras={}, sources={}, entrypoint="missing.py")
    finally:
        body.close()


def test_physics_storage_failure_is_reported_without_blocking_release(tmp_path, monkeypatch):
    import world_use.simulation as module
    original = module.save_arrays

    def fail_chunks(path, arrays):
        if Path(path).parent.name == "physics":
            raise OSError("disk full")
        return original(path, arrays)

    with session_at(tmp_path / "run") as session:
        monkeypatch.setattr(module, "save_arrays", fail_chunks)
        session.client.enable()
        time.sleep(.6)
        session.client.release()
        saved = session.close(dict(verdict="inconclusive"))
        assert "disk full" in saved["final_status"]["physics_recording"]["error"]
        assert not saved["final_status"]["enabled"]


def test_physics_reset_is_not_a_new_episode(tmp_path):
    body = bodies.make("sim", World())
    body.connect()
    body.read()
    recorder = MujocoRecording(body, tmp_path)
    try:
        with body.lock:
            body.data.time = -1.
        recorder.capture()
        assert "backwards" in recorder.error
    finally:
        recorder.close()
        body.close()


def test_trace_failure_still_closes_a_powered_session(tmp_path, monkeypatch):
    session = session_at(tmp_path / "run")
    original = Path.open

    def fail_trace(path, *args, **kwargs):
        if path == session.record.root / "trace.jsonl":
            raise OSError("trace disk full")
        return original(path, *args, **kwargs)

    with pytest.raises(RecordingError, match="trace disk full"), session:
        session.client.enable()
        monkeypatch.setattr(Path, "open", fail_trace)
        session.client.run([dict(do="hold", seconds=.1)], wait=2, check=False)
    saved = json.loads((session.record.root / "result.json").read_text())
    assert "trace disk full" in saved["recording_error"]
    assert not saved["final_status"]["enabled"]
    assert session.closed and not session.daemon.control.is_alive()


@pytest.mark.rendering
def test_external_replay_uses_saved_scene_and_preserves_raw_observations(tmp_path):
    with session_at(tmp_path / "run", camera=True) as session:
        frame = session.client.frame("overview", depth=True)
        assert frame.depth is not None
    artifact = next((tmp_path / "run/task/artifacts").glob("*.frame.json"))
    raw = json.loads(artifact.read_text())
    assert raw["id"] == frame.id and raw["view"] == frame.view.to_dict()
    gif = replay(tmp_path / "run/flight", tmp_path / "scene.gif", speed=10)
    with Image.open(gif) as image:
        assert image.size == (160, 120)
    # The scene is restored from its own integration state, including the hidden free body.
    from world_use.bodies.mujoco_scene import mj
    model = mj.MjModel.from_binary_path(str(tmp_path / "run/scene.mjb"))
    assert model.body("box/hidden").id > 0
