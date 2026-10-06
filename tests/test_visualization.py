import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from world_use.config import load_robot, manifest_data
from world_use.recorder import Tape, save_arrays, save_summary
from world_use.visualization import RecordReader, view
from world_use.world import World


def samples(times, angles):
    tape = Tape(2)
    for t, angle in zip(times, angles, strict=True):
        tape.add(t, True, True, 1, [0, 0], [angle, 0], [1, 2], [30, 40], None, None, None)
    return tape.arrays()


@pytest.fixture
def run_folder(tmp_path):
    manifest = load_robot(Path(__file__).parents[1] / "examples/adapters/planar.toml")
    folder = tmp_path / "record"
    folder.mkdir()
    shutil.copyfile(manifest.urdf, folder / "robot.urdf")
    model = manifest_data(manifest) | {"urdf": "robot.urdf"}
    world = World()
    world.add_box("block", "object", [.35, 0, .2], [.04] * 3)
    world.held = ("block", np.eye(4))
    (folder / "session.json").write_text(json.dumps(dict(
        body="test arm", mode="simulation", package_version="test",
        initial=dict(model=model, world=world.to_dict(), q=[0, 0], q_start=[0, 0], gripper=None))))
    return folder


def test_follow_reader_retains_partial_events_and_reads_new_chunks_once(run_folder):
    folder = run_folder
    (folder / "tape").mkdir()
    event = json.dumps(dict(seq=1, t=0, kind="connected", level="info", message="ready")) + "\n"
    (folder / "events.jsonl").write_text(event[:20])
    save_arrays(folder / "tape/000000.npz", samples([0], [0]))
    reader = RecordReader(folder)
    a, events = reader.poll()
    assert a["t"].tolist() == [0] and events == []
    (folder / "events.jsonl").write_text(event)
    save_arrays(folder / "tape/000001.npz", samples([1], [.5]))
    a, events = reader.poll()
    assert a["t"].tolist() == [1] and events[0]["message"] == "ready"
    assert reader.poll() == ({}, [])


def component_rows(path, entity, component):
    from rerun.chunk import RrdReader
    out = []
    for chunk in RrdReader(path).stream():
        if chunk.entity_path != entity:
            continue
        batch = chunk.to_record_batch()
        if component not in batch.schema.names:
            continue
        times = batch.column("elapsed").cast("int64").to_pylist()
        out.extend(zip(times, batch.column(component).to_pylist(), strict=True))
    return sorted(out, key=lambda pair: pair[0])


def test_a_measurement_shows_its_picture_at_capture_and_its_points_when_measured(run_folder):
    rr = pytest.importorskip("rerun")
    rr.set_strict_mode(True)
    (run_folder / "perception").mkdir()
    Image.new("RGB", (16, 16), "orange").save(run_folder / "perception/one.png")
    save_arrays(run_folder / "perception/one.npz", dict(points=np.array([[.3, .1, .2]]), pixels=np.array([[8, 8]])))
    event = dict(seq=1, t=2, kind="measurement", level="info", message="block at F+0.300 L+0.100 U+0.200",
                 data=dict(measurement=dict(id="one", camera="side", target="block", valid=True, capture_t=1,
                                            image="perception/one.png")))
    (run_folder / "events.jsonl").write_text(json.dumps(event) + "\n")
    output = view(run_folder, output=run_folder / "measurement.rrd")
    assert component_rows(output, "/observations/side", "EncodedImage:blob")[0][0] == 1_000_000_000
    rows = component_rows(output, "/scene/observed/block", "Points3D:positions")
    assert rows[0][0] == 2_000_000_000
    assert rows[0][1][0] == pytest.approx([.3, .1, .2])


def test_export_preserves_measurements_world_changes_and_camera_timing_after_move(run_folder, monkeypatch):
    rr = pytest.importorskip("rerun")
    rr.set_strict_mode(True)
    from world_use import bodies

    monkeypatch.setattr(bodies, "make", lambda *a, **kw: pytest.fail("viewer opened a robot"))
    (run_folder / "tape").mkdir()
    save_arrays(run_folder / "tape/000000.npz", samples([0, .1, .2], [0, np.pi / 2, 0]))
    (run_folder / "views").mkdir()
    Image.new("RGB", (16, 16), "red").save(run_folder / "views/camera.png")
    events = [dict(seq=1, t=.1, kind="look", level="info", message="captured",
                   data=dict(camera="side", path="views/camera.png")),
              dict(seq=2, t=.2, kind="world_state", level="info", message="removed",
                   data=dict(world=World().to_dict())),
              dict(seq=3, t=.3, kind="closed", level="info", message="closed")]
    (run_folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    moved = run_folder.with_name("moved")
    run_folder.rename(moved)
    output = view(moved, output=moved / "view.rrd")
    from rerun.chunk import RrdReader
    assert len(RrdReader(output).blueprints()) == 1
    measured = component_rows(output, "/signals/joints/shoulder/measured", "Scalars:scalars")
    assert [v[0] for _, v in measured] == pytest.approx([0, 90, 0])
    assert [t for t, _ in measured] == [0, 100_000_000, 200_000_000]
    rotations = component_rows(output, "/transforms/shoulder", "Transform3D:quaternion")
    assert next(v[0] for t, v in rotations if t == 100_000_000) == pytest.approx([0, 0, 2**-.5, 2**-.5])
    positions = component_rows(output, "/scene/world/block", "Transform3D:translation")
    assert next(v[0] for t, v in positions if t == 100_000_000) == pytest.approx([0, .35, .2], abs=1e-6)
    parents = component_rows(output, "/scene/world/block", "Transform3D:parent_frame")
    assert all(value == ["robot/base"] for _, value in parents)
    frames = component_rows(output, "/scene/world/block", "CoordinateFrame:frame")
    assert all(value == ["world/block"] for _, value in frames)
    clears = component_rows(output, "/scene/world/block", "Clear:is_recursive")
    assert clears == [(200_000_000, [True])]
    images = component_rows(output, "/observations/side", "EncodedImage:blob")
    assert images[0][0] == 100_000_000
    assert component_rows(output, "/events", "TextLog:text")[-1] == (300_000_000, ["closed: closed"])


def test_follow_waits_for_the_final_chunk_after_close(run_folder, monkeypatch):
    pytest.importorskip("rerun")
    from world_use import visualization

    (run_folder / "tape").mkdir()
    save_arrays(run_folder / "tape/000000.npz", samples([0], [0]))
    polls = 0

    def advance(_):
        nonlocal polls
        polls += 1
        if polls == 1:
            (run_folder / "events.jsonl").write_text(json.dumps(
                dict(seq=1, t=.2, kind="closed", level="info", message="closed")) + "\n")
        elif polls == 5:
            save_arrays(run_folder / "tape/000001.npz", samples([.1, .2], [.5, 1]))
        elif polls == 7:
            save_summary(run_folder / "complete.json",
                         dict(parts=2, events_bytes=(run_folder / "events.jsonl").stat().st_size))
        elif polls > 7:
            pytest.fail("following did not stop after close")

    monkeypatch.setattr(visualization.time, "sleep", advance)
    output = view(run_folder, output=run_folder / "follow.rrd", follow=True)
    rows = component_rows(output, "/signals/joints/shoulder/measured", "Scalars:scalars")
    assert [t for t, _ in rows] == [0, 100_000_000, 200_000_000]
    assert polls == 7


def test_reader_observes_completion_and_final_chunks_in_one_snapshot(run_folder, monkeypatch):
    from world_use import visualization

    (run_folder / "tape").mkdir()
    save_arrays(run_folder / "tape/000000.npz", samples([0], [0]))
    read = visualization.read_chunks

    def close_during_poll(paths):
        a = read(paths)
        save_arrays(run_folder / "tape/000001.npz", samples([.1], [.5]))
        save_summary(run_folder / "complete.json", dict(parts=2, events_bytes=0))
        return a

    monkeypatch.setattr(visualization, "read_chunks", close_during_poll)
    reader = RecordReader(run_folder)
    a, _ = reader.poll()
    assert a["t"].tolist() == [0] and not reader.complete
    a, _ = reader.poll()
    assert a["t"].tolist() == [.1] and reader.complete


@pytest.mark.parametrize("parts_delta, bytes_delta", [(1, 0), (-1, 0), (0, 1), (0, -1)])
def test_reader_does_not_finish_with_inconsistent_completion_counts(run_folder, parts_delta, bytes_delta):
    (run_folder / "tape").mkdir()
    save_arrays(run_folder / "tape/000000.npz", samples([0], [0]))
    line = json.dumps(dict(seq=1, t=0, kind="closed", level="info", message="closed")) + "\n"
    (run_folder / "events.jsonl").write_text(line)
    save_summary(run_folder / "complete.json", dict(parts=1 + parts_delta, events_bytes=len(line) + bytes_delta))
    reader = RecordReader(run_folder)
    reader.poll()
    assert not reader.complete


def test_follow_finishes_an_empty_run_only_after_its_final_events(run_folder, monkeypatch):
    pytest.importorskip("rerun")
    from world_use import visualization

    log = run_folder / "events.jsonl"
    line = json.dumps(dict(seq=1, t=.1, kind="closed", level="info", message="closed")) + "\n"
    log.write_text(line[:20])
    save_summary(run_folder / "complete.json", dict(parts=0, events_bytes=len(line.encode())))
    polls = 0

    def advance(_):
        nonlocal polls
        polls += 1
        assert polls == 1, "following did not stop after the committed final event became visible"
        log.write_text(line)

    monkeypatch.setattr(visualization.time, "sleep", advance)
    output = view(run_folder, output=run_folder / "empty.rrd", follow=True)
    assert component_rows(output, "/events", "TextLog:text") == [(100_000_000, ["closed: closed"])]
    assert polls == 1


def test_a_gripper_the_viewer_cannot_move_is_drawn_static_with_a_warning(run_folder):
    rr = pytest.importorskip("rerun")
    rr.set_strict_mode(True)
    urdf = run_folder / "robot.urdf"
    urdf.write_text(urdf.read_text().replace("</robot>", """
  <link name="jaw"><visual><geometry><box size="0.04 0.01 0.01"/></geometry></visual></link>
  <joint name="jaw" type="revolute">
    <parent link="forearm"/><child link="jaw"/><origin xyz="0.15 0 0"/><axis xyz="0 0 1"/>
    <limit lower="0" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>"""))
    (run_folder / "tape").mkdir()
    save_arrays(run_folder / "tape/000000.npz", samples([0, .1], [0, .5]))
    output = view(run_folder, output=run_folder / "jaw.rrd")
    assert [t for t, _ in component_rows(output, "/transforms/jaw", "Transform3D:quaternion")] == [0]
    assert any("jaw" in text[0] for _, text in component_rows(output, "/events", "TextLog:text"))


def test_interactive_view_launches_the_sibling_app_and_streams_the_record(run_folder, monkeypatch):
    pytest.importorskip("rerun")
    from rerun import sinks

    app = run_folder / "tool-bin/rerun"
    app.parent.mkdir()
    app.touch(mode=0o755)
    monkeypatch.setattr(sys, "executable", str(app.parent / "python"))
    launched = {}
    monkeypatch.setattr(sinks, "_spawn_viewer", lambda **options: launched.update(options))
    output = run_folder / "interactive.rrd"
    monkeypatch.setattr(sinks, "connect_grpc", lambda _, *, recording, **kw: recording.save(output))
    (run_folder / "tape").mkdir()
    save_arrays(run_folder / "tape/000000.npz", samples([0, .1], [0, .5]))
    view(run_folder)
    assert launched["executable_path"] == str(app)
    assert [t for t, _ in component_rows(output, "/signals/joints/shoulder/measured", "Scalars:scalars")] == [
        0, 100_000_000]


def test_missing_extra_gives_installation_hint(run_folder, monkeypatch):
    monkeypatch.setitem(sys.modules, "rerun", None)
    with pytest.raises(ValueError, match=r"world-use\[rerun\]"):
        view(run_folder, output=run_folder / "view.rrd")
