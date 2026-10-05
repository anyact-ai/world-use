"""The daemon and its client, end to end on a simulated arm in real time."""
import shutil
import socket
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest
from conftest import serving
from PIL import Image

from world_use import Client, Refused
from world_use.cameras import Frame
from world_use.client import DaemonError


def unused_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


def test_status_card_and_a_run_with_wait(client):
    assert client.status()["session"]["mode"] == "simulation"
    assert "idle" in client.status()["line"]
    assert "reBot" in client.card()
    r = client.run({"do": "line", "forward": 0.03, "up": 0.03, "duration": 1.0}, wait=10)
    assert r["status"] == "done" and "tool F" in r["line"]


def test_up_reuses_only_the_requested_adapter_and_workcell(daemon, capsys):
    from world_use import cli
    from world_use.daemon import session_identity

    d, c = daemon
    up = ["--url", c.url, "up"]
    assert cli.main(up + ["--body", "sim"]) == 0
    assert cli.main(up + ["--body", "sim:rebot"]) == 0
    assert cli.main(up + ["--workcell", "block"]) == 2
    assert cli.main(up + ["--body", "rebot"]) == 2
    d.session = session_identity("rebot", {})           # simulate an already-running hardware daemon
    assert cli.main(up + ["--body", "sim", "--enable"]) == 2
    assert "requested sim:rebot" in capsys.readouterr().err


def test_up_refuses_a_robot_the_twin_cannot_model_and_leaves_no_daemon(tmp_path, capsys):
    from world_use import cli

    example = Path(__file__).resolve().parents[1] / "examples" / "adapters"
    shutil.copy(example / "planar.urdf", tmp_path)
    # A gripper the planar URDF has no joint for: the twin has nothing to move.
    gripper = "\n[gripper]\nclosed = 0.0\nopen = 1.0\napproach = [1.0, 0.0, 0.0]\nopens_along = [0.0, 1.0, 0.0]\n"
    (tmp_path / "robot.toml").write_text((example / "planar.toml").read_text() + gripper)
    (tmp_path / "cell.toml").write_text('robot = "robot.toml"\n')
    url = unused_url()
    assert cli.main(["--url", url, "up", "--workcell", str(tmp_path / "cell.toml"), "--runs", str(tmp_path)]) == 1
    assert "cannot model this robot" in capsys.readouterr().err
    assert not Client(url).alive()


def test_up_stops_a_daemon_that_does_not_answer_in_time(tmp_path, monkeypatch, capsys):
    from world_use import cli

    spawned = []
    popen = subprocess.Popen

    def spawn(*args, **kwargs):
        spawned.append(popen(*args, **kwargs))
        return spawned[-1]
    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(cli, "UP_TIMEOUT_S", 0.2)                 # a cold start takes longer than this
    assert cli.main(["--url", unused_url(), "up", "--runs", str(tmp_path)]) == 1
    assert spawned[0].poll() is not None and "was stopped" in capsys.readouterr().err


def test_up_enable_also_applies_to_a_matching_existing_daemon(client):
    from world_use import cli

    client.release()
    assert cli.main(["--url", client.url, "up", "--body", "sim", "--enable"]) == 0
    assert client.status()["enabled"]


def test_refusal_is_reported_as_an_incident(client):
    r = client.run({"do": "line", "forward": 0.40}, wait=5)
    assert r["status"] == "refused" and "incident" in r


def test_malformed_spec_is_a_409_with_the_reason(client):
    with pytest.raises(DaemonError) as e:
        client.run({"do": "teleport"})
    assert e.value.code == 409 and "unknown behavior" in str(e.value)


def test_a_refusal_prints_as_json_when_json_is_asked_for(client, capsys):
    import json

    from world_use import cli
    capsys.readouterr()
    assert cli.main(["--url", client.url, "--json", "run", '{"do": "teleport"}']) == 2
    body = json.loads(capsys.readouterr().out)
    assert "unknown behavior" in json.dumps(body)


def test_http_boundary_rejects_foreign_origins_hosts_and_non_json(client):
    from contextlib import closing
    from http.client import HTTPConnection
    from urllib.parse import urlparse

    client.release()
    url = urlparse(client.url)
    for headers, body, expected in [
        ({"Origin": "https://example.invalid", "Content-Type": "application/json"}, "{}", 403),
        ({"Host": f"example.invalid:{url.port}", "Content-Type": "application/json"}, "{}", 403),
        ({"Content-Type": "text/plain"}, "{}", 415),
        ({"Content-Type": "application/json"}, "[]", 400),
    ]:
        with closing(HTTPConnection(url.hostname, url.port)) as conn:
            conn.request("POST", "/enable", body, headers)
            response = conn.getresponse()
            assert response.status == expected
            response.read()
        assert not client.status()["enabled"]
    client.enable()
    assert client.status()["enabled"]


@pytest.mark.usefixtures("file_camera")
def test_cli_returns_structured_json_and_concise_input_errors(client, tmp_path, capsys):
    import json

    from world_use import cli

    for command, key in [("world", "frames"), ("events", "events"), ("look", "path")]:
        assert cli.main(["--url", client.url, command, "--json"]) == 0
        assert key in json.loads(capsys.readouterr().out)
    for args, message in [
        (["run", "[{"], "invalid JSON"),
        (["check", str(tmp_path / "missing.json")], "cannot read JSON file"),
        (["inspect", str(tmp_path / "missing")], "not a flight record"),
        (["replay", str(tmp_path), "--speed", "0"], "speed"),
    ]:
        assert cli.main(["--url", client.url, *args]) == 2
        assert message in capsys.readouterr().err


def test_checkpoint_round_trip(client):
    client.run({"do": "line", "up": 0.05, "duration": 1.0}, wait=10)
    r = client.run([{"do": "checkpoint", "ask": "clear to go on?"}, {"do": "line", "up": 0.01}], wait=5)
    assert r["status"] == "waiting" and r["question"]["ask"] == "clear to go on?"
    r = client.answer(r["id"], "yes", wait=10)
    assert r["status"] == "done"


def test_home_routes_reject_checkpoints_and_can_be_cleared_from_the_cli(client):
    from world_use import cli

    with pytest.raises(DaemonError, match="not allowed in a home route"):
        client.home_route([{"do": "checkpoint", "ask": "clear?"}])
    route = client.home_route([])
    assert route["ok"] and route["home"].startswith("set")
    assert cli.main(["--url", client.url, "home-route", "null"]) == 0
    assert "not available" in client.status()["home"]


def test_a_home_route_is_rehearsed_when_set_and_must_be_given(client, capsys):
    from world_use import cli

    with pytest.raises(DaemonError, match="needs steps"):
        client._call("POST", "/home_route", {"note": "no steps"})
    r = client.home_route([{"do": "line", "forward": 0.40}])
    assert not r["ok"] and r["text"]
    capsys.readouterr()
    assert cli.main(["--url", client.url, "home-route", '[{"do": "line", "forward": 0.40}]']) == 4
    assert r["text"].splitlines()[0] in capsys.readouterr().out


def test_check_does_not_move_the_robot(daemon):
    d, client = daemon
    before = d.k.cmd.q.copy()
    r = client.check({"do": "line", "up": 0.05})
    assert r["ok"] and "check passed" in r["text"]
    np.testing.assert_array_equal(d.k.cmd.q, before)


def test_frame_returns_native_pixels_without_creating_records(daemon, tmp_path):
    from world_use.cameras import FileCamera

    d, c = daemon
    path = tmp_path / "source.png"
    image = Image.new("RGB", (1300, 40), "red")
    image.putpixel((1100, 20), (20, 60, 90))
    image.save(path)
    d.cameras["side view"] = FileCamera("side view", path, rotate=90)
    shots, events = d.shots, d.k.events.seq
    frame = c.frame("side view")
    assert frame.camera == "side view" and frame.image.size == (40, 1300)
    assert np.array_equal(frame.image, image.transpose(Image.Transpose.ROTATE_270))
    assert c.frame("side view").id == frame.id and frame.age_s < 3
    assert d.shots == shots and not (d.k.run_dir / "views").exists()
    assert not any(e["kind"] == "look" for e in d.k.events.since(events))
    with pytest.raises(DaemonError, match="no camera"):
        c.frame("missing")


def test_stop_interrupts_a_long_move(client):
    client.run({"do": "line", "up": 0.03, "duration": 1.0}, wait=10)
    r = client.run({"do": "line", "forward": 0.05, "duration": 5.0})
    threading.Event().wait(0.5)
    client.stop("test")
    job = client.job(r["id"], wait=2)
    assert job["status"] == "stopped"


def test_facts_and_events(client):
    client.world(fact=dict(key="table.z", value=0.205, source="touchdown"))
    assert client.status()["facts"]["table.z"] == 0.205
    ev = client.events(0)
    assert ev["last"] >= 1 and any(e["kind"] == "connected" for e in ev["events"])


def test_event_poll_reports_a_missed_window_and_advances_only_through_returned_events():
    from types import SimpleNamespace

    from world_use.daemon import Daemon
    from world_use.events import EventLog

    log = EventLog(keep=2)
    for i in range(4):
        log.emit("test", str(i))
    daemon = SimpleNamespace(k=SimpleNamespace(events=log))
    status, result = Daemon.api(daemon, "GET", "/events", {"since": 0}, {})
    assert status == 200 and result["missed"] == 2
    assert [e["seq"] for e in result["events"]] == [3, 4] and result["last"] == 4
    _, result = Daemon.api(daemon, "GET", "/events", {"since": 4}, {})
    assert result == dict(events=[], missed=0, last=4, more=False)
    _, result = Daemon.api(daemon, "GET", "/events", {"since": 0, "limit": "1"}, {})
    assert [e["seq"] for e in result["events"]] == [3] and result["last"] == 3 and result["more"]


def test_run_rehearses_and_refuses_the_whole_plan_with_every_problem_before_anything_moves(daemon):
    d, c = daemon
    before = d.k.cmd.q.copy()
    jobs = len(d.k.jobs)
    r = c.run([{"do": "line", "up": 0.03}, {"do": "line", "left": 0.05}, {"do": "joints", "delta_deg": {"1": 130}}],
              wait=5)
    assert r["status"] == "refused" and r["id"] is None
    assert "step 2/3" in r["incident"] and "step 3/3" in r["incident"] and "from here" in r["incident"]
    assert len(d.k.jobs) == jobs                                        # not even the first step was submitted
    np.testing.assert_array_equal(d.k.cmd.q, before)


def test_run_without_the_rehearsal_is_refused_by_the_kernel_at_the_step(client):
    r = client.run([{"do": "line", "up": 0.03}, {"do": "line", "left": 0.05}], wait=10, check=False)
    assert r["status"] == "refused" and "step 2/2" in r["incident"]


def test_checked_run_refuses_while_another_job_is_waiting(daemon):
    d, c = daemon
    c.run({"do": "checkpoint", "ask": "hold here?"}, wait=5)
    before, count = d.k.cmd.q.copy(), len(d.k.jobs)
    with pytest.raises(DaemonError, match="idle robot"):
        c.run([{"do": "line", "up": 0.03}, {"do": "line", "forward": 0.4}])
    assert len(d.k.jobs) == count and np.array_equal(d.k.cmd.q, before)


@pytest.mark.parametrize("state", ["off", "faulted", "uncertain"])
def test_checked_run_never_skips_rehearsal_in_an_unready_state(daemon, state):
    d, c = daemon
    if state == "off":
        c.release()
    elif state == "faulted":
        d.k.faulted = True
    else:
        d.k.power_uncertain = True
    with pytest.raises(DaemonError, match="confirmed power"):
        c.run({"do": "line", "up": 0.03})
    assert not d.k.jobs


@pytest.mark.parametrize("change", ["world", "stop", "job", "pose"])
def test_checked_run_rejects_changes_during_rehearsal(daemon, monkeypatch, change):
    d, c = daemon
    original = d.rehearser.check

    def changed(spec, k, **kwargs):
        report = original(spec, k, **kwargs)
        with k.lock:
            if change == "world":
                d._world({"fact": {"key": "scene", "value": "changed"}})
            elif change == "stop":
                k.stop()
            elif change == "job":
                k.submit({"do": "hold", "seconds": 0.01})
            else:
                k.cmd.q[0] += 0.02
        return report

    monkeypatch.setattr(d.rehearser, "check", changed)
    with pytest.raises(DaemonError, match="changed during rehearsal"):
        c.run({"do": "line", "up": 0.03})
    assert not any(j.behavior.kind == "line" for j in d.k.jobs.values())


def test_checked_run_revalidates_before_its_first_tick(daemon):
    d, _ = daemon
    d.stop_loop.set()
    d.control.join(2)
    assert not d.control.is_alive()
    _, result = d._run({"do": "line", "up": 0.03}, 0, True)
    job = d.k.jobs[result["id"]]
    before = d.k.cmd.q.copy()
    d._world({"fact": {"key": "scene", "value": "changed after admission"}})
    for _ in range(d.k.residuals.need):
        d.k.tick()
    assert job.status == "refused" and np.array_equal(d.k.cmd.q, before)


def test_observation_events_do_not_invalidate_checked_admission(daemon, monkeypatch):
    d, c = daemon
    original = d.rehearser.check

    def observed(*args, **kwargs):
        report = original(*args, **kwargs)
        c.record(note="looked at the scene", context={"observation": "unchanged"})
        c.measure(d.measurements.keep(Frame(Image.new("RGB", (8, 8)), "side")).id, point=[4, 4])
        return report

    monkeypatch.setattr(d.rehearser, "check", observed)
    assert c.run({"do": "hold", "seconds": .01}, wait=5)["status"] == "done"


def test_a_consumed_stop_still_invalidates_rehearsal(daemon, monkeypatch):
    d, c = daemon
    d.stop_loop.set()
    d.control.join(2)
    assert not d.control.is_alive()
    original = d.rehearser.check

    def stopped(*args, **kwargs):
        report = original(*args, **kwargs)
        c.stop()
        d.k.tick()
        assert d.k._stop is None
        return report

    monkeypatch.setattr(d.rehearser, "check", stopped)
    with pytest.raises(DaemonError, match="changed during rehearsal"):
        c.run({"do": "hold", "seconds": .01})


@pytest.mark.rendering
def test_look_saves_a_picture_with_what_the_kernel_knows_drawn_on_it(daemon):
    _, c = daemon
    r = c.look("top", [{"do": "line", "up": 0.03}])
    img = Image.open(r["path"])
    assert img.size == (800, 600) and Path(r["path"]).parent.name == "views"
    assert "magenta cross" in r["drawn"] and "blue" in r["drawn"] and "check passed" in r["check"]
    assert np.asarray(img).std() > 10                                   # a picture, not a blank
    with pytest.raises(DaemonError) as e:
        c.look("nowhere")
    assert "cameras: side, front, top" in str(e.value)


def test_a_box_the_policy_adds_goes_into_the_model_not_into_the_simulation(daemon):
    d, c = daemon
    line = c.box("tray", "surface", [0.32, 0.0, 0.14], [0.3, 0.4, 0.02], source="side camera")["line"]
    assert line.startswith("surface 'tray': centre F+0.320 L+0.000 U+0.140") and "from side camera" in line
    assert "tray" in d.k.world.boxes and "tray" not in d.k.body.world.boxes
    assert "surface 'tray'" in c.world()["text"] and "surface 'tray'" in c.card()
    assert c.remove("tray")["line"] == "removed 'tray'"


def test_a_workcell_box_marked_unknown_is_only_in_the_simulation(tmp_path, rehearser):
    cell = {"box": [dict(name="shelf", kind="surface", center=[0.3, 0, 0.1], size=[0.2, 0.2, 0.02], known=False)]}
    with serving(tmp_path, cell, rehearser) as (d, _):
        assert "shelf" in d.k.body.world.boxes and "shelf" not in d.k.world.boxes


def test_a_world_change_applies_whole_or_not_at_all(client):
    for change, message in [(dict(box=dict(name="tray", kind="surface", center=[0.3, 0], size=[.1, .1, .1])),
                             "box.center"),
                            (dict(frame=[]), "frame"),
                            (dict(frame=dict(name="tilted", yaw_deg=30)), "yaw_deg")]:
        with pytest.raises(DaemonError, match=message) as e:
            client.world(fact=dict(key="seen", value=True), **change)
        assert e.value.code == 400
    world = client.world()
    assert "seen" not in world["facts"] and "tray" not in world["boxes"] and "tilted" not in world["frames"]
    line = client.world(fact=dict(key="table.z", value=0.205, source="touchdown"))["line"]
    assert line == "fact table.z = 0.205 (from touchdown)"


def test_every_request_gets_an_answer_even_after_an_unexpected_error(daemon):
    d, c = daemon

    def broken():
        raise AttributeError("a bug")
    d.k.reset = broken
    with pytest.raises(DaemonError) as e:
        c.reset()
    assert e.value.code == 500 and "a bug" in str(e.value)
    assert "idle" in c.status()["line"]


def test_help_lists_every_step_with_an_example(client):
    from world_use.behaviors import REGISTRY, build
    steps = client.help()
    assert set(steps) == set(REGISTRY)
    for kind, h in steps.items():
        assert h["summary"] and h["example"]["do"] == kind
        build(h["example"])


def test_an_error_reaches_the_operator_with_its_notes(daemon):
    d, c = daemon
    c.release()

    def fails():
        e = ConnectionError("feedback disagrees with the start pose")
        e.add_note("could not confirm torque-off on: joint5. Treat the arm as energised.")
        raise e
    d.k.body.enable = fails
    with pytest.raises(DaemonError) as e:
        c.enable()
    assert e.value.code == 502 and "Treat the arm as energised" in str(e.value)


def test_the_daemon_shuts_down_once(daemon):
    """A second Ctrl+C during the release ramp must not start another release."""
    d, c = daemon
    c.shutdown()
    with pytest.raises(Refused, match="already shutting down"):
        d.shutdown()



@pytest.mark.usefixtures("file_camera")
def test_cli_input_errors_and_rehearsals_have_their_exit_status(client, capsys):
    from world_use import cli
    url = ["--url", client.url]
    assert cli.main(url + ["look", "side", "--plan", '[{"do": "line", "forward": 0.40}]']) == 4
    assert cli.main(url + ["box", "tray", "surface", "0.3,0", "0.1,0.1,0.1"]) == 2
    assert cli.main(url + ["box", "tray"]) == 2
    assert cli.main(url + ["help", "teleport"]) == 2
    assert cli.main(["--url", unused_url(), "view"]) == 3
    assert "start it with: wu up" in capsys.readouterr().err
    assert cli.main(url + ["events", "--limit", "2"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3 and "wu events --since" in lines[-1]


def test_the_cli_exit_status_says_how_the_job_ended(client, capsys):
    """`wu run ... && wu home` carried on after a surprise on the real reBot and took a gripped roll of tape home."""
    from world_use import cli
    url = ["--url", client.url]
    assert cli.main(url + ["run", '[{"do": "line", "up": 0.03, "duration": 1.0}]']) == 0
    assert cli.main(url + ["run", '[{"do": "line", "forward": 0.40}]']) == 4                    # refused, unmoved
    assert cli.main(url + ["check", '[{"do": "line", "forward": 0.40}]']) == 4                  # would be refused
    assert cli.main(url + ["check", '[{"do": "line", "up": 0.02}]']) == 0
    assert cli.main(url + ["run", '[{"do": "checkpoint", "ask": "go on?"}]', "--wait", "5"]) == 5   # waits
    capsys.readouterr()
    assert cli.main(url + ["status", "--json"]) == 0 and '"line"' in capsys.readouterr().out


def test_the_flight_record_can_be_written_without_stopping(client):
    c = client
    c.run({"do": "line", "up": 0.02, "duration": 0.5}, wait=10)
    r = c.record()
    run = Path(r["run"])
    assert {"tape", "summary.json", "world.json"} <= {f.name for f in run.iterdir()}
    assert list((run / "tape").glob("[0-9]*.npz"))
    assert r["summary"]["moving_s"] > 0 and "idle" in c.status()["line"]          # still serving


@pytest.mark.usefixtures("file_camera")
def test_look_with_a_grid_draws_a_ruler_and_nothing_the_kernel_believes(client):
    from world_use import cli
    r = client.look("side", grid=True)
    assert "pixel grid every 100 px" in r["drawn"] and "magenta" not in r["drawn"]
    img = np.asarray(Image.open(r["path"]).convert("RGB")).astype(int)
    believed = client.look("side")
    assert "magenta" in believed["drawn"]
    assert (np.abs(img[:, 99:102] - img[:, 95:98]).sum() > 0)                   # a grid line at x = 100
    assert cli.main(["--url", client.url, "look", "side", "--grid"]) == 0
