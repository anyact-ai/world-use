"""The daemon and its client, end to end on a simulated arm in real time."""
import json
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import serving
from PIL import Image

from world_use import Client, Refused, cli
from world_use.behaviors import REGISTRY, build
from world_use.cameras import FileCamera, Frame
from world_use.client import DaemonError
from world_use.daemon import Daemon, session_identity
from world_use.events import EventLog

OUT_OF_REACH = [{"do": "line", "forward": 0.40}]


def unused_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


def test_a_run_waits_for_its_end_and_the_flight_record_is_written_without_stopping(client):
    assert client.status()["session"]["mode"] == "simulation" and "reBot" in client.card()
    r = client.run({"do": "line", "forward": 0.03, "up": 0.03, "duration": 1.0}, wait=10)
    assert r["status"] == "done" and "tool F" in r["line"]
    r = client.record()
    run = Path(r["run"])
    assert {"tape", "summary.json", "world.json"} <= {f.name for f in run.iterdir()}
    assert list((run / "tape").glob("[0-9]*.npz"))
    assert r["summary"]["moving_s"] > 0 and "idle" in client.status()["line"]          # still serving


def test_up_reuses_only_a_daemon_with_the_requested_adapter_and_workcell(daemon, capsys):
    d, c = daemon
    up = ["--url", c.url, "up"]
    c.release()
    assert cli.main(up + ["--body", "sim", "--enable"]) == 0 and c.status()["enabled"]
    assert cli.main(up + ["--body", "sim:rebot"]) == 0
    assert cli.main(up + ["--workcell", "block"]) == 2
    assert cli.main(up + ["--body", "rebot"]) == 2
    d.session = session_identity("rebot", {})           # as if a hardware daemon were running
    assert cli.main(up + ["--body", "sim", "--enable"]) == 2
    assert "requested sim:rebot" in capsys.readouterr().err


def test_up_refuses_a_robot_the_twin_cannot_model_and_leaves_no_daemon(tmp_path, capsys):
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
    spawned = []
    popen = subprocess.Popen

    def spawn(*args, **kwargs):
        spawned.append(popen(*args, **kwargs))
        return spawned[-1]
    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(cli, "UP_TIMEOUT_S", 0.2)                 # a cold start takes longer than this
    assert cli.main(["--url", unused_url(), "up", "--runs", str(tmp_path)]) == 1
    assert spawned[0].poll() is not None and "was stopped" in capsys.readouterr().err


def test_errors_reach_the_client_with_their_code_and_notes(daemon):
    d, c = daemon
    with pytest.raises(DaemonError, match="unknown behavior") as e:
        c.run({"do": "teleport"})
    assert e.value.code == 409

    def bug():
        raise AttributeError("a bug")
    d.k.reset = bug
    with pytest.raises(DaemonError, match="a bug") as e:
        c.reset()
    assert e.value.code == 500 and "idle" in c.status()["line"]          # answered, and still serving

    def unconfirmed():
        e = ConnectionError("feedback disagrees with the start pose")
        e.add_note("could not confirm torque-off on: joint5. Treat the arm as energised.")
        raise e
    c.release()
    d.k.body.enable = unconfirmed
    with pytest.raises(DaemonError, match="Treat the arm as energised") as e:
        c.enable()
    assert e.value.code == 502


@pytest.mark.usefixtures("file_camera")
def test_the_cli_exit_status_says_how_the_job_ended(client, tmp_path, capsys):
    """`wu run ... && wu home` carried on after a surprise on the real reBot and took a gripped roll of tape home."""
    url = ["--url", client.url]
    for args, code, message in [
        (["run", '[{"do": "line", "up": 0.03, "duration": 1.0}]'], 0, ""),
        (["run", json.dumps(OUT_OF_REACH)], 4, ""),                       # refused, unmoved
        (["check", json.dumps(OUT_OF_REACH)], 4, ""),                     # would be refused
        (["check", '[{"do": "line", "up": 0.02}]'], 0, ""),
        (["look", "side", "--plan", json.dumps(OUT_OF_REACH)], 4, ""),
        (["run", "[{"], 2, "invalid JSON"),
        (["check", str(tmp_path / "missing.json")], 2, "cannot read JSON file"),
        (["inspect", str(tmp_path / "missing")], 2, "flight record"),
        (["replay", str(tmp_path), "--speed", "0"], 2, "speed"),
        (["box", "tray", "surface", "0.3,0", "0.1,0.1,0.1"], 2, ""),
        (["box", "tray"], 2, ""),
        (["help", "teleport"], 2, ""),
        (["run", '[{"do": "checkpoint", "ask": "go on?"}]', "--wait", "5"], 5, ""),   # waits for an answer
    ]:
        assert cli.main(url + args) == code, args
        assert message in capsys.readouterr().err
    assert cli.main(["--url", unused_url(), "view"]) == 3
    assert "start it with: wu up" in capsys.readouterr().err
    for command, key in [("world", "frames"), ("events", "events"), ("look", "path"), ("status", "line")]:
        assert cli.main(url + [command, "--json"]) == 0
        assert key in json.loads(capsys.readouterr().out)
    assert cli.main(url + ["--json", "run", '{"do": "teleport"}']) == 2      # a refusal parses as JSON too
    assert "unknown behavior" in json.dumps(json.loads(capsys.readouterr().out))
    assert cli.main(url + ["events", "--limit", "2"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3 and "wu events --since" in lines[-1]


AGENT = '''
import sys
from world_use import Client
plan = [dict(do="line", up=.03, duration=3)]
if sys.argv[2] == "checkpoint":
    plan += [dict(do="checkpoint", ask="continue?"), dict(do="line", up=.02)]
Client(sys.argv[1]).run(plan, wait=30)
sys.stdin.read()  # the agent remains alive while deciding how to answer
'''


@pytest.mark.parametrize("disconnect_at", ["motion", "checkpoint"])
def test_losing_the_client_process_neither_stops_nor_replays_its_job(client, disconnect_at):
    c = client
    before = c.status()["tool"]["work"]
    cursor = c.events()["last"]
    process = subprocess.Popen([sys.executable, "-c", AGENT, c.url, disconnect_at], stdin=subprocess.PIPE)
    job_id = None
    try:
        deadline = time.monotonic() + 15
        while True:
            assert time.monotonic() < deadline, f"never reached {disconnect_at}: {c.status()}"
            assert process.poll() is None, "the client exited before the interruption"
            batch = c.events(since=cursor, wait=.05)
            cursor = batch["last"]
            job_id = next((e["data"]["job"] for e in batch["events"] if e["kind"] == "submitted"), job_id)
            if job_id is None:
                continue
            status, position = c.job(job_id)["status"], c.status()["tool"]["work"]
            if disconnect_at == "checkpoint":
                if status == "waiting":
                    break
            elif status == "running" and .002 < position[2] - before[2] < .025:
                break
        process.kill()
        process.wait(timeout=5)

        # A new client knows the job ID: it queries the job and does not resubmit the plan.
        assert c.status()["enabled"]
        if disconnect_at == "checkpoint":
            time.sleep(.2)
            waiting = c.job(job_id)
            assert waiting["status"] == "waiting" and waiting["question"]["ask"] == "continue?"
            assert c.status()["tool"]["work"] == pytest.approx(position, abs=.002)
            result = c.answer(job_id, "yes", wait=10)
        else:
            result = c.job(job_id, wait=10)
        assert result["status"] == "done", result
        rise = .05 if disconnect_at == "checkpoint" else .03
        assert np.subtract(c.status()["tool"]["work"], before) == pytest.approx([0, 0, rise], abs=.003)
        log = c.events()["events"]
        for kind in ("submitted", "started", "finished"):
            assert [e["data"]["job"] for e in log if e["kind"] == kind] == [job_id]
        assert sum(e["kind"] == "enabled" for e in log) == 1 and not c.status()["faulted"]
        c.home_route([])
        assert c.home(wait=15)["status"] == "done"
        c.release()
        status = c.status()
        assert not status["enabled"] and not status["power_uncertain"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdin.close()


def test_a_home_route_is_rehearsed_before_it_replaces_the_installed_one(daemon, capsys):
    d, c = daemon
    c.release()                             # a return can be prepared without powering the robot
    with pytest.raises(DaemonError, match="needs steps"):
        c._call("POST", "/home_route", {"note": "no steps"})
    with pytest.raises(DaemonError, match="not allowed in a home route"):
        c.home_route([{"do": "checkpoint", "ask": "clear?"}])
    assert c.home_route([])["home"].startswith("set")
    installed = d.k.home_route
    candidate = [{"do": "line", "up": 0.02}, *OUT_OF_REACH]
    r = c.home_route(candidate)
    assert not r["ok"] and d.k.home_route == installed
    capsys.readouterr()
    assert cli.main(["--url", c.url, "home-route", json.dumps(candidate)]) == 4
    assert r["text"].splitlines()[0] in capsys.readouterr().out
    assert cli.main(["--url", c.url, "home-route", "null"]) == 0
    assert "not available" in c.status()["home"]
    assert not c.home_route(candidate)["ok"] and d.k.home_route is None


def test_home_candidate_must_still_be_current_after_rehearsal(daemon, monkeypatch):
    d, c = daemon
    assert c.home_route([])["ok"]
    installed = d.k.home_route
    check = d.rehearser.check

    def changed(*args, **kwargs):
        report = check(*args, **kwargs)
        d.k.stop("scene changed while checking the way home")
        return report

    monkeypatch.setattr(d.rehearser, "check", changed)
    with pytest.raises(DaemonError, match="changed during rehearsal"):
        c.home_route([])
    assert d.k.home_route == installed


def test_check_does_not_move_the_robot(daemon):
    d, client = daemon
    before = d.k.cmd.q.copy()
    r = client.check({"do": "line", "up": 0.05})
    assert r["ok"] and "check passed" in r["text"]
    np.testing.assert_array_equal(d.k.cmd.q, before)


def test_frame_returns_native_pixels_without_creating_records(daemon, tmp_path):
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
    time.sleep(0.5)
    client.stop("test")
    assert client.job(r["id"], wait=2)["status"] == "stopped"


def test_event_poll_reports_a_missed_window_and_advances_only_through_returned_events():
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
    r = c.run([{"do": "line", "up": 0.03}, {"do": "line", "left": 0.05}, {"do": "joints", "delta_deg": {"1": 130}}],
              wait=5)
    assert r["status"] == "refused" and r["id"] is None
    assert "step 2/3" in r["incident"] and "step 3/3" in r["incident"] and "from here" in r["incident"]
    assert not d.k.jobs                                                 # not even the first step was submitted
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


def stop_control_loop(d):
    """The test ticks the kernel itself."""
    d.stop_loop.set()
    d.control.join(2)
    assert not d.control.is_alive()


def test_checked_run_revalidates_before_its_first_tick(daemon):
    d, _ = daemon
    stop_control_loop(d)
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
    stop_control_loop(d)
    original = d.rehearser.check

    def stopped(*args, **kwargs):
        report = original(*args, **kwargs)
        c.stop()
        d.k.tick()
        assert d.k._stop is None                  # the tick consumed the stop
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
    with pytest.raises(DaemonError, match="cameras: side, front, top"):
        c.look("nowhere")


@pytest.mark.usefixtures("file_camera")
def test_look_with_a_grid_draws_a_ruler_and_nothing_the_kernel_believes(client):
    r = client.look("side", grid=True)
    assert "pixel grid every 100 px" in r["drawn"] and "magenta" not in r["drawn"]
    img = np.asarray(Image.open(r["path"]).convert("RGB")).astype(int)
    assert (np.abs(img[:, 99:102] - img[:, 95:98]).sum() > 0)                   # a grid line at x = 100
    assert "magenta" in client.look("side")["drawn"]
    assert cli.main(["--url", client.url, "look", "side", "--grid"]) == 0


def test_policy_boxes_go_only_into_the_model_and_unknown_workcell_boxes_only_into_the_simulation(tmp_path,
                                                                                                  rehearser):
    cell = {"box": [dict(name="shelf", kind="surface", center=[0.3, 0, 0.1], size=[0.2, 0.2, 0.02], known=False)]}
    with serving(tmp_path, cell, rehearser) as (d, c):
        assert "shelf" in d.k.body.world.boxes and "shelf" not in d.k.world.boxes
        line = c.box("tray", "surface", [0.32, 0.0, 0.14], [0.3, 0.4, 0.02], source="side camera")["line"]
        assert line.startswith("surface 'tray': centre F+0.320 L+0.000 U+0.140") and "from side camera" in line
        assert "tray" in d.k.world.boxes and "tray" not in d.k.body.world.boxes
        assert "surface 'tray'" in c.world()["text"] and "surface 'tray'" in c.card()
        c.remove("tray")
        assert "tray" not in d.k.world.boxes


def test_help_lists_every_step_with_an_example(client):
    steps = client.help()
    assert set(steps) == set(REGISTRY)
    for kind, h in steps.items():
        assert h["summary"] and h["example"]["do"] == kind
        build(h["example"])


def test_a_failed_shutdown_keeps_the_session_for_power_recovery_and_a_shutdown_runs_once(daemon, monkeypatch):
    d, c = daemon
    disable = d.k.body.disable

    def failed_release():
        raise OSError("could not confirm power off")

    monkeypatch.setattr(d.k.body, "disable", failed_release)
    with pytest.raises(DaemonError, match="could not confirm power off"):
        c.shutdown()
    assert d.control.is_alive() and c.status()["power_uncertain"]
    monkeypatch.setattr(d.k.body, "disable", disable)
    c.release()
    c.reset()
    c.enable()
    assert c.status()["enabled"]
    c.shutdown()
    assert not d.control.is_alive() and not d.k.body.enabled
    with pytest.raises(Refused, match="already shutting down"):     # a second Ctrl+C during the release ramp
        d.shutdown()


def test_shutdown_refuses_an_enable_already_queued_during_release(daemon, monkeypatch):
    d, _ = daemon
    releasing, finish_release, queued = threading.Event(), threading.Event(), threading.Event()
    disable = d.k.body.disable

    def blocked_disable():
        releasing.set()
        assert finish_release.wait(5)
        disable()

    class NotifyingQueue(queue.SimpleQueue):
        def put(self, item):
            super().put(item)
            if item[0] == d.k._enable:
                queued.set()

    monkeypatch.setattr(d.k.body, "disable", blocked_disable)
    monkeypatch.setattr(d.k, "_posted", NotifyingQueue())
    with ThreadPoolExecutor(2) as pool:
        shutdown = pool.submit(d.shutdown)
        try:
            assert releasing.wait(5)
            enable = pool.submit(d.k.enable)
            assert queued.wait(5)
        finally:
            finish_release.set()
        shutdown.result(5)
        with pytest.raises((Refused, RuntimeError), match=r"shutting down|control loop.*stopped"):
            enable.result(5)
    assert not d.control.is_alive() and not d.k.enabled and not d.k.body.enabled
