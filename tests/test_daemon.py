"""The daemon and its client, end to end on a simulated arm in real time."""
import threading
from pathlib import Path

import numpy as np
import pytest
from conftest import serve
from PIL import Image

from world_use import Refused
from world_use.client import DaemonError


def test_status_card_and_a_run_with_wait(client):
    assert "idle" in client.status()["line"]
    assert "reBot" in client.card()
    r = client.run({"do": "line", "forward": 0.03, "up": 0.03, "duration": 1.0}, wait=10)
    assert r["status"] == "done" and "tool F" in r["line"]


def test_refusal_is_reported_as_an_incident(client):
    r = client.run({"do": "line", "forward": 0.40}, wait=5)
    assert r["status"] == "refused" and "incident" in r


def test_malformed_spec_is_a_409_with_the_reason(client):
    with pytest.raises(DaemonError) as e:
        client.run({"do": "teleport"})
    assert e.value.code == 409 and "unknown behavior" in str(e.value)


def test_checkpoint_round_trip(client):
    client.run({"do": "line", "up": 0.05, "duration": 1.0}, wait=10)
    r = client.run([{"do": "checkpoint", "ask": "clear to go on?"}, {"do": "line", "up": 0.01}], wait=5)
    assert r["status"] == "waiting" and r["question"]["ask"] == "clear to go on?"
    r = client.answer(r["id"], "yes", wait=10)
    assert r["status"] == "done"


def test_check_does_not_move_the_robot(client):
    before = client.status()["joints_deg"]
    r = client.check({"do": "line", "up": 0.05})
    assert r["ok"] and "check passed" in r["text"]
    assert np.allclose(client.status()["joints_deg"], before, atol=0.01)


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


def test_run_rehearses_and_refuses_the_whole_plan_with_every_problem_before_anything_moves(daemon):
    d, c = daemon
    before = c.status()["joints_deg"]
    jobs = len(d.k.jobs)
    r = c.run([{"do": "line", "up": 0.03}, {"do": "line", "left": 0.05}, {"do": "joints", "delta_deg": {"1": 130}}],
              wait=5)
    assert r["status"] == "refused" and r["id"] is None
    assert "step 2/3" in r["incident"] and "step 3/3" in r["incident"] and "from here" in r["incident"]
    assert len(d.k.jobs) == jobs                                        # not even the first step was submitted
    assert np.allclose(c.status()["joints_deg"], before, atol=0.01)


def test_run_without_the_rehearsal_is_refused_by_the_kernel_at_the_step(client):
    r = client.run([{"do": "line", "up": 0.03}, {"do": "line", "left": 0.05}], wait=10, check=False)
    assert r["status"] == "refused" and "step 2/2" in r["incident"]


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


def test_a_workcell_box_marked_unknown_is_only_in_the_simulation(tmp_path):
    cell = {"box": [dict(name="shelf", kind="surface", center=[0.3, 0, 0.1], size=[0.2, 0.2, 0.02], known=False)]}
    d, _ = serve(tmp_path, cell)
    try:
        assert "shelf" in d.k.body.world.boxes and "shelf" not in d.k.world.boxes
    finally:
        d.stop_loop.set()
        d.http.shutdown()


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
