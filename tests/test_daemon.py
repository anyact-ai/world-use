"""The daemon and its client, end to end on a simulated arm in real time."""
import threading

import numpy as np
import pytest

from conftest import Q_REST
from world_use import Kernel, RealClock, World, bodies
from world_use.client import Client, DaemonError
from world_use.daemon import Daemon


@pytest.fixture
def client():
    world = World()
    k = Kernel(bodies.make("sim", world, q=Q_REST, gripper=1.0), world, RealClock(100.0))
    k.connect()
    k.enable()
    d = Daemon(k, port=0)                       # port 0: any free port
    d.start()
    c = Client(f"http://127.0.0.1:{d.http.server_address[1]}")
    yield c
    d.stop_loop.set()
    d.http.shutdown()


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
