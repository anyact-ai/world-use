"""Public JSON requests refuse invalid input before committing a mutation."""
from contextlib import closing
from http.client import HTTPConnection
from urllib.parse import urlparse

import numpy as np
import pytest

from world_use.client import DaemonError


def test_invalid_run_requests_never_submit_a_job(daemon):
    d, c = daemon
    before = d.k.cmd.q.copy()
    for change in [dict(require=[{"evidence": "missing", "max_age_s": 10}]), dict(check="false"),
                   dict(wait=float("nan")), dict(wait=True),
                   dict(requires=[{"evidence": "missing", "max_age_s": True}]),
                   dict(requires=[{"evidence": "missing", "max_age_s": 10, "typo": 1}])]:
        with pytest.raises(DaemonError) as error:
            c._call("POST", "/run", dict(spec={"do": "hold", "seconds": 0.01}) | change)
        assert error.value.code == 400, change
    with pytest.raises(DaemonError) as error:          # the client must not drop a falsy requirement
        c.run({"do": "hold", "seconds": 0.01}, requires=False)
    assert error.value.code == 400
    assert not d.k.jobs
    np.testing.assert_array_equal(d.k.cmd.q, before)


def test_a_bad_enable_request_cannot_change_power(client):
    """Foreign origins and hosts, non-JSON bodies and unknown arguments."""
    client.release()
    port = urlparse(client.url).port
    json_type = {"Content-Type": "application/json"}
    for path, headers, body, code, reason in [
        ("/enable", {"Origin": "https://example.invalid"} | json_type, "{}", 403, ""),
        ("/enable", {"Host": f"example.invalid:{port}"} | json_type, "{}", 403, ""),
        ("/enable", {"Content-Type": "text/plain"}, "{}", 415, ""),
        ("/enable", json_type, "[]", 400, ""),
        ("/enable", json_type, '{"typo": true}', 400, "typo"),
        ("/enable?typo=", json_type, "{}", 400, "typo"),
        ("/enable?typo", json_type, "{}", 400, "typo"),
    ]:
        with closing(HTTPConnection("127.0.0.1", port)) as conn:
            conn.request("POST", path, body, headers)
            response = conn.getresponse()
            assert response.status == code and reason in response.read().decode(), (path, headers, body)
        assert not client.status()["enabled"]
    client.enable()
    assert client.status()["enabled"]


def test_a_world_change_applies_whole_or_not_at_all(daemon):
    d, c = daemon
    c.box("glass", "fragile", [0.8, 0, 0.5], [0.1] * 3, dtau=0.1)
    glass = d.k.world.boxes["glass"]
    for change, field in [
        (dict(box=dict(name="tray", kind="surface", center=[0.3, 0], size=[.1, .1, .1])), "box.center"),
        (dict(box=dict(name="glass", kind="fragile", center=[0.8, 0, 0.5], size=[0.2] * 3, datu=0.03)), "datu"),
        (dict(frame=[]), "frame"),
        (dict(frame=dict(name="tilted", yaw_deg=30)), "yaw_deg"),
    ]:
        with pytest.raises(DaemonError, match=field) as error:
            c.world(fact=dict(key="seen", value=True), **change)
        assert error.value.code == 400
    world = c.world()
    assert "seen" not in world["facts"] and "tray" not in world["boxes"] and "tilted" not in world["frames"]
    assert d.k.world.boxes["glass"] is glass
    line = c.world(fact=dict(key="table.z", value=0.205, source="touchdown"))["line"]
    assert "table.z = 0.205 (from touchdown)" in line and c.status()["facts"]["table.z"] == 0.205


def test_answer_rejects_a_string_job_id_without_advancing_the_checkpoint(daemon):
    d, c = daemon
    job = c.run({"do": "checkpoint", "ask": "continue?"}, wait=5, check=False)["id"]
    with pytest.raises(DaemonError) as error:
        c._call("POST", "/answer", dict(job=str(job), answer="yes"))
    assert error.value.code == 400 and d.k.jobs[job].status == "waiting"
    assert c.answer(job, "yes", wait=5)["status"] == "done"


@pytest.mark.usefixtures("file_camera")
def test_request_values_are_never_coerced_and_queries_are_decoded_as_text(daemon):
    d, c = daemon
    assert c._call("GET", "/events?since=0&wait=0.0&limit=1")["events"]
    assert c._call("GET", "/frame?camera=side&depth=false")["camera"] == "side"
    for method, path, body in [("GET", "/events?wait=NaN", None), ("GET", "/frame?depth=yes", None),
                               ("POST", "/record", dict(context={"value": float("nan")})),
                               ("POST", "/measure", dict(frame="missing", point=[True, 0]))]:
        with pytest.raises(DaemonError) as error:
            c._call(method, path, body)
        assert error.value.code == 400, path
    assert not d.k.jobs
