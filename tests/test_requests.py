"""Public JSON requests refuse invalid input before committing a mutation."""
import numpy as np
import pytest

from world_use.client import DaemonError


@pytest.mark.parametrize("change", [dict(require=[{"evidence": "missing", "max_age_s": 10}]),
                                    dict(check=None), dict(check="false"), dict(check=1),
                                    dict(wait=float("nan")), dict(wait=True), dict(wait="0"),
                                    dict(requires=False),
                                    dict(requires=[{"evidence": "missing", "max_age_s": True}]),
                                    dict(requires=[{"evidence": "missing", "max_age_s": 10, "typo": 1}])])
def test_invalid_run_requests_never_submit_a_job(daemon, change):
    d, c = daemon
    before = d.k.cmd.q.copy()
    with pytest.raises(DaemonError) as error:
        c._call("POST", "/run", dict(spec={"do": "hold", "seconds": 0.01}) | change)
    assert error.value.code == 400
    assert not d.k.jobs
    np.testing.assert_array_equal(d.k.cmd.q, before)


def test_client_cannot_discard_an_invalid_falsy_requirement(daemon):
    d, c = daemon
    with pytest.raises(DaemonError) as error:
        c.run({"do": "hold", "seconds": 0.01}, requires=False)
    assert error.value.code == 400 and not d.k.jobs


def test_unknown_enable_argument_cannot_change_power(client):
    client.release()
    with pytest.raises(DaemonError, match="typo") as error:
        client._call("POST", "/enable", {"typo": True})
    assert error.value.code == 400 and not client.status()["enabled"]


def test_invalid_world_request_is_atomic(daemon):
    d, c = daemon
    c.box("glass", "fragile", [0.8, 0, 0.5], [0.1] * 3, dtau=0.1)
    original = d.k.world.boxes["glass"]
    with pytest.raises(DaemonError, match="datu") as error:
        c.world(fact=dict(key="new", value="uncommitted"),
                box=dict(name="glass", kind="fragile", center=[0.8, 0, 0.5], size=[0.2] * 3, datu=0.03))
    assert error.value.code == 400
    assert d.k.world.boxes["glass"] is original and "new" not in d.k.world.facts


def test_answer_rejects_a_string_job_id_without_advancing_the_checkpoint(daemon):
    d, c = daemon
    job = c.run({"do": "checkpoint", "ask": "continue?"}, wait=5, check=False)["id"]
    with pytest.raises(DaemonError) as error:
        c._call("POST", "/answer", dict(job=str(job), answer="yes"))
    assert error.value.code == 400 and d.k.jobs[job].status == "waiting"
    assert c.answer(job, "yes", wait=5)["status"] == "done"


@pytest.mark.parametrize(("path", "body"), [
    ("/look", dict(grid="false")),
    ("/stop", dict(reason=True)),
    ("/calibrate", dict(camera="side", points="8")),
    ("/withdraw", dict(measurements=[], reason=False)),
    ("/record", dict(context={"value": float("nan")})),
    ("/measure", dict(frame="missing", point=[True, 0])),
    ("/home_route", dict(steps=None, note=1)),
])
def test_other_mutations_reject_coercion_and_nonfinite_data(client, path, body):
    with pytest.raises(DaemonError) as error:
        client._call("POST", path, body)
    assert error.value.code == 400


@pytest.mark.usefixtures("file_camera")
def test_url_query_values_are_decoded_as_text(daemon):
    d, c = daemon
    assert c._call("GET", "/events?since=0&wait=0.0&limit=1")["events"]
    assert c._call("GET", "/frame?camera=side&depth=false")["camera"] == "side"
    for path in ("/events?wait=NaN", "/frame?depth=yes", "/frame?depth=garbage"):
        with pytest.raises(DaemonError) as error:
            c._call("GET", path)
        assert error.value.code == 400
    assert not d.k.jobs
