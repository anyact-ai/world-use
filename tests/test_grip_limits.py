import numpy as np
import pytest


@pytest.mark.parametrize("params", [
    {"speed": 100}, {"speed": 0}, {"speed": -1}, {"speed": float("nan")},
    {"min": -1}, {"min": 2}, {"min": float("inf")},
    {"squeeze": 10}, {"squeeze": -0.1}, {"squeeze": float("nan")},
    {"effort": 100}, {"lag": 10}, {"start": 5}, {"start": float("nan")},
    {"expect": 2}, {"expect": [0.5, float("nan")]},
])
def test_invalid_grip_parameters_never_move_the_gripper(k, params):
    before = k.cmd.gripper
    out = k.run({"do": "grip", **params})
    assert out.status == "refused" and not k.faulted
    assert np.all(k.tape.arrays()["grip_cmd"] == before)


def test_contact_squeeze_obeys_position_and_speed_limits(lifted, monkeypatch):
    k = lifted
    k.run({"do": "gripper", "to": 0.4})
    g = k.manifest.gripper
    from dataclasses import replace
    read = k.body.read
    # A repeatable contact measurement close to the lower limit isolates command bounding.
    monkeypatch.setattr(k.body, "read", lambda: replace(read(), gripper_tau=.7 if k.cmd.gripper < .12 else 0))
    first = len(k.tape)
    out = k.run({"do": "grip", "speed": g.v_max})
    assert out.ok
    commands = k.tape.arrays()["grip_cmd"][first - 1:]
    assert commands.min() >= g.closed and commands.max() <= g.open
    assert np.abs(np.diff(commands)).max() * k.manifest.rate_hz <= g.v_max + 1e-9


def test_grip_reads_contact_after_issuing_the_final_close_command(lifted, monkeypatch):
    from dataclasses import replace

    k = lifted
    assert k.run({"do": "gripper", "to": .15}).ok
    read = k.body.read
    closed = k.manifest.gripper.closed
    monkeypatch.setattr(k.body, "read", lambda: replace(read(), gripper_tau=.7 if k.cmd.gripper == closed else 0))
    assert k.run({"do": "grip"}).ok
    assert any(e["kind"] == "grip" for e in k.events.since(0))
