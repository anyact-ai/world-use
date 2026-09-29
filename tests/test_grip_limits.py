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


def test_contact_squeeze_obeys_position_and_speed_limits(lifted):
    k = lifted
    k.run({"do": "gripper", "to": 0.4})
    g = k.manifest.gripper
    p = k.tool[:3, 3]
    k.world.add_box("thin", "object", center=p, size=[0.04, 0.001, 0.04], frame="base", grip_width=0.001)
    first = len(k.tape)
    out = k.run({"do": "grip", "speed": g.v_max})
    assert out.ok
    commands = k.tape.arrays()["grip_cmd"][first - 1:]
    assert commands.min() >= g.closed and commands.max() <= g.open
    assert np.abs(np.diff(commands)).max() * k.manifest.rate_hz <= g.v_max + 1e-9
