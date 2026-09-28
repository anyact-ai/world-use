"""The reBot adapter against a fake motorbridge driver: what it sends, and what it never sends."""
import sys
import types

import numpy as np
import pytest
from conftest import Q_REST

from world_use import Kernel, Refused, VirtualClock, World
from world_use.bodies import rebot
from world_use.kinematics import Chain

CHAIN = Chain(rebot.MANIFEST.urdf, rebot.MANIFEST.tool_link)


class FakeBus:
    def __init__(self, start):
        self.q = np.asarray(start, float).copy()          # 7 motor angles as the motors report them
        self.enables = self.disables = self.frames = 0
        self.closed_bus = False
        self.rng = np.random.default_rng(0)


class FakeMotor:
    def __init__(self, bus, i):
        self.bus, self.i, self.on, self.last = bus, i, False, None

    def robstride_get_param_f32(self, param, timeout):
        return float(self.bus.q[self.i]) if param == rebot.P_MECH_POS else 48.0

    def robstride_get_param_u8(self, param, timeout):
        return 1

    def robstride_get_fault_report(self):
        return 0, 0

    def ensure_mode(self, mode, timeout):
        pass

    def enable(self):
        self.on = True
        self.bus.enables += 1

    def disable(self):
        self.on = False
        self.bus.disables += 1

    def send_mit(self, pos, vel, kp, kd, tau):
        self.bus.frames += 1
        self.last = (pos, vel, kp, kd, tau)
        if kp > 0:
            self.bus.q[self.i] = pos                       # perfect tracking

    def get_state(self):
        g = np.append(CHAIN.gravity(self.bus.q[:6]), 0.0)
        return types.SimpleNamespace(pos=float(self.bus.q[self.i]), vel=0.0,
                                     torq=float(g[self.i] + self.bus.rng.normal(0, 0.01)), t_mos=35.0,
                                     status_code=0, arbitration_id=(2 if self.on else 0) << 22)


@pytest.fixture
def fake(monkeypatch):
    holder = {}

    class Controller:
        def __init__(self, channel):
            self.bus = holder["bus"]

        def add_robstride_motor(self, mid, host, model):
            return FakeMotor(self.bus, mid - 1)

        def poll_feedback_once(self):
            pass

        def close_bus(self):
            self.bus.closed_bus = True

        def close(self):
            pass

    mb = types.ModuleType("motorbridge")
    core, errors, models = (types.ModuleType(f"motorbridge.{n}") for n in ("core", "errors", "models"))
    core.Controller = Controller
    errors.CallError = type("CallError", (Exception,), {})
    models.Mode = types.SimpleNamespace(MIT="mit")
    for name, mod in (("motorbridge", mb), ("motorbridge.core", core), ("motorbridge.errors", errors),
                      ("motorbridge.models", models)):
        monkeypatch.setitem(sys.modules, name, mod)

    def make(start):
        holder["bus"] = FakeBus(start)
        return holder["bus"]
    return make


def test_connect_is_read_only_and_unwraps_a_gripper_parked_open(fake):
    bus = fake(np.append(Q_REST, -3.11))                  # parked open at 3.17 rad, came back one turn low
    body = rebot.ReBotBody()
    st = body.connect()
    assert np.allclose(st.q, Q_REST) and abs(st.gripper - (2 * np.pi - 3.11)) < 1e-9
    assert any("gripper" in w for w in body.warnings)
    body.close()
    assert bus.enables == bus.disables == bus.frames == 0 and bus.closed_bus


def test_enable_away_from_rest_is_refused_and_sends_nothing(fake):
    raised = Q_REST.copy()
    raised[1:4] = [0.8, 0.3, 0.4]
    bus = fake(np.append(raised, 1.0))
    body = rebot.ReBotBody()
    body.connect()
    with pytest.raises(Refused):
        body.enable()
    assert bus.enables == bus.frames == 0


def test_enable_ramps_gains_in_from_nothing(fake):
    bus = fake(np.append(Q_REST, 1.0))
    body = rebot.ReBotBody()
    body.connect()
    kps = []
    orig = FakeMotor.send_mit

    def spy(self, pos, vel, kp, kd, tau):
        if self.i == 1:
            kps.append(kp)
        return orig(self, pos, vel, kp, kd, tau)
    FakeMotor.send_mit = spy
    try:
        body.enable()
    finally:
        FakeMotor.send_mit = orig
    assert kps[0] < 0.05 * 150 and abs(kps[-1] - 150) < 1e-9
    assert bus.enables == 7


def test_a_bad_value_sends_nothing_to_any_motor(fake):
    bus = fake(np.append(Q_REST, 1.0))
    body = rebot.ReBotBody()
    body.connect()
    body.enable()
    frames = bus.frames
    q = Q_REST.copy()
    q[2] = np.nan
    with pytest.raises(ValueError):
        body.command(q, np.zeros(6), 1.0)
    assert bus.frames == frames


def test_kernel_session_on_the_adapter_moves_folds_and_releases(fake):
    bus = fake(np.append(Q_REST, 1.0))
    world = World()
    k = Kernel(rebot.ReBotBody(), world, VirtualClock(100.0))
    k.connect()
    k.enable()
    assert k.run({"do": "line", "forward": 0.05, "up": 0.05}).ok
    assert "work" in world.frames
    k.set_home_route([])
    assert k.run({"do": "seq", "steps": k.home_plan()}).ok
    k.release()
    assert bus.disables == 7 and not k.enabled
    k.close()
    assert bus.disables == 7                              # closing sends no further disable frames


def test_a_failed_engage_switches_every_motor_off_again(fake, monkeypatch):
    bus = fake(np.append(Q_REST, 1.0))
    body = rebot.ReBotBody()
    body.connect()
    alive = FakeMotor.get_state

    def get_state(self):                                  # the elbow goes quiet once switched on
        return None if (self.i == 2 and self.on) else alive(self)
    monkeypatch.setattr(FakeMotor, "get_state", get_state)
    with pytest.raises(ConnectionError, match="disagrees with the start pose"):
        body.enable()
    assert bus.enables == 7 and bus.disables == 7 and not body.enabled


def test_an_unconfirmed_switch_off_says_so(fake, monkeypatch):
    fake(np.append(Q_REST, 1.0))
    body = rebot.ReBotBody()
    body.connect()
    alive, off = FakeMotor.get_state, FakeMotor.disable
    monkeypatch.setattr(FakeMotor, "get_state", lambda self: None if (self.i == 2 and self.on) else alive(self))
    monkeypatch.setattr(FakeMotor, "disable", lambda self: None if self.i == 4 else off(self))   # j5 stays on
    with pytest.raises(ConnectionError) as e:
        body.enable()
    assert "joint5" in " ".join(e.value.__notes__) and "energised" in " ".join(e.value.__notes__)
    assert body.enabled                                   # counted as on: nothing was confirmed off for j5
