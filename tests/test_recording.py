import json

import numpy as np
import pytest

from world_use import Kernel, VirtualClock, World, bodies
from world_use.recorder import Tape


def test_elapsed_durations_include_slow_control_ticks():
    tape = Tape(1)
    for t, moving in [(0, True), (0.01, True), (0.51, False), (1.01, False)]:
        tape.add(t, True, moving, 1, [0], [0], None, None, None, None, None)
    s = tape.summary(100)
    assert s["powered_s"] == 1.0 and s["moving_s"] == 0.5
    assert s["moving_share"] == pytest.approx(0.505, abs=0.001)
    assert s["tick_ms"]["max"] == 500


def test_events_tape_and_enable_ramp_share_one_elapsed_clock(tmp_path):
    clock = VirtualClock(100)
    world = World()
    body = bodies.make("sim", world)
    k = Kernel(body, world, clock, run_dir=tmp_path)
    connect, enable, disable = body.connect, body.enable, body.disable

    def slow_connect():
        clock.t += 2
        return connect()

    def slow_enable():
        clock.t += 1
        enable()

    def slow_disable():
        clock.t += 0.5
        disable()

    body.connect, body.enable, body.disable = slow_connect, slow_enable, slow_disable
    k.connect()
    k.enable()
    k.tick()
    clock.t += 1
    k.tick()
    k.emit("contact", "same timeline as tape")
    k.release()
    summary = k.save_record()
    assert summary["powered_s"] == 2.5 and summary["wall_s"] == 4.5
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    tape = np.load(tmp_path / "tape.npz")
    assert events[0]["t"] == 2 and tape["t"].tolist() == [3, 4]
    assert next(e["t"] for e in events if e["kind"] == "contact") == tape["t"][-1]
    assert tape["power_t"].tolist() == [0, 2, 4.5]
    k.close()


def test_failed_enable_counts_unconfirmed_power_even_without_ticks(k, monkeypatch):
    k.release()

    def partial_enable():
        k.clock.t += 0.5
        raise OSError("unconfirmed power")

    monkeypatch.setattr(k.body, "enable", partial_enable)
    with pytest.raises(OSError):
        k.enable()
    k.clock.t += 2
    s = k.save_record()
    assert s["ticks"] == 0 and s["powered_s"] == 2.5
    assert s["power_basis"] == "enable_attempt_to_confirmed_disable"
