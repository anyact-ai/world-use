import json

import numpy as np
import pytest

from world_use import Kernel, VirtualClock, World, bodies
from world_use.recorder import Journal, Tape, load_tape


def test_recording_failure_does_not_interrupt_motion_or_release(tmp_path, monkeypatch):
    from world_use import recorder

    world = World()
    k = Kernel(bodies.make("sim", world), world, VirtualClock(100), run_dir=tmp_path)
    k.connect()
    k.enable()

    def full_disk(*args, **kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(recorder, "open", full_disk, raising=False)
        with pytest.raises(OSError, match="disk full"):
            k.journal.flush()
        assert k.run({"do": "line", "up": .03}).ok
        assert not k.faulted and not k.power_uncertain
        assert k.run({"do": "line", "up": -.03}).ok
        k.release()
        assert not k.enabled
        assert k.journal.error == "disk full"
        assert k.close()["recording_error"] == "disk full"
    k.save_record()
    log = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [e["seq"] for e in log] == list(range(1, k.events.seq + 1))
    assert log[-1]["kind"] == "closed"


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
    tape = load_tape(tmp_path)
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


def test_record_captures_plans_answers_outcomes_and_startup(daemon):
    from world_use.records import inspect

    d, c = daemon
    spec = [{"do": "checkpoint", "ask": "block visible?"}, {"do": "hold", "seconds": .1}]
    r = c.run(spec, wait=5)
    c.answer(r["id"], "yes", wait=5)
    c.record(context={"model": "test-policy", "input": "place the block"}, note="operator checked the scene")
    record = inspect(d.k.run_dir)
    assert record["session"]["initial"]["q_start"]
    assert len(record["session"]["source_sha256"]) == 64
    assert record["jobs"][1]["spec"]["steps"] == spec
    assert record["jobs"][1]["outcome"]["status"] == "done"
    assert {e["kind"] for e in record["observations"]} == {"answer", "annotation"}
    assert record["observations"][-1]["data"]["context"]["model"] == "test-policy"


def test_committed_record_survives_process_kill(tmp_path):
    import subprocess
    import sys

    from world_use.records import inspect, replay

    script = tmp_path / "record.py"
    script.write_text('''
import sys, time
from pathlib import Path
from world_use import Kernel, World, VirtualClock, bodies
from world_use.recorder import load_tape
folder = Path(sys.argv[1])
w = World()
k = Kernel(bodies.make("sim", w), w, VirtualClock(100), run_dir=folder)
k.connect()
k.enable()
k.run({"do": "hold", "seconds": 0.2})
while len(load_tape(folder).get("t", [])) < 20:
    time.sleep(.01)
print("persisted", flush=True)
time.sleep(30)
''')
    folder = tmp_path / "run"
    p = subprocess.Popen([sys.executable, str(script), str(folder)], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "persisted"
        p.kill()
        p.wait(timeout=5)
        record = inspect(folder)
        assert not (folder / "tape.npz").exists()
        assert record["summary"]["ticks"] >= 20 and not record["closed"]
        assert record["jobs"][1]["outcome"]["status"] == "done"
        assert replay(folder, tmp_path / "replay.gif").stat().st_size > 1000
    finally:
        if p.poll() is None:
            p.kill()
            p.wait(timeout=5)


def test_recovery_prefers_new_chunks_over_an_older_manual_save(tmp_path):
    from world_use.recorder import Journal, load_tape

    tape = Tape(1)
    journal = Journal(tape, tmp_path, interval=60)
    try:
        tape.add(0, True, False, 1, [0], [0], None, None, None, None, None)
        tape.save(tmp_path / "tape.npz", 100)
        journal.flush()
        tape.add(.01, True, False, 1, [1], [1], None, None, None, None, None)
        journal.flush()
        a = load_tape(tmp_path)
        assert a["q"].ravel().tolist() == [0, 1]
        assert not list(tmp_path.rglob(".writing-*"))
    finally:
        journal.close()


def test_journal_retires_memory_but_preserves_history_and_lifetime_summary(tmp_path, monkeypatch):
    from world_use.events import EventLog

    monkeypatch.setattr(Tape, "CHUNK", 4)
    tape, reference = Tape(2), Tape(2)
    events = EventLog(keep=6)
    journal = Journal(tape, tmp_path, interval=60, events=events)
    try:
        for i in range(80):
            on = i % 10 < 5
            for target in (tape, reference):
                target.mark_power(i * .01, on)
                target.add(i * .01, on, i % 3 == 0, 1, [0, 0], [i * .001, -i * .001],
                           [-80 + i, i], [30 + i if on else 200, 31], None, None, None)
            events.emit("test", str(i))
            if i % 3 == 0:
                journal.flush()
                assert len(tape.arrays()["t"]) <= Tape.CHUNK
                assert len(tape.arrays()["power_t"]) <= 1
            assert journal.summary(until=i * .01) == reference.summary(100, until=i * .01)
        journal.flush()
        assert journal.error is None
        assert journal.summary(until=1.0) == reference.summary(100, until=1.0)
        saved = load_tape(tmp_path)
        for key, expected in reference.arrays().items():
            np.testing.assert_array_equal(saved[key], expected)
        log = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        assert [e["seq"] for e in log] == list(range(1, 81))
        assert len(events.buf) == 6
    finally:
        journal.close()


def test_long_run_summary_bounds_percentile_sample_and_keeps_early_extrema(tmp_path, monkeypatch):
    from world_use.recorder import _Summary

    monkeypatch.setattr(Tape, "CHUNK", 100)
    monkeypatch.setattr(_Summary, "SAMPLE", 32)
    tape = Tape(1)
    journal = Journal(tape, tmp_path, interval=60)
    try:
        for i in range(2000):
            tape.add(i * .01 + (0.5 if i else 0), True, True, 1, [0], [1 if i == 0 else 0],
                     [50 if i == 0 else 0], [80 if i == 0 else 30], None, None, None)
            if i % 100 == 99:
                journal.flush()
        summary = journal.summary(until=20.5)
        assert summary["ticks"] == 2000
        assert summary["moving_s"] == summary["powered_s"] == 20.5
        assert summary["max_temp_c"] == [80] and summary["max_abs_torque"] == [50]
        assert summary["max_tracking_error_deg"] == [57.3]
        assert summary["tick_ms"] == dict(median=10, p99=10, max=510)
        assert summary["tick_percentile_sample"] == dict(used=32, total=1999)
        assert len(tape.arrays()["t"]) == 100
        assert len(load_tape(tmp_path)["t"]) == 2000
    finally:
        journal.close()


def test_overrun_is_bounded_and_remains_visible_after_storage_recovers(tmp_path, monkeypatch):
    from world_use import recorder
    from world_use.events import EventLog
    from world_use.records import describe, inspect
    from world_use.visualization import RecordReader

    monkeypatch.setattr(Tape, "CHUNK", 2)
    monkeypatch.setattr(Journal, "MAX_BLOCKS", 2)
    monkeypatch.setattr(Journal, "MAX_POWER", 4)
    tape = Tape(1)
    events = EventLog(keep=4)
    journal = Journal(tape, tmp_path, interval=60, events=events)

    def add(i):
        tape.mark_power(i * .01, i % 2 == 0)
        tape.add(i * .01, True, True, 1, [0], [i], [i], [30], None, None, None)
        events.emit("test", str(i))

    def full_disk(*args, **kwargs):
        raise OSError("disk full")

    try:
        add(0)
        journal.flush()
        with monkeypatch.context() as patch:
            patch.setattr(recorder, "open", full_disk, raising=False)
            for i in range(1, 12):
                add(i)
                with pytest.raises(OSError, match="disk full"):
                    journal.flush()
            assert "disk full" in journal.error and "lost" in journal.error
            assert len(tape.arrays()["t"]) <= 4 and len(events.buf) == 4
            assert len(tape.arrays()["power_t"]) <= 4
        journal.flush()
        lost = dict(samples=7, events=7, power_transitions=7)
        assert journal.losses == lost
        assert journal.error and "incomplete" in journal.error and "disk full" not in journal.error
        assert journal.summary()["recording_lost"] == lost
        saved = load_tape(tmp_path)
        assert saved["sample_index"].tolist() == [0, 8, 9, 10, 11]
        assert saved["power_index"].tolist() == [0, 8, 9, 10, 11]
        record = inspect(tmp_path)
        assert record["summary"] == journal.summary()
        assert "INCOMPLETE RECORD" in describe(record)
        _, log = RecordReader(tmp_path).poll()
        assert any(e["kind"] == "recording" and e["level"] == "alarm" for e in log)
        journal.flush()
        assert journal.losses == lost
        assert len(load_tape(tmp_path)["t"]) == 5
    finally:
        journal.close()


def test_failed_chunk_commit_retries_without_duplicates_or_early_retirement(tmp_path, monkeypatch):
    from world_use import recorder

    monkeypatch.setattr(Tape, "CHUNK", 2)
    tape = Tape(1)
    journal = Journal(tape, tmp_path, interval=60)

    def full_disk(*args):
        raise OSError("disk full")

    try:
        for i in range(4):
            tape.add(i * .01, True, False, 1, [0], [i], None, None, None, None, None)
        with monkeypatch.context() as patch:
            patch.setattr(recorder.os, "replace", full_disk)
            with pytest.raises(OSError, match="disk full"):
                journal.flush()
        assert len(tape.arrays()["t"]) == 4
        assert not list(tmp_path.rglob(".writing-*"))
        journal.flush()
        journal.flush()
        assert journal.error is None
        assert load_tape(tmp_path)["q"].ravel().tolist() == [0, 1, 2, 3]
        assert journal.summary()["ticks"] == 4
        assert len(tape.arrays()["t"]) == 2
    finally:
        journal.close()
