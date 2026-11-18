import json
import stat
import subprocess
import sys

import numpy as np
import pytest

from world_use import Kernel, VirtualClock, World, bodies, recorder
from world_use.events import EventLog
from world_use.recorder import Journal, Tape, load_tape, save_arrays, save_summary
from world_use.records import describe, events, inspect

CLOSED = json.dumps(dict(kind="closed", level="info", t=0, message="closed"))


def full_disk(*args, **kwargs):
    raise OSError("disk full")


def sim_kernel(run_dir, clock=None):
    world = World()
    return Kernel(bodies.make("sim", world), world, clock or VirtualClock(100), run_dir=run_dir)


def test_recording_failure_does_not_interrupt_motion_or_release(tmp_path, monkeypatch):
    k = sim_kernel(tmp_path)
    k.connect()
    k.enable()
    with monkeypatch.context() as patch:
        patch.setattr(recorder, "open", full_disk, raising=False)
        with pytest.raises(OSError, match="disk full"):
            k.journal.flush()
        assert k.run({"do": "line", "up": .03}).ok
        assert not k.faulted and not k.power_uncertain
        assert k.run({"do": "line", "up": -.03}).ok
        k.release()
        assert not k.enabled and k.journal.error == "disk full"
        assert k.close()["recording_error"] == "disk full"
        assert not (tmp_path / "complete.json").exists()
    k.save_record()
    log = events(tmp_path)
    assert [e["seq"] for e in log] == list(range(1, k.events.seq + 1))
    assert log[-1]["kind"] == "closed" and inspect(tmp_path)["closed"]


def test_failed_final_telemetry_write_keeps_committed_prefix_open_until_retry(tmp_path, monkeypatch):
    k = sim_kernel(tmp_path)
    k.connect()
    k.enable()
    assert k.run({"do": "hold", "seconds": .05}).ok
    prefix = k.save_record()["ticks"]
    assert prefix > 0
    assert k.run({"do": "hold", "seconds": .05}).ok
    k.release()
    total = k.journal.summary()["ticks"]
    assert total > prefix
    with monkeypatch.context() as patch:
        patch.setattr(recorder, "save_arrays", full_disk)
        assert k.close()["recording_error"] == "disk full"
    assert events(tmp_path)[-1]["kind"] == "closed"
    assert not (tmp_path / "complete.json").exists()
    record = inspect(tmp_path)
    assert not record["closed"] and record["summary"]["ticks"] == prefix
    assert "open or interrupted record" in describe(record)

    k.save_record()
    record = inspect(tmp_path)
    assert record["closed"] and record["summary"]["ticks"] == total
    assert load_tape(tmp_path)["sample_index"].tolist() == list(range(total))
    assert "closed normally" in describe(record)


@pytest.mark.parametrize("parts, extra_bytes, ending", [
    (1, 0, "\n"), (2, 0, "\n"), (0, 0, "\n"), (1, 1, "\n"), (1, -1, "\n"),
    (1, 0, ""),     # an unterminated event line is not committed
])
def test_current_record_closes_only_with_matching_completion_counts(tmp_path, parts, extra_bytes, ending):
    save_summary(tmp_path / "session.json", dict(format_version=3))
    (tmp_path / "tape").mkdir()
    tape = Tape(1)
    tape.add(0, False, False, 0, [0], [0], None, None, None, None, None)
    save_arrays(tmp_path / "tape/000000.npz", tape.arrays())
    (tmp_path / "events.jsonl").write_text(CLOSED + ending)
    assert not inspect(tmp_path)["closed"]
    save_summary(tmp_path / "complete.json", dict(parts=parts, events_bytes=len(CLOSED + ending) + extra_bytes))
    assert inspect(tmp_path)["closed"] == (parts == 1 and extra_bytes == 0 and ending == "\n")


@pytest.mark.parametrize("format_version", [None, 1, 2])
def test_legacy_record_is_closed_by_its_closed_event(tmp_path, format_version):
    if format_version is None:
        save_arrays(tmp_path / "tape.npz", {})
    else:
        save_summary(tmp_path / "session.json", dict(format_version=format_version))
    (tmp_path / "events.jsonl").write_text(CLOSED + "\n")
    assert inspect(tmp_path)["closed"]


def test_elapsed_durations_include_slow_control_ticks():
    tape = Tape(1)
    for t, moving in [(0, True), (0.01, True), (0.51, False), (1.01, False)]:
        tape.add(t, True, moving, 1, [0], [0], None, None, None, None, None)
    s = tape.summary()
    assert s["powered_s"] == 1.0 and s["moving_s"] == 0.5
    assert s["moving_share"] == pytest.approx(0.505, abs=0.001)
    assert s["tick_ms"]["max"] == 500


def test_events_tape_and_enable_ramp_share_one_elapsed_clock(tmp_path):
    clock = VirtualClock(100)
    k = sim_kernel(tmp_path, clock)

    def taking(seconds, action):
        def slow():
            clock.t += seconds
            return action()
        return slow

    body = k.body
    body.connect, body.enable, body.disable = taking(2, body.connect), taking(1, body.enable), taking(.5, body.disable)
    k.connect()
    k.enable()
    k.tick()
    clock.t += 1
    k.tick()
    k.emit("contact", "same timeline as tape")
    k.release()
    summary = k.save_record()
    assert summary["powered_s"] == 2.5 and summary["wall_s"] == 4.5
    log, tape = events(tmp_path), load_tape(tmp_path)
    assert log[0]["t"] == 2 and tape["t"].tolist() == [3, 4]
    assert next(e["t"] for e in log if e["kind"] == "contact") == tape["t"][-1]
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


KILLED_RUN = '''
import sys, time
from pathlib import Path
from world_use import Kernel, World, VirtualClock, bodies
from world_use.recorder import load_tape
from world_use.records import events
folder = Path(sys.argv[1])
w = World()
k = Kernel(bodies.make("sim", w), w, VirtualClock(100), run_dir=folder)
k.connect()
k.enable()
k.run({"do": "hold", "seconds": 0.2})
# Wait until the background journal has committed the outcome and the telemetry.
while len(load_tape(folder).get("t", [])) < 20 or not any(e["kind"] == "finished" for e in events(folder)):
    time.sleep(.01)
print("persisted", flush=True)
time.sleep(30)
'''


@pytest.fixture(scope="module")
def interrupted_record(tmp_path_factory):
    folder = tmp_path_factory.mktemp("killed") / "run"
    p = subprocess.Popen([sys.executable, "-c", KILLED_RUN, str(folder)], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "persisted"
    finally:
        p.kill()
        p.wait(timeout=5)
    return folder


def test_committed_record_survives_process_kill(interrupted_record):
    record = inspect(interrupted_record)
    assert not (interrupted_record / "tape.npz").exists()
    assert record["summary"]["ticks"] >= 20 and not record["closed"]
    assert record["jobs"][1]["outcome"]["status"] == "done"


@pytest.mark.rendering
def test_interrupted_record_can_be_replayed(interrupted_record, tmp_path):
    from world_use.records import replay

    assert replay(interrupted_record, tmp_path / "replay.gif").stat().st_size > 1000


def test_a_0_2_0_record_stays_readable(tmp_path):
    """0.2.0 saved tape.npz and summary.json at close, without session.json."""
    from world_use import fit
    from world_use.bodies.rebot import MANIFEST

    tape = Tape(6)
    for i in range(3):
        tape.add(i * .01, True, False, 1, np.zeros(6), np.zeros(6), np.zeros(6), np.full(6, 30.0), 1, 1, 0)
    save_arrays(tmp_path / "tape.npz", {key: v for key, v in tape.arrays().items() if not key.startswith("power")})
    (tmp_path / "summary.json").write_text(json.dumps(dict(body=MANIFEST.name, ticks=3)))
    assert inspect(tmp_path)["summary"]["ticks"] == 3
    assert fit.robot_of([tmp_path]) is MANIFEST


def test_record_files_are_readable_like_the_event_log(tmp_path):
    save_summary(tmp_path / "summary.json", {})
    assert stat.S_IMODE((tmp_path / "summary.json").stat().st_mode) == 0o644


def test_journal_retires_memory_but_preserves_history_and_lifetime_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(Tape, "CHUNK", 4)
    tape, reference = Tape(2), Tape(2)
    log = EventLog(keep=6)
    journal = Journal(tape, tmp_path, interval=60, events=log)
    try:
        for i in range(80):
            on = i % 10 < 5
            for target in (tape, reference):
                target.mark_power(i * .01, on)
                target.add(i * .01, on, i % 3 == 0, 1, [0, 0], [i * .001, -i * .001],
                           [-80 + i, i], [30 + i if on else 200, 31], None, None, None)
            log.emit("test", str(i))
            if i % 3 == 0:
                journal.flush()
                assert len(tape.arrays()["t"]) <= Tape.CHUNK
                assert len(tape.arrays()["power_t"]) <= 1
            assert journal.summary(until=i * .01) == reference.summary(until=i * .01)
        journal.flush()
        assert journal.error is None
        assert journal.summary(until=1.0) == reference.summary(until=1.0)
        saved = load_tape(tmp_path)
        for key, expected in reference.arrays().items():
            np.testing.assert_array_equal(saved[key], expected)
        assert [e["seq"] for e in events(tmp_path)] == list(range(1, 81))
        assert len(log.buf) == 6
    finally:
        journal.close()


def test_long_run_summary_bounds_percentile_sample_and_keeps_early_extrema(tmp_path, monkeypatch):
    monkeypatch.setattr(Tape, "CHUNK", 100)
    monkeypatch.setattr(recorder._Summary, "SAMPLE", 32)
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
    from world_use.visualization import RecordReader

    monkeypatch.setattr(Tape, "CHUNK", 2)
    monkeypatch.setattr(Journal, "MAX_BLOCKS", 2)
    monkeypatch.setattr(Journal, "MAX_POWER", 4)
    tape = Tape(1)
    log = EventLog(keep=4)
    journal = Journal(tape, tmp_path, interval=60, events=log)

    def add(i):
        tape.mark_power(i * .01, i % 2 == 0)
        tape.add(i * .01, True, True, 1, [0], [i], [i], [30], None, None, None)
        log.emit("test", str(i))

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
            assert len(tape.arrays()["t"]) <= 4 and len(log.buf) == 4
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
        _, alarms = RecordReader(tmp_path).poll()
        assert any(e["kind"] == "recording" and e["level"] == "alarm" for e in alarms)
        journal.flush()
        assert journal.losses == lost
        assert len(load_tape(tmp_path)["t"]) == 5
    finally:
        journal.close()


def test_failed_chunk_commit_retries_without_duplicates_or_early_retirement(tmp_path, monkeypatch):
    monkeypatch.setattr(Tape, "CHUNK", 2)
    tape = Tape(1)
    journal = Journal(tape, tmp_path, interval=60)
    try:
        for i in range(4):
            tape.add(i * .01, True, False, 1, [0], [i], None, None, None, None, None)
        with monkeypatch.context() as patch:
            patch.setattr(recorder.os, "replace", full_disk)
            with pytest.raises(OSError, match="disk full"):
                journal.close()
        assert len(tape.arrays()["t"]) == 4
        assert not list(tmp_path.rglob(".writing-*"))
        assert not (tmp_path / "complete.json").exists()
        journal.flush()
        journal.flush()
        assert journal.error is None
        assert load_tape(tmp_path)["q"].ravel().tolist() == [0, 1, 2, 3]
        assert journal.summary()["ticks"] == 4
        assert len(tape.arrays()["t"]) == 2
        assert json.loads((tmp_path / "complete.json").read_text()) == dict(parts=1, events_bytes=0)
    finally:
        journal.close()
