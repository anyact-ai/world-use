"""Flight recorder: every control tick while connected, plus the numbers that say how a run went.

The summary answers the questions that decide whether a policy is worth running on hardware: how long were
the motors on, how much of that time did the robot actually move, how hot did it get.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections import deque
from contextlib import suppress
from copy import deepcopy
from pathlib import Path

import numpy as np


class Tape:
    """Preallocated telemetry blocks. A journal bounds the pending buffer and retires committed blocks;
    embedded rehearsals without a journal retain their complete tape."""
    CHUNK = 6000                      # rows per block: a minute at 100 Hz

    def __init__(self, n: int):
        self.n = n
        self._blocks: list[dict[str, np.ndarray]] = []
        self._used = self.CHUNK       # rows written in the last block
        self._lock = threading.Lock() # the control thread writes while a request thread may copy
        self._power: deque[tuple[float, bool]] = deque()
        self._power_count = 0
        self._start = 0
        self.max_blocks: int | None = None

    def mark_power(self, t: float, on: bool):
        """Power budget: from an enable attempt through confirmed disable, including incomplete transitions."""
        with self._lock:
            if not self._power or self._power[-1][1] != on:
                self._power.append((t, on))
                self._power_count += 1

    def _block(self, rows: int | None = None) -> dict[str, np.ndarray]:
        c, n = self.CHUNK if rows is None else rows, self.n
        return dict(t=np.zeros(c), enabled=np.zeros(c, bool), moving=np.zeros(c, bool), job=np.zeros(c, int),
                    q_cmd=np.zeros((c, n)), q=np.zeros((c, n)), tau=np.full((c, n), np.nan),
                    temp=np.full((c, n), np.nan), grip_cmd=np.full(c, np.nan), grip=np.full(c, np.nan),
                    grip_tau=np.full(c, np.nan))

    def add(self, t, enabled, moving, job, q_cmd, q, tau, temp, grip_cmd, grip, grip_tau):
        with self._lock:
            if self._used == self.CHUNK:
                if self.max_blocks is not None and len(self._blocks) >= self.max_blocks:
                    self._blocks.pop(0)
                    self._start += self.CHUNK
                self._blocks.append(self._block())
                self._used = 0
            b, i = self._blocks[-1], self._used
            b["t"][i], b["enabled"][i], b["moving"][i], b["job"][i] = t, enabled, moving, job
            b["q_cmd"][i], b["q"][i] = q_cmd, q
            if tau is not None:
                b["tau"][i] = tau
            if temp is not None:
                b["temp"][i] = temp
            for key, v in (("grip_cmd", grip_cmd), ("grip", grip), ("grip_tau", grip_tau)):
                if v is not None:
                    b[key][i] = v
            self._used = i + 1

    def __len__(self) -> int:
        return self._start + (len(self._blocks) - 1) * self.CHUNK + self._used if self._blocks else 0

    def arrays(self) -> dict:
        """Copy retained samples. Persisted runs' complete history is available through load_tape."""
        _, _, a = self.snapshot(0, 0)
        if not len(a["power_t"]):
            a.pop("power_t")
            a.pop("power_on")
        return a if len(a["t"]) or "power_t" in a else {}

    @property
    def bounds(self) -> tuple[int, int]:
        with self._lock:
            return self._start, self._power_count - len(self._power)

    def snapshot(self, row: int, power: int) -> tuple[int, int, dict]:
        """Copy pending samples and their absolute cursors atomically, including any buffer overrun."""
        with self._lock:
            row = max(row, self._start)
            relative = row - self._start
            parts = []
            for i in range(relative // self.CHUNK, len(self._blocks)):
                begin = relative % self.CHUNK if i == relative // self.CHUNK else 0
                end = self._used if i == len(self._blocks) - 1 else self.CHUNK
                parts.append({key: v[begin:end].copy() for key, v in self._blocks[i].items()})
            first_power = self._power_count - len(self._power)
            power = max(power, first_power)
            transitions = list(self._power)[power - first_power:]
        a = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]} if parts else self._block(0)
        return row, power, a | dict(power_t=np.array([t for t, _ in transitions]),
                                   power_on=np.array([on for _, on in transitions], bool))

    def discard(self, row: int, power: int):
        with self._lock:
            while len(self._blocks) > 1 and self._start + self.CHUNK <= row:
                self._blocks.pop(0)
                self._start += self.CHUNK
            # Keep the last transition to suppress duplicate power markers.
            while len(self._power) > 1 and self._power_count - len(self._power) < power:
                self._power.popleft()

    def summary(self, rate_hz: float, *, until: float | None = None) -> dict:
        """Elapsed durations, not tick counts divided by the nominal rate (kept for API compatibility)."""
        return self._summary(self.arrays(), until)

    @staticmethod
    def _summary(a: dict, until: float | None) -> dict:
        if not a:
            return dict(ticks=0)
        summary = _Summary(a["q"].shape[1] if "q" in a else 0, exact=True)
        summary.add(a)
        return summary.result(until)

    def save(self, path: Path, rate_hz: float, *, until: float | None = None):
        a = self.arrays()
        if a:
            save_arrays(path, a)
        return self._summary(a, until)


class _Summary:
    """Lifetime totals and extrema; a bounded uniform sample estimates long-run tick percentiles.

    Updated by the journal thread, never by the control thread. Offline readers can request exact
    percentiles because they already hold the complete tape.
    """
    SAMPLE = 8192

    def __init__(self, n: int, *, exact=False):
        self.exact = exact
        self.ticks = self.gap_count = 0
        self.first: float | None = None
        self.end = self.powered = self.moving = self.gap_max = 0.0
        self.previous: tuple[float, bool, bool] | None = None
        self.power_last: tuple[float, bool] | None = None
        self.explicit_power = self.any_on = self.any_moving = False
        self.tracking = np.full(n, np.nan)
        self.torque = np.full(n, np.nan)
        self.temp_on = np.full(n, np.nan)
        self.temp_all = np.full(n, np.nan)
        self.gaps = np.array([])
        self.keys = np.array([])
        self.rng = np.random.default_rng(0)

    def _power(self, times, on, indices=None):
        if not len(times):
            return
        if self.power_last is not None:
            t, enabled = self.power_last
            self.powered += max(0.0, float(times[0]) - t) * enabled
        contiguous = np.diff(indices) == 1 if indices is not None else np.ones(len(times) - 1, bool)
        self.powered += float(np.maximum(0, np.diff(times)) @ (on[:-1] & contiguous))
        self.power_last = float(times[-1]), bool(on[-1])
        self.first = min(self.first if self.first is not None else float(times[0]), float(times[0]))
        self.end = max(self.end, float(times[-1]))

    def add(self, a: dict, *, row_gap=False, power_gap=False):
        if row_gap:
            self.previous = None
            if not self.explicit_power:
                self.power_last = None
        if power_gap:
            self.power_last = None
        if len(a.get("power_t", [])):
            if not self.explicit_power:
                self.power_last, self.powered = None, 0.0
            self.explicit_power = True
            self._power(a["power_t"], a["power_on"], a.get("power_index"))
        t, on = a.get("t", np.array([])), a.get("enabled", np.array([], bool))
        if not len(t):
            return
        if not self.explicit_power:
            self._power(t, on, a.get("sample_index"))
        self.first = min(self.first if self.first is not None else float(t[0]), float(t[0]))
        self.end = max(self.end, float(t[-1]))
        self.ticks += len(t)
        moving = on & a["moving"]
        contiguous = np.diff(a["sample_index"]) == 1 if "sample_index" in a else np.ones(len(t) - 1, bool)
        self.moving += float(np.maximum(0, np.diff(t)) @ (moving[:-1] & contiguous))
        gaps = np.diff(t)[on[1:] & on[:-1] & contiguous] * 1000
        if self.previous is not None:
            prev_t, prev_on, prev_moving = self.previous
            self.moving += max(0.0, float(t[0]) - prev_t) * prev_moving
            if prev_on and on[0]:
                gaps = np.append(gaps, (float(t[0]) - prev_t) * 1000)
        self.previous = float(t[-1]), bool(on[-1]), bool(moving[-1])
        self.any_on |= bool(on.any())
        self.any_moving |= bool(moving.any())
        self.tracking = np.fmax(self.tracking, np.fmax.reduce(np.abs(a["q"] - a["q_cmd"])[moving],
                                                           axis=0, initial=np.nan))
        self.torque = np.fmax(self.torque, np.fmax.reduce(np.abs(a["tau"]), axis=0, initial=np.nan))
        self.temp_on = np.fmax(self.temp_on, np.fmax.reduce(a["temp"][on], axis=0, initial=np.nan))
        self.temp_all = np.fmax(self.temp_all, np.fmax.reduce(a["temp"], axis=0, initial=np.nan))
        if len(gaps):
            self.gap_count += len(gaps)
            self.gap_max = max(self.gap_max, float(gaps.max()))
            self.gaps = np.concatenate((self.gaps, gaps))
            if not self.exact:
                # Independent random priorities select a uniform sample, independent of chunk boundaries.
                self.keys = np.concatenate((self.keys, self.rng.random(len(gaps))))
                if len(self.gaps) > self.SAMPLE:
                    keep = np.argpartition(self.keys, self.SAMPLE - 1)[:self.SAMPLE]
                    self.gaps, self.keys = self.gaps[keep], self.keys[keep]

    def result(self, until: float | None) -> dict:
        if self.first is None:
            return dict(ticks=0)
        end = max(until or 0.0, self.end)
        powered = self.powered
        if self.power_last is not None:
            powered += max(0.0, end - self.power_last[0]) * self.power_last[1]
        moving = self.moving
        if self.previous is not None:
            moving += max(0.0, end - self.previous[0]) * self.previous[2]
        s: dict = dict(ticks=self.ticks, wall_s=round(end - self.first, 1), powered_s=round(powered, 1),
                 moving_s=round(moving, 1), moving_share=round(moving / powered, 3) if powered > 0 else None,
                 time_basis="elapsed_monotonic",
                 power_basis="enable_attempt_to_confirmed_disable" if self.explicit_power else "sampled_enabled")
        if not self.ticks:
            return s | dict(moving_share=0.0)
        temp = self.temp_on if self.any_on else self.temp_all
        s.update(tick_ms=dict(median=round(float(np.median(self.gaps)), 1),
                              p99=round(float(np.percentile(self.gaps, 99)), 1),
                              max=round(self.gap_max, 1)) if len(self.gaps) else None,
                 max_tracking_error_deg=np.round(np.degrees(self.tracking), 2).tolist() if self.any_moving else None,
                 max_abs_torque=np.round(self.torque, 2).tolist() if np.isfinite(self.torque).any() else None,
                 max_temp_c=np.round(temp, 1).tolist() if np.isfinite(temp).any() else None)
        if self.gap_count > len(self.gaps):
            s["tick_percentile_sample"] = dict(used=len(self.gaps), total=self.gap_count)
        return s


def _atomic(path: Path, write):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".writing-", delete=False) as f:
        temp = Path(f.name)
        try:
            write(f)
            f.flush()
            os.fsync(f.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def save_summary(path: Path, summary: dict):
    from .events import _plain
    _atomic(path, lambda f: f.write(json.dumps(summary, indent=2, default=_plain).encode()))


def save_arrays(path: Path, arrays: dict):
    _atomic(path, lambda f: np.savez_compressed(f, **arrays))


class Journal:
    """Persist once a second, off the control thread. Retire committed telemetry and cap pending history.

    Overruns during prolonged storage failures are reported in status and recording.json. They never
    block control or motor release. Embedded tapes without a journal remain complete in memory.
    """
    MAX_BLOCKS = 5
    MAX_POWER = 5000

    def __init__(self, tape: Tape, folder: Path, interval: float = 1.0, *, events=None):
        self.tape, self.folder = tape, folder / "tape"
        self.events = events
        self.seq, self.offset = 0, 0
        self.folder.mkdir(exist_ok=True)
        self.row = self.power = self.part = 0
        self._error: str | None = None
        self._lost = dict(samples=0, events=0, power_transitions=0)
        self._summary = _Summary(tape.n)
        with tape._lock:
            tape.max_blocks = self.MAX_BLOCKS
            tape._power = deque(tape._power, maxlen=self.MAX_POWER)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, args=(interval,), name="flight-recorder", daemon=True)
        self._thread.start()

    @property
    def losses(self) -> dict:
        row, power = self.tape.bounds
        seq = self.events.first_seq - 1 if self.events else self.seq
        return dict(samples=self._lost["samples"] + max(0, row - self.row),
                    events=self._lost["events"] + max(0, seq - self.seq),
                    power_transitions=self._lost["power_transitions"] + max(0, power - self.power))

    @property
    def error(self) -> str | None:
        errors = [self._error] if self._error else []
        losses = self.losses
        if any(losses.values()):
            errors.append("recording incomplete: " + ", ".join(f"{v} {k} lost" for k, v in losses.items() if v))
        return "; ".join(errors) or None

    def summary(self, *, until: float | None = None) -> dict:
        with self._lock:
            summary = deepcopy(self._summary)
            row, power, a = self.tape.snapshot(self.row, self.power)
            summary.add(a, row_gap=row > self.row, power_gap=power > self.power)
            result = summary.result(until)
            losses = self.losses
            if any(losses.values()):
                result["recording_lost"] = losses
            return result

    def flush(self):
        with self._lock:
            try:
                self._flush()
            except OSError as e:
                self._error = str(e)
                raise
            self._error = None

    def _flush(self):
        from .events import _plain
        events = self.events.since(self.seq) if self.events else []
        row, power, a = self.tape.snapshot(self.row, self.power)
        seq = events[0]["seq"] - 1 if events else self.seq
        if row > self.row or power > self.power or seq > self.seq:
            lost = dict(samples=self._lost["samples"] + row - self.row,
                        events=self._lost["events"] + seq - self.seq,
                        power_transitions=self._lost["power_transitions"] + power - self.power)
            save_summary(self.folder.parent / "recording.json", lost)
            if row > self.row:
                self._summary.previous = None
                if not self._summary.explicit_power:
                    self._summary.power_last = None
            if power > self.power:
                self._summary.power_last = None
            self._lost = lost
            self.row, self.power, self.seq = row, power, seq
        # Retry from the last complete batch after a partial write (e.g. a full disk).
        if events:
            path = self.folder.parent / "events.jsonl"
            with open(path, "r+b" if path.exists() else "w+b") as f:
                f.seek(self.offset)
                f.truncate()
                f.write("".join(json.dumps(e, default=_plain) + "\n" for e in events).encode())
                f.flush()
                os.fsync(f.fileno())
                self.offset = f.tell()
            self.seq = events[-1]["seq"]
        if len(a["t"]) or len(a["power_t"]):
            a["sample_index"] = np.arange(self.row, self.row + len(a["t"]))
            a["power_index"] = np.arange(self.power, self.power + len(a["power_t"]))
            save_arrays(self.folder / f"{self.part:06d}.npz", a)
            self._summary.add(a)
            self.row += len(a["t"])
            self.power += len(a["power_t"])
            self.part += 1
            self.tape.discard(self.row, self.power)

    def _loop(self, interval):
        while not self._stop.wait(interval):
            with suppress(OSError):            # visible in status; a later flush can recover
                self.flush()

    def close(self):
        self._stop.set()
        self._thread.join()
        self.flush()


def load_tape(folder: Path | str) -> dict:
    """Read a normal save or recover committed chunks after process loss. Never replay commands to a robot."""
    folder = Path(folder)
    parts = []
    for path in sorted((folder / "tape").glob("[0-9]*.npz")):
        with np.load(path) as f:
            parts.append(dict(f))
    a = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]} if parts else {}
    if (folder / "tape.npz").exists():
        with np.load(folder / "tape.npz") as f:
            saved = dict(f)
        if not a or len(saved.get("t", [])) >= len(a["t"]):
            a = saved
    if (folder / "tape" / "power.npz").exists():
        with np.load(folder / "tape" / "power.npz") as f:
            a.update(dict(f))
    return a


def session_record(k, **context) -> dict:
    from dataclasses import asdict
    from datetime import UTC, datetime

    from . import __version__, bodies
    from .plan import snapshot

    source = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.rglob("*.py")):
        source.update(str(path.relative_to(root)).encode())
        source.update(path.read_bytes())
    snap = snapshot(k)
    initial = asdict(snap)
    urdf = Path(k.manifest.urdf).read_bytes()
    if k.run_dir:
        from .robot_assets import archive
        archive(Path(k.manifest.urdf), k.run_dir)
        _atomic(k.run_dir / "robot.urdf", lambda f: f.write(urdf))
        initial["model"]["urdf"] = "robot.urdf"
    return dict(format_version=3, created_at=datetime.now(UTC).isoformat(), package_version=__version__,
                source_sha256=source.hexdigest(),
                adapter=next((n for n, m in bodies.manifests().items() if m is k.manifest),
                             f"{type(k.body).__module__}:{type(k.body).__qualname__}"),
                body=k.manifest.name, mode="simulation" if getattr(k.body, "simulated", False) else "hardware",
                urdf_sha256=hashlib.sha256(urdf).hexdigest(),
                initial=initial, **context)
