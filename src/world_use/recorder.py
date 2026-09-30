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
from pathlib import Path

import numpy as np


class Tape:
    """Every control tick, kept in preallocated arrays a minute at a time: a tick writes one row and allocates
    nothing, and a copy of the record so far is a few array copies. So it can be saved while the robot runs,
    without churning Python on the interpreter the control loop shares."""
    CHUNK = 6000                      # rows per block: a minute at 100 Hz

    def __init__(self, n: int):
        self.n = n
        self._blocks: list[dict[str, np.ndarray]] = []
        self._used = self.CHUNK       # rows written in the last block
        self._lock = threading.Lock() # the control thread writes while a request thread may copy
        self._power: list[tuple[float, bool]] = []

    def mark_power(self, t: float, on: bool):
        """Power budget: from an enable attempt through confirmed disable, including incomplete transitions."""
        with self._lock:
            if not self._power or self._power[-1][1] != on:
                self._power.append((t, on))

    def _block(self, rows: int | None = None) -> dict[str, np.ndarray]:
        c, n = self.CHUNK if rows is None else rows, self.n
        return dict(t=np.zeros(c), enabled=np.zeros(c, bool), moving=np.zeros(c, bool), job=np.zeros(c, int),
                    q_cmd=np.zeros((c, n)), q=np.zeros((c, n)), tau=np.full((c, n), np.nan),
                    temp=np.full((c, n), np.nan), grip_cmd=np.full(c, np.nan), grip=np.full(c, np.nan),
                    grip_tau=np.full(c, np.nan))

    def add(self, t, enabled, moving, job, q_cmd, q, tau, temp, grip_cmd, grip, grip_tau):
        with self._lock:
            if self._used == self.CHUNK:
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
        return (len(self._blocks) - 1) * self.CHUNK + self._used if self._blocks else 0

    def arrays(self) -> dict:
        with self._lock:
            power = dict(power_t=np.array([t for t, _ in self._power]),
                         power_on=np.array([on for _, on in self._power], bool)) if self._power else {}
            if not self._blocks:
                return self._block(0) | power if power else {}
            blocks, used = list(self._blocks), self._used
            last = {key: v[:used].copy() for key, v in blocks[-1].items()}
        return {key: np.concatenate([b[key] for b in blocks[:-1]] + [last[key]]) for key in last} | power

    def since(self, row: int) -> dict:
        """Copy only new samples; completed blocks are immutable, and the live block is copied under the lock."""
        with self._lock:
            parts = []
            for i in range(row // self.CHUNK, len(self._blocks)):
                begin = row % self.CHUNK if i == row // self.CHUNK else 0
                end = self._used if i == len(self._blocks) - 1 else self.CHUNK
                parts.append({key: v[begin:end].copy() for key, v in self._blocks[i].items()})
        return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]} if parts else {}

    def power(self) -> dict:
        with self._lock:
            return dict(power_t=np.array([t for t, _ in self._power]),
                        power_on=np.array([on for _, on in self._power], bool))

    def summary(self, rate_hz: float, *, until: float | None = None) -> dict:
        """Elapsed durations, not tick counts divided by the nominal rate (kept for API compatibility)."""
        return self._summary(self.arrays(), until)

    @staticmethod
    def _summary(a: dict, until: float | None) -> dict:
        if not a:
            return dict(ticks=0)
        t = a.get("t", np.array([]))
        power_t, power_on = a.get("power_t", t), a.get("power_on", a.get("enabled", np.array([], bool)))
        end = max(until or 0.0, float(t[-1]) if len(t) else 0.0, float(power_t[-1]) if len(power_t) else 0.0)
        powered = float(np.maximum(0, np.diff(np.append(power_t, end))) @ power_on)
        timing = dict(time_basis="elapsed_monotonic", powered_s=round(powered, 1),
                      power_basis="enable_attempt_to_confirmed_disable" if "power_t" in a else "sampled_enabled")
        if not len(t):
            return dict(ticks=0, wall_s=round(end - power_t[0], 1), moving_s=0.0, moving_share=0.0, **timing)
        on = a["enabled"]
        moving = np.maximum(0, np.diff(np.append(t, end))) @ (on & a["moving"])
        with np.errstate(all="ignore"):
            err = np.abs(a["q"] - a["q_cmd"])[on & a["moving"]]
            temp = a["temp"][on] if on.any() else a["temp"]
            # how late the control loop ran while the motors were on: a busy host shows up here, not in the motion
            gaps = np.diff(a["t"])[on[1:] & on[:-1]] * 1000
            return dict(
                tick_ms=dict(median=round(float(np.median(gaps)), 1), p99=round(float(np.percentile(gaps, 99)), 1),
                             max=round(float(gaps.max()), 1)) if len(gaps) else None,
                ticks=len(t), wall_s=round(end - min(float(t[0]), float(power_t[0])), 1),
                **timing, moving_s=round(float(moving), 1),
                moving_share=round(float(moving / powered), 3) if powered > 0 else None,
                max_tracking_error_deg=np.round(np.degrees(np.nanmax(err, axis=0)), 2).tolist() if len(err) else None,
                max_abs_torque=(np.round(np.nanmax(np.abs(a["tau"]), axis=0), 2).tolist()
                                if np.isfinite(a["tau"]).any() else None),
                max_temp_c=np.round(np.nanmax(temp, axis=0), 1).tolist() if np.isfinite(temp).any() else None)

    def save(self, path: Path, rate_hz: float, *, until: float | None = None):
        a = self.arrays()
        if a:
            save_arrays(path, a)
        return self._summary(a, until)


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
    """Persist incremental tape chunks once a second, away from the control thread.

    Readers ignore unfinished temporary files. Process loss can lose the last interval; no database or
    crash handler is needed to read everything already committed. Memory remains available for live summaries.
    """
    def __init__(self, tape: Tape, folder: Path, interval: float = 1.0, *, events=None):
        self.tape, self.folder = tape, folder / "tape"
        self.events = events
        self.seq, self.offset = 0, 0
        self.folder.mkdir(exist_ok=True)
        self.row, self.part = 0, 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, args=(interval,), name="flight-recorder", daemon=True)
        self._thread.start()

    def flush(self):
        with self._lock:
            try:
                self._flush()
            except OSError as e:
                self.error = str(e)
                raise
            self.error = None

    def _flush(self):
        # Retry from the last complete batch after a partial write (e.g. a full disk).
        from .events import _plain
        events = self.events.since(self.seq) if self.events else []
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
        a = self.tape.since(self.row)
        if a and len(a["t"]):
            save_arrays(self.folder / f"{self.part:06d}.npz", a)
            self.row += len(a["t"])
            self.part += 1
        save_arrays(self.folder / "power.npz", self.tape.power())

    def _loop(self, interval):
        while not self._stop.wait(interval):
            try:
                self.flush()
            except OSError as e:
                self.error = str(e)              # visible in status; a later flush can recover

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
    return dict(format_version=1, created_at=datetime.now(UTC).isoformat(), package_version=__version__,
                source_sha256=source.hexdigest(),
                adapter=next((n for n, m in bodies.manifests().items() if m is k.manifest), None),
                body=k.manifest.name, mode="simulation" if getattr(k.body, "simulated", False) else "hardware",
                urdf_sha256=hashlib.sha256(Path(k.manifest.urdf).read_bytes()).hexdigest(),
                initial=asdict(snap), **context)
