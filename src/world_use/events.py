"""Events: one stream of everything that happened, numbered, so a policy can ask "what changed since N?"."""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

LEVELS = ("info", "warn", "alarm")


class EventLog:
    def __init__(self, path: Path | None = None, keep: int = 5000, *,
                 clock: Callable[[], float] = time.monotonic, t0: float | None = None):
        self.buf: deque[dict] = deque(maxlen=keep)
        self.seq = 0
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.clock = clock
        self.t0 = clock() if t0 is None else t0
        self.file = open(path, "a", buffering=1) if path else None       # noqa: SIM115 - open for the log's life

    def emit(self, kind: str, message: str, level: str = "info", **data) -> dict:
        with self.cond:
            self.seq += 1
            e: dict = dict(seq=self.seq, t=round(self.clock() - self.t0, 6), kind=kind, level=level, message=message)
            if data:
                e["data"] = data
            self.buf.append(e)
            if self.file:
                self.file.write(json.dumps(e, default=_plain) + "\n")
            self.cond.notify_all()
            return e

    def since(self, seq: int = 0, kinds=None, min_level: str = "info") -> list[dict]:
        floor = LEVELS.index(min_level)
        with self.lock:
            return [e for e in self.buf if e["seq"] > seq and (kinds is None or e["kind"] in kinds)
                    and LEVELS.index(e["level"]) >= floor]

    def wait(self, seq: int, timeout: float) -> list[dict]:
        """Block until something newer than seq arrives (or timeout); return what is new."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq > seq, timeout)
        return self.since(seq)

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


def _plain(o):
    try:
        return o.tolist()
    except AttributeError:
        return str(o)
