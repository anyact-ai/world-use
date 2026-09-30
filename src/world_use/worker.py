"""CPU-heavy robot calculations in a persistent spawned process, without hardware handles.

Rehearsal, reach probes and per-step preparation share this worker. Requests are serialized on a helper
thread; the control loop only submits a snapshot and polls a Future. A timeout kills the process through
multiprocessing's public API, which works on Python 3.13 and later.
"""
from __future__ import annotations

import multiprocessing
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import plan, views
from .behaviors import REGISTRY, PathBehavior, build
from .errors import Refused


def _signature():
    return sorted((key, cls.__module__, cls.__qualname__) for key, cls in REGISTRY.items())


def _serve(conn):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    parent = multiprocessing.parent_process()
    if parent is not None:
        def orphaned():
            parent.join()
            os._exit(0)
        threading.Thread(target=orphaned, daemon=True).start()
    from . import bodies
    from .kinematics import Chain
    for m in bodies.manifests().values():
        Chain(m.urdf, m.tool_link)
    conn.send((os.getpid(), _signature()))
    try:
        while (request := conn.recv()) is not None:
            fn, args = request
            try:
                result = (True, fn(*args))
            except Exception as e:
                result = (False, e)
            conn.send(result)
    except EOFError:
        pass
    finally:
        conn.close()


def _check(snap: plan.Snapshot, spec, timeout_s: float) -> plan.Report:
    return plan.rehearse(spec, plan.twin_from(snap), timeout_s)


def _reach_line(snap: plan.Snapshot) -> str:
    t = plan.twin_from(snap)
    t.cmd.q = np.asarray(snap.q_cmd, float)
    return views.reach_line(t)


def _prepare(snap: plan.Snapshot, spec) -> dict:
    t = plan.twin_from(snap)
    t.cmd.q = np.asarray(snap.q_cmd, float)
    b = build(spec)
    if not isinstance(b, PathBehavior):
        raise Refused("only path behaviors need trajectory preparation", "spec")
    b.prepare(t)
    return vars(b)


class Rehearser:
    """Snapshot-only computation. Unsupported extensions fail explicitly; never fall back to the control process."""

    def __init__(self, timeout_s: float = 120.0):
        self.timeout_s = timeout_s
        self._closed = threading.Event()
        self._executor = ThreadPoolExecutor(1, thread_name_prefix="robot-planning")
        self._start()

    def _start(self):
        ctx = multiprocessing.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._process = ctx.Process(target=_serve, args=(child,), daemon=True)
        self._process.start()
        child.close()
        if not self._conn.poll(120):
            self._kill()
            raise RuntimeError("the planning worker did not start")
        self.pid, self._steps = self._conn.recv()

    def _kill(self):
        if self._process.is_alive():
            self._process.kill()
        self._process.join()
        self._conn.close()

    def _call(self, k, fn, *args):
        for attempt in (1, 2):
            if self._closed.is_set():
                raise Refused("the planning worker is closed", "worker_closed")
            try:
                self._conn.send((fn, args))
                if not self._conn.poll(self.timeout_s):
                    self._kill()
                    self._start()
                    raise Refused(f"planning did not finish in {self.timeout_s:.0f} s; this step did not start",
                                  "rehearsal_timeout", "split the plan into shorter phases")
                ok, result = self._conn.recv()
            except (EOFError, BrokenPipeError, ConnectionError):
                if self._closed.is_set():
                    raise Refused("the planning worker is closed", "worker_closed") from None
                self._kill()
                self._start()
                k.emit("rehearser", f"the planning worker died; started another (pid {self.pid})", "warn")
                if attempt == 2:
                    raise RuntimeError("the planning worker died twice; this step did not start") from None
                continue
            if not ok:
                raise result
            return result
        raise AssertionError("unreachable")

    def _snapshot(self, k, snap=None):
        snap = snap or plan.snapshot(k)
        if snap.body is None or self._steps != _signature():
            raise Refused("the worker cannot rebuild this body or its behaviors", "worker_extension",
                          "register the extension in an importable module; use plan.check for embedded offline work")
        return snap

    def check(self, spec, k, timeout_s: float = 900.0, *, snap: plan.Snapshot | None = None) -> plan.Report:
        if isinstance(spec, plan.Plan):
            spec = spec.spec()
        return self._executor.submit(self._call, k, _check, self._snapshot(k, snap), spec, timeout_s).result()

    def reach_line(self, k) -> str:
        return self._executor.submit(self._call, k, _reach_line, self._snapshot(k)).result()

    def prepare(self, spec, k):
        snap = self._snapshot(k)
        return self._executor.submit(self._call, k, _prepare, snap, spec), snap

    def close(self):
        self._closed.set()
        self._kill()
        self._executor.shutdown(wait=True, cancel_futures=True)
