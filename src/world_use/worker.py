"""Rehearsals in a worker process, so a plan check cannot take the control loop's ticks.

A rehearsal is pure computation: inverse kinematics and a simulated kernel ticking through the whole plan, 0.2 to
0.6 s of CPU for a small one. In the daemon's own process it competes with the 100 Hz control loop for the
interpreter. On the physical reBot every `wu run` stalled the loop 140-300 ms while the arm held a raised pose. On
the simulator the loop's p99 tick went from 12.4 ms to 54.7 ms during three checks in the same process, and stayed
at 12.4 ms with the checks in a worker.

The worker gets a snapshot of the kernel (plain data) and sends back a report, so it never touches the robot. It is
started with spawn, so it inherits no CAN handles or threads, ignores the terminal's Ctrl+C (the daemon decides when
to stop) and exits when the daemon does, however the daemon ends.
"""
import multiprocessing
import os
import signal
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np

from . import plan, views
from .behaviors import REGISTRY
from .errors import Refused


def _init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    parent = multiprocessing.parent_process()
    if parent is not None:
        def orphaned():
            parent.join()
            os._exit(0)
        threading.Thread(target=orphaned, daemon=True).start()


def _warm() -> tuple[int, list[str]]:
    """Build every registered robot once, so the first real check is quick; say who we are and what steps we know."""
    from . import bodies
    from .kinematics import Chain
    for m in bodies.manifests().values():
        Chain(m.urdf, m.tool_link)
    return os.getpid(), sorted(REGISTRY)


def _check(snap: plan.Snapshot, spec, timeout_s: float) -> plan.Report:
    return plan.rehearse(spec, plan.twin_from(snap), timeout_s)


def _reach_line(snap: plan.Snapshot) -> str:
    t = plan.twin_from(snap)
    t.cmd.q = np.asarray(snap.q_cmd, float)          # the kernel plans from its command, not the measured pose
    return views.reach_line(t)


class Rehearser:
    """Checks plans and probes reach in one persistent worker process, from snapshots. It keeps no robot state, so
    one serves any number of kernels. A body the worker cannot rebuild (a manifest that is not registered, or steps
    registered in this process but not there) is rehearsed here instead. A worker that dies is replaced once; one
    that runs past `timeout_s` is killed, replaced, and the plan refused."""

    def __init__(self, timeout_s: float = 120.0):
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._start()

    def _start(self):
        self._pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"), initializer=_init)
        self.pid, self._steps = self._pool.submit(_warm).result(timeout=120)

    def _restart(self):
        with self._lock:
            self._pool.kill_workers()
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._start()

    def _here(self, snap: plan.Snapshot) -> bool:
        return snap.body is None or self._steps != sorted(REGISTRY)

    def _call(self, k, fn, *args):
        for attempt in (1, 2):
            try:
                return self._pool.submit(fn, *args).result(timeout=self.timeout_s)
            except BrokenProcessPool:
                self._restart()
                k.emit("rehearser", f"the rehearsal worker died; started another (pid {self.pid})", "warn")
                if attempt == 2:
                    raise RuntimeError("the rehearsal worker died twice on this plan; nothing moved") from None
            except TimeoutError:
                self._restart()
                raise Refused(f"the rehearsal did not finish in {self.timeout_s:.0f} s, so nothing moved",
                              "rehearsal_timeout", "split the plan into shorter phases") from None
        raise AssertionError("unreachable")

    def check(self, spec, k, timeout_s: float = 900.0, *, snap: plan.Snapshot | None = None) -> plan.Report:
        """plan.check, in the worker."""
        if isinstance(spec, plan.Plan):
            spec = spec.spec()
        snap = snap or plan.snapshot(k)
        if self._here(snap):
            return plan.rehearse(spec, plan.twin_from(snap, k.manifest), timeout_s)
        return self._call(k, _check, snap, spec, timeout_s)

    def reach_line(self, k) -> str:
        """views.reach_line, in the worker."""
        snap = plan.snapshot(k)
        if self._here(snap):
            return views.reach_line(k)
        return self._call(k, _reach_line, snap)

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)
