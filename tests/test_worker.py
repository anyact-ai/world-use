"""Rehearsals in a worker process: the daemon's control loop keeps its ticks while plans are checked."""
import dataclasses
import os
import signal

import pytest
from conftest import Q_REST, make_kernel

from world_use import Kernel, Refused, VirtualClock, World, plan, views
from world_use.bodies.rebot import MANIFEST
from world_use.bodies.sim import SimBody
from world_use.worker import Rehearser

PLAN = [{"do": "line", "up": 0.03}, {"do": "checkpoint", "ask": "clear?"}, {"do": "line", "forward": 0.02}]


def test_the_daemon_rehearses_in_the_worker(daemon, monkeypatch):
    """Sabotage rehearsing in this process: checks, runs, refusals and the card still work, so they ran elsewhere."""
    def broken(*a, **kw):
        raise AssertionError("rehearsed in the daemon's process")
    monkeypatch.setattr(plan, "rehearse", broken)
    monkeypatch.setattr(plan, "reach", broken)
    _, c = daemon
    assert c.check(PLAN)["ok"]
    r = c.run({"do": "line", "forward": 0.40}, wait=5)
    assert r["status"] == "refused" and "a 3 cm line can go" in r["incident"]
    assert "a 3 cm line can go" in c.card()
    assert c.run({"do": "line", "up": 0.02, "duration": 0.5}, wait=10)["status"] == "done"


def test_the_same_report_as_in_process(rehearser, lifted):
    assert rehearser.check(PLAN, lifted).to_dict() == plan.check(PLAN, lifted).to_dict()
    assert rehearser.reach_line(lifted) == views.reach_line(lifted)


def test_a_body_the_worker_cannot_rebuild_is_refused(rehearser):
    custom = dataclasses.replace(MANIFEST, name="a custom arm")            # not in bodies.manifests()
    world = World()
    k = Kernel(SimBody(custom, world, q=Q_REST, gripper=1.0), world, VirtualClock(100.0))
    k.connect()
    k.enable()
    with pytest.raises(Refused, match="cannot rebuild"):
        rehearser.check(PLAN, k)
    assert plan.check(PLAN, k).ok          # still available for embedded offline extensions


def test_a_worker_that_dies_is_replaced():
    r = Rehearser()
    try:
        k = make_kernel()
        first = r.pid
        os.kill(first, signal.SIGKILL)
        assert r.check(PLAN, k).ok and r.pid != first
        assert any(e["kind"] == "rehearser" for e in k.events.since(0))
    finally:
        r.close()


def test_a_rehearsal_that_runs_too_long_is_refused_and_the_worker_replaced():
    r = Rehearser(timeout_s=0.2)
    try:
        k = make_kernel()
        with pytest.raises(Refused, match="did not finish"):
            r.check({"do": "hold", "seconds": 3000}, k)
        r.timeout_s = 60.0
        assert r.check(PLAN, k).ok
    finally:
        r.close()


def test_preparation_keeps_feedback_and_stop_responsive(k):
    from concurrent.futures import Future
    from types import SimpleNamespace

    import numpy as np

    pending = Future()
    k.planner = SimpleNamespace(prepare=lambda spec, robot: (pending, plan.snapshot(robot)))
    job = k.submit([{"do": "line", "up": 0.03}])
    for _ in range(50):
        k.tick()
        k.clock.wait()
    assert job.status == "running" and not job.behavior.moves
    assert len(k.tape) == 50 and k.feedback_at == pytest.approx(k.clock.now() - k.clock.dt)
    assert np.allclose(k.cmd.q, k.q_start)
    k.stop("operator stop during planning")
    k.tick()
    assert job.status == "stopped" and not pending.done()


def test_prepared_path_is_refused_after_scene_change(k, rehearser):
    from types import SimpleNamespace

    k.planner = SimpleNamespace(prepare=lambda spec, robot: rehearser.prepare(spec, robot))
    job = k.submit({"do": "line", "up": 0.03})
    while job.status == "queued":
        k.tick()
        k.clock.wait()
    job.behavior._pending.result(timeout=10)
    k.world.add_box("new obstacle", "keep_out", [1, 1, 1], [.1, .1, .1])
    k.tick()
    assert job.status == "refused" and "changed" in job.outcome.message


def test_execution_paths_are_prepared_outside_the_control_process(daemon, monkeypatch):
    from world_use.behaviors import Line

    def fail_here(*args):
        raise AssertionError("planning on the control thread")

    monkeypatch.setattr(Line, "plan", fail_here)
    _, c = daemon
    r = c.run([{"do": "line", "up": 0.06},
               {"do": "guarded", "up": 0.01, "expect_contact": False}], check=False, wait=10)
    assert r["status"] == "done", r
