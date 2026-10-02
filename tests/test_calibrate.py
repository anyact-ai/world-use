"""Calibrating a camera from the arm: the tool visits a box, the policy says where it sees the tool point."""
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import make_kernel

from world_use import Outcome, Refused, cameras
from world_use.calibrate import GRAY, solve, tour
from world_use.cameras import View

TRUE = View.look_at([0.45, -0.75, 0.55], [0.30, 0.0, 0.30], 55.0, (1024, 576))
BOX = np.array([[0.30 + (2 * f - 1) * 0.12, (2 * s - 1) * 0.12, 0.30 + 0.18 * u] for f, s, u in GRAY])
WORKSPACE = np.array([[f, s, u] for f in (0.2, 0.3, 0.4) for s in (-0.1, 0.0, 0.1) for u in (0.2, 0.3, 0.4)])


def _miss(view, truth=TRUE) -> float:
    return float(np.median(np.linalg.norm(view.project(WORKSPACE)[0] - truth.project(WORKSPACE)[0], axis=1)))


def test_a_camera_is_found_from_where_the_tool_was_seen():
    px = TRUE.project(BOX)[0]
    fit = solve(BOX, px, (1024, 576), fov_deg=60.0)
    assert fit.rms < 2 and _miss(fit.view) < 2
    noisy = px + np.random.default_rng(1).normal(0, 12, px.shape)
    assert _miss(solve(BOX, noisy, (1024, 576), fov_deg=60.0).view) < 12
    known = solve(BOX, px, (1024, 576), focal=TRUE.fx)
    assert known.rms < 0.5 and np.linalg.norm(known.C - TRUE.T[:3, 3]) < 0.005


def test_a_wild_answer_is_left_out():
    px = TRUE.project(BOX)[0] + np.random.default_rng(2).normal(0, 5, (8, 2))
    px[3] += [150, -110]
    fit = solve(BOX, px, (1024, 576), fov_deg=60.0)
    assert fit.left_out == [3] and _miss(fit.view) < 12


def test_what_cannot_fix_a_camera_is_refused():
    px = TRUE.project(BOX)[0]
    top = [i for i, (_, _, u) in enumerate(GRAY) if u == 1]
    with pytest.raises(Refused, match="at least 6"):
        solve(BOX[top], px[top], (1024, 576))
    flat = BOX.copy()
    flat[:, 2] = 0.3
    with pytest.raises(Refused, match="one plane"):
        solve(flat, TRUE.project(flat)[0], (1024, 576))
    with pytest.raises(Refused, match="on the arm"):                   # the tool stays put in the picture
        solve(BOX, np.full((8, 2), 500.0) + np.random.default_rng(0).normal(0, 3, (8, 2)), (1024, 576))


def test_a_tour_on_the_simulator_calibrates_a_camera_that_sees_it():
    """Answers from the simulated camera's true lens, scattered like a model's (10 px), recover it."""
    k = make_kernel()
    assert k.run({"do": "line", "forward": 0.04, "up": 0.10}).ok
    front = cameras.sim_cameras(k.body, k.world)["front"]
    steps, a = tour(k, "front", (800, 600), 8)
    job = k.submit(steps)
    rng = np.random.default_rng(0)
    while not job.finished:
        k.tick()
        k.clock.wait()
        if job.status == "waiting":
            (u,), _ = front.lens.project([k.chain.fk(k.state.q)[:3, 3]])
            u = u + rng.normal(0, 10, 2)
            k.answer(job.id, f"{u[0]:.0f},{u[1]:.0f}" if 0 <= u[0] <= 800 and 0 <= u[1] <= 600 else "unseen")
    assert job.outcome.ok and a in (0.12, 0.09, 0.06)
    answers = job.outcome.data["answers"]
    seen = [x for x in answers if x["answer"] != "unseen"]
    fit = solve([k.world.to_base("work", x["tool"]) for x in seen],
                [[float(v) for v in x["answer"].split(",")] for x in seen], (800, 600), fov_deg=60.0)
    grid = np.array([k.world.to_base("work", p) for p in WORKSPACE])
    miss = np.median(np.linalg.norm(fit.view.project(grid)[0] - front.lens.project(grid)[0], axis=1))
    assert len(seen) >= 6 and fit.rms < 16 and miss < 15


def test_the_daemon_calibrates_a_camera_and_installs_it(client, daemon):
    d, c = daemon
    lens = d.cameras["side"].lens
    assert c.run({"do": "line", "forward": 0.04, "up": 0.10}, wait=20)["status"] == "done"
    r = c.calibrate("side", points=6, spread=0.05, wait=30)
    while r["status"] == "waiting":
        assert "nothing else drawn, because this camera is being calibrated" in c.look("side")["drawn"]
        (u,), _ = lens.project([d.k.chain.fk(d.k.state.q)[:3, 3]])
        r = c.answer(r["id"], f"{u[0]:.1f},{u[1]:.1f}", wait=30)
    text = r["calibration"]["text"]
    assert r["status"] == "done" and r["calibration"]["installed"], text
    assert 'frame = "base"' in text and d.cameras["side"].view is not lens
    assert "magenta" in c.look("side")["drawn"] and "camera.side" in c.world()["facts"]


def test_the_daemon_fits_a_large_360_cut_from_resized_answers(daemon):
    d, _ = daemon
    cut = cameras.EquirectCut("panorama", d.cameras["side"], fov_deg=55.0, size=(1920, 1080))
    d.cameras[cut.name] = cut
    pixels = TRUE.project(BOX)[0]                     # answers from the 1024x576 picture saved by look
    answers = [dict(tool=d.k.world.from_base("work", point).tolist(), answer=f"{x:.6f},{y:.6f}")
               for point, (x, y) in zip(BOX, pixels, strict=True)]
    job = SimpleNamespace(id=123, outcome=Outcome("done", "seq", "calibrated", dict(answers=answers)))
    result = d._fit(job, cut.name, (1024, 576))
    assert result["installed"], result["text"]
    assert cut.view is not None
    assert _miss(cut.view, TRUE.scaled(1920, 1080)) < 0.5
