import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from world_use import Kernel, World, fit, records
from world_use.bodies.sim import SimBody
from world_use.client import Client
from world_use.config import load_robot, load_workcell
from world_use.daemon import Daemon, apply_workcell, make_body, session_identity

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "adapters"


def test_external_robot_through_daemon_worker_and_portable_record(tmp_path, monkeypatch, rehearser):
    setup = tmp_path / "setup"
    shutil.copytree(EXAMPLE, setup)
    monkeypatch.syspath_prepend(str(setup))
    cell = load_workcell(setup / "workcell.toml")
    cell["frame"][0]["origin"] = [.01, .02, .03]
    body = make_body(cell["body"], cell)
    k = Kernel(body, World(), run_dir=tmp_path / "run")
    k.connect()
    apply_workcell(cell, k)
    identity = session_identity(cell["body"], cell)
    d = Daemon(k, port=0, rehearser=rehearser, session=identity, config=cell)
    d.start()
    c = Client(f"http://127.0.0.1:{d.http.server_port}")
    try:
        assert len(c.status()["joints_deg"]) == 2
        assert np.allclose(k.world.frame("work").T[:3, 3], [.01, .02, .03])
        phase = {"do": "joints", "delta_deg": {"1": 10, "2": -5}}
        c.enable()
        assert c.check(phase)["ok"]
        assert c.run(phase, wait=10)["status"] == "done"
        c.release()
        c.shutdown()
    finally:
        d.stop_loop.set()
        d.control.join(timeout=2)
        d.http.shutdown()
        d.http.server_close()
        if k.journal._thread.is_alive():
            k.close()

    sim = make_body("sim", cell, World())
    assert sim.manifest.n == 2 and np.allclose(sim.q, np.radians([0, 28.65]))
    # A changed model at the same path is a different startup configuration.
    model_path = setup / "planar.toml"
    model_path.write_text(model_path.read_text().replace("v_max = 0.8", "v_max = 0.4"))
    assert session_identity(cell["body"], cell) != identity
    shutil.rmtree(setup)
    archived = tmp_path / "archived"
    shutil.move(tmp_path / "run", archived)
    assert records.inspect(archived)["closed"]
    assert fit.robot_of([archived]).joints[0].v_max == .8
    assert records.replay(archived, tmp_path / "replay.gif").stat().st_size > 1000


def test_workcell_paths_and_typos_are_not_silently_ignored(tmp_path, monkeypatch):
    assert load_workcell("block")["body"] == "sim"
    with pytest.raises(FileNotFoundError):
        load_workcell(tmp_path / "missing" / "block")
    path = tmp_path / "bench.toml"
    path.write_text('fit = "fit.json"\n[[camera]]\nname = "side"\npath = "images/side.jpg"\n')
    monkeypatch.chdir(tmp_path.parent)
    cell = load_workcell(path)
    assert cell["fit"] == str(tmp_path / "fit.json")
    assert cell["camera"][0]["path"] == str(tmp_path / "images/side.jpg")
    path.write_text('[[camrea]]\nname = "side"\n')
    with pytest.raises(ValueError, match="unknown fields: camrea"):
        load_workcell(path)
    cell = load_workcell(EXAMPLE / "workcell.toml")
    cell["simulation"]["lag_seconds"] = .1
    with pytest.raises(ValueError, match="lag_seconds"):
        make_body("sim", cell)


@pytest.mark.parametrize(("old", "new", "message"), [
    ("self_supporting = true", "self_supporting = false", "rest"),
    ('name = "shoulder"', 'name = "wrong_joint"', "URDF chain order"),
    ("lower = -2.5", "lower = -3.0", "exceed the URDF limits"),
    ("v_max = 0.8", "v_max = nan", "finite number"),
])
def test_invalid_robot_descriptions_fail_before_a_driver_is_loaded(tmp_path, old, new, message):
    shutil.copy(EXAMPLE / "planar.urdf", tmp_path)
    path = tmp_path / "robot.toml"
    path.write_text((EXAMPLE / "planar.toml").read_text().replace(old, new))
    with pytest.raises(ValueError, match=message):
        make_body("no_such_driver:Body", {"robot": str(path)})


def test_prismatic_joints_are_offline_kinematics_only(tmp_path):
    urdf = tmp_path / "linear.urdf"
    urdf.write_text((EXAMPLE / "planar.urdf").read_text().replace('type="revolute"', 'type="prismatic"', 1))
    path = tmp_path / "robot.toml"
    path.write_text((EXAMPLE / "planar.toml").read_text().replace('urdf = "planar.urdf"', 'urdf = "linear.urdf"'))
    body = SimBody(replace(load_robot(EXAMPLE / "planar.toml"), urdf=urdf))
    assert body.chain.fk([.1, 0])[2, 3] == pytest.approx(.3)
    with pytest.raises(ValueError, match="rotational arm joints only; prismatic joints are unsupported"):
        load_robot(path)
    with pytest.raises(ValueError, match="rotational arm joints only; prismatic joints are unsupported"):
        Kernel(body)


def test_supported_rest_uses_joint_names_and_preserves_release_rules(tmp_path):
    shutil.copy(EXAMPLE / "planar.urdf", tmp_path)
    text = (EXAMPLE / "planar.toml").read_text().replace("self_supporting = true", "self_supporting = false")
    path = tmp_path / "robot.toml"
    path.write_text(text + '\n[rest]\nq = [0.0, 0.5]\njoints = ["elbow"]\ntol = 0.1\n')
    model = load_robot(path)
    assert model.rest.holds([1, .5]) and not model.rest.holds([0, .7])
