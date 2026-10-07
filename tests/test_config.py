import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from conftest import serving

from world_use import World, fit, records
from world_use.bodies import manifests
from world_use.bodies.sim import SimBody
from world_use.config import load_robot, load_workcell, manifest_data, manifest_from_data
from world_use.daemon import make_body, session_identity
from world_use.robot_assets import archive, resolve

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "adapters"


@pytest.fixture
def external_robot_record(tmp_path, monkeypatch, rehearser):
    setup = tmp_path / "setup"
    shutil.copytree(EXAMPLE, setup)
    monkeypatch.syspath_prepend(str(setup))
    cell = load_workcell(setup / "workcell.toml")
    cell["frame"][0]["origin"] = [.01, .02, .03]
    with serving(tmp_path, cell, rehearser) as (d, c):
        assert len(c.status()["joints_deg"]) == 2
        assert np.allclose(d.k.world.frame("work").T[:3, 3], [.01, .02, .03])
        phase = {"do": "joints", "delta_deg": {"1": 10, "2": -5}}
        assert c.check(phase)["ok"]
        assert c.run(phase, wait=10)["status"] == "done"
        c.release()

    sim = make_body("sim", cell, World())
    assert sim.manifest.n == 2 and np.allclose(sim.q, np.radians([0, 28.65]))
    # A changed model at the same path is a different startup configuration.
    model_path = setup / "planar.toml"
    model_path.write_text(model_path.read_text().replace("v_max = 0.8", "v_max = 0.4"))
    assert session_identity(cell["body"], cell) != d.session
    shutil.rmtree(setup)
    archived = tmp_path / "archived"
    shutil.move(tmp_path / "run", archived)
    return archived


def test_external_robot_through_daemon_worker_and_portable_record(external_robot_record):
    assert records.inspect(external_robot_record)["closed"]
    assert fit.robot_of([external_robot_record]).joints[0].v_max == .8


@pytest.mark.rendering
def test_external_robot_replays_after_the_original_model_is_removed(external_robot_record, tmp_path):
    assert records.replay(external_robot_record, tmp_path / "replay.gif").stat().st_size > 1000


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
    path.write_text('[[box]]\nname = "cup"\nkind = "object"\ncenter = [0.3, 0, 0.05]\nsize = [0.05, 0.05, 0.1]\n'
                    'mass_kg = 0.2\nfriction = 0.5\n')
    assert load_workcell(path)["box"][0]["mass_kg"] == .2
    cell = load_workcell(EXAMPLE / "workcell.toml")
    cell["simulation"]["lag_seconds"] = .1
    with pytest.raises(ValueError, match="lag_seconds"):
        make_body("sim", cell)


@pytest.mark.parametrize("settings", [
    "max_age_s = nan",                           # bypassed the stale-frame check
    'max_age_s = "3"',                           # silently coerced a string
    "rotate = 90.5",                             # truncated to a valid rotation
    'projection = "equirec"',                    # silently became a pinhole camera
    "fov_deg = 180",                             # degenerate pinhole projection
    "size = [512, 0]",
    "eye = [0, 0, 1]",                           # ignored an incomplete calibration
    'projection = "equirect"\nlook_at = [0, 0, 0]',
])
def test_invalid_camera_configuration_fails_when_loading_the_workcell(tmp_path, settings):
    path = tmp_path / "camera.toml"
    path.write_text('[[camera]]\nname = "side"\npath = "side.png"\n' + settings + "\n")
    with pytest.raises(ValueError):
        load_workcell(path)


def test_camera_configuration_keeps_simulated_and_uncalibrated_360_views(tmp_path):
    from world_use.cameras import EquirectCut, from_config, view_from_config

    path = tmp_path / "cameras.toml"
    path.write_text('[[camera]]\nname="sim"\nframe="base"\neye=[0,0,1]\nlook_at=[0,0,0]\n'
                    '[[camera]]\nname="panorama"\npath="pano.png"\nprojection="equirect"\nyaw_deg=30\n')
    simulated, panorama = load_workcell(path)["camera"]
    assert view_from_config(simulated, World()) is not None
    cut = from_config(panorama, World())
    assert isinstance(cut, EquirectCut) and cut.view is None


@pytest.mark.parametrize(("old", "new", "message"), [
    ("self_supporting = true", "self_supporting = false", "rest"),
    ('name = "shoulder"', 'name = "wrong_joint"', "URDF chain order"),
    ("lower = -2.5", "lower = -3.0", "exceed the URDF limits"),
    ("v_max = 0.8", "v_max = nan", "finite number"),
    ('sensing = ["position"]', 'sensing = ["position"]\nik_weights = [1, 1, 1]', "ik_weights"),
])
def test_invalid_robot_descriptions_fail_before_a_driver_is_loaded(tmp_path, old, new, message):
    shutil.copy(EXAMPLE / "planar.urdf", tmp_path)
    path = tmp_path / "robot.toml"
    path.write_text((EXAMPLE / "planar.toml").read_text().replace(old, new))
    with pytest.raises(ValueError, match=message):
        make_body("no_such_driver:Body", {"robot": str(path)})


@pytest.mark.parametrize("extra", ["dtau = true", 'dtau = "0.03"', "dtau = nan", "datu = 0.03",
                                   "known = 1", "speed = 0.01"])
def test_invalid_workcell_boxes_are_rejected_during_loading(tmp_path, extra):
    path = tmp_path / "cell.toml"
    path.write_text('body = "no_such_driver:Body"\n[[box]]\nname = "glass"\nkind = "fragile"\n'
                    'center = [0, 0, 0]\nsize = [1, 1, 1]\n' + extra)
    with pytest.raises(ValueError):
        load_workcell(path)


def test_prismatic_arm_joints_are_rejected(tmp_path):
    urdf = tmp_path / "linear.urdf"
    urdf.write_text((EXAMPLE / "planar.urdf").read_text().replace('type="revolute"', 'type="prismatic"', 1))
    path = tmp_path / "robot.toml"
    path.write_text((EXAMPLE / "planar.toml").read_text().replace('urdf = "planar.urdf"', 'urdf = "linear.urdf"'))
    with pytest.raises(ValueError, match="rotational arm joints only; prismatic joints are unsupported"):
        load_robot(path)
    with pytest.raises(ValueError, match="rotational arm joints only; prismatic joints are unsupported"):
        SimBody(replace(load_robot(EXAMPLE / "planar.toml"), urdf=urdf))


def test_supported_rest_uses_joint_names_and_preserves_release_rules(tmp_path):
    shutil.copy(EXAMPLE / "planar.urdf", tmp_path)
    text = (EXAMPLE / "planar.toml").read_text().replace("self_supporting = true", "self_supporting = false")
    path = tmp_path / "robot.toml"
    path.write_text(text + '\n[rest]\nq = [0.0, 0.5]\njoints = ["elbow"]\ntol = 0.1\n')
    model = load_robot(path)
    assert model.rest.holds([1, .5]) and not model.rest.holds([0, .7])


def test_gripper_limits_scale_with_its_travel_and_the_worker_gets_the_ik_weights():
    data = manifest_data(load_robot(Path(__file__).with_name("jaw_arm.toml")))
    data["gripper"] = dict(closed=0.0, open=0.08, unit="m", m_per_unit=1.0, approach=[0, 0, 1], opens_along=[0, 1, 0])
    data["ik_weights"] = [1, 1, 1, 1, 1, 0]
    model = manifest_from_data(data)
    g = model.gripper
    assert (g.v_max, g.track_tol, g.squeeze) == pytest.approx((.08, .0108, .00088))
    assert manifest_from_data(manifest_data(model)) == model            # the description a worker builds a twin from
    del data["gripper"]["opens_along"]
    with pytest.raises(ValueError, match="opens_along"):
        manifest_from_data(data)


def test_package_mesh_uris_resolve_inside_their_package(tmp_path):
    package = tmp_path / "arm_description"
    (package / "urdf").mkdir(parents=True)
    (package / "meshes").mkdir()
    (package / "meshes" / "link.stl").write_text("solid link\nendsolid link\n")
    urdf = package / "urdf" / "arm.urdf"
    urdf.write_text("<robot/>")
    assert resolve(urdf, "package://arm_description/meshes/link.stl") == package / "meshes" / "link.stl"
    with pytest.raises(ValueError, match="package://other/meshes/link"):
        resolve(urdf, "package://other/meshes/link.stl")


def test_records_copy_only_meshes_that_world_use_does_not_ship(tmp_path):
    source = tmp_path / "arm"
    (source / "meshes").mkdir(parents=True)
    (source / "meshes" / "link.stl").write_text("solid link\nendsolid link\n")
    (source / "arm.urdf").write_text('<robot><link name="a"><visual><geometry><mesh filename="meshes/link.stl"/>'
                                     "</geometry></visual></link></robot>")
    rebot = manifests()["rebot"].urdf
    for urdf, name in ((source / "arm.urdf", "custom"), (rebot, "built-in")):
        folder = tmp_path / name
        folder.mkdir()
        archive(urdf, folder)
        shutil.copyfile(urdf, folder / "robot.urdf")
    shutil.rmtree(source)
    assert resolve(tmp_path / "custom" / "robot.urdf", "meshes/link.stl").read_text().startswith("solid link")
    assert not [path for path in (tmp_path / "built-in").rglob("*") if path.suffix.lower() == ".stl"]
    assert resolve(tmp_path / "built-in" / "robot.urdf", "../meshes/shared/base_link.STL").is_file()
