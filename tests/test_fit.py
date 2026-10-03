"""A robot model fitted from flight records (fit.py): what the motors measured, not what the URDF says."""
import json
import threading
from dataclasses import replace

import numpy as np
import pytest
from conftest import Q_REST, make_kernel

from world_use import Kernel, RealClock, VirtualClock, World, bodies, card, check, cli, fit, twin
from world_use.bodies.rebot import MANIFEST
from world_use.daemon import apply_workcell, load_workcell
from world_use.kinematics import Chain


def truth() -> fit.Model:
    """The 'real' reBot of these tests: its forearm 20% heavier than the URDF says, with the centre of mass 2 cm
    further out, and friction in every joint. Close to what the physical elbow showed (2026-09-28)."""
    chain = Chain(MANIFEST.urdf, MANIFEST.tool_link)
    links = {name: (m, list(c)) for name, (m, c) in chain.links.items()}
    m, c = links["link3"]
    links["link3"] = (1.2 * m, [c[0] + 0.02, c[1], c[2]])
    return fit.Model(MANIFEST.name, links, [(0.3, 0.5), (0.2, 1.0), (0.5, 1.5), (0.2, 0.5), (0.1, 0.2), (0.05, 0.1)])


TOURS = {
    "a": [{"2": 55, "3": 54, "4": -19}, {"3": 70}, {"2": 40}, {"4": 10}, {"2": 55, "3": 54, "4": -19}],
    "b": [{"2": 45, "3": 60, "4": -30}, {"2": 70, "3": 50}, {"4": -50}, {"3": 75, "4": -10}],
    "c": [{"2": 50, "3": 60, "4": -20}, {"5": 30}, {"5": -30}, {"2": 35, "3": 65, "5": 0}],
}


def record(folder, name) -> str:
    """A flight record from the 'real' robot, driven by a kernel that only knows the URDF."""
    world = World()
    body = bodies.make("sim", world, q=Q_REST, gripper=1.0)
    body.use_fit(truth())
    k = Kernel(body, world, VirtualClock(100.0), run_dir=folder / name)
    k.connect()
    k.enable()
    for target in TOURS[name]:
        out = k.run({"do": "joints", "target_deg": target, "speed": 0.25})
        assert out.ok, out.message
    k.save_record()
    return folder / name


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    folder = tmp_path_factory.mktemp("runs")
    return [record(folder, name) for name in TOURS]


def test_the_fit_finds_the_heavier_forearm_and_says_how_well_it_predicts(records):
    model = fit.fit(records, MANIFEST)
    assert model.records == ["a", "b", "c"] and model.samples > 100
    elbow = model.check["joint3"]                         # each record predicted by a fit made without it
    assert elbow["fit"] < 0.5 * elbow["urdf"], model.describe()
    # a pose none of the tours visited: the fit weighs the arm as the robot does, the URDF does not
    q = np.radians([0, 50, 45, -35, 10, 0])
    real, urdf, fitted = (Chain(MANIFEST.urdf, MANIFEST.tool_link) for _ in range(3))
    truth().apply(real)
    model.apply(fitted)
    miss_urdf, miss_fit = (abs(c.gravity(q)[2] - real.gravity(q)[2]) for c in (urdf, fitted))
    assert miss_urdf > 0.3 and miss_fit < 0.3 * miss_urdf
    assert abs(model.friction[2][0] - 0.5) < 0.2          # the elbow's Coulomb friction
    assert all(v >= 0 for _, v in model.friction)          # no viscous friction below zero
    assert "joint3" in model.describe(urdf) and "link3" in model.describe(urdf)


def test_a_record_of_another_robot_or_never_powered_is_not_used(records, tmp_path):
    (tmp_path / "empty").mkdir()
    other = replace(MANIFEST, name="another arm")
    with pytest.raises(ValueError, match="no record"):
        fit.fit(records, other)
    assert fit.samples(tmp_path / "empty", MANIFEST) is None


def test_a_model_round_trips_and_changes_what_the_chain_weighs(tmp_path):
    truth().save(tmp_path / "fit.json")
    model = fit.load(tmp_path / "fit.json")
    assert model.to_dict() == truth().to_dict()
    chain = Chain(MANIFEST.urdf, MANIFEST.tool_link)
    q = np.radians([0, 55, 54, -19, 0, 0])
    before = chain.gravity(q)[2]
    model.apply(chain)
    assert chain.gravity(q)[2] > before + 0.3
    assert np.allclose(model.friction_torque([0, 0, 0.2, 0, 0, 0])[2], 0.5 * np.tanh(20) + 1.5 * 0.2)


def test_the_kernel_judges_torque_by_the_fit_and_so_do_its_body_and_twin():
    k = make_kernel()
    with pytest.raises(ValueError, match="another arm"):
        k.use_fit(replace(truth(), body="another arm"))
    k.use_fit(truth())
    assert k.fit is not None
    assert k.body.model.body("link3").mass[0] == pytest.approx(truth().links["link3"][0])
    assert any(e["kind"] == "fit" for e in k.events.since(0))
    q = np.radians([0, 55, 54, -19, 0, 0])
    k.cmd.dq = np.array([0, 0, 0.2, 0, 0, 0])
    assert np.isclose(k.expected_torque(q)[2] - k.chain.gravity(q)[2], truth().friction_torque(k.cmd.dq)[2])
    t = twin(k)                                           # rehearsals weigh the arm the same way
    assert t.fit is not None and np.allclose(t.chain.gravity(q), k.chain.gravity(q))
    k.cmd.dq = np.zeros(6)
    assert check([{"do": "joints", "target_deg": {"2": 55, "3": 54, "4": -19}, "speed": 0.3}], k).ok


def test_a_workcell_names_its_fit_beside_itself(tmp_path):
    truth().save(tmp_path / "rebot-fit.json")
    (tmp_path / "bench.toml").write_text('fit = "rebot-fit.json"\n')
    k = make_kernel()
    apply_workcell(load_workcell(tmp_path / "bench.toml"), k)
    assert k.fit is not None and k.fit.body == MANIFEST.name


def test_a_fit_taken_while_the_loop_runs_lands_between_ticks():
    world = World()
    body = bodies.make("sim", world, q=Q_REST, gripper=1.0)
    k = Kernel(body, world, RealClock(100.0))
    k.connect()
    stop = threading.Event()
    loop = threading.Thread(target=k.loop, args=(stop,), daemon=True)
    loop.start()
    seen = []
    body.use_fit = lambda model: seen.append(threading.current_thread())
    try:
        k.use_fit(truth())
        assert seen == [loop] and k.fit is not None and loop.is_alive()
    finally:
        stop.set()
        loop.join(2.0)


def test_the_card_says_which_torque_model_is_in_use():
    k = make_kernel()
    assert "torque model" not in card(k)
    k.use_fit(truth())
    assert "torque model: robot model fitted" in card(k)


def test_wu_fit_finds_the_robot_from_the_records_and_writes_the_model(records, tmp_path, capsys):
    recorded = fit.robot_of(records)
    assert recorded.joints == MANIFEST.joints and recorded.urdf.read_bytes() == MANIFEST.urdf.read_bytes()
    out = tmp_path / "fit.json"
    assert cli.main(["fit", *map(str, records), str(tmp_path / "no-record"), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "each record predicted by a fit made without it" in text and f'fit = "{out}"' in text
    assert fit.load(out).records == ["a", "b", "c"]
    path = records[0] / "session.json"
    original = path.read_text()
    other = json.loads(original)
    other["initial"]["model"]["name"] = "another arm"
    path.write_text(json.dumps(other))
    try:
        with pytest.raises(ValueError, match="one robot at a time"):
            fit.robot_of(records)
    finally:
        path.write_text(original)
