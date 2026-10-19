"""Regression checks for the measured pushing example's physical and data contracts."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from world_use.bodies.mujoco_scene import mj
from world_use.tasks import digest

HERE = Path(__file__).resolve().parents[1] / "examples" / "scene-validation"
sys.path.insert(0, str(HERE))
try:
    spec = importlib.util.spec_from_file_location("measured_scene", HERE / "measured.py")
    measured = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measured)
finally:
    sys.path.remove(str(HERE))


@pytest.mark.parametrize("floor_friction", [0.1, 0.8])
def test_probe_contact_friction_is_independent_of_fitted_floor(floor_friction):
    geometry = measured.REFERENCE["rect1_geometry"]
    model = measured.build_model([0.04965, 0], [0, 0, 0], floor_friction, geometry)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    probe_id = model.geom("probe").id
    probe_contacts = [contact for contact in data.contact if probe_id in (contact.geom1, contact.geom2)]
    assert probe_contacts
    assert all(np.allclose(contact.friction[:2], 0.25) for contact in probe_contacts)
    support_id = model.geom("support").id
    floor_contacts = [contact for contact in data.contact if support_id in (contact.geom1, contact.geom2)]
    assert floor_contacts
    assert all(np.allclose(contact.friction[:2], floor_friction) for contact in floor_contacts)


def test_dynamic_probe_has_prescribed_velocity_and_fixed_contact_height():
    tip = np.array([[0, 0.2, 0, 0], [0.15, 0.215, 0, 0]])
    samples, state = measured.simulate(tip, [0, 0, 0], 0.15, 0.4)
    moving = samples[:, 0] > 0.03
    assert np.max(np.abs(state["probe_velocity"][moving, 0] - 0.1)) < 0.005
    assert np.max(np.abs(state["probe_xy"][:, 0] - (0.2 + samples[:, 0] * 0.1))) < 0.001
    model = measured.build_model([0.04965, 0.025], [0, 0, 0], 0.4,
                                 measured.REFERENCE["rect1_geometry"])
    data = mj.MjData(model)
    probe = model.geom("probe").id
    for _ in range(200):
        data.mocap_pos[0, 0] -= 0.000025
        mj.mj_step(model, data)
    mj.mj_forward(model, data)
    assert data.geom_xpos[probe, 2] - model.geom_size[probe, 1] == pytest.approx(0.007)
    assert data.geom_xmat[probe].reshape(3, 3) == pytest.approx(np.eye(3))
    assert np.linalg.norm(data.geom_xpos[probe, :2] - data.mocap_pos[0, :2]) < 0.001
    assert data.xpos[model.body("object").id, 0] < -0.001


def test_loader_aligns_asynchronous_clocks_and_unwraps_yaw(tmp_path):
    path = tmp_path / "measured.json"
    path.write_text(json.dumps(dict(
        tip_pose=[[10, 0, 0, 0], [12, 0, 0, 0]],
        object_pose=[[9, 0, 0, 3.1], [11, 0, 0, -3.1], [13, 0, 0, -3.0]])))
    sha = digest(path.read_bytes())
    tip, obj, initial, duration, origin = measured.load(path, sha)
    assert origin == 10 and duration == 2
    assert tip[:, 0] == pytest.approx([0, 2])
    assert obj[:, 0] == pytest.approx([1])
    assert initial[2] == pytest.approx(np.pi)
    assert obj[0, 3] > np.pi
    with pytest.raises(ValueError, match="hash"):
        measured.load(path, "incorrect")
