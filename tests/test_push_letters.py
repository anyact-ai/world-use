"""Regressions found while running the camera-driven pushing example."""
import importlib.util
import sys
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest
from PIL import Image

from world_use.cameras import Frame, View

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "push-letters"


@pytest.fixture
def pushing(monkeypatch):
    modules = {}
    for name in ("scene", "policy", "audit"):
        spec = importlib.util.spec_from_file_location(name, EXAMPLE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def test_colour_selection_rejects_sky_and_handles_shadow(pushing):
    scene, policy = pushing["scene"], pushing["policy"]
    colors = np.array(scene.COLORS[:3])
    pixels = np.vstack([colors, colors*.3, [.27, .37, .48], [.75, .75, .75]])
    image = Image.fromarray(np.uint8(pixels[None, :]*255))
    for i in range(3):
        selected = policy.color_mask(image, colors, i)[0]
        assert np.flatnonzero(selected).tolist() == [i, i+3]


@pytest.mark.parametrize("letter,yaw", [("L", .35), ("A", -1.8), ("B", 2.2), ("B", -.94)])
def test_camera_pose_fit_distinguishes_letter_orientation(pushing, letter, yaw):
    scene, policy = pushing["scene"], pushing["policy"]
    view = View.look_at([.45, 0, .7], [.45, 0, 0], fov_deg=35, size=(500, 500), up=(1, 0, 0))
    yy, xx = np.indices((500, 500))
    top = .012
    xy = view.unproject(np.stack([xx+.5, yy+.5], axis=-1), np.full((500, 500), .7-top))[:, :, :2]
    origin = np.array([.47, -.025])
    mask = scene.inside((xy-origin) @ scene.rotation(yaw), scene.bars(letter))
    rgb = np.full((500, 500, 3), 190, np.uint8)
    rgb[mask] = np.array(scene.COLORS[0])*255
    depth = np.where(mask, .7-top, .7)
    frame = Frame(Image.fromarray(rgb), "top", view=view, depth=depth)
    result = policy.estimate(frame, letter, [scene.COLORS[0]], 0)
    assert np.linalg.norm(result["xy"]-origin) < .001
    assert abs(policy.angle_difference(result["yaw"], yaw)) < np.radians(2)


def test_audit_rejects_contact_failures_and_unresolved_power(pushing):
    model = mj.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="bench" type="plane" size="1 1 .1"/>
      <body name="letter_0_T" pos=".4 0 .006"><freejoint name="letter_0_T_free"/>
        <geom type="box" name="letter_0_T_0" size=".02 .02 .006" mass=".02"/>
      </body><body name="push_tool" pos=".2 0 .2"/>
      </worldbody></mujoco>''')
    data = mj.MjData(model)
    for _ in range(500):
        mj.mj_step(model, data)
    replay = dict(time=np.linspace(0, 2.1, 43), qpos=np.tile(data.qpos, (43, 1)),
                  qvel=np.zeros((43, model.nv)))
    task = dict(letters=[dict(name="letter_0_T", letter="T", target_xy=[.4, 0], target_yaw=0)])
    telemetry = dict(max_penetration_m=.0001, pusher_contacts={"letter_0_T_0": 5},
                     torque_off=True, error=None)
    evaluate = pushing["audit"].evaluate
    assert evaluate(model, data, task, replay, telemetry)["passed"]
    for change in (dict(max_penetration_m=.004), dict(torque_off=False), dict(pusher_contacts={}),
                   dict(unexpected_contacts={"robot0_link1/letter_0_T_0": 1}),
                   dict(solver_warnings={"bad_qacc": 1}), dict(error="unfinished")):
        assert not evaluate(model, data, task, replay, telemetry | change)["passed"]
    displaced = dict(task, letters=[dict(task["letters"][0], target_xy=[.5, 0])])
    assert not evaluate(model, data, displaced, replay, telemetry)["passed"]
