"""Physics and rendered pixels must come from MuJoCo, independently of planner beliefs."""
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from world_use import Kernel, VirtualClock, World, bodies, views
from world_use.bodies.mujoco_scene import mj
from world_use.bodies.rebot import MANIFEST
from world_use.cameras import View
from world_use.config import load_robot, load_workcell
from world_use.examples.pick_place import pickup, setup

JAW_ARM = load_robot(Path(__file__).with_name("jaw_arm.toml"))     # five joints and a one-joint revolute jaw


def jaw_arm(truth=None, start_deg=(0, 64.8, -56.9, -97.9, 0), **changes):
    """A kernel on the jaw arm, by default with the tool pointing down 10 cm above the floor."""
    truth = truth if truth is not None else World()
    body = bodies.make("sim", truth, manifest=replace(JAW_ARM, **changes), q=np.radians(start_deg), gripper=0.0)
    k = Kernel(body, World.from_dict(truth.to_dict()), VirtualClock(100))
    k.connect()
    k.enable()
    return k


def test_urdf_kinematics_and_gravity_match_mujoco():
    body = bodies.make("sim", q=[.3, .7, .8, -.3, .1, .2], gripper=2)
    try:
        body.reset(body.q, body.grip)
        m, d = body.model, body.data
        assert isinstance(m, mj.MjModel)
        assert m.nmesh > 25 and m.opt.timestep <= .002
        tool = d.body(MANIFEST.tool_link)
        expected = body.chain.fk(body.q)
        np.testing.assert_allclose(tool.xpos, expected[:3, 3], atol=1e-10)
        np.testing.assert_allclose(tool.xmat.reshape(3, 3), expected[:3, :3], atol=1e-10)
        np.testing.assert_allclose(d.qfrc_bias[:6], body.chain.gravity(body.q), atol=.002)
    finally:
        body.close()


def test_free_objects_fall_and_land_with_motors_off():
    world = World()
    world.add_box("block", "object", [.8, 0, .4], [.04, .04, .04])
    body = bodies.make("sim", world)
    try:
        for _ in range(10):
            body.read()
        z = world.boxes["block"].pose[2, 3]
        assert .33 < z < .37                 # about 1/2 g t² of free fall
        for _ in range(100):
            body.read()
        assert world.boxes["block"].pose[2, 3] == pytest.approx(.02, abs=.001)
        assert np.all(body.data.qfrc_actuator == 0)
        assert body.t == pytest.approx(1.1)
    finally:
        body.close()


def test_grasp_uses_contacts_and_release_does_not_teleport_the_object():
    k, truth = setup()
    try:
        k.enable()
        assert k.run(pickup(k)).ok
        assert truth.held is not None and k.body.data.ncon > 0
        assert k.run({"do": "line", "up": .06}).ok
        block = truth.boxes["block"]
        before = block.pose.copy()
        assert before[2, 3] > .24
        # A planner belief cannot weld the body or reposition simulation truth.
        k.world.boxes["block"].pose[0, 3] += .5
        k.world.held = None
        np.testing.assert_array_equal(block.pose, before)
        g = k.manifest.gripper
        k.body.command(k.state.q, np.zeros(6), g.open)
        k.body.read()
        assert abs(block.pose[2, 3] - before[2, 3]) < .002
        for _ in range(150):
            k.body.read()
        assert truth.held is None
        assert block.pose[2, 3] == pytest.approx(.2, abs=.002)
    finally:
        k.close()


@pytest.mark.parametrize("brakes", [False, True])
def test_disable_removes_actuation_and_only_brakes_hold_a_raised_arm(brakes):
    manifest = replace(MANIFEST, rest=None) if brakes else MANIFEST
    body = bodies.make("sim", manifest=manifest, q=[.3, .7, .8, -.3, .1, .2], gripper=2)
    try:
        body.enable()
        for _ in range(20):
            body.read()
        before = body.q.copy()
        body.disable()
        for _ in range(20):
            state = body.read()
        moved = np.linalg.norm(state.q - before)
        assert moved < 1e-4 if brakes else moved > .01
        np.testing.assert_array_equal(body.data.qfrc_actuator, 0)
    finally:
        body.close()


def test_with_torque_off_a_folded_arm_and_its_gripper_stay_put():
    q = np.radians(load_workcell("block")["body_options"]["start_deg"])
    body = bodies.make("sim", q=q)
    g = body.manifest.gripper
    try:
        start = body.read()
        for _ in range(3000):                    # 30 s: unpowered geared motors hold what gravity does not load
            state = body.read()
        assert np.degrees(np.abs(state.q - start.q)).max() < 1
        assert abs(g.aperture(state.gripper) - g.aperture(start.gripper)) < .001
    finally:
        body.close()


@pytest.mark.rendering
def test_camera_projection_matches_off_axis_intrinsics_and_does_not_step_physics():
    world = World()
    point = np.array([1.08, .1, .4])
    world.add_box("target", "object", point, [.035, .035, .035])
    body = bodies.make("sim", world)
    view = View.look_at([1., -.8, .55], [1., 0, .4], size=(400, 300))
    view = replace(view, fx=380, fy=310, cx=175, cy=163)
    try:
        picture = np.asarray(body.render(view))
        mask = (picture[:, :, 0] > 1.6 * picture[:, :, 1]) & (picture[:, :, 0] > 70)
        y, x = np.nonzero(mask)
        assert len(x) > 20
        (expected,), _ = view.project(point)
        np.testing.assert_allclose([x.mean(), y.mean()], expected, atol=2)
        assert body.t == 0 and body.data.time == 0
    finally:
        body.close()


def test_an_incompatible_gripper_mapping_is_rejected():
    gripper = replace(MANIFEST.gripper, opens_along=(1, 0, 0))
    body = bodies.make("sim", manifest=replace(MANIFEST, gripper=gripper))
    try:
        with pytest.raises(ValueError, match="must oppose along opens_along"):
            body.enable()
        assert not body.enabled
    finally:
        body.close()


def test_a_one_joint_jaw_is_driven_in_its_own_units_and_holds_what_it_grips():
    truth = World()
    truth.add_box("pad", "surface", [.17, 0, .005], [.12, .12, .01])
    truth.add_box("block", "object", [.17, 0, .025], [.02, .02, .03])
    k = jaw_arm(truth)
    try:
        assert k.run({"do": "gripper", "to": .5}).ok
        assert k.body.data.joint("jaw").qpos[0] == pytest.approx(.5, abs=.005)
        out = k.run([{"do": "line", "up": -.065}, {"do": "grip", "expect_mm": [10, 25]},
                     {"do": "line", "up": .05}])
        assert out.ok, out.message
        assert truth.held is not None and truth.held[0] == "block"
        assert truth.boxes["block"].pose[2, 3] > .06
    finally:
        k.close()


def test_fixed_child_finger_pads_hold_and_release_by_contact(tmp_path):
    robot = ET.parse(JAW_ARM.urdf)
    root = robot.getroot()
    for name in ("hand", "jaw"):
        link = root.find(f"link[@name='{name}']")
        collision = link.find("collision")
        link.remove(collision)
        pad = ET.SubElement(root, "link", name=f"{name}_pad")
        pad.append(collision)
        joint = ET.SubElement(root, "joint", name=f"{name}_pad", type="fixed")
        ET.SubElement(joint, "parent", link=name)
        ET.SubElement(joint, "child", link=f"{name}_pad")
    urdf = tmp_path / "padded-jaw.urdf"
    robot.write(urdf)
    truth = World()
    truth.add_box("pad", "surface", [.17, 0, .005], [.12, .12, .01])
    truth.add_box("block", "object", [.17, 0, .025], [.02, .02, .03])
    k = jaw_arm(truth, urdf=urdf)
    try:
        out = k.run([{"do": "gripper", "to": .5}, {"do": "line", "up": -.065},
                     {"do": "grip", "expect_mm": [10, 25]}, {"do": "line", "up": .05}])
        assert out.ok, out.message
        assert truth.held is not None and truth.held[0] == "block"
        assert truth.boxes["block"].pose[2, 3] > .06
        assert k.run({"do": "gripper", "to": .8}).ok
        for _ in range(100):
            k.tick()
        assert truth.held is None
        assert truth.boxes["block"].pose[2, 3] == pytest.approx(.025, abs=.002)
    finally:
        k.close()


def test_without_a_gripper_description_the_jaw_rides_along():
    body = bodies.make("sim", manifest=replace(JAW_ARM, gripper=None))
    try:
        body.read()
        assert body.model.njnt == JAW_ARM.n
    finally:
        body.close()


@pytest.mark.parametrize("weights", [None, (1.0,) * 6])
def test_a_five_joint_arm_moves_sideways_by_letting_its_heading_turn(weights):
    k = jaw_arm(start_deg=(0, 80.9, -84.4, -41.6, 0), ik_weights=weights)      # pointing 45 deg down
    try:
        steps = [{"do": "line", "left": .03}, {"do": "line", "forward": .03}, {"do": "move_to", "to": [.2, -.04, .12]}]
        results = [k.run(step).status for step in steps]
        assert ("heading turns" in views.tool_line(k)) == (weights is None)       # the card says which
        if weights is None:
            assert results == ["done"] * 3
        else:                                   # holding the heading as well puts the same line out of reach
            assert results[0] == "refused"
    finally:
        k.close()


@pytest.mark.parametrize("kind", ["object", "surface"])
def test_estimated_parameters_cannot_change_physical_mass_or_friction(kind):
    world = World()
    world.add_box("block", kind, [.8, 0, .4], [.04] * 3, mass_kg=.05, friction=.8)
    body = bodies.make("sim", world)
    k = Kernel(body, world, VirtualClock(100))
    try:
        k.connect()
        k.tick()
        k.world.boxes["block"].params.update(mass_kg=5.0, friction=0.0)
        k.tick()
        assert body.world.boxes["block"].params == dict(mass_kg=.05, friction=.8)
        assert body.model.body("box/block").mass[0] == pytest.approx(.05)
        assert body.model.geom("box/block").friction[0] == pytest.approx(.8)
    finally:
        k.close()


def test_restored_worlds_own_parameters_and_nested_fact_values():
    world = World()
    world.add_box("block", "object", [.8, 0, .4], [.04] * 3, mass_kg=.05)
    world.assert_fact("target", {"position": [.1, .2, .3]}, "test")
    snapshot = world.to_dict()
    estimate, truth = World.from_dict(snapshot), World.from_dict(snapshot)
    estimate.boxes["block"].params["mass_kg"] = 9
    estimate.facts["target"].value["position"][0] = 9
    assert truth.boxes["block"].params["mass_kg"] == .05
    assert truth.facts["target"].value["position"] == [.1, .2, .3]
    assert snapshot == world.to_dict()
