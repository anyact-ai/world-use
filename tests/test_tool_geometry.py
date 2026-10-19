"""Declared collision bounds, independent of the scene backend's hidden geometry."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from world_use import Refused, World
from world_use.body import JointSpec, JointState, ToolBox
from world_use.config import manifest_data, manifest_from_data
from world_use.envelope import Envelope
from world_use.geometry import rot_z


def rig(manifest, *, boxes=(), payload=False):
    # An analytic fixture isolates collision checks from IK and contact dynamics.
    def fk(q):
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rot_z(q[3]), q[:3]
        return pose

    chain = SimpleNamespace(fk=fk, links={},
                            points=lambda q: np.array([[-1, 0, 1], [-1, 0, 1], q[:3]]),
                            motion_bound=lambda a, b: float(np.linalg.norm(np.asarray(b)[:3] - np.asarray(a)[:3])))
    model = replace(manifest, joints=tuple(JointSpec(n, -10, 10, v_max=1000, a_max=1e6) for n in "xyza"),
                    gripper=None, tool_boxes=tuple(boxes), turn_clearance=None, max_excursion=None, link_radius_m=0)
    world = World()
    q = np.array([0., 0., .1, 0.])
    if payload:
        world.add_box("package", "object", q[:3], [.16, .02, .04], source="measured dimensions")
        world.grab(fk(q))
    return Envelope(model, chain, world, q), q


def feedback(q):
    return JointState(t=1.0, q=np.asarray(q))


@pytest.mark.parametrize("role", ["tool", "finger"])
def test_declared_geometry_hits_a_zone_that_the_arm_and_tcp_miss(k, role):
    bounds = ToolBox("jaw travel", (0, .08, 0), (.04, .02, .04), "caliper measurement", role)
    env, q = rig(k.manifest, boxes=[bounds])
    env.world.add_box("divider", "keep_out", [0, .08, .1], [.005] * 3)
    with pytest.raises(Refused, match=f"{role}:jaw travel"):
        env.check_path([q], q)
    trip = env.watch(feedback(q), q)
    assert trip is not None and trip.kind == "keep_out" and role in trip.message
    bare = Envelope(replace(env.m, tool_boxes=()), env.chain, env.world, q)
    bare.check_path([q], q)
    assert bare.watch(feedback(q), q) is None


def test_rotating_payload_checks_its_swept_corners_and_attachment(k):
    env, q = rig(k.manifest, payload=True)
    obstacle = env.world.add_box("corner", "keep_out", [.05, .05, .1], [.005] * 3)
    end = q + [0, 0, 0, np.pi / 2]
    original = env.world.boxes["package"].pose.copy()
    assert not obstacle.intersects_box(original, env.world.boxes["package"].size)
    assert not obstacle.intersects_box(env.chain.fk(end), env.world.boxes["package"].size)
    with pytest.raises(Refused, match="payload:package"):
        env.check_path([end], q)
    middle = q + [0, 0, 0, np.pi / 4]
    trip = env.watch(feedback(middle), middle)
    assert trip is not None and trip.kind == "keep_out"
    np.testing.assert_array_equal(env.world.boxes["package"].pose, original)  # planning is not a state update
    env.world.drop()
    assert env.watch(feedback(middle), middle) is None


def test_payload_placement_release_and_withdrawal_keep_support_geometry(k):
    env, q = rig(k.manifest, payload=True)
    env.world.add_box("table", "surface", [0, 0, -.01], [.4, .4, .02])
    landing = np.array([0., 0., .02, 0.])
    env.check_path(np.linspace(q, landing, 100), q)
    env.begin_geometry(q)
    assert env.watch(feedback(landing), landing) is None
    below = landing - [0, 0, .004, 0]
    with pytest.raises(Refused, match="payload:package"):
        env.check_path(np.linspace(q, below, 100), q)
    env.world.carry(env.chain.fk(landing))
    placed = env.world.drop()
    assert placed is not None and placed.pose[2, 3] == pytest.approx(.02)
    env.hold_geometry(landing)
    env.check_path(np.linspace(landing, q, 100), landing)
    assert env.watch(feedback(q), q) is None


def test_guarded_support_contact_is_bounded_and_does_not_waive_another_part(k):
    finger = ToolBox("low finger", (0, .07, -.04), (.02, .02, .02), "measured", "finger")
    env, q = rig(k.manifest, boxes=[finger], payload=True)
    env.world.add_box("table", "surface", [0, 0, -.01], [.4, .4, .02])
    delta, contact = env.contact_target(q, np.array([0, 0, -.12]), .02)
    assert contact is not None and contact[:2] == (("finger:low finger",), "table")
    env.check_path(np.linspace(q, q + [*delta, 0], 100), q, allow_contact=True, contact=contact)
    env.begin_geometry(q, contact)
    touching = q + [0, 0, -.065, 0]
    assert env.watch(feedback(touching), touching) is None
    deeper = q + [0, 0, -.08, 0]
    assert env.watch(feedback(deeper), deeper).kind == "surface"
    # Holding after stop discards the unconsumed contact allowance. It permits withdrawal only.
    env.hold_geometry(touching)
    assert env.watch(feedback(touching - [0, 0, .003, 0]), touching).kind == "surface"
    env.check_path(np.linspace(touching, q, 100), touching)
    env.begin_geometry(q, contact)
    env.world.add_box("new support", "surface", [0, 0, .06], [.4, .4, .02])
    assert env.watch(feedback(q), q).kind == "surface"


def test_contact_allowance_cannot_follow_a_changed_support_or_payload(k):
    env, q = rig(k.manifest, payload=True)
    surface = env.world.add_box("table", "surface", [0, 0, -.01], [.4, .4, .02])
    _, contact = env.contact_target(q, np.array([0, 0, -.1]), .02)
    env.begin_geometry(q, contact)
    touching = q - [0, 0, .09, 0]
    assert env.watch(feedback(touching), touching) is None
    surface.pose[2, 3] += .002
    assert env.watch(feedback(touching), touching).kind == "surface"
    surface.pose[2, 3] -= .002
    env.world.boxes["package"].size[2] += .002
    assert env.watch(feedback(touching), touching).kind == "surface"
    # Re-measured geometry restarts from the measured overlap: withdrawal only.
    env.hold_geometry(touching)
    assert env.watch(feedback(touching), touching) is None
    env.check_path(np.linspace(touching, q, 100), touching)
    with pytest.raises(Refused, match="payload:package"):
        env.check_path(np.linspace(touching, touching - [0, 0, .002, 0], 100), touching)


@pytest.mark.parametrize("bottom", ["fingers", "level with tool point"])
def test_guarded_contact_covers_every_part_reaching_the_support(k, bottom):
    if bottom == "fingers":
        boxes = [ToolBox(side, (0, y, -.04), (.02, .02, .02), "measured", "finger")
                 for side, y in (("left", .03), ("right", -.03))]
    else:
        boxes = [ToolBox("housing", (0, 0, .01), (.04, .04, .02), "measured")]
    env, q = rig(k.manifest, boxes=boxes)
    env.world.add_box("table", "surface", [0, 0, -.01], [.4, .4, .02])
    delta, contact = env.contact_target(q, np.array([0, 0, -.12]), .02)
    assert set(contact[0]) == {f"{box.role}:{box.name}" for box in boxes}
    env.check_path(np.linspace(q, q + [*delta, 0], 100), q, allow_contact=True, contact=contact)
    env.begin_geometry(q, contact)
    end = q + [*delta, 0]
    assert env.watch(feedback(end), end) is None


@pytest.mark.parametrize("guarded", [False, True])
def test_small_steps_cannot_replenish_surface_allowance(k, guarded):
    env, q = rig(k.manifest, payload=True)
    table = env.world.add_box("table", "surface", [0, 0, -.01], [.4, .4, .02])
    if guarded:
        _, contact = env.contact_target(q, np.array([0, 0, -.1]), .02)
        env.begin_geometry(q, contact)
    q = np.array([0., 0., .015 if guarded else .02, 0.])
    initial_depth = table.box_depth(env.chain.fk(q), env.world.boxes["package"].size)
    env.hold_geometry(q)
    for _ in range(20):
        end = q - [0, 0, .0005, 0]
        try:
            env.check_path(np.linspace(q, end, 10), q)
        except Refused:
            break
        env.begin_geometry(q)
        assert env.watch(feedback(end), end) is None
        env.hold_geometry(end)
        q = end
    else:
        pytest.fail("ordinary steps accumulated 10 mm of penetration")
    depth = table.box_depth(env.chain.fk(q), env.world.boxes["package"].size)
    assert depth <= initial_depth + .001
    env.begin_geometry(q)
    assert env.watch(feedback(q - [0, 0, .001, 0]), q).kind == "surface"
    lifted = q + [0, 0, .01, 0]
    env.check_path(np.linspace(q, lifted, 100), q)
    env.hold_geometry(lifted)
    with pytest.raises(Refused, match="payload:package"):
        env.check_path(np.linspace(lifted, q - [0, 0, .002, 0], 100), lifted)


def test_worker_keeps_consumed_surface_allowance(lifted, rehearser):
    k = lifted
    bounds = ToolBox("probe", (0, 0, 0), (.01, .01, .01), "fixture dimensions")
    k.manifest = k.body.manifest = replace(k.manifest, tool_boxes=(bounds,))
    extent = np.abs(k.tool[2, :3]) @ np.array(bounds.size) / 2
    k.world.add_box("table", "surface", k.tool[:3, 3] - [0, 0, extent + .01], [.4, .4, .02])
    k.envelope = Envelope(k.manifest, k.chain, k.world, k.state.q)
    assert k.run(dict(do="line", down=.0005)).ok
    before = k.state.q.copy()
    future, _ = rehearser.prepare(dict(do="line", down=.0007), k)
    with pytest.raises(Refused, match="tool:probe"):
        future.result(timeout=10)
    np.testing.assert_array_equal(k.state.q, before)


def test_tool_bounds_are_serialized_for_worker_and_record(k):
    box = ToolBox("left finger", (0, .03, -.02), (.02, .06, .04), "measured swept bounds", "finger")
    manifest = replace(k.manifest, tool_boxes=(box,))
    restored = manifest_from_data(manifest_data(manifest))
    assert restored.tool_boxes == (box,)
    data = manifest_data(manifest)
    data["tool_boxes"][0]["size"] = [1, -1, 1]
    with pytest.raises(ValueError, match="positive"):
        manifest_from_data(data)


def test_worker_receives_declared_bounds_and_still_leaves_the_robot_untouched(lifted, rehearser):
    from world_use.plan import same_start, snapshot

    k = lifted
    bounds = ToolBox("side tool", (0, .15, 0), (.01, .01, .01), "fixture dimensions")
    k.manifest = k.body.manifest = replace(k.manifest, tool_boxes=(bounds,))
    k.envelope = Envelope(k.manifest, k.chain, k.world, k.q_start)
    center = k.tool[:3, 3] + k.tool[:3, :3] @ np.array(bounds.center) + [0, 0, .018]
    k.world.add_box("cabinet", "keep_out", center, [.003] * 3)
    before = k.state.q.copy()
    report = rehearser.check(dict(do="line", up=.02), k)
    assert not report.ok
    assert any(p.get("geometry") == "tool:side tool" for p in report.problems)
    assert report.to_dict()["coverage"]["collision"]["tool_boxes"][0]["source"] == "fixture dimensions"
    np.testing.assert_array_equal(k.state.q, before)
    saved = snapshot(k)
    k.manifest = replace(k.manifest, tool_boxes=())
    assert not same_start(saved, k)


def test_stopped_guarded_job_allows_withdrawal_without_retaining_its_contact_budget(lifted):
    k = lifted
    bounds = ToolBox("probe", (0, 0, 0), (.01, .01, .01), "fixture dimensions")
    k.manifest = k.body.manifest = replace(k.manifest, tool_boxes=(bounds,))
    k.envelope = Envelope(k.manifest, k.chain, k.world, k.q_start)
    point = k.tool[:3, 3]
    table = k.world.add_box("declared table", "surface", point - [0, 0, .03], [.4, .4, .02])
    job = k.submit(dict(do="guarded", down=.06, expect_contact=False))
    for _ in range(1000):
        k.tick()
        k.clock.wait()
        if table.box_depth(k.tool, np.array(bounds.size)) > .003:
            break
        assert not job.finished, job.outcome
    else:
        pytest.fail("probe never reached declared contact")
    k.stop("test interrupt during contact")
    k.tick()
    assert job.status == "stopped"
    out = k.run(dict(do="line", down=.005))
    assert out.status == "refused" and "probe" in out.message
    out = k.run(dict(do="line", up=.04))
    assert out.ok, out.message


@pytest.mark.parametrize("change", [dict(size=(0, 1, 1)), dict(center=(True, 0, 0)), dict(source=""),
                                   dict(rpy=(float("nan"), 0, 0)), dict(role="payload")])
def test_invalid_declared_bounds_are_rejected(change):
    args = dict(name="finger", center=(0, 0, 0), size=(.01, .01, .01), source="measurement") | change
    with pytest.raises(ValueError):
        ToolBox(**args)
