from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from world_use import Refused, World
from world_use.body import JointSpec
from world_use.envelope import Envelope


def test_keep_out_catches_a_link_between_joint_origins(lifted):
    k = lifted
    points = k.chain.points(k.state.q)
    i = 1 + np.linalg.norm(np.diff(points[1:], axis=0), axis=1).argmax()
    center = (points[i] + points[i + 1]) / 2
    box = k.world.add_box("link interior", "keep_out", center, [0.002] * 3, yaw_deg=37)
    assert not any(box.contains(p) for p in points)
    with pytest.raises(Refused, match="link interior"):
        k.envelope.check_path([k.state.q], k.state.q)
    trip = k.envelope.watch(k.state, k.state.q, k.state.gripper)
    assert trip is not None and trip.kind == "keep_out"


def test_keep_out_covers_motion_between_control_samples(k):
    # A tiny prismatic stick crosses a zone between two poses; neither endpoint is in it.
    def points(q):
        return np.array([[0, 0, 0], [q[0], 0, 0], [q[0], 0, 0.01]])

    def fk(q):
        t = np.eye(4)
        t[:3, 3] = points(q)[-1]
        return t

    chain = SimpleNamespace(points=points, fk=fk, links={}, motion_bound=lambda a, b: abs(b[0] - a[0]))
    manifest = replace(k.manifest, joints=(JointSpec("slide", -1, 1, v_max=10, a_max=10000),),
                       turn_clearance=None, max_excursion=None, link_radius_m=0)
    world = World()
    world.add_box("thin wall", "keep_out", [0.005, 0, 0.005], [0.0001, 0.01, 0.01])
    env = Envelope(manifest, chain, world, [0])
    with pytest.raises(Refused, match="thin wall"):
        env.check_path([[0.01]], [0])


def test_rotated_box_segment_intersection_handles_parallel_and_padded_links():
    box = World().add_box("rotated", "keep_out", [0, 0, 0], [0.1, 0.2, 0.1], yaw_deg=45)
    assert box.intersects_segment([-1, -1, 0], [1, 1, 0])
    assert not box.intersects_segment([-1, -1, 0.06], [1, 1, 0.06])
    assert box.intersects_segment([-1, -1, 0.06], [1, 1, 0.06], margin=0.02)


def test_slow_zone_refuses_fast_paths_and_names_a_duration_that_passes(lifted):
    k = lifted
    k.world.add_box("careful", "slow", k.tool[:3, 3], [0.2] * 3, speed=0.01)
    before = k.cmd.q.copy()
    out = k.run({"do": "line", "up": 0.03, "duration": 1})
    assert out.status == "refused" and "slow zone" in out.message
    assert np.array_equal(k.cmd.q, before)
    assert k.run({"do": "line", "up": 0.03, "duration": out.data["min_seconds"]}).ok


@pytest.mark.parametrize("speed", [None, 0, float("nan")])
def test_slow_zone_requires_a_positive_finite_speed(speed):
    params = {} if speed is None else {"speed": speed}
    with pytest.raises(ValueError, match="speed"):
        World().add_box("careful", "slow", [0, 0, 0], [1, 1, 1], **params)


# One case per rule: explicit null, nonpositive, nonfinite, boolean and numeric-string coercion.
@pytest.mark.parametrize("dtau", [None, 0, float("nan"), True, "0.3"])
def test_fragile_zone_rejects_invalid_limits_on_creation_and_restore(dtau):
    world = World()
    original = world.add_box("glass", "fragile", [0, 0, 0], [1, 1, 1])
    assert original.params["dtau"] == 0.3
    with pytest.raises(ValueError, match="dtau"):
        world.add_box("glass", "fragile", [0, 0, 0], [1, 1, 1], dtau=dtau)
    assert world.boxes["glass"] is original
    saved = world.to_dict()
    saved["boxes"]["glass"]["params"]["dtau"] = dtau
    with pytest.raises(ValueError, match="dtau"):
        World.from_dict(saved)


def test_misspelled_box_parameter_preserves_the_previous_zone():
    world = World()
    original = world.add_box("glass", "fragile", [0, 0, 0], [1, 1, 1], dtau=0.1)
    with pytest.raises(ValueError, match="datu"):
        world.add_box("glass", "fragile", [0, 0, 0], [2, 2, 2], datu=0.03)
    assert world.boxes["glass"] is original


@pytest.mark.parametrize("change", [dict(center=[True, 0, 0]), dict(size=["1", 1, 1]), dict(speed=0.1)])
def test_box_geometry_and_kind_parameters_are_checked_before_replacement(change):
    world = World()
    original = world.add_box("object", "object", [0, 0, 0], [1, 1, 1])
    spec = dict(name="object", kind="object", center=[0, 0, 0], size=[1, 1, 1]) | change
    with pytest.raises(ValueError):
        world.add_box(**spec)
    assert world.boxes["object"] is original


def test_rejected_fragile_update_cannot_disable_contact_detection(k):
    from world_use.daemon import Daemon

    k.tick()
    original = k.world.add_box("glass", "fragile", k.tool[:3, 3], [1, 1, 1], dtau=0.3)
    with pytest.raises(ValueError, match="dtau"):
        Daemon._world(SimpleNamespace(k=k), {"box": dict(name="glass", kind="fragile", center=[0, 0, 0],
                                                       size=[2, 2, 2], dtau="nan")})
    assert k.world.boxes["glass"] is original
    k.rebias()
    k.state = replace(k.state, tau=k.state.tau + [0.8, 0, 0, 0, 0, 0])
    trips = [k._collision() for _ in range(5)]
    assert trips[-1] is not None and trips[-1].kind == "contact"
