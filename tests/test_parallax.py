"""Positions from a camera fixed to the hand: on a synthetic hand that moves level over a table, and on the
simulated arm with a rendered wrist camera, where simulator truth appears only in assertions."""
import numpy as np
import pytest

from world_use.behaviors import Outcome
from world_use.cameras import MountedSimCamera, View
from world_use.examples.pick_place import setup
from world_use.parallax import Look, footprint, locate, triangulate
from world_use.vision import dominant_region

# The camera looks along the tool's approach axis from 5 cm beside the grasp point and 10 cm behind it.
MOUNT = View([[-1, 0, 0, 0], [0, -1, 0, 0.05], [0, 0, 1, -0.1], [0, 0, 0, 1]], 170.0, 170.0, 128, 128, 256, 256)
DOWN = np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, -1]])      # tool axes in the base frame: approach straight down


class Hand:
    """A hand at a height over a table, looking at a point or at a level square of some half-width."""

    def __init__(self, thing, *, half=0.0, z=0.65, mount=MOUNT, visible=lambda looks: True, reach=None):
        self.thing, self.half, self.mount = np.asarray(thing, float), half, mount
        self.position, self.visible, self.reach = np.array([0.0, 0.0, z]), visible, reach
        self.looks, self.goals = 0, []

    @property
    def view(self):
        tool = np.eye(4)
        tool[:3, :3], tool[:3, 3] = DOWN, self.position
        return self.mount.on(tool)

    def look(self):
        self.looks += 1
        view = self.view
        if not self.visible(self.looks):
            return Look(view, self.position.copy())
        corners = [self.thing + [dx, dy, 0] for dx in (-self.half, self.half) for dy in (-self.half, self.half)]
        pixels = view.project(corners)[0]
        low, high = pixels.min(axis=0), pixels.max(axis=0)
        if not self.half:
            low, high = low - [10, 15], high + [10, 15]
        # A tracker reports the part inside the image: its box, ending one past the last pixel, and that
        # part's centre.
        box = [*np.clip(low, 0, 256), *np.clip(high, 0, 256)]
        return Look(view, self.position.copy(), [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], box)

    def move(self, goal):
        self.goals.append(np.array(goal, float))
        step = np.asarray(goal, float) - self.position[:2]
        distance = float(np.linalg.norm(step))
        if self.reach is not None and distance > 0:
            step *= max(0.0, distance - self.reach) / distance      # the arm arrives this far short
        self.position[:2] += step
        return Outcome("done", "move", "fixture")


def found(hand, **options):
    return locate(hand.move, hand.look, arrive_m=0.012, **options)


def test_the_hand_ends_over_the_region_and_reports_where_it_is():
    hand = Hand([0.07, -0.04, 0.45])
    result = found(hand, assumed_z=0.5)
    assert result.ok and [result.data[k] for k in "xyz"] == pytest.approx(hand.thing, abs=5e-3)
    # The search ends once the hand is within the arrival distance of the point the looks fix.
    assert hand.position[:2] == pytest.approx(hand.thing[:2], abs=0.012) and result.data["looks"] >= 2
    # The first move already heads for where the look meets the assumed height.
    assert hand.goals[0] == pytest.approx(hand.thing[:2], abs=0.02)
    # The box, 20 by 30 pixels from 30 cm above, is reported in metres, and the point in the image the
    # caller sees next: under the fingers, beside the image centre.
    assert result.data["extent_m"] == pytest.approx([0.035, 0.053], abs=2e-3)
    assert result.data["pixel"] == pytest.approx([128, 156], abs=6)


def test_a_whole_region_with_no_height_to_assume_needs_only_a_second_look_from_a_sidestep_away():
    hand = Hand([-0.06, 0.09, 0.5])
    result = found(hand)
    assert result.ok and [result.data[k] for k in "xyz"] == pytest.approx(hand.thing, abs=5e-3)
    assert np.linalg.norm(hand.goals[0]) == pytest.approx(0.045)


def test_a_region_straight_below_the_hand_is_placed_even_when_the_arm_arrives_short():
    # The first look is nearly along the ray, so a step toward the region gains no baseline by itself.
    hand = Hand([0.0, -0.045, 0.45], reach=0.011)
    result = found(hand, assumed_z=0.5)
    assert result.ok and result.data["moves"] <= 2
    assert [result.data[k] for k in "xy"] == pytest.approx(hand.thing[:2], abs=5e-3)


def test_a_region_the_image_cuts_off_is_placed_only_from_looks_at_all_of_it():
    hand = Hand([0.22, 0.0, 0.45], half=0.09)
    result = found(hand)
    # The first look sees only the near part, whose centre is not the square's, yet leads most of the way.
    assert [result.data[k] for k in "xyz"] == pytest.approx(hand.thing, abs=5e-3)
    assert 0.12 < hand.goals[0][0] < 0.3
    assert result.data["extent_m"] == pytest.approx([0.18, 0.18], abs=6e-3)


def test_a_region_is_whole_unless_its_box_reaches_the_image_border():
    hand = Hand([0.0, 0.0, 0.45])
    # A box ends one past its last pixel: 255 leaves the last column of a 256-pixel image free.
    assert Look(hand.view, hand.position, [100, 100], [1, 1, 255, 255]).whole
    assert not Look(hand.view, hand.position, [100, 100], [1, 1, 256, 200]).whole
    assert not Look(hand.view, hand.position, [100, 100], [0, 1, 200, 200]).whole


def test_a_move_that_fails_ends_the_search_with_its_outcome():
    hand = Hand([0.07, -0.04, 0.45])
    result = locate(lambda goal: Outcome("stopped", "move", "still converging"), hand.look,
                    arrive_m=0.012)
    assert result.status == "stopped" and result.message == "still converging" and hand.looks == 1


def test_a_height_to_assume_that_is_not_below_the_camera_is_not_used():
    # The ray never meets a height above the lens; heading for where it would have leads away from the region.
    hand = Hand([0.07, -0.04, 0.45])
    result = found(hand, assumed_z=0.9)
    assert result.ok and [result.data[k] for k in "xyz"] == pytest.approx(hand.thing, abs=5e-3)
    assert np.linalg.norm(hand.goals[0]) == pytest.approx(0.045) and hand.goals[0] @ hand.thing[:2] > 0


def test_looks_that_meet_behind_the_camera_place_nothing():
    hand = Hand([0.07, -0.04, 0.45])
    look = hand.look

    def moving_with_the_camera():
        # A region that seems to travel with the hand: the rays through it fan out below and cross
        # behind the lens.
        seen = look()
        return Look(seen.view, seen.hand, [128 + 500 * hand.position[0], 128 + 500 * hand.position[1]], seen.bbox)

    result = locate(hand.move, moving_with_the_camera, arrive_m=0.012)
    assert not result.ok and "x" not in result.data


def test_a_region_that_is_not_visible_ends_the_search_before_any_move():
    hand = Hand([0.07, -0.04, 0.45], visible=lambda looks: False)
    result = found(hand)
    assert result.status == "surprise" and not hand.goals


def test_a_region_lost_after_it_was_placed_keeps_its_position():
    hand = Hand([0.07, -0.04, 0.45], visible=lambda looks: looks <= 2)
    result = found(hand)
    assert result.ok and result.data["region"].startswith("lost") and "pixel" not in result.data
    assert [result.data[k] for k in "xy"] == pytest.approx(hand.thing[:2], abs=5e-3)


def test_a_position_found_earlier_is_confirmed_by_one_look_from_over_it():
    hand = Hand([0.07, -0.04, 0.45])
    first = found(hand, assumed_z=0.5)
    known = [first.data[k] for k in "xyz"]
    looks, moves = hand.looks, len(hand.goals)
    again = found(hand, known=known)
    assert again.ok and [again.data[k] for k in "xyz"] == known and again.data["moves"] == 0
    assert hand.looks == looks + 1 and len(hand.goals) == moves and "pixel" in again.data
    # From elsewhere the first move goes straight over it, and that look confirms it.
    hand.position[:2] = [-0.05, 0.06]
    back = found(hand, known=known)
    assert back.ok and back.data["moves"] == 1 and hand.goals[-1] == pytest.approx(known[:2])
    assert [back.data[k] for k in "xy"] == pytest.approx(hand.thing[:2], abs=5e-3)


def test_a_region_that_moved_since_it_was_found_is_placed_afresh():
    hand = Hand([0.07, -0.04, 0.45])
    known = [found(hand, assumed_z=0.5).data[k] for k in "xyz"]
    hand.thing = np.array([0.1, -0.01, 0.45])                       # pushed 4 cm aside since
    result = found(hand, known=known)
    assert result.ok and result.data["looks"] >= 2
    assert [result.data[k] for k in "xyz"] == pytest.approx(hand.thing, abs=5e-3)


def test_a_mirrored_image_places_the_region_in_the_same_spot():
    # The same camera published with its rows flipped: a negative focal length, and the row offset mirrored.
    flipped = View(MOUNT.T, MOUNT.fx, -MOUNT.fy, MOUNT.cx, MOUNT.height - 1 - MOUNT.cy, MOUNT.width, MOUNT.height)
    hand, mirrored = Hand([0.07, -0.04, 0.45]), Hand([0.07, -0.04, 0.45], mount=flipped)
    a, b = found(hand, assumed_z=0.5), found(mirrored, assumed_z=0.5)
    assert [b.data[k] for k in "xyz"] == pytest.approx([a.data[k] for k in "xyz"], abs=1e-9)
    assert b.data["pixel"][1] == 255 - a.data["pixel"][1]


@pytest.mark.parametrize("turn_deg", [0, 35, -70])
def test_a_region_seen_from_above_gets_its_narrow_width_and_the_direction_across_it(turn_deg):
    # A 12 by 5 cm box top at z = 0.45, turned about the vertical, seen from 30 cm above.
    hand = Hand([0.0, 0.05, 0.45])
    view, turn = hand.view, np.radians(turn_deg)
    long, narrow = np.array([np.cos(turn), np.sin(turn), 0]), np.array([-np.sin(turn), np.cos(turn), 0])
    grid = np.stack(np.meshgrid(np.arange(256), np.arange(256)), axis=-1).reshape(-1, 2)
    points = view.unproject(grid, np.full(len(grid), 0.3)) - hand.thing
    mask = ((abs(points @ long) <= 0.06) & (abs(points @ narrow) <= 0.025)).reshape(256, 256)
    seen = footprint(view, dominant_region(mask), 0.3)
    assert seen["across_m"] == pytest.approx(0.05, abs=5e-3) and seen["along_m"] == pytest.approx(0.12, abs=5e-3)
    assert abs(np.dot(seen["across"], narrow)) > 0.99
    assert footprint(view, dominant_region(np.zeros((256, 256), bool)), 0.3) is None


def test_rays_from_nearly_one_place_and_parallel_rays_fix_nothing():
    hand = Hand([0.07, -0.04, 0.45])
    first = hand.look()
    hand.position[:2] += [0.02, 0.0]
    near = hand.look()
    assert triangulate([first, near]) is None
    hand.position[:2] += [0.02, 0.0]
    assert triangulate([first, near, hand.look()]) == pytest.approx(hand.thing, abs=1e-6)
    # The same pixel from two places: something that travels with the camera, at no distance at all.
    assert triangulate([first, Look(hand.view, hand.position.copy(), first.center, first.bbox)]) is None


@pytest.mark.rendering
@pytest.mark.parametrize("start", [(0.02, 0.0), (-0.04, 0.05)])
def test_the_simulated_arm_finds_a_block_with_its_wrist_camera_and_picks_it_up_there(start):
    k, truth = setup()
    # Looks straight down from 5 cm ahead of the fingertips and 4 cm above them.
    mount = View([[0, 1, 0, 0.05], [1, 0, 0, 0], [0, 0, -1, 0.04], [0, 0, 0, 1]], 208.5, 208.5, 160, 120, 320, 240)
    camera = MountedSimCamera("wrist", mount, k.body)
    k.world.boxes.pop("block")                      # the kernel is told nothing about where the block is

    def look():
        frame = camera.capture(k)
        picture = np.asarray(frame.image).astype(float)
        # The block's lit top face, by colour alone.
        ys, xs = np.nonzero((picture[..., 0] > 200) & (picture[..., 0] > 1.5 * picture[..., 2]))
        if len(xs) < 50:
            return Look(frame.view, frame.tool[:3, 3])
        box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        return Look(frame.view, frame.tool[:3, 3], [xs.mean(), ys.mean()], box)

    def move(goal):
        to = k.world.from_base("work", [*goal, k.tool[2, 3]])
        return k.run([{"do": "move_to", "to": to.tolist()}, {"do": "settle"}])

    try:
        k.enable()
        assert k.run([{"do": "line", "up": 0.14}, {"do": "line", "forward": start[0], "left": start[1]}]).ok
        found = locate(move, look, arrive_m=0.01)
        top = truth.boxes["block"].pose[:3, 3] + [0, 0, 0.05]
        assert found.ok and found.data["looks"] >= 2
        assert [found.data[key] for key in "xyz"] == pytest.approx(top, abs=0.01)
        assert found.data["extent_m"] == pytest.approx([0.04, 0.04], abs=0.008)
        # The grasp comes from the located top face alone: fingertips 2 cm past it, pads 3 cm below it.
        face = k.world.from_base("work", [found.data[key] for key in "xyz"])
        plan = [{"do": "gripper", "aperture_mm": 65},
                {"do": "move_to", "to": (face + [0.02, 0, 0.047]).tolist()},
                {"do": "move_to", "to": (face + [0.02, 0, -0.03]).tolist()},
                {"do": "grip", "expect_mm": [35, 45]},
                {"do": "line", "up": 0.06}]
        assert k.run(plan).ok
        assert truth.held is not None and truth.boxes["block"].pose[2, 3] > 0.24
    finally:
        k.close()
