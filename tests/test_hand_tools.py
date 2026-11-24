"""The tool set's sequences, on a fake arm that moves exactly where it is told unless told it meets contact."""
import dataclasses
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import serving

from world_use.behaviors import Outcome
from world_use.bodies.mujoco_scene import mj
from world_use.hand_tools import DOWN, DaemonArm, Hand, HandTools, region_evidence, tilted
from world_use.tracking import Tracking
from world_use.vision import Observation

HAND = Hand(opening_mm=78, open_mm=(70, 85), fit_mm=70, arrive_m=0.012, hand_camera=None,
            fixed_camera=None, in_view_m=0.3, hover="15-25 cm")


class Arm:
    """An executor whose hand goes where it is told; with `short`, a line that may end on contact stops that far
    short of its pose; with `lost`, its controller stops answering at that stage."""

    failures = (ConnectionError,)

    def __init__(self, short=None, refusal=None, lost=None):
        self.at, self.axis, self.aperture = np.array([0.3, 0.0, 0.3]), DOWN.copy(), 80.0
        self.commands, self.short, self.refusal, self.steps, self.lost = [], short, refusal, 0, lost

    def execute(self, command):
        if self.lost is not None and command.get("stage") == self.lost:
            raise ConnectionError("the controller stopped answering")
        if command["kind"] == "confirm_state":
            same = command["fingerprint"] == self.fingerprint()
            return Outcome("done" if same else "refused", "confirm_state", "inspected state comparison")
        self.commands.append(command)
        self.steps += 1
        if command["kind"] == "check_grip":
            lo, hi = command["expect_mm"]
            within = lo <= self.aperture <= hi
            return Outcome("done" if within else "surprise", "check_grip", "measured", self.measure())
        targets = command["targets"]
        if "gripper" in targets:
            self.aperture = 30.0 if targets["gripper"] else 80.0
        if "point" in targets:
            self.axis = np.asarray(targets["point"], float)
        goal = np.array([targets.get(k, v) for k, v in zip("xyz", self.at, strict=True)], float)
        if command.get("rise_m") is not None:
            goal[2] = self.at[2] + command["rise_m"]
        data = dict(arm="arrived")
        if command.get("via") is not None and self.short:
            start = np.array([command["via"][k] for k in "xyz"]) if command["via"] else goal + [0, 0, 0.1]
            goal = goal - self.short * (goal - start) / np.linalg.norm(goal - start)
            data = dict(arm="stopped_on_contact", short_m=self.short)
        self.at = goal
        return Outcome("done", "move", "moved", dict(self.measure(), **data))

    def measure(self):
        return dict(step=self.steps, xyz=self.at.tolist(), aperture_mm=self.aperture)

    def position(self, observation):
        return np.asarray(observation["xyz"])

    def pose(self):
        return self.at.copy(), self.axis.copy()

    def orientation(self, point, jaws=None):
        return dict(point=[float(v) for v in point], jaws=None if jaws is None else [float(v) for v in jaws]), jaws

    def lean_refusal(self, direction, tilt_deg, path):
        return self.refusal

    def hand_look(self, frame=None):
        return None

    def cameras(self):
        return ["front"]

    def frame(self, camera):
        return SimpleNamespace(id="frame-1")

    def validate_pose(self, targets):
        if "z" not in targets:
            raise ValueError("a pose needs z")

    def fingerprint(self):
        return str(self.steps)

    def record(self, name, row):
        pass

    def key(self):
        return self.steps


class Tracks:
    def __init__(self):
        self.targets, self.lost, self.seen = {}, {}, None

    def select(self, frame, target, point=None, box=None, space="image"):
        self.targets[target] = dict(camera="front")
        return dict(target=target, valid=True, camera="front", frame=frame)

    def observe(self, names):
        return [dict(target=name, valid=True, camera="front", seen=self.seen) for name in names]

    def close(self):
        pass


def tools(arm, hand=HAND):
    return HandTools(arm, hand, Tracks())


def moves(arm):
    return [c for c in arm.commands if c["kind"] == "move"]


GRASP = dict(object_target="mug", select=dict(camera="front", point=[10, 10]), approach=dict(x=0.3, y=0.0, z=0.1),
             lift=dict(z=0.14), expect_grip_mm=[20, 40])


def test_a_leaning_grasp_turns_in_place_then_comes_in_along_its_own_axis():
    arm = Arm(short=0.02)
    record, _ = tools(arm).call("grasp", dict(GRASP, tilt_deg=45, toward=dict(x=0.3, y=0.2)))
    turn, approach = moves(arm)[0:2]
    # One turn where the hand is, with the fingers opening; the executor may make it in parts.
    assert turn["stage"] == "grasp_lean" and turn["in_place"] and turn["settle"] == "pass"
    assert {"x", "y", "z"}.isdisjoint(turn["targets"]) and turn["targets"]["gripper"] == 0
    axis = np.array(turn["targets"]["point"])
    assert axis == pytest.approx([0, np.sqrt(0.5), -np.sqrt(0.5)])
    assert turn["targets"]["jaws"] == pytest.approx([1, 0, 0])
    # From 10 cm back along that axis, ending on contact 2 cm short.
    via = np.array([approach["via"][k] for k in "xyz"])
    assert via == pytest.approx(np.array([0.3, 0.0, 0.1]) - 0.1 * axis)
    assert record["leaning"] == dict(tilt_deg=45, toward=[0.0, 1.0]) and record["approach_ended_short_m"] == 0.02
    assert record["status"] == "done"


def test_a_lean_the_arm_cannot_make_is_refused_before_the_hand_turns():
    arm = Arm(refusal=("the wrist cannot lean that way", "lean another way"))
    record, text = tools(arm).call("grasp", dict(GRASP, tilt_deg=45, toward=dict(x=0.5, y=0.0)))
    assert record["status"] == "refused" and "the wrist cannot lean that way" in text
    assert moves(arm) == []


def test_an_arm_lost_during_a_call_ends_it_stopped_after_what_ran():
    arm = Arm(lost="grasp_close")
    record, text = tools(arm).call("grasp", GRASP)
    assert record["status"] == "stopped" and "the controller stopped answering" in text
    assert [c["stage"] for c in moves(arm)] == ["grasp_approach"]


@pytest.mark.parametrize("arguments,message", [
    (dict(tilt_deg=45), "tilt_deg needs toward"),
    (dict(toward=dict(x=0.3, y=0.2)), "toward is where tilt_deg leans"),
    (dict(tilt_deg=45, yaw_deg=10, toward=dict(x=0.3, y=0.2)), "give tilt_deg or yaw_deg"),
    (dict(tilt_deg=75, toward=dict(x=0.3, y=0.2)), "tilt_deg is between 0 and 60"),
    (dict(lift=dict(z=0.2)), "the test lift rises 3-5 cm"),
    (dict(expect_grip_mm=[20, 90]), "max <= 78"),
    (dict(open_mm=60), "give no open_mm"),
])
def test_grasp_arguments_are_refused_before_any_motion(arguments, message):
    arm = Arm()
    record, _ = tools(arm).call("grasp", dict(GRASP, **arguments))
    assert record["status"] == "refused" and message in record["message"]
    assert arm.commands == []


class PartlyOpeningArm(Arm):
    """An executor whose fingers can open partway."""

    opens_partly = True


def test_a_hand_that_opens_partly_takes_an_opening_wider_than_the_object():
    arm = PartlyOpeningArm()
    hand = tools(arm)
    assert "open_mm" in next(s for s in hand.schemas() if s["name"] == "grasp")["parameters"]["properties"]
    record, _ = hand.call("grasp", dict(GRASP, open_mm=50))
    assert moves(arm)[0]["targets"]["opening_mm"] == 50 and record["status"] == "done"
    record, _ = hand.call("grasp", dict(GRASP, open_mm=30))
    assert record["status"] == "refused" and "wider than expect_grip_mm" in record["message"]


def test_a_grasp_stopped_before_its_close_closes_only_on_its_checkpoint_while_nothing_moved():
    arm = Arm()
    hand = tools(arm)
    record, text = hand.call("grasp", dict(GRASP, stop_before_close=True))
    assert record["status"] == "stopped" and [m["stage"] for m in moves(arm)] == ["grasp_approach"]
    assert arm.aperture == 80.0  # still open
    checkpoint = next(a["data"]["checkpoint"] for a in json.loads(text)["assessments"])
    _, text = hand.call("release_placement", dict(checkpoint=checkpoint, note="not a place"))
    assert "release checkpoint is missing" in text  # nor does a grasp's checkpoint release anything
    record, _ = hand.call("grasp", dict(GRASP, stop_before_close=True))
    checkpoint = record["assessments"][-1]["outcome"]["data"]["checkpoint"]
    looked = []  # where the hand was each time the tracker looked
    track = hand.tracking.observe
    hand.tracking.observe = lambda names: looked.append(arm.at[2]) or track(names)
    record, _ = hand.call("close_grasp", dict(checkpoint=checkpoint, note="fingers either side of the mug"))
    assert record["status"] == "done" and [m["stage"] for m in moves(arm)][-2:] == ["grasp_close", "grasp_test_lift"]
    assert looked == [pytest.approx(0.14)]  # after the lift only: a hand down over the object hides it
    _, text = hand.call("close_grasp", dict(checkpoint=checkpoint, note="again"))
    assert "grasp checkpoint is missing, stale or consumed" in text
    record, _ = hand.call("grasp", dict(GRASP, stop_before_close=True))
    checkpoint = record["assessments"][-1]["outcome"]["data"]["checkpoint"]
    arm.steps += 1  # the arm moved since
    _, text = hand.call("close_grasp", dict(checkpoint=checkpoint, note="moved"))
    assert "stale" in text and arm.aperture == 80.0


@pytest.mark.parametrize("max_lean_deg,lean_deg", [(60, 30), (20, 20)])
def test_a_push_leans_back_meets_contact_and_backs_off_along_its_line_to_its_start(max_lean_deg, lean_deg):
    arm = Arm(short=0.05)
    start, end = dict(x=0.3, y=-0.05, z=0.1), dict(x=0.3, y=0.1, z=0.1)
    hand = tools(arm, dataclasses.replace(HAND, max_lean_deg=max_lean_deg))
    record, _ = hand.call("push", dict(start=start, end=end))
    close, line, back = moves(arm)
    assert close["targets"] == {"gripper": 1} and line["via"] == start
    lean = np.radians(lean_deg)
    assert np.array(line["targets"]["point"]) == pytest.approx([0, np.sin(lean), -np.cos(lean)])
    assert record["pushed"] == dict(ended="stopped_on_contact", travelled_mm=100, short_mm=50)
    # Back along the line, never past where it began, standing up as it goes.
    assert [back["targets"][k] for k in "xyz"] == pytest.approx([0.3, -0.05, 0.1])
    assert back["targets"]["point"] == list(DOWN)


def test_a_push_the_wrist_cannot_lean_into_stays_upright():
    arm = Arm(refusal=("no", "no"))
    tools(arm).call("push", dict(start=dict(x=0.3, y=0.0, z=0.1), end=dict(x=0.45, y=0.0, z=0.1)))
    line = moves(arm)[1]
    assert line["targets"]["point"] == pytest.approx(list(DOWN))


PLACE = dict(object_target="mug", move=dict(x=0.2, y=0.1, z=0.12), withdraw=dict(x=0.2, y=0.1, z=0.2),
             expect_grip_mm=[20, 40])


@pytest.mark.parametrize("between", [None, "push", "changed"])
def test_release_needs_an_unchanged_inspected_state_and_happens_once(between):
    arm = Arm()
    hand = tools(arm)
    hand.call("grasp", GRASP)
    record, _ = hand.call("place", PLACE)
    (decision,) = (a["outcome"] for a in record["assessments"])
    assert record["status"] == "stopped" and decision["kind"] == "placement_decision"
    checkpoint = decision["data"]["checkpoint"]
    if between == "push":
        hand.call("push", dict(start=dict(x=0.2, y=0.0, z=0.2), end=dict(x=0.2, y=0.1, z=0.2)))
    elif between == "changed":
        arm.steps += 1
    released, _ = hand.call("release_placement", dict(checkpoint=checkpoint, note="it rests in the bowl"))
    if between:
        assert released["status"] == "refused" and "missing, stale or consumed" in released["message"]
        return
    assert released["status"] == "done" and arm.aperture == 80.0
    assert [c["stage"] for c in moves(arm)[-2:]] == ["place_release", "place_withdraw"]
    again, _ = hand.call("release_placement", dict(checkpoint=checkpoint, note="again"))
    assert again["status"] == "refused"


def observation(z, *, iou, shift, frame="f2", camera="front"):
    target = dict(target="cube", camera=camera, frame=frame, size=[640, 480],
                  motion=dict(from_frame="f1", mask_iou=iou, center_delta_px=[0.0, shift]))
    return dict(xyz=[0.3, 0.0, z], targets=[target])


@pytest.mark.parametrize("iou,shift,unchanged", [(0.95, 2.0, True), (0.95, 40.0, False), (0.6, 2.0, False)])
def test_region_evidence_flags_a_region_that_stayed_while_the_hand_rose(iou, shift, unchanged):
    before = dict(xyz=[0.3, 0.0, 0.1], targets=[dict(target="cube", camera="front", frame="f1")])
    after = observation(0.14, iou=iou, shift=shift)
    facts = region_evidence("cube", before, after, np.array([0.0, 0.0, 0.04]), "front")
    assert facts["unchanged"] is unchanged and facts["hand_vertical_travel_m"] == 0.04


@pytest.mark.parametrize("travel,camera", [([0, 0, 0.01], "front"), ([0.05, 0, 0.04], "front"),
                                           ([0, 0, 0.04], "top")])
def test_region_evidence_compares_only_vertical_travel_seen_by_its_camera(travel, camera):
    before = dict(xyz=[0.3, 0.0, 0.1], targets=[dict(target="cube", camera="front", frame="f1")])
    after = observation(0.14, iou=0.99, shift=0.0, camera=camera)
    assert region_evidence("cube", before, after, np.array(travel), "front") is None


# -- on the twin, through the daemon ------------------------------------------------------------------------

REBOT = Hand(opening_mm=89, open_mm=(80, 89), fit_mm=80, arrive_m=0.012, hand_camera=None, fixed_camera=None,
             in_view_m=0.3, hover="15-25 cm", max_lean_deg=90, standoff_m=0.05, stands_up=False)
TRAY = dict(name="tray", kind="surface", center=[0.32, 0.0, 0.14], size=[0.30, 0.40, 0.02])
BLOCK = dict(name="block", kind="object", center=[0.34, 0.03, 0.20], size=[0.04, 0.04, 0.10])


@pytest.fixture
def twin(tmp_path, rehearser):
    with serving(tmp_path, dict(box=[TRAY, BLOCK]), rehearser=rehearser) as (daemon, client):
        assert client.run({"do": "line", "up": 0.08}, wait=60)["status"] == "done"   # above the turn height
        yield daemon, client, HandTools(DaemonArm(client, REBOT), REBOT, Tracks())


def radial(daemon, distance=0.0):
    """The block's centre in the base frame, and a point `distance` further out from the base."""
    centre = daemon.k.world.boxes["block"].pose[:3, 3]
    out = np.array([centre[0], centre[1], 0.0]) / np.hypot(centre[0], centre[1])
    return centre, centre + distance * out


def xyz(point):
    return {k: float(v) for k, v in zip("xyz", point, strict=False)}


def test_on_the_twin_a_side_grasp_is_a_lean_of_90_degrees_that_comes_in_along_the_fingers(twin):
    daemon, client, hand = twin
    beyond = radial(daemon, 0.2)[1]
    grip = radial(daemon, 0.02)[1] + [0, 0, 0.02]       # the tool point is at the fingertips: block between the pads
    record, text = hand.call("grasp", dict(object_target="block", approach=xyz(grip), tilt_deg=90,
                                           lift=dict(z=float(grip[2]) + 0.04), toward=xyz(beyond[:2]),
                                           expect_grip_mm=[35, 45]))
    assert record["status"] == "done", text
    assert record["leaning"]["tilt_deg"] == 90
    status = client.status()
    assert 35 <= status["gripper"]["aperture_mm"] <= 45
    assert status["tool_axes"]["approach"][2] == pytest.approx(0, abs=0.05)
    assert status["tool"]["base"][2] == pytest.approx(grip[2] + 0.04, abs=0.01)


@pytest.mark.parametrize("toward,reason", [
    ([-0.2, 0.0], "the gripper cannot turn"),           # fingertips toward the base: out of the reBot's reach
    ([0.15, -0.15], "turning needs U+0.267 or higher"),  # off to the side of the way out: j1 turns low down
])
def test_on_the_twin_a_lean_the_arm_cannot_make_is_refused_with_its_reason_before_it_moves(twin, toward, reason):
    daemon, client, hand = twin
    centre, _ = radial(daemon)
    toward = centre[:2] + toward
    before = client.status()["joints_deg"]
    record, text = hand.call("grasp", dict(object_target="block", approach=xyz(centre), tilt_deg=45,
                                           lift=dict(z=float(centre[2]) + 0.04), toward=xyz(toward[:2]),
                                           expect_grip_mm=[35, 45]))
    assert record["status"] == "refused" and "cannot lean that way" in text and reason in text
    assert client.status()["joints_deg"] == pytest.approx(before, abs=0.05)


def test_on_the_twin_a_move_longer_than_one_segment_goes_in_pieces(twin):
    _, client, hand = twin
    goal = np.array(client.status()["tool"]["base"]) + [0.05, 0.27, 0]       # the reBot's segment is 25 cm
    moved = hand.arm.execute(dict(kind="move", targets=xyz(goal), stage="carry", settle="move"))
    assert moved.ok and moved.data["arm"] == "arrived", moved.message
    assert client.status()["tool"]["base"] == pytest.approx(goal.tolist(), abs=0.01)


def test_on_the_twin_the_same_lean_twice_turns_once(twin):
    daemon, _, hand = twin
    approach, across = tilted(radial(daemon, 0.2)[1] - radial(daemon)[0], 45)
    turn = dict(kind="move", targets=dict(point=approach.tolist(), jaws=across.tolist()), stage="grasp_lean",
                settle="pass", in_place=True)
    for _ in range(2):
        turned = hand.arm.execute(turn)
        assert turned.ok, turned.message


def test_on_the_twin_a_release_checkpoint_outlasts_reading_noise_but_not_another_client_s_job(twin):
    daemon, client, hand = twin
    before = hand.arm.fingerprint()
    body, read = daemon.k.body, daemon.k.body.read
    # A held arm's encoders read a count either side of where it stands: nothing commanded that.
    body.read = lambda: dataclasses.replace(st := read(), q=st.q + [0, 0, 1e-3, 0, 0, 0])
    joints = client.status()["joints_deg"]
    while client.status()["joints_deg"] == joints:
        time.sleep(0.01)
    assert hand.arm.fingerprint() == before
    assert client.run({"do": "line", "up": 0.01}, wait=30)["status"] == "done"
    assert hand.arm.fingerprint() != before


def test_on_the_twin_observe_shows_a_change_made_while_the_arm_stood_still(twin):
    _, _, hand = twin
    hand.tracking.targets["block"] = dict(camera="front")
    hand.tracking.seen = "moved"                        # someone moved the block, and the tracker saw it
    _, before = hand.call("observe_targets", {})
    hand.tracking.seen = "moved back"
    _, after = hand.call("observe_targets", {})
    assert "moved back" in after and "moved back" not in before


def test_on_the_twin_a_push_in_free_air_runs_its_line_and_backs_off_to_its_start(twin):
    _, client, hand = twin
    here = np.array(client.status()["tool"]["base"])
    out = here / np.hypot(*here[:2]) * [1, 1, 0]
    start, end = here + 0.02 * out, here + 0.07 * out
    record, text = hand.call("push", dict(start=xyz(start), end=xyz(end), tilt_deg=90))
    assert record["status"] == "done", text
    assert record["pushed"] == dict(ended="reached_end", travelled_mm=50, short_mm=0)
    assert client.status()["tool"]["base"] == pytest.approx(start.tolist(), abs=0.01)


STAND = 0.10                                             # the mug stands on a known 10 cm block


def mug(spec):
    """A 7 cm square mug of boxes, 8 cm tall and 0.15 kg, with a handle 3 cm out on the side toward the base."""
    body = spec.worldbody.add_body(name="mug", pos=[0.30, 0.0, STAND])
    body.add_freejoint(name="mug_free")
    parts = dict(bottom=([0, 0, 0.0025], [0.035, 0.035, 0.0025], 0.03),
                 wall_out=([0.0325, 0, 0.04], [0.0025, 0.035, 0.04], 0.02),
                 wall_in=([-0.0325, 0, 0.04], [0.0025, 0.035, 0.04], 0.02),
                 wall_left=([0, 0.0325, 0.04], [0.035, 0.0025, 0.04], 0.02),
                 wall_right=([0, -0.0325, 0.04], [0.035, 0.0025, 0.04], 0.02),
                 handle=([-0.065, 0, 0.045], [0.006, 0.006, 0.025], 0.01),
                 handle_top=([-0.05, 0, 0.066], [0.015, 0.006, 0.004], 0.005),
                 handle_low=([-0.05, 0, 0.024], [0.015, 0.006, 0.004], 0.005))
    # The handle resists turning in a pinch, as under rubber pads: torsional friction 3 cm. At MuJoCo's default, 0.5 cm,
    # the mug turned 20 degrees in the fingers as it rose and turned again as the hand dragged it along the stand, and
    # where it ended varied between macOS runs of the same code.
    for name, (centre, half, mass) in parts.items():
        body.add_geom(name=name, type=mj.mjtGeom.mjGEOM_BOX, pos=centre, size=half, mass=mass,
                      friction=[1.0, 0.03 if name.startswith("handle") else 0.005, 0.0001], condim=4)


def test_on_the_twin_a_leaning_hand_carries_a_mug_by_its_handle_in_under_a_roof(tmp_path, rehearser):
    """The handle faces the base, so the hand comes at it from outside, leaning 45 degrees away from the base, and
    carries the mug ahead of its fingertips into a box whose opening faces the base."""
    stand = dict(name="stand", kind="surface", center=[0.30, 0.0, STAND / 2], size=[0.14, 0.14, STAND], frame="base")
    floor = dict(name="floor", kind="surface", center=[0.42, 0.0, STAND / 2], size=[0.16, 0.16, STAND], frame="base")
    top = STAND + 0.12
    walls = [dict(name="roof", center=[0.42, 0.0, top + 0.005], size=[0.16, 0.16, 0.01]),
             dict(name="back", center=[0.505, 0.0, STAND + 0.06], size=[0.01, 0.16, 0.12]),
             dict(name="left", center=[0.42, 0.085, STAND + 0.06], size=[0.16, 0.01, 0.12]),
             dict(name="right", center=[0.42, -0.085, STAND + 0.06], size=[0.16, 0.01, 0.12])]
    # The simulator has the walls and roof; the kernel keeps out of them.
    boxes = [stand, floor, *(dict(w, kind="surface", frame="base", known=False) for w in walls),
             *(dict(w, name=w["name"] + "_out", kind="keep_out", frame="base") for w in walls)]
    with serving(tmp_path, dict(box=boxes), rehearser=rehearser, scenery=mug) as (daemon, client):
        assert client.run({"do": "line", "up": 0.08}, wait=60)["status"] == "done"
        hand = HandTools(DaemonArm(client, REBOT), REBOT, Tracks())

        def mug_pose():
            with daemon.k.body.lock:
                found = daemon.k.body.data.body("mug")
                return found.xpos.copy(), np.degrees(np.arccos(np.clip(found.xmat[8], -1, 1)))

        lean = np.array([np.sqrt(0.5), 0, -np.sqrt(0.5)])
        grip = np.array([0.235, 0, STAND + 0.045]) + 0.015 * lean        # the handle between the finger pads
        record, text = hand.call("grasp", dict(object_target="mug", approach=xyz(grip), tilt_deg=45,
                                               lift=dict(z=float(grip[2]) + 0.04), toward=dict(x=0.30, y=0.0),
                                               expect_grip_mm=[6, 20]))
        assert record["status"] == "done" and record["leaning"] == dict(tilt_deg=45, toward=[1.0, 0.0]), text
        at, tilt = mug_pose()
        # Off its stand and held near upright by the handle, whose thickness the fingers' opening reads.
        assert at[2] - STAND > 0.02 and tilt < 10 and 8 <= client.status()["gripper"]["aperture_mm"] <= 16

        here = np.array(client.status()["tool"]["base"])
        # The hand comes down at via to pass under the roof's keep-out: 2.5 cm or less and the carry would enter it.
        # The mug then brushes the stand on its way in; 5 cm or more presses it down, and the carry stops on contact
        # with the mug still on the stand. The drop is near the middle of the 3-4.25 cm that passed.
        release = np.array([0.42 - 0.065, 0.0, here[2] - 0.035])          # the mug's centre over the box's middle
        record, text = hand.call("place", dict(object_target="mug", move=xyz(release), via=xyz(release - [0.12, 0, 0]),
                                               withdraw=xyz(release - [0.10, 0, 0]), expect_grip_mm=[6, 20]))
        (decision,) = (a["outcome"] for a in record["assessments"])
        assert decision["kind"] == "placement_decision", text
        record, text = hand.call("release_placement", dict(checkpoint=decision["data"]["checkpoint"], note="in"))
        assert record["status"] == "done", text
        at, tilt = mug_pose()
        assert 0.36 < at[0] < 0.48 and abs(at[1]) < 0.05 and at[2] == pytest.approx(STAND, abs=0.01) and tilt < 5
        assert client.status()["tool"]["base"][0] < 0.30                    # the hand is back out of the box


class Orange:
    """A tracker that follows the twin's orange block by its colour: a stand-in for EdgeTAM with the same calls."""
    ready = True

    def select(self, target, frame, **prompt):
        return self.update(target, frame)

    def update(self, target, frame):
        rgb = np.asarray(frame.image).astype(float)
        mask = (rgb[..., 0] > 150) & (rgb[..., 0] > 1.6 * rgb[..., 2]) & (rgb[..., 1] < 0.75 * rgb[..., 0])
        ys, xs = np.nonzero(mask)
        if len(xs) < 20:
            return Observation(frame.camera, frame.id, frame.timestamp, None, None, None, 30)
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        return Observation(frame.camera, frame.id, frame.timestamp, box, (float(xs.mean()), float(ys.mean())), mask, 30)

    def forget(self, target):
        pass

    def close(self):
        pass


@pytest.mark.rendering
def test_on_the_twin_locate_places_the_block_through_a_camera_on_the_hand_and_a_grasp_takes_it_there(tmp_path,
                                                                                                    rehearser):
    camera = dict(name="hand", mount="tool", eye=[0.05, 0, 0.04], look_at=[0.05, 0, -1], up=[-1, 0, 0], fov_deg=75,
                  size=[320, 240])
    with serving(tmp_path, dict(box=[TRAY, BLOCK], camera=[camera]), rehearser=rehearser) as (daemon, client):
        assert client.run({"do": "line", "up": 0.08}, wait=60)["status"] == "done"
        seeing = dataclasses.replace(REBOT, hand_camera="hand")
        hand = HandTools(DaemonArm(client, seeing), seeing, Tracking(client, Orange()))
        frame = client.frame("hand")
        box = Orange().update("block", frame).bbox
        record, text = hand.call("locate", dict(target="block", select=dict(camera="hand", box=list(box))))
        assert record["status"] == "done", text
        centre = daemon.k.world.boxes["block"].pose[:3, 3]
        located = np.array([record["located"][k] for k in "xyz"])
        assert np.linalg.norm(located[:2] - centre[:2]) < 0.015     # the top face's centre, over the block's
        # Centred over the region again, then 2 cm further out, so the block sits between the finger pads.
        out = np.array([centre[0], centre[1], 0.0]) / np.hypot(*centre[:2])
        grip = np.array([*located[:2], centre[2] + 0.02]) + 0.02 * out
        record, text = hand.call("grasp", dict(object_target="block", approach=dict(z=float(grip[2])), tilt_deg=90,
                                               offset=[float(v) for v in 0.02 * out[:2]],
                                               lift=dict(z=float(grip[2]) + 0.04), toward=xyz((grip + 0.2 * out)[:2]),
                                               expect_grip_mm=[35, 45]))
        assert record["status"] == "done", text
