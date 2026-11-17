"""Distinct execution, settling, release-admission, capture, and spending regressions."""

import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from budget import BASE_URL, MODEL, AzureBudget, BudgetStop
from inspect_robots.types import Observation, StepResult
from inspect_robots_agent._llm import ToolCall
from inspect_robots_libero.embodiment import LiberoEmbodiment
from native import Driver, Frames, fingerprint
from world_tools import WorldTools, wrist_view

from world_use.behaviors import Outcome


def observation():
    return Observation(
        images={"agentview": np.zeros((256, 256, 3), dtype=np.uint8)},
        # Tool y along world x and the approach axis straight down, as at the start of an episode.
        state={"eef_state": np.array([0, 0, 0, 0, 1, 0, 1, 0, 0, 0.0]), "gripper_qpos": np.array([0.02, -0.02])},
    )


def test_capture_identity_survives_wrapper_and_new_capture_changes_it():
    frames, obs = Frames(), observation()
    frames.sync(obs)
    first = frames.current["agentview"]
    frames.sync(replace(obs, extra={"env_step": 0}))
    assert frames.current["agentview"] is first
    assert frames.current["agentview"].timestamp == first.timestamp
    frames.sync(replace(obs, images={k: v.copy() for k, v in obs.images.items()}))
    assert frames.current["agentview"].id != first.id
    for _ in range(12):
        frames.sync(observation())
    assert len(frames.cache) == 8


@pytest.mark.parametrize("change", ["pixels", "state", "step"])
def test_release_fingerprint_covers_actual_observation(change):
    obs = observation()
    before = fingerprint(obs)
    if change == "pixels":
        obs.images["agentview"][0, 0, 0] = 1
    elif change == "state":
        obs.state["eef_state"][0] = 0.01
    else:
        obs = replace(obs, state_time=0.05)
    assert fingerprint(obs) != before


def response(status=200, usage=True, completed=True):
    return httpx.Response(
        status,
        json=dict(
            status="completed" if completed else "incomplete",
            usage={"input_tokens": 100, "output_tokens": 10} if usage else {},
            output=[],
        ),
    )


def request():
    return httpx.Request("POST", BASE_URL + "/responses", json=dict(model=MODEL, input=[], tools=[]))


def guard(folder, handle, *, key="test", total=15.0, waits=None):
    def sleep(s):
        waits.append(s)

    transport = httpx.MockTransport(handle)
    return AzureBudget(folder / "api", key=key, total=total, per_arm=7.4, seconds=7200, inner=transport, sleep=sleep)


@pytest.mark.parametrize("case", ["network", "status", "usage", "oversized"])
def test_unknown_usage_blocks_any_following_dispatch(tmp_path, case):
    sent = []

    def handle(req):
        sent.append(req)
        if case == "network":
            raise httpx.ReadTimeout("interrupted")
        if case == "status":
            return response(503)
        if case == "usage":
            return response(usage=False)
        return httpx.Response(200, json={"status": "completed", "usage": {"input_tokens": 999999, "output_tokens": 1}})

    budget = guard(tmp_path, handle)
    budget.begin("baseline")
    for _ in range(2):
        with pytest.raises(BudgetStop):
            budget.handle_request(request())
    assert len(sent) == 1
    assert budget.uncertain and budget.used > 0
    with pytest.raises(BudgetStop):
        budget.begin("world_use")


def test_caps_deadline_and_destination_prevent_dispatch(tmp_path):
    sent = []
    budget = guard(tmp_path, sent.append, total=0.001)
    budget.begin("baseline")
    with pytest.raises(BudgetStop):
        budget.handle_request(request())
    budget.total = 15
    with pytest.raises(BudgetStop):
        budget.handle_request(httpx.Request("POST", "https://example.org/responses", json={"model": MODEL}))
    budget.deadline = 0
    with pytest.raises(BudgetStop):
        budget.handle_request(request())
    assert not sent


def test_settlement_and_no_credentials_in_journal(tmp_path):
    sent = []

    def handle(req):
        sent.append(req)
        return response()

    budget = guard(tmp_path, handle, key="test-private-key")
    budget.begin("baseline")
    budget.handle_request(request())
    assert sent[0].headers["api-key"] == "test-private-key"
    assert "authorization" not in sent[0].headers
    assert budget.used == pytest.approx(0.00035)
    assert budget.estimated == pytest.approx(0.0003)
    assert not budget.uncertain
    for p in (tmp_path / "api").iterdir():
        assert "test-private-key" not in p.read_text()


@pytest.mark.parametrize(("failures", "sent"), [(2, 3), (4, 4)])
def test_a_request_whose_connection_was_never_made_is_sent_again_a_few_times(tmp_path, failures, sent):
    attempts, waits = [], []

    def handle(req):
        attempts.append(req)
        if len(attempts) <= failures:
            raise httpx.ConnectError("no tunnel")
        return response()

    budget = guard(tmp_path, handle, waits=waits)
    budget.begin("baseline")
    if failures < sent:
        budget.handle_request(request())
        assert not budget.uncertain and budget.calls == 1 and budget.estimated == pytest.approx(0.0003)
    else:
        with pytest.raises(BudgetStop):
            budget.handle_request(request())
        assert budget.uncertain
    assert len(attempts) == sent and len(waits) == min(failures, 3)
    journal = [json.loads(line)["event"] for line in (tmp_path / "api" / "ledger.jsonl").read_text().splitlines()]
    assert journal.count("connect_retry") == len(waits) and journal.count("reserved") == 1


class FakeDriver:
    def __init__(self):
        self.observation = observation()
        self.frames = Frames()
        self.frames.sync(self.observation)
        self.tools = SimpleNamespace(execute=lambda *a: SimpleNamespace(error=None))
        self.commands = []
        self.output = []
        self.pose = dict(x=0.0, y=0.0, z=0.5)
        self.steps = 0

    def execute(self, command):
        self.commands.append(command)
        self.steps += command["kind"] == "move"
        self.pose.update({k: v for k, v in command.get("targets", {}).items() if k in self.pose})
        self.pose["z"] += command.get("rise_m") or 0
        self.observation.state["eef_state"][:3] = list(self.pose.values())
        return Outcome("done", command["kind"], "fixture measured check")

    def record(self, *args):
        self.output.append(args)

    def record_observation(self):
        pass

    def measure(self):
        return {"aperture_mm": 40, "eef_state": list(self.pose.values())}


def call(name, **args):
    return ToolCall(id="test", name=name, arguments=json.dumps(dict(note="Fixture observation.", **args)))


def tools_fixture(camera=None, **motion):
    # Exercise the real procedure and adapter without inventing a physical grasp.
    tools = object.__new__(WorldTools)
    tools.driver = FakeDriver()
    tools.native = tools.driver.tools
    tools.pending = tools.seen = None
    tools.located, tools.outlined = {}, set()
    captures = iter(range(100))

    def observe(names):
        if camera is None:
            return [dict(target=n, valid=True) for n in names]
        # Each observation is a new capture whose region is compared with the previous one.
        index = next(captures)
        return [
            dict(
                target=n,
                valid=True,
                camera=camera,
                frame=str(index),
                size=[256, 256],
                motion=dict(from_frame=str(index - 1), **motion),
            )
            for n in names
        ]

    tools.selected = []
    tools.tracking = SimpleNamespace(
        targets={"object": {"camera": camera}, "support": {"camera": camera}},
        lost={},
        observe=observe,
        select=lambda frame, target, **where: tools.selected.append((frame, target, where)) or dict(target=target),
    )
    return tools


def pause(tools):
    result = tools.execute(
        call(
            "place",
            object_target="object",
            destination_target="support",
            move={"z": 0.55},
            withdraw={"z": 0.65},
            expect_grip_mm=[5, 70],
        ),
        tools.driver.observation,
    )
    assert tools.pending is not None
    assert json.loads(result.note)["checkpoint"] == "before_release"
    assert len(tools.driver.commands) == 1
    return tools.pending["id"]


def test_release_stops_for_inspection_then_continues_once_without_approach():
    tools = tools_fixture()
    receipt = pause(tools)
    release = call("release_placement", checkpoint=receipt)
    result = tools.execute(release, tools.driver.observation)
    assert json.loads(result.note)["status"] == "done"
    assert [c.get("stage", c["kind"]) for c in tools.driver.commands] == [
        "place_approach",
        "confirm_state",
        "place_release",
        "check_grip",
        "place_withdraw",
    ]
    count = len(tools.driver.commands)
    result = tools.execute(release, tools.driver.observation)
    assert json.loads(result.note)["status"] == "refused"
    assert len(tools.driver.commands) == count


@pytest.mark.parametrize("change", ["pixels", "state", "motion"])
def test_release_refuses_changed_evidence_without_opening(change):
    tools = tools_fixture()
    receipt = pause(tools)
    if change == "pixels":
        tools.driver.observation.images["agentview"][0, 0, 0] = 5
    elif change == "state":
        tools.driver.observation.state["eef_state"][0] = 0.02
    else:
        tools.execute(call("move_to", targets={"z": 0.7}), tools.driver.observation)
    count = len(tools.driver.commands)
    result = tools.execute(call("release_placement", checkpoint=receipt), tools.driver.observation)
    assert json.loads(result.note)["status"] in ("refused", "stopped")
    assert len(tools.driver.commands) == count


STAYED = dict(mask_iou=0.97, center_delta_px=[0.3, -0.5])
MOVED = dict(mask_iou=0.5, center_delta_px=[1.0, -9.0])


def grasp(tools, **extra):
    pose = dict(x=0.0, y=0.0, z=0.5)
    request = call("grasp", object_target="object", approach=pose, lift={"z": 0.54}, expect_grip_mm=[20, 60], **extra)
    return json.loads(tools.execute(request, tools.driver.observation).note)


@pytest.mark.parametrize(
    ("camera", "motion", "status", "assessment"),
    [
        ("agentview", STAYED, "surprise", "unknown"),
        # A region that moved is no confirmation, so the grasp reports as it would unassessed.
        ("agentview", MOVED, "done", "unknown"),
    ],
)
def test_grasp_reports_a_selected_region_that_stayed_behind_the_lift(camera, motion, status, assessment):
    tools = tools_fixture(camera, **motion)
    result = grasp(tools)
    assert (result["status"], result["assessment"]) == (status, assessment)
    # The lift already ran; the finding adds no opening, retreat or retry.
    assert list(result["stages"])[-2:] == ["test_lift", "check_after_lift"]
    assert tools.driver.commands[-1]["kind"] == "check_grip"
    if status == "surprise":
        assert result["checkpoint"] == "after" and result["observed"] == dict(mask_iou=0.97, center_shift_px=0.58)


def wrist_scene(tools, thing, visible=lambda looks: True, half=0.0, shape=None):
    """A fixed point, or a level square of that half-width, seen by the wrist camera of the fixture's hand."""
    driver, looks = tools.driver, []
    driver.pose.update(z=0.65)

    def eef():
        # Tool y along world x and the approach axis straight down, as at the start of an episode.
        return np.array([*driver.pose.values(), 0, 1, 0, 1, 0, 0, 0], dtype=float)

    def look(names):
        looks.append(driver.steps)
        driver.observation.state["eef_state"][:] = eef()
        if not visible(len(looks)):
            return [dict(target=n, valid=False, tracking="lost") for n in names]
        u, v = (float(c) for c in wrist_view(eef()).project(thing)[0][0])
        box = [u - 10, v - 15, u + 10, v + 15]
        if half:
            # The tracker reports the part inside the image: its box and that part's centre.
            corners = [thing + [dx, dy, 0] for dx in (-half, half) for dy in (-half, half)]
            px = np.array([wrist_view(eef()).project(c)[0][0] for c in corners], dtype=float)
            box = [*np.clip(px.min(axis=0), 0, 256), *np.clip(px.max(axis=0), 0, 256)]
            u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        seen = dict(valid=True, camera="wrist", center=[u, v], bbox=box)
        if shape:
            # The tracker's narrowest box around the region: its long side's direction and both sides in pixels.
            axis_deg, length_px, width_px = shape
            seen["dominant_region"] = dict(center=[u, v], axis_deg=axis_deg, length_px=length_px, width_px=width_px)
        return [dict(target=n, **seen) for n in names]

    tools.tracking.observe = look
    tools.tracking.targets["object"]["camera"] = "wrist"


def test_a_wrist_selection_centres_the_hand_over_the_region_before_descending():
    tools = tools_fixture()
    driver, thing = tools.driver, np.array([0.07, -0.04, 0.45])
    wrist_scene(tools, thing)
    result = grasp(tools)
    approach = next(c for c in driver.commands if c.get("stage") == "grasp_approach")
    assert result["status"] == "done" and [approach["targets"][k] for k in "xy"] == pytest.approx(thing[:2], abs=2e-3)
    # The region was placed from looks at least 3 cm apart, and the policy is told where it is.
    assert [c["stage"] for c in driver.commands[:2]] == ["grasp_centre", "grasp_centre"]
    assert result["centred"]["looks"] >= 2 and result["centred"]["z"] == pytest.approx(0.45, abs=5e-3)
    # The first move already heads for where the look meets the height assumed for a grasp.
    assert [driver.commands[0]["targets"][k] for k in "xy"] == pytest.approx(thing[:2], abs=0.02)
    # A hand left closed by an earlier attempt opens before it comes down, and fingers still
    # creeping over their last millimetres do not stop the grasp.
    assert approach["targets"]["gripper"] == 0 and driver.commands[0]["targets"]["gripper"] == 0
    assert approach["expect_mm"] == driver.commands[0]["expect_mm"] == [70, 85]
    assert "x" not in driver.commands[-2]["targets"] and driver.commands[-2]["rise_m"] == pytest.approx(0.04)
    # The region's box, 20 by 30 pixels from 30 cm above it, is reported in millimetres.
    assert result["centred"]["extent_mm"] == pytest.approx([36, 53], abs=2)


@pytest.mark.parametrize(("lift", "status"), [(0.54, "done"), (0.52, "done"), (0.545, "refused"), (0.515, "refused")])
def test_a_test_lift_of_exactly_three_or_five_centimetres_is_accepted(lift, status):
    tools = tools_fixture()
    pose = dict(x=0.0, y=0.0, z=0.49)
    request = call("grasp", object_target="object", approach=pose, lift={"z": lift}, expect_grip_mm=[20, 60])
    assert json.loads(tools.execute(request, tools.driver.observation).note)["status"] == status


def test_a_wrist_selected_grasp_needs_no_x_or_y_and_any_other_grasp_does():
    tools = tools_fixture()
    thing = np.array([0.07, -0.04, 0.45])
    wrist_scene(tools, thing)
    heights = dict(object_target="object", approach={"z": 0.5}, lift={"z": 0.54}, expect_grip_mm=[20, 60])
    done = json.loads(tools.execute(call("grasp", **heights), tools.driver.observation).note)
    approach = next(c for c in tools.driver.commands if c.get("stage") == "grasp_approach")
    assert done["status"] == "done" and [approach["targets"][k] for k in "xy"] == pytest.approx(thing[:2], abs=2e-3)
    tools.tracking.targets["object"]["camera"] = "agentview"
    count = len(tools.driver.commands)
    refused = json.loads(tools.execute(call("grasp", **heights), tools.driver.observation).note)
    assert refused["status"] == "refused" and len(tools.driver.commands) == count


def test_an_offset_moves_the_grasp_point_off_the_centre_of_a_wrist_region():
    tools = tools_fixture()
    thing = np.array([0.07, -0.04, 0.45])
    wrist_scene(tools, thing)
    result = grasp(tools, offset=[0, 0.04])
    approach = next(c for c in tools.driver.commands if c.get("stage") == "grasp_approach")
    assert result["status"] == "done" and result["centred"]["y"] == pytest.approx(thing[1], abs=2e-3)
    assert [approach["targets"][k] for k in "xy"] == pytest.approx(thing[:2] + [0, 0.04], abs=2e-3)
    # Without a wrist region there is no centre to measure the offset from.
    tools.tracking.targets["object"]["camera"] = "agentview"
    count = len(tools.driver.commands)
    refused = grasp(tools, offset=[0, 0.04])
    assert refused["status"] == "refused" and len(tools.driver.commands) == count


def test_a_yaw_turns_the_finger_axis_about_the_vertical_and_keeps_the_approach_axis_down():
    tools = tools_fixture()
    assert grasp(tools, yaw_deg=90)["status"] == "done"
    approach = next(c for c in tools.driver.commands if c.get("stage") == "grasp_approach")["targets"]
    a, b = (np.array([approach[f"r{i}{j}"] for i in range(3)]) for j in range(2))
    # World x for the fingers, and the same downward approach axis as at the start of an episode.
    assert a == pytest.approx([1, 0, 0]) and np.cross(a, b) == pytest.approx([0, 0, -1])
    count = len(tools.driver.commands)
    refused = grasp(tools, yaw_deg=120, select=dict(camera="wrist", point=[5, 5]))
    # A refused call changes nothing, the selection included.
    assert refused["status"] == "refused" and len(tools.driver.commands) == count and not tools.selected


def test_a_position_found_once_is_not_worked_out_again_until_the_region_is_reselected_or_taken():
    tools = tools_fixture()
    wrist_scene(tools, np.array([0.07, -0.04, 0.45]))
    first = json.loads(tools.execute(call("locate", target="object"), tools.driver.observation).note)["located"]
    moves = len(tools.driver.commands)
    again = json.loads(tools.execute(call("locate", target="object"), tools.driver.observation).note)["located"]
    assert [again[k] for k in "xyz"] == [first[k] for k in "xyz"] and len(tools.driver.commands) == moves
    # A grasp of it goes straight to the approach, and fingers that closed on it may have moved it.
    assert grasp(tools)["status"] == "done"
    assert tools.driver.commands[moves]["stage"] == "grasp_approach" and not tools.located
    # A new selection is a new region.
    tools.located["object"] = [0.0, 0.0, 0.45]
    tools.execute(call("select_target", target="object", camera="agentview", point=[5, 5]), tools.driver.observation)
    assert not tools.located


@pytest.mark.parametrize(
    ("extra", "outlined", "turned"),
    [({}, True, True), ({}, False, False), (dict(yaw_deg=0), True, False), (dict(offset=[0.02, 0]), True, False)],
)
def test_fingers_turn_across_the_narrow_side_of_an_outlined_long_region_unless_the_policy_placed_them(
    extra, outlined, turned
):
    tools = tools_fixture()
    # A box top 34 pixels long at 30 degrees in the published wrist image and 14 wide, from 30 cm above.
    wrist_scene(tools, np.array([0.07, -0.04, 0.45]), shape=(30, 34, 14))
    # Only a region selected with a box around it is taken to be the whole object.
    tools.outlined = {"object"} if outlined else set()
    result = grasp(tools, **extra)
    approach = next(c for c in tools.driver.commands if c.get("stage") == "grasp_approach")
    assert result["status"] == "done" and result["centred"]["across_mm"] == pytest.approx(25, abs=3)
    fit = result["centred"]["fit_yaw_deg"]
    assert (result.get("fingers_turned_to_yaw_deg") == fit) == turned
    if turned:
        # The finger axis, tool x, lies across the narrow side and the approach axis stays down.
        axis = np.array([approach["targets"][k] for k in ("r00", "r10", "r20")])
        assert axis == pytest.approx([np.sin(np.radians(fit)), np.cos(np.radians(fit)), 0], abs=1e-9)
    elif "yaw_deg" not in extra:
        assert "r00" not in approach["targets"]


def test_a_round_or_wide_region_leaves_the_fingers_as_they_are():
    for shape in ((30, 20, 19), (30, 80, 55)):
        tools = tools_fixture()
        wrist_scene(tools, np.array([0.07, -0.04, 0.45]), shape=shape)
        tools.outlined = {"object"}
        result = grasp(tools)
        approach = next(c for c in tools.driver.commands if c.get("stage") == "grasp_approach")
        assert result["status"] == "done" and "fingers_turned_to_yaw_deg" not in result
        assert "r00" not in approach["targets"]


@pytest.mark.parametrize(("stalls", "status"), [(1, "done"), (2, "surprise")])
def test_one_centring_move_that_stops_short_of_a_guessed_goal_still_brings_a_new_look(stalls, status):
    tools = tools_fixture()
    thing = np.array([0.07, -0.04, 0.45])
    wrist_scene(tools, thing)
    execute, short = tools.driver.execute, []

    def reach(command):
        outcome = execute(command)
        if command.get("stage") == "locate" and len(short) < stalls:
            short.append(command)
            return Outcome("surprise", "move", "fixture: past the arm's reach", dict(arm="not_approaching"))
        return outcome

    tools.driver.execute = reach
    found = json.loads(tools.execute(call("locate", target="object"), tools.driver.observation).note)
    assert found["status"] == status
    if status == "done":
        assert [found["located"][k] for k in "xy"] == pytest.approx(thing[:2], abs=5e-3)
    else:
        assert len(tools.driver.commands) == 2 and "reach" in found["message"]


def test_fingers_that_will_not_open_end_a_grasp_at_its_first_centring_move():
    tools = tools_fixture()
    wrist_scene(tools, np.array([0.07, -0.04, 0.45]))
    execute = tools.driver.execute

    def jammed(command):
        outcome = execute(command)
        if command.get("stage") == "grasp_centre":
            return Outcome("surprise", "move", "the gripper stopped outside the expected opening", dict(arm="arrived"))
        return outcome

    tools.driver.execute = jammed
    result = grasp(tools)
    assert result["status"] == "surprise" and len(tools.driver.commands) == 1


def test_a_region_missing_from_the_wrist_view_stops_before_any_descent():
    tools = tools_fixture()
    tools.tracking.observe = lambda names: [dict(target=n, valid=False, tracking="lost") for n in names]
    tools.tracking.targets["object"]["camera"] = "wrist"
    result = grasp(tools)
    assert result["status"] == "surprise" and "not visible" in result["message"] and not tools.driver.commands


def test_locate_reports_where_a_wrist_region_is_without_descending():
    tools = tools_fixture()
    thing = np.array([-0.06, 0.09, 0.5])
    wrist_scene(tools, thing)
    found = json.loads(tools.execute(call("locate", target="object"), tools.driver.observation).note)
    assert found["status"] == "done"
    assert [found["located"][k] for k in "xyz"] == pytest.approx(thing, abs=5e-3)
    # The policy is told where that point is in the image it sees next: under the fingers, above centre.
    assert found["located"]["pixel"] == pytest.approx([128, 93], abs=6)
    # Every move was level and left the fingers alone, whatever they hold, and the hand ended over the region.
    assert all(c["stage"] == "locate" and c["targets"].keys() == {"x", "y"} for c in tools.driver.commands)
    assert [tools.driver.pose[k] for k in "xy"] == pytest.approx(thing[:2], abs=5e-3)
    # A region in the fixed camera gives one direction only and no second look. Asking for one
    # is refused before it replaces the wrist selection.
    count = len(tools.driver.commands)
    elsewhere = call("locate", target="object", select=dict(camera="agentview", point=[5, 5]))
    refused = json.loads(tools.execute(elsewhere, tools.driver.observation).note)
    assert refused["status"] == "refused" and "wrist" in refused["message"]
    assert len(tools.driver.commands) == count and not tools.selected


def test_a_lift_too_small_to_compare_is_not_judged(monkeypatch):
    tools = tools_fixture("agentview", **STAYED)
    execute = tools.driver.execute
    monkeypatch.setattr(tools.driver, "execute", lambda c: execute({**c, "rise_m": 0.01} if c.get("rise_m") else c))
    result = grasp(tools)
    assert result["status"] == "done" and result["assessment"] == "unknown" and "assessments" not in result


def test_a_region_hidden_by_the_hand_does_not_stop_the_grasp():
    tools = tools_fixture("agentview", **MOVED)
    looks = []
    observe = tools.tracking.observe

    def lost_after_first_look(names):
        looks.append(tools.driver.steps)
        if len(looks) == 1:
            return observe(names)
        return [dict(target=n, valid=False, tracking="lost", reason="the tracker reports lost") for n in names]

    tools.tracking.observe = lost_after_first_look
    result = grasp(tools)
    # The tracker is asked before the approach and after the lift only, and its loss is reported, not fatal.
    assert looks == [0, 3] and result["status"] == "done" and result["assessment"] == "unknown"
    assert result["observations"]["after"]["targets"] == [
        dict(target="object", valid=False, reason="the tracker reports lost")
    ]
    assert tools.driver.commands[-1]["kind"] == "check_grip"


def test_a_lost_region_is_reported_and_blocks_neither_grasp_nor_place():
    tools = tools_fixture("agentview", **MOVED)
    tools.tracking.observe = lambda names: [dict(target=n, valid=False, tracking="lost") for n in names]
    grasped = grasp(tools)
    assert grasped["status"] == "done" and grasped["observations"]["after"]["targets"] == [
        dict(target="object", valid=False)
    ]
    # The grip itself is measured; the destination is optional evidence as well.
    request = call("place", object_target="object", move={"z": 0.45}, withdraw={"z": 0.55}, expect_grip_mm=[5, 70])
    placed = json.loads(tools.execute(request, tools.driver.observation).note)
    assert placed["status"] == "stopped" and placed["checkpoint"] == "before_release" and tools.pending


def contact(tools, monkeypatch, *stages, short_m=0.096):
    """The fixture's driver meets contact that far short of the pose in these stages."""
    execute = tools.driver.execute
    held = Outcome("done", "move", "fixture: contact", dict(arm="stopped_on_contact", short_m=short_m))

    def run(command):
        tools.driver.commands.append(command) if command.get("stage") in stages else None
        return held if command.get("stage") in stages else execute(command)

    monkeypatch.setattr(tools.driver, "execute", run)


def test_grasp_and_place_report_how_far_short_of_the_pose_an_approach_ended(monkeypatch):
    tools = tools_fixture()
    contact(tools, monkeypatch, "grasp_approach", "place_approach")
    assert grasp(tools)["approach_ended_short_m"] == 0.096
    request = call("place", object_target="object", move={"z": 0.45}, withdraw={"z": 0.6}, expect_grip_mm=[5, 70])
    placed = json.loads(tools.execute(request, tools.driver.observation).note)
    # A can that came down on a rim still pauses for the policy, which is told how high it stopped.
    assert placed["checkpoint"] == "before_release" and placed["approach_ended_short_m"] == 0.096


def test_a_place_through_a_point_comes_in_and_leaves_by_it():
    tools = tools_fixture()
    via = dict(x=0.0, y=-0.2, z=0.5)
    request = dict(object_target="object", move={"y": 0.0, "z": 0.5}, withdraw={"z": 0.7}, expect_grip_mm=[5, 70])
    tools.execute(call("place", via=via, travel_z=0.7, **request), tools.driver.observation)
    approach = tools.driver.commands[0]
    assert (approach["stage"], approach["via"], approach["travel_z"]) == ("place_approach", via, 0.7)
    tools.execute(call("release_placement", checkpoint=tools.pending["id"]), tools.driver.observation)
    out = [c["targets"] for c in tools.driver.commands if c.get("stage") == "place_withdraw"]
    assert out == [via, {"z": 0.7}]
    refused = call("place", via={"x": 0.0, "y": -0.2}, **request)
    assert json.loads(tools.execute(refused, tools.driver.observation).note)["status"] == "refused"


def test_a_push_closes_the_fingers_runs_its_line_and_backs_off_along_it(monkeypatch):
    tools = tools_fixture()
    start, end = dict(x=0.0, y=-0.1, z=0.45), dict(x=0.0, y=0.1, z=0.45)
    told = json.loads(tools.execute(call("push", start=start, end=end, travel_z=0.6), tools.driver.observation).note)
    close, line, back = tools.driver.commands
    assert (close["stage"], close["targets"]) == ("push_close", {"gripper": 1})
    assert (line["targets"], line["via"], line["travel_z"]) == (end, start, 0.6)
    assert told["status"] == "done" and told["pushed"] == dict(ended="reached_end", travelled_mm=200, short_mm=0)
    assert back["targets"] == pytest.approx(dict(x=0.0, y=0.08, z=0.45))
    # A drawer that reaches its stop holds the hand there: the push says how far it went.
    tools.driver.commands.clear()
    contact(tools, monkeypatch, "push", short_m=0.05)
    tools.driver.pose.update(y=0.05, z=0.45)
    told = json.loads(tools.execute(call("push", start=start, end=end), tools.driver.observation).note)
    assert told["pushed"] == dict(ended="stopped_on_contact", travelled_mm=150, short_mm=50)
    assert tools.driver.commands[-1]["targets"]["y"] == pytest.approx(0.03)


@pytest.mark.parametrize(
    "change", [dict(start={"x": 0.0, "y": 0.0}), dict(end=dict(x=0.0, y=-0.09, z=0.45)), dict(yaw_deg=120)]
)
def test_a_push_without_two_points_a_line_between_them_or_a_possible_turn_is_refused(change):
    tools = tools_fixture()
    request = dict(start=dict(x=0.0, y=-0.1, z=0.45), end=dict(x=0.0, y=0.1, z=0.45)) | change
    told = json.loads(tools.execute(call("push", **request), tools.driver.observation).note)
    assert told["status"] == "refused" and not tools.driver.commands


def test_a_grasp_that_stops_before_its_lift_reports_no_lift_evidence(monkeypatch):
    tools = tools_fixture("agentview", **MOVED)
    execute = tools.driver.execute
    refuse = Outcome("surprise", "move", "fixture: nothing between the fingers")
    monkeypatch.setattr(tools.driver, "execute", lambda c: refuse if c.get("stage") == "grasp_close" else execute(c))
    result = grasp(tools)
    assert result["status"] == "surprise" and list(result["stages"])[-1] == "close"
    assert "assessments" not in result


def test_the_policy_gets_a_brief_copy_and_the_record_keeps_everything():
    tools = tools_fixture("agentview", **STAYED)
    pose = dict(x=0.0, y=0.0, z=0.5)
    request = call("grasp", object_target="object", approach=pose, lift={**pose, "z": 0.54}, expect_grip_mm=[20, 60])
    note = tools.execute(request, tools.driver.observation).note
    told = json.loads(note)
    (recorded,) = (row for name, row in tools.driver.output if name == "world-use.jsonl")
    # Both observations, state vectors and capture identifiers stay on disk only.
    assert list(recorded["observations"]) == ["before_lift", "after"] and list(told["observations"]) == ["after"]
    assert "eef_state" in recorded["observations"]["after"] and "eef_state" not in note and "frame" not in note
    assert told["status"] == recorded["status"] and told["assessment"] == recorded["assessment"]
    assert told["assessments"][0]["data"] == recorded["assessments"][-1]["outcome"]["data"]
    # Only the procedure's own target is repeated; the destination stays in the record.
    assert [t["target"] for t in told["observations"]["after"]["targets"]] == ["object"]
    assert len(recorded["observations"]["after"]["targets"]) == 2 and len(note) < 1000


def test_a_selection_in_the_grasp_call_binds_the_target_before_any_motion():
    tools = tools_fixture("agentview", **MOVED)
    result = grasp(tools, select=dict(camera="agentview", box=[10, 20, 30, 40]))
    frame = tools.driver.frames.current["agentview"].id
    where = dict(point=None, box=[10, 20, 30, 40], space="image")
    assert result["status"] == "done" and tools.selected == [(frame, "object", where)]
    count = len(tools.driver.commands)
    refused = grasp(tools, select=dict(camera="overhead", point=[1, 2]))
    assert refused["status"] == "refused" and "agentview" in refused["message"]
    assert len(tools.driver.commands) == count and len(tools.selected) == 1


def test_a_plain_move_result_carries_no_sentence_and_a_shortfall_does(monkeypatch):
    tools = tools_fixture()
    plain = json.loads(tools.execute(call("move_to", targets={"z": 0.6}), tools.driver.observation).note)
    assert plain == {"status": "done"}
    short = Outcome(
        "done", "move", "held short", dict(aperture_mm=40.0, applied=dict(last_command_reached_request=False))
    )
    monkeypatch.setattr(tools.driver, "execute", lambda command: short)
    told = json.loads(tools.execute(call("move_to", targets={"z": 0.7}), tools.driver.observation).note)
    assert told["message"] == "held short" and told["applied"] == {"last_command_reached_request": False}


@pytest.mark.parametrize(
    ("motion", "move", "issued"),
    [
        (STAYED, {"z": 0.45}, False),
        (MOVED, {"z": 0.45}, True),
        # Only a vertical approach is compared: sideways travel was never measured.
        (STAYED, {"x": 0.1, "z": 0.45}, True),
    ],
)
def test_place_issues_a_release_receipt_unless_the_region_stayed_behind_a_vertical_approach(motion, move, issued):
    tools = tools_fixture("agentview", **motion)
    request = call(
        "place",
        object_target="object",
        destination_target="support",
        move=move,
        withdraw={"z": 0.55},
        expect_grip_mm=[5, 70],
    )
    result = json.loads(tools.execute(request, tools.driver.observation).note)
    assert (tools.pending is not None) == issued
    assert result["status"] == ("stopped" if issued else "surprise")
    assert [c.get("stage") for c in tools.driver.commands] == ["place_approach"]


def test_place_refuses_a_travel_height_that_is_not_a_number():
    tools = tools_fixture()
    request = dict(object_target="object", move={"x": 0.1, "z": 0.45}, withdraw={"z": 0.7}, expect_grip_mm=[5, 70])
    refused = json.loads(tools.execute(call("place", travel_z="high", **request), tools.driver.observation).note)
    assert refused["status"] == "refused" and not tools.driver.commands


class Hand(LiberoEmbodiment):
    """The published hand and arm timing without a simulator.

    The hand's command state moves 0.125 per step toward whichever end the request is
    past, and the fingers follow it until an object stops them. The arm trails its
    command, so a move issued straight after another starts from a pose the last
    command has already left behind.
    """

    def __init__(
        self,
        *,
        object_mm=None,
        finger_gain=0.5,
        floor_z=None,
        wall_x=None,
        arm_gain=0.3,
        ramp=0.125,
        shove=0.0,
        momentum=0.0,
    ):
        super().__init__(task_id=0, output_dir="unused")
        self.pos, self.command, self.half_mm = np.array([0.0, 0.0, 0.6]), 0.0, 40.0
        self.momentum, self.velocity = momentum, np.zeros(3)
        self.object_mm, self.finger_gain, self.floor_z, self.shove = object_mm, finger_gain, floor_z, shove
        self.wall_x = wall_x
        self.arm_gain, self.ramp = arm_gain, ramp
        self.last = dict(steps=0)

    def observation(self):
        rot = [1.0, 0.0, 0.0, 0.0, -1.0, 0.0]
        return Observation(
            images={"agentview": np.full((4, 4, 3), self.last["steps"] % 251, dtype=np.uint8)},
            state={
                "eef_state": np.concatenate([self.pos, rot, [self.command]]),
                "gripper_qpos": np.array([self.half_mm, -self.half_mm]) / 1000,
            },
        )

    def step(self, action):
        data = np.asarray(action.data, dtype=float)
        self.velocity = self.momentum * self.velocity + self.arm_gain * (data[:3] - self.pos)
        self.pos = self.pos + self.velocity
        if self.floor_z is not None and self.pos[2] < self.floor_z:
            self.pos[2] = self.floor_z
            self.pos[0] += self.shove  # an obstacle that deflects the hand sideways
        if self.wall_x is not None and self.pos[0] > self.wall_x:
            self.pos[0] = self.wall_x
            self.pos[1] += self.shove
        self.command = float(np.clip(self.command + self.ramp * np.sign(2 * data[9] - 1), 0, 1))
        half = self.half_mm + self.finger_gain * (40 * (1 - self.command) - self.half_mm)
        if self.object_mm is not None:
            half = max(half, self.object_mm / 2)
        self.half_mm = float(half)
        self.last = dict(steps=self.last["steps"] + 1)
        return StepResult(self.observation(), reward=0.0, terminated=False, info={"success": False})


def driver(tmp_path, **hand):
    env = Hand(**hand)
    return Driver(env, env.observation(), tmp_path / "execution")


def stage(targets, name, **extra):
    return dict(kind="move", targets=targets, stage=name, settle_steps=20, **extra)


@pytest.mark.parametrize(
    ("object_mm", "status", "gripper"),
    [(None, "surprise", "reached_end"), (55, "surprise", "stopped_apart"), (30, "done", "stopped_apart")],
)
def test_close_returns_a_stable_hand_with_its_measured_opening(tmp_path, object_mm, status, gripper):
    d = driver(tmp_path, object_mm=object_mm)
    out = d.execute(stage({"gripper": 1}, "grasp_close", expect_mm=[20, 40]))
    assert (out.status, out.data["gripper"]) == (status, gripper)
    assert out.data["aperture_mm"] == pytest.approx(object_mm or 0, abs=3)
    # A hand at rest ends the wait; only motion still under way may use the whole bound.
    assert out.data["executed_steps"] < 31
    if status == "surprise":
        assert out.expected == [20, 40] and out.observed == out.data["aperture_mm"]


def test_a_hand_still_moving_at_the_bound_is_not_reported_as_stable(tmp_path):
    d = driver(tmp_path, finger_gain=0.01)
    out = d.execute(stage({"gripper": 1}, "grasp_close", expect_mm=[20, 78]))
    assert out.status == "stopped" and out.data["gripper"] == "still_moving"
    assert out.data["executed_steps"] == 31


def test_fingers_still_creeping_open_inside_the_expected_opening_have_released(tmp_path):
    d = driver(tmp_path, object_mm=50, finger_gain=0.06)
    d.execute(stage({"gripper": 1}, "grasp_close", expect_mm=[40, 78]))
    # Slow fingers are still short of fully open when the bound ends, but well clear of the object.
    out = d.execute(stage({"gripper": 0}, "place_release", expect_mm=[60, 85]))
    assert out.ok and out.data["gripper"] == "still_moving" and 60 < out.data["aperture_mm"] < 77


def test_an_opening_hand_is_not_judged_before_its_command_ramp_ends(tmp_path):
    # A slow ramp leaves the fingers resting on the object well into the wait. That pause is not a stop.
    d = driver(tmp_path, object_mm=50, ramp=0.05)
    assert d.execute(stage({"gripper": 1}, "grasp_close", expect_mm=[40, 60])).ok
    out = d.execute(stage({"gripper": 0}, "place_release", expect_mm=[70, 85]))
    assert out.ok and out.data["gripper"] == "reached_end" and out.data["aperture_mm"] > 77


def test_a_blocked_arm_returns_early_as_not_approaching(tmp_path):
    d = driver(tmp_path, floor_z=0.5)
    out = d.execute(stage({"z": 0.47}, "grasp_approach"))
    assert out.status == "surprise" and out.data["arm"] == "not_approaching"
    assert out.observed["position_error_m"] == pytest.approx(0.03, abs=1e-3)
    assert out.data["executed_steps"] < 13 + 10
    # Where it stopped relative to the request: held up, not shoved aside.
    assert out.data["above_request_m"] == pytest.approx(0.03, abs=1e-3) and out.data["aside_m"] == 0


@pytest.mark.parametrize(
    ("shove", "status", "arm"), [(0.0, "done", "stopped_on_contact"), (0.01, "surprise", "not_approaching")]
)
def test_only_contact_directly_above_the_pose_ends_a_descent_that_allows_it(tmp_path, shove, status, arm):
    d = driver(tmp_path, floor_z=0.5, shove=shove)
    out = d.execute(stage({"z": 0.47}, "grasp_approach", via={}))
    assert (out.status, out.data["arm"]) == (status, arm)
    if out.ok:
        assert out.data["short_m"] == pytest.approx(0.03, abs=2e-3)
        # The next stage rises from where the hand stopped, not from the height it asked for.
        lift = d.execute(stage({}, "grasp_test_lift", rise_m=0.04))
        assert lift.ok and d.observation.state["eef_state"][2] == pytest.approx(0.54, abs=0.012)


@pytest.mark.parametrize(
    ("shove", "status", "arm"), [(0.0, "done", "stopped_on_contact"), (0.01, "surprise", "not_approaching")]
)
def test_a_level_line_also_ends_on_contact_met_along_it(tmp_path, shove, status, arm):
    d = driver(tmp_path, wall_x=0.05, shove=shove)
    out = d.execute(stage({"x": 0.1}, "push", via={"x": 0.0}))
    assert (out.status, out.data["arm"]) == (status, arm)
    if out.ok:
        # What the fingers pushed stopped 5 cm short of the line's end, and the hand rests there on the line.
        assert out.data["short_m"] == pytest.approx(0.05, abs=2e-3)
        assert d.last_action.data[:3] == pytest.approx([0.05, 0, 0.6], abs=2e-3)


def test_a_point_to_pass_through_is_reached_level_and_then_in_height_before_the_line_to_the_pose(tmp_path):
    d = driver(tmp_path)
    out = d.execute(stage({"x": 0.1, "y": 0.0, "z": 0.5}, "place_approach", via={"x": 0.0, "y": 0.1, "z": 0.5}))
    rows = [json.loads(line) for line in (d.output / "steps.jsonl").read_text().splitlines()]
    sinks = next(i for i, row in enumerate(rows) if row["approved"][2] < 0.6 - 1e-9)
    leaves = next(i for i, row in enumerate(rows) if row["approved"][0] > 1e-9)
    assert rows[sinks - 1]["state"]["eef_state"][:3] == pytest.approx([0, 0.1, 0.6], abs=0.012)
    assert rows[leaves - 1]["state"]["eef_state"][:3] == pytest.approx([0, 0.1, 0.5], abs=0.012)
    assert out.ok and d.observation.state["eef_state"][:3] == pytest.approx([0.1, 0, 0.5], abs=0.012)


def test_a_level_first_approach_is_over_the_pose_before_it_descends(tmp_path):
    d = driver(tmp_path)
    out = d.execute(stage({"x": 0.1, "z": 0.5}, "grasp_approach", via={}))
    rows = [json.loads(line) for line in (d.output / "steps.jsonl").read_text().splitlines()]
    descent = next(i for i, row in enumerate(rows) if row["approved"][2] < 0.6 - 1e-9)
    assert abs(rows[descent - 1]["state"]["eef_state"][0] - 0.1) <= 0.012
    assert out.ok and out.data["executed_steps"] == len(rows) and out.data["position_error_m"] <= 0.012


def test_a_descent_blocked_far_above_its_request_ends_early_and_stops_pushing(tmp_path):
    d = driver(tmp_path, floor_z=0.55)
    out = d.execute(stage({"z": 0.43}, "place_approach", via={}))
    assert out.ok and out.data["arm"] == "stopped_on_contact"
    assert out.data["short_m"] == pytest.approx(0.12, abs=2e-3)
    # The command had 17 steps to run and then a wait; it ended a few steps after the hand stopped following.
    assert out.data["executed_steps"] < 21
    # The applied command came back to the hand, so the next move is not held back by the approvers.
    assert abs(d.last_action.data[2] - 0.55) < 0.011
    up = d.execute(dict(kind="move", targets={"z": 0.6}, stage="move_to"))
    assert up.data.get("applied", {}).get("last_command_reached_request", True)


def test_a_hand_that_met_contact_slightly_off_the_pose_comes_to_rest_over_it(tmp_path):
    d = driver(tmp_path, floor_z=0.5, shove=0.002)
    out = d.execute(stage({"z": 0.47}, "grasp_approach", via={}))
    assert out.ok and out.data["arm"] == "stopped_on_contact"
    rows = [json.loads(line) for line in (d.output / "steps.jsonl").read_text().splitlines()]
    slid = max(row["state"]["eef_state"][0] for row in rows)
    # The palm slid while it pressed. At rest it is held over the pose, not left where it slid to.
    assert slid > 0.004 and d.observation.state["eef_state"][0] < 0.8 * slid and d.last_action.data[0] == 0


def test_a_level_leg_that_stalls_reports_how_far_the_hand_is_from_the_whole_request(tmp_path):
    d = driver(tmp_path, wall_x=0.05)
    out = d.execute(stage({"x": 0.1, "z": 0.45}, "place_approach", via={}))
    # The hand never came down: it is still 15 cm above the release height and 5 cm short of it.
    assert out.status == "surprise" and out.data["arm"] == "not_approaching"
    assert out.data["above_request_m"] == pytest.approx(0.15, abs=2e-3)
    assert out.data["aside_m"] == pytest.approx(0.05, abs=2e-3)


def test_a_swinging_arm_has_arrived_only_at_rest_and_its_turning_points_are_not_stalls(tmp_path):
    # An arm with inertia swings past its pose and back before it settles.
    d = driver(tmp_path, arm_gain=0.08, momentum=0.8)
    out = d.execute(dict(stage({"x": 0.1}, "grasp_centre"), settle_steps=40))
    rows = [json.loads(line)["state"]["eef_state"][0] for line in (d.output / "steps.jsonl").read_text().splitlines()]
    # It passed more than the arrival tolerance beyond the pose, turned round there, and was
    # judged neither arrived on the way through nor stalled at the turn.
    assert max(rows) > 0.1 + 0.012
    assert out.ok and out.data["arm"] == "arrived" and abs(rows[-1] - rows[-2]) <= 0.002
    assert out.data["position_error_m"] <= 0.012


@pytest.mark.parametrize(
    ("case", "hand", "command", "arm"),
    [
        ("free", {}, dict(targets={"x": 0.15}), "arrived"),
        ("descent", {}, dict(targets={"x": 0.1, "z": 0.5}, via={}), "arrived"),
        (
            "contact",
            dict(floor_z=0.5),
            dict(targets={"x": 0.1, "z": 0.45}, via={}),
            "stopped_on_contact",
        ),
        ("blocked", dict(wall_x=0.05), dict(targets={"x": 0.15}), "not_approaching"),
    ],
)
def test_the_thresholds_tell_motion_contact_and_a_block_apart_on_arms_that_lag_and_swing_differently(
    tmp_path, case, hand, command, arm
):
    # The thresholds were set on one simulated arm. A fake arm that closes 8 to 70 percent of its gap to the
    # command each step, and keeps none to 70 percent of its speed, is judged the same way. A slow arm that
    # keeps 85 percent of its speed is not: its free descent reads as blocked.
    for gain in (0.08, 0.2, 0.45, 0.7):
        for momentum in (0.0, 0.5, 0.7):
            d = driver(tmp_path / f"{gain}-{momentum}", arm_gain=gain, momentum=momentum, **hand)
            out = d.execute(dict(kind="move", stage=case, settle_steps=40, **command))
            assert out.data["arm"] == arm, (gain, momentum)


def test_a_stage_that_only_passes_through_its_pose_does_not_wait_for_the_hand_to_stop(tmp_path):
    resting, passing = driver(tmp_path / "a", momentum=0.7), driver(tmp_path / "b", momentum=0.7)
    stopped = resting.execute(stage({"x": 0.1}, "carry"))
    through = passing.execute(dict(stage({"x": 0.1}, "look"), rest=False))
    assert stopped.data["arm"] == through.data["arm"] == "arrived" and through.data["position_error_m"] < 0.012
    assert passing.steps < resting.steps - 3


def test_a_travel_height_is_reached_before_any_level_motion(tmp_path):
    d = driver(tmp_path)
    d.execute(stage({"z": 0.5}, "grasp_approach"))
    start = len((d.output / "steps.jsonl").read_text().splitlines())
    out = d.execute(stage({"x": 0.1, "z": 0.52}, "place_approach", travel_z=0.62, via={}))
    rows = [json.loads(line) for line in (d.output / "steps.jsonl").read_text().splitlines()][start:]
    level = next(i for i, row in enumerate(rows) if row["approved"][0] > 1e-9)
    assert rows[level - 1]["state"]["eef_state"][2] == pytest.approx(0.62, abs=0.012)
    # The hand then crossed at that height and only came down over the pose.
    descent = next(i for i, row in enumerate(rows) if i > level and row["approved"][2] < 0.6)
    assert abs(rows[descent - 1]["state"]["eef_state"][0] - 0.1) <= 0.012
    assert out.ok and d.observation.state["eef_state"][2] == pytest.approx(0.52, abs=0.012)
    # A hand already above the travel height does not come down to it.
    d.execute(stage({"z": 0.66}, "move"))
    start = len((d.output / "steps.jsonl").read_text().splitlines())
    d.execute(stage({"x": 0.0, "z": 0.52}, "place_approach", travel_z=0.62, via={}))
    rows = [json.loads(line) for line in (d.output / "steps.jsonl").read_text().splitlines()][start:]
    assert rows[0]["approved"][2] == pytest.approx(0.66, abs=0.012) and rows[0]["approved"][0] < 0.1


def test_feedback_separates_the_request_from_the_command_the_approvers_applied(tmp_path):
    d = driver(tmp_path)
    first = d.execute(dict(kind="move", targets={"z": 0.5}, stage="move_to"))
    assert "applied" not in first.data and "arm" not in first.data
    # The arm still trails its last command, so the per-step limit, measured from that
    # command, holds every step of a reversed move back.
    lag = d.observation.state["eef_state"][2] - 0.5
    assert lag > 0.01
    raw = d.execute(dict(kind="move", targets={"z": 0.6}, stage="move_to"))
    applied = raw.data["applied"]
    assert raw.ok and "short of the request" in raw.message
    assert applied["reasons"] == ["delta_clamped"] and not applied["last_command_reached_request"]
    assert applied["last_command_position_offset_m"] == pytest.approx(0.6 - d.last_action.data[2], abs=1e-4)
    assert applied["last_command_position_offset_m"] > 0.01


def test_settling_judges_arrival_only_against_a_fully_applied_request(tmp_path):
    d = driver(tmp_path, arm_gain=0.2)
    d.execute(dict(kind="move", targets={"z": 0.5}, stage="move_to"))
    # Holding the measured pose is within tolerance at once, but the approvers need several
    # steps to bring the trailing command back to it, and the arm keeps sinking meanwhile.
    here = float(d.observation.state["eef_state"][2])
    assert here - d.last_action.data[2] > 0.03
    out = d.execute(stage({"z": here}, "grasp_approach"))
    assert out.ok and out.data["arm"] == "arrived"
    assert out.data["applied"]["changed_steps"] and out.data["applied"]["last_command_reached_request"]


def test_an_unrequested_gripper_entry_is_not_reported_as_a_shortfall(tmp_path):
    d = driver(tmp_path, object_mm=30)
    d.execute(stage({"gripper": 1}, "grasp_close", expect_mm=[20, 40]))
    d.execute(dict(kind="move", targets={"gripper": 0}, stage="move_to"))
    # The opening command is still ramping, so the next move's unnamed gripper entry is
    # mid-ramp and gets limited. The hand keeps opening either way.
    out = d.execute(dict(kind="move", targets={"z": 0.61}, stage="move_to"))
    assert out.data["applied"]["changed_steps"] and out.data["applied"]["last_command_reached_request"]
    assert "short of the request" not in out.message
