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
from world_tools import WorldTools

from world_use.behaviors import Outcome


def observation():
    return Observation(
        images={"agentview": np.zeros((256, 256, 3), dtype=np.uint8)},
        state={"eef_state": np.zeros(10), "gripper_qpos": np.array([0.02, -0.02])},
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

    budget = AzureBudget(tmp_path / "api", key="test", inner=httpx.MockTransport(handle))
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
    budget = AzureBudget(tmp_path / "api", key="test", total=0.001, inner=httpx.MockTransport(lambda r: sent.append(r)))
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

    budget = AzureBudget(tmp_path / "api", key="test-private-key", inner=httpx.MockTransport(handle))
    budget.begin("baseline")
    budget.handle_request(request())
    assert sent[0].headers["api-key"] == "test-private-key"
    assert "authorization" not in sent[0].headers
    assert budget.used == pytest.approx(0.00035)
    assert budget.estimated == pytest.approx(0.0003)
    assert not budget.uncertain
    for p in (tmp_path / "api").iterdir():
        assert "test-private-key" not in p.read_text()


class FakeDriver:
    def __init__(self):
        self.observation = observation()
        self.frames = Frames()
        self.frames.sync(self.observation)
        self.tools = SimpleNamespace(execute=lambda *a: SimpleNamespace(error=None))
        self.commands = []
        self.output = []
        self.pose = dict(x=0.0, y=0.0, z=0.5)

    def execute(self, command):
        self.commands.append(command)
        self.pose.update({k: v for k, v in command.get("targets", {}).items() if k in self.pose})
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
    tools.pending = tools.latest = None
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


@pytest.mark.parametrize("change", ["pixels", "state", "motion", "loss"])
def test_release_refuses_changed_evidence_without_opening(change):
    tools = tools_fixture()
    receipt = pause(tools)
    if change == "pixels":
        tools.driver.observation.images["agentview"][0, 0, 0] = 5
    elif change == "state":
        tools.driver.observation.state["eef_state"][0] = 0.02
    elif change == "motion":
        tools.execute(call("move_to", targets={"z": 0.7}), tools.driver.observation)
    else:
        tools.tracking.observe = lambda names: [dict(target=n, valid=n != "object") for n in names]
    count = len(tools.driver.commands)
    result = tools.execute(call("release_placement", checkpoint=receipt), tools.driver.observation)
    assert json.loads(result.note)["status"] in ("refused", "stopped")
    assert len(tools.driver.commands) == count


STAYED = dict(mask_iou=0.97, center_delta_px=[0.3, -0.5])
MOVED = dict(mask_iou=0.5, center_delta_px=[1.0, -9.0])


def grasp(tools, **extra):
    pose = dict(x=0.0, y=0.0, z=0.5)
    request = call(
        "grasp", object_target="object", approach=pose, lift={**pose, "z": 0.54}, expect_grip_mm=[20, 60], **extra
    )
    return json.loads(tools.execute(request, tools.driver.observation).note)


@pytest.mark.parametrize(
    ("camera", "motion", "status", "assessment"),
    [
        ("agentview", STAYED, "surprise", "unknown"),
        ("agentview", MOVED, "done", "passed_caller_checks"),
        # The wrist camera rises with the hand, so a held object's region stays put there.
        ("wrist", STAYED, "done", "unknown"),
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


def test_a_lift_too_small_to_compare_is_not_judged(monkeypatch):
    tools = tools_fixture("agentview", **STAYED)
    execute = tools.driver.execute
    monkeypatch.setattr(tools.driver, "execute", lambda c: execute({**c, "targets": {}}))
    result = grasp(tools)
    assert result["status"] == "stopped" and "could not be compared" in result["message"]


def test_a_grasp_that_stops_before_its_lift_reports_no_lift_evidence(monkeypatch):
    tools = tools_fixture("agentview", **MOVED)
    execute = tools.driver.execute
    refuse = Outcome("surprise", "move", "fixture: nothing between the fingers")
    monkeypatch.setattr(tools.driver, "execute", lambda c: refuse if c.get("stage") == "grasp_close" else execute(c))
    result = grasp(tools)
    assert result["status"] == "surprise" and list(result["stages"])[-1] == "close"
    assert "assessments" not in result


def test_the_policy_gets_a_brief_copy_and_the_record_keeps_everything():
    tools = tools_fixture("agentview", **MOVED)
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


class Hand(LiberoEmbodiment):
    """The published hand and arm timing without a simulator.

    The hand's command state moves 0.125 per step toward whichever end the request is
    past, and the fingers follow it until an object stops them. The arm trails its
    command, so a move issued straight after another starts from a pose the last
    command has already left behind.
    """

    def __init__(self, *, object_mm=None, finger_gain=0.5, floor_z=None, arm_gain=0.3, ramp=0.125):
        super().__init__(task_id=0, output_dir="unused")
        self.pos, self.command, self.half_mm = np.array([0.0, 0.0, 0.6]), 0.0, 40.0
        self.object_mm, self.finger_gain, self.floor_z = object_mm, finger_gain, floor_z
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
        self.pos = self.pos + self.arm_gain * (data[:3] - self.pos)
        if self.floor_z is not None:
            self.pos[2] = max(self.pos[2], self.floor_z)
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
    assert out.data["aperture_mm"] == pytest.approx(object_mm or 0, abs=2)
    assert abs(out.data["aperture_change_mm"]) < 0.5
    # A hand at rest ends the wait; only motion still under way may use the whole bound.
    assert out.data["executed_steps"] < 31
    if status == "surprise":
        assert out.expected == [20, 40] and out.observed == out.data["aperture_mm"]


def test_a_hand_still_moving_at_the_bound_is_not_reported_as_stable(tmp_path):
    d = driver(tmp_path, finger_gain=0.01)
    out = d.execute(stage({"gripper": 1}, "grasp_close", expect_mm=[20, 78]))
    assert out.status == "stopped" and out.data["gripper"] == "still_moving"
    assert out.data["aperture_change_mm"] < 0 and out.data["executed_steps"] == 31


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
