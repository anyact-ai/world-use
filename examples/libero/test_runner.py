"""Distinct execution, release-admission, capture, and spending regressions."""

import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from budget import BASE_URL, MODEL, AzureBudget, BudgetStop
from inspect_robots.types import Observation
from inspect_robots_agent._llm import ToolCall
from native import Frames, fingerprint
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

    def execute(self, command):
        self.commands.append(command)
        return Outcome("done", command["kind"], "fixture measured check")

    def record(self, *args):
        self.output.append(args)

    def record_observation(self):
        pass

    def measure(self):
        return {"aperture_mm": 40}


def call(name, **args):
    return ToolCall(id="test", name=name, arguments=json.dumps(dict(note="Fixture observation.", **args)))


def tools_fixture():
    # Exercise the real procedure and adapter without inventing a physical grasp.
    tools = object.__new__(WorldTools)
    tools.driver = FakeDriver()
    tools.native = tools.driver.tools
    tools.pending = tools.latest = None
    tools.tracking = SimpleNamespace(
        targets={"object": {}, "support": {}},
        lost={},
        observe=lambda names: [dict(target=n, valid=True) for n in names],
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
    release = call(
        "release_placement", checkpoint=receipt, placement_evidence="Fixture approval, not physical evidence."
    )
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
    result = tools.execute(
        call("release_placement", checkpoint=receipt, placement_evidence="Fixture."), tools.driver.observation
    )
    assert json.loads(result.note)["status"] in ("refused", "stopped")
    assert len(tools.driver.commands) == count
