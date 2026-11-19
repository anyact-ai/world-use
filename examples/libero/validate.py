"""Validate a LIBERO runtime without paid calls and write the validation.json that run.py requires.

Runs the runner's tests, checks that the raw arm acts as upstream's does, and runs the World Use tools
on the real simulator and EdgeTAM with a scripted policy.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
from budget import BASE_URL, MODEL, AzureBudget
from inspect_robots import eval as evaluate
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots_agent.policy import LLMAgentPolicy
from inspect_robots_libero.embodiment import LiberoEmbodiment
from inspect_robots_libero.tasks import libero_10_task
from run import MIRRORED, SOURCE, episode, runtime_manifest, source_hashes
from world_tools import wrist_view

# Where two cans stand in task 0, state 0, for the synthetic policy and for scoring it; never used by paid runs.
FIXTURES = dict(soup=(-0.1002, -0.152), tomato=(-0.1173, 0.0557))


class Scripted(AzureBudget):
    def __init__(self, folder):
        self.driver = self.world = self.reference = self.script = None
        self.requests = []
        inner = httpx.MockTransport(self.respond)
        super().__init__(folder, key="offline-test-key", total=7.0, per_arm=3.4, seconds=6000, inner=inner)

    def bind_driver(self, driver, world):
        self.driver, self.world = driver, world

    def respond(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        index = self.arm_calls
        if self.world is None:
            state = self.driver.observation.state["eef_state"] if self.driver else self.reference.last["eef_state"]
            commands = [
                ("move_to", dict(targets={"z": float(state[2]) + 0.03}, note="Offline 3 cm lift.")),
                ("move_to", dict(targets={"gripper": 1.0}, note="Offline close.")),
                ("move_to", dict(targets={"gripper": 0.0}, note="Offline open.")),
                ("give_up", dict(reason="Offline execution comparison complete.", hindsight="none")),
            ]
            name, args = commands[index - 1]
        else:
            # Pixel fixtures for task 0, state 0 serve only this synthetic policy, never paid runs.
            if self.script is None:
                state = self.driver.observation.state["eef_state"]
                here = dict(zip(("x", "y", "z"), map(float, state[:3]), strict=True))
                soup, tomato = (dict(zip("xy", FIXTURES[name], strict=True)) for name in ("soup", "tomato"))

                def grasp(at, **select):
                    return (
                        "grasp",
                        dict(
                            object_target="fixture_object",
                            approach={**at, "z": 0.49},
                            lift={**at, "z": 0.53},
                            expect_grip_mm=[50, 75],
                            **select,
                        ),
                    )

                def in_wrist(at):
                    # Where the top of a can at this fixture position appears from the measured pose.
                    view = wrist_view(self.driver.observation.state["eef_state"])
                    u, v = view.project([at["x"], at["y"], 0.51])[0][0]
                    return dict(camera="wrist", point=[float(u), float(v)])

                def lean():
                    # A leaning grasp of nothing, 10 cm above where the hand is.
                    x, y, z = map(float, self.driver.observation.state["eef_state"][:3])
                    at = dict(x=x, y=y, z=z + 0.1)
                    return "grasp", dict(
                        object_target="fixture_object",
                        approach=at,
                        lift={**at, "z": at["z"] + 0.04},
                        tilt_deg=45,
                        toward=dict(x=x, y=y + 0.2),
                        expect_grip_mm=[5, 70],
                    )

                def push():
                    # A level line in free air, 10 cm above where the hand is.
                    x, y, z = map(float, self.driver.observation.state["eef_state"][:3])
                    start = dict(x=x, y=y, z=z + 0.1)
                    return "push", dict(start=start, end=dict(start, y=y + 0.1), travel_z=z + 0.1)

                aside = dict(x=tomato["x"] + 0.04, y=tomato["y"] - 0.03)
                tomato_box = dict(camera="agentview", box=[95, 120, 125, 161])
                self.script = [
                    ("select_target", dict(target="fixture_object", **tomato_box)),
                    ("select_target", dict(target="fixture_support", camera="agentview", box=[0, 110, 58, 193])),
                    ("observe_targets", {}),
                    # Closing in free air: nothing between the fingers.
                    (
                        "grasp",
                        dict(
                            object_target="fixture_object",
                            approach=here,
                            lift={**here, "z": here["z"] + 0.04},
                            expect_grip_mm=[5, 70],
                        ),
                    ),
                    ("move_to", dict(targets={"gripper": 0.0})),
                    # A reversal issued while the arm still trails its last command.
                    ("move_to", dict(targets={"z": 0.55})),
                    ("move_to", dict(targets={"z": 0.62})),
                    # A hover 5 cm off the soup; the wrist views place it and bring the hand over it.
                    ("move_to", dict(targets=dict(x=soup["x"] + 0.04, y=soup["y"] + 0.03, z=0.64))),
                    lambda: ("locate", dict(target="fixture_located", select=in_wrist(soup))),
                    # The selected region is the tomato sauce; this grasp lifts the soup instead.
                    grasp(soup),
                    ("move_to", dict(targets={"gripper": 0.0})),
                    ("move_to", dict(targets={"z": 0.62})),
                    ("move_to", dict(targets={**tomato, "z": 0.62})),
                    # Reselecting in the grasp call itself must bind the same region.
                    grasp(tomato, select=tomato_box),
                    # Set it down, hover 5 cm off it and take it again from a wrist selection alone.
                    ("move_to", dict(targets={"gripper": 0.0})),
                    ("move_to", dict(targets={**aside, "z": 0.64})),
                    lambda: (
                        "grasp",
                        dict(
                            object_target="fixture_wrist",
                            select=in_wrist(tomato),
                            approach={**aside, "z": 0.49},
                            lift={**aside, "z": 0.53},
                            expect_grip_mm=[50, 75],
                        ),
                    ),
                    ("move_to", dict(targets={"gripper": 0.0})),
                    ("move_to", dict(targets={"z": 0.62})),
                    lean,
                    ("move_to", dict(targets={"gripper": 0.0})),
                    push,
                    ("give_up", dict(reason="Offline procedure checks complete.", hindsight="none")),
                ]
            entry = self.script[index - 1]
            name, args = entry() if callable(entry) else entry
            args = dict(args)
            args["note"] = "Synthetic offline fixture; no task-completion claim."
        return httpx.Response(
            200,
            json=dict(
                id=f"mock-{self.calls}",
                status="completed",
                output=[
                    dict(
                        type="function_call",
                        id=f"fc_{self.calls}",
                        call_id=f"call_{self.calls}",
                        name=name,
                        arguments=json.dumps(args),
                    )
                ],
                usage=dict(input_tokens=100, output_tokens=100),
            ),
        )


class RecordedEmbodiment(LiberoEmbodiment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.executed = []

    def step(self, action):
        result = super().step(action)
        self.executed.append(
            dict(
                approved=np.asarray(action.data).tolist(),
                state={k: v.tolist() for k, v in result.observation.state.items()},
            )
        )
        return result


def reference(folder, transport):
    transport.begin("reference")
    env = RecordedEmbodiment(task_id=0, output_dir=folder / "simulator")
    transport.reference = env
    transport.driver = transport.world = None
    policy = LLMAgentPolicy(
        model=MODEL,
        base_url=BASE_URL,
        api_key_env="FOUNDRY_API_KEY",
        env={"FOUNDRY_API_KEY": "offline"},
        wire="responses",
        effort="low",
        max_llm_calls=80,
        image_horizon=2,
        depth="off",
        transport=transport,
        wire_capture=False,
    )
    policy._client._max_retries = 1
    try:
        (log,) = evaluate(
            libero_10_task(0),
            policy,
            env,
            log_dir=str(folder / "inspect"),
            seed=42,
            store_frames=True,
            store_actions=True,
            grader=None,
            approver=ChainApprover(ClampApprover(env.info.action_space), DeltaLimitApprover(env.info.action_space)),
        )
        assert log.status == "success"
        assert transport.arm_calls == 4
        (folder / "system-prompt.txt").write_text(policy._messages[0]["content"])
        return env.executed
    finally:
        policy._client.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        passed=False,
        paid_calls=0,
        source_sha256=source_hashes(),
        real_simulator=True,
        real_edgetam=True,
        synthetic_policy=True,
        runtime=runtime_manifest(),
    )
    transport = Scripted(args.output / "mock-api")
    try:
        with (args.output / "tests.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    str(SOURCE / "test_campaign.py"),
                    "--basetemp",
                    str(args.output / "pytest"),
                ],
                stdout=log,
                stderr=log,
                check=True,
            )
        report["tests_passed"] = True
        original = reference(args.output / "reference", transport)
        (args.output / "reference-actions.json").write_text(json.dumps(original))
        rebuilt = episode(args.output / "baseline", "baseline", transport)
        steps = [json.loads(line) for line in (args.output / "baseline/execution/steps.jsonl").read_text().splitlines()]
        assert len(steps) == len(original)
        for a, b in zip(original, steps, strict=True):
            np.testing.assert_array_equal(a["approved"], b["approved"])
            for key in a["state"]:
                np.testing.assert_allclose(a["state"][key], b["state"][key], atol=1e-12, rtol=0)
        report["native_parity"] = dict(actions_identical=True, states_match_atol=1e-12, steps=len(steps))
        # What the raw arm is told differs from upstream by one sentence, which both arms get once.
        upstream = (args.output / "reference/system-prompt.txt").read_text()
        told = (args.output / "baseline/system-prompt.txt").read_text()
        assert told != upstream and told.replace(" " + MIRRORED, "") == upstream
        report["raw_prompt"] = dict(upstream_plus=MIRRORED)
        augmented = episode(
            args.output / "world_use", "world_use", transport, initial_expected=rebuilt["initial_observation_sha256"]
        )
        assert augmented["calls"] == len(transport.script)
        assert (args.output / "world_use/system-prompt.txt").read_text().count(MIRRORED) == 1
        assert augmented["status"] == "give_up" and not augmented["success"]
        outcomes = [
            json.loads(line) for line in (args.output / "world_use/execution/world-use.jsonl").read_text().splitlines()
        ]
        assert all(o["valid"] for o in outcomes[:2])
        empty, wrong, held, centred, leant = (o for o in outcomes if o.get("procedure") == "grasp")
        # An empty hand is reported at rest, fully closed, before the settling bound and before any lift.
        closed = empty["last_command"]["outcome"]["data"]
        assert empty["status"] == "surprise" and empty["stages"][-1]["stage"] == "close"
        assert closed["gripper"] == "reached_end" and closed["executed_steps"] < 31
        assert empty["observations"]["after"]["aperture_mm"] < 5
        actual_steps = [
            json.loads(line) for line in (args.output / "world_use/execution/steps.jsonl").read_text().splitlines()
        ]
        assert len(actual_steps) == augmented["steps"]
        before_reselection = actual_steps[: empty["observations"]["after"]["env_step"]]
        assert not any(s["source"] == "grasp_test_lift" for s in before_reselection)
        # The reversed move's last applied command differs from the request, and the feedback says so.
        (reversal,) = (
            o["data"]
            for o in outcomes
            if not o.get("data", {}).get("applied", {}).get("last_command_reached_request", True)
        )
        last = actual_steps[reversal["env_step"] - 1]
        offset = float(np.linalg.norm(np.subtract(last["requested"][:3], last["approved"][:3])))
        assert offset > 0.005 and abs(reversal["applied"]["last_command_position_offset_m"] - offset) < 1e-3
        # The same selected region across two real lifts: left behind, then carried by the hand.
        for grasp in (wrong, held):
            assert [s["status"] for s in grasp["stages"]] == ["done"] * 5 and grasp["checkpoint"] == "after"
        left = wrong["assessments"][-1]["outcome"]
        assert wrong["status"] == "surprise" and left["kind"] == "lift_evidence" and left["data"]["unchanged"]
        # A region that moved proves nothing, so the held grasp reports as an unassessed one does.
        carried = next(t for t in held["observations"]["after"]["targets"] if t["target"] == "fixture_object")["motion"]
        assert held["status"] == "done" and held["assessment"] == "unknown" and not held["completion_verified"]
        assert carried["mask_iou"] < 0.9 and all(a["outcome"]["kind"] == "not_assessed" for a in held["assessments"])
        # From hovers 5 cm off, the wrist views put both cans within the fingers' reach of their true places.
        (located,) = (o["located"] for o in outcomes if "located" in o)
        found = dict(soup=located, tomato=centred["centred"])
        placed = {
            name: float(np.hypot(found[name]["x"] - x, found[name]["y"] - y)) for name, (x, y) in FIXTURES.items()
        }
        assert max(placed.values()) < 0.015 and located["looks"] >= 2
        assert centred["status"] == "done" and [s["status"] for s in centred["stages"]] == ["done"] * 5
        assert 50 <= centred["observations"]["after"]["aperture_mm"] <= 75
        # A leaning grasp turns the hand where it is and comes in along its own axis to the point asked for.
        assert leant["status"] == "surprise" and [s["status"] for s in leant["stages"]] == ["done", "surprise"]
        assert leant["leaning"] == dict(tilt_deg=45, toward=[0.0, 1.0])
        came_in = [s for s in actual_steps if s["source"] == "grasp_approach"][-1]
        ended = np.array(leant["observations"]["after"]["eef_state"])
        assert np.linalg.norm(ended[:3] - came_in["requested"][:3]) < 0.012
        assert np.cross(ended[3:6], ended[6:9]) == pytest.approx([0, 0.5**0.5, -(0.5**0.5)], abs=0.1)
        assert any(s["source"] == "grasp_lean" for s in actual_steps)
        # A push with nothing in its way runs its whole line, leaning, and comes back along it upright.
        (pushed,) = (o for o in outcomes if "pushed" in o)
        assert pushed["status"] == "done"
        assert pushed["pushed"] == dict(ended="reached_end", travelled_mm=100, short_mm=0)
        line = [s for s in actual_steps if s["source"] == "push"]
        assert pushed["eef_state"][1] == pytest.approx(line[-1]["requested"][1] - 0.1, abs=0.012)
        assert abs(line[-1]["requested"][8]) == pytest.approx(0.5, abs=1e-6) and abs(pushed["eef_state"][8]) < 0.12
        # Selection and repeated observation share the actual simulator capture.
        assert outcomes[0]["frame"] == outcomes[1]["frame"]
        observed = outcomes[2]["targets"]
        for selected, current in zip(outcomes[:2], observed, strict=True):
            assert selected["frame"] == current["frame"] and selected["bbox"] == current["bbox"]
        report.update(
            passed=True,
            baseline=rebuilt,
            world_use=augmented,
            empty_grasp_stopped_before_lift=True,
            applied_command_offset_reported_m=offset,
            lift_evidence=dict(left_behind=left["data"], carried_mask_iou=carried["mask_iou"]),
            wrist_placement_error_m={k: round(v, 4) for k, v in placed.items()},
            initial_observations_identical=True,
            repeated_capture_regions_identical=True,
        )
    finally:
        transport.dispose()
        (args.output / "validation.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
