"""No paid calls: compare native execution and exercise real RGB tracking/procedures."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import httpx
import numpy as np
from budget import BASE_URL, MODEL, AzureBudget
from inspect_robots import eval as evaluate
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots_agent.policy import LLMAgentPolicy
from inspect_robots_libero.embodiment import LiberoEmbodiment
from inspect_robots_libero.tasks import libero_10_task
from run import SOURCE, episode, runtime_manifest, source_hashes


class Scripted(AzureBudget):
    def __init__(self, folder):
        self.driver = self.world = self.reference = None
        self.requests = []
        super().__init__(folder, key="offline-test-key", inner=httpx.MockTransport(self.respond))

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
            # Pixel fixtures are used only by this synthetic policy, never by paid runs.
            captures = self.world.context()["captures"]
            frame = next(c["frame"] for c in captures if c["camera"] == "agentview")
            state = self.driver.observation.state["eef_state"]
            approach = dict(zip(("x", "y", "z"), map(float, state[:3]), strict=True))
            lift = {**approach, "z": approach["z"] + 0.04}
            commands = [
                ("select_target", dict(target="fixture_object", frame=frame, box=[95, 120, 125, 161])),
                ("select_target", dict(target="fixture_support", frame=frame, box=[0, 110, 58, 193])),
                ("observe_targets", {}),
                ("grasp", dict(object_target="fixture_object", approach=approach, lift=lift, expect_grip_mm=[5, 70])),
                ("move_to", dict(targets={"gripper": 0.0})),
                ("give_up", dict(reason="Offline empty-grasp check complete.", hindsight="none")),
            ]
            name, args = commands[index - 1]
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
                    str(SOURCE / "test_runner.py"),
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
        augmented = episode(
            args.output / "world_use", "world_use", transport, initial_expected=rebuilt["initial_observation_sha256"]
        )
        assert augmented["calls"] == 6
        assert augmented["status"] == "give_up" and not augmented["success"]
        outcomes = [
            json.loads(line) for line in (args.output / "world_use/execution/world-use.jsonl").read_text().splitlines()
        ]
        assert all(o["valid"] for o in outcomes[:2])
        grasp = next(o for o in outcomes if o.get("procedure") == "grasp")
        assert grasp["status"] in ("stopped", "surprise")
        assert grasp["stages"][-1]["stage"] == "close"
        assert grasp["observations"]["after"]["aperture_mm"] < 5
        assert "test_lift" not in [s["stage"] for s in grasp["stages"]]
        actual_steps = [
            json.loads(line) for line in (args.output / "world_use/execution/steps.jsonl").read_text().splitlines()
        ]
        assert len(actual_steps) == augmented["steps"]
        assert not any(s["source"] == "grasp_test_lift" for s in actual_steps)
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
            initial_observations_identical=True,
            repeated_capture_regions_identical=True,
        )
    finally:
        transport.dispose()
        (args.output / "validation.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
