"""Fresh matched LIBERO pilot; requires offline validation and explicit paid admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from importlib.metadata import distributions
from pathlib import Path

import numpy as np
from budget import BASE_URL, MODEL, RATES, AzureBudget
from inspect_robots.scene import Scene
from inspect_robots_agent._llm import Provider
from inspect_robots_agent._png import png_data_url
from inspect_robots_agent._responses import ResponsesClient
from inspect_robots_agent.policy import LLMAgentPolicy
from inspect_robots_libero.embodiment import LiberoEmbodiment
from inspect_robots_libero.tasks import ENV_SEED_BASE
from native import Driver, EpisodeEnded, fingerprint
from world_tools import DOCS, WorldTools

SOURCE = Path(__file__).resolve().parent

# The simulator publishes the front camera's render turned half a turn, which is upright but
# mirrored, while LIBERO's instructions name sides as the scene has them: in every paired left
# and right region of its task files the left one is at the smaller world y, which that image
# shows on its right. Both arms are told.
MIRRORED = (
    "The agentview image is mirrored left to right: left and right in the instruction are the "
    "reverse of left and right as that image shows them."
)


def source_hashes():
    """Every file here that decides what a run does. validation.json is the record of these, not one of them."""
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(SOURCE.iterdir())
        if p.suffix in (".py", ".txt", ".toml", ".json") and p.name != "validation.json"
    }


def runtime_manifest():
    import inspect_robots_agent._tools as native_tools
    import inspect_robots_agent.policy as native_policy
    import inspect_robots_libero.embodiment as embodiment

    import world_use
    from world_use import vision

    modules = (native_tools, native_policy, embodiment)
    package_versions = {d.metadata["Name"]: d.version for d in distributions()}
    simulator = subprocess.check_output(
        [
            os.environ["LIBERO_PYTHON"],
            "-c",
            "import json; from importlib.metadata import distributions; "
            'print(json.dumps({d.metadata["Name"]: d.version for d in distributions()}))',
        ],
        text=True,
    )
    paths = {m.__name__: Path(m.__file__) for m in modules}
    paths["native_simulator_worker"] = Path(embodiment.__file__).with_name("_worker.py")
    # The whole library, so a change to any module the tools use invalidates the validation.
    library = hashlib.sha256()
    for path in sorted(Path(world_use.__file__).parent.rglob("*.py")):
        library.update(path.relative_to(Path(world_use.__file__).parent).as_posix().encode())
        library.update(path.read_bytes())
    return dict(
        world_use_sha256=library.hexdigest(),
        agent_packages=package_versions,
        simulator_packages=json.loads(simulator),
        installed_source_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        vision_model=vision.MODEL,
        vision_revision=vision.REVISION,
        python=sys.version,
    )


def episode(folder, arm, transport, *, task_id=0, state=0, initial_expected=None, seconds=1800, held_out=False):
    if (state == 2) != held_out:
        raise ValueError(
            "state 2 is the reserved held-out state: it runs only as a held-out test, and nothing else does"
        )
    started = time.monotonic()
    env = LiberoEmbodiment(task_id=task_id, output_dir=folder / "simulator")
    policy = world = driver = None
    transport.begin(arm)
    row = dict(arm=arm, task_id=task_id, state=state, success=False, status="error")
    try:
        scene = Scene(
            id=f"task{task_id:02d}_init{state:02d}",
            instruction="loaded from native scene",
            init_seed=ENV_SEED_BASE + state,
            metadata=dict(libero_task_id=task_id, init_state=state),
        )
        obs = env.reset(scene)
        scene = replace(scene, instruction=obs.instruction)
        initial = fingerprint(obs)
        row.update(initial_observation_sha256=initial, instruction=obs.instruction)
        if initial_expected is not None and initial != initial_expected:
            raise RuntimeError("matched arms did not reset to identical images and robot state")
        if env.last["success"]:
            raise RuntimeError("initial scene already succeeds")
        driver = Driver(env, obs, folder / "execution", deadline=min(transport.deadline, time.monotonic() + seconds))
        # Both arms use the published policy, template, image eviction and native tools.
        policy = LLMAgentPolicy(
            model=MODEL,
            base_url=BASE_URL,
            api_key_env="FOUNDRY_API_KEY",
            env={"FOUNDRY_API_KEY": "transport-injects-scoped-key"},
            wire="responses",
            effort="low",
            max_llm_calls=80,
            image_horizon=2,
            prior_learnings=None,
            depth="off",
            transport=transport,
            wire_capture=False,
            transcript_echo=True,
        )
        policy._client._max_retries = 1
        info = replace(env.info, docs=((env.info.docs or "") + " " + MIRRORED).strip())
        if arm == "world_use":
            info = replace(info, docs=info.docs + "\n\n" + DOCS)
        policy.bind(info)
        if arm == "world_use":
            world = WorldTools(driver)
            policy._toolset = world
        policy.reset(scene)
        (folder / "system-prompt.txt").write_text(policy._messages[0]["content"])
        if hasattr(transport, "bind_driver"):
            transport.bind_driver(driver, world)
        while True:
            driver.check_open()
            driver.record_observation()
            chunk = policy.act(driver.observation)
            for action in chunk.actions:
                if action.meta.get("runner_no_step"):
                    if arm != "world_use":
                        raise RuntimeError("unexpected non-step marker in baseline")
                    continue
                driver.step(action, source="baseline_move")
            if driver.success:
                raise EpisodeEnded("success")
    except EpisodeEnded as exc:
        row.update(success=bool(driver and driver.success), status=str(exc))
    except Exception as exc:
        row.update(status="execution_error", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if policy:
            (folder / "transcript.json").write_text(json.dumps(policy.transcript(), indent=2))
            policy._client.close()
        if world:
            world.close()
        env.close()
        row.update(
            steps=driver.steps if driver else 0,
            calls=transport.arm_calls,
            estimated_usd=transport.arm_estimated,
            budget_charge_usd=transport.arm_used,
            wall_seconds=time.monotonic() - started,
        )
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "result.json").write_text(json.dumps(row, indent=2))
    return row


def live_probe(transport):
    transport.begin("preflight")
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:, :128] = [255, 0, 0]
    image[:, 128:] = [0, 255, 0]
    client = ResponsesClient(
        Provider(BASE_URL, "transport-injects-scoped-key", MODEL), transport=transport, max_retries=1
    )
    try:
        answer = client.complete(
            [
                dict(
                    role="user",
                    content=[
                        dict(type="text", text="Inspect the image. Call report_colors with the left and right colors."),
                        dict(type="image_url", image_url=dict(url=png_data_url(image))),
                    ],
                )
            ],
            [
                dict(
                    type="function",
                    function=dict(
                        name="report_colors",
                        description="Report the observed colors.",
                        parameters=dict(
                            type="object",
                            properties={k: dict(type="string") for k in ("left", "right")},
                            required=["left", "right"],
                        ),
                    ),
                )
            ],
            reasoning_effort="low",
        )
        if len(answer.tool_calls) != 1 or answer.tool_calls[0].name != "report_colors":
            raise RuntimeError("image/tool probe did not return the expected function")
        colors = json.loads(answer.tool_calls[0].arguments)
        if colors != {"left": "red", "right": "green"}:
            raise RuntimeError(f"image/tool probe failed: {colors}")
        return dict(passed=True, colors=colors, estimated_usd=transport.arm_estimated)
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--execute-paid-rollouts", action="store_true")
    parser.add_argument(
        "--authorization", required=True, help="the user's own words approving this exact paid scope; recorded verbatim"
    )
    parser.add_argument("--task", type=int, default=0, choices=range(10), help="LIBERO-10 task id")
    parser.add_argument("--state", type=int, default=0, help="initial state; 2 is reserved and needs --held-out")
    parser.add_argument(
        "--held-out", action="store_true", help="run reserved state 2 as a test of finished tools, never to tune them"
    )
    parser.add_argument("--total-budget", type=float, default=15.0, help="conservative USD cap for this pair")
    parser.add_argument("--arm-budget", type=float, default=7.4, help="conservative USD cap for each arm")
    parser.add_argument("--deadline-seconds", type=float, default=7200, help="wall-clock limit for this pair")
    parser.add_argument("--episode-seconds", type=float, default=1800, help="wall-clock limit for each episode")
    args = parser.parse_args()
    if (args.state == 2) != args.held_out:
        parser.error("initial state 2 is reserved held-out: pass --held-out with it and with no other state")
    if not args.execute_paid_rollouts:
        parser.error("paid calls require --execute-paid-rollouts")
    checks = json.loads(args.validation.read_text())
    if not checks.get("passed") or checks.get("source_sha256") != source_hashes():
        parser.error("offline validation must pass on these exact runner sources")
    if checks.get("runtime") != runtime_manifest():
        parser.error("installed runtime differs from the validated runtime")
    key = os.environ.get("FOUNDRY_API_KEY")
    if not key:
        parser.error("FOUNDRY_API_KEY is missing")
    if "LIBERO_AZURE_BASE_URL" not in os.environ:
        parser.error("LIBERO_AZURE_BASE_URL is missing; use the approved Azure endpoint")
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(
        runner="world-use-candidate-20261116",
        historical_comparison=False,
        model=MODEL,
        effort="low",
        max_output_tokens=3072,
        image_horizon=2,
        task_id=args.task,
        initial_state=args.state,
        held_out=args.held_out,
        added_to_both_prompts=MIRRORED,
        max_steps=900,
        max_calls=80,
        zero_shot=True,
        demonstrations=False,
        task_notes=False,
        total_budget_usd=args.total_budget,
        per_arm_budget_usd=args.arm_budget,
        deadline_seconds=args.deadline_seconds,
        episode_seconds=args.episode_seconds,
        price_assumptions=RATES,
        offline_validation=checks,
        source_sha256=source_hashes(),
        authorization=args.authorization,
    )
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    transport = AzureBudget(
        args.output / "api", key=key, total=args.total_budget, per_arm=args.arm_budget, seconds=args.deadline_seconds
    )
    report = dict(protocol=protocol, results=[])
    try:
        report["api_preflight"] = live_probe(transport)
        initial = None
        for arm in ("baseline", "world_use"):
            row = episode(
                args.output / arm,
                arm,
                transport,
                task_id=args.task,
                state=args.state,
                initial_expected=initial,
                seconds=args.episode_seconds,
                held_out=args.held_out,
            )
            initial = row["initial_observation_sha256"]
            report["results"].append(row)
            (args.output / "summary.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)
    except Exception as exc:
        report["stopped"] = dict(type=type(exc).__name__, reason=str(exc))
        raise
    finally:
        transport.dispose()
        report.update(
            total_estimated_usd=transport.estimated,
            total_budget_charge_usd=transport.used,
            usage_unknown=transport.uncertain,
            requests=transport.calls,
        )
        (args.output / "summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
