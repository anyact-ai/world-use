"""Run one arm of the pilot on a split of cases.toml, through robowild-harness eval.

Both arms use the same model, the react_move_points Robot API profile, frozen simulation, seed 0, one round and
each task's own call budget. "alone" is the harness's react backend; "world-use" is agent.py, the same loop with the
hand tools added. Run it in RoboWild's environment (source activate_robowild.sh) on a Linux machine with an NVIDIA
RTX-class GPU:

    python run.py --arm world-use --split development --model MODEL --credentials creds.json --budget-usd 50 \\
        --agent-python /path/to/agent-venv/bin/python --output results/development \\
        --authorization "the approval of this paid run, verbatim"

creds.json is {"base_url": the model's Responses base URL, "key": its API key}; it is deleted once read, and the key
then lives only in model_proxy.py, which the agents call with a placeholder key and which stops the run's spending
at --budget-usd. Tasks run one harness evaluation each, in the order given, and the next starts only while the
budget left covers the costliest so far. Model calls cost money: such a run refuses to start without
--authorization, which it records with the run. With --scripted PLAN.json instead of a model, scripted_model.py
plays the plan's tool calls, at no cost, to check a setup.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ("alone", "world-use")


def agent(args) -> dict:
    common = dict(model=args.model, base_url=args.base_url, api_key_env=args.api_key_env)
    if args.reasoning_effort:
        common["reasoning_effort"] = args.reasoning_effort
    if args.arm == "alone":
        return dict(backend="react", **common)
    command = [str(args.agent_python), str(HERE / "agent.py"), "--model", args.model, "--base-url", args.base_url,
               "--api-key-env", args.api_key_env]
    if args.reasoning_effort:
        command += ["--reasoning-effort", args.reasoning_effort]
    return dict(backend="command", command=command)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_for(port: int, process: subprocess.Popen):
    for _ in range(100):
        if process.poll() is not None:
            sys.exit("the local model endpoint did not start")
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    sys.exit("the local model endpoint did not start")


def spent(ledger: Path) -> float:
    """What the arm's model calls have cost so far, by model_proxy.py's ledger."""
    lines = ledger.read_text().splitlines() if ledger.exists() else []
    return json.loads(lines[-1])["spent_usd"] if lines else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--split", choices=("development", "held_out"), required=True)
    parser.add_argument("--tasks", help="a comma-separated subset of the split")
    parser.add_argument("--model")
    parser.add_argument("--credentials", type=Path, help="JSON {base_url, key}; deleted once read")
    parser.add_argument("--budget-usd", type=float, help="the most this arm's model calls may cost")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--scripted", type=Path, help="a plan of tool calls to play instead of a model")
    parser.add_argument("--authorization", help="the approval of a paid run in the approver's own words")
    parser.add_argument("--agent-python", type=Path, help="the world-use arm's interpreter (see README.md)")
    parser.add_argument("--worker-python", type=Path,
                        help="the harness's agent-side interpreter, when the simulator's cannot be copied")
    parser.add_argument("--robowild", type=Path, default=Path(os.environ.get("ROBOWILD_CHECKOUT", "RoboWild")))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent-timeout", type=float, default=3600)
    args = parser.parse_args()
    if args.arm == "world-use" and args.agent_python is None:
        parser.error("--agent-python is required for the world-use arm")
    cases = tomllib.loads((HERE / "cases.toml").read_text())
    if not args.scripted and not (args.model and args.credentials and args.budget_usd and args.authorization):
        parser.error("a model run needs --model, --credentials, --budget-usd and --authorization")
    if not args.scripted and args.model not in cases["rates"]:
        parser.error(f"cases.toml has no rates for {args.model}")
    tasks = cases["tasks"][args.split]
    if args.tasks:
        chosen = args.tasks.split(",")
        if not set(chosen) <= set(tasks):
            parser.error(f"--tasks must come from the {args.split} split")
        tasks = chosen
    out = args.output / args.arm
    out.mkdir(parents=True, exist_ok=False)
    port = free_port()
    if args.scripted:
        server = subprocess.Popen([sys.executable, str(HERE / "scripted_model.py"), str(port), str(args.scripted)])
        args.model, args.authorization = "scripted", "a scripted check: no model is called"
    else:
        credentials = json.loads(args.credentials.read_text())
        args.credentials.unlink()
        server = subprocess.Popen([sys.executable, str(HERE / "model_proxy.py"), str(port), str(out / "ledger.jsonl"),
                                   str(args.budget_usd), json.dumps(cases["rates"][args.model])],
                                  env=dict(os.environ, MODEL_UPSTREAM=credentials["base_url"],
                                           MODEL_KEY=credentials["key"]))
        del credentials
    wait_for(port, server)
    os.environ["MODEL_PLACEHOLDER_KEY"] = "local"
    args.base_url, args.api_key_env = f"http://127.0.0.1:{port}/v1", "MODEL_PLACEHOLDER_KEY"
    agent_file = out / "agent.json"
    agent_file.write_text(json.dumps(agent(args), indent=2) + "\n")
    run = dict(
        authorization=args.authorization, arm=args.arm, split=args.split, tasks=tasks, model=args.model,
        reasoning_effort=args.reasoning_effort, budget_usd=args.budget_usd, robowild_commit=cases["robowild_commit"],
        harness_commit=cases["harness_commit"],
        world_use_commit=os.environ.get("WORLD_USE_COMMIT") or subprocess.run(
            ["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
    )
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    ledger = out / "ledger.jsonl"
    costliest = 0.0
    try:
        for index, task in enumerate(tasks):
            before = spent(ledger)
            # The proxy's cap ends an attempt that runs over, which wastes what it spent: a task starts only while
            # the budget left covers the costliest task so far.
            if not args.scripted and args.budget_usd - before < costliest:
                (out / "run.json").write_text(json.dumps(dict(run, not_run=tasks[index:]), indent=2) + "\n")
                sys.exit(f"stopped before {task} with ${args.budget_usd - before:.2f} of the budget left; the "
                         f"costliest task took ${costliest:.2f}. Not run: {','.join(tasks[index:])}")
            budget = "long" if task in cases["tasks"]["long"] else "short"
            options = dict(robot_type=cases["robot_type"], control_mode="joint_position",
                           assets_dir=os.environ["ASSETS_DIR"], curobo_source=os.environ["CUROBO_SOURCE"],
                           call_limit=cases["call_limit"][budget])
            command = ["robowild-harness", "eval", "--environment", "robowild",
                       "--environment-checkout", str(args.robowild), "--tasks", task,
                       "--agent", str(agent_file), "--output", str(out / task),
                       "--assignment-options", json.dumps(options), "--tool-profile", "react_move_points",
                       "--mode", "independent", "--seed", "0", "--rounds", "1", "--gpus", "1", "--envs-per-gpu", "1",
                       "--agent-timeout", str(args.agent_timeout)]
            if args.worker_python:
                command += ["--worker-python", str(args.worker_python)]
            print("+", " ".join(command), flush=True)
            try:
                failed = subprocess.run(command).returncode
            finally:
                # The harness copies its agent-side environment into each task's workspace: some 3,700 files of no
                # use once the task has run, which filled a 500,000-file volume within 40 tasks.
                for workspace in (out / task).glob("runs/*/*/*/agent-workspaces/*"):
                    for copy in (".venv", "dependencies"):
                        shutil.rmtree(workspace / copy, ignore_errors=True)
            if failed:
                sys.exit(f"{task}'s evaluation failed; resume it with robowild-harness eval --resume")
            costliest = max(costliest, spent(ledger) - before)
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
