"""Run matched LIBERO pairs under one campaign cap: one admission authority, a separate run.py output per pair.

A pair is admitted only while the conservative charge of finished attempts plus the full caps of running
ones leaves room for another full pair cap. Any attempt with unknown usage stops further admission. An
interrupted attempt keeps its folder and its charge; its retry goes to a new `_retryN` folder. Each repeat
of a case is its own pair, in a `_repeatN` folder after the first.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from run import ARMS, ROLES

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[1]


def charge(folder: Path) -> tuple[float, bool]:
    """Conservative charge and unknown-usage flag of one attempt, from its own durable ledger."""
    ledger = folder / "api" / "ledger.jsonl"
    if not ledger.exists():
        return 0.0, False
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    # A reservation with no settlement after it is retained in full by total_charge.
    total = max((r.get("total_charge", 0.0) for r in rows), default=0.0)
    unknown = any(r["event"] == "usage_unknown" or r.get("unknown_usage") for r in rows)
    reserved = {r["request"] for r in rows if r["event"] == "reserved"}
    settled = {r["request"] for r in rows if r["event"] == "settled"}
    return total, unknown or bool(reserved - settled)


def complete(folder: Path) -> bool:
    summary = folder / "summary.json"
    if not summary.exists():
        return False
    report = json.loads(summary.read_text())
    return len(report.get("results", [])) == len(report["protocol"].get("arms", ARMS["both"]))


class Campaign:
    def __init__(self, folder, *, cap, pair_cap, run_args, call=subprocess.call):
        self.folder, self.cap, self.pair_cap = folder, cap, pair_cap
        self.run_args, self.call = run_args, call
        self.lock = threading.Lock()
        self.running: dict[str, float] = {}

    def spent(self) -> tuple[float, bool]:
        total, unknown = 0.0, False
        for attempt in sorted(self.folder.glob("task*_init*")):
            if attempt.is_dir() and attempt.name not in self.running:
                c, u = charge(attempt)
                total, unknown = total + c, unknown or u
        return total, unknown

    def admit(self, name: str) -> bool:
        with self.lock:
            total, unknown = self.spent()
            running = sum(self.running.values())
            if unknown or total + running + self.pair_cap > self.cap:
                print(
                    f"NOT ADMITTED {name}: finished {total:.2f} + running {running:.2f} + pair {self.pair_cap:.2f} "
                    f"> cap {self.cap:.2f} or unknown usage={unknown}",
                    flush=True,
                )
                return False
            self.running[name] = self.pair_cap
            return True

    def run(self, task: int, state: int, repeat: int) -> None:
        base = f"task{task:02d}_init{state:02d}" + (f"_repeat{repeat}" if repeat > 1 else "")
        retry = 0
        while (self.folder / (name := base + (f"_retry{retry}" if retry else ""))).exists():
            if complete(self.folder / name):
                print(f"SKIP {base}: already complete", flush=True)
                return
            retry += 1
        if not self.admit(name):
            return
        started = time.monotonic()
        try:
            with (self.folder / f"{name}.stdout").open("w") as out, (self.folder / f"{name}.stderr").open("w") as err:
                code = self.call(self.run_args(task, state, self.folder / name), cwd=ROOT, stdout=out, stderr=err)
        finally:
            with self.lock:
                self.running.pop(name, None)
        c, u = charge(self.folder / name)
        wall = time.monotonic() - started
        print(f"FINISHED {name} exit={code} charge={c:.3f} unknown={u} wall={wall:.0f}s", flush=True)

    def run_all(self, cases, workers: int, stagger: float = 20) -> tuple[float, bool]:
        self.folder.mkdir(parents=True, exist_ok=True)
        queue = list(cases)

        def worker():
            while True:
                with self.lock:
                    if not queue:
                        return
                    case = queue.pop(0)
                self.run(*case)

        threads = [threading.Thread(target=worker) for _ in range(workers)]
        for thread in threads:
            thread.start()
            time.sleep(stagger)  # stagger simulator and tracker start-up
        for thread in threads:
            thread.join()
        total, unknown = self.spent()
        print(f"CAMPAIGN conservative charge {total:.3f} of cap {self.cap:.2f}; unknown usage: {unknown}", flush=True)
        return total, unknown


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="campaign folder; finished pairs in it are skipped and charged")
    parser.add_argument("cases", nargs="*", metavar="TASK:STATE")
    parser.add_argument("--role", choices=("development", "held_out"), help="select every case of this role")
    parser.add_argument("--cap", type=float, required=True, help="conservative USD cap for the whole campaign")
    parser.add_argument("--pair-cap", type=float, required=True, help="conservative USD cap for each pair")
    parser.add_argument("--arm-cap", type=float, required=True, help="conservative USD cap for each arm")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1, help="pairs per case")
    parser.add_argument("--arms", choices=ARMS, default="both")
    parser.add_argument("--validation", type=Path, default=SOURCE / "validation.json")
    parser.add_argument("--authorization", required=True, help="the user's words approving this paid scope")
    parser.add_argument("--held-out-authorization", help="the user's words approving these held-out tests")
    args = parser.parse_args()
    if bool(args.role) == bool(args.cases):
        parser.error("give TASK:STATE cases or --role, not both")
    if args.role:
        cases = [case for case, role in ROLES.items() if role == args.role]
    else:
        cases = [tuple(map(int, item.split(":"))) for item in args.cases]
    if unknown := [case for case in cases if case not in ROLES]:
        parser.error(f"not cases in cases.toml: {unknown}")
    held_out = {case for case in cases if ROLES[case] == "held_out"}
    if held_out and not args.held_out_authorization:
        parser.error("held-out cases need --held-out-authorization")

    def run_args(task, state, output):
        test = (task, state) in held_out
        return [
            sys.executable, str(SOURCE / "run.py"), "--validation", str(args.validation.resolve()),
            "--output", str(output.resolve()), "--task", str(task), "--state", str(state), "--arms", args.arms,
            "--authorization", args.held_out_authorization if test else args.authorization,
            *(["--held-out"] if test else []), "--total-budget", str(args.pair_cap), "--arm-budget", str(args.arm_cap),
            "--episode-seconds", "3000", "--deadline-seconds", "6000", "--execute-paid-rollouts",
        ]  # fmt: skip

    os.environ["PYTHONPATH"] = str(ROOT / "src")
    campaign = Campaign(args.folder, cap=args.cap, pair_cap=args.pair_cap, run_args=run_args)
    campaign.run_all([(*case, repeat) for repeat in range(1, args.repeats + 1) for case in cases], args.workers)


if __name__ == "__main__":
    main()
