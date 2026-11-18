"""Run matched LIBERO pairs under one campaign cap: one admission authority, a separate run.py output per pair.

A pair is admitted only while the conservative charge of finished attempts plus the full caps of running
ones leaves room for another full pair cap. An attempt that did not finish both arms, stopped by the
transport, a lost simulator or an interruption, keeps its folder and its charge, an unsettled request's
whole reservation included, and the pair is tried again in a `_retryN` folder, three attempts in all.
Each repeat of a case is its own pair, in a `_repeatN` folder after the first.
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

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[1]
sys.path.insert(0, str(ROOT / "src"))  # the library of this tree, as the pairs it starts import it

from run import ARMS, ROLES  # noqa: E402

ATTEMPTS = 3


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
    return len(report.get("results", [])) == len(report["protocol"]["arms"])


class Campaign:
    def __init__(self, folder, *, cap, pair_cap, run_args, call=subprocess.call):
        self.folder, self.cap, self.pair_cap = folder, cap, pair_cap
        self.run_args, self.call = run_args, call
        self.lock = threading.Lock()
        self.running: dict[str, float] = {}
        self.stopping = threading.Event()

    def spent(self) -> float:
        attempts = (a for a in self.folder.glob("task*_init*") if a.is_dir() and a.name not in self.running)
        return sum(charge(attempt)[0] for attempt in attempts)

    def admit(self, name: str) -> bool:
        with self.lock:
            if self.stopping.is_set():
                return False
            if any('"over_reservation"' in ledger.read_text() for ledger in self.folder.glob("*/api/ledger.jsonl")):
                print(f"NOT ADMITTED {name}: a request used more than its reservation", flush=True)
                return False
            total, running = self.spent(), sum(self.running.values())
            if total + running + self.pair_cap > self.cap:
                print(
                    f"NOT ADMITTED {name}: finished {total:.2f} + running {running:.2f} + pair {self.pair_cap:.2f} "
                    f"> cap {self.cap:.2f}",
                    flush=True,
                )
                return False
            self.running[name] = self.pair_cap
            return True

    def run(self, task: int, state: int, repeat: int) -> None:
        base = f"task{task:02d}_init{state:02d}" + (f"_repeat{repeat}" if repeat > 1 else "")
        for attempt in range(ATTEMPTS):
            name = base + (f"_retry{attempt}" if attempt else "")
            folder = self.folder / name
            if not folder.exists():
                if not self.admit(name):
                    return
                started = time.monotonic()
                logs = [(self.folder / f"{name}.{stream}").open("w") for stream in ("stdout", "stderr")]
                try:
                    code = self.call(self.run_args(task, state, folder), cwd=ROOT, stdout=logs[0], stderr=logs[1])
                finally:
                    for log in logs:
                        log.close()
                    with self.lock:
                        self.running.pop(name, None)
                c, u = charge(folder)
                wall = time.monotonic() - started
                print(f"FINISHED {name} exit={code} charge={c:.3f} unknown={u} wall={wall:.0f}s", flush=True)
                if code < 0 or code == 130:
                    # A pair stopped by a signal was stopped on purpose: no retry, and no more pairs.
                    self.stopping.set()
            if complete(folder) or self.stopping.is_set():
                return
        print(f"UNFINISHED {base}: {ATTEMPTS} attempts", flush=True)

    def run_all(self, cases, workers: int, stagger: float = 20) -> float:
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
        try:
            for thread in threads:
                thread.start()
                time.sleep(stagger)  # stagger simulator and tracker start-up
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            # The running pairs got the interrupt too. Start nothing more, and wait for them to stop.
            self.stopping.set()
            for thread in threads:
                if thread.is_alive():
                    thread.join()
            raise
        total = self.spent()
        print(f"CAMPAIGN conservative charge {total:.3f} of cap {self.cap:.2f}", flush=True)
        return total


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
    parser.add_argument("--validation", type=Path, required=True, help="validation.json made in this runtime")
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
    if len(set(cases)) != len(cases):
        parser.error("a case is listed twice; use --repeats for more pairs of one case")
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
            "--execute-paid-rollouts",
        ]  # fmt: skip

    os.environ["PYTHONPATH"] = str(ROOT / "src")
    campaign = Campaign(args.folder, cap=args.cap, pair_cap=args.pair_cap, run_args=run_args)
    campaign.run_all([(*case, repeat) for repeat in range(1, args.repeats + 1) for case in cases], args.workers)


if __name__ == "__main__":
    main()
