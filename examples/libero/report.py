"""Paid-run results: collect campaign folders into compact rows, and print comparison tables from them.

usage: report.py collect RESULTS_JSONL COMMIT=CAMPAIGN_DIR [COMMIT=CAMPAIGN_DIR ...]
       report.py tables RESULTS_JSONL
"""

from __future__ import annotations

import collections
import json
import re
import sys
import tomllib
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
CASES = tomllib.loads((SOURCE / "cases.toml").read_text())["case"]
ROLE = {(c["task"], c["state"]): c["role"] for c in CASES}
PAIR = re.compile(r"task\d\d_init\d\d(?:_repeat(\d+))?(?:_retry\d+)?")
ARM = dict(baseline="raw", world_use="world_use")


def usage(ledger):
    totals = collections.defaultdict(collections.Counter)
    for r in ledger:
        if r["event"] == "settled":
            u, a = r["usage"], totals[r["arm"]]
            a["input_tokens"] += u["input_tokens"]
            a["cached_input_tokens"] += u.get("input_tokens_details", {}).get("cached_tokens", 0)
            a["output_tokens"] += u["output_tokens"]
    return totals


def pair_rows(commit, pair):
    """Rows for one pair folder: a result per arm, or one row for the arm the transport stopped."""
    report = json.loads((pair / "summary.json").read_text())
    protocol = report["protocol"]
    ledger = [json.loads(line) for line in (pair / "api" / "ledger.jsonl").read_text().splitlines()]
    tokens = usage(ledger)
    repeat = PAIR.fullmatch(pair.name).group(1)
    head = dict(commit=commit, pair=pair.name, task=protocol["task_id"], state=protocol["initial_state"])
    if repeat:
        head["repeat"] = int(repeat)
    done = {r["arm"] for r in report.get("results", [])}
    stopped = [arm for arm in protocol["arms"] if arm not in done]
    if report.get("stopped") and stopped:
        # The transport refused a request. The pair is no comparison: one row, for the arm that was running,
        # charged with the whole pair, and no result for an arm that finished before it.
        arm = stopped[0]
        steps = pair / arm / "execution" / "steps.jsonl"
        return [dict(
            head, arm=ARM[arm], success=None, ended="infrastructure stop: " + report["stopped"]["reason"],
            calls=sum(r["event"] == "settled" and r["arm"] == arm for r in ledger),
            steps=len(steps.read_text().splitlines()) if steps.exists() else 0, wall_seconds=0, **tokens[arm],
            estimated_usd=round(report["total_estimated_usd"], 3),
            conservative_usd=round(report["total_budget_charge_usd"], 3), tool_calls={},
        )]  # fmt: skip
    rows = []
    for result in report.get("results", []):
        arm = result["arm"]
        tools, outcomes = collections.Counter(), collections.Counter()
        for message in json.loads((pair / arm / "transcript.json").read_text()):
            for call in message.get("tool_calls") or []:
                tools[(call.get("function") or call).get("name")] += 1
        record = pair / arm / "execution" / "world-use.jsonl"
        if arm == "world_use" and record.exists():
            for line in record.read_text().splitlines():
                o = json.loads(line)
                if o.get("procedure"):
                    outcomes[f"{o['procedure']}:{o['status']}"] += 1
                elif o.get("status") == "refused":
                    outcomes["refused"] += 1
        rows.append(dict(
            head, arm=ARM[arm], success=bool(result["success"]), ended=result["status"], calls=result["calls"],
            steps=result["steps"], wall_seconds=round(result["wall_seconds"]), **tokens[arm],
            estimated_usd=round(result["estimated_usd"], 3), conservative_usd=round(result["budget_charge_usd"], 3),
            tool_calls=dict(tools), **({"procedure_outcomes": dict(outcomes)} if arm == "world_use" else {}),
            initial=result["initial_observation_sha256"][:12],
        ))  # fmt: skip
    return rows


def collect(out: Path, campaigns):
    """Append the rows of every pair folder with a summary; rows already in the file are not added again."""
    seen = set()
    if out.exists():
        seen = {(r["commit"], r["pair"], r["arm"]) for r in map(json.loads, out.read_text().splitlines())}
    rows = []
    for item in campaigns:
        commit, folder = item.split("=", 1)
        for pair in sorted(Path(folder).iterdir()):
            if pair.is_dir() and PAIR.fullmatch(pair.name) and (pair / "summary.json").exists():
                rows += [r for r in pair_rows(commit, pair) if (r["commit"], r["pair"], r["arm"]) not in seen]
    with out.open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return rows


def latest(rows):
    """For each case and arm, the finished runs (every repeat) of the last commit in the file that ran it."""
    rank = {commit: i for i, commit in enumerate(dict.fromkeys(r["commit"] for r in rows))}
    chosen = {}
    for r in rows:
        if r["success"] is None:
            continue
        key = (r["task"], r["state"], r["arm"])
        if key not in chosen or rank[r["commit"]] > rank[chosen[key][0]["commit"]]:
            chosen[key] = [r]
        elif r["commit"] == chosen[key][0]["commit"]:
            chosen[key].append(r)
    return chosen


def summarize(runs):
    n = len(runs)
    return dict(
        solved=sum(r["success"] for r in runs),
        n=n,
        calls=sum(r["calls"] for r in runs) / n,
        estimated_usd=sum(r["estimated_usd"] for r in runs) / n,
        steps=sum(r["steps"] for r in runs) / n,
    )


def number(value):
    return f"{value:,.0f}" if value == round(value) else f"{value:,.1f}"


def totals(runs):
    """Sums over episodes: solved, calls (and the median episode's), estimated cost, steps, input tokens."""
    calls = sorted(r["calls"] for r in runs)
    median = (calls[(len(calls) - 1) // 2] + calls[len(calls) // 2]) / 2
    return (
        f"{sum(r['success'] for r in runs)} of {len(runs)} | {sum(calls):,}, {number(median)} "
        f"| {sum(r['estimated_usd'] for r in runs):.2f} | {sum(r['steps'] for r in runs):,} "
        f"| {sum(r['input_tokens'] for r in runs):,}"
    )


def tables(rows):
    chosen = latest(rows)
    cases = sorted({(task, state) for task, state, _ in chosen}, key=lambda c: (ROLE[c], c))
    lines = [
        "Latest run of each case; calls, cost and steps are means over its repeats.",
        "",
        "| task | state | role | raw commit | raw solved | World Use commit | World Use solved "
        "| calls raw / WU | est. USD raw / WU | steps raw / WU |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    unmatched = False
    for case in cases:
        runs = {arm: chosen.get((*case, arm)) for arm in ("raw", "world_use")}
        s = {arm: summarize(r) if r else None for arm, r in runs.items()}
        cells = []
        for arm in ("raw", "world_use"):
            cells += [f"`{runs[arm][0]['commit']}`", f"{s[arm]['solved']} of {s[arm]['n']}"] if s[arm] else ["", ""]
        for key, fmt in (("calls", number), ("estimated_usd", "{:.2f}".format), ("steps", number)):
            cells.append(" / ".join(fmt(s[arm][key]) if s[arm] else "-" for arm in ("raw", "world_use")))
        # Arms run in one pair start from one observation. Rows paired across campaigns are checked here.
        starts = {r["initial"] for arm_runs in runs.values() for r in arm_runs or () if "initial" in r}
        unmatched = unmatched or len(starts) > 1
        mark = " †" if len(starts) > 1 else ""
        lines.append(f"| {case[0]} | {case[1]}{mark} | {ROLE[case]} | " + " | ".join(cells) + " |")
    if unmatched:
        lines += ["", "† The arms' runs of this case did not start from the same initial observation."]
    lines += [
        "",
        "Totals over every episode of those runs.",
        "",
        "| cases | arm | solved | calls, total and median | est. USD | steps | input tokens |",
        "|---|---|---|---|---|---|---|",
    ]
    groups = [(role, [c for c in cases if ROLE[c] == role]) for role in dict.fromkeys(ROLE[c] for c in cases)]
    groups += [("all", cases)] + [(f"task {t}", [c for c in cases if c[0] == t]) for t in sorted({c[0] for c in cases})]
    for name, which in groups:
        for arm, label in (("raw", "raw"), ("world_use", "World Use")):
            runs = [r for case in which for r in chosen.get((*case, arm), ())]
            if runs:
                lines.append(f"| {name} | {label} | {totals(runs)} |")
    return "\n".join(lines)


def main():
    command, out = sys.argv[1], Path(sys.argv[2])
    if command == "collect":
        for r in collect(out, sys.argv[3:]):
            ended = "success" if r["success"] else r["ended"][:28]
            print(f"{r['commit']:8s} {r['pair']:22s} {r['arm']:9s} {ended:28s} calls {r['calls']:3d} "
                  f"steps {r['steps']:3d} ${r['estimated_usd']:.2f}")  # fmt: skip
    elif command == "tables":
        print(tables([json.loads(line) for line in out.read_text().splitlines()]))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
