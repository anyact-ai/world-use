"""One JSON row per task of one or more run.py outputs, as in development-check.jsonl.

    python report.py results/development/alone results/development/world-use > rows.jsonl

A row has the harness's verdict and ending, the Robot API calls made (from the harness's audit log, so a hand tool's
moves count), the policy's tool calls, and the model requests and estimated cost the arm's ledger recorded while
the task ran.
"""

from __future__ import annotations

import collections
import json
import re
import sys
import tomllib
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = ("move_robot", "get_camera_point_3d", "done", "grasp", "close_grasp", "place", "release_placement", "push",
         "tilt")


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rows(arm_dir: Path, long_tasks) -> list[dict]:
    run = json.loads((arm_dir / "run.json").read_text())
    ledger = lines(arm_dir / "ledger.jsonl") if (arm_dir / "ledger.jsonl").exists() else []
    found = []
    for task in sorted(arm_dir.glob("*/runs/robowild/*"), key=lambda p: p.name):
        # A resumed evaluation can leave more than one run of a task: the row is the latest that finished.
        finished = [(p, v) for p in sorted(task.iterdir()) if p.is_dir() and (v := next(p.rglob("result.json"), None))]
        if not finished:  # still running
            continue
        record, verdict = finished[-1]
        # An attempt the harness judged before the agent's first call may have no events, audit or conversation.
        events, audit, conversation = (lines(file) if (file := next(record.rglob(name), None)) else []
                                       for name in ("events.jsonl", "audit.jsonl", "conversation.jsonl"))
        times = [datetime.fromisoformat(events[i]["time"]).timestamp() for i in (0, -1)] if events else [0, -1]
        spent = [r for r in ledger if times[0] <= r["at"] <= times[1]]
        result = json.loads(verdict.read_text())
        robot = sum(1 for r in audit if r.get("event") == "capability_call" and r.get("capability") != "done")
        called = [m[1] for m in (re.match(r"(\w+)\(", c.get("text") or "") for c in conversation) if m]
        final = [c.get("text") or "" for c in conversation if c.get("phase") == "final"]
        found.append(dict(
            arm=run["arm"], task=task.name, budget="long" if task.name in long_tasks else "short",
            model=run["model"], reasoning_effort=run.get("reasoning_effort"),
            world_use_commit=run["world_use_commit"], success=result.get("success"),
            ended=result.get("termination_reason"), robot_calls=robot,
            tool_calls=dict(collections.Counter(n for n in called if n in TOOLS)),
            model_requests=len(spent), usd=round(sum(r["usd"] for r in spent), 2),
            closing=final[-1][:300] if final else ""))
    return found


def main():
    long_tasks = tomllib.loads((HERE / "cases.toml").read_text())["tasks"]["long"]
    for arm_dir in sys.argv[1:]:
        for row in rows(Path(arm_dir), long_tasks):
            print(json.dumps(row))


if __name__ == "__main__":
    main()
