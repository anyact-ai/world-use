"""report.py on a run.py output laid out as the harness writes it.

    uv run pytest examples/robowild/test_report.py
"""

import json

from report import rows


def write(path, *records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_a_row_counts_the_tasks_robot_calls_and_the_spending_while_it_ran(tmp_path):
    arm = tmp_path / "world-use"
    write(arm / "run.json", dict(arm="world-use", model="m", reasoning_effort="high", world_use_commit="abc"))
    # 1800000000 is 2027-01-15T08:00:00Z; the task runs for its first 100 seconds.
    write(arm / "ledger.jsonl", dict(at=1800000050, usd=1.5), dict(at=1800000090, usd=0.5), dict(at=1800000200, usd=9))
    record = arm / "23-03/runs/robowild/23-03/long-0000-23-03"
    write(record / "events.jsonl", dict(event="run_started", time="2027-01-15T08:00:00+00:00"),
          dict(event="run_finished", time="2027-01-15T08:01:40+00:00"))
    write(record / "attempts/01/result.json", dict(success=True, termination_reason="success"))
    write(record / "model/audit.jsonl", *[dict(event="capability_call", capability=c)
                                           for c in ("move_robot", "move_robot", "get_camera_point_3d", "done")])
    write(record / "viewer/conversation.jsonl", dict(text='grasp({"approach": {}})'),
          dict(text="grasp returned {}"), dict(text="done({})"), dict(text="Placed.", phase="final"))
    (row,) = rows(arm, long_tasks=["23-03"])
    assert row["budget"] == "long" and row["success"] and row["closing"] == "Placed."
    assert row["robot_calls"] == 3 and row["tool_calls"] == {"grasp": 1, "done": 1}
    assert row["model_requests"] == 2 and row["usd"] == 2.0
