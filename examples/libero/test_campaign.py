"""Case roles, campaign admission and result collection, without a simulator or network."""

import json
import sys

import campaign
import pytest
import report
from campaign import Campaign
from run import check_case, episode


@pytest.mark.parametrize(
    ("task", "state", "held_out", "refusal"),
    [(0, 4, False, "held-out"), (4, 6, False, "held-out"), (0, 0, True, "held-out"), (2, 3, False, "not a case")],
)
def test_a_case_runs_only_in_its_listed_role(tmp_path, task, state, held_out, refusal):
    with pytest.raises(ValueError, match=refusal):
        episode(tmp_path / "arm", "baseline", None, task_id=task, state=state, held_out=held_out)
    assert not (tmp_path / "arm").exists()


def test_listed_cases_run_in_their_role():
    check_case(9, 1, False)
    check_case(7, 4, True)
    check_case(9, 6, True)


def test_held_out_cases_need_their_own_authorization(monkeypatch, tmp_path, capsys):
    argv = ["campaign.py", str(tmp_path), "0:0", "0:4", "--cap", "9", "--pair-cap", "3", "--arm-cap", "1.5"]
    monkeypatch.setattr(sys, "argv", [*argv, "--authorization", "yes", "--validation", "validation.json"])
    with pytest.raises(SystemExit):
        campaign.main()
    assert "--held-out-authorization" in capsys.readouterr().err
    assert not tmp_path.joinpath("task00_init00").exists()


def ledger(folder, charge, settled=True):
    (folder / "api").mkdir(parents=True)
    rows = [dict(arm="baseline", event="reserved", request="0001-baseline", total_charge=charge)]
    if settled:
        rows.append(dict(arm="baseline", event="settled", request="0001-baseline", total_charge=charge))
    (folder / "api" / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def fake_pairs(charges, launched):
    """A run.py stand-in: each attempt charges the next amount and finishes both arms. None is an attempt the
    transport stopped in its first arm, with a request's usage unknown."""

    def call(args, **_):
        output = args[0]
        launched.append(output.name)
        charge = charges.pop(0)
        ledger(output, charge or 1.0, settled=charge is not None)
        summary = dict(protocol=dict(arms=["baseline", "world_use"]), results=[] if charge is None else [{}, {}])
        (output / "summary.json").write_text(json.dumps(summary))
        return 0

    return call


def fake_campaign(folder, cap, charges, launched):
    return Campaign(
        folder, cap=cap, pair_cap=4, run_args=lambda task, state, out: [out], call=fake_pairs(charges, launched)
    )


def test_admission_counts_finished_charges_and_running_caps(tmp_path):
    launched = []
    pairs = fake_campaign(tmp_path, 10, [3] * 4, launched)
    pairs.run_all([(task, 0, 1) for task in range(4)], workers=1, stagger=0)
    # 0 + 4, 3 + 4 and 6 + 4 fit the cap; 9 + 4 does not.
    assert launched == ["task00_init00", "task01_init00", "task02_init00"]
    assert pairs.spent() == 9


def test_running_pairs_hold_their_full_cap(tmp_path):
    pairs = Campaign(tmp_path, cap=10, pair_cap=4, run_args=None)
    assert [pairs.admit(name) for name in "abc"] == [True, True, False]


def test_a_pair_that_did_not_finish_is_tried_again_and_stays_charged(tmp_path):
    launched = []
    pairs = fake_campaign(tmp_path, 100, [None, 2, None, None, None, 1], launched)
    pairs.run_all([(0, 0, 1), (1, 0, 1), (2, 0, 1)], workers=1, stagger=0)
    assert launched == [
        "task00_init00", "task00_init00_retry1",                          # stopped, then finished
        "task01_init00", "task01_init00_retry1", "task01_init00_retry2",  # three attempts and no more
        "task02_init00",
    ]  # fmt: skip
    # Each unsettled reservation is kept in full beside the settled charges.
    assert pairs.spent() == 1 + 2 + 3 * 1 + 1


def test_repeats_get_their_own_folders_and_an_interrupted_attempt_is_retried(tmp_path):
    (tmp_path / "task00_init00").mkdir()  # interrupted: no summary
    launched = []
    pairs = fake_campaign(tmp_path, 100, [1] * 4, launched)
    pairs.run_all([(0, 0, 1), (0, 0, 2)], workers=1, stagger=0)
    pairs.run_all([(0, 0, 1), (0, 0, 2)], workers=1, stagger=0)
    assert launched == ["task00_init00_retry1", "task00_init00_repeat2"]


def pair(folder, results, *, task=0, state=0, stopped=None, initial="a" * 64):
    folder.mkdir()
    rows = []
    for i, (arm, _, calls) in enumerate(results):
        usage = dict(input_tokens=100 * calls, input_tokens_details=dict(cached_tokens=10), output_tokens=5)
        rows.append(dict(arm=arm, event="settled", request=f"{i:04d}-{arm}", usage=usage, total_charge=0.1))
    (folder / "api").mkdir()
    (folder / "api" / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = dict(
        protocol=dict(task_id=task, initial_state=state, arms=["baseline", "world_use"]),
        results=[],
        total_estimated_usd=0.2,
        total_budget_charge_usd=0.3,
    )
    for arm, success, calls in results:
        if success is None:  # the arm the transport stopped has no result
            continue
        (folder / arm).mkdir()
        (folder / arm / "transcript.json").write_text(json.dumps([dict(tool_calls=[dict(name="move_to")] * calls)]))
        summary["results"].append(
            dict(
                arm=arm, task_id=task, state=state, success=success, status="success" if success else "give_up",
                calls=calls, steps=10 * calls, wall_seconds=12.4, estimated_usd=0.01 * calls, budget_charge_usd=0.02,
                initial_observation_sha256=initial,
            )
        )  # fmt: skip
    if stopped:
        summary["stopped"] = dict(type="BudgetStop", reason=stopped)
    (folder / "summary.json").write_text(json.dumps(summary))


def test_report_collects_repeats_and_infrastructure_stops_and_tables_the_latest_commit(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    pair(old / "task00_init00", [("baseline", True, 40), ("world_use", False, 20)])
    pair(new / "task00_init00", [("baseline", False, 30), ("world_use", True, 10)])
    pair(new / "task00_init00_repeat2", [("baseline", True, 50), ("world_use", True, 12)])
    pair(new / "task00_init02", [("baseline", None, 64)], state=2, stopped="request would exceed budget")
    # A pair stopped in its second arm: the finished first arm is no comparison either.
    pair(new / "task00_init03", [("baseline", True, 30), ("world_use", None, 5)], state=3, stopped="request failed")
    out = tmp_path / "results.jsonl"
    report.collect(out, [f"a={old}", f"b={new}"])
    report.collect(out, [f"b={new}"])  # nothing new to add
    rows = [json.loads(line) for line in out.read_text().splitlines()]

    assert [(r["commit"], r["pair"], r["arm"], r["success"], r.get("repeat")) for r in rows] == [
        ("a", "task00_init00", "raw", True, None),
        ("a", "task00_init00", "world_use", False, None),
        ("b", "task00_init00", "raw", False, None),
        ("b", "task00_init00", "world_use", True, None),
        ("b", "task00_init00_repeat2", "raw", True, 2),
        ("b", "task00_init00_repeat2", "world_use", True, 2),
        ("b", "task00_init02", "raw", None, None),
        ("b", "task00_init03", "world_use", None, None),
    ]
    stop = rows[-2]
    assert stop["ended"] == "infrastructure stop: request would exceed budget"
    assert (stop["calls"], stop["input_tokens"], stop["estimated_usd"]) == (1, 6400, 0.2)
    assert rows[2]["tool_calls"] == {"move_to": 30} and rows[3]["procedure_outcomes"] == {}

    table = report.tables(rows)
    assert "| 0 | 0 | development | `b` | 1 of 2 | `b` | 2 of 2 | 40 / 11 |" in table
    assert "held_out" not in table  # a stopped pair is not a result
    # Totals add up episodes, not the cases' means: 10 + 12 calls, median 11.
    assert "| development | World Use | 2 of 2 | 22, 11 |" in table and "| task 0 | raw | 1 of 2 | 80, 40 |" in table
    assert "†" not in table

    # A later World Use run alone is tabled beside the earlier raw run, and marked if it began elsewhere.
    alone = tmp_path / "alone"
    alone.mkdir()
    pair(alone / "task00_init00", [("world_use", True, 9)], initial="b" * 64)
    report.collect(out, [f"c={alone}"])
    table = report.tables([json.loads(line) for line in out.read_text().splitlines()])
    assert "| 0 | 0 † | development | `b` | 1 of 2 | `c` | 1 of 1 | 40 / 9 |" in table
    assert "did not start from the same initial observation" in table
