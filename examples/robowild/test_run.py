"""run.py against a stand-in robowild-harness whose every task costs $30.

    uv run pytest examples/robowild/test_run.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = """#!{python}
import json, sys
from pathlib import Path
args = sys.argv[1:]
out = Path(args[args.index("--output") + 1])
with open(out.parent / "ran.txt", "a") as stream:
    stream.write(args[args.index("--tasks") + 1] + "\\n")
workspace = out / "runs/robowild" / args[args.index("--tasks") + 1] / "run-0/agent-workspaces/job-0"
for name in (".venv/bin/python", "dependencies/robot_sdk.py", "observations/0.png"):
    (workspace / name).parent.mkdir(parents=True, exist_ok=True)
    (workspace / name).write_text("")
ledger = out.parent / "ledger.jsonl"
lines = ledger.read_text().splitlines() if ledger.exists() else []
spent = json.loads(lines[-1])["spent_usd"] if lines else 0.0
with open(ledger, "a") as stream:
    stream.write(json.dumps(dict(usd=30.0, spent_usd=spent + 30.0)) + "\\n")
"""


def test_tasks_run_one_at_a_time_while_the_budget_covers_the_costliest_and_leave_no_environment_copy(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    harness = bin_dir / "robowild-harness"
    harness.write_text(HARNESS.format(python=sys.executable))
    harness.chmod(0o755)
    credentials = tmp_path / "creds.json"
    credentials.write_text(json.dumps(dict(base_url="http://127.0.0.1:9/v1", key="secret-key")))
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", ASSETS_DIR="assets",
               CUROBO_SOURCE="curobo", WORLD_USE_COMMIT="test")
    run = subprocess.run(
        [sys.executable, str(HERE / "run.py"), "--arm", "alone", "--split", "development",
         "--tasks", "01-01,02-06,04-02", "--model", "gpt-6.2-astra", "--credentials", str(credentials),
         "--budget-usd", "70", "--authorization", "a test", "--output", str(tmp_path / "out")],
        env=env, capture_output=True, text=True, timeout=60)
    # $70: the first task takes $30, the second starts with $40 left and takes $30, and $10 cannot cover a third.
    assert (tmp_path / "out/alone/ran.txt").read_text().split() == ["01-01", "02-06"]
    assert run.returncode != 0 and "Not run: 04-02" in run.stderr
    assert not credentials.exists()
    assert json.loads((tmp_path / "out/alone/run.json").read_text())["not_run"] == ["04-02"]
    # The copy of the harness's environment in each workspace goes once the task has run; its records stay.
    (workspace,) = (tmp_path / "out/alone/01-01").glob("runs/*/*/*/agent-workspaces/*")
    assert sorted(p.name for p in workspace.iterdir()) == ["observations"]
