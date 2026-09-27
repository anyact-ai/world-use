"""`wu`: the command line a policy (or a person) drives the robot with. Output is short and plain on purpose:
every line ends up in a model's context.

    wu up --body sim            start the daemon (detached: it outlives this shell and any agent session)
    wu card                     what this robot is and can do
    wu status                   one line: job, tool position, gripper, torques, heat
    wu check '<spec>'|file      rehearse on a twin from the measured state; nothing real moves
    wu run '<spec>'|file        run it; waits up to --wait seconds, then prints the outcome and the state line
    wu answer JOB yes|no|...    answer a checkpoint question
    wu stop | wu home | wu events | wu down
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .client import DEFAULT_URL, Client, DaemonError


def _spec(text: str):
    p = Path(text)
    if not text.lstrip().startswith(("{", "[")) and p.exists():
        text = p.read_text()
    return json.loads(text)


def _print_job(d: dict):
    if d.get("incident"):
        print(d["incident"])
        return
    out = d.get("outcome")
    if out:
        print(f"job {d['id']} {out['status']}: {out['message']}")
    elif d.get("question"):
        q = d["question"]
        where = f" (look at: {q['view']}" + (f" {q['roi']}" if q.get("roi") else "") + ")" if q.get("view") else ""
        print(f"job {d['id']} waiting: {q['ask']}{where}; answer with: wu answer {d['id']} <answer>")
    else:
        print(f"job {d['id']} {d['status']}: {d['what']}")
    print(d["line"])


def cmd_up(a):
    c = Client(a.url)
    if c.alive():
        print(c.status()["line"])
        return 0
    log_dir = Path(a.runs)
    log_dir.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, "-m", "world_use.daemon", "--port", a.url.rsplit(":", 1)[-1].split("/")[0], "--runs", str(log_dir)]
    if a.body:
        args += ["--body", a.body]
    if a.workcell:
        args += ["--workcell", a.workcell]
    if a.enable:
        args.append("--enable")
    with open(log_dir / "daemon.log", "a") as log:
        subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    for _ in range(100):
        time.sleep(0.1)
        if c.alive():
            print(c.status()["line"])
            return 0
    print(f"the daemon did not come up; see {log_dir / 'daemon.log'}", file=sys.stderr)
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wu", description="drive a robot through the world-use daemon")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("up", help="start the daemon")
    up.add_argument("--body")
    up.add_argument("--workcell")
    up.add_argument("--runs", default=os.environ.get("WORLD_USE_RUNS", "runs"))
    up.add_argument("--enable", action="store_true")
    sub.add_parser("down", help="release at rest and stop the daemon")
    sub.add_parser("status")
    sub.add_parser("card")
    for name in ("run", "check"):
        p = sub.add_parser(name)
        p.add_argument("spec", help="JSON spec or a file containing one")
        if name == "run":
            p.add_argument("--wait", type=float, default=60.0, help="seconds to wait for the outcome")
    p = sub.add_parser("answer")
    p.add_argument("job", type=int)
    p.add_argument("answer")
    p.add_argument("--wait", type=float, default=60.0)
    p = sub.add_parser("job")
    p.add_argument("job", type=int)
    p.add_argument("--wait", type=float, default=0.0)
    p = sub.add_parser("stop")
    p.add_argument("reason", nargs="?", default="stop requested")
    p = sub.add_parser("home")
    p.add_argument("--wait", type=float, default=60.0)
    p = sub.add_parser("home-route", help="set the way home, e.g. '[]' = fold straight home from here")
    p.add_argument("steps")
    p.add_argument("--note", default="")
    p = sub.add_parser("events")
    p.add_argument("--since", type=int, default=0)
    p.add_argument("--wait", type=float, default=0.0)
    p = sub.add_parser("fact", help="record something measured or seen")
    p.add_argument("key")
    p.add_argument("value")
    p.add_argument("--source", default="policy")
    for name in ("enable", "release", "reset"):
        sub.add_parser(name)
    a = ap.parse_args(argv)
    c = Client(a.url)
    try:
        if a.cmd == "up":
            return cmd_up(a)
        if a.cmd == "down":
            r = c.shutdown()
            print(json.dumps(r["summary"]) if a.json else _summary(r["summary"]))
            return 0
        r = _dispatch(a, c)
        if a.json:
            print(json.dumps(r, indent=1))
        elif isinstance(r, dict) and "id" in r and "status" in r:
            _print_job(r)
        elif isinstance(r, dict) and "text" in r:
            print(r["text"])
        elif isinstance(r, dict) and "line" in r:
            print(r["line"])
        elif isinstance(r, dict) and "home" in r:
            print(f"home: {r['home']}")
        elif isinstance(r, str):
            print(r)
        else:
            print(json.dumps(r, indent=1))
        return 0
    except DaemonError as e:
        refused = e.body.get("refused")
        if refused:
            print(f"refused: {refused['message']}" + (f" (hint: {refused['hint']})" if refused.get("hint") else ""))
        else:
            print(f"error: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"cannot reach the daemon at {a.url} ({e}); start it with: wu up", file=sys.stderr)
        return 3


def _dispatch(a, c: Client):
    if a.cmd == "status":
        return c.status()
    if a.cmd == "card":
        return c.card()
    if a.cmd == "run":
        return c.run(_spec(a.spec), wait=a.wait)
    if a.cmd == "check":
        return c.check(_spec(a.spec))
    if a.cmd == "answer":
        return c.answer(a.job, a.answer, wait=a.wait)
    if a.cmd == "job":
        return c.job(a.job, wait=a.wait)
    if a.cmd == "stop":
        return c.stop(a.reason)
    if a.cmd == "home":
        return c.home(wait=a.wait)
    if a.cmd == "home-route":
        return c.home_route(_spec(a.steps), a.note)
    if a.cmd == "events":
        r = c.events(a.since, a.wait)
        return "\n".join(f"[{e['seq']}] {e['t']:>7.1f}s {e['level']:5s} {e['kind']}: {e['message']}" for e in r["events"]) or "(none)"
    if a.cmd == "fact":
        try:
            value = json.loads(a.value)
        except json.JSONDecodeError:
            value = a.value
        return c.world(fact=dict(key=a.key, value=value, source=a.source))
    return getattr(c, a.cmd)()


def _summary(s: dict) -> str:
    share = s.get("moving_share")
    return (f"powered {s.get('powered_s', 0)} s, moving {s.get('moving_s', 0)} s"
            + (f" ({100 * share:.0f}%)" if share is not None else "") + f"; max temps {s.get('max_temp_c')}")


if __name__ == "__main__":
    sys.exit(main())
