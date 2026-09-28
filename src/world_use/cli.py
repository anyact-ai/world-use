"""`wu`: the command line a policy (or a person) drives the robot with. Output is short and plain on purpose:
every line ends up in a model's context.

    wu up --body sim            start the daemon (detached: it outlives this shell and any agent session)
    wu card                     what this robot is and can do
    wu status                   one line: job, tool position, gripper, torques, heat
    wu look [CAMERA]            save a picture, with the tool and the known boxes drawn on it; prints its path
    wu run '<spec>'|file        rehearse, then run; waits up to --wait seconds, then prints the outcome
    wu run --checked            run the plan the last `wu check` rehearsed, without pasting it again
    wu check '<spec>'|file      rehearse only: the forecast, nothing real moves
    wu answer JOB yes|no|...    answer a checkpoint question
    wu world | wu box ...       what the kernel knows about the scene; tell it about a surface or object
    wu help [STEP]              the steps a plan can use, from the running daemon
    wu fact KEY VALUE           record a measurement with its source
    wu home-route '<steps>'     the way home from here ('[]' = fold straight back); wu home runs it
    wu record                   write the flight record so far (tape, summary, world), without stopping
    wu stop | events | enable | release | down
    wu mcp                      the same verbs as MCP tools, over stdio

The exit status says how it went, so a shell chain stops where the robot did: 0 done (a job) or passed (a
check); 4 refused, surprise, stopped, faulted or cancelled, or a check that would not pass; 5 waiting at a
checkpoint (`wu answer`); 6 still running when the wait ran out (`wu job ID --wait 60`); 2 the daemon refused the
request; 3 no daemon. `--json` works before or after the command.
"""
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


def job_text(d: dict) -> str:
    """A job as a policy reads it: the outcome (or the question it waits on) and the state line."""
    lines = [f"warning: {d['warning']}"] if d.get("warning") else []
    if d.get("incident"):
        return "\n".join(lines + [d["incident"]])
    out = d.get("outcome")
    if out:
        lines.append(f"job {d['id']} {out['status']}: {out['message']}")
    elif d.get("question"):
        q = d["question"]
        where = f" (look at: {q['view']}" + (f" {q['roi']}" if q.get("roi") else "") + ")" if q.get("view") else ""
        lines.append(f"job {d['id']} waiting: {q['ask']}{where}; answer with: wu answer {d['id']} <answer>")
    else:
        more = f"; keep waiting with: wu job {d['id']} --wait 60" if d["status"] in ("queued", "running") else ""
        lines.append(f"job {d['id']} {d['status']}: {d['what']}{more}")
    return "\n".join(lines + [d["line"]])


JOB_EXIT = {"done": 0, "waiting": 5, "queued": 6, "running": 6}


def exit_status(a, r) -> int:
    """How a command went, as a shell reads it: see the module docstring."""
    if isinstance(r, dict) and "id" in r and "status" in r:
        return JOB_EXIT.get(r["status"], 4)
    if a.cmd == "check" and isinstance(r, dict):
        return 0 if r.get("ok") else 4
    return 0


def cmd_up(a):
    c = Client(a.url)
    if c.alive():
        print(c.status()["line"])
        return 0
    log_dir = Path(a.runs)
    log_dir.mkdir(parents=True, exist_ok=True)
    port = a.url.rsplit(":", 1)[-1].split("/")[0]
    args = [sys.executable, "-m", "world_use.daemon", "--port", port, "--runs", str(log_dir)]
    if a.body:
        args += ["--body", a.body]
    if a.workcell:
        args += ["--workcell", a.workcell if not Path(a.workcell).exists() else str(Path(a.workcell).resolve())]
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
    sub.add_parser("record", help="write the flight record so far, without stopping")
    sub.add_parser("status")
    sub.add_parser("card")
    for name in ("run", "check"):
        p = sub.add_parser(name)
        p.add_argument("spec", nargs="?" if name == "run" else None, help="JSON spec or a file containing one")
        if name == "run":
            p.add_argument("--wait", type=float, default=60.0, help="seconds to wait for the outcome")
            p.add_argument("--no-check", action="store_true", help="skip the rehearsal")
            p.add_argument("--checked", action="store_true", help="run the plan the last `wu check` rehearsed")
    p = sub.add_parser("look", help="save a picture from a camera and print its path")
    p.add_argument("camera", nargs="?")
    p.add_argument("--plan", help="draw this plan's tool path on the picture (JSON spec or file)")
    p.add_argument("--grid", action="store_true", help="a pixel ruler and nothing the kernel believes")
    p = sub.add_parser("help", help="the steps a plan can use")
    p.add_argument("step", nargs="?")
    sub.add_parser("world", help="frames, boxes and facts the kernel knows")
    sub.add_parser("mcp", help="serve these commands as MCP tools over stdio (needs world-use[mcp])")
    p = sub.add_parser("box", help="tell the kernel about a surface, object or zone (work frame, metres)")
    p.add_argument("name")
    p.add_argument("kind", nargs="?", help="surface | object | keep_out | fragile | slow")
    p.add_argument("center", nargs="?", help="F,L,U of its centre, e.g. 0.32,0,0.14")
    p.add_argument("size", nargs="?", help="its size forward,left,up, e.g. 0.3,0.4,0.02")
    p.add_argument("--yaw", type=float, default=0.0, help="degrees it is turned about up")
    p.add_argument("--frame", default="work")
    p.add_argument("--source", default="policy", help="where you learned it, e.g. 'side camera'")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="e.g. grip_width=0.04, dtau=0.3")
    p.add_argument("--remove", action="store_true", help="forget this box")
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
    for p in sub.choices.values():                  # `wu status --json` as well as `wu --json status`
        p.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="print raw JSON")
    a = ap.parse_args(argv)
    c = Client(a.url)
    try:
        if a.cmd == "up":
            return cmd_up(a)
        if a.cmd == "mcp":
            try:
                from .mcp_server import serve
            except ImportError:
                print("the MCP server needs the mcp extra: uv tool install 'world-use[mcp] @ git+https://github.com/anyact-ai/world-use'",
                      file=sys.stderr)
                return 1
            serve(a.url)
            return 0
        if a.cmd == "down":
            r = c.shutdown()
            print(json.dumps(r["summary"]) if a.json else _summary(r["summary"]))
            return 0
        if a.cmd == "record":
            r = c.record()
            print(json.dumps(r) if a.json else f"{r['run'] or '(no run folder)'}\n{_summary(r['summary'])}")
            return 0
        r = _dispatch(a, c)
        if a.json:
            print(json.dumps(r, indent=1))
        elif isinstance(r, dict) and "id" in r and "status" in r:
            print(job_text(r))
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
        return exit_status(a, r)
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
        if a.spec is None and not a.checked:
            raise SystemExit("wu run '<plan>' (or wu run --checked, for the plan the last wu check rehearsed)")
        return c.run(None if a.checked else _spec(a.spec), wait=a.wait, check=not a.no_check, checked=a.checked)
    if a.cmd == "look":
        r = c.look(a.camera, None if a.plan is None else _spec(a.plan), a.grid)
        return f"{r['path']}\n{r['camera']} camera, {r['size'][0]}x{r['size'][1]}: {r['drawn']}" + (
            f"\n{r['check']}" if r.get("check") else "")
    if a.cmd == "help":
        return help_text(c, a.step)
    if a.cmd == "world":
        return c.world()["text"]
    if a.cmd == "box":
        if a.remove:
            return c.remove(a.name)["line"]
        if not (a.kind and a.center and a.size):
            raise SystemExit("wu box NAME KIND CENTER SIZE, e.g. wu box tray surface 0.32,0,0.14 0.3,0.4,0.02")
        extra = {}
        for kv in a.set:
            key, _, value = kv.partition("=")
            try:
                extra[key] = json.loads(value)
            except json.JSONDecodeError:
                extra[key] = value
        return c.box(a.name, a.kind, _vec(a.center), _vec(a.size), frame=a.frame, yaw_deg=a.yaw, source=a.source,
                     **extra)["line"]
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
        return "\n".join(f"[{e['seq']}] {e['t']:>7.1f}s {e['level']:5s} {e['kind']}: {e['message']}"
                         for e in r["events"]) or "(none)"
    if a.cmd == "fact":
        try:
            value = json.loads(a.value)
        except json.JSONDecodeError:
            value = a.value
        return c.world(fact=dict(key=a.key, value=value, source=a.source))
    return getattr(c, a.cmd)()


def _vec(text: str) -> list[float]:
    v = [float(x) for x in text.replace(" ", "").split(",")]
    if len(v) != 3:
        raise SystemExit(f"expected three numbers like 0.3,0,0.14; got {text!r}")
    return v


def help_text(c: Client, step: str | None = None) -> str:
    """The steps a plan can use, from the running daemon (so plugins show up), else from this installation."""
    try:
        steps = c.help()
    except OSError:                                     # no daemon: the steps this installation knows
        from .behaviors import REGISTRY
        steps = {kind: cls.help() for kind, cls in REGISTRY.items()}
    if step:
        h = steps.get(step)
        if h is None:
            return f"no step {step!r}; steps: {', '.join(steps)}"
        return "\n".join([f"{step}: {h['summary']}", h["params"], "every step also takes \"label\"",
                          f"e.g. {json.dumps(h['example'])}" if h.get("example") else ""]).strip()
    lines = ["steps (a plan is a JSON list of them; wu help STEP for parameters):"]
    for kind, h in steps.items():
        lines.append(f"  {kind:10s} {h['summary']}")
        if h.get("example"):
            lines.append(f"  {'':10s} {json.dumps(h['example'])}")
    return "\n".join(lines)


def _summary(s: dict) -> str:
    share = s.get("moving_share")
    return (f"powered {s.get('powered_s', 0)} s, moving {s.get('moving_s', 0)} s"
            + (f" ({100 * share:.0f}%)" if share is not None else "") + f"; max temps {s.get('max_temp_c')}")


if __name__ == "__main__":
    sys.exit(main())
