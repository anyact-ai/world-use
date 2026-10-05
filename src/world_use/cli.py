"""`wu`: the command line a policy (or a person) drives the robot with. Output is short and plain on purpose:
every line ends up in a model's context. `wu --help` adds the exit statuses and environment variables.

    wu up --body sim            start the daemon (detached: it outlives this shell and any agent session)
    wu card                     what this robot is and can do
    wu status                   one line: job, tool position, gripper, torques, heat
    wu look [CAMERA]            save a picture, with the tool and the known boxes drawn on it; prints its path
    wu run PLAN                 rehearse, then run; waits up to --wait seconds, then prints the outcome
    wu check PLAN               rehearse only: the forecast, nothing real moves
    wu answer JOB yes|no|...    answer a checkpoint question
    wu world | wu box ...       what the kernel knows about the scene; tell it about a surface or object
    wu help [STEP]              the steps a plan can use, from the running daemon
    wu fact KEY VALUE           record a measurement with its source
    wu home-route STEPS         the way home from here ('[]' = fold straight back); wu home runs it
    wu record                   write the flight record so far (tape, summary, world), without stopping
    wu calibrate CAMERA         find where a camera is from the arm: say where you see the tool point, 6-8 times
    wu fit RUN...               fit the robot's link masses and joint friction from flight records (no daemon)
    wu stop | events | enable | release | down
    wu mcp                      the same verbs as MCP tools, over stdio

A plan is JSON (one step or a list of steps) or a file containing it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import views
from .client import DEFAULT_URL, Client, DaemonError

EPILOG = """\
exit status (a shell chain stops where the robot did):
  0  done (a job) or passed (a rehearsal: check, look --plan, home-route)
  4  refused, surprise, stopped, faulted or cancelled, or a rehearsal that would not pass
  5  waiting at a checkpoint (wu answer)
  6  still running when the wait ran out (wu job ID --wait 60)
  2  the daemon refused the request, or the input is invalid
  3  no daemon (wu up starts one)
  1  the daemon or the MCP server could not start

environment:
  WORLD_USE_URL   the daemon's address (default http://127.0.0.1:7431)
  WORLD_USE_RUNS  where wu up and wu demo write flight records (default ./runs)

--json prints raw JSON; it works before or after the command."""

HINTS = dict(answer="wu answer {id} <answer>", wait="wu job {id} --wait 60")
JOB_EXIT = {"done": 0, "waiting": 5, "queued": 6, "running": 6}
UP_TIMEOUT_S = 60.0


def _spec(text: str):
    if not text.lstrip().startswith(("{", "[")) and text != "null":
        try:
            text = Path(text).expanduser().read_text()
        except OSError as e:
            raise ValueError(f"cannot read JSON file: {e}") from e
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}") from e


def exit_status(r) -> int:
    """How a command went, as a shell reads it: see EPILOG."""
    if isinstance(r, dict) and "id" in r and "status" in r:
        return JOB_EXIT.get(r["status"], 4)
    if isinstance(r, dict) and "ok" in r:
        return 0 if r["ok"] else 4
    return 0


def _unreachable(url: str, e: OSError) -> int:
    print(f"cannot reach the daemon at {url} ({e}); start it with: wu up", file=sys.stderr)
    return 3


def cmd_up(a):
    from .daemon import load_workcell, session_identity

    c = Client(a.url)
    cell = load_workcell(a.workcell)
    wanted = session_identity(a.body or cell.get("body", "sim"), cell)

    def connected():
        status = c.status()
        actual = status.get("session")
        if actual != wanted:
            adapter = actual.get("adapter", "unknown") if actual else "unknown (older daemon)"
            print(f"refused: {a.url} serves {adapter} with a different or unknown startup configuration; "
                  f"requested {wanted['adapter']}. Use its matching --body/--workcell, or stop that daemon "
                  "at rest before starting this one.", file=sys.stderr)
            return 2
        if a.enable:
            c.enable()
            status = c.status()
        print(json.dumps(status, indent=1) if a.json else status["line"])
        return 0

    if c.alive():
        return connected()
    log_dir = Path(a.runs)
    log_dir.mkdir(parents=True, exist_ok=True)
    port = a.url.rsplit(":", 1)[-1].split("/")[0]
    args = [sys.executable, "-m", "world_use.daemon", "--port", port, "--runs", str(log_dir)]
    if a.body:
        args += ["--body", a.body]
    if a.workcell:
        args += ["--workcell", a.workcell if not Path(a.workcell).exists() else str(Path(a.workcell).resolve())]
    path = log_dir / "daemon.log"
    with open(path, "a") as log:
        start = log.tell()
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
    # A cold start takes several seconds (more under load): wait as long as the daemon is still starting. Once its
    # port is open, a request waits until it serves, so no request may outlast the deadline either.
    deadline = time.monotonic() + UP_TIMEOUT_S
    while process.poll() is None and (left := deadline - time.monotonic()) > 0:
        if Client(a.url, timeout=left).alive():
            return connected()
        time.sleep(0.1)
    if process.poll() is None:
        process.terminate()           # before serving it has no signal handler, so this stops it
        try:
            process.wait(10)
        except subprocess.TimeoutExpired:
            print(f"the daemon did not answer within {UP_TIMEOUT_S:.0f} s and is still running (pid {process.pid}); "
                  f"see {path}", file=sys.stderr)
            return 1
        print(f"the daemon did not answer within {UP_TIMEOUT_S:.0f} s and was stopped; see {path}", file=sys.stderr)
        return 1
    if c.alive():                     # another daemon took the port meanwhile
        return connected()
    with open(path) as log:
        log.seek(start)
        lines = [line for line in log.read().splitlines() if line.strip()]
    reason = lines[-1] if lines else f"it exited with status {process.returncode}"
    print(f"the daemon did not start: {reason}\nsee {path}", file=sys.stderr)
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wu", description="drive a robot through the world-use daemon",
                                 epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL, help="the daemon's address (default: $WORLD_USE_URL)")
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="COMMAND")
    up = sub.add_parser("up", help="start the daemon in the background; it outlives this shell")
    up.add_argument("--body", help="sim | sim:<built-in> | rebot | module:Class (default: the workcell's, else sim)")
    up.add_argument("--workcell", help="TOML file, or a bundled name such as block: robot, cameras and scene")
    up.add_argument("--runs", default=os.environ.get("WORLD_USE_RUNS", "runs"),
                    help="folder for flight records and daemon.log (default: $WORLD_USE_RUNS, else ./runs)")
    up.add_argument("--enable", action="store_true", help="switch torque on once the daemon answers")
    sub.add_parser("down", help="release at rest, save the flight record and stop the daemon")
    sub.add_parser("status", help="one line: job, tool position, gripper, torques, heat")
    sub.add_parser("card", help="what this robot is and can do; read it once")
    sub.add_parser("policy", help="print the installed agent brief (no daemon needed)")
    for name, what in (("run", "rehearse a plan, then run it and wait for the outcome"),
                       ("check", "rehearse a plan without moving anything")):
        p = sub.add_parser(name, help=what)
        p.add_argument("plan", help="JSON (one step or a list of steps) or a file containing it")
        if name == "run":
            p.add_argument("--wait", type=float, default=60.0, help="seconds to wait for the outcome")
            p.add_argument("--no-check", action="store_true", help="skip the rehearsal")
    p = sub.add_parser("look", help="save a picture from a camera and print its path")
    p.add_argument("camera", nargs="?")
    p.add_argument("--plan", help="draw this plan's tool path on the picture (JSON or a file)")
    p.add_argument("--grid", action="store_true", help="a pixel ruler and nothing the kernel believes")
    p = sub.add_parser("answer", help="answer the question a checkpoint is waiting on")
    p.add_argument("job", type=int)
    p.add_argument("answer")
    p.add_argument("--wait", type=float, default=60.0, help="seconds to wait for the outcome")
    p = sub.add_parser("job", help="a job's outcome or the question it waits on")
    p.add_argument("job", type=int)
    p.add_argument("--wait", type=float, default=0.0, help="seconds to wait for it to finish or ask")
    p = sub.add_parser("stop", help="stop now: hold where it is and cancel anything queued")
    p.add_argument("reason", nargs="?", default="stop requested")
    p = sub.add_parser("home", help="go home along the home route, then fold to the rest pose")
    p.add_argument("--wait", type=float, default=60.0, help="seconds to wait for the outcome")
    p = sub.add_parser("home-route", help="set the way home from here and rehearse it ('[]' folds straight home)")
    p.add_argument("steps", help="JSON list of motion and gripper steps, [] or null (clears the route)")
    p.add_argument("--note", default="")
    sub.add_parser("world", help="frames, boxes and facts the kernel knows")
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
    p = sub.add_parser("fact", help="record something measured or seen, with its source")
    p.add_argument("key")
    p.add_argument("value", help="JSON (0.205, true, [1, 2]) or text")
    p.add_argument("--source", default="policy")
    p = sub.add_parser("events", help="what happened, oldest first, after event number --since")
    p.add_argument("--since", type=int, default=0)
    p.add_argument("--wait", type=float, default=0.0, help="seconds to wait for a newer event")
    p.add_argument("--limit", type=int, default=40, help="at most this many events (default 40)")
    sub.add_parser("enable", help="switch torque on where the arm is")
    sub.add_parser("release", help="switch torque off; only at the rest pose")
    sub.add_parser("reset", help="clear a fault once an operator has checked the robot")
    p = sub.add_parser("record", help="write the flight record so far, without stopping")
    p.add_argument("--note", default="", help="record an intervention or observation")
    p.add_argument("--context", help="JSON or a JSON file with agent/model inputs to retain")
    p = sub.add_parser("inspect", help="summarize a recorded run offline")
    p.add_argument("run", type=Path)
    p = sub.add_parser("replay", help="render recorded measurements and the world model; never operates hardware")
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, help="GIF to write (default: replay.gif in the run folder)")
    p.add_argument("--speed", type=float, default=1.0)
    p = sub.add_parser("view", help="open a read-only Rerun viewer for the local daemon or a recorded run")
    p.add_argument("run", nargs="?", type=Path, help="run folder; defaults to the local daemon's active record")
    p.add_argument("--follow", action="store_true", help="follow new samples in the supplied run folder")
    p.add_argument("--out", type=Path, help="save a portable .rrd instead of opening a window")
    p = sub.add_parser("calibrate", help="find where a camera is from the arm: answer where it sees the tool point")
    p.add_argument("camera")
    p.add_argument("--points", type=int, default=8, help="corners of the box to visit (6-8)")
    p.add_argument("--spread", type=float, help="half-width of the box, m (default: the largest that passes)")
    p.add_argument("--wait", type=float, default=60.0)
    p = sub.add_parser("fit", help="fit the robot's model (link masses, friction) from flight records")
    p.add_argument("runs", nargs="+", type=Path, help="flight record folders")
    p.add_argument("--body", help="built-in robot name or robot TOML file (default: the recorded model)")
    p.add_argument("--out", type=Path, default=Path("fit.json"))
    p = sub.add_parser("demo", help="run the scripted block task in simulation, with a success check")
    p.add_argument("--out", type=Path, help="an empty folder (default: a new block-demo folder in $WORLD_USE_RUNS)")
    p.add_argument("--scenario", choices=["nominal", "shifted", "missing", "misplaced"], default="nominal")
    p.add_argument("--no-video", action="store_true")
    p = sub.add_parser("help", help="the steps a plan can use")
    p.add_argument("step", nargs="?")
    p = sub.add_parser("mcp", help="serve these commands as MCP tools over stdio (needs the mcp extra)")
    p.add_argument("--vision", action="store_true", help="load optional EdgeTAM selection before powered work")
    p.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    p.add_argument("--model-path", help="local EdgeTAM checkpoint; otherwise use the pinned public revision")
    for p in sub.choices.values():                  # `wu status --json` as well as `wu --json status`
        p.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="print raw JSON")
    a = ap.parse_args(argv)
    c = Client(a.url)
    try:
        if a.cmd == "view":
            from .visualization import view
            folder = a.run
            if folder is None:
                try:
                    recorded = c.status().get("recording", {}).get("path")
                except OSError as e:
                    return _unreachable(a.url, e)
                if not recorded:
                    raise ValueError("the daemon has no run folder; start it with --runs or supply a saved run")
                folder = Path(recorded)
            result = view(folder, output=a.out, follow=a.follow or a.run is None)
            if result is not None:
                print(json.dumps(dict(path=str(result))) if a.json else result)
            return 0
        if a.cmd == "demo":
            return cmd_demo(a)
        if a.cmd in ("inspect", "replay"):
            from . import records
            if a.cmd == "inspect":
                r = records.inspect(a.run)
                print(json.dumps(r, indent=2) if a.json else records.describe(r))
            else:
                print(records.replay(a.run, a.out or a.run / "replay.gif", speed=a.speed))
            return 0
        if a.cmd == "policy":
            from . import policy_text
            print(policy_text())
            return 0
        if a.cmd == "up":
            return cmd_up(a)
        if a.cmd == "fit":
            return cmd_fit(a)
        if a.cmd == "mcp":
            try:
                from .mcp_server import serve
            except ImportError:
                print("the MCP server needs the mcp extra: uv tool install 'world-use[mcp] @ git+https://github.com/anyact-ai/world-use'",
                      file=sys.stderr)
                return 1
            serve(a.url, vision=a.vision, device=a.device, model_path=a.model_path)
            return 0
        if a.cmd == "down":
            r = c.shutdown()
            print(json.dumps(r["summary"]) if a.json else views.record_line(r["summary"]))
            return 0
        if a.cmd == "record":
            r = c.record(note=a.note, context=_spec(a.context) if a.context else None)
            print(json.dumps(r) if a.json else f"{r['run'] or '(no run folder)'}\n{views.record_line(r['summary'])}")
            return 0
        r = _dispatch(a, c)
        if a.json:
            print(json.dumps(r, indent=1))
        elif a.cmd == "look":
            print(f"{r['path']}\n{r['camera']} camera, {r['size'][0]}x{r['size'][1]}: {r['drawn']}" + (
                f"\n{r['check']}" if r.get("check") else ""))
        elif a.cmd == "events":
            lines = [views.event_line(e) for e in r["events"]] or ["(none)"]
            if r["more"]:
                lines.append(f"more after [{r['last']}]: wu events --since {r['last']}")
            print("\n".join(lines))
        elif a.cmd == "home-route":
            print(views.home_text(r))
        elif isinstance(r, dict) and "id" in r and "status" in r:
            print(views.job_text(r, **HINTS))
        elif isinstance(r, dict) and "text" in r:
            print(r["text"])
        elif isinstance(r, dict) and "line" in r:
            print(r["line"])
        elif isinstance(r, str):
            print(r)
        else:
            print(json.dumps(r, indent=1))
        return exit_status(r)
    except DaemonError as e:
        refused = e.body.get("refused")
        if getattr(a, "json", False):                 # a script asked for JSON: a refusal must parse too
            print(json.dumps(e.body))
        elif refused:
            print(f"refused: {refused['message']}" + (f" (hint: {refused['hint']})" if refused.get("hint") else ""))
        else:
            print(f"error: {e}", file=sys.stderr)
        return 2
    except (ValueError, KeyError) as e:
        print(f"{a.cmd}: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        if a.cmd in ("demo", "inspect", "replay", "view", "fit", "policy", "up"):
            print(f"{a.cmd}: {e}", file=sys.stderr)
            return 2
        return _unreachable(a.url, e)


def _dispatch(a, c: Client):
    if a.cmd == "status":
        return c.status()
    if a.cmd == "card":
        return c.card()
    if a.cmd == "run":
        return c.run(_spec(a.plan), wait=a.wait, check=not a.no_check)
    if a.cmd == "calibrate":
        return c.calibrate(a.camera, a.points, a.spread, a.wait)
    if a.cmd == "look":
        return c.look(a.camera, None if a.plan is None else _spec(a.plan), a.grid)
    if a.cmd == "help":
        return _steps(c, a.step)
    if a.cmd == "world":
        return c.world()
    if a.cmd == "box":
        if a.remove:
            return c.remove(a.name)
        if not (a.kind and a.center and a.size):
            raise ValueError("wu box NAME KIND CENTER SIZE, e.g. wu box tray surface 0.32,0,0.14 0.3,0.4,0.02")
        extra = {key: views.parse_value(value) for key, _, value in (kv.partition("=") for kv in a.set)}
        return c.box(a.name, a.kind, _vec(a.center), _vec(a.size), frame=a.frame, yaw_deg=a.yaw, source=a.source,
                     **extra)
    if a.cmd == "check":
        return c.check(_spec(a.plan))
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
        return c.events(a.since, a.wait, a.limit)
    if a.cmd == "fact":
        return c.world(fact=dict(key=a.key, value=views.parse_value(a.value), source=a.source))
    return getattr(c, a.cmd)()


def cmd_demo(a) -> int:
    from .examples.pick_place import run
    folder = a.out or _fresh(Path(os.environ.get("WORLD_USE_RUNS", "runs")) / "block-demo")
    r = run(folder, a.scenario, video=not a.no_video)
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        jobs = [*r["outcomes"], r["return_outcome"]]
        lines = [f"{r['scenario']}: the block was {'' if r['success'] else 'not '}placed at the target; jobs "
                 + ", ".join(o["status"] for o in jobs) + "; torque " + ("off" if r["torque_off"] else "STILL ON")]
        lines += [f"  {o['status']}: {o['message']}" for o in jobs if o["status"] != "done"]
        lines.append(f"record: {folder}")
        if (folder / "demo.gif").exists():
            lines.append(f"animation: {folder / 'demo.gif'}")
        print("\n".join(lines))
    return 0 if r["success"] and r["torque_off"] else 4


def _fresh(base: Path) -> Path:
    """base, else the first of base-2, base-3, ... that is missing or empty."""
    path, n = base, 1
    while path.exists() and any(path.iterdir()):
        n += 1
        path = base.with_name(f"{base.name}-{n}")
    return path


def cmd_fit(a) -> int:
    from . import bodies, fit
    from .config import load_robot
    from .kinematics import Chain
    runs = [r for r in a.runs if (r / "tape.npz").exists() or any((r / "tape").glob("[0-9]*.npz"))]
    if not runs:
        raise ValueError("no recorded telemetry among those paths")
    manifest = fit.robot_of(runs) if not a.body else (
        bodies.manifests()[a.body] if a.body in bodies.manifests() else load_robot(a.body))
    model = fit.fit(runs, manifest)
    model.save(a.out)
    print(json.dumps(model.to_dict()) if a.json else
          model.describe(Chain(manifest.urdf, manifest.tool_link)) + f"\nwritten to {a.out}; use it with "
          f"`fit = \"{a.out}\"` in the workcell")
    return 0


def _vec(text: str) -> list[float]:
    try:
        v = [float(x) for x in text.replace(" ", "").split(",")]
    except ValueError:
        v = []
    if len(v) != 3:
        raise ValueError(f"expected three numbers like 0.3,0,0.14; got {text!r}")
    return v


def _steps(c: Client, step: str | None) -> str:
    """The steps a plan can use, from the running daemon (so plugins show up), else from this installation."""
    try:
        steps = c.help()
    except OSError:
        from .behaviors import REGISTRY
        steps = {kind: cls.help() for kind, cls in REGISTRY.items()}
    return views.steps_text(steps, step)


if __name__ == "__main__":
    sys.exit(main())
