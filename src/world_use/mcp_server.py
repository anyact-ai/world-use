"""MCP server: the `wu` verbs as tools, for agents that speak MCP rather than a shell.

    wu mcp                                   # stdio; the daemon must already be running (wu up)
    claude mcp add world-use -- wu mcp       # for example, in Claude Code

It is a thin layer on the daemon's client, like the CLI, and returns the same short text; `look` returns the
picture itself.
"""
import json
from collections.abc import Callable

from .cli import help_text, job_text
from .client import DEFAULT_URL, Client, DaemonError

INSTRUCTIONS = (
    "You drive a robot. A kernel underneath plans, checks and executes every motion and holds still whenever "
    "something unexpected happens. Read `card` once. Plan a phase as a list of steps (`help` lists them) and `run` "
    "it: it is rehearsed first, and if the kernel would refuse any step, nothing moves and every problem is listed. "
    "Answer checkpoints with `answer`; `look` at a camera when you need to see. Distances are metres in the work "
    "frame: forward, left, up. The robot is slow to heat and fast to act: think with torque off, act in phases.")


def build(url: str = DEFAULT_URL):
    from mcp.server.mcpserver import Image, MCPServer

    c = Client(url)
    server = MCPServer("world-use", instructions=INSTRUCTIONS)

    def call(fn: Callable[[], object]):
        """Errors come back as text a model can act on, never as a failed tool call."""
        try:
            return fn()
        except DaemonError as e:
            refused = e.body.get("refused")
            if refused:
                return f"refused: {refused['message']}" + (f" (hint: {refused['hint']})" if refused.get("hint") else "")
            return f"error: {e}"
        except OSError as e:
            return f"cannot reach the daemon at {url} ({e}); start it with: wu up"

    @server.tool()
    def card() -> str:
        """What this robot is and can do: joints, gripper, which way the tool points, the frames, the surfaces and
        objects the kernel knows, cameras, and which short moves are possible from here. Read it once."""
        return call(c.card)

    @server.tool()
    def status() -> str:
        """One line: the running job, the tool position (work frame), gripper, joint torques, the hottest motor."""
        return call(lambda: c.status()["line"])

    @server.tool()
    def run(plan: list[dict] | dict | None = None, wait_s: float = 60.0, rehearse: bool = True,
            checked: bool = False) -> str:
        """Run a plan: a list of steps, e.g. [{"do": "line", "up": 0.05}, {"do": "grip", "expect_mm": [35, 45]}].
        Rehearsed on a twin first; if any step would break a limit nothing moves and every problem is listed.
        checked=true runs the plan the last `check` rehearsed. Returns the outcome and the state line, or the
        question a checkpoint is waiting on."""
        return call(lambda: job_text(c.run(plan, wait=wait_s, check=rehearse, checked=checked)))

    @server.tool()
    def check(plan: list[dict] | dict) -> str:
        """Rehearse a plan on a twin without running it: time, contacts, heat and every limit it would break."""
        return call(lambda: c.check(plan)["text"])

    @server.tool()
    def answer(job: int, answer: str, wait_s: float = 60.0) -> str:
        """Answer the question a checkpoint is waiting on; the expected answer (usually "yes") carries on."""
        return call(lambda: job_text(c.answer(job, answer, wait=wait_s)))

    @server.tool()
    def look(camera: str | None = None, plan: list[dict] | dict | None = None, grid: bool = False):
        """A picture from a camera (default: the first), with the tool point, the work axes and the boxes the kernel
        knows drawn on it; given a plan, its rehearsed tool path too. grid: a pixel ruler and nothing else, for
        reading off where something is (as calibrating asks)."""
        def shot():
            r = c.look(camera, plan, grid)
            text = f"{r['camera']} camera: {r['drawn']}" + (f"\n{r['check']}" if r.get("check") else "")
            return [text, Image(path=r["path"])]
        return call(shot)

    @server.tool()
    def world() -> str:
        """Everything the kernel knows about the scene: frames, boxes (work frame) and facts, with their sources."""
        return call(lambda: c.world()["text"])

    @server.tool()
    def add_box(name: str, kind: str, center: list[float], size: list[float], yaw_deg: float = 0.0,
                source: str = "policy", grip_width: float | None = None,
                speed: float | None = None, dtau: float | None = None) -> str:
        """Tell the kernel about something you see. kind: surface (a table; plans may not pass through it), object
        (a thing to grip; grip_width in metres), keep_out, fragile or slow. center [forward, left, up] and size
        [forward, left, up] in metres, work frame. slow requires speed (planned tool speed in m/s); fragile
        accepts dtau (contact threshold in Nm). Guarded moves and every plan check use it from then on."""
        extra = {key: value for key, value in dict(grip_width=grip_width, speed=speed, dtau=dtau).items()
                 if value is not None}
        return call(lambda: c.box(name, kind, center, size, yaw_deg=yaw_deg, source=source, **extra)["line"])

    @server.tool()
    def remove_box(name: str) -> str:
        """Forget a box the kernel knows."""
        return call(lambda: c.remove(name)["line"])

    @server.tool()
    def fact(key: str, value: str, source: str) -> str:
        """Record something you measured or saw, with where it came from; it goes stale after a surprise."""
        def record():
            try:
                v = json.loads(value)
            except json.JSONDecodeError:
                v = value
            c.world(fact=dict(key=key, value=v, source=source))
            return f"recorded {key} = {v}"
        return call(record)

    @server.tool()
    def help(step: str | None = None) -> str:
        """The steps a plan can use, with an example each; with a step name, its parameters."""
        return call(lambda: help_text(c, step))

    @server.tool()
    def stop(reason: str = "stop requested") -> str:
        """Stop now: the arm holds where it is and anything queued is cancelled."""
        return call(lambda: c.stop(reason)["line"])

    @server.tool()
    def home_route(steps: list[dict], note: str = "") -> str:
        """Set the way home from here, checked against what you can see: [] folds straight home; otherwise the
        moves that get clear first. It goes stale when anything is touched."""
        return call(lambda: f"home: {c.home_route(steps, note)['home']}")

    @server.tool()
    def home(wait_s: float = 60.0) -> str:
        """Go home along the home route, then fold to the rest pose."""
        return call(lambda: job_text(c.home(wait=wait_s)))

    @server.tool()
    def enable() -> str:
        """Switch torque on (the arm holds where it is). Every held second heats the motors."""
        return call(lambda: c.enable()["line"])

    @server.tool()
    def release() -> str:
        """Switch torque off; only allowed at the rest pose."""
        return call(lambda: c.release()["line"])

    @server.tool()
    def events(since: int = 0) -> str:
        """What happened since event number `since`: contacts, grips, warnings, questions."""
        def recent():
            r = c.events(since)
            return "\n".join(f"[{e['seq']}] {e['t']:>7.1f}s {e['level']:5s} {e['kind']}: {e['message']}"
                             for e in r["events"][-40:]) or "(none)"
        return call(recent)

    @server.tool()
    def calibrate(camera: str, points: int = 8, wait_s: float = 60.0) -> str:
        """Find where a camera is from the arm: the tool visits the corners of a box, and at each a question asks
        where the tool point is in `look(camera, grid=True)`; answer x,y pixels (or unseen) with `answer`. The
        reply to the last answer has the fit, installed if it is good, and the workcell lines to keep it."""
        return call(lambda: job_text(c.calibrate(camera, points, None, wait_s)))

    @server.tool()
    def record() -> str:
        """Write the flight record so far (tape, summary, world) without stopping anything; returns where."""
        def saved():
            r = c.record()
            s = r["summary"]
            return (f"{r['run']}: powered {s.get('powered_s', 0)} s, moving {s.get('moving_s', 0)} s, "
                    f"max temps {s.get('max_temp_c')}")
        return call(saved)

    return server


def serve(url: str = DEFAULT_URL):
    build(url).run("stdio")
