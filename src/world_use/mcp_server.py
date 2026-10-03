"""MCP server: the `wu` verbs as tools, for agents that speak MCP rather than a shell.

    wu mcp                                   # stdio; the daemon must already be running (wu up)
    claude mcp add world-use -- wu mcp       # for example, in Claude Code

It is a thin layer on the daemon's client, like the CLI, and returns the same short text; `look` returns the
picture itself.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from collections.abc import Callable
from io import BytesIO

from .cli import help_text, job_text
from .client import DEFAULT_URL, Client, DaemonError

INSTRUCTIONS = ("Read the policy tool or world-use://policy resource before operating the robot, then card and status. "
                "Plans use metres in the work frame. Every powered hold heats the motors. "
                "Use job to wait for a submitted run and inspect its final outcome.")


def build(url: str = DEFAULT_URL):
    from mcp.server.mcpserver import Image, MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    c = Client(url)
    server = MCPServer("world-use", instructions=INSTRUCTIONS)
    frames = OrderedDict()
    frame_lock = threading.Lock()

    def call(fn: Callable[[], object]):
        """Preserve the daemon's error as an MCP tool error, with actionable text."""
        try:
            return fn()
        except DaemonError as e:
            refused = e.body.get("refused")
            if refused:
                raise ToolError(f"refused: {refused['message']}" + (
                    f" (hint: {refused['hint']})" if refused.get("hint") else "")) from e
            raise ToolError(f"error: {e}") from e
        except OSError as e:
            raise ToolError(f"cannot reach the daemon at {url} ({e}); start it with: wu up") from e

    @server.resource("world-use://policy")
    def policy_resource() -> str:
        from . import policy_text
        return policy_text()

    @server.tool()
    def policy() -> str:
        """The complete installed operating brief. Read before operating a robot."""
        return policy_resource()

    @server.tool()
    def job(job: int, wait_s: float = 60.0) -> dict | str:
        """Wait for a run to finish or ask a question; returns its structured outcome, even after completion."""
        return call(lambda: c.job(job, wait=wait_s))

    @server.tool()
    def reset() -> str:
        """Clear a fault after the operator resolves it; unconfirmed motor power still blocks reset."""
        return call(lambda: c.reset()["line"])

    @server.tool()
    def shutdown() -> dict | str:
        """Release at rest, save the record, and stop the daemon. Refused while raised or busy."""
        return call(c.shutdown)

    @server.tool()
    def card() -> str:
        """What this robot is and can do: joints, gripper, which way the tool points, the frames, the surfaces and
        objects the kernel knows, cameras, and which short moves are possible from here. Read it once."""
        return call(c.card)

    @server.tool()
    def status() -> dict | str:
        """Structured session, power, job and feedback state, plus a concise human-readable line."""
        return call(c.status)

    @server.tool()
    def run(plan: list[dict] | dict, wait_s: float = 60.0, rehearse: bool = True,
            requires: list[dict] | None = None) -> str:
        """Run a plan: a list of steps, e.g. [{"do": "line", "up": 0.05}, {"do": "grip", "expect_mm": [35, 45]}].
        Rehearsed on a twin first; if any step would break a limit nothing moves and every problem is listed.
        Returns the outcome and the state line, or the
        question a checkpoint is waiting on.
        Evidence prerequisites are [{"evidence": receipt_id, "max_age_s": seconds}]; they apply even without rehearsal.
        """
        return call(lambda: job_text(c.run(plan, wait=wait_s, check=rehearse, requires=requires)))

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
    def camera_frame(camera: str | None = None, depth: bool = False):
        """Inspect native upright RGB pixels. Request aligned metric depth for measurement (MuJoCo cameras).
        Returns a frame ID for measure_pixels. Captures are bounded; recapture if an old ID has expired."""
        def capture():
            from .perception import EvidenceStore
            frame = c.frame(camera, depth=depth)
            with frame_lock:
                frames[frame.id] = frame
                while len(frames) > 8 or sum(EvidenceStore.size(f) for f in frames.values()) > 64 * 1024 * 1024:
                    frames.popitem(last=False)
            image = BytesIO()
            frame.image.save(image, format="PNG")
            return [json.dumps(dict(frame=frame.id, camera=frame.camera, size=frame.image.size,
                                    depth=frame.depth is not None, age_s=frame.age_s)),
                    Image(data=image.getvalue(), format="png")]
        return call(capture)

    @server.tool()
    def measure_pixels(frame: str, point: list[float] | None = None, box: list[int] | None = None,
                       target: str | None = None) -> dict:
        """Measure a point or rectangular visible region from camera_frame; register its source evidence.
        Coordinates are native pixels; box is [left, top, right, bottom], right/bottom exclusive.
        A region must contain one visible surface. Geometry is in base-frame metres, not an object pose.
        Receipt ID can be used in run.requires. World assertions remain explicit add_box/fact calls."""
        def measurement():
            import numpy as np

            from .perception import measure
            with frame_lock:
                source = frames.get(frame)
            if source is None:
                raise ToolError("frame expired; call camera_frame again")
            try:
                if (point is None) == (box is None):
                    raise ValueError("provide exactly one point or box")
                mask = None
                if box is not None:
                    if (len(box) != 4 or not all(isinstance(v, int) for v in box)
                            or not 0 <= box[0] < box[2] <= source.image.width
                            or not 0 <= box[1] < box[3] <= source.image.height):
                        raise ValueError("box must be within the native image")
                    mask = np.zeros((source.image.height, source.image.width), dtype=bool)
                    mask[box[1]:box[3], box[0]:box[2]] = True
                return c.record(evidence=measure(source, point=point, mask=mask, target=target))
            except ValueError as e:
                raise ToolError(str(e)) from e
        return call(measurement)

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
    def home_route(steps: list[dict] | None, note: str = "") -> str:
        """Set the way home from here, checked against what you can see: [] folds straight home; otherwise the
        moves that get clear first. Only motion and gripper steps; no checkpoints or holds. null clears a route
        when the scene changes. It goes stale when anything is touched."""
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
    def record(note: str = "", context: dict | None = None) -> str:
        """Write the flight record so far (tape, summary, world) without stopping anything; returns where."""
        def saved():
            r = c.record(note=note, context=context)
            s = r["summary"]
            return (f"{r['run']}: powered {s.get('powered_s', 0)} s, moving {s.get('moving_s', 0)} s, "
                    f"max temps {s.get('max_temp_c')}")
        return call(saved)

    return server


def serve(url: str = DEFAULT_URL):
    build(url).run("stdio")
