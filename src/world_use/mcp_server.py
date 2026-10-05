"""MCP server: the `wu` verbs as tools, for agents that speak MCP rather than a shell.

    wu mcp                                   # stdio; the daemon must already be running (wu up)
    claude mcp add world-use -- wu mcp       # for example, in Claude Code

It is a thin layer on the daemon's client, like the CLI, and returns the same short text; `look` returns the
picture itself.
"""
from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Literal

from . import views
from .client import DEFAULT_URL, Client, DaemonError
from .errors import Refused
from .observations import Perception, crop_image

INSTRUCTIONS = ("Read the policy tool or world-use://policy resource before operating the robot, then card and status. "
                "Plans use metres in the work frame. Every powered hold heats the motors. "
                "Use job to wait for a submitted run and inspect its final outcome.")
HINTS = dict(answer="answer(job={id}, answer=...)", wait="job(job={id})")


def build(url: str = DEFAULT_URL, *, vision=False, device="cpu", model_path=None, perception=None):
    from mcp.server.mcpserver import Image, MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import CallToolResult, TextContent, ToolAnnotations

    c = Client(url)
    if vision:
        initial = c.status()
        if initial["enabled"] or initial["power_uncertain"]:
            raise Refused("load vision before enabling the arm", "not_ready")
        from .vision_worker import TrackerProcess
        perception = Perception(c, TrackerProcess(device=device, model_path=model_path))
    p = perception if perception is not None else Perception(c)

    @asynccontextmanager
    async def lifespan(_):
        try:
            yield {}
        finally:
            await asyncio.to_thread(p.close)

    server = MCPServer("world-use", instructions=INSTRUCTIONS, lifespan=lifespan)

    def result(data, text=None, image=None):
        content = [TextContent(type="text", text=json.dumps(data) if text is None else text)]
        if image is not None:
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            content.append(Image(data=buffer.getvalue(), format="png").to_image_content())
        return CallToolResult(content=content, structured_content=data)

    def job_reply(data):
        return result(data, views.job_text(data, **HINTS))

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
        except (Refused, ValueError, KeyError) as e:
            raise ToolError(str(e)) from e

    @server.resource("world-use://policy")
    def policy_resource() -> str:
        from . import policy_text
        return policy_text()

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def policy() -> str:
        """The complete installed operating brief. Read before operating a robot."""
        return policy_resource()

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def job(job: int, wait_s: float = 60.0):
        """Wait for a run to finish or ask a question; returns its outcome, even after completion."""
        return call(lambda: job_reply(c.job(job, wait=wait_s)))

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def reset() -> str:
        """Clear a fault after the operator resolves it; unconfirmed motor power still blocks reset."""
        return call(lambda: c.reset()["line"])

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def shutdown():
        """Release at rest, save the record, and stop the daemon. Refused while raised or busy."""
        def stop_daemon():
            data = c.shutdown()
            return result(data, "daemon stopped; " + views.record_line(data["summary"]))
        return call(stop_daemon)

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def card():
        """What this robot is and can do: joints, gripper, which way the tool points, the frames, the surfaces and
        objects the kernel knows, cameras, and which short moves are possible from here. Read it once."""
        def describe():
            data = c.capabilities()
            data["perception"] = dict(configured=p.tracker is not None,
                                      ready=p.tracker is not None and getattr(p.tracker, "ready", True),
                                      max_targets=p.MAX_TARGETS)
            return result(data, data["card"])
        return call(describe)

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def status():
        """The one-line status; session, power, job and feedback state come as structured content."""
        def describe():
            data = c.status()
            return result(data, data["line"])
        return call(describe)

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def run(plan: list[dict] | dict | None = None, wait_s: float = 60.0, rehearse: bool = True,
            requires: list[dict] | None = None, plan_id: str | None = None, request_id: str | None = None):
        """Run a plan: a list of steps, e.g. [{"do": "line", "up": 0.05}, {"do": "grip", "expect_mm": [35, 45]}].
        Rehearsed on a twin first; if any step would break a limit nothing moves and every problem is listed.
        Returns the outcome and the state line, or the
        question a checkpoint is waiting on.
        Evidence prerequisites are [{"evidence": receipt_id, "max_age_s": seconds}]; they apply even without rehearsal.
        Supply plan OR plan_id. A prepared ID executes once; retrying it returns its job. Literal-plan retries
        need the same request_id, formed as status.session_id + ':' + a unique value. Completion is not verification.
        """
        def execute():
            data = c.run(plan, wait=wait_s, check=rehearse, requires=requires, plan_id=plan_id, request_id=request_id)
            return job_reply(data)
        return call(execute)

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def check(plan: list[dict] | dict, requires: list[dict] | None = None,
              max_age_s: float | None = None, effects: list[dict] | None = None):
        """Resolve geometry references and rehearse without motion. Returns a prepared ID for run(plan_id=...).
        move_to accepts {geometry: ID, component: center, offset_m: [x,y,z], offset_frame: work} in to;
        references require max_age_s. Freeze optional lift/placement criteria from card.effects before acting."""
        def prepare():
            data = c.check(plan, prepare=True, requires=requires, max_age_s=max_age_s, effects=effects)
            return result(data, data["text"])
        return call(prepare)

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def answer(job: int, answer: str, wait_s: float = 60.0):
        """Answer the question a checkpoint is waiting on; the expected answer (usually "yes") carries on."""
        return call(lambda: job_reply(c.answer(job, answer, wait=wait_s)))

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def look(camera: str | None = None, plan: list[dict] | dict | None = None, grid: bool = False):
        """A picture from a camera (default: the first), with the tool point, the work axes and the boxes the kernel
        knows drawn on it; given a plan, its rehearsed tool path too. grid: a pixel ruler and nothing else, for
        reading off where something is (as calibrating asks)."""
        def shot():
            r = c.look(camera, plan, grid)
            text = f"{r['camera']} camera: {r['drawn']}" + (f"\n{r['check']}" if r.get("check") else "")
            return [text, Image(path=r["path"])]
        return call(shot)

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def camera_frame(camera: str | None = None, depth: bool = False):
        """Inspect native upright RGB pixels. Request aligned metric depth for measurement (MuJoCo cameras).
        Returns a frame ID for measure_pixels. Captures are bounded; recapture if an old ID has expired."""
        def capture():
            frame = p.capture(camera, depth=depth)
            return result(dict(frame=frame.id, camera=frame.camera, size=frame.image.size,
                               depth=frame.depth is not None, age_s=frame.age_s, session=frame.session,
                               calibration=frame.calibration), image=frame.image)
        return call(capture)

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def measure_pixels(frame: str, point: list[float] | None = None, box: list[int] | None = None,
                       target: str | None = None):
        """Measure a point or rectangular visible region from camera_frame; register its source evidence.
        Coordinates are native pixels; box is [left, top, right, bottom], right/bottom exclusive.
        A region must contain one visible surface. Geometry is in base-frame metres, not an object pose.
        Receipt ID can be used in run.requires. World assertions remain explicit add_box/fact calls."""
        return call(lambda: result(p.measure_pixels(frame, point=point, box=box, target=target)))

    if p.tracker is not None:
        @server.tool(annotations=ToolAnnotations(read_only_hint=False))
        def select_target(frame: str, point: list[float] | None = None, box: list[int] | None = None,
                          label: str | None = None, replace_target: str | None = None):
            """Segment one native point/box selection. Returns target ID, evidence and exact-frame mask preview.
            The label is an agent name, not a semantic prompt. Replacement invalidates the old target's plans."""
            def select():
                data = p.select_target(frame, point=point, box=box, label=label, replace_target=replace_target)
                preview = None
                if data["status"] == "tracked":
                    preview, mapping = p.inspect_image(frame, target=data["target"])
                    data["preview"] = mapping
                return result(data, image=preview)
            return call(select)

        @server.tool(annotations=ToolAnnotations(read_only_hint=False))
        def observe_targets(targets: list[str], depth: bool = True, frames: dict[str, str] | None = None):
            """Refresh 1..4 selected targets, capturing once per camera, or use explicit camera:frame IDs.
            Reports masks and metric evidence separately. Lost targets need reselection; no background tracking."""
            return call(lambda: result(p.observe_targets(targets, depth=depth, frames=frames)))

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def inspect_image(frame: str | None = None, evidence: str | None = None, crop: list[int] | None = None,
                      target: str | None = None, max_side: int = 1024):
        """Inspect a cached capture OR archived evidence image, with exact native-pixel crop mapping.
        Optional target overlay must match the live capture. Archived evidence stays historical."""
        def inspect():
            if (frame is None) == (evidence is None):
                raise ValueError("provide frame or evidence, not both")
            if frame is not None:
                image, metadata = p.inspect_image(frame, box=crop, target=target, max_side=max_side)
            else:
                from PIL import Image as PILImage
                assert evidence is not None
                if target is not None:
                    raise ValueError("live target overlays cannot be drawn on archived evidence")
                saved = c.evidence_image(evidence)
                source = PILImage.open(BytesIO(base64.b64decode(saved["png"]))).convert("RGB")
                image, metadata = crop_image(source, box=crop, max_side=max_side)
                metadata.update(evidence=evidence, source=saved["metadata"], historical=True)
            return result(metadata, image=image)
        return call(inspect)

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def fit_geometry(evidence: str, kind: Literal["known_box", "plane", "axis"] = "known_box", frame: str = "work",
                     size_m: list[float] | None = None, max_residual_m: float = .003):
        """Fit registered support: known_box (explicit dimensions, upright visible top), plane, or axis.
        Returns base-metre geometry IDs for check. Fits expose assumptions; directions can be unsigned."""
        return call(lambda: result(c.fit_geometry(evidence, kind=kind, frame=frame, size_m=size_m,
                                                  max_residual_m=max_residual_m)))

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def verify_effect(job: int, after: str, effect: int = 0):
        """Evaluate a job's predeclared effect using a new geometry ID and captured tool/gripper feedback.
        Returns pass/fail/unknown with evidence. Does not move, rewrite criteria or consult simulator truth."""
        return call(lambda: result(c.verify_effect(job, after, effect=effect)))

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def inspect_run(since: int = 0, limit: int = 50, job: int | None = None):
        """Inspect this session's committed history plus live tail, with event cursor and visible record gaps.
        Includes plans, outcomes, evidence and verification; use inspect_image for archived evidence pixels."""
        return call(lambda: result(c.inspect_run(since=since, limit=limit, job=job)))

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def world():
        """Everything the kernel knows about the scene: frames, boxes (work frame) and facts, with their sources."""
        def describe():
            data = c.world()
            return result(data, data["text"])
        return call(describe)

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def add_box(name: str, kind: str, center: list[float], size: list[float], yaw_deg: float = 0.0,
                source: str = "policy", grip_width: float | None = None,
                speed: float | None = None, dtau: float | None = None, frame: str = "work") -> str:
        """Tell the kernel about something you see. kind: surface (a table; plans may not pass through it), object
        (a thing to grip; grip_width in metres), keep_out, fragile or slow. center [forward, left, up] and size
        [forward, left, up] in metres, work frame. slow requires speed (planned tool speed in m/s); fragile
        accepts dtau (contact threshold in Nm). Guarded moves and every plan check use it from then on."""
        extra = {key: value for key, value in dict(grip_width=grip_width, speed=speed, dtau=dtau).items()
                 if value is not None}
        return call(lambda: c.box(name, kind, center, size, frame=frame, yaw_deg=yaw_deg,
                                  source=source, **extra)["line"])

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def remove_box(name: str) -> str:
        """Forget a box the kernel knows."""
        return call(lambda: c.remove(name)["line"])

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def fact(key: str, value: str, source: str) -> str:
        """Record something you measured or saw, with where it came from; it goes stale after a surprise.
        value is JSON (0.205, true, [1, 2]) or text."""
        return call(lambda: c.world(fact=dict(key=key, value=views.parse_value(value), source=source))["line"])

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def help(step: str | None = None):
        """The steps a plan can use, with an example each; with a step name, its parameters."""
        def describe():
            steps = c.help()
            text = views.steps_text(steps, step)
            return result(dict(steps=steps if not step else {step: steps[step]}), text)
        return call(describe)

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def stop(reason: str = "stop requested") -> str:
        """Stop now: the arm holds where it is and anything queued is cancelled."""
        return call(lambda: c.stop(reason)["line"])

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def home_route(steps: list[dict] | None, note: str = "") -> str:
        """Set the way home from here, checked against what you can see: [] folds straight home; otherwise the
        moves that get clear first. Only motion and gripper steps; no checkpoints or holds. null clears a route
        when the scene changes. It goes stale when anything is touched. The reply says whether the way home,
        rehearsed from here, would pass."""
        return call(lambda: views.home_text(c.home_route(steps, note)))

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def home(wait_s: float = 60.0, request_id: str | None = None):
        """Go home along the home route, then fold to the rest pose."""
        return call(lambda: job_reply(c.home(wait=wait_s, request_id=request_id)))

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def enable() -> str:
        """Switch torque on (the arm holds where it is). Every held second heats the motors."""
        return call(lambda: c.enable()["line"])

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def release() -> str:
        """Switch torque off; only allowed at the rest pose."""
        return call(lambda: c.release()["line"])

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def events(since: int = 0, limit: int = 40):
        """A bounded page after event number since; continue from last while more is true. Gaps stay visible."""
        def recent():
            if not 1 <= limit <= 100:
                raise ValueError("limit must be in 1..100")
            r = c.events(since, limit=limit)
            text = "\n".join(map(views.event_line, r["events"])) or "(none)"
            return result(r, text + (f"\nmore after [{r['last']}]: events(since={r['last']})" if r["more"] else ""))
        return call(recent)

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def calibrate(camera: str, points: int = 8, wait_s: float = 60.0, request_id: str | None = None):
        """Find where a camera is from the arm: the tool visits the corners of a box, and at each a question asks
        where the tool point is in `look(camera, grid=True)`; answer x,y pixels (or unseen) with `answer`. The
        reply to the last answer has the fit, installed if it is good, and the workcell lines to keep it."""
        return call(lambda: job_reply(c.calibrate(camera, points, None, wait_s, request_id=request_id)))

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def record(note: str = "", context: dict | None = None) -> str:
        """Write the flight record so far (tape, summary, world) without stopping anything; returns where."""
        def saved():
            r = c.record(note=note, context=context)
            return f"{r['run'] or '(no run folder)'}: {views.record_line(r['summary'])}"
        return call(saved)

    return server


def serve(url: str = DEFAULT_URL, **options):
    build(url, **options).run("stdio")
