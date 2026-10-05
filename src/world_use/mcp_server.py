"""MCP server: the `wu` verbs as tools, for agents that speak MCP rather than a shell.

    wu mcp                                   # stdio; the daemon must already be running (wu up)
    claude mcp add world-use -- wu mcp       # for example, in Claude Code

It is a thin layer on the daemon's client, like the CLI, and returns the same short text; `look` returns the
picture itself. Camera frames and measurements come with their pictures too.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from contextlib import asynccontextmanager
from io import BytesIO

from . import views
from .client import DEFAULT_URL, Client, DaemonError
from .errors import Refused
from .perception import rectangle

INSTRUCTIONS = ("Read the policy tool or world-use://policy resource before operating the robot, then card and status. "
                "Plans use metres in the work frame. Every powered hold heats the motors. "
                "Use job to wait for a submitted run and inspect its final outcome.")
HINTS = dict(answer="answer(job={id}, answer=...)", wait="job(job={id})")


def crop_image(image, box=None, max_side=1024):
    """A box [left, top, right, bottom] of native pixels scaled so its longer side is max_side, and the matrix that
    maps the result's pixel edges back to native ones: native = native_from_image @ [x, y, 1]."""
    if isinstance(max_side, bool) or not isinstance(max_side, int) or not 64 <= max_side <= 2048:
        raise ValueError("max_side must be in 64..2048")
    bounds = [0, 0, image.width, image.height] if box is None else list(box)
    rectangle(bounds, image.size)
    crop = image.crop(bounds)
    scale = max_side / max(crop.size)
    out = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))))
    return out, dict(size=list(out.size), native_size=list(image.size), crop=bounds,
                     native_from_image=[[crop.width / out.width, 0, bounds[0]],
                                        [0, crop.height / out.height, bounds[1]], [0, 0, 1]])


def build(url: str = DEFAULT_URL, *, vision=False, device="cpu", model_path=None, tracker=None):
    """vision loads EdgeTAM in its own process for select_target and observe_targets; tests and examples can pass
    any object with the same select/update/forget/close methods as tracker instead."""
    from mcp.server.mcpserver import Image, MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import CallToolResult, TextContent, ToolAnnotations

    c = Client(url)
    if vision:
        initial = c.status()
        if initial["enabled"] or initial["power_uncertain"]:
            raise Refused("load vision before enabling the arm", "not_ready")
        from .vision_worker import TrackerProcess
        tracker = TrackerProcess(device=device, model_path=model_path)
    tracking = None
    if tracker is not None:
        from .tracking import Tracking
        tracking = Tracking(c, tracker)

    @asynccontextmanager
    async def lifespan(_):
        try:
            yield {}
        finally:
            if tracking is not None:
                await asyncio.to_thread(tracking.close)

    server = MCPServer("world-use", instructions=INSTRUCTIONS, lifespan=lifespan)

    def result(data, text=None, images=()):
        """Text, the same data as structured content, and pictures: PIL images or paths of saved ones."""
        content = [TextContent(type="text", text=json.dumps(data) if text is None else text)]
        for image in images:
            if isinstance(image, str):
                content.append(Image(path=image).to_image_content())
            else:
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                content.append(Image(data=buffer.getvalue(), format="png").to_image_content())
        return CallToolResult(content=content, structured_content=data)

    def measured(measurements: list[dict], data: dict):
        return result(data, images=[m["image"] for m in measurements if m.get("image")])

    def job_reply(data, camera=None, depth=False):
        images = []
        text = views.job_text(data, **HINTS)
        if camera is not None and data.get("status") not in ("queued", "running"):
            try:
                frame = c.frame(camera, depth=depth)
                data["frame"] = dict(id=frame.id, camera=frame.camera, size=list(frame.image.size),
                                     depth=frame.depth is not None, age_s=round(frame.age_s, 2))
                text += f"\nframe: {json.dumps(data['frame'])}"
                images.append(frame.image)
            except Exception as e:
                # Motion has already been submitted. A camera failure must preserve its job/outcome.
                data["camera_error"] = str(e)
                text += f"\ncamera unavailable: {e}; the job above is unchanged"
        return result(data, text, images)

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
    def job(job: int, wait_s: float = 60.0, camera: str | None = None, depth: bool = False):
        """Wait for a run to finish or ask a question; returns its outcome, even after completion.
        camera adds a fresh picture and frame id at a checkpoint or outcome; depth adds simulated metric depth."""
        return call(lambda: job_reply(c.job(job, wait=wait_s), camera, depth))

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def reset() -> str:
        """Clear a fault after the operator resolves it; unconfirmed motor power still blocks reset."""
        return call(lambda: c.reset()["line"])

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def shutdown():
        """Release at rest, save the record, and stop the daemon. Refused while raised or busy."""
        def stop_daemon():
            data = c.shutdown()
            return result(data, f"daemon stopped; record {data['run'] or '(none)'}; "
                                f"{views.record_line(data['summary'])}")
        return call(stop_daemon)

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def card():
        """What this robot is and can do: joints, gripper, which way the tool points, the frames, the surfaces and
        objects the kernel knows, cameras (and which give depth), and which short moves are possible from here.
        Read it once."""
        def describe():
            data = c.capabilities()
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
    def run(plan: list[dict] | dict, wait_s: float = 60.0, rehearse: bool = True, requires: list[dict] | None = None,
            camera: str | None = None, depth: bool = False):
        """Run a plan: a list of steps, e.g. [{"do": "line", "up": 0.05}, {"do": "grip", "expect_mm": [35, 45]}].
        Rehearsed on a twin first; if any step would break a limit nothing moves and every problem is listed.
        Returns the outcome and the state line, or the question a checkpoint is waiting on.
        requires: [{"evidence": measurement id, "max_age_s": seconds}]; the run is refused, or stops before its
        next step, once a measurement is older than that or its camera was calibrated again.
        camera adds a fresh picture and frame id at a checkpoint or outcome; depth adds simulated metric depth."""
        return call(lambda: job_reply(c.run(plan, wait=wait_s, check=rehearse, requires=requires), camera, depth))

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def check(plan: list[dict] | dict):
        """Rehearse a plan on a twin without running it: time, contacts, heat and every limit it would break."""
        def rehearse():
            data = c.check(plan)
            return result(data, data["text"])
        return call(rehearse)

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def answer(job: int, answer: str, wait_s: float = 60.0, camera: str | None = None, depth: bool = False):
        """Answer the question a checkpoint is waiting on; the expected answer (usually "yes") carries on.
        camera adds a fresh picture and frame id at the next checkpoint or outcome; depth adds simulated depth."""
        return call(lambda: job_reply(c.answer(job, answer, wait=wait_s), camera, depth))

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
        """A camera's picture at native resolution, with nothing drawn on it, and its frame id for measure_pixels.
        depth: also capture aligned metric depth (simulated cameras), which measuring needs."""
        def capture():
            frame = c.frame(camera, depth=depth)
            return result(dict(frame=frame.id, camera=frame.camera, size=list(frame.image.size),
                               depth=frame.depth is not None, age_s=round(frame.age_s, 2)), images=[frame.image])
        return call(capture)

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def measure_pixels(frame: str, point: list[float] | None = None, box: list[int] | None = None,
                       target: str | None = None):
        """Measure the visible surface under a point [x, y] or a box [left, top, right, bottom] (right and bottom
        exclusive) of a camera_frame, in its native pixels; the frame needs depth. Returns an id, valid and
        reason, and in work-frame metres: surface_center (of what the camera sees, not of a hidden object),
        visible_bounds and from_tool (surface_center minus the tool point). in_tool expresses that surface in
        the captured tool's axes, in metres: compare the same visible feature before/after a lift or rotation.
        The measured pixels are drawn on the picture. Pass the id to run(requires=...) for freshness checks."""
        def measure():
            data = c.measure(frame, point=point, box=box, target=target)
            return measured([data], data)
        return call(measure)

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def inspect_image(frame: str, crop: list[int] | None = None, max_side: int = 1024):
        """A camera_frame again, optionally cropped to [left, top, right, bottom] native pixels and scaled to
        max_side; native_from_image maps its pixels back to the frame's native pixels for measure_pixels."""
        def inspect():
            source = c.frame(id=frame)
            image, mapping = crop_image(source.image, crop, max_side)
            return result(dict(mapping, frame=source.id, camera=source.camera, age_s=round(source.age_s, 2)),
                          images=[image])
        return call(inspect)

    if tracking is not None:
        @server.tool(annotations=ToolAnnotations(read_only_hint=False))
        def select_target(frame: str, target: str, point: list[float] | None = None, box: list[int] | None = None):
            """Select an object by a point or box in a camera_frame, name it target, and measure it as
            measure_pixels does. observe_targets measures it again later without selecting. A frame older than the
            tracker accepts is followed into a new one first. tracking says tracked or lost."""
            def select():
                data = tracking.select(frame, target, point=point, box=box)
                return measured([data], data)
            return call(select)

        @server.tool(annotations=ToolAnnotations(read_only_hint=False))
        def observe_targets(targets: list[str]):
            """Measure selected targets again in one new frame per camera. A lost target is dropped and its
            measurements are withdrawn, so runs that require them stop; select it again to continue."""
            def observe():
                data = tracking.observe(targets)
                return measured(data, dict(measurements=data))
            return call(observe)

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
    def home(wait_s: float = 60.0):
        """Go home along the home route, then fold to the rest pose."""
        return call(lambda: job_reply(c.home(wait=wait_s)))

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
    def calibrate(camera: str, points: int = 8, wait_s: float = 60.0):
        """Find where a camera is from the arm: the tool visits the corners of a box, and at each a question asks
        where the tool point is in `look(camera, grid=True)`; answer x,y pixels (or unseen) with `answer`. The
        reply to the last answer has the fit, installed if it is good, and the workcell lines to keep it."""
        return call(lambda: job_reply(c.calibrate(camera, points, None, wait_s)))

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
