"""World-use's agent for RoboWild-Harness: the harness's own ReAct loop, with the hand tools added.

The harness runs it as a `command` backend, once per attempt, in the task workspace, with the attempt prompt on stdin.
It is the `react` backend's loop unchanged (instructions, one tool call per turn, images after each result, context
elision, retries), so it differs from that baseline only in the extra tools and their help text. Robot API tools pass
through as they do there; grasp, place, release_placement and push run world_use.hand_tools sequences through
`RoboWildArm`, whose moves count toward the same call budget.

    {"backend": "command", "command": ["<venv>/bin/python", "<world-use>/examples/robowild/agent.py",
                                       "--model", "MODEL", "--base-url", "URL", "--api-key-env", "NAME"]}

The interpreter needs world-use, robowild-harness and its robot-sdk package; see README.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from robot_sdk import RobotError
from robowild_harness.agents.common import AgentConfig
from robowild_harness.agents.react import ReactRunner, _call_id, _compact, _reject_constant
from robowild_harness.runtime.batch import _REACT_DEVELOPER_INSTRUCTIONS, _attempt_prompt

from world_use.hand_tools import HandTools

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm import HAND, MAX_TILT_DEG, NoTracking, RoboWildArm

TOOLS = ("grasp", "place", "release_placement", "push")
POINT = dict(type="object", description="{x, y, z} in metres, in robot_base.")
TILT = dict(
    type="function", name="tilt", strict=False,
    description=(
        "Turn the held object about pivot, a point of it that should stay where it is, such as a container's mouth "
        "over where it pours, until the hand leans tilt_deg from upright with its fingertips toward a level point. "
        "With over, the hand first carries the pivot there. It turns at most 30 degrees per move_robot call, with "
        "the grip monitored, and reports where the pivot ended. tilt_deg 0 stands the hand upright again about the "
        "same point. The finger axis ends level and across the lean."
    ),
    parameters=dict(type="object", required=["object_target", "pivot", "toward", "tilt_deg", "expect_grip_mm"],
                    properties=dict(
                        object_target=dict(type="string"),
                        pivot=POINT,
                        over=POINT,
                        toward=dict(type="object", description="{x, y}: the level point the fingertips lean toward."),
                        tilt_deg=dict(type="number", description="Lean of the hand from upright, 0 to 150."),
                        expect_grip_mm=dict(type="array", items=dict(type="number"), minItems=2, maxItems=2,
                                            description="The measured opening that means the object is still held."),
                    )),
)
# The loop keeps the images of its newest IMAGE_HISTORY image-bearing messages and drops older ones' images as new
# ones come. Azure OpenAI's prompt cache ends at the first message whose images were dropped, so dropping one per
# move made every request after a move pay for its whole context: in 01-01, 25 of 35 requests had only the
# instructions cached. Dropped IMAGE_BATCH at a time, the cache breaks once in that many moves.
IMAGE_BATCH = 6
# Arguments that need region tracking or a camera on the hand, which this robot's tools do not have.
UNSUPPORTED = {"grasp": ("select", "offset"), "place": ("select_destination", "destination_target")}
EMBODIMENT_DOCS = (
    "Positions for these tools are x, y, z in the current robot_base frame, the frame of move_robot and of "
    "robot_state: get_camera_point_3d gives them from a pixel. The base moves that frame with it. Each tool runs "
    "several move_robot calls, which count toward the call budget, and returns what was measured, ending with the "
    "budget's count after it in _harness as a Robot API result does: the arm's move_robot stays available for "
    "everything else, including the base and the lift. The tool point is 5 mm above the fingertips, near the bottom "
    "of the finger pads, which reach 12 mm above it. To pinch something thinner than about 1 cm, give an approach z "
    "at or a few millimetres below the surface it rests on: contact stops the fingertips there. A move counts as "
    "arrived within 2 cm of its goal, and approach_ended_short_m also reports an approach that ended more than 3 mm "
    "above its pose, where fingers may close over a thin part. A large turn of the wrist in one move_robot call can "
    "shake a held object loose: a keychain pinched by its tag fell out of a 100-degree turn. tilt turns what the "
    "hand holds in steps, about a point of it."
)


class Watched:
    """The Robot API client, handing every result to the executor so its state stays the latest."""

    def __init__(self, robot, absorb):
        self._robot, self._absorb = robot, absorb

    def call(self, name, **arguments):
        try:
            result = self._robot.call(name, **arguments)
        except RobotError as error:
            self._absorb(error.details or {})
            raise
        self._absorb(result)
        return result

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda **arguments: self.call(name, **arguments)


class WorldUseReact(ReactRunner):
    def __init__(self, config: AgentConfig, workspace: Path, trusted_root: Path):
        super().__init__(config, workspace, trusted_root, _REACT_DEVELOPER_INSTRUCTIONS)
        scene = json.loads((workspace / "latest_observation.json").read_text())
        arm = RoboWildArm(None, scene, failures=(RobotError,), record_dir=trusted_root)
        self._robot = arm.robot = Watched(self._robot, arm.absorb)
        self._hand = HandTools(arm, HAND, NoTracking())
        added = []
        for schema in self._hand.schemas():
            if schema["name"] in TOOLS:
                parameters = schema["parameters"]
                for name in UNSUPPORTED.get(schema["name"], ()):
                    parameters["properties"].pop(name)
                added.append(dict(type="function", name=schema["name"], description=schema["description"],
                                  strict=False, parameters=parameters))
        self._tools.extend([*added, TILT])
        self._hand_names = frozenset(TOOLS) | {"tilt"}
        self._instructions += f"\n\n# Hand tools\n\n{self._hand.docs()} {EMBODIMENT_DOCS}"
        self._usage = None

    def _elide_old_images(self):
        bearing = sum(1 for item in self._input if item.get("type") == "message" and item.get("role") == "user"
                      and any(isinstance(part, dict) and part.get("type") == "input_image"
                              for part in item.get("content", ())))
        if bearing > self._image_history + IMAGE_BATCH:
            super()._elide_old_images()

    def _invoke(self, turn_id, iteration, call):
        name = str(call.get("name") or "")
        if name not in self._hand_names:
            return super()._invoke(turn_id, iteration, call)
        raw = call.get("arguments")
        try:
            arguments = json.loads(raw, parse_constant=_reject_constant) if isinstance(raw, str) else raw
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must decode to a JSON object")
        except (TypeError, ValueError) as error:
            return self._refuse(turn_id, iteration, call, f"Invalid tool arguments for {name}: {error}")
        self._conversation.write(self._record(
            turn_id, f"{iteration}:call:{_call_id(call)}", f"{name}({_compact(arguments)})"))
        before = self._hand.arm.key()
        record, text = self._tilt(arguments) if name == "tilt" else self._hand.call(name, arguments)
        text = self._hand.arm.with_budget(text)
        self._hand.arm.record("world-use.jsonl", dict(tool=name, arguments=arguments, result=record))
        self._conversation.write(self._record(
            turn_id, f"{iteration}:result:{_call_id(call)}", f"{name} returned {text}"))
        moved = self._hand.arm.key() != before
        return text, self._result_images(self._hand.arm.scene) if moved else (), False

    def _tilt(self, args):
        try:
            if not isinstance(args.get("object_target"), str):
                raise ValueError("tilt needs object_target, the held object")
            tilt = args.get("tilt_deg")
            if type(tilt) not in (float, int) or not 0 <= tilt <= MAX_TILT_DEG:
                raise ValueError(f"tilt_deg is between 0 and {MAX_TILT_DEG}")
            points = []
            for key in ("pivot", "over"):
                if key in args:
                    self._hand.arm.validate_pose(args[key])
                    if args[key].keys() != {"x", "y", "z"}:
                        raise ValueError(f"{key} is x, y and z in metres")
                points.append([args[key][k] for k in "xyz"] if key in args else None)
            toward = args.get("toward")
            if not isinstance(toward, dict) or toward.keys() != {"x", "y"} or any(
                    type(v) not in (float, int) or not math.isfinite(v) for v in toward.values()):
                raise ValueError("toward is a level point: x and y in metres")
            self._hand.pending = None  # any motion voids a release checkpoint
            record = self._hand.arm.tilt(points[0], [toward["x"], toward["y"]], tilt,
                                         self._hand._range(args["expect_grip_mm"]), over=points[1])
        except (KeyError, ValueError, TypeError) as error:
            record = dict(status="refused", message=str(error))
        except RobotError as error:
            record = dict(status="stopped", message=f"the arm's controller failed during this call: {error}",
                          hint="what ran before the failure may have moved the arm: observe before going on")
        return record, json.dumps(record, separators=(",", ":"))

    # The harness reads a command agent's model usage from ROBOT_TRACE_TELEMETRY_PATH, as it records the
    # react backend's itself, so both arms' reports count tokens the same way.
    def _record_usage(self, request_id, body):
        super()._record_usage(request_id, body)
        self._usage = request_id, body.get("model"), body.get("usage")

    def _respond(self, turn_id, iteration, **options):
        start, self._usage = time.time_ns(), None
        output = super()._respond(turn_id, iteration, **options)
        path = os.environ.get("ROBOT_TRACE_TELEMETRY_PATH")
        if self._usage is not None and isinstance(self._usage[2], dict) and path:
            request_id, model, usage = self._usage
            counts = dict(input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                          cached_input_tokens=(usage.get("input_tokens_details") or {}).get("cached_tokens"),
                          reasoning_tokens=(usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
                          total_tokens=usage.get("total_tokens"))
            row = dict(version=1, request_id=request_id, phase="complete", start_utc_ns=start,
                       end_utc_ns=time.time_ns(), model=model or self._config.model, provider="openai",
                       usage={k: v for k, v in counts.items() if isinstance(v, int)})
            with open(path, "a") as stream:
                stream.write(json.dumps(row) + "\n")
        return output


def react_prompt(prompt: str) -> str:
    """The attempt prompt the react backend gets, from the one a command backend is sent."""
    found = re.match(r"Attempt (\d+)/(\d+)\.", prompt)
    attempt, attempts = (int(found[1]), int(found[2])) if found else (1, 1)
    failed = "marked the previous attempt unsuccessful" in prompt
    return _attempt_prompt(attempt, attempts, (), failed=failed, backend="react")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--image-history", type=int)
    parser.add_argument("--max-iterations", type=int)
    args = parser.parse_args()
    workspace = Path.cwd()
    telemetry = os.environ.get("ROBOT_TRACE_TELEMETRY_PATH")
    trusted = Path(telemetry).parent / "world-use" if telemetry else workspace.parent / f"{workspace.name}-world-use"
    config = AgentConfig(backend="react", model=args.model, base_url=args.base_url, api_key_env=args.api_key_env,
                         reasoning_effort=args.reasoning_effort, image_history=args.image_history,
                         max_iterations=args.max_iterations)
    runner = WorldUseReact(config, workspace, trusted)
    images = tuple(Path(image["path"]) for image in runner._hand.arm.scene.get("images", []))
    turn = runner.run(react_prompt(sys.stdin.read()), images)
    print(turn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
