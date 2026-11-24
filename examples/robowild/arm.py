"""The hand tools' executor for RoboWild's mobile Franka, through the harness's Robot API.

Positions are in the current robot_base frame (+X forward, +Y left, +Z up), the frame of move_robot goals and of
robot_state. It moves with the base: a policy that drives the base sees its targets move with it. The tool frame
is panda_tcp, 0.107 m past the hand along its approach axis, +Z, with the fingers opening along +Y.

The harness runs the simulator frozen between calls (`--mode independent`), so the state in the latest result
is the state now: reading it costs no Robot API call. Every move is one move_robot call, which counts toward the
task's call budget like any of the policy's own calls.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from world_use import geometry
from world_use.behaviors import Outcome
from world_use.hand_tools import DOWN, Hand, tilted

ARM = "left"  # the mobile Franka's one arm, as move_robot names it
APPROACH, OPENS = (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)  # in panda_tcp
TRACKED_M = 0.02  # move_robot's own tracking limit; a command's last leg must end this near its goal
# A leg on the way, which the next leg's absolute goal corrects, may end this far off: a free move with a large turn
# of the wrist ended 2.4 cm off in the simulator, with nothing in the way.
TRANSIT_M = 0.05
# move_robot counts an approach within TRACKED_M as arrived, but grasp approaches ended up to 17 mm above their goal
# so: a pinch of a 3.5 mm part that ends this far short closes above it, and the result says by how much.
SHORT_REPORT_M = 0.003
GRIP_SLACK_MM = 1.0  # finger readings settle within this of the opening they stop at
# Empty fingers read 0.2-2.7 mm in the development check. The tools take no expected opening below 4 mm, as a
# reading there cannot be told from an empty close.
EMPTY_MM = 4.0

MAX_MOVES = 8  # move_robot calls in one move_sequence
TILT_STEP_DEG = 30  # a tilt turns the hand at most this far per move, so the pivot stays near where it was
MAX_TILT_DEG = 150  # past horizontal, as pouring out of a narrow container needs
PIVOT_REACH_M = 0.3  # the pivot is on the held object: further from the hand than this, it is not

# The Panda's fingers each travel 0-40 mm; the opening is their sum.
HAND = Hand(opening_mm=80, open_mm=(70, 80), fit_mm=70, arrive_m=TRACKED_M, hand_camera=None, fixed_camera=None,
            in_view_m=0.3, hover="15-25 cm")


def quaternion(R) -> list[float]:
    """wxyz of a rotation matrix, from its largest of trace and diagonal. A hand pointing down is a half turn,
    whose w is 0: the signs of x, y and z then come from the matrix's symmetric part, not from w."""
    R = np.asarray(R, float)
    trace = float(np.trace(R))
    if trace > 0:
        s = 2 * math.sqrt(1 + trace)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = 2 * math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] >= R[2, 2]:
        s = 2 * math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2 * math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4]
    return [float(v) for v in q]


def rotation(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _partial(turn, fraction) -> np.ndarray:
    """The rotation `fraction` of the way along turn, about its axis."""
    angle = math.acos(max(-1.0, min(1.0, (float(np.trace(turn)) - 1) / 2)))
    if angle < 1e-9:
        return np.eye(3)
    if math.pi - angle < 1e-6:
        # A half turn: its axis is the column of turn + I with the largest norm.
        sym = turn + np.eye(3)
        axis = sym[:, int(np.argmax(np.linalg.norm(sym, axis=0)))]
    else:
        axis = np.array([turn[2, 1] - turn[1, 2], turn[0, 2] - turn[2, 0], turn[1, 0] - turn[0, 1]])
    axis = axis / np.linalg.norm(axis)
    a = fraction * angle
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


class NoTracking:
    """This pilot tracks no regions: the policy gives positions, from get_camera_point_3d."""

    targets: dict = {}
    lost: dict = {}

    def select(self, *args, **kwargs):
        raise ValueError("this robot's tools track no regions")

    def observe(self, names):
        return []

    def close(self):
        pass


class RoboWildArm:
    """An Executor over the Robot API client. `scene` is the latest published scene_state; `absorb` takes each
    later result, whoever made the call."""

    opens_partly = True  # its gripper value runs from 0, open, to 1, closed

    def __init__(self, robot, scene: dict, *, failures=(), record_dir: Path | None = None):
        self.robot, self.scene = robot, scene
        # The client's error type. A rejected move becomes a refusal; one whose execution is unknown ends the call.
        self.failures = failures
        self.record_dir = record_dir
        self.accounting = {}  # the harness's call count after the latest call: robot_api_calls_used and _limit

    # -- state ------------------------------------------------------------------------------------------
    def absorb(self, result) -> None:
        """Keep the newest state and call count a call returned."""
        if not isinstance(result, dict):
            return
        if isinstance(result.get("_harness"), dict):
            self.accounting = result["_harness"]
        scene = result.get("scene_state")
        if isinstance(scene, dict) and "robot_state" in scene:
            self.scene = scene
        elif isinstance(result.get("robot_state"), dict):
            self.scene = dict(self.scene, robot_state=result["robot_state"])

    @property
    def hand(self) -> dict:
        return self.scene["robot_state"]["arms"][ARM]

    def opening_mm(self) -> float:
        return 1000 * float(sum(self.hand["gripper_positions"]))

    def measure(self):
        return dict(tool=[float(v) for v in self.hand["eef_position"]], aperture_mm=round(self.opening_mm(), 1),
                    observation=self.scene.get("sequence"))

    def position(self, observation):
        return np.asarray(observation["tool"], float)

    def pose(self):
        R = rotation(self.hand["eef_quaternion_wxyz"])
        return np.asarray(self.hand["eef_position"], float), R @ APPROACH

    def orientation(self, point, jaws=None):
        R = geometry.aim(rotation(self.hand["eef_quaternion_wxyz"]), APPROACH, OPENS, point, jaws)
        return dict(quaternion_wxyz=quaternion(R)), R @ OPENS

    def lean_refusal(self, direction, tilt_deg, path):
        # This profile has no planning call to rehearse with: a lean the arm cannot make fails at its first
        # move with the planner's reason, before the hand turns.
        return None

    def hand_look(self, frame=None):
        return None

    def cameras(self):
        return [image["name"] for image in self.scene.get("images", [])]

    def frame(self, camera):
        raise ValueError("this robot's tools track no regions")

    def validate_pose(self, targets):
        if not isinstance(targets, dict) or not targets or not targets.keys() <= {"x", "y", "z"}:
            raise ValueError("a pose is x, y and z in metres, in robot_base")
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in targets.values()):
            raise ValueError("a pose is x, y and z in metres, in robot_base")

    def fingerprint(self):
        return json.dumps(self.measure())

    def record(self, name, row):
        if self.record_dir is not None:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            with (self.record_dir / name).open("a") as stream:
                stream.write(json.dumps(row, default=str) + "\n")

    def key(self):
        return self.scene.get("sequence")  # each call that changes the world publishes a new scene; none else does

    def with_budget(self, text: str) -> str:
        """A tool's brief result with the call count after it, as the harness ends its own results."""
        if not self.accounting:
            return text
        return json.dumps(dict(json.loads(text), _harness=self.accounting), separators=(",", ":"))

    # -- commands -----------------------------------------------------------------------------------------
    def execute(self, command):
        kind = command["kind"]
        if kind == "confirm_state":
            same = command["fingerprint"] == self.fingerprint()
            return Outcome("done" if same else "refused", kind, "inspected state comparison", self.measure())
        if kind == "check_grip":
            return self._opening(command["expect_mm"], Outcome("done", kind, "measured opening", self.measure()))
        outcome = Outcome("done", kind, "nothing to do: the hand is already there", dict(self.measure(),
                                                                                           arm="arrived"))
        legs = self._legs(command)
        for index, (goal, contact) in enumerate(legs):
            outcome = self._leg(goal, kind, contact, transit=index < len(legs) - 1)
            if not outcome.ok:
                return outcome
            if command.get("monitor_mm"):
                outcome = self._opening(command["monitor_mm"], outcome, "the grip changed during the move")
                if not outcome.ok:
                    return outcome
        gripper = command["targets"].get("gripper")
        if gripper is not None and command.get("expect_mm"):
            return self._opening(command["expect_mm"], outcome)
        return outcome

    def _legs(self, command):
        """(move_robot arm goal, whether it may end on contact) for each leg of a move."""
        targets = dict(command["targets"])
        here = np.asarray(self.hand["eef_position"], float)
        goal = np.array([targets.get(k, v) for k, v in zip("xyz", here, strict=True)], float)
        if command.get("rise_m") is not None:
            goal[2] = here[2] + command["rise_m"]
        turn = {k: targets[k] for k in ("quaternion_wxyz",) if k in targets}
        gripper = {"gripper": float(targets["gripper"])} if "gripper" in targets else {}
        if gripper and not gripper["gripper"] and "opening_mm" in targets:
            gripper["gripper"] = round(1 - targets["opening_mm"] / HAND.opening_mm, 4)

        def leg(to=None, **extra):
            return dict(target=ARM, **({} if to is None else dict(position=[float(v) for v in to])), **extra)

        legs = []
        travel = command.get("travel_z")
        if travel is not None and travel - here[2] > HAND.arrive_m:
            here = np.array([here[0], here[1], travel])
            legs.append((leg(here), False))
        via = command.get("via")
        if via is not None:
            start = np.array([via.get("x", goal[0]), via.get("y", goal[1]), via.get("z", here[2])], float)
            # Level to above start, turning and opening on the way (the gripper acts on arrival), then straight
            # down to start and along the line to the goal, which may end on contact.
            legs.append((leg([start[0], start[1], here[2]], **turn, **gripper), False))
            if abs(start[2] - here[2]) > 1e-4:
                legs.append((leg(start), False))
            if np.linalg.norm(goal - start) > 1e-4:
                legs.append((leg(goal), True))
        elif np.linalg.norm(goal - here) > 1e-4 or turn:
            legs.append((leg(goal if np.linalg.norm(goal - here) > 1e-4 else None, **turn, **gripper), False))
        elif gripper:
            legs.append((leg(**gripper), False))  # the fingers open or close where the hand is
        return legs

    def _leg(self, goal, kind, contact, transit=False):
        """One move_robot call, judged against its goal."""
        try:
            result = self.robot.move_robot(goals=[goal])
        except self.failures as error:
            code = getattr(error, "code", "")
            if code in ("connection_closed", "internal_error"):  # whether it moved is unknown: stop the call
                raise
            self.absorb(getattr(error, "details", None) or {})
            status = "stopped" if code.endswith("budget_exceeded") else "refused"
            return Outcome(status, kind, str(error), self.measure())
        self.absorb(result)
        data = self.measure()
        self.record("moves.jsonl", dict(goal=goal, contact=contact, moved=result.get("moved"),
                                        status=result.get("execution_status"), failure=result.get("failure"),
                                        goals=(result.get("move") or {}).get("goals"), at=data))
        if not result.get("moved"):
            failure = result.get("failure") or {}
            return Outcome("refused", kind, f"{failure.get('category', 'not moved')}: {failure.get('message', '')}",
                           data, hint="nothing moved; choose a pose the arm can reach")
        if result.get("execution_status") != "completed":
            stop = result.get("safety_stop") or {}
            return Outcome("stopped", kind, f"the move ended {result.get('execution_status')}: "
                           f"{stop.get('message', '')}".rstrip(": "), data)
        reached = (result.get("move") or {}).get("goals", [{}])[0]
        if "position" not in goal:
            data["arm"] = "arrived"
            return Outcome("done", kind, "done; inspect the measured result", data)
        target = np.asarray(goal["position"], float)
        at = np.asarray(data["tool"], float)
        line = target - np.asarray(reached.get("start_position", at), float)
        unit = line / max(float(np.linalg.norm(line)), 1e-9)
        short = float((target - at) @ unit)
        if reached.get("tracked", True):
            data["arm"] = "arrived"
            if contact and short > SHORT_REPORT_M:
                data["short_m"] = round(short, 4)
            return Outcome("done", kind, "done; inspect the measured result", data)
        if contact:
            # A hand blocked on its way is pushed aside and tilted a little as the controller keeps driving: the
            # fingertips on a counter 5.7 cm above the goal ended 2.2 cm aside and 0.12 rad turned.
            if short > TRACKED_M and short >= float(np.linalg.norm((target - at) - short * unit)):
                data.update(arm="stopped_on_contact", short_m=round(short, 4))
                return Outcome("done", kind, "the approach met contact short of the requested pose", data)
        elif transit and float(np.linalg.norm(target - at)) <= TRANSIT_M:
            data["arm"] = "arrived"
            return Outcome("done", kind, "on the way", data)
        data["arm"] = "not_approaching"
        return Outcome("surprise", kind, f"the hand ended {100 * float(np.linalg.norm(target - at)):.1f} cm from "
                       "its goal", data, goal["position"], data["tool"],
                       "something blocks the way or the pose is at the edge of reach; look before going on")

    def tilt(self, pivot, toward, tilt_deg, expect_mm, over=None) -> dict:
        """Turn the hand, and what it holds, about pivot until it leans tilt_deg from upright toward a level point,
        in moves of at most TILT_STEP_DEG with the grip monitored. With over, the hand first carries the pivot
        there. The finger axis ends level and across the lean, as a leaning grasp's does."""
        p0 = np.asarray(self.hand["eef_position"], float)
        pivot = np.asarray(pivot, float)
        if float(np.linalg.norm(pivot - p0)) > PIVOT_REACH_M:
            raise ValueError(f"the pivot is a point of the held object, within {PIVOT_REACH_M:g} m of the hand")
        end = pivot if over is None else np.asarray(over, float)
        level = np.asarray(toward, float) - end[:2]
        if float(np.linalg.norm(level)) < 0.01:
            raise ValueError("toward is a level point at least 1 cm from where the pivot turns")
        calls = 0
        if over is not None:
            moved = self._leg(dict(target=ARM, position=[float(v) for v in p0 + end - pivot]), "tilt", False)
            calls = 1
            if moved.ok:
                moved = self._opening(expect_mm, moved, "the grip changed during the carry")
            if not moved.ok:
                return dict(status=moved.status, message=moved.message, stage="carry", moves=calls, **moved.data)
            # Arrived counts within 2 cm: turn about where the pivot is, carried with the hand, not where it was sent.
            at = np.asarray(self.hand["eef_position"], float)
            p0, pivot = at, at + pivot - p0
        R0 = rotation(self.hand["eef_quaternion_wxyz"])
        approach, across = tilted(level, tilt_deg) if tilt_deg else (DOWN, tilted(level, 0)[1])
        if across @ (R0 @ OPENS) < 0:
            across = -across  # the finger axis has no sign: keep the one that turns the wrist least
        Rt = np.column_stack([np.cross(across, approach), across, approach])
        turn = Rt @ R0.T
        degrees = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(turn)) - 1) / 2))))
        steps = math.ceil(degrees / TILT_STEP_DEG - 1e-6) if degrees >= 1 else 0
        outcome = Outcome("done", "tilt", "nothing to do: the hand already leans so",
                          dict(self.measure(), arm="arrived"))
        for k in range(1, steps + 1):
            Rk = _partial(turn, k / steps) @ R0
            goal = dict(target=ARM, position=[float(v) for v in pivot + Rk @ R0.T @ (p0 - pivot)],
                        quaternion_wxyz=quaternion(Rk))
            outcome = self._leg(goal, "tilt", False, transit=k < steps)
            calls += 1
            if outcome.ok:
                outcome = self._opening(expect_mm, outcome, "the grip changed during the tilt")
            if not outcome.ok:
                break
        # Where the pivot is now, carried rigidly with the hand from where it was.
        R = rotation(self.hand["eef_quaternion_wxyz"])
        moved = np.asarray(self.hand["eef_position"], float) + R @ R0.T @ (pivot - p0)
        return dict(status=outcome.status, message=outcome.message, moves=calls,
                    pivot=[round(float(v), 4) for v in moved],
                    pivot_moved_m=round(float(np.linalg.norm(moved - pivot)), 4), **outcome.data)

    def sequence(self, moves) -> dict:
        """move_robot calls in order, each with its own arguments, stopping after the first that is refused, does
        not move, does not complete or leaves a goal untracked. The record holds each move's outcome and, as last,
        the result of the last move run, as move_robot returns it."""
        if not isinstance(moves, list) or not 1 <= len(moves) <= MAX_MOVES or any(
                not isinstance(move, dict) or move.keys() != {"goals"} or not isinstance(move["goals"], list)
                for move in moves):
            raise ValueError(f"moves is a list of 1 to {MAX_MOVES} move_robot arguments, each {{\"goals\": [...]}}")
        ran = []
        for move in moves:
            try:
                last = self.robot.move_robot(goals=move["goals"])
            except self.failures as error:
                if getattr(error, "code", "") in ("connection_closed", "internal_error"):
                    raise  # whether it moved is unknown: the caller ends the call
                self.absorb(getattr(error, "details", None) or {})
                last = dict(ok=False, error=str(error), code=getattr(error, "code", None),
                            execution_started=getattr(error, "execution_started", None))
                ran.append(dict(ok=False, code=last["code"]))
                break
            self.absorb(last)
            goals = (last.get("move") or {}).get("goals") or []
            tracked = last.get("tracked", all(goal.get("tracked", True) for goal in goals))
            ran.append(dict(moved=last.get("moved"), execution_status=last.get("execution_status"),
                            tracked=tracked))
            self.record("moves.jsonl", dict(goal=move["goals"], sequence=True, moved=last.get("moved"),
                                            status=last.get("execution_status"), failure=last.get("failure"),
                                            goals=goals, at=self.measure()))
            if not last.get("moved") or last.get("execution_status") != "completed" or not tracked:
                break
        else:
            return dict(status="done", ran=len(ran), of=len(moves), moves=ran, last=last)
        return dict(status="stopped", ran=len(ran), of=len(moves), moves=ran, last=last)

    def _opening(self, expect, outcome, message="the gripper stopped outside the expected opening"):
        observed = round(self.opening_mm(), 1)
        if max(expect[0] - GRIP_SLACK_MM, EMPTY_MM) <= observed <= expect[1] + GRIP_SLACK_MM:
            return outcome
        # Fingers stopped wider than an empty close met something, even where that is narrower than expected.
        gripper = "reached_end" if observed < EMPTY_MM else "stopped_apart"
        return Outcome("surprise", outcome.kind, message, dict(outcome.data, aperture_mm=observed, gripper=gripper),
                       list(expect), observed, "inspect the cameras; the opening does not show what is held")
