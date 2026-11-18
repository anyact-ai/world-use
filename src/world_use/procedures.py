"""Grasp and place as short sequences of the caller's commands, run outside the control loop.

Every command, including each grip check and the withdrawal, goes through the caller's execute,
which validates it and returns an Outcome. A sequence stops at the first command or check that
does not succeed, and never opens, retreats or retries on its own. The caller chooses the grasp,
the poses and any recovery; a completed sequence does not show that the object is held or placed.
"""
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Generator
from copy import deepcopy
from typing import Any

from .behaviors import Outcome

Execute = Callable[[dict], Outcome]
Observe = Callable[[], dict]
Assess = Callable[[str, dict], Outcome]
AsyncExecute = Callable[[dict], Awaitable[Outcome]]
AsyncObserve = Callable[[], Awaitable[dict]]
AsyncAssess = Callable[[str, dict], Awaitable[Outcome]]


def _drive(steps):
    value = None
    try:
        while True:
            value = steps.send(value)
            if inspect.isawaitable(value):
                if inspect.iscoroutine(value):
                    value.close()
                raise TypeError("synchronous procedures require synchronous callbacks")
    except StopIteration as done:
        return done.value
    finally:
        steps.close()


async def _drive_async(steps):
    value = None
    try:
        while True:
            value = steps.send(value)
            if inspect.isawaitable(value):
                value = await value
    except StopIteration as done:
        return done.value
    finally:
        steps.close()


def feedback(outcome: Outcome) -> dict:
    """A short report of a procedure's outcome for the policy; record outcome.to_dict() in full.

    It gives each stage's status, the last checkpoint and the next stage, the latest assessment,
    and two observations: the reference one (before the lift for a grasp, otherwise before the
    first command) and the latest. A failure adds the last command run and the outcome's
    expected, observed and hint.
    """
    data = outcome.data
    observations = data["observations"]
    reference = "before_lift" if "before_lift" in observations else "before"
    latest = next(reversed(observations))
    result = dict(status=outcome.status, procedure=outcome.kind, message=outcome.message,
                  completion_verified=False, intent=data["intent"], assessment=data["assessment"],
                  stages=[dict(stage=s["stage"], status=s["outcome"]["status"]) for s in data["stages"]],
                  observations={name: observations[name] for name in dict.fromkeys((reference, latest))})
    result.update(checkpoint=data.get("checkpoint"), next_stage=data.get("next_stage"),
                  assessments=data.get("assessments", [])[-1:])
    if not outcome.ok and data["stages"]:
        result["last_command"] = data["stages"][-1]
    result.update({k: v for k, v in outcome.feedback().items() if k in ("expected", "observed", "hint")})
    return result


def _run(kind: str, execute: Execute | AsyncExecute, observe: Observe | AsyncObserve,
         stages: list[tuple[str, dict]], intent: dict[str, str], checkpoints: dict[str, str],
         assess: Assess | AsyncAssess | None) -> Generator[object, Any, Outcome]:
    observations = {}
    outcomes = []
    data: dict = dict(stages=outcomes, observations=observations, completion_verified=False, intent=intent,
                      assessment="unknown", assessments=[], checkpoint=None, next_stage=stages[0][0])
    if assess is not None and not callable(assess):
        raise TypeError("assess must be callable")
    for role, target in intent.items():
        if not isinstance(target, str) or not target:
            raise ValueError(f"{role} must name a selected target")

    def checkpoint(name):
        data["checkpoint"] = name
        observation = observations[name] = deepcopy((yield observe()))
        selected = {t["target"]: t for t in observation.get("targets", [])}
        for role, target in intent.items():
            if selected.get(target, {}).get("valid") is not True:
                return Outcome("refused" if not outcomes else "stopped", kind,
                               f"{name}: {role} {target!r} has no current evidence", data,
                               expected="a currently observed selection", observed=selected.get(target),
                               hint="inspect and reselect the intended region; completed stages are not replayed")
        if assess is not None and name != "before":
            assessment = yield assess(name, deepcopy(observations))
            if assessment is None and name == "after":
                # No stage follows, so a caller with nothing to say leaves the assessment unknown.
                return None
            if not isinstance(assessment, Outcome):
                raise TypeError("assess must return an Outcome; unknown evidence must stop the procedure")
            data["assessments"].append(dict(checkpoint=name, outcome=deepcopy(assessment.feedback())))
            if not assessment.ok:
                return Outcome(assessment.status, kind, f"{name}: {assessment.message}", data,
                               assessment.expected, assessment.observed, assessment.hint)
            if name == "after":
                data["assessment"] = "passed_caller_checks"
        return None

    if failure := (yield from checkpoint("before")):
        return failure
    for index, (name, command) in enumerate(stages):
        outcome = yield execute(command)
        outcomes.append(dict(stage=name, command=deepcopy(command), outcome=deepcopy(outcome.feedback())))
        if not outcome.ok:
            # No recovery motion follows a failed check. In particular, an uncertain
            # grip must not cause an automatic opening or retreat with a payload.
            observations["after"] = deepcopy((yield observe()))
            return Outcome(outcome.status, kind, f"{name}: {outcome.message}",
                           data,
                           outcome.expected, outcome.observed, outcome.hint)
        data["next_stage"] = stages[index + 1][0] if index + 1 < len(stages) else None
        if name in checkpoints and (failure := (yield from checkpoint(checkpoints[name]))):
            return failure
    return Outcome("done", kind, "commands and measured checks completed; inspect the outcome evidence",
                   data, hint="compare the intended object's before/after appearance and motion; "
                   "judge retention or placement from fresh views, and choose a correction if uncertain")


def grasp(execute: Execute, observe: Observe, *, approach: dict, close: dict, check: dict, lift: dict,
          object_target: str | None = None, assess: Assess | None = None) -> Outcome:
    """Approach a chosen grasp, close, check the grip, lift a little and check it again.

    Each command is a dict passed unchanged to execute, which validates it, counts its steps and
    returns an Outcome. approach must wait for the arm to arrive before closing; lift must be short
    and monitor the expected grip range; check must measure the opening afresh, not read the
    command. Nothing is carried after the lift.

    observe() is called at the checkpoints before (ahead of the first command), before_close,
    before_lift and after (following the second check). A command that fails ends the grasp with
    one more observation, also recorded as after. Compare before_lift with after to judge whether
    the object rose with the hand.

    object_target names a region in the observations' targets list: the caller's claim of which
    object this is, not the tracker's. Each checkpoint's observation must show it valid, or the
    grasp stops there. assess(checkpoint, observations) is called at every checkpoint except before
    and returns an Outcome: only done lets the grasp continue, and unknown evidence should return
    stopped. At after, where no stage follows, it may return None to leave the assessment unknown.
    Exceptions propagate with no further motion. Nothing is retried.
    """
    return _drive(_run("grasp", execute, observe,
        [("approach", approach), ("close", close), ("check_before_lift", check),
         ("test_lift", lift), ("check_after_lift", check)],
        {} if object_target is None else {"object": object_target},
        {"approach": "before_close", "check_before_lift": "before_lift", "check_after_lift": "after"}, assess))


def place(execute: Execute, observe: Observe, *, move: dict, release: dict,
          withdraw: dict, check: dict, object_target: str | None = None,
          destination_target: str | None = None, assess: Assess | None = None) -> Outcome:
    """Move to a chosen release pose, open, check the opening, then withdraw.

    Commands go through execute as in grasp. move must wait for the arm to arrive and monitor the
    grip on the way; the release pose and a clear withdrawal are the caller's choice.
    object_target and destination_target name regions each checkpoint's observation must show
    valid, as in grasp.

    The checkpoints are before, before_release and after (following the withdrawal); a failed
    command adds an after observation, as in grasp. assess is called at before_release and after;
    one that does not return done at before_release keeps the gripper closed. Judge support and
    stability from the last observation: an open gripper, overlapping image regions and completed
    commands do not show that the object was placed. Completed stages are never replayed; recover
    with new commands through the executor.
    """
    return _drive(_place(execute, observe, move, release, check, withdraw, object_target, destination_target,
                         assess))


async def async_place(execute: AsyncExecute, observe: AsyncObserve, *, move: dict, release: dict,
                      withdraw: dict, check: dict, object_target: str | None = None,
                      destination_target: str | None = None, assess: Assess | AsyncAssess | None = None) -> Outcome:
    """place for async clients: execute and observe are awaitable, assess may be either.

    It runs in the caller's event loop, not a background thread. Cancelling it stops later stages,
    but not a command a remote executor has already accepted: the caller must find out how that
    job ended and resolve motor power.
    """
    return await _drive_async(_place(execute, observe, move, release, check, withdraw, object_target,
                                     destination_target, assess))


def _place(execute, observe, move, release, check, withdraw, object_target, destination_target, assess):
    return _run("place", execute, observe,
                [("arrive", move), ("release", release), ("check_release", check), ("withdraw", withdraw)],
                {role: name for role, name in (("object", object_target), ("destination", destination_target))
                 if name is not None}, {"arrive": "before_release", "withdraw": "after"}, assess)
