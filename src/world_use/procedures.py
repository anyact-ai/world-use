"""Small manipulation procedures outside the control loop.

The caller supplies checked, bounded commands and observations. Every command,
including a check or withdrawal, goes through the same executor. These functions
do not select grasps, invent geometry, retry, or certify physical support.
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
        # Cancellation cannot leave a background thread issuing later stages.
        steps.close()


def feedback(outcome: Outcome) -> dict:
    """Current procedure evidence for the policy; keep the full outcome in its record.

    Successful stage telemetry stays in the record. A failure retains its expected
    and observed values and recovery hint alongside the before/after evidence.
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
        data["next_stage"] = name
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
    """Arrive at a chosen grasp, close, check, test lift, then check again.

    approach must require measured arm arrival before closing; lift must be short,
    bounded and guarded by the expected grip range. check must use fresh measured
    opening, not command state. The executor owns validation and step accounting.
    Observations before and after the lift let the policy judge co-motion and
    choose a new attempt explicitly if needed. No subsequent carry is implied.
    object_target optionally binds the attempt to a selected region in observe's
    targets list. A label is the caller's identity claim, not tracker recognition.
    Bindings are rechecked before close, before lift and after lift. assess receives
    that checkpoint name and the observation history; only a done Outcome permits
    continuation. It runs outside the control loop. Unknown evidence must return
    stopped; exceptions propagate without subsequent motion. There is no implicit retry.
    """
    return _drive(_run("grasp", execute, observe,
        [("approach", approach), ("close", close), ("check_before_lift", check),
         ("test_lift", lift), ("check_after_lift", check)],
        {} if object_target is None else {"object": object_target},
        {"approach": "before_close", "check_before_lift": "before_lift", "check_after_lift": "after"}, assess))


def place(execute: Execute, observe: Observe, *, move: dict, release: dict,
          withdraw: dict, check: dict, object_target: str | None = None,
          destination_target: str | None = None, assess: Assess | None = None) -> Outcome:
    """Arrive at a chosen release pose, release, check opening, then withdraw.

    move must require measured arrival and monitor retention; release must check
    measured opening. Targets and a clear withdrawal are the caller's decisions.
    Inspect the final fresh observation for support and stability; image overlap,
    an open gripper and completed commands do not establish placement success.
    Optional target names bind the observation evidence to the intended object
    and destination. The caller owns identity and semantic outcome assessment.
    Recheck bindings and call assess at before_release and after withdrawal, with
    fresh observations. The assessor can prevent opening when measured placement
    is wrong or unknown. Recovery and any remaining stages require a new explicit
    call through the executor; this function never replays completed stages.
    """
    return _drive(_run("place", execute, observe, [("arrive", move), ("release", release),
                                           ("check_release", check), ("withdraw", withdraw)],
                {role: name for role, name in (("object", object_target), ("destination", destination_target))
                 if name is not None}, {"arrive": "before_release", "withdraw": "after"}, assess))


async def async_place(execute: AsyncExecute, observe: AsyncObserve, *, move: dict, release: dict,
                      withdraw: dict, check: dict, object_target: str | None = None,
                      destination_target: str | None = None, assess: Assess | AsyncAssess | None = None) -> Outcome:
    """The same place contract for async tool clients, without a background procedure thread.

    Cancellation prevents later stages; it does not cancel a command already accepted
    by a remote executor. The caller must reconcile that job and resolve motor power.
    """
    return await _drive_async(_run("place", execute, observe,
        [("arrive", move), ("release", release), ("check_release", check), ("withdraw", withdraw)],
        {role: name for role, name in (("object", object_target), ("destination", destination_target))
         if name is not None}, {"arrive": "before_release", "withdraw": "after"}, assess))
