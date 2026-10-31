"""Small manipulation procedures outside the control loop.

The caller supplies checked, bounded commands and observations. Every command,
including a check or withdrawal, goes through the same executor. These functions
do not select grasps, invent geometry, retry, or certify physical support.
"""
from __future__ import annotations

from collections.abc import Callable

from .behaviors import Outcome

Execute = Callable[[dict], Outcome]
Observe = Callable[[], dict]


def feedback(outcome: Outcome) -> dict:
    """Current procedure evidence for the policy; keep the full outcome in its record.

    Successful stage telemetry stays in the record. A failure retains its expected
    and observed values and recovery hint alongside the before/after evidence.
    """
    data = outcome.data
    result = dict(status=outcome.status, procedure=outcome.kind, message=outcome.message,
                  completion_verified=False, intent=data["intent"], assessment=data["assessment"],
                  stages=[dict(stage=s["stage"], status=s["outcome"]["status"]) for s in data["stages"]],
                  observations=data["observations"])
    result.update({k: v for k, v in outcome.feedback().items() if k in ("expected", "observed", "hint")})
    return result


def _run(kind: str, execute: Execute, observe: Observe, stages: list[tuple[str, dict]],
         intent: dict[str, str]) -> Outcome:
    observations = {"before": observe()}
    outcomes = []
    data = dict(stages=outcomes, observations=observations, completion_verified=False, intent=intent,
                assessment="unknown")
    selected = {t["target"]: t for t in observations["before"].get("targets", [])}
    for role, target in intent.items():
        if not isinstance(target, str) or not target:
            raise ValueError(f"{role} must name a selected target")
        if not selected.get(target, {}).get("valid"):
            return Outcome("refused", kind, f"{role} {target!r} has no current evidence", data,
                           expected="a currently observed selection", observed=selected.get(target),
                           hint="inspect the current view and reselect the intended region before moving")
    for name, command in stages:
        outcome = execute(command)
        outcomes.append(dict(stage=name, outcome=outcome.feedback()))
        if not outcome.ok:
            # No recovery motion follows a failed check. In particular, an uncertain
            # grip must not cause an automatic opening or retreat with a payload.
            observations["after"] = observe()
            return Outcome(outcome.status, kind, f"{name}: {outcome.message}",
                           data,
                           outcome.expected, outcome.observed, outcome.hint)
        if name == "check_before_lift":
            observations["before_lift"] = observe()
    observations["after"] = observe()
    return Outcome("done", kind, "commands and measured checks completed; inspect the outcome evidence",
                   data, hint="compare the intended object's before/after appearance and motion; "
                   "judge retention or placement from fresh views, and choose a correction if uncertain")


def grasp(execute: Execute, observe: Observe, *, approach: dict, close: dict, check: dict, lift: dict,
          object_target: str | None = None) -> Outcome:
    """Arrive at a chosen grasp, close, check, test lift, then check again.

    approach must require measured arm arrival before closing; lift must be short,
    bounded and guarded by the expected grip range. check must use fresh measured
    opening, not command state. The executor owns validation and step accounting.
    Observations before and after the lift let the policy judge co-motion and
    choose a new attempt explicitly if needed. No subsequent carry is implied.
    object_target optionally binds the attempt to a selected region in observe's
    targets list. A label is the caller's identity claim, not tracker recognition.
    """
    return _run("grasp", execute, observe, [("approach", approach), ("close", close), ("check_before_lift", check),
                                           ("test_lift", lift), ("check_after_lift", check)],
                {} if object_target is None else {"object": object_target})


def place(execute: Execute, observe: Observe, *, move: dict, release: dict,
          withdraw: dict, check: dict, object_target: str | None = None,
          destination_target: str | None = None) -> Outcome:
    """Arrive at a chosen release pose, release, check opening, then withdraw.

    move must require measured arrival and monitor retention; release must check
    measured opening. Targets and a clear withdrawal are the caller's decisions.
    Inspect the final fresh observation for support and stability; image overlap,
    an open gripper and completed commands do not establish placement success.
    Optional target names bind the observation evidence to the intended object
    and destination. The caller owns identity and semantic outcome assessment.
    """
    return _run("place", execute, observe, [("arrive", move), ("release", release),
                                           ("check_release", check), ("withdraw", withdraw)],
                {role: name for role, name in (("object", object_target), ("destination", destination_target))
                 if name is not None})
