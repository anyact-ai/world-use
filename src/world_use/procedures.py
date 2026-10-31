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


def _run(kind: str, execute: Execute, observe: Observe, stages: list[tuple[str, dict]]) -> Outcome:
    observations = {"before": observe()}
    outcomes = []
    for name, command in stages:
        outcome = execute(command)
        outcomes.append(dict(stage=name, outcome=outcome.feedback()))
        if not outcome.ok:
            # No recovery motion follows a failed check. In particular, an uncertain
            # grip must not cause an automatic opening or retreat with a payload.
            observations["after"] = observe()
            return Outcome(outcome.status, kind, f"{name}: {outcome.message}",
                           dict(stages=outcomes, observations=observations, completion_verified=False),
                           outcome.expected, outcome.observed, outcome.hint)
        if name == "check_before_lift":
            observations["before_lift"] = observe()
    observations["after"] = observe()
    return Outcome("done", kind, "commands and measured checks completed; inspect the outcome evidence",
                   dict(stages=outcomes, observations=observations, completion_verified=False))


def grasp(execute: Execute, observe: Observe, *, approach: dict, close: dict, check: dict, lift: dict) -> Outcome:
    """Arrive at a chosen grasp, close, check, test lift, then check again.

    approach must require measured arm arrival before closing; lift must be short,
    bounded and guarded by the expected grip range. check must use fresh measured
    opening, not command state. The executor owns validation and step accounting.
    Observations before and after the lift let the policy judge co-motion and
    choose a new attempt explicitly if needed. No subsequent carry is implied.
    """
    return _run("grasp", execute, observe, [("approach", approach), ("close", close), ("check_before_lift", check),
                                           ("test_lift", lift), ("check_after_lift", check)])


def place(execute: Execute, observe: Observe, *, move: dict, release: dict,
          withdraw: dict, check: dict) -> Outcome:
    """Arrive at a chosen release pose, release, withdraw, and check opening.

    move must require measured arrival and monitor retention; release must check
    measured opening. Targets and a clear withdrawal are the caller's decisions.
    Inspect the final fresh observation for support and stability; image overlap,
    an open gripper and completed commands do not establish placement success.
    """
    return _run("place", execute, observe, [("arrive", move), ("release", release),
                                           ("check_release", check), ("withdraw", withdraw)])
