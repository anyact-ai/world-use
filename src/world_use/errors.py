"""A refused step: it starts no motion; earlier steps in a running plan may have moved."""
from __future__ import annotations


class Refused(ValueError):
    """A step was refused before it moved; preceding steps may have completed.

    rule      which check refused it (e.g. "joint_limit", "speed", "reach", "keep_out")
    hint      the nearest thing that would pass, when there is one
    problems  every limit the command would break, so one refusal can name all of them
    """

    def __init__(self, message: str, rule: str = "", hint: str = "", **data):
        super().__init__(message)
        self.rule, self.hint, self.data = rule, hint, data
        self.problems: list[Refused] = [self]

    @classmethod
    def several(cls, problems: list[Refused]) -> Refused:
        """One refusal that names every problem found."""
        if len(problems) == 1:
            return problems[0]
        hints = list(dict.fromkeys(p.hint for p in problems if p.hint))
        r = cls("; ".join(str(p) for p in problems), problems[0].rule, "; ".join(hints))
        r.problems = list(problems)
        return r

    def to_dict(self) -> dict:
        d = dict(message=str(self), rule=self.rule, hint=self.hint, **self.data)
        if len(self.problems) > 1:
            d["problems"] = [p.to_dict() for p in self.problems]
        return d


def explain(e: BaseException) -> str:
    """An exception as an operator reads it, notes included: "treat the arm as energised" must not get lost."""
    return " ".join([f"{type(e).__name__}: {e}", *getattr(e, "__notes__", ())])
