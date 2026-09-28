"""The one exception a policy needs to understand: a refused command. Nothing moved."""


class Refused(ValueError):
    """A command was refused before anything moved.

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
