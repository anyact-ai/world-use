"""The one exception a policy needs to understand: a refused command. Nothing moved."""
from __future__ import annotations


class Refused(ValueError):
    """A command was refused before anything moved.

    rule   which check refused it (e.g. "joint_limit", "speed", "reach", "keep_out")
    hint   the nearest thing that would pass, when there is one
    """

    def __init__(self, message: str, rule: str = "", hint: str = "", **data):
        super().__init__(message)
        self.rule, self.hint, self.data = rule, hint, data

    def to_dict(self) -> dict:
        return dict(message=str(self), rule=self.rule, hint=self.hint, **self.data)
