"""Strict public request data; domain methods still enforce the limits they own."""
from __future__ import annotations

import inspect
import math
from typing import Annotated, Any, Literal, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model, field_validator, model_validator

Number = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[Number, Field(gt=0)]
Nonnegative = Annotated[Number, Field(ge=0)]
Name = Annotated[str, Field(min_length=1)]
BoxKind = Literal["surface", "object", "keep_out", "fragile", "slow"]
Vector = Annotated[list[Number], Field(min_length=3, max_length=3)]
PositiveVector = Annotated[list[Positive], Field(min_length=3, max_length=3)]
Point = Annotated[list[Number], Field(min_length=2, max_length=2)]
Rectangle = Annotated[list[int], Field(min_length=4, max_length=4)]
Plan = list | dict


def finite_data(value, path="request"):
    """JSON's numeric values must be finite, including values inside untyped plan/context data."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path}: numbers must be finite")
    if isinstance(value, dict):
        for key, item in value.items():
            finite_data(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_data(item, f"{path}[{index}]")
    return value


def validation_error(error: ValidationError) -> str:
    """Name invalid fields without echoing whole plans or Pydantic's diagnostic URLs."""
    return "; ".join(f"{'.'.join(map(str, item['loc'])) or 'request'}: "
                     f"{item['msg'].removeprefix('Value error, ')}"
                     for item in error.errors(include_url=False, include_input=False))


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def finite(cls, value):
        return finite_data(value)


class BoxParameters(Request):
    kind: BoxKind
    grip_width: Positive | None = None
    mass_kg: Positive | None = None
    friction: Nonnegative | None = None
    dtau: Positive | None = None
    speed: Positive | None = None

    @field_validator("grip_width", "mass_kg", "friction", "dtau", "speed")
    @classmethod
    def supplied_number(cls, value):
        if value is None:
            raise ValueError("must be a number")
        return value

    @model_validator(mode="after")
    def kind_parameters(self):
        allowed = {"object": {"grip_width", "mass_kg", "friction"}, "surface": {"mass_kg", "friction"},
                   "fragile": {"dtau"}, "slow": {"speed"}}
        supplied = self.model_fields_set & {"grip_width", "mass_kg", "friction", "dtau", "speed"}
        if unexpected := supplied - allowed.get(self.kind, set()):
            raise ValueError(f"{self.kind} box: unexpected parameters {', '.join(sorted(unexpected))}")
        if self.kind == "slow" and "speed" not in supplied:
            raise ValueError("a slow zone needs a finite, positive speed in m/s")
        return self


class BoxRequest(BoxParameters):
    name: Name
    center: Vector
    size: PositiveVector
    frame: Name = "base"
    yaw_deg: Number = 0.0
    source: str = "config"


class Requirement(Request):
    evidence: Name
    max_age_s: Positive


Requirements = Annotated[list[Requirement], Field(max_length=16)]


class Wait(Request):
    wait: Nonnegative = 0.0


class Run(Wait):
    spec: Plan
    check: bool = True
    requires: Requirements | None = None


class Check(Request):
    spec: Plan


class Look(Request):
    camera: str | None = None
    spec: Plan | None = None
    grid: bool = False


class Answer(Wait):
    job: Annotated[int, Field(ge=1)]
    answer: str


class Stop(Request):
    reason: str = "stop requested"


class HomeRoute(Request):
    steps: list | None
    note: str = ""

    @model_validator(mode="before")
    @classmethod
    def needs_steps(cls, value):
        if isinstance(value, dict) and "steps" not in value:
            raise ValueError("home_route needs steps: a list of moves, [] to fold straight home, or null to clear")
        return value


class FrameChange(Request):
    name: Name
    origin: Vector = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    rpy_deg: Vector = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    source: str = "policy"


class FactChange(Request):
    key: Name
    value: Any
    source: str = "policy"
    note: str = ""


class WorldChange(Request):
    frame: FrameChange | None = None
    fact: FactChange | None = None
    box: BoxRequest | None = None
    remove: Name | None = None

    @model_validator(mode="after")
    def changes(self):
        if not self.model_fields_set or any(getattr(self, name) is None for name in self.model_fields_set):
            raise ValueError("world change: give a frame, a fact, a box or a box to remove")
        return self


class Calibrate(Wait):
    camera: Name
    points: int = 8
    spread: Positive | None = None


class Plane(Request):
    box: Rectangle
    max_error_m: Positive = 0.001


class Measure(Request):
    frame: Name
    point: Point | None = None
    box: Rectangle | None = None
    mask: str | None = None
    target: str | None = None
    plane: Plane | None = None


class Withdraw(Request):
    measurements: list[Name]
    reason: str = ""


class Record(Request):
    context: dict | None = None
    note: str = ""


class Events(Wait):
    since: Annotated[int, Field(ge=0)] = 0
    limit: Annotated[int, Field(ge=1)] | None = None


class FrameQuery(Request):
    id: str | None = None
    camera: str | None = None
    depth: bool = False

    @field_validator("depth", mode="before")
    @classmethod
    def depth_text(cls, value):
        if isinstance(value, str) and value.lower() not in ("true", "false", "1", "0"):
            raise ValueError("depth must be true or false")
        return value


POST = {"run": Run, "check": Check, "look": Look, "answer": Answer, "stop": Stop,
        "enable": Request, "release": Request, "reset": Request, "home_route": HomeRoute,
        "home": Wait, "world": WorldChange, "calibrate": Calibrate, "measure": Measure,
        "withdraw": Withdraw, "record": Record, "shutdown": Request}


def http_request(method: str, route: list[str], query: dict, body: dict) -> tuple[dict, dict]:
    """Decode URL text explicitly; JSON request bodies receive no type coercion."""
    name = route[0]
    query_model = (Events if method == "GET" and route == ["events"] else
                   FrameQuery if method == "GET" and route == ["frame"] else
                   Wait if (method == "GET" and name == "jobs" and len(route) == 2)
                   or (method == "POST" and len(route) == 1 and issubclass(POST.get(name, Request), Wait))
                   else Request)
    query = {key: str(value) if type(value) in (int, float) else value for key, value in query.items()}
    try:
        query = query_model.model_validate_strings(query, strict=False).model_dump(exclude_unset=True)
        if method == "POST" and len(route) == 1 and (model := POST.get(name)):
            body = model.model_validate(body).model_dump(exclude_unset=True)
    except ValidationError as error:
        raise ValueError(validation_error(error)) from None
    return query, body


def tool_model(fn) -> type[Request]:
    """Derive MCP argument validation and its published schema from the public tool signature."""
    hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {name: (hints[name], ... if p.default is inspect.Parameter.empty else p.default)
                              for name, p in inspect.signature(fn).parameters.items()}
    return create_model(f"{fn.__name__}Arguments", __base__=Request, **fields)
