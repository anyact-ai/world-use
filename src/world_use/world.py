"""World: named frames, boxes (surfaces, objects, zones) and facts that remember where they came from.

Every fact carries its source and time, and becomes stale when the world may have changed under it (a contact,
a human touching the scene). Policies read the world instead of re-deriving it from images every step.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

KINDS = ("surface", "object", "keep_out", "fragile", "slow")


@dataclass
class Frame:
    name: str
    T: np.ndarray                    # 4x4 pose in the base frame
    source: str = "config"
    t: float = field(default_factory=time.time)

    @property
    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(forward, left, up) = the frame's x, y, z axes in base coordinates."""
        return self.T[:3, 0].copy(), self.T[:3, 1].copy(), self.T[:3, 2].copy()


@dataclass
class Box:
    """An oriented box. Its z axis is "up": surfaces are touched from above.

    surface   something the tool may touch (a table, a printer bed); plans may not pass through it
    object    a thing to manipulate; grip_width (m) is how wide it is between the jaws. Not solid to the
              planner: the fingers have to reach around it
    keep_out  nothing of the arm may enter it
    fragile   contact inside it is judged with a much smaller torque change (dtau, Nm)
    slow      motion inside it is capped at speed (m/s)
    """
    name: str
    kind: str
    pose: np.ndarray                 # 4x4, box centre in the base frame
    size: np.ndarray                 # full extents along the box's own x, y, z
    params: dict = field(default_factory=dict)
    source: str = "config"
    t: float = field(default_factory=time.time)

    def local(self, p) -> np.ndarray:
        return self.pose[:3, :3].T @ (np.asarray(p, float) - self.pose[:3, 3])

    def contains(self, p, margin: float = 0.0) -> bool:
        return bool(np.all(np.abs(self.local(p)) <= self.size / 2 + margin))

    def depth(self, p) -> float:
        """How far p is below the top face, along the box's up axis; -inf outside its footprint. A surface is
        solid all the way down: nothing passes through a table, however thin the box that stands for it."""
        d = self.local(p)
        if np.any(np.abs(d[:2]) > self.size[:2] / 2):
            return -np.inf
        return float(self.size[2] / 2 - d[2])

    @property
    def top(self) -> float:
        return float(self.pose[2, 3] + self.pose[2, 2] * self.size[2] / 2)

    @property
    def solid(self) -> bool:
        return self.kind == "surface"


@dataclass
class Fact:
    key: str
    value: object
    source: str
    t: float = field(default_factory=time.time)
    note: str = ""
    stale: str | None = None         # why it may no longer hold


class World:
    def __init__(self):
        self.frames: dict[str, Frame] = {"base": Frame("base", np.eye(4), "definition")}
        self.boxes: dict[str, Box] = {}
        self.facts: dict[str, Fact] = {}

    # -- frames -----------------------------------------------------------------------------------
    def add_frame(self, name: str, T, source: str = "config") -> Frame:
        self.frames[name] = Frame(name, np.asarray(T, float), source)
        return self.frames[name]

    def frame(self, name: str) -> Frame:
        if name not in self.frames:
            raise KeyError(f"unknown frame {name!r}; known: {sorted(self.frames)}")
        return self.frames[name]

    def to_base(self, frame: str, p) -> np.ndarray:
        T = self.frame(frame).T
        return T[:3, :3] @ np.asarray(p, float) + T[:3, 3]

    def from_base(self, frame: str, p) -> np.ndarray:
        T = self.frame(frame).T
        return T[:3, :3].T @ (np.asarray(p, float) - T[:3, 3])

    # -- boxes ------------------------------------------------------------------------------------
    def add_box(self, name: str, kind: str, center, size, frame: str = "base", yaw_deg: float = 0.0,
                source: str = "config", **params) -> Box:
        if kind not in KINDS:
            raise ValueError(f"box kind must be one of {KINDS}")
        F = self.frame(frame).T
        c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
        pose = np.eye(4)
        pose[:3, :3] = F[:3, :3] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        pose[:3, 3] = F[:3, :3] @ np.asarray(center, float) + F[:3, 3]
        self.boxes[name] = Box(name, kind, pose, np.asarray(size, float), params, source)
        return self.boxes[name]

    def solids(self):
        return [b for b in self.boxes.values() if b.solid]

    def of_kind(self, kind: str):
        return [b for b in self.boxes.values() if b.kind == kind]

    def zones_at(self, p) -> list[Box]:
        return [b for b in self.boxes.values() if b.kind in ("keep_out", "fragile", "slow") and b.contains(p)]

    # -- facts ------------------------------------------------------------------------------------
    def assert_fact(self, key: str, value, source: str, note: str = "") -> Fact:
        self.facts[key] = Fact(key, value, source, note=note)
        return self.facts[key]

    def invalidate(self, reason: str, keys=None) -> list[str]:
        """Mark facts stale (all of them unless keys are given). Returns the keys that changed."""
        changed = []
        for k, f in self.facts.items():
            if (keys is None or k in keys) and f.stale is None:
                f.stale = reason
                changed.append(k)
        return changed

    # -- persistence ------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return dict(
            frames={n: dict(T=np.round(f.T, 6).tolist(), source=f.source, t=f.t) for n, f in self.frames.items()},
            boxes={n: dict(kind=b.kind, pose=np.round(b.pose, 6).tolist(), size=b.size.tolist(), params=b.params,
                           source=b.source, t=b.t) for n, b in self.boxes.items()},
            facts={k: dict(value=f.value, source=f.source, t=f.t, note=f.note, stale=f.stale) for k, f in self.facts.items()})

    @classmethod
    def from_dict(cls, d: dict) -> World:
        w = cls()
        for n, f in (d.get("frames") or {}).items():
            w.frames[n] = Frame(n, np.array(f["T"], float), f.get("source", "config"), f.get("t", time.time()))
        for n, b in (d.get("boxes") or {}).items():
            w.boxes[n] = Box(n, b["kind"], np.array(b["pose"], float), np.array(b["size"], float), b.get("params", {}),
                             b.get("source", "config"), b.get("t", time.time()))
        for k, f in (d.get("facts") or {}).items():
            w.facts[k] = Fact(k, f["value"], f["source"], f.get("t", time.time()), f.get("note", ""), f.get("stale"))
        return w
