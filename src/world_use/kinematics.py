"""Serial-chain kinematics from a URDF: forward kinematics, Jacobian, IK and gravity torques. numpy only.

The chain runs from the URDF root to one tool link, through rotational joints only. Movable joints off that path
(gripper fingers) are treated as fixed at zero; their links still count for gravity.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import axis_angle, pose_error, rpy

G = 9.81
MOVABLE = ("revolute", "continuous", "prismatic")


@dataclass(frozen=True)
class _Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray            # 4x4, parent link -> joint frame at q = 0
    axis: np.ndarray              # unit vector in the joint frame
    lower: float
    upper: float


def _attr(element, name: str) -> str:
    """A required URDF attribute."""
    value = None if element is None else element.get(name)
    if value is None:
        raise ValueError(f"URDF element is missing its {name!r}")
    return value


def _floats(text, default):
    return np.array([float(v) for v in text.split()]) if text else np.array(default, float)


class Chain:
    def __init__(self, urdf: str | Path, tool_link: str):
        root = ET.parse(urdf).getroot()
        self.links: dict[str, tuple[float, np.ndarray]] = {}           # link -> (mass, centre of mass in link frame)
        for link in root.findall("link"):
            inertial = link.find("inertial")
            mass = None if inertial is None else inertial.find("mass")
            if inertial is not None and mass is not None:
                origin = inertial.find("origin")
                com = _floats(origin.get("xyz") if origin is not None else None, [0, 0, 0])
                self.links[_attr(link, "name")] = (float(_attr(mass, "value")), com)
        joints = []
        for j in root.findall("joint"):
            o, a, lim = j.find("origin"), j.find("axis"), j.find("limit")
            T = np.eye(4)
            T[:3, :3] = rpy(*_floats(o.get("rpy") if o is not None else None, [0, 0, 0]))
            T[:3, 3] = _floats(o.get("xyz") if o is not None else None, [0, 0, 0])
            axis = _floats(a.get("xyz") if a is not None else None, [1, 0, 0])
            kind = j.get("type")
            lower = float(lim.get("lower", -np.inf)) if lim is not None and kind != "continuous" else -np.inf
            upper = float(lim.get("upper", np.inf)) if lim is not None and kind != "continuous" else np.inf
            norm = np.linalg.norm(axis)
            joints.append(_Joint(_attr(j, "name"), _attr(j, "type"), _attr(j.find("parent"), "link"),
                                 _attr(j.find("child"), "link"), T, axis / norm if norm > 0 else np.array([1.0, 0, 0]),
                                 lower, upper))
        by_child = {j.child: j for j in joints}
        children = {j.parent for j in joints} | {j.child for j in joints}
        roots = [c for c in children if c not in by_child]
        if len(roots) != 1:
            raise ValueError(f"URDF must have exactly one root link, found {roots}")
        self.root, self.tool_link = roots[0], tool_link
        if tool_link not in by_child and tool_link != self.root:
            raise ValueError(f"tool link {tool_link!r} is not in the URDF")
        path, link = [], tool_link
        while link in by_child:                     # walk up from the tool to the root
            path.append(by_child[link])
            link = by_child[link].parent
        path.reverse()
        self._length = sum(float(np.linalg.norm(j.origin[:3, 3])) for j in path)
        self.active = [j for j in path if j.type in MOVABLE]
        if any(j.type == "prismatic" for j in self.active):
            raise ValueError("the runtime supports rotational arm joints only; prismatic joints are unsupported")
        self.joint_names = [j.name for j in self.active]
        self.n = len(self.active)
        self.lower = np.array([j.lower for j in self.active])
        self.upper = np.array([j.upper for j in self.active])
        self._index = {j.name: i for i, j in enumerate(self.active)}
        ordered, known, pending = [], {self.root}, list(joints)
        while pending:                              # parent before child, so link_frames is one pass
            ready = [j for j in pending if j.parent in known]
            if not ready:
                raise ValueError("URDF joints do not form a tree")
            for j in ready:
                ordered.append(j)
                known.add(j.child)
                pending.remove(j)
        self._joints = ordered
        self._carried = {}                          # links each active joint holds up (for gravity)
        for j in self.active:
            carried = {j.child}
            for k in ordered:
                if k.parent in carried:
                    carried.add(k.child)
            self._carried[j.name] = [link for link in carried if link in self.links]
        self._carried_mass = [sum(self.links[link][0] for link in self._carried[j.name]) for j in self.active]
        self._memo: tuple[np.ndarray, dict] | None = None   # the last pose's link frames: many callers ask per tick

    # -- forward kinematics ---------------------------------------------------------------------
    def link_frames(self, q) -> dict[str, np.ndarray]:
        """Every link's pose in the root frame. Read-only: the result is cached for the next call with the same q."""
        q = np.asarray(q, float)
        memo = self._memo
        if memo is not None and np.array_equal(memo[0], q):
            return memo[1]
        T = {self.root: np.eye(4)}
        for j in self._joints:
            Tj = j.origin
            i = self._index.get(j.name)
            if i is not None:
                Tq = np.eye(4)
                Tq[:3, :3] = axis_angle(j.axis, q[i])
                Tj = Tj @ Tq
            T[j.child] = T[j.parent] @ Tj
        self._memo = (q.copy(), T)
        return T

    def fk(self, q) -> np.ndarray:
        """Pose of the tool link in the root frame."""
        return self.link_frames(q)[self.tool_link].copy()

    def points(self, q) -> np.ndarray:
        """Joint origins and the tool point: a coarse stick model for clearance checks."""
        F = self.link_frames(q)
        return np.array([F[self.root][:3, 3]] + [F[j.child][:3, 3] for j in self.active] + [F[self.tool_link][:3, 3]])

    def motion_bound(self, a, b) -> float:
        """Upper bound on any stick-model point's travel during linear joint interpolation (metres)."""
        return float(self._length * np.abs(np.asarray(b) - np.asarray(a)).sum())

    def axes(self, q) -> np.ndarray:
        """Each moving joint's axis in the root frame, one row per joint."""
        F = self.link_frames(q)
        return np.array([F[j.child][:3, :3] @ j.axis for j in self.active])

    def jacobian(self, q) -> np.ndarray:
        """6 x n geometric Jacobian of the tool point: linear rows, then angular rows, root frame."""
        F = self.link_frames(q)
        p = F[self.tool_link][:3, 3]
        J = np.zeros((6, self.n))
        for i, j in enumerate(self.active):
            axis = F[j.child][:3, :3] @ j.axis
            J[:3, i] = np.cross(axis, p - F[j.child][:3, 3])
            J[3:, i] = axis
        return J

    def ik(self, T_des, q_seed, lower=None, upper=None, weights=None, iters=60, damping=1e-4, tol=1e-7,
           max_step=0.05) -> tuple[np.ndarray, float]:
        """Damped least squares from a nearby seed, joints clamped to limits. Returns (q, weighted residual).

        weights: 6 values for (x, y, z, rx, ry, rz); zeros drop an axis (a 5-joint arm cannot hold every
        orientation, so it asks for position plus the orientation axes it can reach).
        """
        lower = self.lower if lower is None else lower
        upper = self.upper if upper is None else upper
        W = np.ones(6) if weights is None else np.asarray(weights, float)
        q = np.clip(np.asarray(q_seed, float).copy(), lower, upper)
        for _ in range(iters):
            e = pose_error(self.fk(q), T_des) * W
            if np.linalg.norm(e) < tol:
                break
            J = self.jacobian(q) * W[:, None]
            dq = np.linalg.solve(J.T @ J + damping * np.eye(self.n), J.T @ e)
            q = np.clip(q + np.clip(dq, -max_step, max_step), lower, upper)
        return q, float(np.linalg.norm(pose_error(self.fk(q), T_des) * W))

    # -- statics --------------------------------------------------------------------------------
    def gravity(self, q) -> np.ndarray:
        """Torque each joint must supply to hold q against gravity, from the URDF inertials."""
        F = self.link_frames(q)
        weighted = {link: m * (F[link][:3, :3] @ com + F[link][:3, 3])
                    for link, (m, com) in self.links.items() if link in F}
        g = np.zeros(self.n)
        for i, j in enumerate(self.active):
            axis = F[j.child][:3, :3] @ j.axis
            mass = self._carried_mass[i]
            # z of axis x (moment of the carried mass about the joint origin): gravity only pulls along -z
            r = sum((weighted[link] for link in self._carried[j.name]), np.zeros(3)) - mass * F[j.child][:3, 3]
            g[i] = G * (axis[0] * r[1] - axis[1] * r[0])
        return g

    @property
    def carried_links(self) -> list[str]:
        """Links with an inertial that some joint holds up: the ones whose weight shows in joint torque."""
        carried = set().union(*(self._carried[j.name] for j in self.active))
        return [link for link in self.links if link in carried]

    def gravity_regressor(self, q, links) -> np.ndarray:
        """n x 4k matrix Y with gravity(q) = Y @ phi, phi being each of the k links' [mass, mass * centre of mass
        (its own frame)]: holding torque is linear in those, so recorded torques can fit them (see fit.py)."""
        F = self.link_frames(q)
        Y = np.zeros((self.n, 4 * len(links)))
        for i, j in enumerate(self.active):
            axis = F[j.child][:3, :3] @ j.axis
            carried = self._carried[j.name]
            for c, link in enumerate(links):
                if link not in carried:
                    continue
                arm = F[link][:3, 3] - F[j.child][:3, 3]
                Y[i, 4 * c] = G * (axis[0] * arm[1] - axis[1] * arm[0])
                R = F[link][:3, :3]
                Y[i, 4 * c + 1:4 * c + 4] = G * (axis[0] * R[1] - axis[1] * R[0])
        return Y

    def set_links(self, links: dict[str, tuple[float, np.ndarray | list[float]]]):
        """Replace links' masses and centres of mass (in their own frames), e.g. with values fitted from records."""
        for name, (mass, com) in links.items():
            if name not in self.links:
                raise KeyError(f"no link {name!r} with an inertial in this URDF")
            self.links[name] = (float(mass), np.asarray(com, float))
        self._carried_mass = [sum(self.links[link][0] for link in self._carried[j.name]) for j in self.active]

    def potential(self, q) -> float:
        F = self.link_frames(q)
        return sum(m * G * (F[link] @ np.append(c, 1.0))[2] for link, (m, c) in self.links.items() if link in F)

    @property
    def mass(self) -> float:
        return sum(m for m, _ in self.links.values())
