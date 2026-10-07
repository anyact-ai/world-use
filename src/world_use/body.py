"""Body: what a robot is (its manifest) and the small I/O contract every robot adapter implements.

An adapter only moves joints and reports what it measures. Planning, limits, watchdogs, behaviors and
logging live in the kernel, so a new arm needs a manifest and six methods.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class JointSpec:
    name: str
    lower: float                      # radians
    upper: float
    v_max: float = 0.8                # plan gate: peak planned speed
    a_max: float = 6.0                # plan gate: peak planned acceleration
    track_tol: float = 0.15           # watchdog: |measured - commanded| that means "something is in the way"
    tau_max: float = math.inf         # watchdog: |measured torque| that stops everything
    tau_hold_max: float = math.inf    # plan gate: refuse poses whose gravity load exceeds this
    excursion_exempt: bool = False    # e.g. wrist roll: turning it swings nothing, so no excursion limit
    contact_dtau: float = 3.0         # any move stops when torque departs this far from the gravity model


@dataclass(frozen=True)
class GripperSpec:
    """Positions are native units: what the driver reads and commands, and for a gripper with one joint, that
    joint's URDF coordinate. Speed, tracking tolerance and squeeze left unset scale with the travel
    |open - closed|; tau_max is an absolute effort in the driver's units."""
    closed: float                     # native units at fully closed
    open: float                       # native units at fully open
    approach: tuple[float, float, float]      # tool-frame direction the fingers point
    opens_along: tuple[float, float, float]   # tool-frame axis the jaws open along
    unit: str = "rad"
    m_per_unit: float | None = None   # opening in metres per native unit, if calibrated
    v_max: float = math.nan           # unset: the whole travel in a second
    track_tol: float = math.nan       # unset: 13.5% of the travel
    tau_max: float = 4.0
    squeeze: float = math.nan         # unset: 1.1% of the travel. A grip closes this far past contact (kp times it
                                      # must stay under tau_max)
    tool_point: str = "between the fingertips"   # where the tool link sits, in words

    def __post_init__(self):
        travel = abs(self.open - self.closed)
        for key, share in (("v_max", 1.0), ("track_tol", 0.135), ("squeeze", 0.011)):
            if math.isnan(getattr(self, key)):
                object.__setattr__(self, key, float(f"{share * travel:.3g}"))

    def aperture(self, position: float | None) -> float | None:
        """Opening between the fingers in metres, if the mapping is known."""
        if position is None or self.m_per_unit is None:
            return None
        return (position - self.closed) * self.m_per_unit

    def position(self, aperture_m: float) -> float:
        if self.m_per_unit is None:
            raise ValueError("this gripper has no aperture calibration; command it in native units")
        return self.closed + aperture_m / self.m_per_unit


@dataclass(frozen=True)
class Rest:
    """The pose where switching torque off moves nothing (arms without brakes fall anywhere else)."""
    q: tuple[float, ...]
    joints: tuple[int, ...]           # the joints that carry weight: only they must be near q
    tol: float = 0.15
    stops: tuple[int, ...] = ()       # joints that fold onto a hard stop at q; the stop carries part of their load

    def holds(self, q) -> bool:
        q = np.asarray(q, float)
        idx = list(self.joints)
        return bool(np.abs(q[idx] - np.asarray(self.q)[idx]).max() <= self.tol)

    def off_stop(self, i: int, joint: JointSpec) -> float:
        """+1 or -1: the way joint i leaves the stop it rests on, towards the middle of its range."""
        return 1.0 if (joint.lower + joint.upper) / 2 > self.q[i] else -1.0


@dataclass(frozen=True)
class Manifest:
    name: str
    urdf: Path
    tool_link: str
    joints: tuple[JointSpec, ...]
    rate_hz: float = 100.0
    gripper: GripperSpec | None = None
    rest: Rest | None = None          # None: the robot holds itself when disabled (brakes)
    temp_warn_c: float = 70.0
    temp_limit_c: float = 80.0
    sensing: frozenset[str] = frozenset({"position"})   # + "torque", "temperature", "gripper_effort"
    speed: float = 0.25               # default peak joint speed (rad/s) when a move has no duration
    auto_accel: float = 3.0           # automatic durations keep peak joint acceleration under this
    min_move_s: float = 0.5
    max_segment_m: float = 0.25       # longest single Cartesian segment
    link_radius_m: float = 0.03       # keep-out padding around the coarse joint-to-joint link model
    max_excursion: float | None = None    # rad any joint may travel from the session's start pose
    turn_clearance: tuple[tuple[int, ...], float] | None = None   # (joints, m): only turn these above start height + m
    ik_weights: tuple[float, ...] | None = None   # (x, y, z, rx, ry, rz) the IK holds, base frame; 0 frees an axis.
                                                  # None: all six, or free yaw (rz) on arms with fewer than 6 joints
    notes: tuple[str, ...] = ()       # quirks worth telling the policy about (the embodiment card)
    hardware_notes: tuple[str, ...] = ()  # quirks of the physical robot that a simulation does not reproduce
    frames: Callable | None = None    # (chain, q at session start) -> {name: 4x4}, e.g. a "work" frame
    thermal: dict = field(default_factory=dict)   # optional per-joint heating model for the simulator

    @property
    def n(self) -> int:
        return len(self.joints)

    @property
    def lower(self) -> np.ndarray:
        return np.array([j.lower for j in self.joints])

    @property
    def upper(self) -> np.ndarray:
        return np.array([j.upper for j in self.joints])


@dataclass
class JointState:
    """One measurement. Arrays are per joint in manifest order; None where the robot cannot sense it."""
    t: float
    q: np.ndarray
    dq: np.ndarray | None = None
    tau: np.ndarray | None = None
    temp: np.ndarray | None = None
    gripper: float | None = None
    gripper_tau: float | None = None
    faults: tuple[str, ...] = ()


@runtime_checkable
class Body(Protocol):
    """The I/O contract. The kernel calls it from one thread at a time: once its control loop runs, only that
    thread (hardware drivers are rarely safe to call from two at once).

    A body that computes gravity itself (a feedforward, a simulator's torques) may also take `use_fit(model)`: a
    model fitted from the robot's flight records (fit.py), which it then weighs the links by."""
    manifest: Manifest

    def connect(self) -> JointState:
        """Open the connection read-only and return the measured state. Must not move or release anything."""

    def enable(self) -> None:
        """Switch torque on at the measured pose, without a jump. If it raises, every motor is off again, or the
        error names those it could not confirm off. The kernel treats a failed transition as unconfirmed power
        until disable succeeds at a freshly measured rest pose."""

    def read(self) -> JointState:
        """Latest measurement. Called once per control tick. Preserve its timestamp when serving a cached
        sample; timestamps must be finite and nondecreasing. Arrays must be finite and match the joint count;
        optional sensing may be None. Raise on lost or stale feedback rather than inventing a new timestamp."""

    def command(self, q: np.ndarray, dq: np.ndarray, gripper: float | None, gripper_v: float = 0.0) -> None:
        """Position setpoint for this tick, with velocity feedforward."""

    def disable(self) -> None:
        """Switch torque off. The kernel only calls this where the manifest says it is safe. If it raises, the
        kernel treats power as unconfirmed, faults, and suspends commands until a successful release."""

    def close(self) -> None:
        """Release the connection. Must not switch torque off: that is disable()'s job."""
