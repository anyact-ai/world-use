"""Seeed Studio reBot Arm B601-RS: six RobStride motors plus a gripper on CAN, 48 V, no brakes.

The manifest is what the kernel needs to know. ReBotBody is the hardware adapter: it talks MIT-mode position
control to the motors through Seeed's motorbridge driver (the `rebot` extra).

Everything here was learned on the arm: the soft engage, the refusal to switch on or off away from the folded
rest pose, never sending a disable frame from a read-only connection, and the numbers in the notes.
"""
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ...body import GripperSpec, JointSpec, JointState, Manifest, Rest
from ...errors import Refused
from ...kinematics import Chain

HERE = Path(__file__).resolve().parent

# name, motor id, motor model, kp (Nm/rad), kd (Nm s/rad), track_tol (rad), tau_max (Nm), hold_max (Nm), contact (Nm)
# Gains are Seeed's RS values; track_tol and tau_max sit about 2x above anything a healthy move produced on the
# arm. The contact thresholds are first estimates (a healthy move reached ~1.6 Nm on the shoulder); calibrate them.
MOTORS = (
    ("joint1", 1, "rs-06", 50.0, 3.0, 0.15, 10.0, 1.0, 2.5),
    ("joint2", 2, "rs-06", 150.0, 10.0, 0.15, 20.0, 14.0, 3.0),
    ("joint3", 3, "rs-06", 150.0, 10.0, 0.15, 20.0, 14.0, 3.0),
    ("joint4", 4, "rs-00", 50.0, 5.0, 0.20, 7.0, 5.0, 1.5),
    ("joint5", 5, "rs-00", 50.0, 4.0, 0.20, 4.0, 2.0, 1.2),
    ("joint6", 6, "rs-00", 50.0, 4.0, 0.20, 4.0, 2.0, 1.2),
)
GRIPPER_MOTOR = ("gripper", 7, "rs-00", 50.0, 4.0)
URDF_LIMITS = ((-2.8, 2.8), (0.0, 3.14), (0.0, 3.14), (-1.57, 1.57), (-1.57, 1.57), (-3.14, 3.14))

def work_frame(chain: Chain, q) -> np.ndarray:
    """Where the arm points (set by joint 1 alone), flattened: x forward, y left, z up, at the base origin."""
    zero = chain.link_frames(np.zeros(chain.n))
    f0 = zero[chain.tool_link][:3, 0].copy()
    f0[2] = 0.0
    link1 = chain.active[0].child
    f_local = zero[link1][:3, :3].T @ (f0 / np.linalg.norm(f0))
    f = chain.link_frames(q)[link1][:3, :3] @ f_local
    f[2] = 0.0
    f /= np.linalg.norm(f)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = f, np.cross([0, 0, 1.0], f), [0, 0, 1.0]
    return T


# Folded, the shoulder and elbow rest on hard stops (their lower limits). Powered, the elbow meets its stop about
# 1 deg before the angle it sags to unpowered (2026-09-27), so the stop takes part of its load there.
REST = Rest(q=(0.0,) * 6, joints=(1, 2, 3), tol=0.15, stops=(1, 2))
MANIFEST = Manifest(
    name="reBot Arm B601-RS",
    urdf=HERE / "ReBot_Arm_RS.urdf",
    tool_link="gripper_end",
    joints=tuple(JointSpec(name, lo, hi, v_max=0.8, a_max=6.0, track_tol=tol, tau_max=tmax, tau_hold_max=hold,
                           excursion_exempt=(name == "joint6"), contact_dtau=contact)
                 for (name, _, _, _, _, tol, tmax, hold, contact), (lo, hi) in zip(MOTORS, URDF_LIMITS, strict=True)),
    rate_hz=100.0,
    # squeeze: at the gripper's kp of 50 Nm/rad, 0.05 rad on something rigid is 2.5 Nm; 0.1 would pass tau_max
    gripper=GripperSpec(closed=0.05, open=4.5, unit="rad", m_per_unit=0.020, v_max=4.5, track_tol=0.6, tau_max=4.0,
                        squeeze=0.05, approach=(1.0, 0.0, 0.0), opens_along=(0.0, 1.0, 0.0)),
    rest=REST,
    temp_warn_c=70.0,
    temp_limit_c=80.0,
    sensing=frozenset({"position", "torque", "temperature", "gripper_effort"}),
    speed=0.25,
    auto_accel=3.0,
    min_move_s=0.5,
    max_segment_m=0.25,
    max_excursion=1.6,
    turn_clearance=((0, 4, 5), 0.05),
    frames=lambda chain, q: {"work": work_frame(chain, q)},
    notes=(
        "With torque on, the elbow (j3) carries about 7 Nm, its continuous rating, even folded at rest (the adapter "
        "supports the arm's weight in every pose): it heats about 8 C per minute from cold and cools only with "
        "torque off. Decide with torque off; act in bursts.",
        "Gripper opening is roughly 20 mm per rad (approximate); holding shows as -1.4..-2.2 Nm of gripper effort.",
        "The work frame points where the arm points at rest: forward, left, up. It stays fixed for the session.",
        "Nose-down (move_to with point \"down\") is reachable low and near: about U+0.04 to +0.12 with the tool "
        "F+0.14 to +0.26 in front of the base. That is below the turn height, where the wrist and base may not turn, "
        "so it ends a few degrees off straight down (3.6 from rest).",
    ),
    hardware_notes=(
        "Forward/up moves end 2-5 mm low (the elbow carries about 15% more than the URDF says).",
        "After base turns the tool can be 5-10 mm off sideways: the base gain is soft.",
        "Joint torque strays 1-3 Nm from the gravity model over a 10 cm move (friction and hysteresis, not mass), "
        "so a long guarded move can stop on nothing: line to about 2 cm short of the expected contact, then guard "
        "only the rest. Contact is found at a few newtons.",
    ),
)

# MIT command ranges per motor model. motorbridge clamps out-of-range values silently, and a NaN encodes as
# full-scale negative position, velocity and torque, so every value is checked before it is sent.
MIT_RANGE = {"rs-00": dict(pos=4 * np.pi, vel=33.0, kp=500.0, kd=5.0, tau=14.0),
             "rs-06": dict(pos=4 * np.pi, vel=50.0, kp=5000.0, kd=100.0, tau=36.0)}
P_MECH_POS, P_VBUS, P_ZERO_STA = 0x7019, 0x701C, 0x7029
MODE_RUN = 2                          # bits 23:22 of the feedback frame id: 0 disabled, 1 calibrating, 2 running
HOST_ID = 0xFD
ENGAGE_S, RELEASE_S = 1.0, 1.0
START_POSE_TOL = 0.08                 # rad: feedback after engage must agree with the measured start pose
STALE_TICKS, BLIND_TICKS = 10, 20     # bit-identical feedback this long = a silent motor; this long blind = fault
WRAP_SUSPECT = 3.3                    # rad: a motor reading beyond this has probably lost its zero


class ReBotBody:
    manifest = MANIFEST

    def __init__(self, channel: str | None = None, velocity_ff: bool = True, stiffness: float = 1.0):
        self.channel = channel or os.environ.get("REBOT_CHANNEL", "can0")
        self.velocity_ff, self.stiffness = velocity_ff, float(np.clip(stiffness, 0.2, 1.0))
        self.chain = Chain(MANIFEST.urdf, MANIFEST.tool_link)
        self.ctrl: Any = None
        self.motors: list = []
        self.enabled = False
        self.warnings: list[str] = []
        self.grip_offset = 0.0
        self._ff_key = self._ff_val = None
        self._stale, self._last_sig, self._blind = np.zeros(7, int), [None] * 7, 0
        self._mode_check = False              # armed only if every motor reports run mode after engage
        self._awake = None
        self._last: JointState | None = None

    @property
    def last(self) -> JointState:
        if self._last is None:
            raise RuntimeError("the reBot is not connected")
        return self._last

    # -- Body contract ----------------------------------------------------------------------------
    def connect(self) -> JointState:
        try:
            from motorbridge.core import Controller  # ty: ignore[unresolved-import]
            from motorbridge.errors import CallError  # ty: ignore[unresolved-import]
            from motorbridge.models import Mode  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise ImportError("the reBot adapter needs Seeed's driver, the rebot extra: "
                              "uv tool install 'world-use[rebot] @ git+https://github.com/anyact-ai/world-use'") from e
        self._Mode = Mode
        try:
            self.ctrl = Controller(self.channel)
            self.motors = [self.ctrl.add_robstride_motor(mid, HOST_ID, model)
                           for _, mid, model, *_ in (*MOTORS, GRIPPER_MOTOR)]
        except CallError as e:
            self._close_bus()
            raise ConnectionError(f"{e}\nCould not open the CAN adapter: another program (MotorBridge Studio, "
                                  "another arm script) may hold it, or it is unplugged.") from e
        rows = [self._telemetry(i) for i in range(7)]
        bad = [r for r in rows if r.get("error") or r.get("fault")]
        if bad:
            raise ConnectionError("motor not healthy: "
                                  + "; ".join(f"{r['name']}: {r.get('error') or hex(r['fault'])}" for r in bad))
        low = [r for r in rows if r["vbus"] < 40.0]
        if low:
            raise ConnectionError(f"bus voltage low ({low[0]['vbus']:.1f} V on {low[0]['name']}): "
                                  "is the 48 V supply on?")
        pos = np.array([r["pos"] for r in rows])
        if np.abs(pos[:6]).max() > WRAP_SUSPECT or not all(r["zero_sta"] == 1 for r in rows):
            raise ConnectionError(f"joint angles look wrapped (rad {np.round(pos, 2).tolist()}): a motor lost its zero")
        self.grip_offset, grip = self._gripper_frame(float(pos[6]))
        room = np.minimum(pos[:6] - MANIFEST.lower, MANIFEST.upper - pos[:6])
        for i in (0, 4, 5):
            if room[i] < 0.35:
                self.warnings.append(f"joint{i + 1} starts {np.degrees(room[i]):.0f} deg from its limit; "
                                     "sideways moves may be refused. With torque off, turn it towards 0 by hand.")
        self._last = JointState(time.monotonic(), pos[:6].copy(), gripper=grip)
        return self._last

    def enable(self):
        """Torque on at the measured pose: gains and gravity support ramp in over a second, no jump. If anything
        fails on the way, every motor is switched off again before the error goes up: the arm is at rest, so that
        moves nothing, while a motor left on with nothing commanding it holds its last frame indefinitely."""
        q0 = self._params_positions()
        if not REST.holds(q0[:6]):
            raise Refused(f"the arm is not folded at rest (joints deg {np.round(np.degrees(q0[:6]), 1).tolist()}); "
                          "switching on away from rest is how a raised arm gets jerked", "not_at_rest",
                          "fold it by hand with torque off, or use a takeover recovery")
        if sys.platform == "darwin":        # a sleeping Mac stops sending; the motors would hold the last command
            try:
                self._awake = subprocess.Popen(["caffeinate", "-dims", "-w", str(os.getpid())])
            except OSError:
                self._awake = None
        self.enabled = True
        try:
            self._engage(q0)
        except BaseException as e:
            unconfirmed = self._switch_off()
            if unconfirmed:
                e.add_note(f"could not confirm torque-off on: {', '.join(unconfirmed)}. Treat the arm as energised.")
            raise

    def _engage(self, q0):
        for m in self.motors:
            m.ensure_mode(self._Mode.MIT, 1000)
            time.sleep(0.05)
        time.sleep(0.2)
        for m in self.motors:
            m.enable()
            time.sleep(0.02)
        q = q0[:6].copy()
        grip = q0[6] - self.grip_offset
        n = int(ENGAGE_S * MANIFEST.rate_hz)
        for k in range(n):
            self._send_all(q, np.zeros(6), grip, 0.0, scale=(k + 1) / n)
            if k in (n // 10, n - 1):        # at 10% gain, before there is torque to do harm, and at the end
                pos, _ = self._feedback()
                ref = np.append(q, grip)
                if np.any(np.isnan(pos)) or np.abs(pos - ref).max() > START_POSE_TOL:
                    raise ConnectionError(f"feedback {np.round(pos, 3).tolist()} disagrees with the start pose at "
                                          f"{100 * (k + 1) // n}% gain")
            time.sleep(1.0 / MANIFEST.rate_hz)
        modes = [None if s is None else (int(s.arbitration_id) >> 22) & 3 for s in self._states]
        self._mode_check = all(m == MODE_RUN for m in modes)
        if not self._mode_check:
            self.warnings.append(f"motors report mode bits {modes} while enabled (expected {MODE_RUN}); "
                                 "the run-mode check is off for this session")

    def read(self) -> JointState:
        if not self.enabled:                 # no feedback frames without commands: parameter reads, a few per second
            if self._last is None or time.monotonic() - self._last.t > 0.2:
                pos = self._params_positions()
                self._last = JointState(time.monotonic(), pos[:6], gripper=float(pos[6] - self.grip_offset))
            return self._last
        pos, tau = self._feedback()
        faults = []
        temp = np.array([np.nan if s is None else s.t_mos for s in self._states])
        codes = [0 if s is None else s.status_code for s in self._states]
        modes = [-1 if s is None else (int(s.arbitration_id) >> 22) & 3 for s in self._states]
        for i, (name, *_) in enumerate((*MOTORS, GRIPPER_MOTOR)):
            if codes[i]:
                faults.append(f"{name} reports fault bits {codes[i]:#04x}")
            elif self._mode_check and modes[i] not in (-1, MODE_RUN):
                faults.append(f"{name} dropped out of run mode (it switched itself off)")
        if temp[6] > 85.0:
            faults.append(f"gripper motor at {temp[6]:.0f} C")
        frozen = np.isnan(pos).any() or (self._stale > STALE_TICKS).any()
        self._blind = self._blind + 1 if frozen else 0
        if self._blind > BLIND_TICKS:
            faults.append("no fresh feedback (48 V off? cable? adapter?)")
        last = self.last
        q = np.where(np.isnan(pos[:6]), last.q, pos[:6])
        self._last = JointState(time.monotonic(), q, None, np.nan_to_num(tau[:6]), temp[:6],
                                float(np.nan_to_num(pos[6], nan=last.gripper or 0.0)),
                                float(np.nan_to_num(tau[6])),
                                tuple(faults))
        return self._last

    def command(self, q, dq, gripper, gripper_v=0.0):
        self._send_all(np.asarray(q, float), np.asarray(dq, float) if self.velocity_ff else np.zeros(6),
                       self.last.gripper if gripper is None else gripper, gripper_v if self.velocity_ff else 0.0)

    def disable(self):
        """Gains and gravity support ramp out over a second at the commanded pose, then each motor is switched
        off and its acknowledgement checked. The kernel only calls this at the rest pose."""
        q, grip = self.last.q.copy(), self.last.gripper
        n = int(RELEASE_S * MANIFEST.rate_hz)
        for k in range(n):
            self._send_all(q, np.zeros(6), grip, 0.0, scale=1.0 - (k + 1) / n)
            time.sleep(1.0 / MANIFEST.rate_hz)
        unconfirmed = self._switch_off()
        if unconfirmed:
            raise RuntimeError(f"could not confirm torque-off on: {', '.join(unconfirmed)}. "
                               "Treat the arm as energised.")

    def _switch_off(self) -> list[str]:
        """Disable each motor and check its acknowledgement. Returns the motors whose torque-off is unconfirmed."""
        unconfirmed = []
        for (name, *_), m in zip((*MOTORS, GRIPPER_MOTOR), self.motors, strict=True):
            off = False
            for _ in range(3):
                try:
                    m.disable()
                    st = m.get_state()
                    off = st is None or ((int(st.arbitration_id) >> 22) & 3) != MODE_RUN
                    if off:
                        break
                except Exception:
                    time.sleep(0.02)
            if not off:
                unconfirmed.append(name)
        self.enabled = bool(unconfirmed)
        if self._awake:
            self._awake.terminate()
            self._awake = None
        return unconfirmed

    def close(self):
        """Release the adapter WITHOUT disable frames: if the arm is raised and holding, a disable drops it."""
        self._close_bus()
        if self._awake:
            self._awake.terminate()
            self._awake = None

    # -- internals --------------------------------------------------------------------------------
    def _telemetry(self, i) -> dict:
        m, name = self.motors[i], (*MOTORS, GRIPPER_MOTOR)[i][0]
        row = dict(name=name)
        try:
            pos = m.robstride_get_param_f32(P_MECH_POS, 300)
            if pos == 0.0:                   # the driver returns a FAILED read as 0: ask again
                pos = m.robstride_get_param_f32(P_MECH_POS, 300)
                if pos == 0.0:
                    raise RuntimeError("position read returned exactly 0 twice")
            row.update(pos=pos, vbus=m.robstride_get_param_f32(P_VBUS, 300),
                       zero_sta=m.robstride_get_param_u8(P_ZERO_STA, 300), fault=m.robstride_get_fault_report()[0])
        except Exception as e:
            row["error"] = str(e)
        return row

    def _params_positions(self) -> np.ndarray:
        return np.array([m.robstride_get_param_f32(P_MECH_POS, 300) for m in self.motors])

    def _gripper_frame(self, reading: float) -> tuple[float, float]:
        """(offset, true position). Motors power on reporting -pi..pi, so a gripper parked open past pi comes
        back one turn low; it cannot be below 0, so the correction is unambiguous."""
        for turns in (0, 1):
            true = reading + turns * 2 * np.pi
            if -0.35 <= true <= 5.35:
                if turns:
                    self.warnings.append(f"gripper read {reading:+.2f} rad (parked open past 3.14 at power-off); "
                                         f"using {true:.2f}. Park it closed before switching off.")
                return -turns * 2 * np.pi, true
        raise ConnectionError(f"gripper reads {reading:+.2f} rad, outside 0..5 rad even allowing for a wrap")

    def _feedback(self) -> tuple[np.ndarray, np.ndarray]:
        self.ctrl.poll_feedback_once()
        pos, tau = np.full(7, np.nan), np.full(7, np.nan)
        self._states = []
        for i, m in enumerate(self.motors):
            st = m.get_state()
            self._states.append(st)
            if st is None:
                continue
            pos[i], tau[i] = st.pos, st.torq
            # the driver's cache is never cleared, so a dead receive path looks like a perfectly still motor
            sig = (st.pos, st.vel, st.torq, st.t_mos)
            self._stale[i] = self._stale[i] + 1 if sig == self._last_sig[i] else 0
            self._last_sig[i] = sig
        pos[6] -= self.grip_offset
        return pos, tau

    def _gravity(self, q):
        key = q.tobytes()
        if key != self._ff_key:
            self._ff_key, self._ff_val = key, self.chain.gravity(q)
        return self._ff_val

    def _send_all(self, q, dq, grip, grip_v, scale=1.0):
        """One MIT frame per motor. All seven are checked before any is sent: a bad value never leaves half
        the arm on a new command and half on the old one."""
        ff = self._gravity(np.asarray(q, float))
        frames = [(model, q[i], dq[i], kp * self.stiffness * scale, kd, ff[i] * scale)
                  for i, (_, _, model, kp, kd, *_rest) in enumerate(MOTORS)]
        _, _, model, kp, kd = GRIPPER_MOTOR
        frames.append((model, grip + self.grip_offset, grip_v, kp * scale, kd, 0.0))
        checked = [self._check(i, *f) for i, f in enumerate(frames)]
        for m, vals in zip(self.motors, checked, strict=True):
            m.send_mit(*vals)

    @staticmethod
    def _check(i, model, pos, vel, kp, kd, tau) -> tuple:
        r = MIT_RANGE[model]
        vals = dict(pos=float(pos), vel=float(vel), kp=float(kp), kd=float(kd), tau=float(tau))
        for key, v in vals.items():
            lo = 0.0 if key in ("kp", "kd") else -r[key]
            if not (np.isfinite(v) and lo <= v <= r[key]):
                raise ValueError(f"motor {i + 1}: MIT {key}={v} is outside [{lo:g}, {r[key]:g}]; nothing sent")
        return vals["pos"], vals["vel"], vals["kp"], vals["kd"], vals["tau"]

    def _close_bus(self):
        if self.ctrl is not None:
            try:
                self.ctrl.close_bus()
            finally:
                self.ctrl.close()
                self.ctrl = None
