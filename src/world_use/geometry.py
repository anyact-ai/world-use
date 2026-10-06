"""Small rigid-body helpers. Poses are 4x4 homogeneous matrices; everything is SI (m, rad)."""
from __future__ import annotations

import numpy as np


def rpy(r: float, p: float, y: float) -> np.ndarray:
    """URDF roll-pitch-yaw (fixed axes X, Y, Z) as a rotation matrix."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(axis, angle: float) -> np.ndarray:
    """Rotation of `angle` rad about `axis` (Rodrigues)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def rot_z(angle: float) -> np.ndarray:
    return axis_angle((0, 0, 1), angle)


def align_planar(source, target, tool, *, max_error_m: float) -> dict:
    """Fit corresponding 3D landmarks with XY translation and yaw, without scale or reflection.

    All inputs use the same coordinate frame. Correspondence and a rigid grasp are the caller's claims.
    The proposed tool pose preserves its height and accounts for an off-centre grasp. It is geometry,
    not a motion plan or a grasp/clearance check. Relative landmark heights must also agree, allowing
    a constant height offset between the two surfaces; vertical placement stays with the caller.
    """
    a, b, tcp = (np.asarray(v, dtype=float) for v in (source, target, tool))
    if (a.ndim != 2 or a.shape[1:] != (3,) or len(a) < 3 or b.shape != a.shape
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("source and target need at least 3 corresponding finite XYZ landmarks")
    if (tcp.shape != (4, 4) or not np.isfinite(tcp).all() or not np.allclose(tcp[3], [0, 0, 0, 1])
            or not np.allclose(tcp[:3, :3].T @ tcp[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(tcp[:3, :3]), 1)):
        raise ValueError("tool must be a rigid 4x4 pose in the landmarks' coordinate frame")
    if isinstance(max_error_m, bool) or not np.isfinite(max_error_m) or max_error_m <= 0:
        raise ValueError("max_error_m must be positive and finite")
    ac, bc = a[:, :2].mean(axis=0), b[:, :2].mean(axis=0)
    x, y = a[:, :2] - ac, b[:, :2] - bc
    for points in (x, y):
        singular = np.linalg.svd(points, compute_uv=False)
        if singular[0] < 1e-9 or singular[1] < singular[0] * .01:
            raise ValueError("landmarks must span an area; choose separated, non-collinear features")
    yaw = float(np.arctan2(np.sum(x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]), np.sum(x * y)))
    rotation = rot_z(yaw)
    shift = bc - rotation[:2, :2] @ ac
    errors = np.linalg.norm(a[:, :2] @ rotation[:2, :2].T + shift - b[:, :2], axis=1)
    dz = b[:, 2] - a[:, 2]
    height_error = float(np.max(np.abs(dz - np.median(dz))))
    valid = bool(max(float(errors.max()), height_error) <= max_error_m)
    proposed = tcp.copy()
    proposed[:3, :3] = rotation @ tcp[:3, :3]
    proposed[:2, 3] = rotation[:2, :2] @ tcp[:2, 3] + shift
    return dict(valid=valid, reason=None if valid else "landmarks disagree: recheck correspondence, depth or tilt",
                yaw_delta_deg=float(np.degrees(yaw)), translation_xy_m=shift.tolist(),
                errors_m=errors.tolist(), max_error_m=float(errors.max()), height_error_m=height_error,
                tool_pose=proposed.tolist() if valid else None)


def pose_error(T: np.ndarray, T_des: np.ndarray) -> np.ndarray:
    """6-vector (position error, orientation error) that takes T towards T_des, both in the base frame."""
    ep = T_des[:3, 3] - T[:3, 3]
    Re = T_des[:3, :3] @ T[:3, :3].T
    eo = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    return np.concatenate([ep, eo])


def rotation_log(R: np.ndarray) -> np.ndarray:
    """Axis * angle of a rotation matrix (the inverse of axis_angle)."""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    angle = float(np.arccos(c))
    if angle < 1e-9:
        return np.zeros(3)
    if np.pi - angle < 1e-6:                          # 180 deg: take the axis from the diagonal
        i = int(np.argmax(np.diag(R)))
        axis = (R + R.T)[:, i] / 4
        axis[i] = (R[i, i] + 1) / 2
        return axis / np.linalg.norm(axis) * angle
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
    return w * angle


def interpolate_rotation(R0: np.ndarray, R1: np.ndarray, s: float) -> np.ndarray:
    """Geodesic interpolation between two rotations, s in 0..1."""
    w = rotation_log(R1 @ R0.T)
    n = np.linalg.norm(w)
    return R0 if n < 1e-12 else axis_angle(w / n, n * s) @ R0


def angle(a, b) -> float:
    """The angle between two directions, rad."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)))


def aim(R: np.ndarray, approach, opens, point, jaws=None) -> np.ndarray:
    """The tool rotation nearest R whose `approach` axis (tool frame) points along `point` (base frame) and whose
    `opens` axis lies along `jaws` (sign-free: whichever sign turns least), or else as close to its current
    direction as pointing that way allows."""
    approach, opens = np.asarray(approach, float), np.asarray(opens, float)
    p = np.asarray(point, float) / np.linalg.norm(point)
    now = R @ opens
    if jaws is None:
        j = now - (now @ p) * p
        if np.linalg.norm(j) < 0.2:                    # the jaws lay nearly along it: carry them round the turn
            a0 = R @ approach
            w = np.cross(a0, p)
            turn = axis_angle(w, np.arctan2(np.linalg.norm(w), a0 @ p)) if np.linalg.norm(w) > 1e-9 else \
                axis_angle(now, np.pi)
            j = turn @ now
            j = j - (j @ p) * p
    else:
        j = np.asarray(jaws, float)
        j = j - (j @ p) * p
        if np.linalg.norm(j) < np.sin(np.radians(15)):
            raise ValueError("the jaws cannot open along the way the gripper points")
    j /= np.linalg.norm(j)
    if j @ now < 0:
        j = -j
    tool = np.column_stack([approach, opens, np.cross(approach, opens)])
    return np.column_stack([p, j, np.cross(p, j)]) @ tool.T
