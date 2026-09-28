"""Small rigid-body helpers. Poses are 4x4 homogeneous matrices; everything is SI (m, rad)."""
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
        axis = np.sqrt(np.clip((np.diag(R) + 1) / 2, 0, None))
        axis *= np.sign(np.array([1.0, R[0, 1] + 1e-12, R[0, 2] + 1e-12]))
        return axis / np.linalg.norm(axis) * angle
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
    return w * angle


def interpolate_rotation(R0: np.ndarray, R1: np.ndarray, s: float) -> np.ndarray:
    """Geodesic interpolation between two rotations, s in 0..1."""
    w = rotation_log(R1 @ R0.T)
    n = np.linalg.norm(w)
    return R0 if n < 1e-12 else axis_angle(w / n, n * s) @ R0
