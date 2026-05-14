"""SE(3) 小工具：齐次变换、反对称矩阵（M2 逆解所需几何基础）。"""

from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v.ravel()
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_wxyz_to_R(w: float, x: float, y: float, z: float) -> np.ndarray:
    """单位四元数 (w, x, y, z) -> 3x3 旋转矩阵。"""
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def homogeneous(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p.ravel()
    return T


def trans(x: float, y: float, z: float) -> np.ndarray:
    return homogeneous(np.eye(3), np.array([x, y, z], dtype=np.float64))


def rotvec_to_R(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues：单位轴 axis，转角 angle（弧度）。"""
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    na = np.linalg.norm(a)
    if na < 1e-12:
        return np.eye(3)
    a = a / na
    K = skew(a)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
