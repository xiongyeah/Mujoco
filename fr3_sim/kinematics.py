"""FR3 7 轴链式正运动学 + 几何雅可比（与 menagerie fr3.xml 几何一致）。"""

from __future__ import annotations

import numpy as np

from fr3_sim.se3 import homogeneous, quat_wxyz_to_R, rotz, trans


def _R_wxyz(w: float, x: float, y: float, z: float) -> np.ndarray:
    return quat_wxyz_to_R(w, x, y, z)


def _segment_fixed_A(j: int) -> np.ndarray:
    """第 j 关节对应的固定齐次变换 A_j（不含 Rz(q_j)）。"""
    if j == 0:
        return trans(0.0, 0.0, 0.333)
    if j == 1:
        return homogeneous(_R_wxyz(1.0, -1.0, 0.0, 0.0), np.zeros(3))
    if j == 2:
        return trans(0.0, -0.316, 0.0) @ homogeneous(_R_wxyz(1.0, 1.0, 0.0, 0.0), np.zeros(3))
    if j == 3:
        return trans(0.0825, 0.0, 0.0) @ homogeneous(_R_wxyz(1.0, 1.0, 0.0, 0.0), np.zeros(3))
    if j == 4:
        return trans(-0.0825, 0.384, 0.0) @ homogeneous(_R_wxyz(1.0, -1.0, 0.0, 0.0), np.zeros(3))
    if j == 5:
        return homogeneous(_R_wxyz(1.0, 1.0, 0.0, 0.0), np.zeros(3))
    if j == 6:
        return trans(0.088, 0.0, 0.0) @ homogeneous(_R_wxyz(1.0, 1.0, 0.0, 0.0), np.zeros(3))
    raise IndexError(j)


def fk_fr3_link7(q: np.ndarray) -> np.ndarray:
    """基座 -> fr3_link7 原点，4x4 齐次矩阵（世界系）。"""
    q = np.asarray(q, dtype=np.float64).reshape(7)
    T = np.eye(4, dtype=np.float64)
    for j in range(7):
        Aj = _segment_fixed_A(j)
        T = T @ Aj @ homogeneous(rotz(q[j]), np.zeros(3))
    return T


def fk_fr3_attachment(q: np.ndarray) -> np.ndarray:
    """末端：link7 上 attachment_site (0,0,0.107)。"""
    return fk_fr3_link7(q) @ trans(0.0, 0.0, 0.107)


def geometric_jacobian_world(q: np.ndarray, ee_offset_local: np.ndarray | None = None) -> np.ndarray:
    """
    几何雅可比 J ∈ R^{6x7}，世界系：v = [pdot; omega] = J @ dq。
    ee_offset_local: link7 坐标系内末端偏移，默认 attachment_site。
    """
    q = np.asarray(q, dtype=np.float64).reshape(7)
    if ee_offset_local is None:
        ee_offset_local = np.array([0.0, 0.0, 0.107], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    z_cols: list[np.ndarray] = []
    o_cols: list[np.ndarray] = []

    for j in range(7):
        Aj = _segment_fixed_A(j)
        P = T @ Aj
        R = P[:3, :3]
        p = P[:3, 3]
        z_cols.append(R[:, 2])
        o_cols.append(p.copy())
        T = P @ homogeneous(rotz(q[j]), np.zeros(3))

    Tee = T @ trans(ee_offset_local[0], ee_offset_local[1], ee_offset_local[2])
    p_e = Tee[:3, 3]

    J = np.zeros((6, 7), dtype=np.float64)
    for j in range(7):
        z_j = z_cols[j]
        o_j = o_cols[j]
        J[:3, j] = np.cross(z_j, p_e - o_j)
        J[3:, j] = z_j
    return J


def pose_error_se3(T_cur: np.ndarray, T_des: np.ndarray) -> np.ndarray:
    """
    6 维误差 [p_err; rotvec]，与几何雅可比同一世界系约定：
    p_err = p_des - p_cur；R_err = R_des @ R_cur^T，rotvec = log(R_err)。
    """
    R_err = T_des[:3, :3] @ T_cur[:3, :3].T
    t = T_des[:3, 3] - T_cur[:3, 3]
    tr = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(tr))
    if theta < 1e-8:
        w = np.zeros(3, dtype=np.float64)
    else:
        w_hat = (R_err - R_err.T) / (2.0 * np.sin(theta))
        w = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]], dtype=np.float64) * theta
    return np.concatenate([t, w], axis=0)
