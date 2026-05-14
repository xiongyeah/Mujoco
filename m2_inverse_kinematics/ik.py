"""数值逆运动学（M2）：阻尼最小二乘（DLS）+ 伪逆单步。"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .kinematics import fk_fr3_attachment, geometric_jacobian_world, pose_error_se3

IkMethod = Literal["dls", "pinv"]


def ik_pinv_step(
    q: np.ndarray,
    T_des: np.ndarray,
    *,
    rcond: float = 1e-4,
    step_scale: float = 1.0,
    position_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """单步纯伪逆（无显式阻尼，数值上用 lstsq/pinv 稳定）。"""
    q = np.asarray(q, dtype=np.float64).reshape(7).copy()
    T = fk_fr3_attachment(q)
    if position_only:
        e = T_des[:3, 3] - T[:3, 3]
        J = geometric_jacobian_world(q)[:3, :]
    else:
        e = pose_error_se3(T, T_des)
        J = geometric_jacobian_world(q)
    dq, *_ = np.linalg.lstsq(J, e, rcond=rcond)
    return q + step_scale * dq, e


def ik_dls_step(
    q: np.ndarray,
    T_des: np.ndarray,
    *,
    damping: float = 1e-2,
    step_scale: float = 1.0,
    position_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    单步 DLS：返回 (q_next, err_vec)。
    err_vec 为 3 或 6 维，与 J 使用的行数一致。
    """
    q = np.asarray(q, dtype=np.float64).reshape(7).copy()
    T = fk_fr3_attachment(q)
    if position_only:
        e = T_des[:3, 3] - T[:3, 3]
        J = geometric_jacobian_world(q)[:3, :]
    else:
        e = pose_error_se3(T, T_des)
        J = geometric_jacobian_world(q)

    m, n = J.shape
    I = np.eye(m, dtype=np.float64)
    dq = J.T @ np.linalg.solve(J @ J.T + (damping**2) * I, e)
    return q + step_scale * dq, e


def ik_solve(
    q_init: np.ndarray,
    T_des: np.ndarray,
    *,
    max_iters: int = 200,
    tol: float = 1e-4,
    damping: float = 1e-2,
    step_scale: float = 0.5,
    position_only: bool = False,
    method: IkMethod = "dls",
    rcond: float = 1e-4,
) -> tuple[np.ndarray, int, float]:
    """
    迭代 IK，返回 (q, iters, final_err_norm)。

    method:
        - ``"dls"``: 阻尼最小二乘单步（奇异附近更稳，默认）。
        - ``"pinv"``: 每步用 ``lstsq`` 伪逆（对应课程 M2「Jacobian 伪逆迭代」）。
    """
    q = np.asarray(q_init, dtype=np.float64).reshape(7).copy()
    final_err = np.inf
    it = 0
    for it in range(max_iters):
        if method == "dls":
            q, e = ik_dls_step(q, T_des, damping=damping, step_scale=step_scale, position_only=position_only)
        elif method == "pinv":
            q, e = ik_pinv_step(q, T_des, rcond=rcond, step_scale=step_scale, position_only=position_only)
        else:
            raise ValueError(f"unknown method: {method!r}")
        final_err = float(np.linalg.norm(e))
        if final_err < tol:
            break
    return q, it + 1, final_err


def manipulability_translation(J: np.ndarray) -> float:
    """
    平移子雅可比可操作度（与奇异距离粗指标）：w = sqrt(det(J_t J_t^T))，J_t 为前 3 行。
    越小越接近平移奇异。
    """
    Jt = np.asarray(J, dtype=np.float64)[:3, :]
    m = Jt @ Jt.T
    d = float(np.linalg.det(m))
    if d <= 0.0:
        return 0.0
    return float(np.sqrt(d))


def null_space_vector_from_jacobian(J: np.ndarray) -> np.ndarray:
    """
    最小奇异值对应的右奇异向量 v（单位范数），满足 J v ≈ 0（秩亏时近似零空间方向）。
    J 形状 (6, 7) 且满行秩时近似 1 维零空间方向。
    """
    J = np.asarray(J, dtype=np.float64)
    _, _, vh = np.linalg.svd(J, full_matrices=True)
    v = np.asarray(vh[-1, :], dtype=np.float64).reshape(7)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v
