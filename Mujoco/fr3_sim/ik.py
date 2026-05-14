"""数值逆运动学：阻尼最小二乘（DLS）+ 可选伪逆。"""

from __future__ import annotations

import numpy as np

from fr3_sim.kinematics import fk_fr3_attachment, geometric_jacobian_world, pose_error_se3


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
) -> tuple[np.ndarray, int, float]:
    """迭代 IK，返回 (q, iters, final_err_norm)。"""
    q = np.asarray(q_init, dtype=np.float64).reshape(7).copy()
    final_err = np.inf
    it = 0
    for it in range(max_iters):
        q, e = ik_dls_step(q, T_des, damping=damping, step_scale=step_scale, position_only=position_only)
        final_err = float(np.linalg.norm(e))
        if final_err < tol:
            break
    return q, it + 1, final_err
