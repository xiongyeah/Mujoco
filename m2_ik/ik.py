"""
M2 · 数值解 1：Jacobian 伪逆迭代（单步用 ``numpy.linalg.lstsq`` 稳定求 Δq）。

仅实现 ``method='pinv'``；DLS、零空间等后续再加。
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from m2_ik.kinematics import forward_kinematics, jacobian, pose_error_se3

IkMethod = Literal["pinv"]


def ik_pinv_step(
    q: np.ndarray,
    T_des: np.ndarray,
    *,
    rcond: float = 1e-4,
    step_scale: float = 1.0,
    position_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """一步：Δq = J⁺ e，q_next = q + step_scale * Δq。"""
    q = np.asarray(q, dtype=np.float64).reshape(7).copy()
    T = forward_kinematics(q)
    if position_only:
        e = T_des[:3, 3] - T[:3, 3]
        J = jacobian(q)[:3, :]
    else:
        e = pose_error_se3(T, T_des)
        J = jacobian(q)
    dq, *_ = np.linalg.lstsq(J, e, rcond=rcond)
    return q + step_scale * dq, e


def inverse_kinematics(
    target_pose: np.ndarray,
    q_init: np.ndarray,
    *,
    method: IkMethod = "pinv",
    max_iters: int = 500,
    tol: float = 7e-4,
    rcond: float = 1e-4,
    step_scale: float = 0.55,
    position_only: bool = False,
) -> tuple[np.ndarray, int, float]:
    """
    迭代逆解。

    Parameters
    ----------
    target_pose
        4×4 齐次目标位姿（与 ``forward_kinematics`` 同一末端定义）。
    q_init
        初始关节角 (7,)。
    method
        当前仅支持 ``'pinv'``。

    Returns
    -------
    q, iters, final_err_norm
    """
    if method != "pinv":
        raise NotImplementedError("当前仅实现 method='pinv'（Jacobian 伪逆迭代）")

    T_des = np.asarray(target_pose, dtype=np.float64).reshape(4, 4)
    q = np.asarray(q_init, dtype=np.float64).reshape(7).copy()
    final_err = np.inf
    it = 0
    for it in range(max_iters):
        q, e = ik_pinv_step(q, T_des, rcond=rcond, step_scale=step_scale, position_only=position_only)
        final_err = float(np.linalg.norm(e))
        if final_err < tol:
            break
    return q, it + 1, final_err
