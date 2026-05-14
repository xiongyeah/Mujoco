r"""
M2 · 数值逆解：Jacobian 伪逆迭代 + 阻尼最小二乘（DLS）。

- 伪逆单步：``lstsq(J, e)``，等价最小范数解 :math:`J^+ e`（行满秩、相容时）。
- DLS 单步：:math:`\Delta q = J^{\top}(JJ^{\top}+\lambda^2 I)^{-1} e`
  （与 ``docs/M2_DLS_阻尼最小二乘_数学推导.md`` 右形式一致）。
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from m2_ik.kinematics import forward_kinematics, jacobian, pose_error_se3

IkMethod = Literal["pinv", "dls"]


def _task_error_and_jacobian(
    q: np.ndarray,
    T_des: np.ndarray,
    *,
    position_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """当前 q 下：FK 得 T，再算任务误差 e 与雅可比 J。"""
    q = np.asarray(q, dtype=np.float64).reshape(7)
    T = forward_kinematics(q)
    if position_only:
        e = T_des[:3, 3] - T[:3, 3]
        J = jacobian(q)[:3, :]
    else:
        e = pose_error_se3(T, T_des)
        J = jacobian(q)
    return q, e, J


def ik_pinv_step(
    q: np.ndarray,
    T_des: np.ndarray,
    *,
    rcond: float = 1e-4,
    step_scale: float = 1.0,
    position_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    伪逆一步：:math:`\Delta q = J^+ e`（``lstsq``），:math:`q \leftarrow q + \alpha \Delta q`。
    """
    q, e, J = _task_error_and_jacobian(q, T_des, position_only=position_only)
    dq, *_ = np.linalg.lstsq(J, e, rcond=rcond)
    return q + step_scale * dq, e


def ik_dls_step(
    q: np.ndarray,
    T_des: np.ndarray,
    *,
    damping: float,
    step_scale: float = 1.0,
    position_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    DLS 一步：:math:`\Delta q = J^{\top}(JJ^{\top}+\lambda^2 I)^{-1} e`，
    :math:`q \leftarrow q + \alpha \Delta q`。

    Parameters
    ----------
    damping
        阻尼系数 :math:`\lambda>0`。对应 Tikhonov 项 :math:`\lambda^2 \lVert \Delta q \rVert^2`。
    """
    if damping <= 0.0:
        raise ValueError("damping (λ) 必须为正标量")

    q, e, J = _task_error_and_jacobian(q, T_des, position_only=position_only)
    m = J.shape[0]
    # Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e
    aat = J @ J.T + (damping * damping) * np.eye(m, dtype=np.float64)
    x = np.linalg.solve(aat, np.asarray(e, dtype=np.float64).reshape(m))
    dq = J.T @ x
    return q + step_scale * dq, e


def inverse_kinematics(
    target_pose: np.ndarray,
    q_init: np.ndarray,
    *,
    method: IkMethod = "dls",
    max_iters: int = 500,
    tol: float = 7e-4,
    rcond: float = 1e-4,
    step_scale: float = 0.55,
    position_only: bool = False,
    damping: float = 0.05,
) -> tuple[np.ndarray, int, float]:
    r"""
    迭代逆解。

    Parameters
    ----------
    method
        ``'pinv'``：伪逆单步；``'dls'``：阻尼最小二乘单步（默认 ``'dls'``）。
    damping
        仅 ``method='dls'`` 时使用，为 :math:`\lambda`（默认 ``0.05``）。

    Returns
    -------
    q, iters, final_err_norm
    """
    T_des = np.asarray(target_pose, dtype=np.float64).reshape(4, 4)
    q = np.asarray(q_init, dtype=np.float64).reshape(7).copy()
    final_err = np.inf
    it = 0

    for it in range(max_iters):
        if method == "pinv":
            q, e = ik_pinv_step(
                q,
                T_des,
                rcond=rcond,
                step_scale=step_scale,
                position_only=position_only,
            )
        elif method == "dls":
            q, e = ik_dls_step(
                q,
                T_des,
                damping=damping,
                step_scale=step_scale,
                position_only=position_only,
            )
        else:
            raise ValueError(f"未知 method: {method!r}，应为 'pinv' 或 'dls'")

        final_err = float(np.linalg.norm(e))
        if final_err < tol:
            break

    return q, it + 1, final_err
