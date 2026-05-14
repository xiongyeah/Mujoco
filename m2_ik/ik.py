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
    """
    伪逆迭代的「单步」：在当前 q 上算 e、J，解 JΔq≈e，再 q←q+αΔq。
    对应公式：Δq = J⁺e（lstsq 给出最小范数最小二乘意义下的 J⁺e），α = step_scale。
    """
    q = np.asarray(q, dtype=np.float64).reshape(7).copy()
    # ① 正运动学：由当前关节角算末端位姿 T(q)，用于构造误差
    T = forward_kinematics(q)
    if position_only:
        # ①′（仅位置模式）位置误差 e_p = p_d - p(q)；雅可比取平移前三行 J_p
        e = T_des[:3, 3] - T[:3, 3]
        J = jacobian(q)[:3, :]
    else:
        # ② 工作空间误差：e(q) ∈ R^6，与几何雅可比同一套空间速度约定
        e = pose_error_se3(T, T_des)
        # ③ 在当前 q 处计算几何雅可比 J(q)，用于一阶近似 JΔq ≈ e
        J = jacobian(q)
    # ④ 求本步关节增量：Δq = J⁺ e（超定/欠定均由 lstsq 稳定求解，行满秩时等价 J^T(JJ^T)^{-1}e）
    dq, *_ = np.linalg.lstsq(J, e, rcond=rcond)
    # ⑤ 关节更新：q^{k+1} = q^k + α·Δq（α 防止一步过大，保证线性化有效）
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

    # ⑥ 初始化：给定目标位姿 T_d 与迭代初值 q^{(0)}
    T_des = np.asarray(target_pose, dtype=np.float64).reshape(4, 4)
    q = np.asarray(q_init, dtype=np.float64).reshape(7).copy()
    final_err = np.inf
    it = 0
    # ⑦ 迭代直至 ‖e‖<tol 或达到最大次数（每轮调用 ik_pinv_step = 重复「线性化+伪逆一步」）
    for it in range(max_iters):
        q, e = ik_pinv_step(q, T_des, rcond=rcond, step_scale=step_scale, position_only=position_only)
        final_err = float(np.linalg.norm(e))
        # ⑧ 收敛判据：‖e(q)‖ < ε
        if final_err < tol:
            break
    return q, it + 1, final_err
