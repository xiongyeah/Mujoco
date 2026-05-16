r"""零空间投影：利用冗余自由度做次级任务，不影响末端位姿。

与 ``INTERFACE_SPEC_v0.md`` 第 2.2 节对齐。

用法
----
>>> from m2_ik.null_space import inverse_kinematics_with_nullspace
>>>
>>> q_preferred = np.array([0.0, -0.5, 0.0, -1.8, 0.0, 2.0, 0.7])
>>> q, info = inverse_kinematics_with_nullspace(T_target, q_init, q_preferred)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from m2_ik.ik import _task_error_and_jacobian
from m2_ik.kinematics import forward_kinematics, jacobian, pose_error_se3

IkMethod = str  # "pinv" | "dls"


def null_space_projector(
    J: np.ndarray, rcond: float = 1e-4
) -> np.ndarray:
    r"""零空间投影矩阵 :math:`N = I - J^+ J`.

    Parameters
    ----------
    J
        几何雅可比 :math:`\in \mathbb{R}^{m \times 7}`, :math:`m \in \{3, 6\}`.
    rcond
        传给 ``np.linalg.pinv`` 的截断阈值。

    Returns
    -------
    N : (7, 7) — 任意 :math:`v` 经 :math:`N v` 投影后落在 :math:`J` 的零空间中。
    """
    n = J.shape[1]
    Jpinv = np.linalg.pinv(J, rcond=rcond)
    return np.eye(n, dtype=np.float64) - Jpinv @ J


def joint_limit_repulsion(
    q: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    margin: float = 0.1,
) -> np.ndarray:
    """关节限位排斥速度：当关节接近限位时生成推离速度。

    在 ``margin`` 范围内激活，越靠近限位排斥越强（二次函数）。
    返回与 ``q`` 同形的速度向量，正方向为远离下限、靠近上限。

    Parameters
    ----------
    q : (7,)
        当前关节角。
    q_min : (7,)
        关节下限。
    q_max : (7,)
        关节上限。
    margin : float
        激活区域宽度（弧度，默认 0.1 rad）。

    Returns
    -------
    v : (7,)
        关节空间速度，经零空间投影后即可在保末端位姿的前提下推离限位。
    """
    v = np.zeros_like(q)
    for i in range(len(q)):
        d_low = q[i] - q_min[i]
        d_high = q_max[i] - q[i]
        if 0.0 < d_low < margin:
            v[i] = (1.0 / d_low - 1.0 / margin) * margin
        elif 0.0 < d_high < margin:
            v[i] = -(1.0 / d_high - 1.0 / margin) * margin
    return v


def manipulability(q: np.ndarray) -> float:
    r"""可操作度 :math:`w(q) = \sqrt{\det(J J^\top)}`。

    注意
    ----
    数值上保证 ``det`` 非负，避免近奇异时因浮点误差产生负值导致 ``sqrt`` 失败。
    """
    J = jacobian(q)
    d = float(np.linalg.det(J @ J.T))
    return float(np.sqrt(max(d, 0.0)))


def manipulability_gradient(
    q: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """可操作度梯度 :math:`\nabla w(q)`（数值微分）。"""
    w0 = manipulability(q)
    grad = np.zeros(7, dtype=np.float64)
    for i in range(7):
        qp = q.copy()
        qp[i] += eps
        grad[i] = (manipulability(qp) - w0) / eps
    return grad


def _compute_nullspace_push(
    q: np.ndarray,
    objectives: list[tuple[Callable[[np.ndarray], np.ndarray], float]],
    *,
    rcond: float,
) -> np.ndarray:
    r"""计算零空间投影项 :math:`\sum_i \alpha_i \, N \, v_i(q)`。

    Parameters
    ----------
    q : (7,)
        当前关节角。
    objectives : list of (fn, weight)
        每个目标由 ``fn(q) → (7,)`` 速度向量与对应权重 ``weight`` 组成。
        所有权重项求和后经零空间投影：:math:`\sum \alpha_i N v_i`。

    Returns
    -------
    dq_null : (7,)
    """
    if not objectives:
        return np.zeros(7, dtype=np.float64)

    J = jacobian(q)
    N = null_space_projector(J, rcond=rcond)

    v = np.zeros(7, dtype=np.float64)
    for fn, weight in objectives:
        v = v + weight * fn(q)
    return N @ v


def inverse_kinematics_with_nullspace(
    target_pose: np.ndarray,
    q_init: np.ndarray,
    q_preferred: np.ndarray | None = None,
    nullspace_gain: float = 0.1,
    *,
    # ── 可选的原始回调模式（与 q_preferred 二选一） ──
    null_space_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    # ── 零空间多目标扩展 ──
    joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
    joint_limit_margin: float = 0.1,
    joint_limit_gain: float = 0.1,
    manipulability_gain: float = 0.0,
    # ── 透传给主 IK 的参数 ──
    method: IkMethod = "dls",
    damping: float = 0.05,
    max_iter: int = 200,
    tol_pos: float = 1e-4,
    tol_rot: float = 1e-3,
    rcond: float = 1e-4,
    step_scale: float = 0.55,
    position_only: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""带零空间投影的迭代逆解，支持多个次级目标。

    每步迭代：
    :math:`q \leftarrow q + \alpha \Delta q_{\mathrm{task}}
    + \sum_i \alpha_i \, N(q) \, v_i(q)`

    其中 :math:`N = I - J^+ J` 为零空间投影。

    支持三类次级目标（可叠加）：
    - ``q_preferred``：拉向参考位形
    - ``joint_limits``：关节限位排斥（零空间中推离边界）
    - ``manipulability_gain > 0``：最大化可操作度（远离奇异）

    Parameters
    ----------
    target_pose : (4, 4)
        目标末端位姿。
    q_init : (7,)
        迭代初值。
    q_preferred : (7,) or None
        期望的参考关节角（零空间中拉向该目标）。
    nullspace_gain : float
        ``q_preferred`` 目标权重 :math:`\beta`（默认 0.1）。
    joint_limits : (q_min, q_max) or None
        关节限位，每项 ``(7,)``。指定后在零空间中添加限位排斥目标。
    joint_limit_margin : float
        限位排斥激活区域宽度（弧度，默认 0.1）。
    joint_limit_gain : float
        限位排斥权重（默认 0.1）。
    manipulability_gain : float
        可操作度梯度权重。若 > 0，则在零空间中沿 :math:`\nabla w(q)` 方向
        移动以最大化可操作度（远离奇异）。默认 0（关闭）。

    Returns
    -------
    q : (7,)
    info : dict
        同 ``m2_ik.ik.inverse_kinematics``，额外含 ``'nullspace_objectives'`` 列表。
    """
    T_des = np.asarray(target_pose, dtype=np.float64).reshape(4, 4)
    q = np.asarray(q_init, dtype=np.float64).reshape(7).copy()
    it = 0

    # ── 组装零空间多目标列表 ──
    if q_preferred is not None and null_space_fn is not None:
        raise ValueError("q_preferred 与 null_space_fn 不能同时指定")

    objectives: list[tuple[Callable[[np.ndarray], np.ndarray], float]] = []

    if q_preferred is not None:
        q_pref_const = q_preferred  # 闭包捕获
        objectives.append((lambda qq: q_pref_const - qq, nullspace_gain))
    elif null_space_fn is not None:
        objectives.append((null_space_fn, nullspace_gain))

    if joint_limits is not None:
        q_min, q_max = joint_limits
        objectives.append((
            lambda qq: joint_limit_repulsion(qq, q_min, q_max, joint_limit_margin),
            joint_limit_gain,
        ))

    if manipulability_gain > 0.0:
        objectives.append((manipulability_gradient, manipulability_gain))

    for it in range(max_iter):
        # --- 误差与雅可比 ---
        T = forward_kinematics(q)
        e6 = pose_error_se3(T, T_des)
        J = jacobian(q)

        e = e6[:3] if position_only else e6
        Jp = J[:3, :] if position_only else J

        # --- 主任务步 ---
        if method == "pinv":
            dq, *_ = np.linalg.lstsq(Jp, e, rcond=rcond)
        elif method == "dls":
            if damping <= 0.0:
                raise ValueError("damping (λ) 必须为正标量")
            m = Jp.shape[0]
            aat = Jp @ Jp.T + (damping * damping) * np.eye(m, dtype=np.float64)
            x = np.linalg.solve(aat, np.asarray(e, dtype=np.float64).reshape(m))
            dq = Jp.T @ x
        else:
            raise ValueError(f"未知 method: {method!r}")

        # --- 零空间次级任务（多目标组合） ---
        dq_null = _compute_nullspace_push(q, objectives, rcond=rcond)

        q = q + step_scale * dq + dq_null
        if joint_limits is not None:
            np.clip(q, joint_limits[0], joint_limits[1], out=q)

        # --- 收敛检查（完整 FK 验算） ---
        T = forward_kinematics(q)
        e6_check = pose_error_se3(T, T_des)
        pos_err = float(np.linalg.norm(e6_check[:3]))
        rot_err = float(np.linalg.norm(e6_check[3:]))
        if pos_err < tol_pos and rot_err < tol_rot:
            break

    # 退出循环后：如果指定了 joint_limits，裁剪最终 q 并重新验算误差
    if joint_limits is not None:
        np.clip(q, joint_limits[0], joint_limits[1], out=q)
    T = forward_kinematics(q)
    e6_check = pose_error_se3(T, T_des)
    pos_err = float(np.linalg.norm(e6_check[:3]))
    rot_err = float(np.linalg.norm(e6_check[3:]))

    info: dict[str, Any] = {
        "converged": pos_err < tol_pos and rot_err < tol_rot,
        "iters": it + 1,
        "pos_err": pos_err,
        "rot_err": rot_err,
    }
    return q, info
