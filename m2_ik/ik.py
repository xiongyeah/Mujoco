r"""M2 · 数值逆解：Jacobian 伪逆迭代 + 阻尼最小二乘（DLS）。

与 ``INTERFACE_SPEC_v0.md`` 第 2 节对齐。

接口
----
- ``inverse_kinematics`` — 主入口，返回 ``(q, info_dict)``
- ``ik_pinv_step`` / ``ik_dls_step`` — 单步函数（内部/调试用）
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np

from m2_ik.kinematics import forward_kinematics, jacobian, pose_error_se3

IkMethod = Literal["pinv", "dls"]

# FR3 关节限位（Franka Research 3 规格，弧度）
FRANKA_Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0873, -2.8973])
FRANKA_Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])


class IKConvergenceError(RuntimeError):
    """IK 在最大迭代次数内未收敛时抛出（仅 strict 模式下）。"""

    def __init__(
        self,
        q_last: np.ndarray,
        pos_err: float,
        rot_err: float,
        iters: int,
        *,
        msg: str = "",
    ) -> None:
        self.q_last = q_last
        self.pos_err = pos_err
        self.rot_err = rot_err
        self.iters = iters
        super().__init__(
            msg
            or (
                f"IK 未收敛：迭代 {iters} 次后 "
                f"pos_err={pos_err:.2e}, rot_err={rot_err:.2e}"
            )
        )


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
    r"""伪逆一步：:math:`\Delta q = J^+ e`（``lstsq``），:math:`q \leftarrow q + \alpha \Delta q`。"""
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
    r"""DLS 一步：:math:`\Delta q = J^\top(J J^\top + \lambda^2 I)^{-1} e`。"""
    if damping <= 0.0:
        raise ValueError("damping (λ) 必须为正标量")
    q, e, J = _task_error_and_jacobian(q, T_des, position_only=position_only)
    m = J.shape[0]
    aat = J @ J.T + (damping * damping) * np.eye(m, dtype=np.float64)
    x = np.linalg.solve(aat, np.asarray(e, dtype=np.float64).reshape(m))
    dq = J.T @ x
    return q + step_scale * dq, e


def _split_rot_err(e6: np.ndarray) -> tuple[float, float]:
    """从 6 维误差 [Δp; ω] 中分解位置误差 (m) 与旋转误差 (rad)。"""
    return float(np.linalg.norm(e6[:3])), float(np.linalg.norm(e6[3:]))


def inverse_kinematics(
    target_pose: np.ndarray,
    q_init: np.ndarray,
    method: str = "dls",
    max_iter: int = 200,
    tol_pos: float = 1e-4,
    tol_rot: float = 1e-3,
    *,
    # ── 内部调优参数（调用者通常不需要关心） ──
    damping: float = 0.05,
    rcond: float = 1e-4,
    step_scale: float = 0.55,
    position_only: bool = False,
    strict: bool = False,
    joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""求解 FR3 的逆运动学。

    Parameters
    ----------
    target_pose : (4, 4)
        目标末端（flange）位姿，world frame。
    q_init : (7,)
        迭代初值。

        .. note::

            **初值策略**：FR3 是 7-DOF 冗余臂，数值 IK（伪逆 / DLS）是局部
            迭代算法，**不同 ``q_init`` 会收敛到不同解**（无穷多组 IK 解）。
            实际轨迹跟踪时：

            - **推荐**：将上一控制周期／上一时刻的实际 ``q`` 作为 ``q_init``
              传入，这样前后帧的解自然连贯，相邻时刻 IK 解不会跳变。
            - 若无上一时刻信息（首次求解），可尝试多组随机初值取最优，
              或使用零空间 IK（:func:`~m2_ik.null_space.inverse_kinematics_with_nullspace`）
              配合 ``q_preferred`` 指定期望构型。

    method : {'pinv', 'dls'}
        ``'pinv'`` — 伪逆；``'dls'`` — 阻尼最小二乘（默认）。
    max_iter : int
        最大迭代次数（默认 200）。
    tol_pos : float
        位置收敛阈值，单位 m（默认 1e-4）。
    tol_rot : float
        姿态收敛阈值，单位 rad（默认 1e-3）。
    joint_limits : (q_min, q_max) or None
        关节限位，每项为 ``(7,)`` 数组。指定后在每步迭代末尾将 ``q`` 裁剪到
        ``[q_min, q_max]`` 内，确保输出不超出物理限位。若为 ``None`` 则不检查。
        默认限位见 :data:`FRANKA_Q_MIN` / :data:`FRANKA_Q_MAX`。

    Returns
    -------
    q : (7,)
        求得的关节角（若 ``joint_limits`` 非空则保证在限位内）。
    info : dict
        - ``'converged'`` : bool
        - ``'iters'`` : int
        - ``'pos_err'`` : float
        - ``'rot_err'`` : float
        - ``'time'`` : float  (秒)

    Raises
    ------
    IKConvergenceError
        当 ``strict=True`` 且收敛失败时抛出。
    """
    T_des = np.asarray(target_pose, dtype=np.float64).reshape(4, 4)
    q = np.asarray(q_init, dtype=np.float64).reshape(7).copy()
    t_start = time.perf_counter()
    it = 0

    for it in range(max_iter):
        # FK → 误差 → 雅可比
        T = forward_kinematics(q)
        e6 = pose_error_se3(T, T_des)
        J = jacobian(q)

        e = e6[:3] if position_only else e6
        Jp = J[:3, :] if position_only else J

        # 一步更新
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
            raise ValueError(f"未知 method: {method!r}，应为 'pinv' 或 'dls'")

        q = q + step_scale * dq
        if joint_limits is not None:
            np.clip(q, joint_limits[0], joint_limits[1], out=q)

        # 收敛检查
        pos_err, rot_err = _split_rot_err(e6) if not position_only else (
            float(np.linalg.norm(e6[:3])), 0.0
        )
        if pos_err < tol_pos and rot_err < tol_rot:
            break

    elapsed = time.perf_counter() - t_start
    if joint_limits is not None:
        np.clip(q, joint_limits[0], joint_limits[1], out=q)
    pos_err, rot_err = _split_rot_err(pose_error_se3(forward_kinematics(q), T_des))
    converged = pos_err < tol_pos and rot_err < tol_rot

    if strict and not converged:
        raise IKConvergenceError(q, pos_err, rot_err, it + 1)

    info: dict[str, Any] = {
        "converged": converged,
        "iters": it + 1,
        "pos_err": pos_err,
        "rot_err": rot_err,
        "time": elapsed,
    }
    return q, info
