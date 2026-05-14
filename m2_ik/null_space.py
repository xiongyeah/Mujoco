"""零空间投影：利用冗余自由度做次级任务，不影响末端位姿。

用法
----
>>> from m2_ik.null_space import inverse_kinematics
>>>
>>> def elbow_up(q):
...     q_ref = np.array([0.0, -0.5, 0.0, -1.8, 0.0, 2.0, 0.7])
...     return q_ref - q
>>>
>>> q, iters, err = inverse_kinematics(T_target, q_init, null_space_fn=elbow_up)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from m2_ik.kinematics import forward_kinematics, jacobian, pose_error_se3

IkMethod = Literal["pinv", "dls"]


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


def inverse_kinematics(
    target_pose: np.ndarray,
    q_init: np.ndarray,
    *,
    null_space_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    null_space_gain: float = 0.1,
    method: IkMethod = "dls",
    max_iters: int = 500,
    tol: float = 7e-4,
    rcond: float = 1e-4,
    step_scale: float = 0.55,
    position_only: bool = False,
    damping: float = 0.05,
) -> tuple[np.ndarray, int, float]:
    r"""带零空间投影的迭代逆解。

    每步迭代：
    :math:`q \leftarrow q + \alpha \Delta q_{\text{task}} + \beta\, N(q)\, f(q)`

    其中 :math:`N = I - J^+ J` 为零空间投影，:math:`f(q)` 由 ``null_space_fn`` 给出。

    Parameters
    ----------
    null_space_fn
        回调 ``f(q) -> (7,)``，返回*期望的关节空间推力*（如梯度或参考位形偏差）。
        ``None`` 时退化为普通 IK（行为同 ``m2_ik.ik.inverse_kinematics``）。
    null_space_gain
        零空间步长缩放 :math:`\beta`（通常 0.05~0.3，过大可能干扰主任务收敛）。
    method
        ``'pinv'`` 或 ``'dls'``（默认 ``'dls'``，与 ``m2_ik.ik.inverse_kinematics`` 一致）。
    damping
        仅 ``method='dls'`` 时有效。

    Returns
    -------
    q, iters, final_err_norm
    """
    T_des = np.asarray(target_pose, dtype=np.float64).reshape(4, 4)
    q = np.asarray(q_init, dtype=np.float64).reshape(7)
    final_err = np.inf
    it = 0

    for it in range(max_iters):
        # --- 当前误差与雅可比 ---
        T = forward_kinematics(q)
        if position_only:
            e = T_des[:3, 3] - T[:3, 3]
            J = jacobian(q)[:3, :]
        else:
            e = pose_error_se3(T, T_des)
            J = jacobian(q)

        # --- 主任务步 ---
        if method == "pinv":
            dq_task, *_ = np.linalg.lstsq(J, e, rcond=rcond)
        elif method == "dls":
            if damping <= 0.0:
                raise ValueError("damping (λ) 必须为正标量")
            m = J.shape[0]
            aat = J @ J.T + (damping * damping) * np.eye(m, dtype=np.float64)
            x = np.linalg.solve(aat, np.asarray(e, dtype=np.float64).reshape(m))
            dq_task = J.T @ x
        else:
            raise ValueError(f"未知 method: {method!r}，应为 'pinv' 或 'dls'")

        # --- 零空间次级任务 ---
        dq_null = np.zeros(7, dtype=np.float64)
        if null_space_fn is not None:
            N = null_space_projector(J, rcond=rcond)
            dq_null = null_space_gain * N @ null_space_fn(q)

        q = q + step_scale * dq_task + dq_null

        final_err = float(np.linalg.norm(e))
        if final_err < tol:
            break

    return q, it + 1, final_err
