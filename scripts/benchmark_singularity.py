r"""
FR3 全奇异类型稳定性 Benchmark：伪逆 vs DLS。

覆盖四种奇异工况（手腕/肘部/肩部/边界）× 三种逼近方式
（关节空间扫描／任务空间穿越／随机批量）。

输出：outputs/singularity_benchmark/ 下的 PNG 图与 CSV。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from m2_ik import forward_kinematics
from m2_ik.ik import _task_error_and_jacobian
from m2_ik.kinematics import jacobian

# Franka Research 3 关节限位
_FRANKA_Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0873, -2.8973], dtype=np.float64)
_FRANKA_Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float64)

# 各奇异工况的参考位形（接近奇异但不太过）
_REF_WRIST = np.array([0.0, -0.35, 0.4, -1.35, 0.0, 1.55, -0.25], dtype=np.float64)
_REF_ELBOW = np.array([0.0, -0.35, -0.5, -1.35, 0.05, 1.55, -0.25], dtype=np.float64)
_REF_SHOULDER = np.array([0.0, -0.35, np.pi / 2, -1.35, 0.05, 1.55, -0.25], dtype=np.float64)
_REF_BOUNDARY = np.array([0.0, -0.35, 0.4, -1.35, 0.05, 2.5, -0.25], dtype=np.float64)


@dataclass
class TrialRecord:
    case: str          # wrist / elbow / shoulder / boundary
    sweep_mode: str    # joint / task / random
    singular_dist: float
    method: str        # pinv / dls_0.02 / dls_0.05 / dls_0.2
    converged: bool
    iters: int
    final_err: float
    max_dq_norm: float
    manipulability: float


def manipulability(q: np.ndarray) -> float:
    """可操作度 w = prod(σ_i)，基于 SVD 数值稳定。"""
    s = np.linalg.svd(jacobian(q), compute_uv=False)
    return float(np.prod(s))


def solve_and_record(
    T_des: np.ndarray,
    q_init: np.ndarray,
    *,
    method: str,
    tol: float,
    max_iters: int,
    rcond: float,
    step_scale: float,
) -> tuple[np.ndarray, int, float, float]:
    """返回 (q_sol, iters, final_err, max_dq_norm)。

    与 ``inverse_kinematics`` 不同——此函数记录并返回循环中的最大 ‖Δq‖。
    """
    q = np.asarray(q_init, dtype=np.float64).reshape(7)
    max_dq = 0.0
    it = 0

    for it in range(max_iters):
        q, e, J = _task_error_and_jacobian(q, T_des, position_only=False)

        if method.startswith("dls"):
            lam = float(method.split("_")[1])
            m = J.shape[0]
            aat = J @ J.T + lam * lam * np.eye(m, dtype=np.float64)
            x = np.linalg.solve(aat, np.asarray(e, dtype=np.float64).reshape(m))
            dq = J.T @ x
        else:
            dq, *_ = np.linalg.lstsq(J, e, rcond=rcond)

        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > max_dq:
            max_dq = dq_norm

        q = q + step_scale * dq
        final_err = float(np.linalg.norm(e))
        if final_err < tol:
            break

    return q, it + 1, final_err, max_dq
