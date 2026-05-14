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


def case_wrist_targets(
    q5_vals: np.ndarray,
    ref: np.ndarray,
) -> dict[str, list[tuple[float, np.ndarray]]]:
    """手腕奇异：修改 q5，FK 出目标位姿。"""
    targets: dict[str, list[tuple[float, np.ndarray]]] = {
        "joint": [], "task": [], "random": []
    }
    for q5 in q5_vals:
        q = ref.copy().astype(np.float64)
        q[4] = q5
        T = forward_kinematics(q)
        dist = abs(q5)
        targets["joint"].append((dist, T.copy()))
        for sign in [-1, 1]:
            Tt = T.copy()
            Tt[:3, 3] += np.array([sign * 0.05, 0.0, 0.0])
            targets["task"].append((dist, Tt))
    rng = np.random.default_rng(20260514)
    for _ in range(200):
        q5_pert = rng.uniform(-0.3, 0.3)
        q = ref.copy().astype(np.float64)
        q[4] = q5_pert
        T = forward_kinematics(q)
        targets["random"].append((abs(q5_pert), T.copy()))
    return targets


def case_elbow_targets(
    q3_vals: np.ndarray,
    ref: np.ndarray,
) -> dict[str, list[tuple[float, np.ndarray]]]:
    """肘部奇异：修改 q3 趋近上限（手臂伸直）。"""
    q3_max = float(_FRANKA_Q_MAX[2])
    targets: dict[str, list[tuple[float, np.ndarray]]] = {
        "joint": [], "task": [], "random": []
    }
    for q3 in q3_vals:
        q = ref.copy().astype(np.float64)
        q[2] = q3
        T = forward_kinematics(q)
        dist = abs(q3_max - q3)
        targets["joint"].append((dist, T.copy()))
        for sign in [-1, 1]:
            Tt = T.copy()
            Tt[:3, 3] += np.array([0.0, sign * 0.03, 0.0])
            targets["task"].append((dist, Tt))
    rng = np.random.default_rng(20260515)
    for _ in range(200):
        q3_pert = rng.uniform(-1.5, q3_max - 0.01)
        q = ref.copy().astype(np.float64)
        q[2] = q3_pert
        T = forward_kinematics(q)
        dist = abs(q3_max - q3_pert)
        targets["random"].append((dist, T.copy()))
    return targets


def case_shoulder_targets(
    q1_vals: np.ndarray,
    q3_vals: np.ndarray,
    ref: np.ndarray,
) -> dict[str, list[tuple[float, np.ndarray]]]:
    """肩部奇异：q1 与 q3 组合导致轴对齐。用最小奇异值比作为距离。"""
    targets: dict[str, list[tuple[float, np.ndarray]]] = {
        "joint": [], "task": [], "random": []
    }
    for q1 in q1_vals:
        for q3 in q3_vals:
            q = ref.copy().astype(np.float64)
            q[0] = q1
            q[2] = q3
            T = forward_kinematics(q)
            _, _, J = _task_error_and_jacobian(q, T, position_only=False)
            s = np.linalg.svd(J, compute_uv=False)
            dist = 1.0 - s[-1] / (s[0] + 1e-15)
            targets["joint"].append((dist, T.copy()))
            for sign in [-1, 1]:
                Tt = T.copy()
                Tt[:3, 3] += np.array([sign * 0.04, 0.0, 0.0])
                targets["task"].append((dist, Tt))
    rng = np.random.default_rng(20260516)
    for _ in range(200):
        q1_pert = rng.uniform(-0.5, 0.5)
        q3_pert = np.pi / 2 + rng.uniform(-0.5, 0.5)
        q = ref.copy().astype(np.float64)
        q[0] = q1_pert
        q[2] = q3_pert
        T = forward_kinematics(q)
        _, _, J = _task_error_and_jacobian(q, T, position_only=False)
        s = np.linalg.svd(J, compute_uv=False)
        dist = 1.0 - s[-1] / (s[0] + 1e-15)
        targets["random"].append((dist, T.copy()))
    return targets


def case_boundary_targets(
    q6_vals: np.ndarray,
    ref: np.ndarray,
) -> dict[str, list[tuple[float, np.ndarray]]]:
    """边界奇异：q6 趋近上限。"""
    q6_max = float(_FRANKA_Q_MAX[5])
    targets: dict[str, list[tuple[float, np.ndarray]]] = {
        "joint": [], "task": [], "random": []
    }
    for q6 in q6_vals:
        q = ref.copy().astype(np.float64)
        q[5] = q6
        T = forward_kinematics(q)
        dist = abs(q6_max - q6)
        targets["joint"].append((dist, T.copy()))
        for sign in [-1, 1]:
            Tt = T.copy()
            Tt[:3, 3] += np.array([sign * 0.03, 0.0, 0.0])
            targets["task"].append((dist, Tt))
    rng = np.random.default_rng(20260517)
    for _ in range(200):
        q6_pert = rng.uniform(2.0, q6_max - 0.01)
        q = ref.copy().astype(np.float64)
        q[5] = q6_pert
        T = forward_kinematics(q)
        dist = abs(q6_max - q6_pert)
        targets["random"].append((dist, T.copy()))
    return targets


def define_cases(
    seed: int,
) -> dict[str, dict[str, list[tuple[float, np.ndarray]]]]:
    """返回 {case_name: {sweep_mode: [(singular_dist, T_des), ...]}}。"""
    q5_vals = [
        -2.0, -1.0, -0.5, -0.2, -0.1, -0.05, -0.02, -0.01, -0.005,
        0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0,
    ]
    q3_max = float(_FRANKA_Q_MAX[2])
    q3_vals = np.linspace(-1.5, q3_max, 36).tolist()
    q1_vals = np.linspace(-0.5, 0.5, 7).tolist()
    q3_vals_s = np.linspace(np.pi / 2 - 0.5, np.pi / 2 + 0.5, 7).tolist()

    cases = {
        "wrist": case_wrist_targets(np.array(q5_vals, dtype=np.float64), _REF_WRIST),
        "elbow": case_elbow_targets(np.array(q3_vals, dtype=np.float64), _REF_ELBOW),
        "shoulder": case_shoulder_targets(np.array(q1_vals, dtype=np.float64), np.array(q3_vals_s, dtype=np.float64), _REF_SHOULDER),
        "boundary": case_boundary_targets(np.linspace(2.0, _FRANKA_Q_MAX[5], 20), _REF_BOUNDARY),
    }
    return cases


def collect_records(
    cases: dict[str, dict[str, list[tuple[float, np.ndarray]]]],
    *,
    methods: list[str],
    tol: float,
    max_iters: int,
    rcond: float,
    step_scale: float,
) -> list[TrialRecord]:
    """遍历所有工况 × 逼近方式 × 方法，采集记录。"""
    records: list[TrialRecord] = []

    for case_name, sweep_dict in cases.items():
        for sweep_mode, target_list in sweep_dict.items():
            for singular_dist, T_des in target_list:
                q_init = np.zeros(7, dtype=np.float64)
                for method in methods:
                    q_sol, iters, final_err, max_dq = solve_and_record(
                        T_des,
                        q_init,
                        method=method,
                        tol=tol,
                        max_iters=max_iters,
                        rcond=rcond,
                        step_scale=step_scale,
                    )
                    manip = manipulability(q_sol)
                    converged = final_err < tol
                    records.append(TrialRecord(
                        case=case_name,
                        sweep_mode=sweep_mode,
                        singular_dist=singular_dist,
                        method=method,
                        converged=converged,
                        iters=iters,
                        final_err=final_err,
                        max_dq_norm=max_dq,
                        manipulability=manip,
                    ))
    return records


def write_csv(path: Path, records: list[TrialRecord]) -> None:
    """将记录写入 CSV 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "case", "sweep_mode", "singular_dist", "method",
            "converged", "iters", "final_err", "max_dq_norm", "manipulability",
        ])
        w.writeheader()
        for r in records:
            w.writerow({
                "case": r.case,
                "sweep_mode": r.sweep_mode,
                "singular_dist": f"{r.singular_dist:.6e}",
                "method": r.method,
                "converged": int(r.converged),
                "iters": r.iters,
                "final_err": f"{r.final_err:.6e}",
                "max_dq_norm": f"{r.max_dq_norm:.6e}",
                "manipulability": f"{r.manipulability:.6e}",
            })


@dataclass
class BucketStats:
    dist_mid: float
    count: int
    converge_rate: float
    mean_iters: float
    mean_max_dq: float
    mean_manip: float


def aggregate_by_dist_bins(
    records: list[TrialRecord],
    *,
    n_bins: int = 15,
) -> dict[tuple[str, str, str], dict[int, BucketStats]]:
    """按 (case, sweep_mode, method) 分桶聚合（对数分桶）。"""
    groups: dict[tuple[str, str, str], list[TrialRecord]] = {}
    for r in records:
        key = (r.case, r.sweep_mode, r.method)
        groups.setdefault(key, []).append(r)

    result: dict[tuple[str, str, str], dict[int, BucketStats]] = {}
    for key, recs in groups.items():
        dists = np.array([r.singular_dist for r in recs], dtype=np.float64)
        log_dists = np.log10(np.maximum(dists, 1e-15))
        bin_edges = np.linspace(log_dists.min(), log_dists.max(), n_bins + 1)
        bin_indices = np.digitize(log_dists, bin_edges) - 1
        buckets: dict[int, BucketStats] = {}
        for i in range(n_bins):
            mask = bin_indices == i
            if not mask.any():
                continue
            subset = [recs[j] for j in range(len(recs)) if mask[j]]
            dist_mid = 10.0 ** ((bin_edges[i] + bin_edges[i + 1]) / 2.0)
            conv = [r for r in subset if r.converged]
            buckets[i] = BucketStats(
                dist_mid=dist_mid,
                count=len(subset),
                converge_rate=len(conv) / len(subset),
                mean_iters=float(np.mean([r.iters for r in conv])) if conv else 0.0,
                mean_max_dq=float(np.nanmean([r.max_dq_norm for r in subset])),
                mean_manip=float(np.nanmean([r.manipulability for r in subset])),
            )
        result[key] = buckets
    return result


def plot_sweep_comparison(
    agg: dict[tuple[str, str, str], dict[int, BucketStats]],
    sweep_mode: str,
    *,
    out_png: Path,
    methods: list[str],
) -> None:
    """每张图 4 行（四种工况）× 3 列（收敛率 / ‖Δq‖ / 可操作度）。"""
    import matplotlib.pyplot as plt

    cases = ["wrist", "elbow", "shoulder", "boundary"]
    case_labels = ["手腕奇异", "肘部奇异", "肩部奇异", "边界奇异"]
    colors = {"pinv": "#d62728", "dls_0.02": "#2ca02c", "dls_0.05": "#1f77b4", "dls_0.2": "#9467bd"}
    labels_map = {"pinv": "伪逆"}

    fig, axes = plt.subplots(4, 3, figsize=(14, 16), constrained_layout=True)
    fig.suptitle(f"奇异点稳定性 — 逼近方式: {sweep_mode}", fontsize=14)

    for row, case in enumerate(cases):
        for method in methods:
            key = (case, sweep_mode, method)
            buckets = agg.get(key, {})
            if not buckets:
                continue
            bin_ids = sorted(buckets.keys())
            dists = [buckets[i].dist_mid for i in bin_ids]
            converge = [buckets[i].converge_rate for i in bin_ids]
            dq_vals = [buckets[i].mean_max_dq for i in bin_ids]
            manip = [buckets[i].mean_manip for i in bin_ids]
            label = labels_map.get(method, method)
            c = colors.get(method, "#333")

            axes[row, 0].semilogx(dists, converge, marker=".", color=c, label=label)
            axes[row, 0].set_ylabel("收敛率")
            axes[row, 0].grid(True, alpha=0.3)

            axes[row, 1].loglog(dists, dq_vals, marker=".", color=c, label=label)
            axes[row, 1].set_ylabel(r"最大 $\|\Delta q\|$")
            axes[row, 1].grid(True, alpha=0.3)

            axes[row, 2].semilogx(dists, manip, marker=".", color=c, label=label)
            axes[row, 2].set_ylabel("可操作度")
            axes[row, 2].grid(True, alpha=0.3)

        for col in range(3):
            axes[row, col].set_xlabel("奇异距离")
            axes[row, col].set_title(f"{case_labels[row]}")
            if row == 0 and col == 0:
                axes[row, col].legend(fontsize=8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
