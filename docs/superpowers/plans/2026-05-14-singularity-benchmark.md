# 奇异点稳定性 Benchmark 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 单脚本 `scripts/benchmark_singularity.py`，对 FR3 四种奇异类型（手腕/肘部/肩部/边界）× 三种逼近方式（关节空间扫/任务空间穿越/随机批量），统计 pinv 与 DLS 的收敛率、‖Δq‖ 爆炸程度、可操作度退化。

**Architecture:** 单一脚本，仿 `benchmark_ik_stack_poses.py` 的数据采集+聚合+出图三层。四种奇异工况各自定义扫描参数，共用同一套采集循环和可视化流水线。

**Tech Stack:** numpy, matplotlib, argparse, csv, dataclasses

---

## 文件结构

- **Create:** `scripts/benchmark_singularity.py` — 整个 benchmark
- **Create:** `docs/design_singularity_benchmark.md` — 已完成

无需修改现有 `m2_ik/` 的任何文件。

---

### Task 1: 公共结构与工具函数

**Files:**
- Create: `scripts/benchmark_singularity.py`（累积写入，本 Task 先写开头）

- [ ] **Step 1: 写文件头、导入、类型与常量**

```python
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
from typing import Literal, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from m2_ik import forward_kinematics, inverse_kinematics, pose_error_se3
from m2_ik.kinematics import jacobian

IkMethod = Literal["pinv", "dls"]

# Franka Research 3 关节限位
_FRANKA_Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0873, -2.8973], dtype=np.float64)
_FRANKA_Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float64)

# 各奇异工况的参考位形（接近奇异但不太过）
# wrists: q5 → 0; elbow: q3 → 上限; shoulder: q1≈0, q3≈π/2; boundary: q6 → 上限
_REF_WRIST = np.array([0.0, -0.35, 0.4, -1.35, 0.0, 1.55, -0.25], dtype=np.float64)
_REF_ELBOW = np.array([0.0, -0.35, -0.5, -1.35, 0.05, 1.55, -0.25], dtype=np.float64)
_REF_SHOULDER = np.array([0.0, -0.35, np.pi / 2, -1.35, 0.05, 1.55, -0.25], dtype=np.float64)
_REF_BOUNDARY = np.array([0.0, -0.35, 0.4, -1.35, 0.05, 2.5, -0.25], dtype=np.float64)
```

- [ ] **Step 2: 写数据记录结构体**

```python
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
```

- [ ] **Step 3: 写可操作度工具函数**

```python
def manipulability(q: np.ndarray) -> float:
    """可操作度 w = sqrt(det(J J^T))。"""
    J = jacobian(q)
    return float(np.sqrt(np.linalg.det(J @ J.T)))
```

- [ ] **Step 4: 写 IK 单次求解函数（记录 max_dq_norm）**

这个函数需要记录迭代过程中 ‖Δq‖ 的最大值，因此不能直接调 `inverse_kinematics`（它只返回最终结果）。需要自己实现一个循环，与 `ik.py` 的 `inverse_kinematics` 结构相同，但记录每一步的 dq_norm。

```python
def solve_and_record(
    T_des: np.ndarray,
    q_init: np.ndarray,
    *,
    method: str,
    damping: float,
    tol: float,
    max_iters: int,
    rcond: float,
    step_scale: float,
) -> tuple[np.ndarray, int, float, float]:
    """返回 (q_sol, iters, final_err, max_dq_norm)。"""
    from m2_ik.ik import _task_error_and_jacobian

    q = np.asarray(q_init, dtype=np.float64).reshape(7)
    max_dq = 0.0
    final_err = np.inf
    it = 0

    for it in range(max_iters):
        q, e, J = _task_error_and_jacobian(q, T_des, position_only=False)

        if method.startswith("dls"):
            lam = damping if method == "dls" else float(method.split("_")[1])
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
```

**注意**：这里 `import` 了 `m2_ik.ik._task_error_and_jacobian`（私有的），但这是在同项目的脚本中使用，是合理的。

---

### Task 2: 四种奇异工况的目标位姿生成

**Files:**
- Modify: `scripts/benchmark_singularity.py`（追加）

- [ ] **Step 1: 写手腕奇异目标位姿生成函数**

```python
def case_wrist_targets(
    q5_vals: np.ndarray,
    ref: np.ndarray,
) -> dict[str, list[tuple[float, np.ndarray]]]:
    """手腕奇异：修改 q5，FK 出目标位姿。
    返回 dict[sweep_mode, list[(singular_dist, T_des)]]。
    """
    targets: dict[str, list[tuple[float, np.ndarray]]] = {
        "joint": [], "task": [], "random": []
    }
    for q5 in q5_vals:
        q = ref.copy().astype(np.float64)
        q[4] = q5
        T = forward_kinematics(q)
        dist = abs(q5)  # 到奇异点的距离 = |q5|
        targets["joint"].append((dist, T.copy()))
        # 任务空间穿越：末端沿 x 方向平移 ±5cm，保持姿态不变
        for sign in [-1, 1]:
            Tt = T.copy()
            Tt[:3, 3] += np.array([sign * 0.05, 0.0, 0.0])
            targets["task"].append((dist, Tt))
    # 随机批量：在奇异参数附近采样
    rng = np.random.default_rng(20260514)
    for _ in range(200):
        q5_pert = rng.uniform(-0.3, 0.3)
        q = ref.copy().astype(np.float64)
        q[4] = q5_pert
        T = forward_kinematics(q)
        targets["random"].append((abs(q5_pert), T.copy()))
    return targets
```

- [ ] **Step 2: 写肘部奇异目标位姿生成函数**

```python
def case_elbow_targets(
    q3_vals: np.ndarray,
    ref: np.ndarray,
) -> dict[str, list[tuple[float, np.ndarray]]]:
    """肘部奇异：修改 q3 趋近上限（手臂伸直）。"""
    q3_max = float(_FRANKA_Q_MAX[2])  # -0.0698
    targets: dict[str, list[tuple[float, np.ndarray]]] = {
        "joint": [], "task": [], "random": []
    }
    for q3 in q3_vals:
        q = ref.copy().astype(np.float64)
        q[2] = q3
        T = forward_kinematics(q)
        dist = abs(q3_max - q3)
        targets["joint"].append((dist, T.copy()))
        # 任务穿越：沿垂直于前臂方向移动
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
```

- [ ] **Step 3: 写肩部奇异目标位姿生成函数**

```python
def case_shoulder_targets(
    q1_vals: np.ndarray,
    q3_vals: np.ndarray,
    ref: np.ndarray,
) -> dict[str, list[tuple[float, np.ndarray]]]:
    """肩部奇异：q1 与 q3 组合导致轴对齐。用最小奇异值作为距离度量。"""
    from m2_ik.ik import _task_error_and_jacobian

    targets: dict[str, list[tuple[float, np.ndarray]]] = {
        "joint": [], "task": [], "random": []
    }
    for q1 in q1_vals:
        for q3 in q3_vals:
            q = ref.copy().astype(np.float64)
            q[0] = q1
            q[2] = q3
            T = forward_kinematics(q)
            # 距离 = 1 - 奇异值之比（越小越接近奇异）
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
        # q3 围绕 π/2 扰动
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
```

- [ ] **Step 4: 写边界奇异目标位姿生成函数**

```python
def case_boundary_targets(
    q6_vals: np.ndarray,
    ref: np.ndarray,
) -> dict[str, list[tuple[float, np.ndarray]]]:
    """边界奇异：q6 趋近上限。"""
    q6_max = float(_FRANKA_Q_MAX[5])  # 3.7525
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
```

- [ ] **Step 5: 注册所有工况**

```python
def define_cases(
    seed: int,
) -> dict[str, dict[str, list[tuple[float, np.ndarray]]]]:
    """返回 {case_name: {sweep_mode: [(singular_dist, T_des), ...]}}。"""
    # 手腕奇异：q5 在 0 附近加密
    q5_vals = [
        -2.0, -1.0, -0.5, -0.2, -0.1, -0.05, -0.02, -0.01, -0.005,
        0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0,
    ]
    # 肘部奇异：q3 从 -1.5 线性扫到上限
    q3_max = float(_FRANKA_Q_MAX[2])
    q3_vals = np.linspace(-1.5, q3_max, 36).tolist()
    # 肩部奇异：q1 与 q3 组合扫描
    q1_vals = np.linspace(-0.5, 0.5, 7).tolist()
    q3_vals_s = np.linspace(np.pi / 2 - 0.5, np.pi / 2 + 0.5, 7).tolist()

    cases = {
        "wrist": case_wrist_targets(np.array(q5_vals, dtype=np.float64), _REF_WRIST),
        "elbow": case_elbow_targets(np.array(q3_vals, dtype=np.float64), _REF_ELBOW),
        "shoulder": case_shoulder_targets(np.array(q1_vals, dtype=np.float64), np.array(q3_vals_s, dtype=np.float64), _REF_SHOULDER),
        "boundary": case_boundary_targets(np.linspace(2.0, _FRANKA_Q_MAX[5], 20), _REF_BOUNDARY),
    }
    return cases
```

---

### Task 3: 采集流水线

**Files:**
- Modify: `scripts/benchmark_singularity.py`（追加）

- [ ] **Step 1: 写采集函数 + 求解循环**

```python
def collect_records(
    cases: dict[str, dict[str, list[tuple[float, np.ndarray]]]],
    *,
    methods: list[str],
    dampings: dict[str, float],
    tol: float,
    max_iters: int,
    rcond: float,
    step_scale: float,
) -> list[TrialRecord]:
    """遍历所有工况 × 逼近方式 × 方法，采集记录。"""
    records: list[TrialRecord] = []
    method_list: list[str] = methods[:]  # ["pinv", "dls_0.02", ...]

    for case_name, sweep_dict in cases.items():
        for sweep_mode, target_list in sweep_dict.items():
            for singular_dist, T_des in target_list:
                q_init = np.zeros(7, dtype=np.float64)
                for method in method_list:
                    if method == "pinv":
                        damping_val = 0.0
                    elif method.startswith("dls"):
                        damping_val = dampings[method]
                    else:
                        continue

                    t0 = time.perf_counter()
                    q_sol, iters, final_err, max_dq = solve_and_record(
                        T_des,
                        q_init,
                        method=method,
                        damping=damping_val,
                        tol=tol,
                        max_iters=max_iters,
                        rcond=rcond,
                        step_scale=step_scale,
                    )
                    _ = time.perf_counter() - t0  # 不记录 wall time
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
```

---

### Task 4: 聚合与可视化

**Files:**
- Modify: `scripts/benchmark_singularity.py`（追加）

- [ ] **Step 1: 写 CSV 输出函数**

```python
def write_csv(path: Path, records: list[TrialRecord]) -> None:
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
```

- [ ] **Step 2: 写聚合函数**

按 `(case, sweep_mode, method)` 将 singular_dist 分桶（每桶取均值）：

```python
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
    n_bins: int = 20,
) -> dict[tuple[str, str, str], list[BucketStats]]:
    """按 (case, sweep_mode, method) 分桶聚合。"""
    bins: dict[tuple[str, str, str], dict[int, list[TrialRecord]]] = {}
    for r in records:
        key = (r.case, r.sweep_mode, r.method)
        if key not in bins:
            bins[key] = {}
    # 对每个 key 做 log 分桶
    for key, recs in bins.items():
        dists = np.array([getattr(r, "singular_dist") for r in recs])
        if len(dists) == 0:
            continue
        log_dists = np.log10(np.maximum(dists, 1e-15))
        bin_edges = np.linspace(log_dists.min(), log_dists.max(), n_bins + 1)
        bin_indices = np.digitize(log_dists, bin_edges) - 1
        for i in range(n_bins):
            mask = bin_indices == i
            if not mask.any():
                continue
            subset = [recs[j] for j in range(len(recs)) if mask[j]]
            dist_mid = 10.0 ** ((bin_edges[i] + bin_edges[i + 1]) / 2.0)
            converged = [r for r in subset if r.converged]
            bins[key][i] = BucketStats(
                dist_mid=dist_mid,
                count=len(subset),
                converge_rate=len(converged) / len(subset),
                mean_iters=float(np.mean([r.iters for r in converged])) if converged else 0.0,
                mean_max_dq=float(np.mean([r.max_dq_norm for r in subset])),
                mean_manip=float(np.mean([r.manipulability for r in subset])),
            )
    return bins
```

- [ ] **Step 3: 写出图函数**

每张图 4 行（四种工况）× 3 列（收敛率 / ‖Δq‖ / 可操作度），每条线一种 method：

```python
def plot_sweep_comparison(
    agg: dict[tuple[str, str, str], dict[int, BucketStats]],
    sweep_mode: str,
    *,
    out_png: Path,
    methods: list[str],
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False

    cases = ["wrist", "elbow", "shoulder", "boundary"]
    case_labels = ["手腕奇异", "肘部奇异", "肩部奇异", "边界奇异"]
    colors = {"pinv": "#d62728", "dls_0.02": "#2ca02c", "dls_0.05": "#1f77b4", "dls_0.2": "#9467bd"}

    fig, axes = plt.subplots(4, 3, figsize=(14, 16), constrained_layout=True)
    fig.suptitle(f"奇异点稳定性 — 逼近方式: {sweep_mode}", fontsize=14)

    for row, case in enumerate(cases):
        for method in methods:
            key = (case, sweep_mode, method)
            bins = agg.get(key, {})
            if not bins:
                continue
            bin_ids = sorted(bins.keys())
            dists = [bins[i].dist_mid for i in bin_ids]
            conv = [bins[i].converge_rate for i in bin_ids]
            dq = [bins[i].mean_max_dq for i in bin_ids]
            manip = [bins[i].mean_manip for i in bin_ids]

            # 列 0: 收敛率
            ax = axes[row, 0]
            ax.semilogx(dists, conv, marker=".", color=colors.get(method, "#333"),
                        label={"pinv": "伪逆"}.get(method, method))
            ax.set_ylabel("收敛率")
            ax.grid(True, alpha=0.3)

            # 列 1: 最大 ‖Δq‖
            ax = axes[row, 1]
            ax.loglog(dists, dq, marker=".", color=colors.get(method, "#333"),
                      label={"pinv": "伪逆"}.get(method, method))
            ax.set_ylabel(r"最大 $\|\Delta q\|$")
            ax.grid(True, alpha=0.3)

            # 列 2: 可操作度
            ax = axes[row, 2]
            ax.semilogx(dists, manip, marker=".", color=colors.get(method, "#333"),
                        label={"pinv": "伪逆"}.get(method, method))
            ax.set_ylabel("可操作度")
            ax.grid(True, alpha=0.3)

        for col in range(3):
            axes[row, col].set_xlabel("奇异距离")
            axes[row, col].set_title(f"{case_labels[row]}")
            if row == 0 and col == 0:
                axes[row, col].legend(fontsize=8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
```

---

### Task 5: 主函数入口

**Files:**
- Modify: `scripts/benchmark_singularity.py`（追加）

- [ ] **Step 1: 写 argparse 和 main**

```python
def main() -> None:
    ap = argparse.ArgumentParser(description="FR3 全奇异类型稳定性 Benchmark")
    ap.add_argument("--damping", type=float, nargs="+", default=[0.02, 0.05, 0.2],
                    help="DLS λ 列表")
    ap.add_argument("--tol", type=float, default=7e-4)
    ap.add_argument("--max-iters", type=int, default=500)
    ap.add_argument("--step-scale", type=float, default=0.55)
    ap.add_argument("--rcond", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "singularity_benchmark")
    args = ap.parse_args()

    try:
        import matplotlib as mpl
    except ImportError as e:
        raise SystemExit("需要 matplotlib：请先 pip install -r requirements.txt") from e

    mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False

    # 方法列表
    dampings = {f"dls_{d:.4f}".rstrip("0").rstrip("."): d for d in args.damping}
    methods = ["pinv"] + list(dampings.keys())

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 定义工况
    cases = define_cases(args.seed)

    # 2. 采集
    records = collect_records(
        cases,
        methods=methods,
        dampings=dampings,
        tol=args.tol,
        max_iters=args.max_iters,
        rcond=args.rcond,
        step_scale=args.step_scale,
    )

    # 3. 写 CSV
    csv_path = out_dir / "singularity_benchmark_trials.csv"
    write_csv(csv_path, records)
    print(f"已写入: {csv_path}")

    # 4. 聚合
    agg = aggregate_by_dist_bins(records, n_bins=15)

    # 5. 出图：三种逼近方式各一张
    for sweep_mode in ("joint", "task", "random"):
        png_path = out_dir / f"sweep_{sweep_mode}.png"
        plot_sweep_comparison(agg, sweep_mode, out_png=png_path, methods=methods)
        print(f"已写入: {png_path}")


if __name__ == "__main__":
    main()
```

---

## 自检

对照 spec 逐项检查：

| Spec 需求 | 对应 Task | 覆盖 |
|-----------|----------|------|
| 四种奇异工况（手腕/肘部/肩部/边界） | Task 2 Step 1-4 | ✅ |
| 关节空间扫描逼近 | Task 2 各 case 函数中的 "joint" 分支 | ✅ |
| 任务空间穿越逼近 | Task 2 各 case 函数中的 "task" 分支 | ✅ |
| 随机批量逼近 | Task 2 各 case 函数中的 "random" 分支 | ✅ |
| 采集字段（converged/iters/final_err/max_dq_norm/manipulability） | Task 1 Step 2 的 TrialRecord | ✅ |
| 收敛率/‖Δq‖/迭代次数/可操作度可视化 | Task 4 Step 3 的 4×3 子图 | ✅ |
| CSV 输出 | Task 4 Step 1 | ✅ |
| CLI 参数（damping/tol/max-iters/step-scale/seed/out） | Task 5 Step 1 | ✅ |
| 不修改现有 m2_ik/ | 全部在新脚本中 | ✅ |
