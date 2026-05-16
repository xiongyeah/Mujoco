r"""
堆积木关键位姿 IK 对比实验：抓取 / 抬起 / 放置 各 5 个目标位姿。

对伪逆 (``pinv``) 与阻尼最小二乘 (``dls``) 统计：
迭代次数、收敛精度（末端 SE(3) 误差范数）、单次求解 wall time、收敛失败率。

**注意**：同一 ``(phase, target_idx, trial)`` 下两种方法使用**完全相同**的随机初值 ``q0``，
并对 ``q0`` 按 Franka 关节上下限裁剪；初值在 ``--q0-mean``（默认与名义抓取参考一致）附近
``N(0, σ)`` 采样，否则 pinv/dls 对比会失去可比性。

输出：``outputs/ik_stack_benchmark/`` 下的 PNG 图与 CSV（可通过 ``--out`` 修改）。
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

IkMethod = Literal["pinv", "dls"]

# 堆积木 IK 实验参考构型（与 build_stack_target_poses 名义抓取一致）
Q_STACK_IK_REF = np.array([0.0, -0.35, 0.45, -1.4, 0.0, 1.5, -0.2], dtype=np.float64)

# Franka Research 3 / Panda 系列典型关节限位（弧度），用于随机初值裁剪
_FRANKA_Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0873, -2.8973], dtype=np.float64)
_FRANKA_Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float64)


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = (float(v[0]), float(v[1]), float(v[2]))
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _exp_so3_rotvec(v: np.ndarray) -> np.ndarray:
    """李代数 so(3) 指数映射：旋转向量 v（轴角 = v）→ R。"""
    v = np.asarray(v, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64) + _skew(v)
    k = v / theta
    K = _skew(k)
    s, c = np.sin(theta), np.cos(theta)
    return np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)


def _left_se3_noise(R_delta: np.ndarray, p_delta: np.ndarray) -> np.ndarray:
    Tn = np.eye(4, dtype=np.float64)
    Tn[:3, :3] = R_delta
    Tn[:3, 3] = np.asarray(p_delta, dtype=np.float64).reshape(3)
    return Tn


def perturb_pose_world(
    T: np.ndarray,
    rng: np.random.Generator,
    *,
    pos_sigma_m: float,
    rot_sigma_rad: float,
) -> np.ndarray:
    """世界系左乘小扰动：T' = exp([w]×, Δp) · T。"""
    dp = rng.normal(0.0, pos_sigma_m, size=3)
    w = rng.normal(0.0, rot_sigma_rad, size=3)
    return _left_se3_noise(_exp_so3_rotvec(w), dp) @ T


def build_stack_target_poses(
    pose_seed: int,
    *,
    pos_sigma_m: float = 0.012,
    rot_sigma_rad: float = 0.05,
    lift_dz_m: float = 0.05,
) -> dict[str, list[np.ndarray]]:
    """
    构造 15 个 4×4 目标位姿（抓取/抬起/放置 各 5 个）。

    名义位姿来自「积木桌面作业」典型构型：前方偏右、末端朝下接近竖直向下抓取，
    抬起为沿世界 **+z** 小幅平移（保持与抓取相同的末端姿态 **R**，仅抬高度），
    再对位置加毫米级扰动；**不再**对抬起做世界系左乘旋转扰动，否则易超出 FR3
    可达空间或使 6D 目标不可解（原先 +0.13 m 且左乘 SE(3) 噪声会导致抬起阶段大量失败）。

    放置为沿基座 x 负向叠放并略降 z，仍可对世界系做小 SE(3) 扰动。
    """
    T_grasp0 = forward_kinematics(Q_STACK_IK_REF)

    grasp: list[np.ndarray] = []
    for k in range(5):
        r = np.random.default_rng(pose_seed + 1_003 * k + 11)
        grasp.append(perturb_pose_world(T_grasp0, r, pos_sigma_m=pos_sigma_m, rot_sigma_rad=rot_sigma_rad))

    # 抬起：在抓取姿态 **R 不变** 的前提下 +z 平移；竖直位移过大时 FR3 常不可达。
    lift: list[np.ndarray] = []
    for k, Tg in enumerate(grasp):
        Tl = Tg.copy()
        Tl[:3, 3] = Tg[:3, 3] + np.array([0.0, 0.0, float(lift_dz_m)], dtype=np.float64)
        r = np.random.default_rng(pose_seed + 7_019 * k + 29)
        dp = r.normal(0.0, pos_sigma_m * 0.6, size=3)
        Tl[:3, 3] = Tl[:3, 3] + dp
        lift.append(Tl)

    place: list[np.ndarray] = []
    for k, Tl in enumerate(lift):
        Tp = Tl.copy()
        Tp[:3, 3] = Tl[:3, 3] + np.array([-0.09 - 0.018 * k, 0.01 * ((-1) ** k), -0.05], dtype=np.float64)
        r = np.random.default_rng(pose_seed + 13_031 * k + 41)
        place.append(perturb_pose_world(Tp, r, pos_sigma_m=pos_sigma_m, rot_sigma_rad=rot_sigma_rad))

    return {"grasp": grasp, "lift": lift, "place": place}


@dataclass
class TrialRecord:
    phase: str
    target_idx: int
    method: str
    trial: int
    converged: bool
    iters: int
    wall_time_s: float
    err_task_norm: float
    err_geom_norm: float


def run_trial(
    T_des: np.ndarray,
    q0: np.ndarray,
    *,
    method: IkMethod,
    damping: float,
    tol: float,
    max_iters: int,
    rcond: float,
    step_scale: float,
) -> tuple[bool, int, float, float, float]:
    """返回 (converged, iters, wall_time_s, err_task_norm, err_geom_norm)。"""
    t0 = time.perf_counter()
    q_sol, info = inverse_kinematics(
        T_des,
        q0,
        method=cast(IkMethod, method),
        damping=damping,
        max_iter=max_iters,
        tol_pos=tol,
        tol_rot=tol,
        rcond=rcond,
        step_scale=step_scale,
    )
    wall = time.perf_counter() - t0
    converged = info["converged"]
    err_task = max(info["pos_err"], info["rot_err"])
    iters = info["iters"]
    T_act = forward_kinematics(q_sol)
    e6 = pose_error_se3(T_act, T_des)
    err_geom = float(np.linalg.norm(e6))
    return converged, iters, wall, float(err_task), err_geom


def sample_q0(rng: np.random.Generator, *, spread: float, q_mean: np.ndarray) -> np.ndarray:
    """在 q_mean 附近高斯扰动，再按 FR3 关节上下限裁剪。"""
    q = np.asarray(q_mean, dtype=np.float64).reshape(7) + rng.normal(0.0, spread, size=7)
    return np.minimum(np.maximum(q, _FRANKA_Q_MIN), _FRANKA_Q_MAX)


def collect_records(
    poses: dict[str, list[np.ndarray]],
    *,
    methods: tuple[IkMethod, ...],
    trials_per_target: int,
    base_seed: int,
    damping: float,
    tol: float,
    max_iters: int,
    rcond: float,
    step_scale: float,
    q0_spread: float,
    q0_mean: np.ndarray,
) -> list[TrialRecord]:
    records: list[TrialRecord] = []
    phase_ids = {"grasp": 0, "lift": 1, "place": 2}
    for phase in ("grasp", "lift", "place"):
        pid = phase_ids[phase]
        for ti, T_des in enumerate(poses[phase]):
            for j in range(trials_per_target):
                # 同一 (phase, target, trial) 下 pinv / dls 共用同一初值，才可公平对比
                rng = np.random.default_rng(base_seed + pid * 50_000 + ti * 1_000 + j)
                q0 = sample_q0(rng, spread=q0_spread, q_mean=q0_mean)
                for method in methods:
                    conv, iters, wall, et, eg = run_trial(
                        T_des,
                        q0.copy(),
                        method=method,
                        damping=damping,
                        tol=tol,
                        max_iters=max_iters,
                        rcond=rcond,
                        step_scale=step_scale,
                    )
                    records.append(
                        TrialRecord(
                            phase=phase,
                            target_idx=ti,
                            method=method,
                            trial=j,
                            converged=conv,
                            iters=iters,
                            wall_time_s=wall,
                            err_task_norm=et,
                            err_geom_norm=eg,
                        )
                    )
    return records


def aggregate_by_phase(
    records: list[TrialRecord],
    *,
    methods: tuple[str, ...],
) -> dict[str, dict[str, dict[str, float]]]:
    """phase -> method -> metric -> value。"""
    phases = ("grasp", "lift", "place")
    out: dict[str, dict[str, dict[str, float]]] = {}
    for ph in phases:
        out[ph] = {}
        for m in methods:
            sub = [r for r in records if r.phase == ph and r.method == m]
            n = len(sub)
            succ = [r for r in sub if r.converged]
            ns = len(succ)
            fail_rate = 1.0 - (ns / n) if n else 0.0
            if ns:
                mean_iters = float(np.mean([r.iters for r in succ]))
                std_iters = float(np.std([r.iters for r in succ]))
                mean_time_ms = float(np.mean([r.wall_time_s for r in succ])) * 1000.0
                std_time_ms = float(np.std([r.wall_time_s for r in succ])) * 1000.0
                mean_geom = float(np.mean([r.err_geom_norm for r in succ]))
                std_geom = float(np.std([r.err_geom_norm for r in succ]))
            else:
                mean_iters = std_iters = 0.0
                mean_time_ms = std_time_ms = 0.0
                mean_geom = std_geom = 0.0
            out[ph][m] = {
                "n": float(n),
                "n_succ": float(ns),
                "fail_rate": float(fail_rate),
                "mean_iters": mean_iters,
                "std_iters": std_iters,
                "mean_time_ms": mean_time_ms,
                "std_time_ms": std_time_ms,
                "mean_geom_err": mean_geom,
                "std_geom_err": std_geom,
            }
    return out


def write_csv(path: Path, records: list[TrialRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "target_idx",
                "method",
                "trial",
                "converged",
                "iters",
                "wall_time_s",
                "err_task_norm",
                "err_geom_norm",
            ],
        )
        w.writeheader()
        for r in records:
            w.writerow(
                {
                    "phase": r.phase,
                    "target_idx": r.target_idx,
                    "method": r.method,
                    "trial": r.trial,
                    "converged": int(r.converged),
                    "iters": r.iters,
                    "wall_time_s": r.wall_time_s,
                    "err_task_norm": r.err_task_norm,
                    "err_geom_norm": r.err_geom_norm,
                }
            )


def plot_bar_comparison(
    agg: dict[str, dict[str, dict[str, float]]],
    *,
    out_png: Path,
    title_suffix: str,
) -> None:
    import matplotlib.pyplot as plt

    phases = ["grasp", "lift", "place"]
    labels_cn = ["抓取", "抬起", "放置"]
    methods = ("pinv", "dls")
    x = np.arange(len(phases), dtype=np.float64)
    width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)
    fig.suptitle(f"堆积木关键位姿 IK 对比 ({title_suffix})", fontsize=13)

    def _bars(ax, key_mean: str, key_std: str, ylabel: str, title: str, *, scale: float = 1.0) -> None:
        for i, m in enumerate(methods):
            means = [agg[p][m][key_mean] * scale for p in phases]
            stds = [agg[p][m][key_std] * scale for p in phases]
            offset = (-0.5 + i) * width
            ax.bar(x + offset, means, width, yerr=stds, label=("伪逆 pinv" if m == "pinv" else "DLS"), capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_cn)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend()

    ax = axes[0, 0]
    _bars(ax, "mean_iters", "std_iters", "迭代次数", "平均迭代次数（仅成功试次）")

    ax = axes[0, 1]
    _bars(ax, "mean_geom_err", "std_geom_err", r"$\Vert e_{\mathrm{geom}}\Vert$", "平均末端位姿误差（成功试次）")

    ax = axes[1, 0]
    _bars(ax, "mean_time_ms", "std_time_ms", "时间 / ms", "平均计算时间（成功试次）")

    ax = axes[1, 1]
    for i, m in enumerate(methods):
        fr = [agg[p][m]["fail_rate"] * 100.0 for p in phases]
        offset = (-0.5 + i) * width
        ax.bar(x + offset, fr, width, label=("伪逆 pinv" if m == "pinv" else "DLS"))
    ax.set_xticks(x)
    ax.set_xticklabels(labels_cn)
    ax.set_ylabel("失败率 / %")
    ax.set_title("收敛失败率（全部试次）")
    fr_all = [agg[p][m]["fail_rate"] * 100.0 for p in phases for m in methods]
    ymax = float(np.max(fr_all)) if fr_all else 0.0
    ax.set_ylim(0.0, max(5.0, ymax * 1.25 + 1e-6))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="堆积木关键位姿：pinv vs DLS 对比实验")
    ap.add_argument("--seed", type=int, default=20260514, help="位姿与试次随机种子")
    ap.add_argument(
        "--lift-dz",
        type=float,
        default=0.05,
        help="抬起相对抓取的世界系 +z 平移（米）。过大时易超出 FR3 可达空间，实验会大量失败。",
    )
    ap.add_argument("--trials", type=int, default=24, help="每个目标位姿、每种方法的随机初值试次数")
    ap.add_argument(
        "--q0-spread",
        type=float,
        default=0.55,
        help="初值在 --q0-mean 附近 N(0,σ) 的 σ（弧度）；过大会使失败率偏高",
    )
    ap.add_argument(
        "--q0-mean",
        type=float,
        nargs=7,
        metavar=("q1", "q2", "q3", "q4", "q5", "q6", "q7"),
        default=None,
        help="随机初值分布中心（7 个关节角，弧度）。默认与名义抓取参考一致。",
    )
    ap.add_argument("--damping", type=float, default=0.05, help="DLS 阻尼 λ")
    ap.add_argument("--tol", type=float, default=7e-4, help="IK 收敛阈值（任务误差范数）")
    ap.add_argument("--max-iters", type=int, default=500)
    ap.add_argument("--rcond", type=float, default=1e-4)
    ap.add_argument("--step-scale", type=float, default=0.55)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "ik_stack_benchmark",
        help="输出目录（写入 CSV 与 PNG）",
    )
    args = ap.parse_args()

    try:
        import matplotlib as mpl
    except ImportError as e:  # pragma: no cover
        raise SystemExit("需要 matplotlib：请先 pip install -r requirements.txt") from e

    mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False

    poses = build_stack_target_poses(
        args.seed,
        pos_sigma_m=0.012,
        rot_sigma_rad=0.05,
        lift_dz_m=args.lift_dz,
    )

    q0_mean = np.asarray(args.q0_mean, dtype=np.float64).reshape(7) if args.q0_mean is not None else Q_STACK_IK_REF

    methods: tuple[IkMethod, ...] = ("pinv", "dls")
    records = collect_records(
        poses,
        methods=methods,
        trials_per_target=args.trials,
        base_seed=args.seed + 11_000,
        damping=args.damping,
        tol=args.tol,
        max_iters=args.max_iters,
        rcond=args.rcond,
        step_scale=args.step_scale,
        q0_spread=args.q0_spread,
        q0_mean=q0_mean,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "ik_stack_benchmark_trials.csv"
    write_csv(csv_path, records)

    agg = aggregate_by_phase(records, methods=("pinv", "dls"))
    png_path = args.out / "ik_stack_benchmark_summary.png"
    plot_bar_comparison(
        agg,
        out_png=png_path,
        title_suffix=f"seed={args.seed}, trials/目标/方法={args.trials}",
    )

    print(f"已写入: {csv_path}")
    print(f"已写入: {png_path}")


if __name__ == "__main__":
    main()
