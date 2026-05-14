"""
M2 逆运动学实验输出（对照 FR3_sim_plan.pdf · M2 基本目标）：

- 三个测试位姿下，DLS 与伪逆迭代两种数值解的迭代次数、终误差、耗时；
- 奇异附近（低可操作度初值）稳定性对比；
- 冗余：解处雅可比最小奇异方向 v 的 ||J v||（一阶末端不变量）。

生成 CSV + 中文摘要文本，便于直接贴进报告。

用法（在项目根目录）:
  python scripts/m2_ik_experiment_report.py
  python scripts/m2_ik_experiment_report.py --out outputs/m2_ik
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from m2_inverse_kinematics import (
    fk_fr3_attachment,
    geometric_jacobian_world,
    manipulability_translation,
    null_space_vector_from_jacobian,
    pose_error_se3,
    ik_solve,
)


def _T_des_from_q(q: np.ndarray, pos_delta: np.ndarray | None = None) -> np.ndarray:
    T = fk_fr3_attachment(np.asarray(q, dtype=np.float64).reshape(7)).copy()
    if pos_delta is not None:
        T[:3, 3] += np.asarray(pos_delta, dtype=np.float64).reshape(3)
    return T


def _run_case(
    *,
    case_id: str,
    pose_name: str,
    q_init: np.ndarray,
    T_des: np.ndarray,
    method: str,
    tol: float,
    max_iters: int,
) -> dict[str, object]:
    q0 = np.asarray(q_init, dtype=np.float64).reshape(7)
    J0 = geometric_jacobian_world(q0)
    w0 = manipulability_translation(J0)

    t0 = time.perf_counter()
    q_sol, iters, err_f = ik_solve(
        q0,
        T_des,
        max_iters=max_iters,
        tol=tol,
        damping=0.05,
        step_scale=0.55,
        method=method,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    Js = geometric_jacobian_world(q_sol)
    w_sol = manipulability_translation(Js)
    T_act = fk_fr3_attachment(q_sol)
    err_vec = pose_error_se3(T_act, T_des)
    err_check = float(np.linalg.norm(err_vec))

    return {
        "case_id": case_id,
        "pose_name": pose_name,
        "method": method,
        "converged": int(err_f < tol),
        "iters": iters,
        "final_err": err_f,
        "err_pose_check": err_check,
        "time_ms": elapsed_ms,
        "w_init": w0,
        "w_sol": w_sol,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="M2 IK 对比实验报告（CSV + 摘要）")
    ap.add_argument("--out", type=str, default="", help="输出目录（默认: 项目根下 outputs/m2_ik）")
    ap.add_argument("--max-iters", type=int, default=500, help="单次 IK 最大迭代次数")
    ap.add_argument("--tol", type=float, default=7e-4, help="收敛阈值（与 m6_reaching 默认一致）")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else ROOT / "outputs" / "m2_ik"
    out_dir.mkdir(parents=True, exist_ok=True)

    tol = float(args.tol)
    max_iters = int(args.max_iters)

    q_init_default = np.zeros(7, dtype=np.float64)

    # 三个测试位姿：T_des = FK(q_ref) + 小位置偏置；从零位起步求解（与 PDF「三姿对比」一致）
    poses: list[tuple[str, str, np.ndarray, np.ndarray]] = [
        (
            "P1",
            "形位一(中小关节幅值)",
            q_init_default,
            _T_des_from_q(np.array([0.0, -0.25, 0.32, -1.12, 0.05, 1.22, -0.22], dtype=np.float64), np.array([0.02, -0.02, 0.015])),
        ),
        (
            "P2",
            "形位二(肘部伸展)",
            q_init_default,
            _T_des_from_q(np.array([0.0, -0.45, 0.55, -1.75, 0.05, 1.95, -0.55], dtype=np.float64)),
        ),
        (
            "P3",
            "形位三(非对称)",
            q_init_default,
            _T_des_from_q(np.array([0.15, 0.25, -0.35, -1.25, 0.35, 1.45, 0.2], dtype=np.float64), np.array([-0.03, 0.04, -0.02])),
        ),
    ]

    rows: list[dict[str, object]] = []
    for pid, pname, q0, T_des in poses:
        for method in ("dls", "pinv"):
            rows.append(
                _run_case(
                    case_id=pid,
                    pose_name=pname,
                    q_init=q0,
                    T_des=T_des,
                    method=method,
                    tol=tol,
                    max_iters=max_iters,
                )
            )

    # 奇异附近：初值可操作度较小，同一目标下对比两种方法终误差与迭代次数
    q_near = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.0], dtype=np.float64)
    T_sing = _T_des_from_q(q_near, np.array([0.06, 0.0, 0.04]))
    for method in ("dls", "pinv"):
        rows.append(
            _run_case(
                case_id="S1",
                pose_name="低可操作度初值+小偏置目标",
                q_init=q_near,
                T_des=T_sing,
                method=method,
                tol=tol,
                max_iters=max_iters,
            )
        )

    csv_path = out_dir / "m2_ik_comparison.csv"
    fieldnames = [
        "case_id",
        "pose_name",
        "method",
        "converged",
        "iters",
        "final_err",
        "err_pose_check",
        "time_ms",
        "w_init",
        "w_sol",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})

    # 冗余：取 P1 + DLS 的解做零空间一阶检验
    T_p1 = poses[0][3]
    q_p1, _, _ = ik_solve(
        q_init_default,
        T_p1,
        max_iters=max_iters,
        tol=tol,
        damping=0.05,
        step_scale=0.55,
        method="dls",
    )
    J1 = geometric_jacobian_world(q_p1)
    v = null_space_vector_from_jacobian(J1)
    jv_norm = float(np.linalg.norm(J1 @ v))

    lines_zh: list[str] = []
    lines_zh.append("M2 逆运动学 · 实验输出摘要（自动生成）\n")
    lines_zh.append("对照课程文档：三测试位姿 ×（DLS / 伪逆迭代）；奇异初值；冗余度一阶检验。\n")
    lines_zh.append(f"收敛阈值 tol = {tol:g}，最大迭代 max_iters = {max_iters}\n")
    lines_zh.append("\n--- 主对比表（CSV 同内容）---\n")
    lines_zh.append(
        f"{'位姿':<6} {'方法':<6} {'收敛':>4} {'迭代':>6} {'终误差':>12} {'耗时ms':>10} {'w_init':>12} {'w_sol':>12}\n"
    )
    for r in rows:
        lines_zh.append(
            f"{r['case_id']!s:<6} {r['method']!s:<6} {int(r['converged']):>4} {int(r['iters']):>6} "
            f"{float(r['final_err']):>12.6e} {float(r['time_ms']):>10.3f} "
            f"{float(r['w_init']):>12.6e} {float(r['w_sol']):>12.6e}\n"
        )
    lines_zh.append("\n--- 冗余（末端一阶不变）---\n")
    lines_zh.append(f"P1 收敛解处：||J v|| = {jv_norm:.6e}（v 为最小奇异值对应单位右奇异向量，理想接近 0）\n")
    lines_zh.append(f"可操作度 w（平移子块 sqrt(det(J_t J_t^T))) 在解处 = {manipulability_translation(J1):.6e}\n")
    lines_zh.append(f"\n详细 CSV: {csv_path.resolve()}\n")

    summary_path = out_dir / "m2_ik_summary_zh.txt"
    summary_path.write_text("".join(lines_zh), encoding="utf-8")

    print("已写入:")
    print(" ", csv_path.resolve())
    print(" ", summary_path.resolve())
    print("".join(lines_zh))


if __name__ == "__main__":
    main()
