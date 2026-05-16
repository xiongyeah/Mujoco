"""对比演示：同一目标下伪逆 (pinv) 与 DLS 迭代 IK。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from m2_ik import forward_kinematics, inverse_kinematics, pose_error_se3


def run_case(
    *,
    method: str,
    damping: float,
    T_des: np.ndarray,
    q0: np.ndarray,
) -> None:
    q_sol, info = inverse_kinematics(
        T_des,
        q0,
        method=cast(Literal["pinv", "dls"], method),
        damping=damping,
    )
    T_act = forward_kinematics(q_sol)
    err6 = pose_error_se3(T_act, T_des)
    print(f"[{method}] 迭代={info['iters']}  终pos_err={info['pos_err']:.2e}  rot_err={info['rot_err']:.2e}  耗时={info['time']:.4f}s")
    print(f"       验算||e6||={np.linalg.norm(err6):.6e}")
    print(f"       q_sol={np.array2string(q_sol, precision=4, suppress_small=True)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="伪逆 vs DLS 逆运动学对比")
    ap.add_argument("--method", choices=["pinv", "dls", "both"], default="both")
    ap.add_argument("--damping", type=float, default=0.05, help="DLS 阻尼 λ")
    args = ap.parse_args()

    q_ref = np.array([0.0, -0.35, 0.4, -1.35, 0.05, 1.55, -0.25], dtype=np.float64)
    T_des = forward_kinematics(q_ref)
    T_des[:3, 3] += np.array([0.02, -0.015, 0.01], dtype=np.float64)
    q0 = np.zeros(7, dtype=np.float64)

    if args.method in ("pinv", "both"):
        run_case(method="pinv", damping=args.damping, T_des=T_des, q0=q0.copy())
    if args.method in ("dls", "both"):
        run_case(method="dls", damping=args.damping, T_des=T_des, q0=q0.copy())


if __name__ == "__main__":
    main()
