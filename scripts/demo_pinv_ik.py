"""最小演示：从零初值用伪逆 IK 收敛到目标末端位姿。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from m2_ik import forward_kinematics, inverse_kinematics, pose_error_se3


def main() -> None:
    q_ref = np.array([0.0, -0.35, 0.4, -1.35, 0.05, 1.55, -0.25], dtype=np.float64)
    T_des = forward_kinematics(q_ref)
    T_des[:3, 3] += np.array([0.02, -0.015, 0.01], dtype=np.float64)

    q0 = np.zeros(7, dtype=np.float64)
    q_sol, iters, err = inverse_kinematics(T_des, q0, method="pinv")

    T_act = forward_kinematics(q_sol)
    err6 = pose_error_se3(T_act, T_des)

    print("伪逆 IK（pinv）")
    print(f"  迭代次数: {iters}")
    print(f"  终范数误差 ||e||: {err:.6e}")
    print(f"  位姿误差向量范数: {np.linalg.norm(err6):.6e}")
    print(f"  q_sol: {np.array2string(q_sol, precision=4, suppress_small=True)}")


if __name__ == "__main__":
    main()
