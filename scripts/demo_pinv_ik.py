"""最小演示：从零初值用伪逆 IK 收敛到目标末端位姿。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from m2_ik import forward_kinematics, inverse_kinematics, pose_error_se3


def main() -> None:
    # 构造目标：先取参考关节 q_ref 的 FK，再在位置上加一点偏置，得到待求 T_des
    q_ref = np.array([0.0, -0.35, 0.4, -1.35, 0.05, 1.55, -0.25], dtype=np.float64)
    T_des = forward_kinematics(q_ref)
    T_des[:3, 3] += np.array([0.02, -0.015, 0.01], dtype=np.float64)

    q0 = np.zeros(7, dtype=np.float64)  # 迭代初值 q^{(0)}
    # 调用整条伪逆迭代：内部重复「FK→e→J→lstsq→更新 q」直至 ‖e‖<tol
    q_sol, info = inverse_kinematics(T_des, q0, method="pinv")

    # 验算：用解出的 q_sol 再做 FK，与 T_des 比 6 维位姿误差
    T_act = forward_kinematics(q_sol)
    err6 = pose_error_se3(T_act, T_des)

    print("伪逆 IK（pinv）")
    print(f"  迭代次数: {info['iters']}")
    print(f"  终位置误差: {info['pos_err']:.2e}  终旋转误差: {info['rot_err']:.2e}")
    print(f"  位姿误差向量范数: {np.linalg.norm(err6):.6e}")
    print(f"  q_sol: {np.array2string(q_sol, precision=4, suppress_small=True)}")


if __name__ == "__main__":
    main()
