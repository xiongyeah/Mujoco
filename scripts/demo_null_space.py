"""Demo：零空间投影控制肘部高度，末端位姿不变。

对比两种策略：
1. 不加零空间 → 随机初值收敛到的默认姿态
2. 推肘部抬高 → 同一目标位姿但肘部显著更高

用法
----
    python scripts/demo_null_space.py

与 ``INTERFACE_SPEC_v0.md`` 第 2.2 节对齐：
``inverse_kinematics_with_nullspace`` 通过 ``q_preferred`` 指定零空间目标。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m2_ik.kinematics import forward_kinematics
from m2_ik.null_space import inverse_kinematics_with_nullspace as ik_ns


def main():
    # 一个随机的目标位姿（确保在工作空间内）
    T_target = np.array(
        [
            [0.0, -1.0, 0.0, 0.4],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # 随机初值
    rng = np.random.default_rng(42)
    q_init = rng.uniform(-0.5, 0.5, size=7)

    # ── 1. 无零空间 ──
    q_plain, info_plain = ik_ns(T_target, q_init, q_preferred=None)
    T_plain = forward_kinematics(q_plain)
    pos_plain = T_plain[:3, 3]

    print("=== 无零空间（默认 IK）===")
    print(f"  迭代 {info_plain['iters']} 次，pos_err={info_plain['pos_err']:.2e}  rot_err={info_plain['rot_err']:.2e}")
    print(f"  末端位置: ({pos_plain[0]:.3f}, {pos_plain[1]:.3f}, {pos_plain[2]:.3f})")
    print(f"  关节角: {np.round(q_plain, 3)}")

    # ── 2. 推肘部抬高 ──
    # 次级目标：将关节拉向"肘部抬高"的参考位形
    Q_REF = np.array([0.0, -0.5, 0.0, -1.8, 0.0, 2.0, 0.7], dtype=np.float64)

    q_up, info_up = ik_ns(
        T_target, q_init,
        q_preferred=Q_REF,
        nullspace_gain=0.25,
    )
    T_up = forward_kinematics(q_up)
    pos_up = T_up[:3, 3]

    print("\n=== 推肘部抬高（零空间增益 0.25）===")
    print(f"  迭代 {info_up['iters']} 次，pos_err={info_up['pos_err']:.2e}  rot_err={info_up['rot_err']:.2e}")
    print(f"  末端位置: ({pos_up[0]:.3f}, {pos_up[1]:.3f}, {pos_up[2]:.3f})")
    print(f"  关节角: {np.round(q_up, 3)}")

    # ── 对比 ──
    print("\n── 对比 ──")
    print(f"  末端位置差: {np.linalg.norm(pos_up - pos_plain):.2e}")
    print(f"  肘部抬高策略 q_2: {q_plain[1]:+.3f} → {q_up[1]:+.3f}")
    print(f"  肘部抬高策略 q_3: {q_plain[2]:+.3f} → {q_up[2]:+.3f}")
    print(f"  肘部抬高策略 q_4: {q_plain[3]:+.3f} → {q_up[3]:+.3f}")


if __name__ == "__main__":
    main()
