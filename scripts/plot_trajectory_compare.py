"""M3：同一组起止关节角，对比五次多项式 vs 关节空间直线+LSPB 的速度/加速度曲线。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from fr3_sim.trajectory import lspb_joint_trajectory, quintic_joint_trajectory


def main() -> None:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    q0 = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.2, 0.0], dtype=np.float64)
    qf = np.array([0.4, 0.1, -0.35, -1.2, 0.25, 1.8, -0.5], dtype=np.float64)
    dt = 0.002
    T = 2.0

    tq, qq, qdq, qddq = quintic_joint_trajectory(q0, qf, T, dt)
    tl, ql, qdl, qddl = lspb_joint_trajectory(q0, qf, vmax=0.6, amax=1.2, dt=dt, time_stretch=1.0)

    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=False)
    axs[0].plot(tq, np.linalg.norm(qdq, axis=1), label="quintic |qd|")
    axs[0].plot(tl, np.linalg.norm(qdl, axis=1), label="LSPB line |qd|")
    axs[0].set_ylabel("|qd| (rad/s)")
    axs[0].legend()
    axs[0].set_title("Joint speed norm: quintic vs LSPB (line in q-space)")

    axs[1].plot(tq, np.linalg.norm(qddq, axis=1), label="quintic |qdd|")
    axs[1].plot(tl, np.linalg.norm(qddl, axis=1), label="LSPB line |qdd|")
    axs[1].set_ylabel("|qdd| (rad/s^2)")
    axs[1].legend()
    axs[1].set_title("Joint accel norm")

    axs[2].plot(tq, qq[:, 3], label="quintic q4")
    axs[2].plot(tl, ql[:, 3], label="LSPB q4")
    axs[2].set_xlabel("t (s)")
    axs[2].set_ylabel("q4 (rad)")
    axs[2].legend()
    axs[2].set_title("Joint 4 (example)")

    fig.tight_layout()
    png = out_dir / "m3_joint_compare.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print("已保存:", png)


if __name__ == "__main__":
    main()
