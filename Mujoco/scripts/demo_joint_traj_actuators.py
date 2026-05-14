"""M3+M5：用 menagerie 内置位置执行器跟踪关节轨迹（ctrl = q_des）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from fr3_sim.paths import fr3_scene_xml
from fr3_sim.recorder import SimulationRecorder
from fr3_sim.trajectory import quintic_joint_trajectory


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(fr3_scene_xml()))
    data = mujoco.MjData(model)

    q0 = np.array(data.qpos[:7], copy=True)
    qf = q0 + np.array([0.25, -0.15, 0.2, -0.25, 0.1, 0.35, -0.1], dtype=np.float64)
    q_lo = model.jnt_range[:7, 0]
    q_hi = model.jnt_range[:7, 1]
    qf = np.clip(qf, q_lo + 1e-3, q_hi - 1e-3)

    dt = float(model.opt.timestep)
    T = 2.5
    ts, qs, _, _ = quintic_joint_trajectory(q0, qf, T, dt)

    data.qpos[:7] = q0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    rec = SimulationRecorder()
    for i in range(len(ts)):
        data.ctrl[:7] = qs[i]
        mujoco.mj_step(model, data)
        rec.record(float(data.time), data)

    bundle = rec.stack()
    err = float(np.linalg.norm(bundle["qpos"][-1, :7] - qf))
    print("步数:", bundle["time"].shape[0], "末端关节误差范数:", err)


if __name__ == "__main__":
    main()
