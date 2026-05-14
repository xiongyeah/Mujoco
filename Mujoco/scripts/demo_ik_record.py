"""演示：随机关节目标 -> FK 得位姿 -> IK 求解 -> PD 保持 + 记录（M2 + M5）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from fr3_sim.controller import JointPDGravity
from fr3_sim.ik import ik_solve
from fr3_sim.kinematics import fk_fr3_attachment
from fr3_sim.paths import fr3_scene_xml
from fr3_sim.recorder import SimulationRecorder
from fr3_sim.sim_loop import restore_torque_only_mode, step_torque, torque_only_mode


def main() -> None:
    xml = str(fr3_scene_xml())
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)

    rng = np.random.default_rng(3)
    q_lo = model.jnt_range[:7, 0]
    q_hi = model.jnt_range[:7, 1]
    q_tgt = rng.uniform(q_lo, q_hi)
    T_des = fk_fr3_attachment(q_tgt)

    q_init = np.clip(q_tgt + 0.08 * rng.standard_normal(7), q_lo, q_hi)
    q_sol, iters, err = ik_solve(
        q_init,
        T_des,
        max_iters=400,
        tol=1e-5,
        damping=5e-2,
        step_scale=0.55,
        position_only=False,
    )
    print(f"IK iters={iters}, final pose err={err:.3e}")

    gains = torque_only_mode(model, backup=True)
    data.qpos[:7] = q_sol
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    pd = JointPDGravity(kp=np.full(7, 120.0), kd=np.full(7, 35.0))
    q_des = q_sol.copy()
    dq_des = np.zeros(7)

    rec = SimulationRecorder()
    for _ in range(240):
        tau = pd.compute_torque(model, data, q_des=q_des, dq_des=dq_des)
        step_torque(model, data, tau)
        rec.record(float(data.time), data, ctrl=np.array(data.ctrl, copy=True), qfrc_applied=tau)

    bundle = rec.stack()
    print("记录 time:", bundle["time"].shape, "qpos:", bundle["qpos"].shape)

    restore_torque_only_mode(model, gains)


if __name__ == "__main__":
    main()
