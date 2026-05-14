"""M4：三种力矩控制在 torque_only_mode 下对比（关节 PD、计算力矩、笛卡尔阻抗）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from fr3_sim.controller import CartesianImpedanceSite, ComputedTorquePD, JointPDGravity
from fr3_sim.kinematics import fk_fr3_attachment
from fr3_sim.paths import fr3_scene_xml
from fr3_sim.sim_loop import restore_torque_only_mode, step_torque, torque_only_mode


def _site_pos_err(model: mujoco.MjModel, data: mujoco.MjData, T_des: np.ndarray) -> float:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(data.site_xpos[sid] - T_des[:3, 3]))


def _run(
    name: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl,
    q_home: np.ndarray,
    T_des: np.ndarray,
    *,
    steps: int = 2200,
) -> None:
    mujoco.mj_resetData(model, data)
    rng = np.random.default_rng(42)
    data.qpos[:7] = q_home + 0.08 * rng.standard_normal(7)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    errs: list[float] = []
    qerrs: list[float] = []
    for _ in range(steps):
        tau = ctrl.compute_torque(model, data, q_des=q_home, dq_des=np.zeros(7))
        step_torque(model, data, tau)
        errs.append(_site_pos_err(model, data, T_des))
        qerrs.append(float(np.linalg.norm(data.qpos[:7] - q_home)))

    tail_p = np.mean(errs[-200:])
    tail_q = np.mean(qerrs[-200:])
    print(f"{name:22s} |ee|_末段均值={tail_p:.4f} m   |q-q*|_末段均值={tail_q:.4f} rad")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(fr3_scene_xml()))
    data = mujoco.MjData(model)

    q_home = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=np.float64)
    T_des = fk_fr3_attachment(q_home)

    gains = torque_only_mode(model, backup=True)

    pd = JointPDGravity(kp=np.full(7, 260.0), kd=np.full(7, 55.0))
    ct = ComputedTorquePD(kp=np.full(7, 110.0), kd=np.full(7, 32.0))
    cart = CartesianImpedanceSite(
        "attachment_site",
        T_des,
        kp=np.array([720.0, 720.0, 720.0, 48.0, 48.0, 48.0], dtype=np.float64),
        kd=np.array([72.0, 72.0, 72.0, 12.0, 12.0, 12.0], dtype=np.float64),
        kd_joint=8.0,
    )

    _run("Joint PD + bias", model, data, pd, q_home, T_des)
    _run("Computed torque PD", model, data, ct, q_home, T_des)
    _run("Cartesian impedance", model, data, cart, q_home, T_des)

    restore_torque_only_mode(model, gains)


if __name__ == "__main__":
    main()
