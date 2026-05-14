"""
M6 / M4：同一 reaching 规划（五次关节轨迹 + 固定末端目标位姿），
对比四条曲线（同一 quintic 参考 + 同一末端目标 T_des）：
  1) position：位置执行器 ctrl 跟踪 q_ref（工程基线）
  2) joint_pd：关节 PD + bias + qfrc_applied
  3) computed_torque：计算力矩 PD 跟踪 q_ref,qdot_ref
  4) cartesian：笛卡尔阻抗拉向 T_des（不跟踪关节参考，路径不同）

输出 outputs/m6_reach_three.npz 与 outputs/m6_reach_three.png
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from fr3_sim.controller import CartesianImpedanceSite, ComputedTorquePD, JointPDGravity
from fr3_sim.m6_reaching import build_reach_plan, reset_arm_home, site_position_error
from fr3_sim.paths import fr3_scene_xml
from fr3_sim.sim_loop import restore_torque_only_mode, step_torque, torque_only_mode


def _joint_ref_rmse(q_meas: np.ndarray, q_ref: np.ndarray) -> float:
    return float(np.linalg.norm(q_meas - q_ref))


def _power_step(tau: np.ndarray, dq: np.ndarray, dt: float) -> float:
    return float(np.dot(tau, dq) * dt)


def run_position(model: mujoco.MjModel, data: mujoco.MjData, plan) -> dict:
    reset_arm_home(model, data)
    ee: list[float] = []
    jr: list[float] = []
    pwr: list[float] = []
    for i in range(plan.qs.shape[0]):
        data.ctrl[:7] = plan.qs[i]
        mujoco.mj_forward(model, data)
        dq = np.array(data.qvel[:7], copy=True)
        mujoco.mj_step(model, data)
        ee.append(site_position_error(model, data, plan.T_des[:3, 3]))
        jr.append(_joint_ref_rmse(data.qpos[:7], plan.qs[i]))
        tau_act = data.actuator_force[:7] if model.nu == 7 else np.zeros(7)
        pwr.append(_power_step(tau_act, dq, float(model.opt.timestep)))
    return {"ee": np.array(ee), "joint_rmse": np.array(jr), "power": np.array(pwr)}


def run_torque_controller(model, data, plan, ctrl, *, name: str) -> dict:
    reset_arm_home(model, data)
    bak = torque_only_mode(model, backup=True)
    ee: list[float] = []
    jr: list[float] = []
    pwr: list[float] = []
    try:
        for i in range(plan.qs.shape[0]):
            mujoco.mj_forward(model, data)
            dq = np.array(data.qvel[:7], copy=True)
            tau = ctrl.compute_torque(model, data, plan.qs[i], plan.qds[i])
            step_torque(model, data, tau)
            ee.append(site_position_error(model, data, plan.T_des[:3, 3]))
            jr.append(_joint_ref_rmse(data.qpos[:7], plan.qs[i]))
            pwr.append(_power_step(tau, dq, float(model.opt.timestep)))
    finally:
        restore_torque_only_mode(model, bak)
    return {"ee": np.array(ee), "joint_rmse": np.array(jr), "power": np.array(pwr), "name": name}


def run_cartesian(model: mujoco.MjModel, data: mujoco.MjData, plan) -> dict:
    reset_arm_home(model, data)
    cart = CartesianImpedanceSite(
        "attachment_site",
        plan.T_des,
        kp=np.array([520.0, 520.0, 520.0, 42.0, 42.0, 42.0], dtype=np.float64),
        kd=np.array([58.0, 58.0, 58.0, 10.0, 10.0, 10.0], dtype=np.float64),
        kd_joint=7.0,
    )
    return run_torque_controller(model, data, plan, cart, name="cartesian")


def main() -> None:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(fr3_scene_xml()))
    data = mujoco.MjData(model)
    p_des = np.array([0.44, 0.06, 0.48], dtype=np.float64)
    plan = build_reach_plan(model, data, p_des, duration=2.8)

    joint_pd = JointPDGravity(kp=np.full(7, 200.0), kd=np.full(7, 48.0))
    ct = ComputedTorquePD(kp=np.full(7, 90.0), kd=np.full(7, 26.0))

    res_pos = run_position(model, data, plan)
    res_jpd = run_torque_controller(model, data, plan, joint_pd, name="joint_pd")
    res_ct = run_torque_controller(model, data, plan, ct, name="computed_torque")
    res_cart = run_cartesian(model, data, plan)

    np.savez(
        out_dir / "m6_reach_three.npz",
        time=plan.ts,
        q_ref=plan.qs,
        qd_ref=plan.qds,
        T_des=plan.T_des,
        ee_pos=res_pos["ee"],
        ee_jpd=res_jpd["ee"],
        ee_ct=res_ct["ee"],
        ee_cart=res_cart["ee"],
        jrmse_pos=res_pos["joint_rmse"],
        jrmse_jpd=res_jpd["joint_rmse"],
        jrmse_ct=res_ct["joint_rmse"],
        jrmse_cart=res_cart["joint_rmse"],
        pwr_pos=res_pos["power"],
        pwr_jpd=res_jpd["power"],
        pwr_ct=res_ct["power"],
        pwr_cart=res_cart["power"],
    )

    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axs[0].plot(plan.ts, res_pos["ee"], label="position ctrl")
    axs[0].plot(plan.ts, res_jpd["ee"], label="joint PD+bias")
    axs[0].plot(plan.ts, res_ct["ee"], label="computed torque")
    axs[0].plot(plan.ts, res_cart["ee"], label="cartesian imp.")
    axs[0].set_ylabel("|ee pos err| (m)")
    axs[0].legend(loc="upper right", fontsize=8)
    axs[0].set_title("M6 reaching: end-effector position error vs time")

    axs[1].plot(plan.ts, res_pos["joint_rmse"], label="position ctrl")
    axs[1].plot(plan.ts, res_jpd["joint_rmse"], label="joint PD+bias")
    axs[1].plot(plan.ts, res_ct["joint_rmse"], label="computed torque")
    axs[1].plot(plan.ts, res_cart["joint_rmse"], label="cartesian imp.")
    axs[1].set_ylabel("||q-q_ref|| (rad)")
    axs[1].set_xlabel("t (s)")
    axs[1].legend(loc="upper right", fontsize=8)
    axs[1].set_title("Joint tracking vs nominal quintic")
    fig.tight_layout()
    fig.savefig(out_dir / "m6_reach_three.png", dpi=150)
    plt.close(fig)

    def _summ(tag: str, ee: np.ndarray, jr: np.ndarray, pw: np.ndarray) -> None:
        tail = slice(int(0.8 * len(ee)), None)
        print(
            f"{tag:16s}  ee_max={ee.max():.4f}  ee_tail_mean={ee[tail].mean():.4f}  "
            f"jrmse_tail_mean={jr[tail].mean():.4f}  power_sum_tau_dqdt={pw.sum():.3f}"
        )

    print("--- metrics (same duration, same T_des target) ---")
    _summ("position", res_pos["ee"], res_pos["joint_rmse"], res_pos["power"])
    _summ("joint_pd", res_jpd["ee"], res_jpd["joint_rmse"], res_jpd["power"])
    _summ("comp_torque", res_ct["ee"], res_ct["joint_rmse"], res_ct["power"])
    _summ("cartesian", res_cart["ee"], res_cart["joint_rmse"], res_cart["power"])
    print("saved:", (out_dir / "m6_reach_three.npz").resolve())
    print("saved:", (out_dir / "m6_reach_three.png").resolve())


if __name__ == "__main__":
    main()
