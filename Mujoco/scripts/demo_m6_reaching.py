"""
M6 · Reaching 演示：home → IK 目标位姿 → 五次关节轨迹 → 位置伺服跟踪，并记录末端位置误差。

默认使用 menagerie scene.xml（7 关节 + 位置执行器）。
可选 --torque：关闭执行器 + 计算力矩 PD 跟踪（与 M4 衔接）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from fr3_sim.controller import ComputedTorquePD
from fr3_sim.m6_reaching import build_reach_plan, reset_arm_home, site_position_error
from fr3_sim.paths import fr3_scene_xml
from fr3_sim.recorder import SimulationRecorder
from fr3_sim.sim_loop import restore_torque_only_mode, step_torque, torque_only_mode


def _run_position(model: mujoco.MjModel, data: mujoco.MjData, plan, rec: SimulationRecorder) -> np.ndarray:
    errs: list[float] = []
    for i in range(plan.qs.shape[0]):
        data.ctrl[:7] = plan.qs[i]
        mujoco.mj_step(model, data)
        errs.append(site_position_error(model, data, plan.T_des[:3, 3]))
        rec.record(float(data.time), data, ctrl=data.ctrl.copy())
    return np.asarray(errs, dtype=np.float64)


def _run_torque(model: mujoco.MjModel, data: mujoco.MjData, plan, rec: SimulationRecorder) -> np.ndarray:
    bak = torque_only_mode(model, backup=True)
    pd = ComputedTorquePD(kp=np.full(7, 95.0), kd=np.full(7, 28.0))
    errs: list[float] = []
    try:
        for i in range(plan.qs.shape[0]):
            tau = pd.compute_torque(model, data, q_des=plan.qs[i], dq_des=plan.qds[i])
            step_torque(model, data, tau)
            errs.append(site_position_error(model, data, plan.T_des[:3, 3]))
            rec.record(float(data.time), data, ctrl=data.ctrl.copy(), qfrc_applied=tau)
    finally:
        restore_torque_only_mode(model, bak)
    return np.asarray(errs, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=2.8, help="五次轨迹总时长 (s)")
    ap.add_argument(
        "--px",
        type=float,
        default=0.44,
        help="目标末端位置 x（世界系，约在工作空间前方）",
    )
    ap.add_argument("--py", type=float, default=0.06)
    ap.add_argument("--pz", type=float, default=0.48)
    ap.add_argument("--torque", action="store_true", help="用计算力矩 PD + qfrc_applied（需 mjDSBL_ACTUATION）")
    ap.add_argument("--out", type=str, default="", help="可选：保存 npz 路径，如 outputs/m6_reaching.npz")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(fr3_scene_xml()))
    data = mujoco.MjData(model)
    p_des = np.array([args.px, args.py, args.pz], dtype=np.float64)

    plan = build_reach_plan(model, data, p_des, duration=args.duration)
    reset_arm_home(model, data)
    mujoco.mj_forward(model, data)

    rec = SimulationRecorder()
    if args.torque:
        errs = _run_torque(model, data, plan, rec)
    else:
        errs = _run_position(model, data, plan, rec)

    bundle = rec.stack()
    print(f"轨迹步数 {plan.qs.shape[0]}  仿真时长 {bundle['time'][-1]:.3f} s")
    print(f"末端位置误差: max={errs.max():.5f} m  mean(末20%)={errs[int(0.8*len(errs)):].mean():.5f} m  final={errs[-1]:.5f} m")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            outp,
            time=bundle["time"],
            qpos=bundle["qpos"],
            qvel=bundle["qvel"],
            ctrl=bundle["ctrl"],
            qfrc_applied=bundle["qfrc_applied"],
            ee_pos_err=errs,
            q0=plan.q0,
            q1=plan.q1,
            p_des=p_des,
            T_des=plan.T_des,
            mode=np.array(["torque" if args.torque else "position"]),
        )
        print("已保存:", outp.resolve())


if __name__ == "__main__":
    main()
