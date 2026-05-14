"""
M6 · 桌面 pick-and-place 演示：接近方块 → 下降 → mocap「吸附」跟随末端 → 搬运到放置区 → 释放 → 回 home。

默认位置伺服；可选 --torque 使用计算力矩 PD（与 demo_m6_reaching 一致）。
加 --viewer 会打开 MuJoCo 窗口并实时显示仿真（被动 viewer，脚本驱动物理）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco
import mujoco.viewer
import numpy as np

from fr3_sim.controller import ComputedTorquePD
from fr3_sim.m6_pickplace import (
    DEFAULT_PICK_BLOCK_CENTER,
    DEFAULT_PLACE_CENTER,
    DESK_HOME_Q,
    build_cartesian_segment_plan,
    build_joint_space_plan,
    pick_block_mocapid,
    blend_mocap_block_to_pad,
    pick_block_offset_local,
    reset_desk_home,
    sync_pick_block_mocap,
)
from fr3_sim.paths import fr3_desk_pick_xml
from fr3_sim.recorder import SimulationRecorder
from fr3_sim.sim_loop import restore_torque_only_mode, step_torque, torque_only_mode


def _sync_v(v: Any | None) -> None:
    if v is not None:
        v.sync()


def _run_plan_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    plan,
    rec: SimulationRecorder,
    *,
    mocap_id: int | None,
    mocap_follow: bool,
    mocap_off: np.ndarray | None,
    viewer: Any | None = None,
) -> None:
    for i in range(plan.qs.shape[0]):
        data.ctrl[:7] = plan.qs[i]
        mujoco.mj_step(model, data)
        if mocap_follow and mocap_id is not None and mocap_off is not None:
            sync_pick_block_mocap(model, data, mocap_id, mocap_off)
        rec.record(float(data.time), data, ctrl=data.ctrl.copy())
        _sync_v(viewer)


def _run_plan_torque(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    plan,
    rec: SimulationRecorder,
    pd: ComputedTorquePD,
    *,
    mocap_id: int | None,
    mocap_follow: bool,
    mocap_off: np.ndarray | None,
    viewer: Any | None = None,
) -> None:
    for i in range(plan.qs.shape[0]):
        tau = pd.compute_torque(model, data, q_des=plan.qs[i], dq_des=plan.qds[i])
        step_torque(model, data, tau)
        if mocap_follow and mocap_id is not None and mocap_off is not None:
            sync_pick_block_mocap(model, data, mocap_id, mocap_off)
        rec.record(float(data.time), data, ctrl=data.ctrl.copy(), qfrc_applied=tau)
        _sync_v(viewer)


def _dwell_steps(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_hold: np.ndarray,
    n: int,
    rec: SimulationRecorder,
    *,
    torque: bool,
    pd: ComputedTorquePD | None,
    mocap_id: int | None,
    mocap_follow: bool,
    mocap_off: np.ndarray | None,
    viewer: Any | None = None,
) -> None:
    q_hold = np.asarray(q_hold, dtype=np.float64).reshape(7)
    zdq = np.zeros(7, dtype=np.float64)
    for _ in range(n):
        if torque and pd is not None:
            tau = pd.compute_torque(model, data, q_des=q_hold, dq_des=zdq)
            step_torque(model, data, tau)
            rec.record(float(data.time), data, ctrl=data.ctrl.copy(), qfrc_applied=tau)
        else:
            data.ctrl[:7] = q_hold
            mujoco.mj_step(model, data)
            rec.record(float(data.time), data, ctrl=data.ctrl.copy())
        if mocap_follow and mocap_id is not None and mocap_off is not None:
            sync_pick_block_mocap(model, data, mocap_id, mocap_off)
        _sync_v(viewer)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--torque", action="store_true", help="计算力矩 PD + 关闭内置执行器")
    ap.add_argument("--viewer", action="store_true", help="打开 MuJoCo 窗口实时显示（被动 viewer）")
    ap.add_argument("--out", type=str, default="", help="可选 npz 路径")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(fr3_desk_pick_xml()))
    data = mujoco.MjData(model)
    mid = pick_block_mocapid(model)
    dt = float(model.opt.timestep)

    reset_desk_home(model, data)
    mujoco.mj_forward(model, data)
    q = np.array(data.qpos[:7], dtype=np.float64)

    pb = DEFAULT_PICK_BLOCK_CENTER
    # 接近段仍略抬高，避免蹭桌面；抓取段要让 attachment_site 靠近方块顶面（pb_z + half_geom），
    # 否则 IK 末端停在方块上方数厘米，mocap 虽锁相对位姿也会看起来像「隔空抓」。
    p_approach = pb + np.array([0.0, 0.0, 0.18], dtype=np.float64)
    p_grasp_hi = pb + np.array([0.0, 0.0, 0.055], dtype=np.float64)
    p_grasp_touch = pb + np.array([0.0, 0.0, 0.019], dtype=np.float64)
    p_lift = pb + np.array([0.0, 0.0, 0.18], dtype=np.float64)

    pc = DEFAULT_PLACE_CENTER
    p_place_hover = pc + np.array([0.0, 0.0, 0.16], dtype=np.float64)
    p_place = pc + np.array([0.0, 0.0, 0.03], dtype=np.float64)

    rec = SimulationRecorder()
    bak = torque_only_mode(model, backup=True) if args.torque else None
    pd = ComputedTorquePD(kp=np.full(7, 95.0), kd=np.full(7, 28.0)) if args.torque else None

    viewer_handle: Any | None = None
    if args.viewer:
        mujoco.mj_forward(model, data)
        viewer_handle = mujoco.viewer.launch_passive(model, data)

    mocap_off: np.ndarray | None = None

    def run_cart(plan, *, follow: bool) -> None:
        nonlocal q
        off = mocap_off if follow else None
        if args.torque and pd is not None:
            _run_plan_torque(
                model,
                data,
                plan,
                rec,
                pd,
                mocap_id=mid,
                mocap_follow=follow,
                mocap_off=off,
                viewer=viewer_handle,
            )
        else:
            _run_plan_position(
                model,
                data,
                plan,
                rec,
                mocap_id=mid,
                mocap_follow=follow,
                mocap_off=off,
                viewer=viewer_handle,
            )
        q = np.array(plan.q1, dtype=np.float64)

    try:
        plan_approach = build_cartesian_segment_plan(model, q, p_approach, 2.8)
        run_cart(plan_approach, follow=False)

        plan_grasp_hi = build_cartesian_segment_plan(model, q, p_grasp_hi, 1.35)
        run_cart(plan_grasp_hi, follow=False)

        plan_grasp_touch = build_cartesian_segment_plan(model, q, p_grasp_touch, 1.15)
        run_cart(plan_grasp_touch, follow=False)

        # 抓取瞬间记录 site 局部偏移，避免写死 [0,0,-0.1]（在 FR3 姿态下会映射到世界 +Z，方块会「飞走」）
        mocap_off = pick_block_offset_local(model, data, mid)

        # 短暂停留并开始 mocap 跟随（无物理夹爪）
        _dwell_steps(
            model,
            data,
            plan_grasp_touch.q1,
            max(1, int(round(0.25 / dt))),
            rec,
            torque=args.torque,
            pd=pd,
            mocap_id=mid,
            mocap_follow=True,
            mocap_off=mocap_off,
            viewer=viewer_handle,
        )
        q = np.array(data.qpos[:7], dtype=np.float64)

        plan_lift = build_cartesian_segment_plan(model, q, p_lift, 1.85)
        run_cart(plan_lift, follow=True)

        plan_hover = build_cartesian_segment_plan(model, q, p_place_hover, 3.0)
        run_cart(plan_hover, follow=True)

        plan_lower = build_cartesian_segment_plan(model, q, p_place, 2.0)
        run_cart(plan_lower, follow=True)

        # 释放：更长 quintic 混合，避免末尾再 set 一次造成微抖
        blend_mocap_block_to_pad(model, data, mid, duration=0.28, viewer=viewer_handle)
        _dwell_steps(
            model,
            data,
            plan_lower.q1,
            max(1, int(round(0.2 / dt))),
            rec,
            torque=args.torque,
            pd=pd,
            mocap_id=None,
            mocap_follow=False,
            mocap_off=None,
            viewer=viewer_handle,
        )
        q = np.array(data.qpos[:7], dtype=np.float64)

        # 先抬高并偏向基座侧，再回 home，减少大臂穿过垫上红块的视觉穿模
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        mujoco.mj_forward(model, data)
        p_clear = np.array(data.site_xpos[sid], dtype=np.float64).copy()
        p_clear[0] -= 0.14
        p_clear[1] += 0.12
        p_clear[2] = max(float(p_clear[2]) + 0.26, 0.64)
        plan_clear = build_cartesian_segment_plan(model, q, p_clear, 2.15)
        run_cart(plan_clear, follow=False)
        q = np.array(data.qpos[:7], dtype=np.float64)

        home_plan = build_joint_space_plan(q, DESK_HOME_Q, 2.55, dt)
        if args.torque and pd is not None:
            _run_plan_torque(
                model,
                data,
                home_plan,
                rec,
                pd,
                mocap_id=mid,
                mocap_follow=False,
                mocap_off=None,
                viewer=viewer_handle,
            )
        else:
            _run_plan_position(
                model,
                data,
                home_plan,
                rec,
                mocap_id=mid,
                mocap_follow=False,
                mocap_off=None,
                viewer=viewer_handle,
            )
    finally:
        if bak is not None:
            restore_torque_only_mode(model, bak)
        if viewer_handle is not None:
            viewer_handle.close()

    bundle = rec.stack()
    # 方块中心相对放置区中心的水平距离（释放后 mocap 仍表示方块）
    block_xy = np.array(data.mocap_pos[mid, :2], dtype=np.float64)
    place_xy = pc[:2]
    dist_xy = float(np.linalg.norm(block_xy - place_xy))

    print(f"仿真时长 {bundle['time'][-1]:.3f} s  步数 {len(bundle['time'])}")
    print(f"释放后方块与放置区中心水平偏差: {dist_xy:.4f} m")

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
            mocap_pos_final=np.array(data.mocap_pos[mid].copy()),
            place_center=pc,
            mode=np.array(["torque" if args.torque else "position"]),
        )
        print("已保存:", outp.resolve())


if __name__ == "__main__":
    main()
