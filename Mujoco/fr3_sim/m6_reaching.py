"""M6 · Reaching：末端位置目标 + 关节空间五次轨迹 + 仿真记录。"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from fr3_sim.ik import ik_solve
from fr3_sim.kinematics import fk_fr3_attachment
from fr3_sim.trajectory import quintic_joint_trajectory


def reset_arm_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """使用 menagerie 内置 keyframe `home`（scene.xml 合并自 fr3.xml）。"""
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid < 0:
        raise RuntimeError("未找到 keyframe `home`")
    mujoco.mj_resetDataKeyframe(model, data, kid)


def make_reach_pose(
    q_home: np.ndarray,
    p_des_world: np.ndarray,
) -> np.ndarray:
    """
    构造末端目标齐次矩阵：位置为 p_des_world，姿态与 q_home 下末端姿态一致。
    """
    T = fk_fr3_attachment(np.asarray(q_home, dtype=np.float64).reshape(7))
    T = T.copy()
    T[:3, 3] = np.asarray(p_des_world, dtype=np.float64).reshape(3)
    return T


def solve_reach_ik(
    model: mujoco.MjModel,
    q_init: np.ndarray,
    T_des: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """关节限幅内 IK。"""
    q_sol, iters, err = ik_solve(
        np.asarray(q_init, dtype=np.float64).reshape(7),
        T_des,
        max_iters=500,
        tol=7e-4,
        damping=0.05,
        step_scale=0.55,
        position_only=False,
    )
    lo = model.jnt_range[:7, 0]
    hi = model.jnt_range[:7, 1]
    q_sol = np.clip(q_sol, lo + 1e-4, hi - 1e-4)
    return q_sol, iters, err


def site_position_error(model: mujoco.MjModel, data: mujoco.MjData, p_des: np.ndarray) -> float:
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    p = np.array(data.site_xpos[sid], dtype=np.float64)
    return float(np.linalg.norm(p - np.asarray(p_des, dtype=np.float64).reshape(3)))


@dataclass
class ReachPlan:
    q0: np.ndarray
    q1: np.ndarray
    T_des: np.ndarray
    ts: np.ndarray
    qs: np.ndarray
    qds: np.ndarray
    qdds: np.ndarray


def build_reach_plan(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    p_des_world: np.ndarray,
    duration: float,
) -> ReachPlan:
    reset_arm_home(model, data)
    q0 = np.array(data.qpos[:7], dtype=np.float64)
    T_des = make_reach_pose(q0, p_des_world)
    q1, _, ik_err = solve_reach_ik(model, q0, T_des)
    dt = float(model.opt.timestep)
    ts, qs, qds, qdds = quintic_joint_trajectory(q0, q1, duration, dt)
    return ReachPlan(q0=q0, q1=q1, T_des=T_des, ts=ts, qs=qs, qds=qds, qdds=qdds)
