"""M6 · Pick-and-place（桌面）：关节五次轨迹 + 末端位姿 IK；方块为 mocap，抓取阶段按 attachment_site 位姿同步。"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from fr3_sim.kinematics import fk_fr3_attachment
from fr3_sim.m6_reaching import solve_reach_ik
from fr3_sim.trajectory import quintic_joint_trajectory

# 与 assets/fr3_desk_pick.xml 中 keyframe desk_home 一致
DESK_HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=np.float64)

# 与 XML 中 pick_block / place_zone 初始位姿一致（米）
DEFAULT_PICK_BLOCK_CENTER = np.array([0.55, 0.12, 0.425], dtype=np.float64)
# place_zone body z=0.425，垫片 half 高 0.003 → 顶面 0.428；方块 half 0.025 → 静止时方块中心 z
PLACE_PAD_TOP_Z = 0.425 + 0.003
PICK_BLOCK_HALF_Z = 0.02  # 与 assets/fr3_desk_pick.xml 中 pick_geom half-size 一致（略小减轻与连杆视觉重叠）
PLACE_BLOCK_CENTER_Z = PLACE_PAD_TOP_Z + PICK_BLOCK_HALF_Z
DEFAULT_PLACE_CENTER = np.array([0.38, -0.2, PLACE_BLOCK_CENTER_Z], dtype=np.float64)


def reset_desk_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "desk_home")
    if kid < 0:
        raise RuntimeError("未找到 keyframe `desk_home`（请使用 fr3_desk_pick.xml）")
    mujoco.mj_resetDataKeyframe(model, data, kid)


def pick_block_mocapid(model: mujoco.MjModel) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_block")
    if bid < 0:
        raise RuntimeError("未找到 body `pick_block`")
    mid = int(model.body_mocapid[bid])
    if mid < 0:
        raise RuntimeError("`pick_block` 未设为 mocap")
    return mid


def reach_T_at_position(q: np.ndarray, p_world: np.ndarray) -> np.ndarray:
    """末端目标齐次矩阵：位置 p_world，姿态与当前 q 下 attachment 一致。"""
    T = fk_fr3_attachment(np.asarray(q, dtype=np.float64).reshape(7)).copy()
    T[:3, 3] = np.asarray(p_world, dtype=np.float64).reshape(3)
    return T


def pick_block_offset_local(model: mujoco.MjModel, data: mujoco.MjData, mocap_id: int) -> np.ndarray:
    """
    抓取瞬间：根据当前 mocap 方块中心与 attachment_site 计算常值局部偏移 off。

    满足 delta_world = mocap - site = R @ off，故 off = R.T @ delta_world；跟随阶段每步
    mocap = site + R @ off（R 为当前 site_xmat.reshape(3,3)）。
    """
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if sid < 0:
        raise RuntimeError("未找到 site `attachment_site`")
    mujoco.mj_forward(model, data)
    R = np.asarray(data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
    delta_w = np.asarray(data.mocap_pos[mocap_id], dtype=np.float64).reshape(3) - np.asarray(
        data.site_xpos[sid], dtype=np.float64
    ).reshape(3)
    return (R.T @ delta_w).astype(np.float64)


def sync_pick_block_mocap(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mocap_id: int,
    offset_local: np.ndarray,
) -> None:
    """将 mocap 方块与 attachment_site 保持固定相对位姿（offset_local 在 site 局部系）。"""
    off = np.asarray(offset_local, dtype=np.float64).reshape(3)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if sid < 0:
        raise RuntimeError("未找到 site `attachment_site`")
    mujoco.mj_forward(model, data)
    R = np.asarray(data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
    pos = np.asarray(data.site_xpos[sid], dtype=np.float64) + R @ off
    quat_xyzw = Rotation.from_matrix(R).as_quat()
    wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
    data.mocap_pos[mocap_id] = pos
    data.mocap_quat[mocap_id] = wxyz


def set_mocap_block_on_pad(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mocap_id: int,
    xy: np.ndarray | None = None,
) -> None:
    """释放后把方块对齐到放置垫（世界系水平、底面贴垫顶）。"""
    if xy is None:
        xy = DEFAULT_PLACE_CENTER[:2]
    xy = np.asarray(xy, dtype=np.float64).reshape(2)
    data.mocap_pos[mocap_id, 0] = xy[0]
    data.mocap_pos[mocap_id, 1] = xy[1]
    data.mocap_pos[mocap_id, 2] = float(PLACE_BLOCK_CENTER_Z)
    data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def blend_mocap_block_to_pad(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mocap_id: int,
    *,
    duration: float = 0.28,
    xy: np.ndarray | None = None,
    viewer: object | None = None,
) -> None:
    """
    从当前 mocap 位姿平滑过渡到垫上目标（位置 quintic ease、姿态 Slerp）。

    避免单帧跳变；不再末尾强制 set_mocap_block_on_pad，以免与插值终点重复产生微跳。
    本段只做 mj_forward，不 mj_step，机械臂状态不变。
    """
    if xy is None:
        xy = DEFAULT_PLACE_CENTER[:2]
    xy = np.asarray(xy, dtype=np.float64).reshape(2)
    p0 = np.array(data.mocap_pos[mocap_id], dtype=np.float64, copy=True)
    q0w = np.array(data.mocap_quat[mocap_id], dtype=np.float64, copy=True)
    p1 = np.array([xy[0], xy[1], float(PLACE_BLOCK_CENTER_Z)], dtype=np.float64)
    q1w = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    R0 = Rotation.from_quat(_quat_wxyz_to_xyzw(q0w))
    R1 = Rotation.from_quat(_quat_wxyz_to_xyzw(q1w))
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([R0, R1]))

    dt = float(model.opt.timestep)
    n = max(6, int(round(duration / dt)))

    def smootherstep(t: float) -> float:
        """Perlin 型 quintic：端点一阶、二阶导为 0，比 smoothstep 更顺。"""
        t = float(np.clip(t, 0.0, 1.0))
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    for k in range(1, n + 1):
        t = k / n
        s = smootherstep(t)
        data.mocap_pos[mocap_id] = (1.0 - s) * p0 + s * p1
        Ri = slerp(s)
        qw = _quat_xyzw_to_wxyz(Ri.as_quat())
        nrm = float(np.linalg.norm(qw))
        if nrm > 1e-9:
            data.mocap_quat[mocap_id] = qw / nrm
        mujoco.mj_forward(model, data)
        if viewer is not None:
            viewer.sync()

    data.mocap_pos[mocap_id] = p1
    data.mocap_quat[mocap_id] = q1w
    mujoco.mj_forward(model, data)
    if viewer is not None:
        viewer.sync()


@dataclass
class JointQuinticPlan:
    q0: np.ndarray
    q1: np.ndarray
    T_des: np.ndarray
    ts: np.ndarray
    qs: np.ndarray
    qds: np.ndarray
    qdds: np.ndarray


def build_cartesian_segment_plan(
    model: mujoco.MjModel,
    q0: np.ndarray,
    p_des_world: np.ndarray,
    duration: float,
) -> JointQuinticPlan:
    """从关节初值 q0 出发，末端位置目标 p_des_world（姿态随 q0），五次关节空间轨迹。"""
    q0 = np.asarray(q0, dtype=np.float64).reshape(7)
    T_des = reach_T_at_position(q0, p_des_world)
    q1, _, _ = solve_reach_ik(model, q0, T_des)
    dt = float(model.opt.timestep)
    ts, qs, qds, qdds = quintic_joint_trajectory(q0, q1, duration, dt)
    return JointQuinticPlan(q0=q0, q1=q1, T_des=T_des, ts=ts, qs=qs, qds=qds, qdds=qdds)


def build_joint_space_plan(q0: np.ndarray, q1: np.ndarray, duration: float, dt: float) -> JointQuinticPlan:
    """纯关节空间插值（无笛卡尔 T_des）。"""
    q0 = np.asarray(q0, dtype=np.float64).reshape(7)
    q1 = np.asarray(q1, dtype=np.float64).reshape(7)
    ts, qs, qds, qdds = quintic_joint_trajectory(q0, q1, duration, dt)
    T_des = fk_fr3_attachment(q1)
    return JointQuinticPlan(q0=q0, q1=q1, T_des=T_des, ts=ts, qs=qs, qds=qds, qdds=qdds)
