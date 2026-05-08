"""控制器统一接口（计划：controller(...) -> tau）。"""

from __future__ import annotations

from typing import Protocol

import mujoco
import numpy as np

from fr3_sim.dynamics import inverse_dynamics_tau
from fr3_sim.kinematics import pose_error_se3

# menagerie fr3.xml 关节力矩上限（近似）
_TAU_LIM_FR3 = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float64)


def clip_tau_fr3(tau: np.ndarray) -> np.ndarray:
    t = np.asarray(tau, dtype=np.float64).reshape(-1)
    n = min(t.size, _TAU_LIM_FR3.size)
    lim = _TAU_LIM_FR3[:n]
    return np.clip(t, -lim, lim)


class TorqueController(Protocol):
    """所有力矩控制器实现该接口；输入输出均在模型 nv 维关节空间。"""

    def compute_torque(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        q_des: np.ndarray,
        dq_des: np.ndarray,
    ) -> np.ndarray:
        ...


class ZeroTorque:
    def compute_torque(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        q_des: np.ndarray,
        dq_des: np.ndarray,
    ) -> np.ndarray:
        return np.zeros(model.nu, dtype=np.float64)


class JointPDGravity:
    """关节空间 PD + mj_forward 后的 qfrc_bias（含重力与速度相关项）。"""

    def __init__(self, kp: np.ndarray | float, kd: np.ndarray | float) -> None:
        self.kp = np.asarray(kp, dtype=np.float64).reshape(-1)
        self.kd = np.asarray(kd, dtype=np.float64).reshape(-1)

    def compute_torque(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        q_des: np.ndarray,
        dq_des: np.ndarray,
    ) -> np.ndarray:
        mujoco.mj_forward(model, data)
        q = data.qpos[: model.nu]
        dq = data.qvel[: model.nu]
        tau = self.kp * (q_des - q) + self.kd * (dq_des - dq) + data.qfrc_bias[: model.nu]
        return clip_tau_fr3(tau)


class ComputedTorquePD:
    """计算力矩：tau = ID( qdd_cmd )，其中 qdd_cmd = Kp(qd-q)+Kd(dqd-dq)（关节空间）。"""

    def __init__(self, kp: np.ndarray | float, kd: np.ndarray | float) -> None:
        self.kp = np.asarray(kp, dtype=np.float64).reshape(-1)
        self.kd = np.asarray(kd, dtype=np.float64).reshape(-1)

    def compute_torque(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        q_des: np.ndarray,
        dq_des: np.ndarray,
    ) -> np.ndarray:
        nu = model.nu
        mujoco.mj_forward(model, data)
        q = data.qpos[:nu]
        dq = data.qvel[:nu]
        qdd_cmd = self.kp * (q_des - q) + self.kd * (dq_des - dq)
        qacc = np.zeros(model.nv, dtype=np.float64)
        qacc[:nu] = qdd_cmd
        tau = inverse_dynamics_tau(model, data, qacc)
        return clip_tau_fr3(tau)


class CartesianImpedanceSite:
    """
    笛卡尔阻抗（末端 site）：F = Kp e + Kd (v_des - v)，tau = J^T F − kd_joint·dq。
    q_des / dq_des 不参与计算（占位以满足接口）；目标位姿在构造时给定。
    """

    def __init__(
        self,
        site_name: str,
        T_des: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        *,
        kd_joint: float = 4.0,
    ) -> None:
        self.site_name = site_name
        self.T_des = np.asarray(T_des, dtype=np.float64).reshape(4, 4).copy()
        self.kp = np.asarray(kp, dtype=np.float64).reshape(6)
        self.kd = np.asarray(kd, dtype=np.float64).reshape(6)
        self.kd_joint = float(kd_joint)

    def compute_torque(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        q_des: np.ndarray,
        dq_des: np.ndarray,
    ) -> np.ndarray:
        _ = q_des, dq_des
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.site_name)
        if sid < 0:
            raise ValueError(f"未找到 site: {self.site_name}")

        mujoco.mj_forward(model, data)
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        J = np.vstack([jacp[:, : model.nu], jacr[:, : model.nu]])

        T_cur = np.eye(4, dtype=np.float64)
        T_cur[:3, :3] = data.site_xmat[sid].reshape(3, 3)
        T_cur[:3, 3] = data.site_xpos[sid]

        e = pose_error_se3(T_cur, self.T_des)
        v = J @ data.qvel[: model.nu]
        F = self.kp * e + self.kd * (-v)
        dq = data.qvel[: model.nu]
        tau = J.T @ F - self.kd_joint * dq
        return clip_tau_fr3(tau)
