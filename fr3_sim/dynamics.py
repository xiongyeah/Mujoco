"""MuJoCo 动力学辅助：稠密质量矩阵、逆动力学（给定 qacc）。"""

from __future__ import annotations

import mujoco
import numpy as np


def dense_mass_matrix(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """返回 M(q) ∈ R^{nv×nv}（需先 mj_forward 或本函数内调用）。"""
    mujoco.mj_forward(model, data)
    m = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, m, data.qM)
    return m


def inverse_dynamics_tau(model: mujoco.MjModel, data: mujoco.MjData, qacc: np.ndarray) -> np.ndarray:
    """
    给定广义加速度 qacc（长度 nv），返回与 nu 个执行器维度对齐的广义力前缀
    （Franka 场景 nu=nv=7 时即关节力矩）。
    调用前会 mj_forward。
    """
    qacc = np.asarray(qacc, dtype=np.float64).reshape(-1)
    nv = model.nv
    if qacc.size > nv:
        raise ValueError("qacc 长度不能超过 nv")
    qacc_full = np.zeros(nv, dtype=np.float64)
    qacc_full[: qacc.size] = qacc

    mujoco.mj_forward(model, data)
    backup = np.array(data.qacc, copy=True)
    data.qacc[:] = qacc_full
    mujoco.mj_inverse(model, data)
    tau = np.array(data.qfrc_inverse[: model.nu], dtype=np.float64)
    data.qacc[:] = backup
    return tau
