"""仿真步进辅助：力矩控制试验时关闭内置执行器（位置伺服仍会残留力，需 disable actuation）。"""

from __future__ import annotations

from typing import NamedTuple

import mujoco
import numpy as np


class TorqueModeBackup(NamedTuple):
    actuator_gainprm: np.ndarray
    disableflags: int


def torque_only_mode(model: mujoco.MjModel, backup: bool = True) -> TorqueModeBackup | None:
    """
    1) actuator_gainprm 置零（避免位置伺服刚度）
    2) 设置 mjDSBL_ACTUATION：否则零增益下仍可能有执行器相关力，导致 qfrc_applied 与动力学不一致

    返回备份供 restore_torque_only_mode 恢复。
    """
    gains = np.array(model.actuator_gainprm, copy=True)
    model.actuator_gainprm[:] = 0.0
    dflags = int(model.opt.disableflags)
    model.opt.disableflags = dflags | int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    return TorqueModeBackup(gains, dflags) if backup else None


def restore_torque_only_mode(model: mujoco.MjModel, bak: TorqueModeBackup | None) -> None:
    if bak is None:
        return
    model.actuator_gainprm[:] = bak.actuator_gainprm
    model.opt.disableflags = bak.disableflags


def clear_applied_forces(data: mujoco.MjData) -> None:
    data.qfrc_applied[:] = 0.0


def step_torque(model: mujoco.MjModel, data: mujoco.MjData, tau: np.ndarray) -> None:
    """施加关节力矩并 mj_step 一步（长度与 nv 一致的前缀）。"""
    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    n = min(len(tau), len(data.qfrc_applied))
    data.qfrc_applied[:n] = tau[:n]
    mujoco.mj_step(model, data)
    data.qfrc_applied[:] = 0.0
