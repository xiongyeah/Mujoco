"""
M2 · 逆运动学（课程仿真方案独立模块）

包含：SE(3) 工具、FR3 正运动学/几何雅可比/位姿误差、数值 IK（伪逆单步 + DLS + 迭代求解）。
仿真其余部分（MuJoCo、轨迹、任务）仍通过 `fr3_sim` 引用本包；亦可直接 `import m2_inverse_kinematics`。
"""

from __future__ import annotations

from .ik import (
    ik_dls_step,
    ik_pinv_step,
    ik_solve,
    manipulability_translation,
    null_space_vector_from_jacobian,
)
from .kinematics import (
    fk_fr3_attachment,
    fk_fr3_link7,
    geometric_jacobian_world,
    pose_error_se3,
)
from .se3 import homogeneous, quat_wxyz_to_R, rotvec_to_R, rotz, skew, trans

__all__ = [
    "fk_fr3_attachment",
    "fk_fr3_link7",
    "geometric_jacobian_world",
    "homogeneous",
    "ik_dls_step",
    "ik_pinv_step",
    "ik_solve",
    "manipulability_translation",
    "null_space_vector_from_jacobian",
    "pose_error_se3",
    "quat_wxyz_to_R",
    "rotvec_to_R",
    "rotz",
    "skew",
    "trans",
]
