"""M2 逆运动学：伪逆迭代 + DLS（阻尼最小二乘）+ 零空间投影。

与 ``INTERFACE_SPEC_v0.md`` 第 1–2 节对齐。
"""

from __future__ import annotations

from m2_ik.ik import IKConvergenceError, ik_dls_step, ik_pinv_step, inverse_kinematics
from m2_ik.kinematics import (
    forward_kinematics,
    forward_kinematics_all_links,
    gravity,
    jacobian,
    pose_error_se3,
)
from m2_ik.null_space import (
    inverse_kinematics_with_nullspace,
    joint_limit_repulsion,
    manipulability,
    manipulability_gradient,
    null_space_projector,
)

__all__ = [
    # M1 · 运动学
    "forward_kinematics",
    "forward_kinematics_all_links",
    "jacobian",
    "gravity",
    "pose_error_se3",
    # M2 · 逆运动学
    "inverse_kinematics",
    "ik_pinv_step",
    "ik_dls_step",
    "IKConvergenceError",
    # M2 · 零空间
    "inverse_kinematics_with_nullspace",
    "null_space_projector",
    "joint_limit_repulsion",
    "manipulability",
    "manipulability_gradient",
]
