"""
FR3 正运动学、几何雅可比与重力矩（与 INTERFACE_SPEC_v0.md 第 1 节对齐）。

对外接口
-------
- ``forward_kinematics(q)`` → 末端 4×4 齐次矩阵
- ``jacobian(q)`` → 几何雅可比 J ∈ ℝ^{6×7}
- ``gravity(q)`` → 关节空间重力力矩 τ_g ∈ ℝ^{7}
- ``forward_kinematics_all_links(q)`` → 各连杆 4×4 矩阵列表（可选）
"""

from __future__ import annotations

import numpy as np

from m2_ik.se3 import homogeneous, quat_wxyz_to_R, rotz, trans


def _R_wxyz(w: float, x: float, y: float, z: float) -> np.ndarray:
    return quat_wxyz_to_R(w, x, y, z)


def _segment_fixed_A(j: int) -> np.ndarray:
    # 固定段变换 A_j（不含关节旋转），链式：A_j · Rz(q_j)
    if j == 0:
        return trans(0.0, 0.0, 0.333)
    if j == 1:
        return homogeneous(_R_wxyz(1.0, -1.0, 0.0, 0.0), np.zeros(3))
    if j == 2:
        return trans(0.0, -0.316, 0.0) @ homogeneous(_R_wxyz(1.0, 1.0, 0.0, 0.0), np.zeros(3))
    if j == 3:
        return trans(0.0825, 0.0, 0.0) @ homogeneous(_R_wxyz(1.0, 1.0, 0.0, 0.0), np.zeros(3))
    if j == 4:
        return trans(-0.0825, 0.384, 0.0) @ homogeneous(_R_wxyz(1.0, -1.0, 0.0, 0.0), np.zeros(3))
    if j == 5:
        return homogeneous(_R_wxyz(1.0, 1.0, 0.0, 0.0), np.zeros(3))
    if j == 6:
        return trans(0.088, 0.0, 0.0) @ homogeneous(_R_wxyz(1.0, 1.0, 0.0, 0.0), np.zeros(3))
    raise IndexError(j)


def fk_link7(q: np.ndarray) -> np.ndarray:
    # FK 递推：^0T_link7 = Π_j ( A_j · Rz(q_j) )
    q = np.asarray(q, dtype=np.float64).reshape(7)
    T = np.eye(4, dtype=np.float64)
    for j in range(7):
        Aj = _segment_fixed_A(j)
        T = T @ Aj @ homogeneous(rotz(q[j]), np.zeros(3))
    return T


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """末端 attachment（法兰 + 0.107 m）在基座系下的 4×4 齐次矩阵。"""
    # IK 里记 T(q)：link7 再乘末端偏移，得到与雅可比一致的末端帧
    return fk_link7(q) @ trans(0.0, 0.0, 0.107)


def pose_error_se3(T_cur: np.ndarray, T_des: np.ndarray) -> np.ndarray:
    """6 维误差 [Δp; log(R_err)]，与几何雅可比同一约定。"""
    # 旋转误差：R_err = R_d · R^T；与平移误差一起组成 e ∈ R^6
    R_err = T_des[:3, :3] @ T_cur[:3, :3].T
    t = T_des[:3, 3] - T_cur[:3, 3]  # 位置部分 e_p = p_d - p
    tr = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(tr))
    if theta < 1e-8:
        w = np.zeros(3, dtype=np.float64)
    elif theta > np.pi - 1e-6:
        # θ ≈ π：sinθ ≈ 0，标准轴角公式会 0/0。
        # R_err 对称，轴为 (R_err + I) 最大范数列，幅值 π。
        m = R_err + np.eye(3)
        idx = int(np.argmax(np.linalg.norm(m, axis=0)))
        axis = m[:, idx]
        w = (axis / np.linalg.norm(axis)) * np.pi
    else:
        # 姿态部分：由 R_err 提取等效轴角
        w_hat = (R_err - R_err.T) / (2.0 * np.sin(theta))
        w = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]], dtype=np.float64) * theta
    return np.concatenate([t, w], axis=0)


def jacobian(q: np.ndarray, ee_offset_local: np.ndarray | None = None) -> np.ndarray:
    """几何雅可比 J，世界系 v = [ṗ; ω] = J @ q̇。"""
    q = np.asarray(q, dtype=np.float64).reshape(7)
    if ee_offset_local is None:
        ee_offset_local = np.array([0.0, 0.0, 0.107], dtype=np.float64)

    # 前向递推各关节轴 z_j 与轴上点 o_j（几何法列雅可比）
    T = np.eye(4, dtype=np.float64)
    z_cols: list[np.ndarray] = []
    o_cols: list[np.ndarray] = []

    for j in range(7):
        Aj = _segment_fixed_A(j)
        P = T @ Aj
        R = P[:3, :3]
        p = P[:3, 3]
        z_cols.append(R[:, 2])
        o_cols.append(p.copy())
        T = P @ homogeneous(rotz(q[j]), np.zeros(3))

    Tee = T @ trans(ee_offset_local[0], ee_offset_local[1], ee_offset_local[2])
    p_e = Tee[:3, 3]  # 末端（attachment）位置，用于叉乘列

    # 组装 J：平移列 z×(p_e−o_j)，旋转列 z_j（标准几何雅可比）
    J = np.zeros((6, 7), dtype=np.float64)
    for j in range(7):
        z_j = z_cols[j]
        o_j = o_cols[j]
        J[:3, j] = np.cross(z_j, p_e - o_j)
        J[3:, j] = z_j
    return J


# ── 末端偏移参数（与 jacobian 默认 ee_offset_local 一致） ──
_EE_OFFSET = np.array([0.0, 0.0, 0.107], dtype=np.float64)


def forward_kinematics_all_links(q: np.ndarray) -> list[np.ndarray]:
    """返回 link1 ∼ link7 + flange 在基座系下的 4×4 齐次矩阵，共 8 个。"""
    q = np.asarray(q, dtype=np.float64).reshape(7)
    frames: list[np.ndarray] = []
    T = np.eye(4, dtype=np.float64)
    for j in range(7):
        Aj = _segment_fixed_A(j)
        T = T @ Aj @ homogeneous(rotz(q[j]), np.zeros(3))
        frames.append(T.copy())
    # flange = link7 + 末端偏移
    frames.append(T @ trans(_EE_OFFSET[0], _EE_OFFSET[1], _EE_OFFSET[2]))
    return frames


# ── FR3 连杆质量与质心偏移（来自 menagerie MJCF / franka_ros） ──
# 每项：(mass_kg, com_xyz_in_link_frame)
_LINK_INERTIA: list[tuple[float, np.ndarray]] = [
    (4.0, np.array([0.0, 0.0, 0.015])),        # link1
    (4.0, np.array([0.0001, -0.106, 0.003])),   # link2
    (3.0, np.array([0.070, 0.0, 0.040])),        # link3
    (3.5, np.array([0.035, 0.0, 0.017])),        # link4
    (2.5, np.array([0.010, 0.0, 0.030])),        # link5
    (1.5, np.array([0.0, 0.0, 0.004])),          # link6
    (0.5, np.array([0.0, 0.0, 0.050])),          # link7（含法兰）
]
_GRAVITY = 9.81  # m/s², -z 方向


def gravity(q: np.ndarray) -> np.ndarray:
    r"""计算 FR3 在当前关节角下的关节空间重力力矩。

    Parameters
    ----------
    q : (7,)
        关节角，单位 rad。

    Returns
    -------
    g : (7,)
        重力补偿力矩，单位 N·m。

    Notes
    -----
    基于各连杆 CoM（质心）在 world frame 下的位置，通过
    :math:`\tau_{g,j} = \sum_{i=j}^{7} m_i \, \mathbf{g}
    \cdot \bigl( \mathbf{z}_j \times (\mathbf{p}_{\mathrm{com},i}
    - \mathbf{o}_j) \bigr)` 计算。

    ``_LINK_INERTIA`` 中的质量与 CoM 偏移来自 Franka menagerie MJCF，
    与 MuJoCo 模型一致；**不依赖 MuJoCo 运行时**，适合无仿真环境的 IK /
    控制器调试。
    """
    q = np.asarray(q, dtype=np.float64).reshape(7)

    # 第一步：前向递推各关节的 {z_j, o_j} 与各连杆 CoM 世界坐标
    T = np.eye(4, dtype=np.float64)
    z_cols: list[np.ndarray] = []   # 关节轴方向 (世界系)
    o_cols: list[np.ndarray] = []   # 关节原点 (世界系)
    com_positions: list[np.ndarray] = []  # 各连杆 CoM 世界坐标

    for j in range(7):
        Aj = _segment_fixed_A(j)
        P = T @ Aj                     # 关节 j 的基座侧变换
        z_cols.append(P[:3, :2].copy())  # 注意：z 轴是第 3 列
        # 修正：z_j = R_j[:, 2]
        z_cols[j] = P[:3, 2].copy()
        o_cols.append(P[:3, 3].copy())

        # 关节 j 旋转后的变换
        T_joint = T @ Aj @ homogeneous(rotz(q[j]), np.zeros(3))

        # link j 的 CoM 在 world frame 下的位置
        mass, com_local = _LINK_INERTIA[j]
        T_com = T_joint @ trans(com_local[0], com_local[1], com_local[2])
        com_positions.append(T_com[:3, 3].copy())

        T = T_joint

    # 第二步：计算重力矩
    g_vec = np.array([0.0, 0.0, -_GRAVITY], dtype=np.float64)
    tau_g = np.zeros(7, dtype=np.float64)

    for j in range(7):
        z_j = z_cols[j]
        o_j = o_cols[j]
        total = 0.0
        for i in range(j, 7):  # 只有 j ≤ i 的连杆有贡献
            m_i, _ = _LINK_INERTIA[i]
            p_com = com_positions[i]
            # τ_g,j += m_i * g · (z_j × (p_com - o_j))
            total += m_i * float(np.dot(g_vec, np.cross(z_j, p_com - o_j)))
        tau_g[j] = total

    return tau_g
