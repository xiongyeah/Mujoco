"""
M3 轨迹规划：关节空间五次多项式、关节空间 LSPB（分量独立、同步总时长）、
笛卡尔位置直线 + 姿态 SLERP。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _quintic_scalar_coeffs(
    q0: float,
    qf: float,
    v0: float,
    vf: float,
    a0: float,
    af: float,
    T: float,
) -> np.ndarray:
    """返回 [c0..c5]，使 q(t)=sum c_k t^k 满足端点位置/速度/加速度。"""
    if T <= 0:
        raise ValueError("T 必须为正")
    c0 = q0
    c1 = v0
    c2 = 0.5 * a0
    b0 = qf - (c0 + c1 * T + c2 * T * T)
    b1 = vf - (c1 + 2 * c2 * T)
    b2 = af - (2 * c2)
    A = np.array(
        [
            [T**3, T**4, T**5],
            [3 * T**2, 4 * T**3, 5 * T**4],
            [6 * T, 12 * T**2, 20 * T**3],
        ],
        dtype=np.float64,
    )
    c3, c4, c5 = np.linalg.solve(A, np.array([b0, b1, b2], dtype=np.float64))
    return np.array([c0, c1, c2, c3, c4, c5], dtype=np.float64)


def _quintic_eval(c: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c0, c1, c2, c3, c4, c5 = c
    t = np.asarray(t, dtype=np.float64)
    q = c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5
    qd = c1 + 2 * c2 * t + 3 * c3 * t**2 + 4 * c4 * t**3 + 5 * c5 * t**4
    qdd = 2 * c2 + 6 * c3 * t + 12 * c4 * t**2 + 20 * c5 * t**3
    return q, qd, qdd


def quintic_joint_trajectory(
    q0: np.ndarray,
    qf: np.ndarray,
    T: float,
    dt: float,
    *,
    v0: np.ndarray | None = None,
    vf: np.ndarray | None = None,
    a0: np.ndarray | None = None,
    af: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    各关节独立同总时长 T 的五次多项式。
    返回 (time, q, qd, qdd)，形状 (N, n_dof)。
    """
    q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
    qf = np.asarray(qf, dtype=np.float64).reshape(-1)
    n = q0.size
    z = np.zeros(n, dtype=np.float64)
    v0 = z if v0 is None else np.asarray(v0, dtype=np.float64).reshape(-1)
    vf = z if vf is None else np.asarray(vf, dtype=np.float64).reshape(-1)
    a0 = z if a0 is None else np.asarray(a0, dtype=np.float64).reshape(-1)
    af = z if af is None else np.asarray(af, dtype=np.float64).reshape(-1)

    ts = np.arange(0.0, T + 0.5 * dt, dt, dtype=np.float64)
    coeffs = np.stack([_quintic_scalar_coeffs(q0[i], qf[i], v0[i], vf[i], a0[i], af[i], T) for i in range(n)], axis=0)

    q = np.zeros((len(ts), n), dtype=np.float64)
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    for i in range(n):
        qi, qdi, qddi = _quintic_eval(coeffs[i], ts)
        q[:, i] = qi
        qd[:, i] = qdi
        qdd[:, i] = qddi
    return ts, q, qd, qdd


def lspb_min_duration(D: float, vmax: float, amax: float) -> float:
    """|D| 在对称梯形（vmax, amax）下的最短时间。"""
    D = float(abs(D))
    if D < 1e-12:
        return 0.0
    vmax = float(abs(vmax))
    amax = float(abs(amax))
    Ta = vmax / amax
    Da = 0.5 * amax * Ta**2
    if D <= 2.0 * Da + 1e-12:
        return float(2.0 * np.sqrt(D / amax))
    Tc = (D - 2.0 * Da) / vmax
    return float(2.0 * Ta + Tc)


def _lspb_profile_times(D: float, vmax: float, amax: float) -> tuple[float, float, float, float]:
    """
    返回 (Ta, Tc, T, v_peak)，对称加减速：0..Ta 加速，Ta..Ta+Tc 匀速，Ta+Tc..T 减速。
    """
    D = float(abs(D))
    if D < 1e-12:
        return 0.0, 0.0, 0.0, 0.0
    vmax = float(abs(vmax))
    amax = float(abs(amax))
    Ta = vmax / amax
    Da = 0.5 * amax * Ta**2
    if D <= 2.0 * Da + 1e-12:
        Ta = float(np.sqrt(D / amax))
        v_peak = amax * Ta
        return Ta, 0.0, 2.0 * Ta, v_peak
    Tc = (D - 2.0 * Da) / vmax
    return Ta, float(Tc), float(2.0 * Ta + Tc), vmax


def lspb_scalar_eval(
    t: np.ndarray,
    q0: float,
    qf: float,
    vmax: float,
    amax: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """标量 LSPB：使用最短可行时间；返回 q, qd, qdd。"""
    D = float(qf - q0)
    sgn = 1.0 if D >= 0 else -1.0
    dist = abs(D)
    Ta, Tc, T, v_peak = _lspb_profile_times(dist, vmax, amax)
    t = np.asarray(t, dtype=np.float64)
    q = np.zeros_like(t)
    qd = np.zeros_like(t)
    qdd = np.zeros_like(t)
    amax = float(abs(amax))
    for k in range(len(t)):
        tt = float(t[k])
        if tt <= 0.0:
            u = ud = udd = 0.0
        elif tt >= T:
            u, ud, udd = dist, 0.0, 0.0
        elif tt < Ta:
            udd = amax
            ud = amax * tt
            u = 0.5 * amax * tt**2
        elif tt < Ta + Tc:
            u_acc = 0.5 * amax * Ta**2
            ud = v_peak
            u = u_acc + v_peak * (tt - Ta)
            udd = 0.0
        else:
            tau = tt - (Ta + Tc)
            udd = -amax
            ud = v_peak - amax * tau
            u_acc = 0.5 * amax * Ta**2
            u = u_acc + v_peak * Tc + v_peak * tau - 0.5 * amax * tau**2
        q[k] = q0 + sgn * u
        qd[k] = sgn * ud
        qdd[k] = sgn * udd
    return q, qd, qdd


def lspb_joint_trajectory(
    q0: np.ndarray,
    qf: np.ndarray,
    vmax: float,
    amax: float,
    dt: float,
    *,
    time_stretch: float = 1.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    关节空间直线 + 标量 LSPB：沿 q(s)=q0+s*(qf-q0)，s∈[0,1]，
    标量 s 的“位移长度”取 ||qf-q0||₂，使各关节同步到达终点。
    """
    q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
    qf = np.asarray(qf, dtype=np.float64).reshape(-1)
    dq = qf - q0
    D_norm = float(np.linalg.norm(dq))
    if D_norm < 1e-12:
        ts = np.array([0.0], dtype=np.float64)
        return ts, np.tile(q0, (1, 1)), np.zeros((1, q0.size)), np.zeros((1, q0.size))

    T = float(max(lspb_min_duration(D_norm, vmax, amax) * time_stretch, 1e-3))
    ts = np.arange(0.0, T + 0.5 * dt, dt, dtype=np.float64)
    u, ud, udd = lspb_scalar_eval(ts, 0.0, D_norm, vmax, amax)
    s = (u / D_norm).reshape(-1, 1)
    sd = (ud / D_norm).reshape(-1, 1)
    sdd = (udd / D_norm).reshape(-1, 1)
    q = q0.reshape(1, -1) + s * dq.reshape(1, -1)
    qd = sd * dq.reshape(1, -1)
    qdd = sdd * dq.reshape(1, -1)
    return ts, q, qd, qdd


def cartesian_linear_slerp(
    T_start: np.ndarray,
    T_end: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    """
    对每个 alpha，返回 4x4 齐次位姿：位置线性插值，姿态 SLERP。
    alphas: 一维数组，元素在 [0,1]。
    返回 (len(alphas), 4, 4)。
    """
    alphas = np.asarray(alphas, dtype=np.float64).reshape(-1)
    p0 = T_start[:3, 3]
    p1 = T_end[:3, 3]
    R0 = Rotation.from_matrix(T_start[:3, :3])
    R1 = Rotation.from_matrix(T_end[:3, :3])
    key_times = [0.0, 1.0]
    key_rots = Rotation.concatenate([R0, R1])
    slerp = Slerp(key_times, key_rots)

    out = np.zeros((len(alphas), 4, 4), dtype=np.float64)
    out[:, 3, 3] = 1.0
    for k, a in enumerate(np.clip(alphas, 0.0, 1.0)):
        out[k, :3, 3] = (1.0 - a) * p0 + a * p1
        out[k, :3, :3] = slerp(float(a)).as_matrix()
    return out
