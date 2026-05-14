"""M1 自检：FK / 雅可比 与 MuJoCo mj_forward、mj_jacSite 对比。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from fr3_sim.kinematics import fk_fr3_attachment, geometric_jacobian_world
from fr3_sim.paths import fr3_scene_xml


def _T_from_xmat_xpos(xmat9: np.ndarray, xpos3: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(xmat9, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(xpos3, dtype=np.float64).reshape(3)
    return T


def main() -> None:
    xml = str(fr3_scene_xml())
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if sid < 0:
        raise RuntimeError("未找到 attachment_site")

    rng = np.random.default_rng(0)
    fk_pos_err = []
    fk_rot_err = []
    jac_err = []

    for _ in range(50):
        q = rng.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1])
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)

        T_mj = _T_from_xmat_xpos(data.site_xmat[sid], data.site_xpos[sid])
        T_us = fk_fr3_attachment(q)

        fk_pos_err.append(np.linalg.norm(T_mj[:3, 3] - T_us[:3, 3]))
        fk_rot_err.append(np.linalg.norm(T_mj[:3, :3] - T_us[:3, :3]))

        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        J_mj = np.vstack([jacp[:, :7], jacr[:, :7]])
        J_us = geometric_jacobian_world(q)
        jac_err.append(np.linalg.norm(J_mj - J_us))

    print("FK 位置误差 max:", float(np.max(fk_pos_err)))
    print("FK 旋转矩阵误差 max (Frobenius sense L2):", float(np.max(fk_rot_err)))
    print("Jacobian 误差 max:", float(np.max(jac_err)))


if __name__ == "__main__":
    main()
