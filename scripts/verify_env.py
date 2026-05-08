"""M5 环境自检：加载 menagerie FR3 场景并前向动力学一步。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco

from fr3_sim.paths import fr3_scene_xml


def main() -> None:
    xml = fr3_scene_xml()
    print(f"加载 XML: {xml}")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}")
    print("qpos0 (前 min(14,nq) 个):", data.qpos[: min(14, model.nq)])

    print("\nBody 列表（供对照末端命名）:")
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        print(f"  [{bid}] {name}")

    x = data.xpos.copy()
    print("\nmj_forward 后: xpos 形状", x.shape, "（world 坐标系下各 body 原点）")


if __name__ == "__main__":
    main()
