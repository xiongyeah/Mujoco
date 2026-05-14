"""启动 MuJoCo 官方 Simulate 交互式窗口（launch_from_path）。

场景使用 menagerie 的 FR3 scene.xml（机械臂 + 地面），无额外桌面物体。
运行：在仓库根目录执行  .\\.venv\\Scripts\\python scripts\\run_viewer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco.viewer

from fr3_sim.paths import fr3_scene_xml


def main() -> None:
    xml_path = str(fr3_scene_xml())
    # 官方封装：打开与 mujoco simulate 相同的 GUI，从 XML 路径加载
    mujoco.viewer.launch_from_path(xml_path)


if __name__ == "__main__":
    main()
