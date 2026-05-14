"""打开桌面 + mocap 方块场景的 MuJoCo Simulate 窗口（仅场景，不自动跑搬运轨迹）。

运行：.\\.venv\\Scripts\\python scripts\\run_viewer_desk.py

要边看窗口边跑完整 pick-and-place，请用：
  .\\.venv\\Scripts\\python scripts\\demo_m6_pickplace.py --viewer
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco.viewer

from fr3_sim.paths import fr3_desk_pick_xml


def main() -> None:
    mujoco.viewer.launch_from_path(str(fr3_desk_pick_xml()))


if __name__ == "__main__":
    main()
