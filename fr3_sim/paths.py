from pathlib import Path

# fr3_sim/paths.py -> 仓库根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT_ROOT / "assets"
MENAGERIE_FR3 = PROJECT_ROOT / "third_party" / "mujoco_menagerie" / "franka_fr3"


def fr3_scene_xml() -> Path:
    """官方 menagerie 场景入口（含地面与关键帧）。"""
    p = MENAGERIE_FR3 / "scene.xml"
    if not p.is_file():
        raise FileNotFoundError(
            f"未找到模型文件: {p}\n"
            "请在仓库根目录执行稀疏克隆（仅拉取 franka_fr3）:\n"
            "  git clone --filter=blob:none --sparse "
            "https://github.com/google-deepmind/mujoco_menagerie.git "
            "third_party/mujoco_menagerie\n"
            "  cd third_party/mujoco_menagerie\n"
            "  git sparse-checkout set franka_fr3\n"
            "  git checkout main\n"
        )
    return p


def fr3_desk_pick_xml() -> Path:
    """桌面 + mocap 方块场景（含 FR3，无夹爪抓取由脚本驱动 mocap）。"""
    p = ASSETS / "fr3_desk_pick.xml"
    if not p.is_file():
        raise FileNotFoundError(f"未找到桌面场景: {p}")
    return p
