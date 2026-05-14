"""查看 .npz（NumPy 压缩包）里有哪些数组及形状；可选画关节曲线。

.npz 不是 Excel 表，需用 Python / NumPy（或自己导出为 CSV 再用表格软件打开）。

示例：
  .\\.venv\\Scripts\\python scripts\\inspect_npz.py outputs/m6_pickplace.npz
  .\\.venv\\Scripts\\python scripts\\inspect_npz.py outputs/m6_pickplace.npz --plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="列出 npz 中的数组；可选绘制 qpos–time")
    ap.add_argument("npz", type=str, help="例如 outputs/m6_pickplace.npz")
    ap.add_argument("--plot", action="store_true", help="若有 time 与 qpos，则绘制各关节角")
    args = ap.parse_args()

    path = Path(args.npz)
    if not path.is_file():
        print("文件不存在:", path.resolve(), file=sys.stderr)
        sys.exit(1)

    data = np.load(path, allow_pickle=True)
    print("文件:", path.resolve())
    print("键名:", list(data.files))
    for k in data.files:
        a = data[k]
        if hasattr(a, "shape"):
            print(f"  {k}: shape={a.shape} dtype={a.dtype}")
        else:
            print(f"  {k}: {type(a)} {a}")

    if args.plot:
        if "time" not in data.files or "qpos" not in data.files:
            print("无 time/qpos，跳过作图。", file=sys.stderr)
            return
        import matplotlib.pyplot as plt

        t = np.asarray(data["time"], dtype=np.float64).ravel()
        q = np.asarray(data["qpos"], dtype=np.float64)
        if q.ndim != 2 or q.shape[0] != len(t):
            print("qpos 与 time 长度不匹配，跳过作图。", file=sys.stderr)
            return
        n_j = min(7, q.shape[1])
        _, axs = plt.subplots(n_j, 1, sharex=True, figsize=(8, 10), constrained_layout=True)
        if n_j == 1:
            axs = [axs]
        for j in range(n_j):
            axs[j].plot(t, q[:, j])
            axs[j].set_ylabel(f"q{j}")
        axs[-1].set_xlabel("time (s)")
        plt.suptitle(path.name)
        plt.show()


if __name__ == "__main__":
    main()
