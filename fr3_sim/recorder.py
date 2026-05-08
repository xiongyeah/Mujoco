"""仿真数据记录（M5）：时间序列堆叠为 ndarray / 可选 CSV。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SimulationRecorder:
    time: list[float] = field(default_factory=list)
    qpos: list[np.ndarray] = field(default_factory=list)
    qvel: list[np.ndarray] = field(default_factory=list)
    ctrl: list[np.ndarray] = field(default_factory=list)
    qfrc_applied: list[np.ndarray] = field(default_factory=list)
    extra: dict[str, list[Any]] = field(default_factory=dict)

    def record(self, t: float, data, ctrl: np.ndarray | None = None, qfrc_applied: np.ndarray | None = None) -> None:
        self.time.append(float(t))
        self.qpos.append(np.array(data.qpos, copy=True))
        self.qvel.append(np.array(data.qvel, copy=True))
        if ctrl is None:
            self.ctrl.append(np.array(data.ctrl, copy=True))
        else:
            self.ctrl.append(np.asarray(ctrl, dtype=np.float64).copy())
        if qfrc_applied is None:
            nv = len(data.qfrc_applied)
            self.qfrc_applied.append(np.zeros(nv))
        else:
            self.qfrc_applied.append(np.asarray(qfrc_applied, dtype=np.float64).copy())

    def stack(self) -> dict[str, np.ndarray]:
        """转为 (N, ...) 数组。"""
        out: dict[str, np.ndarray] = {
            "time": np.asarray(self.time, dtype=np.float64),
            "qpos": np.stack(self.qpos, axis=0) if self.qpos else np.zeros((0,)),
            "qvel": np.stack(self.qvel, axis=0) if self.qvel else np.zeros((0,)),
            "ctrl": np.stack(self.ctrl, axis=0) if self.ctrl else np.zeros((0,)),
            "qfrc_applied": np.stack(self.qfrc_applied, axis=0) if self.qfrc_applied else np.zeros((0,)),
        }
        for k, rows in self.extra.items():
            out[k] = np.stack(rows, axis=0) if rows else np.zeros((0,))
        return out
