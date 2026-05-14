# M2 · 阻尼最小二乘（Damped Least Squares, DLS）数学推导

> **公式预览**：建议用 **Markdown+Math**、**Typora** 或 **Obsidian** 打开。下文使用 `\| \|`、分块矩阵等写法，便于 **KaTeX** 渲染。

本文与「雅可比伪逆迭代」配套：在**奇异或接近奇异**的位形下，标准伪逆

$$
J^\dagger = J^{\top} \left( J J^{\top} \right)^{-1}
$$

中 $\left( J J^{\top} \right)^{-1}$ 放大数值误差，关节增量 $\Delta q = J^\dagger e$ 可能过大、迭代振荡。**DLS** 在同样的一步线性子问题中引入**阻尼**，改善病态性。

---

## 1. 问题来源：一步仍写 $J \Delta q \approx e$

与伪逆迭代相同，在当前关节角 $q$ 处线性化，希望一步关节增量 $\Delta q$ 满足

$$
J \, \Delta q \approx e,
$$

其中 $J \in \mathbb{R}^{m \times n}$（对 FR3 末端 6 维约束、7 关节时 $m=6$，$n=7$），$e \in \mathbb{R}^{m}$ 为当前任务空间误差（位置或位姿误差向量，与所用雅可比约定一致）。

**伪逆解**（无阻尼）在相容时取 $\Delta q = J^\dagger e$，等价于在「$J\Delta q=e$ 的所有解」中选 $\| \Delta q \|$ 最小者。当 $J$ 接近秩亏时，$J^\dagger$ **范数很大**，$\Delta q$ 不稳定。

---

## 2. Tikhonov 正则化（岭回归形式）

将硬约束 $J\Delta q = e$ 放松为**带正则的最小二乘**：

$$
\min_{\Delta q \in \mathbb{R}^{n}}
\;
\| J \Delta q - e \|_{2}^{2}
\;+\;
\lambda^{2} \, \| \Delta q \|_{2}^{2},
$$

其中 **$\lambda > 0$** 为**阻尼系数**（标量，也可推广为对角阵，此处从标量推导）。

**目标**：在拟合误差 $J\Delta q \approx e$ 的同时，惩罚过大的 $\Delta q$，从而抑制奇异方向上的爆炸。

---

## 3. 一阶最优性（法方程）

记目标函数

$$
\phi(\Delta q)
=
\left( J \Delta q - e \right)^{\top} \left( J \Delta q - e \right)
+
\lambda^{2} \, \Delta q^{\top} \Delta q.
$$

对 $\Delta q$ 求梯度并令为零：

$$
\frac{\partial \phi}{\partial (\Delta q)^{\top}}
=
2 J^{\top} \left( J \Delta q - e \right)
+
2 \lambda^{2} \Delta q
= 0.
$$

整理得 **法方程（normal equation）**：

$$
\left( J^{\top} J + \lambda^{2} I_{n} \right) \Delta q
=
J^{\top} e.
$$

当 $J^{\top} J + \lambda^{2} I$ **正定**（对 $\lambda>0$ 恒成立）时，有唯一解

$$
\Delta q
=
\left( J^{\top} J + \lambda^{2} I_{n} \right)^{-1}
J^{\top} e.
$$

此式常称为 DLS 的 **「左形式」**（$n \times n$ 求逆，$n=7$ 时规模小，适合实现）。

---

## 4. 等价「右形式」：$J^{\top} \left( J J^{\top} + \lambda^{2} I_{m} \right)^{-1} e$

与专栏 **2.3 阻尼最小二乘法** 中写出的形式一致（记号 $\Delta\theta$ 与 $\Delta q$、$e$ 与误差向量同义）：

$$
\Delta q
=
J^{\top}
\left( J J^{\top} + \lambda^{2} I_{m} \right)^{-1}
e,
\qquad
\lambda \in \mathbb{R},\ \lambda \neq 0.
$$

由 Tikhonov 问题第 2 节可严格推出该式与第 3 节左形式等价（矩阵恒等式 / 逆引理）。**只要 $\lambda>0$**，$J J^{\top} + \lambda^{2} I_{m}$ 为**正定**因而恒可逆，**不要求** $J$ 行满秩（奇异附近仍可用；秩亏时伪逆的 $J J^{\top}$ 不可逆，但阻尼后矩阵始终可逆）。

**与伪逆对比**：伪逆用 $\left( J J^{\top} \right)^{-1}$（仅当 $J J^{\top}$ 可逆）；DLS 用 $\left( J J^{\top} + \lambda^{2} I_{m} \right)^{-1}$。当 $\lambda \to 0$ 且 $J J^{\top}$ 可逆时，上式趋于 $J^\dagger e$，与伪逆一致。

**实现上**：$m=6$ 时 $\left( J J^{\top} + \lambda^{2} I_{6} \right)$ 为 **$6 \times 6$**，求逆或 `solve` 代价低；许多机器人 IK 代码采用此 **「右形式」**：

$$
\Delta q = J^{\top} \, x,
\qquad
\left( J J^{\top} + \lambda^{2} I_{m} \right) x = e.
$$

---

## 5. 奇异值视角（与专栏 SVD 写法对照）

对 $J$ 作奇异值分解（设秩为 $r$），专栏中给出的 DLS 算子可写成

$$
J^{\top} \left( J J^{\top} + \lambda^{2} I \right)^{-1}
=
\sum_{i=1}^{r}
\frac{\sigma_i}{\sigma_i^{2} + \lambda^{2}} \, v_i u_i^{\top},
$$

其中 $\sigma_i$ 为奇异值，$u_i$、$v_i$ 为 $J$ 的 SVD 中对应左右奇异向量（具体排列与 $J=U\Sigma V^{\top}$ 约定一致）。

与**普通伪逆**（无阻尼）在同一展开形式下可写为 $\sum_i \tau_i\, v_i u_i^{\top}$ 时：

- **DLS**：$\displaystyle \tau_i^{\mathrm{DLS}} = \frac{\sigma_i}{\sigma_i^{2} + \lambda^{2}}$；
- **简单伪逆**：$\displaystyle \tau_i^{\mathrm{pinv}} = \frac{1}{\sigma_i}$（当 $\sigma_i \to 0$ 时无界）。

直观上：$\tau_i^{\mathrm{DLS}}$ 在 $\sigma_i \to 0$ 时趋于 **$0$**（有界），而 $\tau_i^{\mathrm{pinv}}$ 爆炸；故 DLS 在奇异附近**稳定**。另：矩阵 $\left( J J^{\top} + \lambda^{2} I \right)^{-1}$ 在 $J J^{\top}$ 的特征基下，特征值由 **$1/\sigma_i^2$** 变为 **$1/(\sigma_i^{2} + \lambda^{2})$**；再左乘 $J^{\top}$ 后得到上式系数 $\sigma_i/(\sigma_i^{2}+\lambda^{2})$，两者叙述一致、层次不同。

**专栏补充（SDLS）**：对各奇异方向采用**不同**阻尼、依赖当前位形与误差，可少迭代、少调参，但需完整 SVD，计算量更大；本文档不展开，实现时与常数 $\lambda$ 的 DLS 区分即可。

---

## 6. 迭代中的 DLS 一步（与伪逆迭代并列）

在数值 IK 外层循环中，每一步：

1. 计算 $e(q)$、$J(q)$；
2. 用 DLS 解 $\Delta q$，例如

$$
\Delta q = J^{\top} \left( J J^{\top} + \lambda^{2} I \right)^{-1} e \,;
$$

3. 更新 $q \leftarrow q + \alpha \, \Delta q$（$\alpha$ 为步长，可与伪逆迭代相同）。

重复直至 $\| e \| < \varepsilon$ 或达最大迭代次数。

---

## 7. 阻尼 $\lambda$ 与步长 $\alpha$ 的选择（工程说明）

- **$\lambda$ 大**：更强调 $\| \Delta q \|$ 小，**更稳**但**收敛变慢**、对误差修正变「钝」。  
- **$\lambda$ 小**：接近伪逆，在远离奇异时跟踪好，但在奇异附近仍可能抖。  
- **$\alpha$**：与伪逆迭代相同，限制单步关节变化，改善线性化有效性。

常根据**可操作度**或**最小奇异值**自适应调 $\lambda$（课程加分项），最小实现可取常数 $\lambda$。

---

## 8. 与伪逆、与 `lstsq` 的关系小结

| 方法 | 一步 $\Delta q$（右形式示意） |
|------|-------------------------------|
| 伪逆 | $J^{\top} \left( J J^{\top} \right)^{-1} e$（$J J^\top$ 可逆时） |
| DLS | $J^{\top} \left( J J^{\top} + \lambda^{2} I \right)^{-1} e$ |

`numpy.linalg.lstsq(J, e)` 求的是 **无正则** 的 $\min \| J\Delta q - e \|^2$ 的最小范数解；**DLS 不能**用「同一个 `lstsq(J,e)`」直接代替，需显式实现上式或左形式 $\left( J^{\top} J + \lambda^{2} I \right)^{-1} J^{\top} e$。

---

## 9. 参考文献

- Nakamura, Y., *Advanced Robotics: Redundancy and Optimization* — 阻尼伪逆与奇异处理。  
- Siciliano et al., *Robotics: Modelling, Planning and Control* — 数值 IK、奇异性。  
- Wampler, C. W., «Manipulator inverse kinematic solutions based on vector formulations and damped least-squares methods», *Communications of the ACM*, 1986（经典 DLS 讨论来源之一）。

---

*与 M2 模块「伪逆 + DLS 对比」实验设计配套；实现待写入 `m2_ik` 时可与本文公式逐行对应。*
