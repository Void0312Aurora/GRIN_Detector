# 反射结构化观测模型实验（2026-07-28）

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-07-28
- Verification Status: VERIFIED
- Version Label: structured_observation_v2_edge

## 结论

结构化静态模型改善了真实圆内纹理的统计分布，但仍不能正确生成真实图中的局部四向边缘聚焦，因此当前判定仍为 `structured_pilot_fail`。

该结果说明应拆开两个问题：

1. 相关纹理可以由“共同纹理 + 低维因子 + 分区高斯随机场”近似，不必使用全图 VAE。
2. 四向亮边不是普通纹理问题。残差层虽然能拟合一个角向修正场，却没有恢复最终图像中的真实边缘幅值和局部形态；下一步必须修改物理仿真器的场级边缘响应，而不是继续扩大观测残差模型。

## 模型

模型不是时序状态空间模型。对第 `i` 颗 lens 的对数强度残差使用：

\[
R_i(r,\theta)=E_i(r,\theta)+H_i(r,\theta)+T_0(r,\theta)+P_i(r,\theta)+G_i(r,\theta)
\]

- `E_i`：固定边缘宽度上的显式 `k=1,2,4` 圆谐波修正（v2）。
- `H_i`：128 个径向 bin 上的圆谐波系数，训练集秩为 3。
- `T_0`：20 张训练图的确定性稳健共同纹理。
- `P_i`：全图纹理残差的秩 2 因子。
- `G_i`：圆内、边缘、外场三个区域分别拟合功率谱后生成的相关随机场。

最终观测为 `clip(log(I_physics) + R_i, 0, 1)`。模型按 lens 划分；patch 和像素没有被计作独立样本。

## 固定数据与执行环境

- 数据：24 颗不同 microlens，每颗一张灰度 BMP。
- 内部划分：20 张训练；`21.bmp`–`24.bmp` 留出。
- 边界：历史仿真调参看过全部 24 张图，所以该留出只是回顾性内部验证，不是独立外部测试。
- 输入尺寸：`512 × 512`。
- GPU：NVIDIA GeForce RTX 3090。
- PyTorch：`2.9.1+cu128`。
- 随机种子：`20260728`。
- 仿真集成：4 次。
- 生成样本：64。

## 正式结果

| 方法 | 留出 log RMSE ↓ | 相对 PCA | 扩展特征 SWD ↓ | 纹理特征误差 ↓ | 生成多样性/真实多样性 |
|---|---:|---:|---:|---:|---:|
| 物理候选 | 0.172812 | +34.4% | 1.277270 | 0.153019 | 0.000 |
| PCA rank-5 | **0.128561** | 基准 | 1.391796 | 0.199287 | 0.706 |
| 结构化 v1 | 0.150741 | +17.3% | 1.095483 | **0.057367** | 0.598 |
| 结构化 v2 edge | 0.145510 | +13.2% | **1.054197** | 0.057403 | 0.644 |

v2 相比 v1：

- 留出 RMSE 降低约 `3.47%`。
- 扩展特征 SWD 降低约 `3.77%`。
- 相比 PCA，扩展特征 SWD 低约 `24.3%`。
- 相比纯物理候选，纹理特征均值误差低约 `62.5%`。

但预设的“结构化 oracle RMSE 不超过 PCA 5%”没有通过。其余三项通过：扩展 SWD 优于 PCA、最近训练图距离比为 `0.539`、纹理误差优于物理候选。

## 视觉与分项否决

综合 SWD 改善不能证明视觉成功：

- 真实留出图的 `k=4` 边缘幅值为 `0.043078`。
- 物理候选为 `0.033420`。
- 结构化 v2 生成图只有 `0.020324`。

因此 v2 的四向边缘幅值比真实值低约 `52.8%`，并且低于原物理候选。显式边缘修正图本身具有角向结构，但加回物理候选、共同纹理和随机场后，没有形成真实图中的局部白色聚焦。

圆内纹理也仍不完全正确：GRF 样本具有相关颗粒，不再是 VAE 的大块圆斑，但缺少真实图中连续的细密纹理；纹理特征 spread ratio 仅 `0.463`，说明生成变化幅度仍不足。

## 复现与工程验证

v2 使用相同命令独立复跑一次：

- verdict、4 项判据、全部重建指标和生成指标逐值一致。
- `structured_factors.npz` 和 `texture_spectrum.npz` 共 14 个数组逐元素一致。
- `reconstruction_metrics.csv` SHA-256：`22CF8D8FE1145C36A89F90134A696F0B304E086AF34C5A88D7B7BB8E60EEF9AF`。
- `generation_metrics.csv` SHA-256：`CDCCCE624A3D70CDF499044F42A3B32F99A74534938935B810913DB2680C1A68`。
- Ruff 检查通过；相关测试 `14 passed`。

工程异常记录：首次烟雾实验暴露 CUDA deterministic `median` 不支持，已改为 GPU 排序后显式取中间值；随后修复一个 NumPy 指标辅助函数错误。两次 v2 正式启动前的路径错误均发生在读取数据前，不产生实验结果。

## 统计边界检查（11/11）

| 检查项 | 状态 | 说明 |
|---|---|---|
| Simpson 悖论 | CAUTION | 只有一个历史采集批次，未能按独立批次分层复核。 |
| 生态谬误 | CAUTION | 像素/patch 不作为独立 lens；结论只针对 lens 级留出。 |
| Berkson 选择偏差 | CAUTION | 数据只包含可检测、可裁剪的有效 lens。 |
| Collider 偏差 | NOTE | 没有把由结果共同决定的变量作为协变量控制。 |
| 基率忽视 | NOTE | 本实验没有分类概率或诊断率指标。 |
| 回归均值 | NOTE | 没有按极端分数选择样本做前后比较。 |
| 幸存者偏差 | CAUTION | 失焦、损坏或未被检测到的 lens 未进入数据。 |
| Look-elsewhere | CAUTION | v1 判据预先固定；v2 是基于视觉失败提出的探索性机制消融。 |
| Garden of Forking Paths | CAUTION | 历史调参看过全部24张图，留出不是盲测；v2不改写v1判定。 |
| 相关不等于因果 | CAUTION | 外观残差拟合不能证明白光聚焦的物理成因。 |
| 反向因果 | NOTE | 没有把横截面观测解释为因果方向或时间演化。 |

没有 p 值或置信区间；4 张留出图不足以进行稳定的显著性检验。这里的数值用于模型工程比较，不是总体效应估计。

## 产物

- v1：`D:\workshop\Research\PICNN\analysis_outputs\reflection_structured_observation_v1_gpu`
- v2：`D:\workshop\Research\PICNN\analysis_outputs\reflection_structured_observation_v2_edge_gpu`
- v2 复跑：`D:\workshop\Research\PICNN\analysis_outputs\reflection_structured_observation_v2_edge_gpu_repro`
- 实现：`src/mini_grin_rebuild/models/structured_observation.py`
- 入口：`scripts/train_reflection_structured_observation.py`

## 下一阶段

1. 保留 v2 的 GRF 作为观测层纹理候选，不把它用于修补边缘机制。
2. 在 `optical_leakage_lite` 的复场/差分形成阶段加入角向可见度，而不是在相机前附加亮度残差：降低均匀基环，同时保留 `k=1` 照明方向和 `k=4` 聚焦方向。
3. 只使用20张校准图选择场级边缘参数；评价必须单列 focus coverage、peak/median、`h1/h4` 幅值和相位，不能再由综合 SWD 决定。
4. 场级边缘通过后，再联合结构化 GRF；最后用新的独立采集批次作外部验证。
