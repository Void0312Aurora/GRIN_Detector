# Gating / QC（门控）用于“可发表/可落地”的相位重建流程

本项目的目标不是只在合成数据上拿到更低的 RMSE，而是要在 **无符号（signless）测量导致的病态逆问题**下，提供一条可审计、可复现、可在真实数据落地的重建与缺陷检测链路。

为此需要一个 **不依赖 ground-truth** 的质量控制（QC / gating）机制，用来回答：

- 这一次重建是否可信？
- 边缘/视场外伪影是否足以污染缺陷判读？
- 在接入真实数据后，是否发生了域偏移（domain shift）或采集异常？

## 1) 门控指标（均为 GT-free）

### A. 物理一致性残差（Physics residual）

对预测缺陷 `d̂`（高度域）与标准面 `h_std`，通过物理前向得到预测的测量：

- `ΔÎ_x = I_x(h_std + d̂) - I_x(h_std)`
- `ΔÎ_y = I_y(h_std + d̂) - I_y(h_std)`

与真实测量 `ΔI_x, ΔI_y` 对比，定义残差幅值：

`r = sqrt((ΔÎ_x-ΔI_x)^2 + (ΔÎ_y-ΔI_y)^2)`

统计（仅在 aperture 内）：

- `physics_rmse`：`r` 的 RMSE
- `physics_p95_abs`：`|r|` 的 95% 分位

这两项直接对应“前向可解释性/物理闭环”，在真实数据上也可计算。

### B. 边缘伪影（Edge band leakage）

很多伪影来自 padding/边界条件/视场截断，表现为在透镜边缘环带出现非物理结构。

在 aperture 内的边缘环带（默认 `r ∈ (0.9R, R]`，由 `training.eval_edge_band_start_frac` 控制）统计：

- `edge_mean_abs`：`mean(|d̂|)`
- `edge_p95_abs`：`p95(|d̂|)`

### C. 视场外泄漏（Outside leakage）

在 aperture 外（`r > R`）统计：

- `outside_mean_abs`
- `outside_p95_abs`

该指标常用于发现：光阑检测错误、FOV 未对准、多透镜入镜、或者算法在无效区域“胡乱填充”。

### D. 不确定性（可选、默认不做硬门控）

若模型输出 `logvar`（像素级不确定性），可记录：

- `logvar_mean`（aperture 内均值）

注意：不同方法/不同 head 的 `logvar` 分布不可直接对齐，且传统算法没有该输出，因此默认只记录、不作为硬门控阈值。

## 2) 阈值如何设定（推荐：从验证集分位数校准）

门控阈值不建议拍脑袋给常数；推荐做法：

1. 选定一个“当前最可靠”的方法（通常是最好的 NN）作为参考；
2. 在 `val` 上计算上述 QC 指标分布；
3. 取 `q=0.99`（或 `0.995`）分位数作为上限阈值；
4. 将阈值固化到配置文件，作为后续模型优化与对比的统一门槛。

对应脚本：

- `mini_grin_rebuild/scripts/gate_report.py`（支持 `--calibrate-split val --quantile 0.99`）

本仓库已提供一个固化阈值的配置示例：

- `mini_grin_rebuild/configs/benchmark_microlens200_srt_gated_q99.json`

## 3) 如何运行

### A. 输出门控报告（推荐用于论文/对比/调参）

示例（CPU 可跑；有 GPU 可把 `--device cpu` 去掉）：

```bash
python mini_grin_rebuild/scripts/gate_report.py \
  --config mini_grin_rebuild/configs/benchmark_microlens200_srt.json \
  --data-root mini_grin_rebuild/data/microlens_srt_dataset64_fov200 \
  --split test \
  --calibrate-split val --quantile 0.99 \
  --include-oracle \
  --device cpu \
  --ckpt nn_reflect_padding=/path/to/best.pt \
  --ckpt nn_prior_residual_input=/path/to/best.pt \
  --num-plots 10
```

产物：

- `runs/<...>/gate_report.json`：每个样本每个方法的 QC 指标与 pass/fail
- `runs/<...>/plots/*.png`：失败样本的可视化（按伪影指标排序）

### B. 在常规评估里记录 QC（推荐用于 suite 表格）

当配置里设置了 `training.gate_*` 阈值后，`mini-grin eval` / `mini-grin baseline` 会在 `eval_metrics.json` 增加 `qc` 字段：

```bash
PYTHONPATH=mini_grin_rebuild/src python -m mini_grin_rebuild.cli eval \
  --config mini_grin_rebuild/configs/benchmark_microlens200_srt_gated_q99.json \
  --data-root mini_grin_rebuild/data/microlens_srt_dataset64_fov200 \
  --checkpoint /path/to/best.pt --split test --num-plots 0
```

## 4) 接入真实数据时的处理建议（最小闭环）

真实数据没有 `defect_true`，因此训练/评估要把“依赖 GT 的指标”与“GT-free 的 QC 指标”分层：

1. **数据预处理标准化**：平场/暗场、强度归一化、坏点处理、FOV 裁剪与对准（建议把这些步骤版本化并写入 run meta）。
2. **光阑/aperture 识别**：从强度场或结构光特征估计透镜口径与中心，得到 `aperture mask`（门控中的 outside/edge 指标依赖它）。
3. **物理层校准**：用标准样或已知样件拟合 `DifferentiableGradientLayer` 的增益/偏置/模糊等参数，避免“模型对错误物理层拟合得很好但不可解释”。
4. **门控阈值重校准**：采集一批人工确认“无异常采集”的样本作为 real-val，重新统计 QC 分布并更新阈值（同样建议用分位数策略）。
5. **漂移监控**：上线后持续记录 QC 指标分布；一旦明显偏移即可判定采集/环境/器件状态变化，触发重新标定或微调。

