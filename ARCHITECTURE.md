# mini_grin_rebuild — 重写架构草案（面向审计与科研产出）

## 1. 重写目标（可审计、可复现实验、可发表）

当前核心研究代码位于 `mini_grin/`，但整体呈“脚本驱动 + 产物堆积”的形态，审计与复现成本高，不利于持续产出科研结果。主要问题包括：

- **运行依赖隐式**：缺少独立的项目元信息（依赖、版本、入口），运行强依赖 CWD/相对路径。
- **库代码与实验脚本耦合**：大量逻辑散落在 `mini_grin/scripts/*.py`，重复的输入拼装/指标计算难以统一维护。
- **产物与源码同级**：实验产物与源码同级堆积，难以做可复现的“run 目录”管理。

本次工作选择 **彻底重写**（不复制粘贴旧实现），目标是：

- 把项目变成一个**可审计的科研软件**：每次实验都能追溯“代码版本 + 配置 + 数据版本 + 环境 + 指标与图”。
- 把“核心算法”与“实验编排”解耦：算法可以被复用、被测试、被基准对比；实验只负责组合与记录。
- 把产物管理工程化：统一 run 目录，便于对比、回滚、复现、导出论文图表。

旧代码在短期内只作为“参考实现/对照基线”（用于验证新实现是否对齐），不作为新项目代码的直接来源。

## 2. 现有代码模块梳理（基于当前仓库）

核心依赖链大致如下：

- 配置：`mini_grin/core/configs.py`（`SimulationConfig`、`TrainingConfig`）
- 合成数据：`mini_grin/data/virtual_objects.py`（标准/参考/测试三元组、缺陷注入）
- 物理前向：
  - NumPy 版：`mini_grin/physics/simulator.py`（`simulate_capture`）
  - Torch 可微版：`mini_grin/physics/layer.py`（`DifferentiableGradientLayer`）
- 模型：`mini_grin/models/unetpp.py`、`mini_grin/models/defect_unet.py`
- 训练：
  - 单样本合成训练：`mini_grin/training/trainer.py`（`DefectTrainer`）
  - 数据集训练脚本：`mini_grin/scripts/train_dataset.py`
- 评估与可视化：`mini_grin/scripts/evaluate_*.py`、`mini_grin/visualization/plots.py`

## 3. 重写后的目录结构（建议）

> 采用“在新文件夹里重写”的方式：`mini_grin_rebuild/`，避免影响现有研究流程；同时让新工程从第一天起就满足审计与复现要求。

建议以 `src/` 作为包根目录（避免把仓库根目录当作 Python 包路径带来的隐式导入问题）：

```
mini_grin_rebuild/
  pyproject.toml
  README.md
  ARCHITECTURE.md
  PLAN.md
  configs/
    default.json
  src/
    mini_grin_rebuild/
      core/            # 配置/路径/seed/logging/typing
      data/            # dataset 格式定义、读取、合成数据生成
      physics/         # simulator(np) 与 layer(torch)
      models/          # 网络结构
      training/        # trainer/losses/metrics（库层）
      evaluation/      # 评估指标与导出（库层）
      visualization/   # 绘图
      cli/             # 统一 CLI 入口（argparse/typer）
  scripts/             # 临时脚本/兼容层（逐步收敛到 cli）
  tests/
  docs/
```

## 4. 审计与复现规范（必须尽早定下来）

### 4.1 每个实验必须可追溯

- 统一 run 目录：`mini_grin_rebuild/runs/<run_id>/`
- 每个 run 至少包含：
  - `config.json`（最终生效配置快照）
  - `meta.json`（时间、git commit、命令行、主机/环境摘要）
  - `metrics.json`（训练/评估指标，建议同时保存逐 step/epoch 与 summary）
  - `checkpoints/`（模型/优化器/随机状态）
  - `plots/`（论文图、对比图）

### 4.2 数据与输出路径禁止硬编码

- 库层代码不得依赖“当前工作目录”；所有 IO 路径必须来自配置或 run 目录。
- 禁止把数据集、checkpoint、plots 直接写到仓库根目录。

### 4.3 “库层”与“编排层”边界

- `src/mini_grin_rebuild/**`：算法与可复用逻辑（不写 argparse，不做隐式全局状态）。
- `cli/`：负责参数解析、run 目录创建、调用库层、落盘审计信息。

### 4.4 Loss 范围（保持旧项目“开启项”不变）

为降低审计复杂度，重写版只保留旧项目默认启用的 4–5 个 loss 组件：

- `diff`（数据项，始终存在；可选异方差 `logvar` 版本）
- `sr_diff`（权重 `1e-6`）
- `curl`（权重 `5e-4`）
- `sparsity`（权重 `5e-4`）
- `edge_suppress`（权重 `0.05`，ROI 半径 `defect_roi_radius=0.6`）

其它未开启的 loss（TV/SSIM/FFT 等）不进入重写版配置与实现。

## 5. 重写路线（高层）

重写按“先规范后算法”的顺序推进：

1. **定义最小可审计闭环**：配置→run 目录→日志→指标落盘（先不做训练也可以）。
2. **确定数据规范**：`.npz` / `.pt` / `.zarr` 选型与字段 schema；写校验器。
3. **实现物理前向与可微物理层**：每个模块都有 shape/dtype/device 的单元测试。
4. **实现训练与评估闭环**：保证同一配置可复现同一结果（至少统计意义一致）。
5. **沉淀论文级输出**：指标表、消融扫描、图表导出脚本与版本化配置。

更细的执行计划见 `mini_grin_rebuild/PLAN.md`。
