# 项目进展梳理（2026-05-22）

## 1. 当前项目基线

当前工作区已经完成从旧 `mini_grin` / `pinns_modular` 向 `mini_grin_rebuild/` 的主线切换。

- 活跃研究代码：`src/mini_grin_rebuild/`
- 活跃实验入口：`mini_grin_rebuild/scripts/` 与 `mini-grin` CLI
- 历史实现：仅作为兼容性对照，不再是日常开发入口

当前仓库的判断应当是：

**`mini_grin_rebuild/` 已经不是“重写准备阶段”，而是进入了“主骨架基本稳定、开始针对方法边界做压力测试”的阶段。**

## 2. 已经完成并稳定下来的部分

### 2.1 工程与审计骨架

以下基础设施已经成型：

- 配置系统、run 目录、`config/meta/metrics` 审计落盘
- 统一 CLI 骨架
- `runs/ + runs_archive/ + run_refs/` 分层 run 管理
- 基本测试体系与核心模块分层

截至目前，`docs/RUNS.md` 记录：

- 总 run 数：`104`
- 总体量：约 `29.7 GiB`
- 顶层 live run：`3`
- archived run：`101`

这说明项目已经摆脱“实验结果散堆在仓库根目录”的早期状态。

### 2.2 synthetic 主闭环

当前 synthetic 主线已经具备完整闭环：

- 合成样本生成
- 物理前向
- PINN/残差训练
- baseline 对比
- 评估与图表输出

从现有文档与脚本看，`microlens_srt` 仍然是当前最核心、最稳定的主训练集。

### 2.3 GT-free QC / gating

`docs/GATING.md` 对项目当前的“可落地性”非常关键，说明主线目标已经不只是追求 synthetic 指标，而是在构建：

- 物理一致性残差
- edge / outside leakage
- 无 GT 的 QC 通过率

这部分已经从“想法”推进到“配置、脚本、阈值校准和评估输出”层面，是当前项目的重要资产。

### 2.4 外部数据与真实实验前置准备

下面这条支线已经铺好接口，但还没有进入正式真实闭环：

- `external_data/` 下载、解压、扫描与 smoke 流程
- `docs/REAL_DATA_EXPERIMENT_PLAN.md`
- `docs/QDIC_WLI_MICROLENS_ACQUISITION_PLAN.md`
- `docs/QDIC_MICROLENS_EXPERIMENT_REPORT.md`

也就是说，真实数据不是“完全没开始”，而是已经完成了方案、接口和实验规范的准备阶段。

## 3. 当前正在推进的最新方向

最近一轮未提交改动的主题非常集中，主线已经明显转向 **wrap-stress / coarse-prior 边界分析**。

### 3.1 粗先验不再只有一条路

`reconstruction/` 当前正在从单一 `pseudo_poisson` 扩展到多种 coarse prior：

- `pseudo_poisson`
- `first_order_poisson`
- `first_order_sign_quadratic_poisson`

同时，训练输入与 baseline 评估已经开始支持 prior method 切换，这说明研究问题已经从“要不要先验”推进到了：

**“哪一种先验在 wrap pressure 下更稳，以及该如何注入网络。”**

### 3.2 wrap / phase-jump 压力测试已经进入实施阶段

这部分不是停留在想法层面，而是已经出现成体系的配置、脚本和测试：

- `configs/wrap_stress_mixed_v1.json`
- `scripts/train_phase_jump_suite.py`
- `scripts/sweep_phase_jump_scale.py`
- `scripts/sweep_wrap_mixed_baselines.py`
- `scripts/in_wrap_hotspot_audit.py`
- `scripts/in_wrap_phase_gradient_audit.py`
- `docs/NEXT_STEP_PHASE_JUMP_PLAN.md`

这表明当前主线的下一阶段并不是“泛泛地继续调参”，而是在系统追问：

**当前 `pseudo_poisson prior + residual input` 骨架在更强 wrap stress 下会先在哪里失效。**

### 3.3 wrap-aware 抽象已经开始落到代码结构

`reconstruction/unwrapping/problem.py` 已经定义了 `UnwrapProblem` / `UnwrapSolution`，说明项目不再只是做经验性脚本试验，而是准备为下一代 wrap-aware coarse reconstruction 预留稳定接口。

这一步很重要，因为它意味着：

- 后续新方法可以在同一抽象下比较
- 诊断信息、权重图和 defect phase 可以统一输出
- wrap-aware 方向开始具备可维护的代码承载点

## 4. 当前测试状态

在当前工作区直接执行：

```bash
conda run -n PINNs bash -lc 'cd /home/void0312/PINNs/mini_grin_rebuild && pytest -q tests'
```

截至 `2026-05-22` 的结果为：

- `25 passed, 6 skipped`

说明如下：

- `25 passed`：当前重写版主线测试通过
- `6 skipped`：`tests/test_compat_legacy.py` 依赖旧版 `mini_grin` 参考实现；当前工作区未放置该历史代码，因此自动跳过

这比“默认直接失败”更符合当前仓库已经切换到 `mini_grin_rebuild` 主线的事实。

## 5. 当前仍需注意的风险与未完成项

### 5.0 非理想物理泄露路线已冻结

截至 `2026-05-25`，无 `raw` 额外观测的非理想物理泄露路线已归档冻结：

- 归档：`docs/archive/physical_leakage_freeze_2026-05-25/`
- 判定：当前 `optical_leakage_lite` 默认参数更像强 domain shift / 相干纹理扰动，而不是已验证的符号泄露模型
- 原因：缺少真实工业 DIC 非理想参数或标定响应，继续深入容易把模型假设当成物理事实

该路线保留脚本和结果用于审计，但不再作为当前主推进方向。后续主线应回到 wrap-stress / coarse-prior 边界分析。

### 5.1 wrap-stress 方向仍处于大块未提交状态

目前与 wrap / prior method 相关的代码、测试、脚本和配置存在一批未提交改动，说明方向已经很明确，但还没有沉淀为稳定版本。

最值得优先收口的文件集中在：

- `src/mini_grin_rebuild/reconstruction/pseudo_poisson.py`
- `src/mini_grin_rebuild/evaluation/evaluator.py`
- `src/mini_grin_rebuild/training/trainer.py`
- `src/mini_grin_rebuild/experiments/suite.py`
- 对应的 `tests/`、`configs/`、`scripts/`

### 5.2 兼容性验证仍然依赖历史参考实现

虽然兼容性测试现在会自动跳过，但如果后续要做“新旧实现数值对齐”的正式结论，仍然需要把旧版 `mini_grin` 放回工作区根目录或 `Archive/`。

### 5.3 临时实验目录仍然存在

当前本地还存在：

- `tmp_wrap_smoke/`
- `tmp_wrap_smoke_srt/`

这类目录更像短期实验残留，后续可以视情况继续清理或转入归档。

## 6. 推荐的下一步推进顺序

基于当前进展，我建议下一步按下面顺序推进。

### 第一步：先把 wrap-stress 主线收口成一个可复述的最小结论

目标不是立刻发散到更多 realism，而是先回答：

- 当前最优骨架在什么 wrap 压力区间开始明显失效？
- `pseudo_poisson / first_order / hybrid` 三类粗先验谁更稳？
- 失效主要发生在 branch sign、edge artifact，还是 residual correction 不足？

如果这个问题不先答清，后续继续加 realism 只会让失败来源更难拆分。

### 第二步：把 wrap-stress 诊断正式并入 evaluation

建议下一轮直接把以下统计沉淀到评估输出中：

- coarse prior method 标签
- branch / sign mismatch 比例
- wrap-stress 样本级摘要
- 与 QC 指标的相关性

这样后面的实验就不只是“看图判断”，而是有结构化证据。

### 第三步：用当前 active runs 重新组织对照叙事

当前 live run 仍然聚焦：

- QC
- pre-real
- 20x OOD

建议下一步补上一组更明确的 wrap-stress 对照 run，让项目主线从：

`synthetic -> QC -> pre-real`

变成：

`synthetic baseline -> wrap-stress boundary -> QC / pre-real`

这样整个方法论会更完整。

### 第四步：在 wrap 边界问题收清楚后，再推进真实闭环

真实数据仍然重要，但更适合放在 wrap-stress 机制问题之后推进。原因是：

- 当前主骨架已经具备真实闭环所需的大部分工程接口
- 真正还没回答清楚的是方法学边界，而不是纯工程接线问题

换句话说，下一步最值钱的工作不是“更花哨的真实实验素材”，而是把当前骨架为什么有效、何时失效讲得更清楚。
