# 下一阶段行动建议（修正版，2026-05-15）

## 1. 先纠正问题定义

这里说的 **相位跃变**，不是表面高度或物理相位本身出现真实突变，而是 **包裹相位（wrapped phase）在模 \(2\pi\) 表示下产生的分支切换 / unwrap 问题**。

也就是说，下一阶段如果讨论“相位跃变”，应该理解为：

- 当前无偏置主线在接近或跨越包裹边界时，如何保持重建稳定；
- `pseudo_poisson` 这类基于分支选择和积分的先验，在包裹/解包裹压力增大时还能否继续提供有效结构先验；
- 网络应该如何利用这个先验，而不是把它替掉。

前一版把“相位跃变”往“相位突变 / jump surface”上带偏了，这个判断不对，这里明确纠正。

## 2. 当前消融真正说明了什么

当前项目最关键的结论，不是在“teacher 很重要”，而是在：

- **`pseudo_poisson prior` 很重要**
- **`residual` 很重要**
- **`prior as input` 很重要**
- **单独加 `teacher loss` 没有帮助，甚至明显破坏结果**

依据现有无偏置消融摘要：

- `nn_prior_residual_input`：
  - 缺陷 F1 = `0.2427`
  - 缺陷 IoU = `0.1565`
  - 全局 RMSE = `0.0100`
- `nn_prior_residual`：
  - 缺陷 F1 = `0.1892`
  - 缺陷 IoU = `0.1175`
  - 缺陷相关性 = `0.9507`
- `nn_teacher_only`：
  - 缺陷 F1 = `0.0309`
  - 缺陷 IoU = `0.0163`
  - 全局 RMSE = `0.0591`
- `nn_prior_residual_teacher`：
  - 缺陷 F1 = `0.0509`
  - 缺陷 IoU = `0.0269`
  - 全局 RMSE = `0.0614`

这组结果很清楚：

- `teacher only` 基本是坏的；
- `prior + residual` 是有效的；
- `prior + residual + teacher` 反而被打崩；
- 当前最优骨架是 **`pseudo_poisson_prior_as_input + residual learning`**。

所以，下一步绝对不应该把 `pseudo_poisson` 说成“降级”，更不应该把它从主骨架里挪走。真正该谨慎对待的是 **teacher 分支 / teacher loss**。

## 3. 当前项目做到哪一步了

截至现在，`mini_grin_rebuild/` 的主线已经不是“随便试一个 PINN”，而是已经收敛到比较明确的一条工程骨架：

- synthetic 主闭环已打通；
- `pseudo_poisson` 与 `oracle_poisson` 已作为基线固化；
- `nn_prior_residual_input` 已在缺陷指标、QC 门控、pre-real 验证里占据当前最优地位；
- `reflect padding`、`phase input`、`coord input`、edge 抑制、gate q=0.99 都已经系统扫过；
- 真实实验准备和外部 topography 接入也已铺好接口。

因此，当前阶段最值钱的不是“推翻主骨架重来”，而是：

- 识别这个主骨架的适用边界；
- 在不破坏主骨架的前提下，把它往“包裹/解包裹更难的场景”推。

## 4. 下一步应该怎么理解“相位跃变方向”

既然这里的相位跃变本质上是 **unwrap / branch selection / wrapped representation** 问题，那么下一步不应表述成：

- “构造相位突变样本”
- “让网络直接学一个 jump surface”
- “把 `pseudo_poisson` 降成普通 auxiliary”

更合理的表述应该是：

- 在保持当前 `pseudo_poisson prior + residual input` 主骨架的前提下；
- 主动构造 **更接近包裹边界、甚至发生分支切换压力** 的样本；
- 分析当前 `pseudo_poisson` 分支选择假设在这些样本上哪里开始失效；
- 再决定是否需要局部引入 wrap-aware 表示或 unwrap-aware 后处理。

换句话说，研究问题应该写成：

**“当前 `pseudo_poisson prior + residual input` 骨架，在接近或跨越包裹边界时，怎样保持分支选择、先验注入和残差学习的稳定性？”**

这和前一版“另起一条 phase-jump track”不是一回事。

## 5. 为什么这个方向仍然值得做

即使把问题定义纠正后，这个方向依然成立，而且比“继续往 `instrument_lite` 里堆 realism 参数”更像下一步主问题。

原因是：

- 当前主线已经在“连续小残差、先验有效”的 regime 上跑得比较顺；
- `pseudo_poisson` 本身就依赖分支选择和积分假设；
- 一旦样本更接近包裹边界，问题会首先暴露在 **branch / unwrap / prior mismatch** 上，而不一定先暴露在 blur、illumination、camera response 上；
- 这类失败模式更接近方法学边界，而不是单纯 domain gap。

所以重点不是“抛开当前骨架去做一个新表示”，而是：

- 先把当前骨架推到更难的 wrap regime；
- 看它是哪里先坏；
- 再对坏掉的那一段做定向修补。

## 6. 当前骨架里哪些东西该保，哪些东西该动

### 6.1 必须保住的

- `pseudo_poisson` 先验主地位
- `prior as input`
- residual 学习框架
- 当前 QC / gating 指标体系
- 现有 `nn_prior_residual_input` 作为主对照

这些不是负担，而是当前项目最核心的资产。

### 6.2 明确不能再误判的

- `teacher loss` 不是下一步主抓手
- 不能把 `teacher` 的失败误读成 `pseudo_poisson prior` 的失败
- 不能把 `pseudo_poisson` 基线门控差，误读成“先验注入没意义”

因为实验已经说明：

- `pseudo_poisson` 单独拿出来做最终解，门控很差；
- 但把 `pseudo_poisson` 注入到神经网络作为 prior input，效果反而最好；
- 这说明问题不在“先验无用”，而在“先验单独作为终解不够，用作结构引导最有用”。

### 6.3 下一步真正该动的

- 数据生成中的 wrap regime
- `pseudo_poisson` 的 branch failure 分析
- residual 分支在 wrap 边界附近的表达能力
- 评估指标中对 wrap stress 的专项统计

## 7. 推荐的下一步技术路线

### 7.1 第一优先：先做 wrap-stress benchmark，不要先改网络主头

当前第一步不建议上来就改 `UNetPP` 输出头，也不建议一开始把 target 改成全新的 `cos/sin` 周期表示。

更稳的顺序应该是：

1. 保持当前 `nn_prior_residual_input` 骨架不变；
2. 在 synthetic 数据中构造更强的 wrap-stress 样本；
3. 测当前骨架在哪类样本上先崩；
4. 再决定要不要改表示或后处理。

因为在现阶段，最值钱的问题不是“有没有更花哨的网络头”，而是：

**当前最好用的骨架，到底是在哪个 wrap 区间开始坏。**

### 7.2 第二优先：把数据从“小心避开包裹问题”改成“系统扫描包裹压力”

当前 `SimulationConfig` 里已经有：

- `wrap_safety`
- `standard_residual_wrap_frac`
- `defect_amplitude_wrap_min`
- `defect_amplitude_wrap_max`

这些参数已经足够搭一个第一版 wrap-stress benchmark。

建议下一步不是马上造新 defect 类型，而是先扫下面这些量：

- `standard_residual_wrap_frac`
  - 例如从 `0.3 -> 0.5 -> 0.7 -> 0.9`
- `defect_amplitude_wrap_max`
  - 例如从 `1.0 -> 1.2 -> 1.5`
- `defect_amplitude_wrap_min`
  - 适当拉高，减少“过于简单的小缺陷”
- `defect_center_sigma_norm` / `defect_center_max_radius_norm`
  - 控制缺陷更靠近主曲率更强的区域

目标不是生成“相位突变表面”，而是让现有分支选择假设更频繁地遇到困难样本。

### 7.3 第三优先：做 branch-failure 分析，而不是先发明新 loss

当前 `reconstruction/pseudo_poisson.py` 的关键假设是：

- `sign(∂phi_test) == sign(∂phi_standard)` almost everywhere

下一步最应该补的是 **失效分析**：

- 哪些样本上这个假设先失效；
- 失效集中在透镜中心、边缘、还是缺陷边界；
- 失效后网络 residual 能纠正多少；
- 失效和 QC 指标之间有没有稳定对应关系。

建议直接补分析输出：

- branch sign mismatch ratio
- mismatch 区域面积占比
- mismatch 到 defect ROI 的距离统计
- mismatch 与 `physics_rmse` / `outside_p95_abs` 的相关性

如果这些统计一旦建立起来，后面再改模型才有方向。

### 7.4 第四优先：如果确实要改表示，先改 residual 支路，不要推翻 prior 主骨架

只有在上一步确认：

- 当前 residual 分支在 wrap-stress 场景下确实表达不够；
- 且失败不是单纯 branch sign 错误传播；

这时才建议考虑更 wrap-aware 的 residual 表示。

即便要改，也建议遵守：

- `pseudo_poisson` 仍然保留为主 prior；
- 新表示只作用在 residual correction 支路；
- 不要把整个主预测目标改成完全脱离当前先验的另一套体系。

这时可以考虑的方向包括：

- residual 分支预测局部 wrap correction
- residual 分支预测 branch reliability / uncertainty
- residual 分支预测 unwrap-sensitive region mask

但这些都应是 **增强现有骨架**，不是替代现有骨架。

## 8. 更真实仿真该放在什么位置

更真实仿真当然仍然有价值，但它应该放在这个顺序之后：

1. 先确认 wrap-stress 下的主失败模式；
2. 先确认 `pseudo_poisson prior + residual input` 在包裹压力下怎么坏；
3. 再把 `instrument_lite` 的 realism 加进来，看这些失败是否被进一步放大。

否则如果现在直接同时上：

- 更高 wrap stress
- 更复杂 illumination
- 更强 blur / response mismatch
- 更复杂 geometry drift

最后会很难判断崩坏到底来自哪一层。

所以，更真实仿真不该取消，但更适合当作第二阶段 stress test，而不是第一步主任务。

## 9. 具体建议改哪些模块

### 9.1 `data/generate_dataset.py` 与 `data/virtual_objects.py`

这是第一优先级。

建议先做：

- wrap-stress 配置扫描
- 记录每个样本的 wrap 难度元数据
- 输出可用于 branch failure 分析的中间量

建议新增到 `sample_meta` / `dataset_meta` 的字段：

- `standard_wrap_frac`
- `defect_wrap_target`
- `estimated_wrap_stress_level`
- 可选 `phase_grad_peak_standard`
- 可选 `phase_grad_peak_test`

### 9.2 `reconstruction/pseudo_poisson.py`

这里第一步不要急着改算法主体，先加分析钩子。

建议补：

- oracle sign 和 pseudo sign 的差异统计
- branch mismatch heatmap
- mismatch 区域 summary

这样可以直接量化：

- 先验什么时候仍然可靠；
- 什么时候只是“有误差但仍可作为 input prior”；
- 什么时候已经完全误导 residual 分支。

### 9.3 `evaluation/evaluator.py` / `evaluation/metrics.py`

新增 wrap-stress 视角下的分析指标：

- 按 wrap 难度分 bucket 的 F1 / IoU / RMSE
- branch mismatch rate
- prior error vs final error 的对比
- QC 指标与 wrap 难度的关系

当前指标体系不用推翻，只要补充分层分析即可。

### 9.4 `training/trainer.py`

短期内不建议大改主训练逻辑。

首先做的应该是：

- 在训练/评估日志里记录 prior 与 residual 的相对贡献
- 在 wrap-stress 数据上比较：
  - `nn_prior_residual_input`
  - `nn_prior_residual`
  - `nn_reflect_phase_residual020`

也就是说，先做结构对比，而不是先写新损失。

### 9.5 `models/unetpp.py`

这里目前不是第一刀。

除非 wrap-stress benchmark 明确证明：

- 当前 residual 输出在表示能力上不够；

否则不建议现在就改成新多头主结构。

## 10. 我建议立刻执行的动作

如果下一步真的要开工，我建议按下面顺序来：

1. 保留 `nn_prior_residual_input` 为当前主 baseline
2. 新建一个 `wrap_stress_v1` 数据/实验配置组
3. 只扫现有 wrap 相关参数，不先改网络结构
4. 给 `pseudo_poisson` 增加 branch-failure 分析输出
5. 用现有 evaluator 做分 bucket 统计
6. 看清楚是先验坏了、还是 residual 不够、还是 teacher 干扰

这一轮回答的核心问题应该是：

**当前最优骨架在包裹压力上限附近是如何失效的。**

这一步回答清楚之后，才值得决定要不要加：

- unwrap-aware residual head
- branch reliability head
- wrap-aware loss

## 11. 总结

下一步仍然可以把“相位跃变”当重点，但必须把它准确地理解成 **包裹/解包裹与分支选择问题**，而不是“相位突变建模问题”。

在这个前提下，当前项目最该保住的是：

- `pseudo_poisson prior`
- `prior as input`
- residual learning 主骨架

最不该误用的是：

- teacher loss

因此，下一阶段最合理的策略不是另起炉灶，而是：

**以 `pseudo_poisson prior + residual input` 为主骨架，系统推进 wrap-stress / unwrap-stress benchmark，先看它在包裹边界附近怎么坏，再决定需要补哪一层。**
