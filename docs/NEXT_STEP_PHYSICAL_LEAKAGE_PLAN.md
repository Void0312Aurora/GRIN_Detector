# 非理想物理泄露取符号分支计划

## 冻结状态：2026-05-25

本分支当前已冻结并归档。原因是我们尚不能确认实际工业 DIC 采集链中的非理想物理参数；在无 `raw` 额外观测的默认情形下，当前 `optical_leakage_lite` 仿真更像强 domain shift / 相干纹理扰动，而不是已验证的可用符号泄露模型。

归档位置：

- [docs/archive/physical_leakage_freeze_2026-05-25/README.md](/home/void0312/PINNs/mini_grin_rebuild/docs/archive/physical_leakage_freeze_2026-05-25/README.md)

保留代码和脚本用于复现与审计，但不再作为当前主推进方向。若未来重新开启，需要先获得真实仪器参数、标定响应，或明确引入额外采集通道的工业可行性。

## 0. 当前路线裁剪：封存额外 raw 观测假设

经过重新确认，标准工业 DIC 采集默认只提供 `I_x` 和 `I_y`，并不自动提供同一样品下的直通强度 `I_raw`。因此，依赖 `I_raw(test) - I_raw(standard)` 的符号判定分支需要额外采集模式、额外光路或额外曝光；它不能作为默认 DIC 方案。

本阶段将 `raw` 路线封存为**额外观测假设**：

- 可以作为“若设备已有直通强度通道时”的扩展路线
- 不用于当前默认工业 DIC 消融
- 当前默认路线只允许 `I_x / I_y`
- 非理想物理泄露必须从 `I_x / I_y` 本身被网络学习

对应的新实验配置是：

- [configs/ideal_dic_smoke64.json](/home/void0312/PINNs/mini_grin_rebuild/configs/ideal_dic_smoke64.json)
- [configs/nonideal_dic_no_raw_smoke64.json](/home/void0312/PINNs/mini_grin_rebuild/configs/nonideal_dic_no_raw_smoke64.json)
- [configs/suites/no_raw_dic_leakage_smoke.json](/home/void0312/PINNs/mini_grin_rebuild/configs/suites/no_raw_dic_leakage_smoke.json)

## 1. 这条分支要回答什么问题

当前主线里，`pseudo_poisson prior + residual input` 的核心困难来自：

- 理想无偏置 DIC / SPDIC 观测本质上更接近平方梯度响应
- 当小扰动假设不再稳、或进入更强 wrap / branch stress 区间时，符号恢复会变困难

`docs/Temp/temp-01.md` 提出的这条分支，不是继续在 “signless + unwrap” 框架里硬扛，而是换一个问题表述：

**真实成像链并不等于理想局部平方梯度算子。有限 NA、离焦、像差、PSF、响应非线性等非理想传播，会把缺陷相位扰动转成一阶强度残差，从而在 raw / 非理想通道中泄露符号信息。**

因此，这条分支的目标不是“再找一个更强的先验”，而是：

**验证并利用真实物理泄露中的符号信息，减少对“小扰动线性项必须成立”这件事的依赖。**

## 2. 当前仓库里已经有什么基础

这条分支并不是从零开始，当前仓库已经有不少可复用基础。

### 2.1 已有离线仿真框架

当前离线数据生成已经支持两个 capture engine：

- `ideal_gradient`
- `instrument_lite`

对应位置：

- [src/mini_grin_rebuild/simulation/engines/ideal_gradient.py](/home/void0312/PINNs/mini_grin_rebuild/src/mini_grin_rebuild/simulation/engines/ideal_gradient.py)
- [src/mini_grin_rebuild/simulation/engines/instrument_lite.py](/home/void0312/PINNs/mini_grin_rebuild/src/mini_grin_rebuild/simulation/engines/instrument_lite.py)

其中 `instrument_lite` 已经具备有序的非理想变换链：

- `response`
- `optics`
- `illumination`
- `geometry`
- `camera`

这些模块已经能模拟：

- 增益 / 偏置 / 饱和 / channel cross-talk
- PSF blur / edge falloff / outside leakage
- illumination field / bias
- capture-level 与 channel-level shift / rotation / scale
- shot noise / read noise / quantization / bad pixels

### 2.2 已有 raw intensity 输入通道

当前数据集和训练输入已经支持把三帧原始观测作为输入：

- `intensity_standard`
- `intensity_reference`
- `intensity_test`

对应位置：

- [src/mini_grin_rebuild/data/datasets.py](/home/void0312/PINNs/mini_grin_rebuild/src/mini_grin_rebuild/data/datasets.py)
- [src/mini_grin_rebuild/training/inputs.py](/home/void0312/PINNs/mini_grin_rebuild/src/mini_grin_rebuild/training/inputs.py)

这意味着从“数据接口”和“模型输入接口”角度，这条分支已经有进入点。

### 2.3 已有非理想 synthetic 配置入口

已经有一份相对接近这条路线的配置：

- [configs/benchmark_microlens200_srt_instrument_lite.json](/home/void0312/PINNs/mini_grin_rebuild/configs/benchmark_microlens200_srt_instrument_lite.json)

说明这条路线至少已经有了一个初始实验壳子。

## 3. 当前真正的缺口

虽然基础设施不少，但这条分支目前还没有真正“打中核心物理命题”。最关键的缺口有三个。

### 3.1 当前 `instrument_lite` 还不是 `temp-01.md` 里的泄露模型

`temp-01.md` 的核心理论是：

- 真实复场 `U_h = H_theta[t_h]`
- 原始强度残差 `Delta I_raw` 中出现关于缺陷 `d` 的一阶干涉项
- 这个一阶项对 `d -> -d` 反号，因此能泄露符号

但当前 `instrument_lite` 的实现方式是：

- 先按理想模型生成 `I_x, I_y`
- 再对这两个强度图施加 response / blur / illumination / geometry / camera 变换

也就是说，当前的非理想项大多是**强度域后处理**，并不是：

- 从相位透射函数出发
- 经过复场传播 / 孔径 / 离焦 / 像差
- 再形成 raw / DIC 观测

所以当前 `instrument_lite` 更像“domain mismatch engine”，还不是“物理泄露 engine”。

### 3.2 当前 raw 输入并不等于“能提供符号的一阶 raw 通道”

现在的 `raw intensity inputs` 本质上还是 `I_x / I_y` 三帧原始观测拼接，而不是：

- 真正的 `I_raw`
- 或 `Delta I_raw = I_raw(test) - I_raw(standard)`

所以现在即使设置了 `use_raw_intensity_inputs=true`，也不能直接说明模型已经在利用 `temp-01.md` 说的物理泄露。

### 3.3 当前可微前向模型仍是理想平方梯度层

当前训练和 QC 依赖的 differentiable forward model 仍然只有：

- [src/mini_grin_rebuild/physics/layer.py](/home/void0312/PINNs/mini_grin_rebuild/src/mini_grin_rebuild/physics/layer.py)
- [src/mini_grin_rebuild/physics/factory.py](/home/void0312/PINNs/mini_grin_rebuild/src/mini_grin_rebuild/physics/factory.py)

它本质上仍然是：

- phase -> gradient -> squared intensity

所以即使离线数据以后加入了物理泄露，训练时的 physics consistency 仍然会把模型往“理想 signless 前向”上拽，这会限制这条分支真正发挥作用。

## 4. 对当前分支的工作判断

基于 `temp-01.md` 和现有代码，我认为这条分支当前应该被定义为：

**“从 ideal signless inverse 问题，转向 weakly sign-revealing observation 问题。”**

它和当前的 wrap / unwrap 分支不是互斥关系，而是两条不同层面的路线：

- `wrap / unwrap` 分支：默认观测仍然 signless，重点做 branch selection / wrapped representation / coarse prior 稳定性
- `physical leakage` 分支：认为真实观测本身含有弱符号信息，重点做前向建模、反事实验证、泄露强度定量

这两条路线最终甚至可以汇合：

- 先用 physical leakage 缓解 sign ambiguity
- 再用 wrap-aware prior 处理仍然存在的 branch / integration 问题

## 5. 推荐的阶段计划

这条分支不建议一开始就改大训练闭环。更稳的推进顺序是“先证明信息存在，再证明模型能用，再决定是否升级前向层”。

### Phase A: 先做“符号泄露存在性”验证

目标：

- 不先问网络是否变强
- 先验证 `d` 与 `-d` 在非理想传播下是否真的可区分

#### A1. 增加一个最小泄露仿真原型

建议新增一个新 engine，而不是继续把所有东西塞进 `instrument_lite`：

- 暂名：`optical_leakage_lite`

第一版只需支持最小物理链：

1. `h = h_s + d`
2. `t_h = exp(i kappa h)`
3. 经过标量频域传播 `H_theta`
4. 生成：
   - `I_raw = |U_h|^2`
   - 可选 `I_x, I_y`（由 propagated field 再做剪切差分）

第一版参数只需要少量可控项：

- effective defocus `z_eff`
- aperture / NA proxy
- PSF / blur
- optional aberration term

#### A2. 做 `d` vs `-d` 反事实实验

围绕同一个标准面 `h_s`，对同一个 defect shape 生成：

- `d_plus = d`
- `d_minus = -d`

然后比较：

- ideal gradient engine 下的差异
- optical leakage engine 下的差异

建议新增的核心统计：

- `lambda_raw`
- `lambda_dic`
- `raw_sign_separation_score`
- 按 defect size / radial position / wrap stress 分组的符号可分性

验收标准：

- 在至少一部分样本与参数区间里，`I_raw(h_s + d)` 和 `I_raw(h_s - d)` 存在稳定非零分离
- 且分离强度随 defocus / aperture / blur 呈系统变化，而不是纯噪声抖动

### Phase B: 把“泄露信息”正式接到数据与评估层

目标：

- 先不急着改训练 loss
- 先让 dataset / evaluator 能看见这类信息

#### B1. 扩展样本 schema

建议在 `.npz` 样本中新增显式 raw 通道：

- `raw_standard`
- `raw_reference`
- `raw_test`
- 可选 `raw_diff_ts`

并在 `sample_meta.json` / `dataset_meta.json` 中记录：

- leakage engine name
- sampled physical leakage params
- per-sample leakage summary

建议新增 meta 字段：

- `leakage_strength_raw`
- `leakage_strength_dic`
- `defocus_level`
- `aperture_proxy`
- `aberration_level`

#### B2. 增加“符号泄露专用评估”

在 evaluator 或独立脚本里新增一套不是 RMSE/F1 的诊断输出：

- `sign_flip_separability`
- `raw_diff_corr_with_defect_sign`
- center / edge / low-gradient 区域分组统计
- 与 `wrap_class`、`estimated_wrap_stress_level` 的相关性

这一步的重点是回答：

**哪些区域、哪些样本、哪些非理想参数最容易提供可用符号信息。**

### Phase C: 小规模验证“网络是否真的在用泄露信息”

目标：

- 只做最小训练对照
- 不急着上完整新 physics loss

#### C1. 做三组最小对照

建议先控制到最小矩阵：

1. ideal signless baseline
2. leakage engine + 仍只喂 DIC 通道
3. leakage engine + 喂 DIC + raw 通道

比较重点不是全量指标，而是：

- 中心低梯度缺陷
- `d` / `-d` 易混区域
- 当前 pseudo-poisson 容易退化的样本

如果第三组明显优于前两组，才说明“泄露信息不仅存在，而且模型能用”。

#### C2. 暂不急着替换 pseudo-poisson 主骨架

这条分支当前不应一上来推翻：

- `pseudo_poisson prior`
- `prior as input`
- residual learning 骨架

更稳的做法是：

- 保持现有骨架
- 仅增加 leakage-aware observation
- 看它是否能减少 coarse prior 的 sign failure

### Phase D: 再考虑升级 differentiable forward model

只有当 A/B/C 都给出正面证据后，才值得进入这一步。

这时再考虑把当前：

- `DifferentiableGradientLayer`

扩成：

- `DifferentiableOpticalLeakageLayer`

并增加：

- raw consistency loss
- leakage-linearized loss
- calibrated forward/QC model

否则太早改训练前向，工程成本很高，也难判断收益是否来自真实符号泄露。

## 6. 当前最值得做的具体任务清单

如果按“先做分支研究、不急着大改训练”的原则，我建议下一步的具体工作内容是下面这 6 项。

### 任务 1. 写一页分支问题定义

把 `temp-01.md` 里的结论压成更项目化的版本，明确：

- 这条分支研究的问题
- 和 wrap/unwrap 分支的关系
- 当前最小验收命题

这份文档就是本文件。

### 任务 2. 新增 `optical_leakage_lite` 仿真 engine

建议放在：

- `src/mini_grin_rebuild/simulation/engines/optical_leakage_lite.py`

第一版只要能输出：

- `I_raw`
- `I_x`
- `I_y`

并接受少量物理参数即可。

### 任务 3. 新增 `sign_flip_counterfactual` 脚本

建议新增脚本：

- `scripts/sign_flip_counterfactual.py`

输入：

- config
- defect family / sample count
- leakage parameter sweep

输出：

- `d` vs `-d` 的分离统计
- 热图 / 报表 / JSON summary

### 任务 4. 扩展 dataset schema 支持 raw 通道

这一步会改：

- `data/generate_dataset.py`
- `data/datasets.py`
- `training/inputs.py`

目标是显式支持真正的 raw 通道，而不是默认把现有 `I_x/I_y` 三帧观测当成 raw 替代物。

### 任务 5. 增加 leakage 专用评估指标

建议优先做“只读诊断”，不先改训练：

- `evaluator.py` 或独立 report 脚本
- 输出 sample-level leakage metrics

### 任务 6. 做最小训练对照

只有前面 1-5 跑通后，再做：

- ideal vs leakage
- no-raw vs raw

的小规模训练对照。

## 7. 验收标准

这条分支是否值得继续，建议按以下门槛判断。

### 通过门槛

- 能在 synthetic 反事实实验中稳定测到 `d` / `-d` 的 raw 分离
- 分离强度对物理参数有可解释趋势
- 加入 raw 后，至少在一类当前困难样本上有稳定收益

### 暂缓门槛

如果出现以下情况，应暂缓大规模推进：

- 分离只在极端参数下出现
- 分离高度依赖噪声/偏置等不稳因素
- 网络增益只出现在 trivial 样本，而对当前真正困难样本无帮助

## 8. 当前建议的优先顺序

如果只选一个近期主任务，我建议：

**先做 Phase A 的 `d` / `-d` 反事实泄露验证。**

原因很简单：

- 这是这条分支最核心、最省成本、最能决定方向价值的一步
- 如果这一步都站不住，后面没必要急着改 dataset、training、forward model
- 如果这一步站住了，后面每一步都有明确物理依据

## 9. 一句话结论

这条分支当前最合理的定位不是“继续调 signless 主线的小变体”，而是：

**系统验证并利用真实非理想传播中的弱符号泄露，把问题从纯 signless inverse 推向 weakly sign-revealing observation。**

在当前仓库里，最适合立即开展的工作不是先改大模型，而是：

**新增最小物理泄露 engine，做 `d` vs `-d` 反事实实验，先证明这条信息源是否真实存在且足够稳定。**
