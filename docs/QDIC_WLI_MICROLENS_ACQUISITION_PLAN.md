# qDIC + 白光干涉仪微透镜联合采集实验方案

## 1. 实验定义

这份文档定义的是一个**具体的数据采集实验**，不是泛化的真实验证建议。

实验名称：

- `qDIC + WLI paired microlens acquisition pilot`

实验目标：

- 用 **qDIC / 定量微分相衬显微镜** 采集与最终应用场景一致的微透镜观测数据
- 用 **WLI / 白光干涉仪** 为同一颗微透镜提供计量级 3D 形貌参考
- 建立同一颗微透镜在 `clean / defect` 两种状态下的**跨模态配对数据**
- 为当前 `mini_grin_rebuild` 主线提供一批可用于真实验证的 pilot 数据

这次实验的核心不是“多采一些显微图”，而是构造一套能够支持下列判断的数据：

1. 当前基线在真实 qDIC 数据上是否仍然成立
2. WLI 是否能为 qDIC 提供足够可靠的形貌参考
3. 微透镜真实样本是否满足当前方法依赖的“平滑标准面 + 稀疏小缺陷”结构假设

## 2. 两台仪器各自承担的角色

### qDIC 的角色

qDIC 是**主观测模态**，地位上对应当前项目里最终希望处理的“真实输入”。

本实验中，qDIC 负责：

- 采集真实微透镜在 `clean / defect` 状态下的主输入数据
- 记录重复采集稳定性
- 提供后续 QC / gating 的真实观测对象
- 保留 raw 数据与仪器导出的中间结果

### WLI 的角色

WLI 是**计量参考模态**，不是主部署模态。

本实验中，WLI 负责：

- 给出同一颗微透镜的 3D 高度图
- 用于估计 clean 状态下的标准面
- 用于估计 defect 状态下相对 clean 标准面的残差
- 用于判断真实样本是否满足当前方法的稀疏性假设
- 为 qDIC 结果提供弱 GT / 计量参考

一句话概括：

- qDIC 负责“真实输入”
- WLI 负责“真实几何参考”

## 3. 当前代码变量与真实实验对象的映射

为了让这次采集能接进当前主线，必须先定义清楚概念映射。

| 当前主线概念 | 真实实验中的对应物 | 备注 |
| --- | --- | --- |
| `standard` | 同一颗 clean 微透镜的 qDIC / WLI 采集 | clean 状态必须重复采集 |
| `test` | 同一颗 defect 微透镜的 qDIC / WLI 采集 | 最好是同一颗 lens 在缺陷注入后的状态 |
| `reference` | 计算参考面或会话级参考件 | 不建议强行要求每颗 lens 都有物理 reference 图 |
| `defect_true` | `WLI(test) - fitted_standard_from_WLI(clean)` | 这是最重要的弱 GT 候选 |
| `diff_ts` | qDIC 的 `test - standard` 配对观测 | 当前主线最关键 |
| `diff_sr` | clean 标准面相对 reference 的差异 | 可由 WLI clean 拟合得到，或在实测中弱化处理 |

这里最重要的决定是：

- **`reference` 不必等于每颗样本都采一个物理 reference 图。**

对当前实验，更实际的做法是：

1. 对 clean WLI 形貌拟合出微透镜名义标准面
2. 将该名义标准面作为**计算参考面**
3. 真实实验中的主配对关系聚焦在 `standard(clean)` 与 `test(defect)`

这比强行追求“每颗 lens 都有物理 reference 图”更稳，也更符合你们当前主线真正依赖的结构。

## 4. 本次 pilot 的核心假设

这次具体采集实验要验证的不是所有可能的问题，而是下面 5 条。

1. qDIC 在同一颗 clean 微透镜上的重复采集是稳定的
2. WLI 在同一颗 clean 微透镜上的重复扫描足够稳定，可作为几何参考
3. 同一颗微透镜在 `clean -> defect` 变化后，WLI 残差表现为局部稀疏缺陷，而不是全域粗糙化
4. qDIC 与 WLI 可以在同一颗 lens 上完成跨模态配对与空间对齐
5. 用 WLI clean 拟合得到的标准面，足以支撑当前方法所需的“标准面 + 缺陷残差”定义

## 5. 本次实验的最小输出

实验完成后，至少要拿到下面这些产物。

### A. qDIC 侧

- clean 状态重复采集
- defect 状态重复采集
- raw 数据
- 仪器导出的 phase / gradient / processed image（若仪器软件提供）
- 会话级参考件或校准数据

### B. WLI 侧

- clean 状态 topography
- defect 状态 topography
- raw 高度图
- valid mask
- 去 tilt / 去 form 后的 residual map
- clean 标准面拟合结果

### C. 跨模态侧

- qDIC 与 WLI 的 lens identity 对应关系
- stage 坐标或视场对应关系
- registration transform
- 每颗 lens 的 `clean / defect` 配对关系

如果缺少这些核心产物中的任意一类，这次实验很难支撑后续模型验证。

## 6. 首轮 pilot 的样本设计

建议不要把第一轮做成“大规模数据采集”，而是做成“高可解释 pilot”。

### 样本组 1：clean repeat

目的：

- 测仪器稳定性
- 测跨模态对齐可行性
- 测 clean 样本是否会被算法误判成缺陷

建议数量：

- 3 颗 clean 微透镜

每颗采集：

- qDIC：5 repeats
- WLI：2 repeats

### 样本组 2：paired seeded defect

目的：

- 构造最强的因果对照
- 让 `clean -> defect` 的变化可以被明确解释

建议数量：

- 3 颗微透镜

每颗状态：

- clean 状态：qDIC 5 次，WLI 2 次
- defect 状态：qDIC 5 次，WLI 2 次

缺陷注入优先顺序：

1. 微粒污染
2. 轻微附着物
3. 可控局部划痕

第一轮不建议上大面积粗糙化或不可逆强破坏。

### 样本组 3：自然缺陷样本（可选）

目的：

- 观察真实工艺缺陷是否偏离当前假设

建议数量：

- 2 到 3 颗

这组只用于边界分析，不作为第一轮主结论依据。

## 7. 单颗 lens 的标准采集流程

对每颗样本建议固定为以下顺序。

### Phase 0: 会话级校准

- qDIC 采参考件 / 平场 / 暗场 / 仪器校准
- WLI 做当天标定与参考检查
- 记录环境条件、操作员、物镜、NA、曝光、光源

### Phase 1: clean 状态 qDIC

- 固定样本方向
- 低倍 overview 记录样本整体位置
- 切到目标视场，只保留单颗 lens
- 采 clean qDIC repeats

### Phase 2: clean 状态 WLI

- 尽量不改变样本朝向
- 记录 stage 坐标与 orientation
- 对同一颗 lens 做 clean WLI 扫描

### Phase 3: 缺陷注入

- 在显微镜外或专用工位上引入可控缺陷
- 记录缺陷类型、注入方式、是否可逆

### Phase 4: defect 状态 qDIC

- 返回同一颗 lens
- 对齐到同一视场
- 采 defect qDIC repeats

### Phase 5: defect 状态 WLI

- 对同一颗 lens 做 defect WLI 扫描

### Phase 6: 当天 quick check

- 检查 raw 文件是否完整
- 检查 clean / defect 是否混淆
- 检查 qDIC 与 WLI 是否能映射到同一 lens
- 检查 clean WLI residual 是否明显小于 defect residual

## 8. 视场与配准要求

### 视场要求

- 单次采集中只包含一颗微透镜
- aperture 边界完整
- 保留少量边缘余量
- 避免严重遮挡、过曝、截断

### 配准要求

建议为样本载台准备以下辅助信息：

- 样本方向标记
- 样本编号
- 低倍 overview 图
- stage 坐标
- 若可能，加入可见 fiducial

跨模态配准至少要做到：

- 能确定“这张 qDIC 图”和“这张 WLI 图”来自同一颗 lens
- 能将两者裁剪到同一 aperture 区域

第一轮不要求像素级完美配准，但必须做到 lens identity 级正确配对。

## 9. 当前主线下推荐的两种实验模式

### 模式 A：最小闭环模式

输入：

- qDIC clean
- qDIC defect
- WLI clean
- WLI defect

用途：

- 先验证主假设是否成立
- 不强依赖真实 `reference` 图

这是第一轮最推荐的模式。

### 模式 B：增强参考模式

额外增加：

- qDIC 参考件 / 光路参考
- WLI 参考平面 / 标准样
- clean WLI 标准面拟合与 form removal

用途：

- 更接近当前代码里的 `reference / standard / test`
- 更有利于后续把真实数据做成结构化 benchmark

如果第一轮 pilot 顺利，第二轮再上这个模式更合适。

## 10. 当天必须记录的元数据

最少应记录：

- `session_id`
- `sample_id`
- `lens_id`
- `state`：`clean` / `defect`
- `modality`：`qdic` / `wli`
- `repeat_index`
- `operator`
- `instrument_name`
- `objective`
- `na`
- `wavelength_nm`
- `pixel_pitch_um`
- `fov_um`
- `exposure_ms`
- `gain`
- `illumination`
- `stage_xyz`
- `orientation_mark`
- `single_lens_in_fov`
- `aperture_visible`
- `paired_clean_capture_id`
- `paired_wli_capture_id`
- `raw_file_paths`
- `notes`

配套模板见：

- `external_data/templates/qdic_wli_session_manifest.example.json`

## 11. 结果验收标准

第一轮不建议设过于激进的性能阈值，但必须有最低验收条件。

### 采集侧最低验收

- qDIC clean repeats 全部可读
- WLI clean / defect 高度图可读
- 同一颗 lens 的 qDIC 与 WLI 成功配对
- raw 文件、元数据、样本编号一致

### 几何侧最低验收

- clean WLI 可拟合出稳定标准面
- defect WLI 相对 clean 标准面出现局部残差
- 若 clean WLI 本身就是大面积纹理/粗糙起伏，则该样本标为 out-of-scope

### 算法侧最低验收

- qDIC clean 重建结果不应大量产生假缺陷
- qDIC defect 相比 clean 应表现出局部可解释差异
- QC 指标在 clean / defect / bad capture 之间应有分层

## 12. 这次实验真正要回答的结论

实验结束后，必须明确回答以下问题：

1. qDIC + WLI 是否能构成稳定的同 lens 配对数据集？
2. 真实微透镜样本中，“平滑标准面 + 稀疏缺陷”是否足够常见？
3. 当前主线是否可以进入真实 pilot 验证阶段？
4. 如果不能，问题出在样本分布、采集流程、跨模态配准，还是方法本身？

## 13. 建议的直接下一步

1. 确认两台仪器的固定参数与可导出原始文件格式
2. 选 3 颗 clean 微透镜作为 pilot lens set
3. 准备 orientation 标记与 overview 拍摄方案
4. 决定第一种 seeded defect 注入方式
5. 按 manifest 模板开始第一轮采集
