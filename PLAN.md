# mini_grin_rebuild — 重写执行计划（可迭代）

## A. 当前问题清单（为什么必须重写）

从当前 `mini_grin/` 的代码结构看，问题不仅是“工程组织”，更关键的是 **缺少可审计与可复现的科研软件规范**：

1. **入口分散且重复**：`mini_grin/scripts/*.py` 中存在重复的输入通道拼装、指标实现、checkpoint 结构推断等逻辑（例如 `evaluate_dataset.py` / `evaluate_metrics.py`）。
2. **路径与产物散落**：数据集、checkpoint、扫描结果在仓库根目录形成大量同级目录，难以复现实验与清理。
3. **审计链断裂**：无法从某个结果文件反推出（配置/代码版本/数据版本/随机种子/运行命令/环境），不利于论文复现实验与同行审阅。
4. **边界不清**：算法（physics/model/loss）与实验编排（argparse/print/IO）耦合，导致很难为关键模块补齐单元测试与行为契约（shape/dtype/device）。

## B. 里程碑（建议按顺序推进）

### R0：定义“研究问题 → 数据 → 指标”的验收标准

- 明确任务定义：输入是什么（强度差/原始强度/相位梯度）、输出是什么（缺陷高度/相位差）、评价指标是什么（RMSE/PSNR/SSIM/物理一致性）。
- 固化数据格式 schema（字段名、shape、dtype、单位），并给出最小样本文件示例。
- 写一页“论文式说明”：模型假设、物理前向、损失项定义（方便审计与写作）。

### R1：建立可审计的实验框架（先不关心算法细节）

- 新目录：`mini_grin_rebuild/`（已创建）
- 目标：任何一次运行都生成标准 run 目录并落盘审计信息（`config.json/meta.json/metrics.json`）。
- 这一阶段产物：配置系统 + run 管理 + 统一 CLI 骨架 + 最小单元测试。

### R2：重写数据层（schema + generator + loader + validator）

- 重写合成数据生成：从“脚本输出”变为“库 API + CLI”。
- 读写与校验：给 `.npz`/`.pt` 建立显式 schema 校验器（缺字段/shape 不对直接报错）。
- 数据版本标识：把生成配置与随机种子写入样本或 dataset-level metadata。

### R3：重写物理层（NumPy 前向 + Torch 可微版本）

- 用单元测试锁定行为契约：输入输出 shape、dtype、device、数值稳定性（避免 silent bug）。
- 所有物理参数与单位写入配置与文档（便于审计/论文撰写）。

### R4：重写训练与模型层（可插拔，便于消融）

- 把训练循环与 loss/metrics 拆开：loss 与 metrics 都是纯函数（易测、易审计）。
- 训练过程输出：每个 epoch/step 记录指标，保存 checkpoint（包含优化器与随机状态）。

### R5：重写评估与可视化（论文级输出）

- 标准化评估：同一 checkpoint 在指定 split 上输出 `metrics.json` + `plots/`。
- 论文产物导出：固定风格与命名规则，确保图表可复现。

### R6：重写实验编排（扫描/消融/对比基线）

- 扫描脚本不直接跑 python 文件串起来；改为调用库 API + 统一 CLI 参数与配置覆盖。
- 对照基线：旧实现只用于“数值/趋势对齐验证”，不进入新包代码。

## C. 下一步我建议我来做什么（马上开始动手）

先把 **R1：可审计实验框架** 做成最小闭环（这一步不涉及复制旧实现，也不涉及改算法）：

1. 在 `mini_grin_rebuild/src/mini_grin_rebuild/core/` 实现：配置加载/保存（`config.json`）、run 目录管理（生成 `meta.json`）、统一日志与随机种子工具。
2. 在 `mini_grin_rebuild/src/mini_grin_rebuild/cli/` 提供一个最小 CLI：`init-run`（只创建 run 目录并写入审计文件）。
3. 在 `mini_grin_rebuild/tests/` 补 2-3 个 `unittest` 用例，锁定 run 目录与配置快照行为。
