# 非项目资料封存说明

## 1. 本次封存的目的

仓库工作区内长期混放了一批**不直接服务当前研发主线**的本地资料，主要包括：

- 中期检查报告与附件
- 答辩汇报 PPT / PDF / 预览图
- 申报书与相关压缩包
- 面向答辩页的局部重绘素材与导出图

这些内容虽然和课题有关，但不属于当前日常开发、实验编排和方法推进的主入口。

为了让项目入口更聚焦，本次将它们从 `docs/` 的日常可见路径中移开，统一转入本地封存区。

## 2. 封存位置

本地封存目录：`local_archive/non_project_reports/`。

封存时间：

- `2026-05-22`

## 3. 当前的边界

### 已封存的内容

以下内容应视为“非项目主线资料”：

- 原 `docs/reports/` 下的本地报告、附件、PPT、申报书、预览图与中间导出物

### 仍保留在项目主线中的内容

以下内容仍保留在 `docs/` 主线中，因为它们直接服务后续研究推进：

- `docs/GATING.md`
- `docs/RUNS.md`
- `docs/REAL_DATA_EXPERIMENT_PLAN.md`
- `docs/QDIC_WLI_MICROLENS_ACQUISITION_PLAN.md`
- `docs/QDIC_MICROLENS_EXPERIMENT_REPORT.md`
- `docs/NEXT_STEP_PHASE_JUMP_PLAN.md`
- `docs/templates/`

其中 `docs/templates/` 虽然与汇报/实验写作有关，但属于**可复用模板资产**，不是一次性的答辩材料，因此保留。

## 4. 使用建议

- 日常查看项目进展时，优先从 `README.md`、`docs/PROJECT_STATUS_2026-05-22.md`、`docs/GATING.md`、`docs/RUNS.md` 进入。
- 只有在需要追溯中期检查、答辩汇报或申报材料时，再进入 `local_archive/non_project_reports/`。
- 如果后续再次生成答辩材料，本地脚本会继续写入 `local_archive/non_project_reports/`；该目录只作为本地封存区，不进入远端主线。
