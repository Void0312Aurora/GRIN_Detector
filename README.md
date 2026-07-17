# mini_grin_rebuild

本仓库用于把原先较为“研究脚本化”的 `mini_grin/` **彻底重写**为更易审计、可复现、可持续产出的科研工程。

从当前阶段开始，本仓库视为 **唯一后续研究基线**；历史实现只作为外部兼容性参考，不属于当前发行内容。

- 当前代码入口：`src/mini_grin_rebuild/`
- 当前 CLI 入口：`python -m mini_grin_rebuild.cli.main` 或安装后的 `mini-grin`
- 重写目标与路线：见 `ARCHITECTURE.md` 与 `PLAN.md`
- 最新状态梳理：`docs/PROJECT_STATUS_2026-05-22.md`
- 非项目资料封存说明：`docs/NON_PROJECT_ARCHIVE.md`

## 目录

- 代码：`src/mini_grin_rebuild/`
- 配置：`configs/`
- 脚本（薄封装/兼容层）：`scripts/`
- 文档：`docs/`
- 测试：`tests/`

## 原则

- **不复制粘贴旧实现**：旧代码仅作为对照基线，用于验证新实现的数值/趋势是否合理。
- **先审计闭环，再写算法**：任何一次运行都必须落盘 `config/meta/metrics`，方便追溯与复现。

## 测试

在已经安装项目依赖与 `pytest` 的环境中执行：

```bash
python -m pip install -e .
python -m pytest -q tests
```

截至 `2026-07-17`，干净仓库快照已验证结果为 `36 passed, 7 skipped`。

- `tests/test_compat_legacy.py` 只有在工作区根目录或 `Archive/` 中能找到旧版 `mini_grin` 参考实现时才会执行；否则会自动跳过。
- `tests/test_external_topography.py` 中依赖本地真实 `.plux` 样本的测试，在样本未随仓库分发时会跳过；其余合成 `.plux` 测试仍会执行。
- 如果只想跑重写版核心测试，也可以在 `tests/` 下按文件筛选。

## Run 管理

当前 run 管理采用“活跃保留、探索归档”的分层方式：

- 规则与总览：`docs/RUNS.md`
- 关键 run 快捷入口：`run_refs/`
- 刷新报告与快捷引用：

```bash
python scripts/manage_runs.py refresh
```

- 将 `archive_candidate` 物理迁移到 `runs_archive/`：

```bash
python scripts/manage_runs.py archive
```

默认只有少量 `active` run 会保留在 `runs/` 顶层。
`reference` 与 `archive_candidate` 都会迁移到 `runs_archive/<kind>/`，并通过 `run_refs/` 暴露快捷入口。
这样新的测试和实验只面对精简后的 `runs/` 顶层目录。

截至 `2026-05-22`，`docs/RUNS.md` 记录的 run 概况为：

- `104` 个 run，总体量约 `29.7 GiB`
- 顶层 `runs/` 只保留 `3` 个 active run
- 历史探索 run 已分流到 `runs_archive/`

## 低分辨率 smoke test 的注意事项

`virtual_objects.py` 中的缺陷宽度/长度参数是以“物理单位”（与 `dx` 同单位，默认 µm）定义的。
如果仅把 `grid_size` 从 512 降到 64 但保持 `dx=0.39`，等价于把视场 (FOV) 缩小 8 倍，
会导致 scratch 相对视场过长、标准相位梯度过大、`data_weight_map` 几乎全域 clamp，从而更容易出现伪影/塌缩。

- 建议使用 `configs/benchmark_smoke64_fov200.json`：通过增大 `dx` 保持约 200µm 的 FOV，使 smoke test 更贴近原物理尺度。

## 门控 / QC（可发表/可落地）

真实数据没有 ground-truth，因此需要一套 **GT-free** 的门控机制来衡量重建质量、抑制边缘伪影并监控域偏移。

- 文档：`docs/GATING.md`
- 示例配置（含 q=0.99 门控阈值）：`configs/benchmark_microlens200_srt_gated_q99.json`

## 当前主线关注点

当前最新一轮工作重心已经从“基础重写”推进到 **wrap-stress / coarse-prior** 分析，主要包括：

- `pseudo_poisson / first_order_poisson / first_order_sign_quadratic_poisson` 三类粗先验切换
- 面向包裹压力场景的 `wrap_stress_mixed_v1.json`、`train_phase_jump_suite.py`、`sweep_phase_jump_scale.py`
- `reconstruction/unwrapping/` 下的 wrap-aware 问题抽象

建议先读 `docs/PROJECT_STATUS_2026-05-22.md`，再进入对应脚本和配置。

## 外部数据

当前主训练集仍然是项目内的 synthetic `microlens_srt` 数据；外部数据的作用是补充真实形貌、做 `height -> SPDIC` 物理链验证，以及后续 `pre-real / hybrid` 测试。

- 目录说明：`external_data/README.md`
- 管理脚本：`scripts/manage_external_data.py`
- 真实实验方案：`docs/REAL_DATA_EXPERIMENT_PLAN.md`
- qDIC+WLI 采集实验：`docs/QDIC_WLI_MICROLENS_ACQUISITION_PLAN.md`
- qDIC-only 实验报告草稿：`docs/QDIC_MICROLENS_EXPERIMENT_REPORT.md`
- 研究汇报模板：`docs/templates/research_reports/EXPERIMENT_REPORT_TEMPLATE.tex`
- 标准实验报告模板：`docs/templates/lab_reports/QDIC_WLI_STANDARD_EXPERIMENT_REPORT_TEMPLATE.tex`

推荐顺序：

- `zenodo_10365872`：先打通导入与扫描
- `zenodo_18014400`：再补真实纹理先验

示例：

在仓库根目录执行：

```bash
python scripts/manage_external_data.py download --dataset zenodo_10365872
python scripts/manage_external_data.py extract --dataset zenodo_10365872
python scripts/manage_external_data.py scan --all
python scripts/external_topography_smoke.py external_data/raw/zenodo_10365872_subset/Fig3_CSI_50x_AM.plux
```

上面的 smoke 脚本会把外部 `.plux` topography 样本转换成项目当前可用的 `height -> phase -> I_x/I_y` 结果，并在 `external_data/manifests/` 记录统计摘要。
