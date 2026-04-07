# mini_grin_rebuild

该目录用于把当前仓库中较为“研究脚本化”的 `mini_grin/` **彻底重写**为更易审计、可复现、可持续产出的科研工程。

从当前阶段开始，`mini_grin_rebuild/` 视为 **唯一后续研究基线**；仓库根目录下的其它历史实现、备份与旧产物统一封存到 `Archive/`。

- 现状代码入口：`mini_grin/scripts/*.py`（例如 `python -m mini_grin.scripts.train_demo`）
- 重写目标与路线：见 `mini_grin_rebuild/ARCHITECTURE.md` 与 `mini_grin_rebuild/PLAN.md`

## 目录（重建后的目标形态）

- 代码：`mini_grin_rebuild/src/mini_grin_rebuild/`
- 配置：`mini_grin_rebuild/configs/`
- 脚本（薄封装/兼容层）：`mini_grin_rebuild/scripts/`
- 文档：`mini_grin_rebuild/docs/`
- 测试：`mini_grin_rebuild/tests/`

## 原则

- **不复制粘贴旧实现**：旧代码仅作为对照基线，用于验证新实现的数值/趋势是否合理。
- **先审计闭环，再写算法**：任何一次运行都必须落盘 `config/meta/metrics`，方便追溯与复现。

## 测试

推荐在 `PINNs` conda 环境中执行：

```bash
conda run -n PINNs bash -lc 'cd /home/void0312/PINNs/mini_grin_rebuild && pytest -q tests'
```

当前仓库下该命令已验证通过，结果为 `17 passed`。

- 兼容性测试会自动从工作区根目录与 `Archive/` 中解析旧版 `mini_grin` 参考实现。
- 如果只想跑重写版核心测试，也可以在 `tests/` 下按文件筛选。

## Run 管理

当前 run 管理采用“活跃保留、探索归档”的分层方式：

- 规则与总览：`mini_grin_rebuild/docs/RUNS.md`
- 关键 run 快捷入口：`mini_grin_rebuild/run_refs/`
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

## 低分辨率 smoke test 的注意事项

`virtual_objects.py` 中的缺陷宽度/长度参数是以“物理单位”（与 `dx` 同单位，默认 µm）定义的。
如果仅把 `grid_size` 从 512 降到 64 但保持 `dx=0.39`，等价于把视场 (FOV) 缩小 8 倍，
会导致 scratch 相对视场过长、标准相位梯度过大、`data_weight_map` 几乎全域 clamp，从而更容易出现伪影/塌缩。

- 建议使用 `mini_grin_rebuild/configs/benchmark_smoke64_fov200.json`：通过增大 `dx` 保持约 200µm 的 FOV，使 smoke test 更贴近原物理尺度。

## 门控 / QC（可发表/可落地）

真实数据没有 ground-truth，因此需要一套 **GT-free** 的门控机制来衡量重建质量、抑制边缘伪影并监控域偏移。

- 文档：`mini_grin_rebuild/docs/GATING.md`
- 示例配置（含 q=0.99 门控阈值）：`mini_grin_rebuild/configs/benchmark_microlens200_srt_gated_q99.json`

## 外部数据

当前主训练集仍然是项目内的 synthetic `microlens_srt` 数据；外部数据的作用是补充真实形貌、做 `height -> SPDIC` 物理链验证，以及后续 `pre-real / hybrid` 测试。

- 目录说明：`mini_grin_rebuild/external_data/README.md`
- 管理脚本：`mini_grin_rebuild/scripts/manage_external_data.py`
- 真实实验方案：`mini_grin_rebuild/docs/REAL_DATA_EXPERIMENT_PLAN.md`
- qDIC+WLI 采集实验：`mini_grin_rebuild/docs/QDIC_WLI_MICROLENS_ACQUISITION_PLAN.md`
- qDIC-only 实验报告草稿：`mini_grin_rebuild/docs/QDIC_MICROLENS_EXPERIMENT_REPORT.md`
- 研究汇报模板：`mini_grin_rebuild/docs/templates/research_reports/EXPERIMENT_REPORT_TEMPLATE.tex`
- 标准实验报告模板：`mini_grin_rebuild/docs/templates/lab_reports/QDIC_WLI_STANDARD_EXPERIMENT_REPORT_TEMPLATE.tex`

推荐顺序：

- `zenodo_10365872`：先打通导入与扫描
- `zenodo_18014400`：再补真实纹理先验

示例：

```bash
cd /home/void0312/PINNs/mini_grin_rebuild
python scripts/manage_external_data.py download --dataset zenodo_10365872
python scripts/manage_external_data.py extract --dataset zenodo_10365872
python scripts/manage_external_data.py scan --all
python scripts/external_topography_smoke.py external_data/raw/zenodo_10365872_subset/Fig3_CSI_50x_AM.plux
```

上面的 smoke 脚本会把外部 `.plux` topography 样本转换成项目当前可用的 `height -> phase -> I_x/I_y` 结果，并在 `external_data/manifests/` 记录统计摘要。
