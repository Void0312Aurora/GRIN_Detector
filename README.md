# mini_grin_rebuild

该目录用于把当前仓库中较为“研究脚本化”的 `mini_grin/` **彻底重写**为更易审计、可复现、可持续产出的科研工程。

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

## 低分辨率 smoke test 的注意事项

`virtual_objects.py` 中的缺陷宽度/长度参数是以“物理单位”（与 `dx` 同单位，默认 µm）定义的。
如果仅把 `grid_size` 从 512 降到 64 但保持 `dx=0.39`，等价于把视场 (FOV) 缩小 8 倍，
会导致 scratch 相对视场过长、标准相位梯度过大、`data_weight_map` 几乎全域 clamp，从而更容易出现伪影/塌缩。

- 建议使用 `mini_grin_rebuild/configs/benchmark_smoke64_fov200.json`：通过增大 `dx` 保持约 200µm 的 FOV，使 smoke test 更贴近原物理尺度。

## 门控 / QC（可发表/可落地）

真实数据没有 ground-truth，因此需要一套 **GT-free** 的门控机制来衡量重建质量、抑制边缘伪影并监控域偏移。

- 文档：`mini_grin_rebuild/docs/GATING.md`
- 示例配置（含 q=0.99 门控阈值）：`mini_grin_rebuild/configs/benchmark_microlens200_srt_gated_q99.json`
