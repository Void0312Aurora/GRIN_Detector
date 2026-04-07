# External Data

该目录用于放置从公开来源获取的外部数据，目标不是替换当前 `microlens_srt` synthetic 主训练集，而是服务于两类工作：

- `height/topography -> SPDIC` 物理链验证
- `pre-real / hybrid` 测试与域迁移

目录约定：

- `raw/`: 原始下载包，不纳入 git
- `processed/`: 解压或转换后的中间结果，不纳入 git
- `manifests/`: 下载与扫描结果，可保留在仓库中

当前纳入管理的数据源：

- `zenodo_10365872`: 小体量 topography 数据，用于快速打通导入和仿真链
- `zenodo_18014400`: micro-milled 纹理数据，用于补充真实表面纹理先验
- `daks_33`: 后续候选，用于更强的干涉物理校准

当前已经落盘的 smoke 子集：

- `raw/zenodo_10365872_subset/Fig3_CSI_50x_AM.plux`
  当前已验证为可直接当 zip 容器读取，内部含 `LAYER_0.raw`、`LAYER_0.stack.raw`、`index.xml`、`metrics.txt` 等成员。
- `raw/zenodo_18014400_subset/Data_File_3.txt`
  当前已验证为可直接解析的制表符文本，表头为 `Filename / Scratch number / Average_CoF_per_cycle`，共 9600 行数据。

对应 probe 结果保存在：

- `manifests/zenodo_10365872_subset_probe.json`
- `manifests/zenodo_18014400_subset_probe.json`

真实实验相关文档与模板：

- `docs/REAL_DATA_EXPERIMENT_PLAN.md`
- `external_data/templates/real_capture_manifest.example.json`
- `docs/QDIC_WLI_MICROLENS_ACQUISITION_PLAN.md`
- `external_data/templates/qdic_wli_session_manifest.example.json`

常用命令：

```bash
cd /home/void0312/PINNs/mini_grin_rebuild
python scripts/manage_external_data.py list
python scripts/manage_external_data.py download --dataset zenodo_10365872
python scripts/manage_external_data.py extract --dataset zenodo_10365872
python scripts/manage_external_data.py scan --all
python scripts/probe_external_sample.py external_data/raw/zenodo_10365872_subset/Fig3_CSI_50x_AM.plux
python scripts/external_topography_smoke.py external_data/raw/zenodo_10365872_subset/Fig3_CSI_50x_AM.plux
```

`external_topography_smoke.py` 会执行以下流程：

- 读取 `.plux` 容器中的 `index.xml` 与 `LAYER_0.raw`
- 解析 FOV / 像素尺寸 / 仪器元数据
- 对 topography 做缺失值填充、平面去趋势、零均值化
- 调用项目内相位缩放与梯度模型，生成 `phase / I_x / I_y`
- 将 `npy` 数组与预览图写到 `external_data/processed/...`
- 将汇总 JSON 写到 `external_data/manifests/<sample>_spdic_smoke.json`

默认假设 `.plux` 中的 `FOV_X/FOV_Y` 单位为 `mm`，并把 topography `LAYER_0.raw` 解释为 `float32` 高度图，单位按 `µm` 进入当前项目物理链。
如果后续确认高度单位不是 `µm`，可在 smoke 时通过 `--height-scale` 进行显式换算。
