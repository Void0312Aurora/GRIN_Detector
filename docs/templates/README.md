# 模板与协议文档

`docs/templates/` 现在按内容拆成三个子目录，避免研究汇报模板、标准实验报告模板和协议文档混放在一起。

## 目录结构

- `research_reports/`
  阶段性研究汇报模板，强调问题定义、关键假设、run / config / 指标总结。
- `lab_reports/`
  标准实验报告模板，采用“实验标题、实验目的、实验原理、实验步骤、数据记录、数据处理、误差分析、问题讨论”体例。
- `protocols/`
  已排版的实验方案、协议和采集计划文档，不作为通用模板复用。

## 当前文件

- `research_reports/EXPERIMENT_REPORT_TEMPLATE.tex`
- `research_reports/EXPERIMENT_REPORT_TEMPLATE.pdf`
- `lab_reports/QDIC_WLI_STANDARD_EXPERIMENT_REPORT_TEMPLATE.tex`
- `lab_reports/QDIC_WLI_STANDARD_EXPERIMENT_REPORT_TEMPLATE.pdf`
- `protocols/QDIC_WLI_MICROLENS_ACQUISITION_PLAN.tex`
- `protocols/QDIC_WLI_MICROLENS_ACQUISITION_PLAN.pdf`

## 推荐用法

先创建新的实验目录：

```bash
cd /home/void0312/PINNs/mini_grin_rebuild
mkdir -p reports/exp_20260323_pilot/figures
```

如果你要写阶段性研究汇报，复制：

```bash
cp docs/templates/research_reports/EXPERIMENT_REPORT_TEMPLATE.tex reports/exp_20260323_pilot/report.tex
```

如果你要写标准实验报告，复制：

```bash
cp docs/templates/lab_reports/QDIC_WLI_STANDARD_EXPERIMENT_REPORT_TEMPLATE.tex reports/exp_20260323_pilot/report.tex
```

## 关键宏

研究汇报模板重点填写：

- `\ReportTitle`
- `\ReportID`
- `\AuthorName`
- `\ProblemStatement`
- `\KeyHypothesis`
- `\DataSummary`
- `\PrimaryRunPaths`
- `\PrimaryConfigPaths`
- `\HardwareSummary`
- `\InstrumentSummary`

标准实验报告模板重点填写：

- `\ExperimentTitle`
- `\ExperimentDate`
- `\ExperimentLocation`
- `\Experimenter`
- `\Partner`
- `\InstrumentA`
- `\InstrumentB`
- `\SampleName`
- `\ReportID`

## 编译方式

推荐使用 `xelatex` 或 `latexmk`，因为模板基于 `ctexart`，更适合中英文混排。

```bash
cd /home/void0312/PINNs/mini_grin_rebuild/reports/exp_20260323_pilot
latexmk -xelatex -interaction=nonstopmode report.tex
```

## 说明

- LaTeX 中间文件已经加入 `.gitignore`，`docs/templates/` 下默认只保留源文件和导出的 PDF。
- `protocols/` 里的文档更接近“已定稿方案”，后续如果继续扩展 SOP，可以继续放在该目录下。
