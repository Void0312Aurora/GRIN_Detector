# Physical Leakage Route Freeze Archive

Date: 2026-05-25

Status: frozen / archived

## Decision

The non-ideal physical leakage route is frozen as a research branch. It should not be treated as the current main path for symbol/sign extraction until real industrial DIC acquisition conditions are clarified.

The immediate reason is not that the idea is physically impossible. The issue is that the current synthetic non-ideal model is underconstrained by real instrument parameters. In the no-raw setting, the default `optical_leakage_lite` response behaves more like a strong domain shift / coherent texture generator than a validated sign-leakage observation model.

## Scope Archived Here

This archive preserves:

- no-raw DIC leakage sample visualizations
- controlled visibility ablation results
- sample 0012 diagnostic panels
- CSV/JSON summaries used to decide the freeze

Key files:

- `summary_metrics.csv`: condition-level visibility summary.
- `sample_metrics.csv`: per-sample visibility metrics.
- `dic_visibility_ablation_summary.json`: full controlled ablation payload.
- `visualization_summary.json`: prediction-vs-ground-truth visualization payload.
- `assets/visibility_heatmap.png`: median visibility heatmap.
- `assets/sample_0012_ablation_grid.png`: sample 0012 resolution/profile ablation.
- `assets/sample_0012_leakage_check.png`: sample 0012 localized visibility check.
- `assets/sample_0012_prediction_panel.png`: sample 0012 prediction/sign panel.

## Main Findings

1. The extra `raw` observation hypothesis was sealed as an optional future assumption, not a default industrial DIC assumption. Standard no-raw DIC only uses `I_x / I_y`.

2. Under no-raw DIC, the current default non-ideal response is too strong for a meaningful positive result. It suppresses or scrambles defect visibility rather than exposing reliable sign information.

3. Increasing resolution alone does not rescue the default non-ideal model. Median visibility z-score for the default profile remains low:
   `64: 0.70`, `128: 0.52`, `256: 0.38`.

4. Low-pass alone is not the main problem. The `lowpass_only` profile remains visibly informative:
   `64: 9.04`, `128: 7.18`, `256: 6.05`.

5. The strongest failure mode appears to come from the current coherent propagation / defocus / aberration combination, which introduces structured texture and washes out defect contrast.

6. Sample 0012 is a representative failure case: the defect exists in ground truth, but the default no-raw physical leakage observation makes it hard to see by eye.

## Reproduction Commands

Generate no-raw DIC leakage smoke data and suite results:

```bash
python scripts/run_no_raw_dic_leakage_ablation.py \
  --train 24 \
  --val 8 \
  --test 16 \
  --seed 42 \
  --num-plots 0 \
  --out /tmp/no_raw_dic_leakage_ablation
```

Generate prediction/sign visualizations:

```bash
python scripts/visualize_no_raw_dic_leakage_samples.py \
  --data-root /tmp/no_raw_dic_leakage_ablation/data/nonideal_dic_no_raw \
  --ideal-data-root /tmp/no_raw_dic_leakage_ablation/data/ideal_dic \
  --config configs/nonideal_dic_no_raw_smoke64.json \
  --checkpoint runs/20260525_170119_f243e82c_20260525_170118_bdcaa656_suite_no_raw_dic_leakage_smoke_nonideal_dic_no_raw_nn_dic_only_s42/checkpoints/best.pt \
  --out-dir /tmp/no_raw_dic_leakage_ablation/visualizations \
  --top-k 4 \
  --device cpu
```

Run the controlled visibility ablation:

```bash
python scripts/run_dic_visibility_ablation.py \
  --out /tmp/dic_visibility_ablation \
  --test 16 \
  --resolutions 64,128,256 \
  --profiles near_identity,weak,mid,default,lowpass_only \
  --sample-index 12 \
  --force
```

## If This Route Is Reopened

Reopen only with at least one of the following:

- real or vendor-supported DIC acquisition parameters
- measured optical response / calibration data
- a controlled optical model whose non-ideal parameters can be tied to instrument settings
- a clear reason to add an extra acquisition channel such as raw/direct intensity

Until then, this route should remain an archived exploratory branch rather than a main research direction.
