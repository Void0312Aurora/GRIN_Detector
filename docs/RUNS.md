# Run Management

该文件可通过 `python scripts/manage_runs.py refresh` 重新生成。

## Summary

- Total runs: `104`
- Total size: `29.7 GiB`
- Live runs in `runs/`: `3`
- Archived runs in `runs_archive/`: `101`
- Active runs: `3`
- Reference runs: `6`
- Archive candidates: `95`

## Policy

- `runs/` only keeps curated `active` runs plus future new outputs.
- Historical exploratory runs are physically moved to `runs_archive/<kind>/`.
- `reference` runs are also archived, but still exposed through `run_refs/`.
- Unlisted runs default to `archive_candidate`; they are the first targets for migration.
- Shortcut symlinks for curated runs are created in `run_refs/`, regardless of live/archive storage.

## Active Runs

| Alias              | Run                                                  | Storage | Kind     | Size      | Timestamp           | Note                                    |
| ------------------ | ---------------------------------------------------- | ------- | -------- | --------- | ------------------- | --------------------------------------- |
| ood_suite_taper_v1 | 20260123_165204_11172324_20x_ood_suite_taper_v1      | live    | 20x      | 9.3 GiB   | 2026-01-23 16:52:04 | 20x OOD suite with taper variant.       |
| pre_real_best_v2   | 20260122_194443_3af24edd_pre_real_validation_best_v2 | live    | pre_real | 455.9 MiB | 2026-01-22 19:44:43 | Updated pre-real validation checkpoint. |
| qc_eval_best       | 20260122_132018_4c5bb1b1_qc_eval_best                | live    | qc       | 6.8 KiB   | 2026-01-22 13:20:18 | Primary QC evaluation summary.          |

## Reference Runs

| Alias                    | Run                                                                      | Storage  | Kind       | Size      | Timestamp           | Note                                                            |
| ------------------------ | ------------------------------------------------------------------------ | -------- | ---------- | --------- | ------------------- | --------------------------------------------------------------- |
| ood_suite_v1             | 20260123_145017_87d0b039_20x_ood_suite_v1                                | archived | 20x        | 9.3 GiB   | 2026-01-23 14:50:17 | Pre-taper 20x OOD baseline; kept for comparison against taper.  |
| pre_real_best            | 20260122_193357_f9f5d803_pre_real_validation_best                        | archived | pre_real   | 455.9 MiB | 2026-01-22 19:33:57 | Older pre-real validation checkpoint; v2 is the live successor. |
| suite_opt_gate_q99_v1    | 20260122_131349_08bf608f_suite_optimize_gate_q99_v1_gated_q99_v1         | archived | suite_root | 64.4 KiB  | 2026-01-22 13:13:49 | Gate optimization suite root.                                   |
| suite_opt_edge_gated_q99 | 20260122_130505_bd1592bc_suite_optimize_edge_v1_gated_q99                | archived | suite_root | 106.5 KiB | 2026-01-22 13:05:05 | Edge optimization suite root.                                   |
| gate_srt_q99             | 20260122_120109_bfeede16_gate_microlens_srt_q99                          | archived | gate       | 3.0 MiB   | 2026-01-22 12:01:09 | GT-free gating reference with q=0.99.                           |
| ablation_signless_report | 20260121_232000_eb006e63_suite_ablation_signless_v1_microlens_srt_report | archived | suite_root | 23.4 KiB  | 2026-01-21 23:20:00 | Microlens SRT ablation report root.                             |

## Largest Runs

| Run                                                                                                                             | Status            | Storage  | Kind         | Size      | Timestamp           |
| ------------------------------------------------------------------------------------------------------------------------------- | ----------------- | -------- | ------------ | --------- | ------------------- |
| 20260123_145017_87d0b039_20x_ood_suite_v1                                                                                       | reference         | archived | 20x          | 9.3 GiB   | 2026-01-23 14:50:17 |
| 20260123_165204_11172324_20x_ood_suite_taper_v1                                                                                 | active            | live     | 20x          | 9.3 GiB   | 2026-01-23 16:52:04 |
| 20260122_194443_3af24edd_pre_real_validation_best_v2                                                                            | active            | live     | pre_real     | 455.9 MiB | 2026-01-22 19:44:43 |
| 20260122_193357_f9f5d803_pre_real_validation_best                                                                               | reference         | archived | pre_real     | 455.9 MiB | 2026-01-22 19:33:57 |
| 20260122_010803_55f49a91_20260122_010643_7fd30346_suite_optimize_edge_v1_B1_nn_coord_inputs_s42                                 | archive_candidate | archived | suite_member | 208.2 MiB | 2026-01-22 01:08:03 |
| 20260122_010910_fb172c0a_20260122_010643_7fd30346_suite_optimize_edge_v1_B1_nn_coord_reflect_s42                                | archive_candidate | archived | suite_member | 208.2 MiB | 2026-01-22 01:09:10 |
| 20260122_010729_288941d5_20260122_010643_7fd30346_suite_optimize_edge_v1_B1_nn_phase_inputs_s42                                 | archive_candidate | archived | suite_member | 208.2 MiB | 2026-01-22 01:07:29 |
| 20260122_003724_b7c58c69_20260122_003456_28f31a0c_suite_ablation_signless_v1_metricsA_fixregions_nn_prior_residual_teacher_s42  | archive_candidate | archived | suite_member | 208.2 MiB | 2026-01-22 00:37:24 |
| 20260122_003146_1b3637f2_20260122_002923_456b7246_suite_ablation_signless_v1_metricsA_nn_prior_residual_teacher_s42             | archive_candidate | archived | suite_member | 208.2 MiB | 2026-01-22 00:31:46 |
| 20260121_232255_61080302_20260121_232000_eb006e63_suite_ablation_signless_v1_microlens_srt_report_nn_prior_residual_teacher_s42 | archive_candidate | archived | suite_member | 208.2 MiB | 2026-01-21 23:22:55 |

## Largest Archived Runs

| Run                                                                                                                             | Kind         | Size      | Timestamp           |
| ------------------------------------------------------------------------------------------------------------------------------- | ------------ | --------- | ------------------- |
| 20260123_145017_87d0b039_20x_ood_suite_v1                                                                                       | 20x          | 9.3 GiB   | 2026-01-23 14:50:17 |
| 20260122_193357_f9f5d803_pre_real_validation_best                                                                               | pre_real     | 455.9 MiB | 2026-01-22 19:33:57 |
| 20260122_010803_55f49a91_20260122_010643_7fd30346_suite_optimize_edge_v1_B1_nn_coord_inputs_s42                                 | suite_member | 208.2 MiB | 2026-01-22 01:08:03 |
| 20260122_010910_fb172c0a_20260122_010643_7fd30346_suite_optimize_edge_v1_B1_nn_coord_reflect_s42                                | suite_member | 208.2 MiB | 2026-01-22 01:09:10 |
| 20260122_010729_288941d5_20260122_010643_7fd30346_suite_optimize_edge_v1_B1_nn_phase_inputs_s42                                 | suite_member | 208.2 MiB | 2026-01-22 01:07:29 |
| 20260122_003724_b7c58c69_20260122_003456_28f31a0c_suite_ablation_signless_v1_metricsA_fixregions_nn_prior_residual_teacher_s42  | suite_member | 208.2 MiB | 2026-01-22 00:37:24 |
| 20260122_003146_1b3637f2_20260122_002923_456b7246_suite_ablation_signless_v1_metricsA_nn_prior_residual_teacher_s42             | suite_member | 208.2 MiB | 2026-01-22 00:31:46 |
| 20260121_232255_61080302_20260121_232000_eb006e63_suite_ablation_signless_v1_microlens_srt_report_nn_prior_residual_teacher_s42 | suite_member | 208.2 MiB | 2026-01-21 23:22:55 |
| 20260122_010945_43a87501_20260122_010643_7fd30346_suite_optimize_edge_v1_B1_nn_edge_band_suppress_s42                           | suite_member | 208.2 MiB | 2026-01-22 01:09:45 |
| 20260122_003113_3d0a3a6d_20260122_002923_456b7246_suite_ablation_signless_v1_metricsA_nn_prior_residual_input_s42               | suite_member | 208.2 MiB | 2026-01-22 00:31:13 |

## Archived Runs By Kind

| Kind         | Count | Combined Size |
| ------------ | ----- | ------------- |
| suite_member | 56    | 8.3 GiB       |
| smoke        | 23    | 1.2 GiB       |
| suite_root   | 5     | 164.6 KiB     |
| scan         | 3     | 622.2 MiB     |
| other        | 3     | 2.3 MiB       |
| gate         | 3     | 1.5 MiB       |
| qc           | 2     | 32.5 KiB      |

## Live Archive Candidates By Kind

| Kind | Count | Combined Size |
| ---- | ----- | ------------- |

## Registry Statuses

| Status            | Meaning                                                                                                 |
| ----------------- | ------------------------------------------------------------------------------------------------------- |
| active            | Current baseline or result that should stay easy to find during ongoing research.                       |
| reference         | Historical run that is still useful for comparison, report reproduction, or QC.                         |
| archive_candidate | Historical exploratory run. Keep the original path stable for now, but treat it as archived by default. |
