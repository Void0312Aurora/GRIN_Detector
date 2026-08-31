# Reflection dark-port sim-to-real calibration (2026-07-21; visual correction 2026-07-23)

## Evidence boundary

The available real archive contains 24 eight-bit dark-port BMP frames, one frame
from each distinct microlens. It does **not** contain paired `reference`,
`standard`, and `test` captures, a calibrated `I_y` channel, repeated flat/dark
frames, or height ground truth. Consequently this work calibrates the appearance
and nuisance distribution of `I_x`; it does not identify a unique optical cause
for every observed defect or establish a quantitative scattering-to-height law.

All 24 frames were visible to earlier exploratory fitting. The split in
`configs/reflection_real_blocked_v1.json` is therefore retrospective:

- five April four-frame blocks are used for leave-one-block-out comparison;
- `21.bmp` through `24.bmp`, whose mtimes are from 2026-03-27, are reported as a
  cross-date temporal check;
- neither result is an independent external test. A new acquisition batch is
  required for that claim.

## Selected implementation

The selected deterministic and nuisance configs are:

- `configs/reflection_microlens520_sim2real_actual.json`;
- `configs/reflection_microlens520_sim2real_noisy.json`.

They replace the principal mismatches of the previous configs:

1. The camera can emit native eight-bit DN with exposure and black level applied
   before shot noise and ADC. The noisy config also has optional correlated read
   noise. Existing configs remain byte-compatible when the new camera keys are
   absent. `output_mode: "dn"` selects the DN saturation default and tells the
   evaluator not to perform a second radiometric fit; the selected configs also
   set an explicit eight-bit ADC and `255 DN` saturation.
2. The fixture has coarse and fine phase scales plus persistent amplitude
   texture. The lens interior now has continuous multi-scale scattering fields
   that can be shared within a capture and partly locked to an acquisition-base
   seed; the deterministic config no longer relies on sparse point scatterers.
3. The 2026-07-23 correction removes the deterministic full rough seam ring
   (`rim_amplitude: 0`, random rim modulation disabled) and adds an explicit
   fourth-order inner-edge focus term after the shared DIC blur. This is an
   empirical unresolved polarization/illumination coupling term, not a claim
   that its unique optical cause has been identified.
4. The camera supports a two-scale fixed spatial residual. In the selected
   deterministic config, `20%` of its variance is coordinate-locked and the
   remainder is capture-specific. This represents the common acquisition base
   without asserting that all observed interior texture is sensor noise; the
   real-data decomposition still shows that most per-frame interior energy is
   lens-specific.
5. The formal dataset path can attach a localized complex scattering modifier to
   the synthetic `test` object. `reference` and `standard` remain unchanged, and
   missing/disabled scattering preserves the legacy path.
6. The selected configs do not emit or train on `I_raw`, because the real
   acquisition does not provide that channel.

The scattering factors (`8-12` amplitude and `0.6-1.2 rad` rough phase) are
provisional heuristics informed by three strong-anomaly frames, not calibrated
material constants. The clean-microlens evaluator below does not inject a defect
modifier and therefore does not validate these scattering ranges; it validates
only the common optical appearance and per-frame nuisance distribution.

## Validation protocol

`scripts/evaluate_reflection_sim2real.py` evaluates configs as written. It does
not replace reflectance parameters with an internal sweep. For legacy configs,
linear radiometry is fit using only each fold's fit frames. DN configs are never
refit. The report includes per-fold metrics, config and detection hashes, hashes
for all 24 raw BMP files, seed, ensemble size, and the Git commit.

The aggregate score covers radial and seam RMSE, rim saturation, the ordered
inner-edge harmonic spectrum (`H1-H8`), fourth-order phase, focus coverage and
peak-to-median ratio, edge rise and FWHM, fixture scale/contrast/level, and
interior texture coverage, kurtosis and spectrum on both the common image and
individual captures. It is a project-specific diagnostic score, not a standard
perceptual metric. Scores produced before the 2026-07-23 metric expansion are
not numerically comparable with the new score.

The historical 2026-07-21 results below use the earlier score and are retained
only for provenance. They use four simulation captures with seeds `20260721`,
`21260724`, `22260727`, and `23260730`:

| Validation | Config | Baseline score | Candidate score | Relative reduction |
| --- | --- | ---: | ---: | ---: |
| Five-block retrospective CV | deterministic | 10.443 | 6.463 | 38.1% |
| Cross-date temporal check | deterministic | 13.137 | 6.877 | 47.7% |
| Five-block retrospective CV | noisy | 13.375 | 5.645 | 57.8% |
| Cross-date temporal check | noisy | 17.457 | 6.414 | 63.3% |

Selected cross-date deterministic metrics show where the gain comes from:

| Metric | Real | Old simulation | New simulation |
| --- | ---: | ---: | ---: |
| Edge rise 10-90 (um) | 7.57 | 1.02 | 5.70 |
| Fixture correlation length (px) | 17.0 | 11.5 | 16.5 |
| Rim angular CV | 0.519 | 0.134 | 0.444 |
| Rim dipole amplitude | 0.557 | 0.019 | 0.613 |
| Interior median (DN) | 3.5 | 5.43 | 3.0 |
| Seam profile RMSE (DN) | - | 67.14 | 29.63 |

The noisy config brings the cross-date interior `p99.9` from `95.7 DN` in the
old noisy model to `11.0 DN`, against `11.5 DN` in the real frames.

## Visual-structure correction (2026-07-23)

The previous deterministic candidate still failed the two most obvious visual
checks: its inner edge remained a nearly complete saturated annulus, and its
log-intensity interior was empty except for isolated points. The evaluator was
part of the problem because it reduced the edge to CV/dipole summaries and the
interior to a median and `p99.9` tail.

The corrected evaluator measures an angularly matched inner-edge contrast
profile, its ordered `H1-H8` spectrum, the fourth-order phase, focus coverage and
peak-to-median ratio. It also radially detrends the interior and records
fine/medium texture standard deviation, dense-pixel coverage, kurtosis and 2-D
spectral bands. These terms enter the aggregate score, so a complete ring or a
set of sparse bright dots can no longer pass by matching only radial averages.

A fair comparison re-evaluated the archived 2026-07-21 deterministic config and
the corrected config under the same expanded score:

| Validation | Archived deterministic | Corrected deterministic | Relative reduction |
| --- | ---: | ---: | ---: |
| Five-block retrospective CV | 16.719 | 10.995 | 34.2% |
| Cross-date temporal check | 20.162 | 5.885 | 70.8% |

> **RETRACTED 2026-08-09.** These two score reductions are artifacts of a
> defective scorer and must not be cited. The evaluator's `_score` had an
> unbounded `interior_common_texture_kurtosis` term: because the archived model
> has a near-empty interior, its kurtosis reached `234` against a real `3.7`,
> and that single term supplied `31.5` of the archived model's `55.9` temporal
> points (56%). The apparent improvement measures mostly the removal of that
> divergence, not better agreement. Two further defects compounded it: eight of
> 24 terms describe interior texture and carried a third of the total weight,
> and the weighted mean had no way to express "this must not get worse".
>
> Under the fixed scorer (per-term ceiling, family-renormalized weights) the same
> recorded folds give archived `13.21` versus corrected `3.94` on the temporal
> check -- the ordering survives, the magnitude does not. More importantly the
> new veto gate **fails on all five raw fidelity metrics**, in this fair
> comparison and in the legacy-baseline one:
>
> | Gate metric | Archived | Corrected | Change |
> | --- | ---: | ---: | ---: |
> | Radial profile RMSE (DN) | 31.30 | 47.76 | 52.6% worse |
> | Radial profile corr | 0.859 | 0.589 | 31.4% worse |
> | Seam profile RMSE (DN) | 43.84 | 83.28 | 90.0% worse |
> | Seam profile corr | 0.868 | 0.370 | 57.4% worse |
> | Low-pass aperture corr | 0.782 | 0.717 | 8.2% worse |
>
> The section below acknowledged only the single-fold seam RMSE cost
> (`29.63 -> 51.69`). The regression is in fact systematic: all 12 folds, all
> five metrics. The corrected config bought angular inner-edge structure by
> giving up radial and seam profile fidelity, including the seam relief this
> project had separately fitted to correlation `0.864`.
>
> Consequently `reflection_microlens520_sim2real_actual.json` is **not a
> validated working point** and must not be promoted to the actual config until
> it passes the gate. Whether the search was misled by the same defect (it calls
> the same `_score`) or the trade was made knowingly is not yet determined.
>
> **Re-run under the fixed scorer** (`analysis_outputs/reflection_sim2real_v4_gated/old_vs_visual/`,
> same seed `20260721`, ensemble `4`, channel `I_x` and split manifest as the
> original, from a clean tree at `742e5aa`):
>
> | Validation | Archived | Corrected | Verdict |
> | --- | ---: | ---: | --- |
> | Five-block CV | 7.259 +- 0.711 | 6.681 +- 1.785 | overlapping; not separable |
> | Cross-date temporal | 9.462 | 4.012 | corrected better |
> | Gate | - | - | **gate_fail, 5/5 metrics** |
>
> The blocked-CV gap the original reported as a `34.2%` reduction is not
> reproducible once the unbounded term is capped and collinear terms are
> renormalized: `7.259 +- 0.711` versus `6.681 +- 1.785` overlap within one
> standard deviation, and the corrected model is now the *less* stable of the
> two across folds. Only the single-fold temporal number still favours the
> corrected config, and a single fold cannot carry a working-point decision.
>
> Gate detail from this re-run: radial RMSE `30.91 -> 46.46` (+50.3%), radial
> corr `0.877 -> 0.621` (-29.2%), seam RMSE `37.43 -> 81.10` (+116.6%), seam
> corr `0.921 -> 0.397` (-56.9%), low-pass aperture corr `0.805 -> 0.732`
> (-9.1%).
>
> This is the disposition: the corrected config is a mechanism study that
> established the four-lobe inner edge is real and reproducible, not a
> calibrated working point. The angular work must be redone under the
> constraint that seam profile correlation stays at or above the `0.864` this
> project reached in the joint seam fit.

Selected cross-date metrics (four-simulation common image unless marked
"individual median") are:

| Metric | Real | Archived simulation | Corrected simulation |
| --- | ---: | ---: | ---: |
| Inner-edge focus coverage | 0.549 | 0.757 | 0.562 |
| Inner-edge peak / median | 1.711 | 1.453 | 1.722 |
| Inner-edge `H1` amplitude | 0.512 | 0.344 | 0.511 |
| Inner-edge `H4` amplitude | 0.364 | 0.238 | 0.277 |
| Inner-edge `H4` phase (deg, period 90) | 89.84 | 46.54 | 88.67 |
| Interior fine texture std (DN) | 0.631 | ~0 | 0.681 |
| Interior fine texture std, individual median (DN) | 1.063 | 0.018 | 0.981 |
| Interior dense coverage, individual median | 0.303 | 0.000 | 0.314 |
| Interior spectral centroid (cycles/px) | 0.193 | 0.000 | 0.193 |
| Fixture median (DN) | 22.5 | 58.5 | 21.0 |
| Fixture correlation length (px) | 17 | 17 | 17 |
| Fixture contrast | 0.590 | 0.543 | 0.578 |

The texture term is intentionally mostly capture-specific: the earlier
sector-coherence and cross-frame tests found the real interior texture to be
isotropic and weakly correlated across lenses. A `20%` coordinate-locked
fraction represents the common acquisition base; it is not evidence that the
entire texture is shared or that its physical source is uniquely known.

## Remaining mismatch

The corrected deterministic model is not a complete image match:

- cross-date common-image rim saturation is `0.119` versus `0.048` real, and the
  rim FWHM is `7.90 um` versus `13.17 um`; the four directions are now present,
  but their radial band is still too narrow and clips too often;
- the corrected `H2` and `H5` amplitudes (`0.293`, `0.239`) remain above the real
  values (`0.120`, `0.100`), so the full ordered angular spectrum is not yet a
  match even though `H1`, focus coverage and fourth-order phase align;
- the cross-date radial RMSE is essentially unchanged (`29.74 -> 29.71 DN`),
  while seam RMSE worsens (`29.63 -> 51.69 DN`). This is the explicit cost of
  removing the unrealistically bright complete annulus; the next seam update
  must recover the broad radial skirt without erasing the four-lobe structure;
- a single deterministic exposure still cannot represent the acquisition-block
  brightness shift, although the cross-date fixture median is now close
  (`21.0 DN` simulation versus `22.5 DN` real);
- the fixed spatial residual is an empirical acquisition-base model. With no
  repeated dark/flat captures, its optical-versus-sensor attribution remains
  unidentified;
- only `I_x` is calibrated. `I_y` must remain provisional until a true two-axis
  capture is available;
- height-only, scattering-only, and coupled defects are not identifiable from
  one unpaired frame per microlens.

The next acquisition should include dark/flat frames, repeated captures of the
same microlens, paired nominal/defect states, both shear axes, and an independent
height measurement. Those data are needed to separate instrument-fixed texture,
sample-specific scattering, temporal drift, and height response.

## Reproduction artifacts

Strict-JSON summaries, comparison figures, implementation source snapshots, and
the tracked working-tree patch are generated under the ignored real-data product
directory:

- `reflection_sim2real_v1/final_ensemble4_final/`;
- `reflection_sim2real_v1/noisy_ensemble4_final/`.

The 2026-07-23 corrected deterministic artifacts are under:

- `analysis_outputs/reflection_sim2real_v3/old_vs_visual/` (fair archived-vs-corrected comparison);
- `analysis_outputs/reflection_sim2real_v3/candidate_eval/` (legacy baseline vs corrected candidate);
- `analysis_outputs/reflection_visual_structure_v3/candidate/` (ordered edge and interior-texture audit).

The noisy config has been migrated to the same mechanisms but was not
recalibrated or revalidated in the 2026-07-23 deterministic-only correction.

The calibration sweep is reproducible with
`scripts/search_reflection_sim2real.py` and the
`configs/reflection_sim2real_search_*.json` manifests. The evaluator and search
tool write strict JSON and fail rather than emitting non-standard `NaN` values.
