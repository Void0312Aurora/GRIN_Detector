# Reflection simulation adaptation (updated 2026-07-21)

## Scope

This adaptation treats the available apparatus information as the working boundary:

- front-surface reflection model;
- wavelength `0.520 um`;
- numerical aperture `0.42`;
- microlens curvature radius `400 um`;
- microlens sag `15.34 um`;
- no sample refractive index in the geometric height-to-phase conversion.

The apparatus family is the weak-measurement differential imaging platform of Liu et al., Phys. Rev. A 106, 023518 (2022); see the correspondence section at the end of this document. Lab feedback (2026-07-21) confirms:

- the 24 frames are 24 distinct microlenses from the same nominal batch, one frame each;
- two-axis shear-differential captures (`I_x`/`I_y`) are planned as the next acquisition;
- the captures are taken at the postselected dark port;
- shear magnitude `10 um` (reference plane not yet confirmed: object plane would be `21.3` simulation px, detector plane would be `10/9.09 = 1.1 um` object, i.e. `2.35` simulation px);
- source: semiconductor laser, centre `520 nm`, bandwidth `2 nm` (coherence length about `135 um`, consistent with the `>= 19 um` data-driven bound);
- camera pixel pitch `2.4 um`; system magnification `9.09x`, giving a nominal object-plane scale of `0.264 um/px` and a full-frame field of view of about `1445 x 963 um`.

The stated magnification conflicts with the lens-geometry self-calibration by a factor of `1.49`: the detected dark-disk radius (`620 px` median) equals the spherical-cap aperture radius (`109.7 um`) only at an effective magnification of `13.56x`. The lab states there is no relay stage, so the conflict must be resolved against internal evidence. Three independent internal checks now favour `13.56x`:

1. cap-geometry self-calibration: disk radius `620 px` matches the `109.7 um` clear aperture only at `13.56x`;
2. interior ring check: if the disk were a larger cell region at `9.09x`, the cap edge (a slope discontinuity that must scatter brightly at a dark port) would sit at `67%` of the disk radius; the measured radial median intensity and high-pass energy are featureless and monotonic from `rho 0.32` to `0.84` with the global energy minimum near `0.68`, and the only bright ring is at the disk edge itself;
3. objective identification: `NA 0.42` is characteristic of a `20x` M-Plan-Apo-class objective (`f = 10 mm`, design tube `200 mm`); with a `135 mm` tube lens this gives `13.5x`, matching the self-calibration to `0.5%`, whereas `9.09x` would require a non-standard `90.9 mm` tube (`9.0909 = 200/22` suggests a focal-length bookkeeping slip).

Working scale therefore remains `0.177 um/px` (`13.56x`). To settle it externally, either check the tube-lens focal length (if it is `135 mm`, the stated `9.09x` was computed with the wrong tube assumption) or the lens array pitch (neighbouring circle centres are `1344 px` apart: `238 um` at `13.56x` versus `355 um` at `9.09x`, immediately decidable from the array datasheet).

Because the port is now known to be the postselected dark port, the bright-field `|U|^2` comparison below must be read as a shape surrogate, not as the physical forward model; see the dark-port model test section. The 24 BMP files remain single-frame raw observations, not `test-standard` arrays. Cross-sample empirical differences are constructed only after aperture registration and intensity normalisation.

## Implemented model changes

1. Height-to-phase conversion now supports two explicit modes:

   - transmission: `2*pi*(n_object-n_air)*h/lambda`;
   - reflection: `4*pi*n_air*cos(theta)*h/lambda`.

2. The synthetic microlens can use a physical spherical cap:

   `h(r) = sqrt(R^2-r^2) - (R-s)` for `r <= sqrt(2*R*s-s^2)`.

3. The complex-field engine can use a coherent circular pupil with cutoff `NA/lambda`, rather than only an empirical Gaussian frequency aperture.

4. The apparatus-oriented config is `configs/reflection_microlens520_actual.json`. Its `-100` defocus and `8 px` raw blur are nuisance-fit values in the current simulator coordinates, not calibrated instrument measurements.

5. The complex-field engine now supports localized coherent ghost paths. Each path can include:

   - a finite smooth source support applied before ghost propagation;
   - a detector-plane visibility envelope representing finite beam overlap or mutual coherence;
   - lateral shift, tilt, defocus and aberration differences;
   - a correlated complex source texture representing nonuniform local reflection or scattering;
   - multiple ghost components when a single parasitic path is insufficient.

   With all new amplitudes disabled, the previous propagation result is unchanged. The apparatus config remains free of fitted ghost parameters; the data-informed nuisance ranges are isolated in `configs/reflection_microlens520_noisy.json`.

## Comparison protocol

`scripts/compare_reflection_capture.py` performs the raw-frame validation:

- crop the one valid central lens from each of the 24 BMP frames;
- resize crops to the physical simulation field of view;
- robustly normalise each crop using the interior baseline and aperture-edge intensity;
- form the pixelwise median of all 24 crops to suppress isolated defects and dust;
- compare radial profiles and two-dimensional aperture images with propagated coherent raw reflection intensity.

The project's internal `I_x/I_y` and `test-standard` arrays are not compared directly with the raw BMP files.

`scripts/build_empirical_reflection_differences.py` then:

- ranks registered frames by robust interior distance to the 24-frame median;
- selects `7.bmp` as the current empirical standard frame;
- constructs `registered_test - registered_standard` for all frames;
- masks the outer `12%` of the aperture radius, where small radius/registration errors dominate;
- records signed difference maps and robust anomaly scores.

## Current result

| Raw-frame comparison | Radial correlation | Aperture image correlation | Edge image correlation | Radial RMSE |
| --- | ---: | ---: | ---: | ---: |
| Coherent raw reflection | 0.991 | 0.924 | 0.876 | 0.103 |

The coherent raw model reproduces the dominant dark-aperture / bright-rim structure. This was originally an in-sample shape fit after sweeping defocus and blur; the held-out check below now shows the selected nuisance values transfer across frames, though this is still not a radiometric or instrument calibration.

The empirical subtraction is useful despite missing original pair labels because the microlenses are assumed to share the same nominal geometry. The current highest residual frames are `5.bmp`, `19.bmp`, `20.bmp`, `18.bmp`, `14.bmp`, `13.bmp`, `2.bmp`, and `23.bmp`, which is broadly consistent with the visibly large oval and scratch-like structures.

The empirical difference has a strict interpretation boundary: it is a cross-sample deviation from a nominally clean lens, not the same lens measured before and after a defect. Manufacturing variation, focus and illumination remain in the residual, so it is appropriate for anomaly display and simulation-shape comparison but not yet quantitative height ground truth.

The remaining mismatch is substantial:

- real edge angular coefficient of variation: `0.379`;
- coherent simulation edge angular coefficient of variation: `0.027`.

The simulation is almost rotationally symmetric, while the real captures contain strong directional illumination, fixture texture, contamination and sample-specific anomalies. These effects should be introduced as nuisance variation only after acquisition semantics are recovered.

Generated local comparison artifacts are under:

`external_data/processed/wechat_2026-07_15-34/reflection_simulation_comparison/`

Empirical difference artifacts are under:

`external_data/processed/wechat_2026-07_15-34/empirical_reflection_differences/`

They remain ignored by Git together with the raw observational data.

## Localized stripe experiment

`scripts/render_reflection_localized_ghost_comparison.py` removes the robust median high-pass image of all 24 registered frames, estimates the excess stripe-energy support in `13.bmp` and `14.bmp`, and fits only the localized ghost path. It does not fit the gross microlens rim in the same objective.

The fitted excess-energy centers are in the upper-left part of the aperture:

| Frame | center x | center y | major sigma | minor sigma |
| --- | ---: | ---: | ---: | ---: |
| `13.bmp` | -0.120 | -0.252 | 0.328 | 0.273 |
| `14.bmp` | -0.168 | -0.194 | 0.336 | 0.281 |

The finite-support model fixes the principal failure of the global ghost model: simulated stripe energy is now spatially localized. Adding correlated complex texture inside the ghost source also breaks the unrealistically continuous fringes and improves localized spectrum correlation from approximately `0.37-0.41` to `0.51-0.53` while retaining energy-map correlation near `0.76`.

This is still not a full morphological match. The simulated pattern retains a visibly dominant fringe family, while the real residual is more broadband and irregular. A second coherent ghost path was tested but did not improve the combined energy/spectrum score and is therefore not enabled in the noisy config. The current conclusion is limited to:

- finite beam overlap or coherence support is necessary to explain localization;
- nonuniform local reflection/scattering is useful to explain broken fringe continuity;
- the available images do not identify the responsible surface or a unique ghost path;
- cross-frame transfer between `13.bmp` and `14.bmp` has now been checked (see the generalization section); a broader validation still requires more frames that show the stripe anomaly.

Generated localized-ghost artifacts are under:

`external_data/processed/wechat_2026-07_15-34/reflection_localized_ghost_simulation/`

## Dark-port model test (2026-07-21)

With the port now confirmed as the postselected dark port, `compare_reflection_capture.py` gained `--channel {I_raw,I_x,I_y}` and `--shear-px` so the sheared-difference channels can be compared directly. First results against the 24-frame median:

| Model | Best profile corr | Best aperture image corr | Comment |
| --- | ---: | ---: | --- |
| Bright `I_raw` surrogate (previous) | 0.991 | 0.924 | shape fit only, port-inconsistent |
| Dark `I_x`, shear `21.33 px` (`10 um` object plane) | 0.682 | 0.393 | dense shear fringes fill the aperture |
| Dark `I_y`, shear `21.33 px` | 0.682 | 0.395 | same, rotated |
| Dark `I_x`, shear `1.57 px` (detector-plane reading at `13.56x`) | 0.677 | 0.346 | one broad ring instead of black interior |

The uniform-amplitude coherent dark-port model fails in an instructive way: it predicts strong shear fringes (period about `2*pi*R_curv/(k*delta)`, roughly `10 um` for the object-plane shear) or a bright ring inside the aperture, while the real interiors are uniformly black at `2-6 DN` with only defects, edges and fixture texture lighting up. The real dark port behaves like the ideal differential edge detector of the PRA scheme. The most likely missing ingredients, in order:

1. an amplitude/reflectance map: the lens front surface is weakly reflective and smooth, while the fixture scatters strongly; the engine currently assumes `amplitude = 1` everywhere;
2. steep-slope pupil rejection: over most of the aperture the doubled surface slope exceeds the `NA 0.42` acceptance, so the smooth-lens contribution never reaches the detector and cannot form fringes;
3. postselection leakage (offset/extinction) adding a weak coherent floor.

The `2-6 DN` interior corresponds to at most a few photoelectrons at the measured `0.46 DN/e-` gain, so the black interior is radiometrically consistent with near-total suppression. Conclusion: the bright-field `I_raw` comparison stays as a shape surrogate for registration purposes, and the physical dark-port adaptation needs the amplitude/pupil ingredients before the fringe-free black aperture can be reproduced. Fit artifacts are under `reflection_darkport_comparison_ix`, `_iy` and `_ix_small`.

Config note: both reflection configs now carry `shear_px = 1.573` (the camera-plane reading of the stated `10 um` at the `13.56x` working magnification, i.e. `0.74 um` in the object plane) and a data-calibrated camera block (`photon_gain` about `200-400` photoelectrons per intensity unit, matching the measured `0.46 DN/e-` at `saturation_level 2.0`; the previous `8000-20000` overstated the photon budget by more than an order of magnitude). The shear value is provisional until the reference plane is confirmed.

## Dark-port physical model fit (2026-07-21, second pass)

The engine now implements the missing dark-port ingredients:

- `reflectance` block: an amplitude/phase map with a weakly reflective smooth lens cap, a strongly scattering rough fixture outside the cap (correlated rough phase), and a rough scattering annulus at the cap-substrate seam (`rim_amplitude`, `rim_width_px`);
- `dark_port` block: postselected output per axis with two placements, `image_shear` (`E(r+d/2)-E(r-d/2)+eps*E`) and `fourier_tilt` (`E*2i*sin(pi*q*r+phi)+eps*E`), where `eps` is the coherent postselection leakage;
- default-off behaviour is bit-identical to the legacy sheared-difference channels (unit-tested).

`scripts/compare_reflection_dark_port.py` fits both placements against the 24-frame native median in absolute DN, with the exposure scale and a small dark offset as the only radiometric nuisances:

| Level (DN) | Real | Best image_shear | Best fourier_tilt |
| --- | ---: | ---: | ---: |
| Lens interior median | 3.0 | 3.1 | 17.7 |
| Mid annulus 0.7-0.9 | 3.0 | 3.0 | 23.8 |
| Fixture median | 79.0 | 87.7 | 88.1 |
| Rim p95 | 255 (clipped) | 179 | 196 |
| Aperture image corr | - | 0.825 | 0.434 |

The image-plane shear placement wins decisively on absolute levels: with lens amplitude at or below about `0.1-0.4` relative units, the smooth-cap dark-port residual falls below the camera dark floor (about `3 DN`), reproducing the featureless black interior exactly; the rough fixture decorrelates under the shear and stays bright; the seam annulus produces the bright rim. The best fit uses `defocus 0`, `blur 2 px`, `leak 0`, so the large `-100` defocus of the bright-field surrogate is revealed as a wrong-port artefact. The Fourier-plane placement cannot make the interior dark while keeping the fixture bright and is rejected. Identifiability notes: the lens specular amplitude is only bounded from above (its dark-port residual sits below the dark floor); rim clipping is under-reproduced (`179` vs `255`), and the rim angular anisotropy (`0.09` vs `0.38`) still lacks the directional illumination/fixture component.

`configs/reflection_microlens520_actual.json` now carries the winning dark-port configuration (`image_shear`, `shear_px 1.573`, reflectance map). The noisy config still describes the bright-field surrogate nuisances and must be migrated to dark-port terms in the calibration step. Artifacts are under `reflection_darkport_model_fit/`.

### Structural evaluation (third pass)

A reviewer question ("why does the simulation look blurrier than the real capture?") exposed three defects in the second-pass evaluation, now fixed:

1. The radial-profile RMSE selection metric is nearly blind to sharpness; it had accepted `blur = 2 px` (about three times the diffraction sigma) and a wide weak seam ring. The evaluation now adds structural metrics: edge 10-90 rise width, rim FWHM and p95, fixture texture correlation length, and fixture contrast.
2. The real reference was the 24-frame median, which softens structure by construction: registration jitter widens the edge rise from `3.6 um` (single frame) to `5.4 um` (median), and per-frame fixture texture partially averages away. Structural metrics are now referenced to a single frame (`7.bmp`).
3. The fixture texture mechanism was wrong. A fully coherent rough surface gives fully developed speckle (contrast about `0.92`), while the real fixture contrast is `0.59`; smoothing the roughness instead produced blurred blobs. The correct mechanism is partially developed speckle: averaging intensities over `K` independent rough-phase realizations (finite illumination spatial coherence). `K = 8` reproduces contrast `0.51` and correlation length `8 px` versus the real `0.59` / `13 px`. This is currently applied at comparison level; engine-level integration belongs to the noisy-config migration.

Current structural match (single-frame reference):

| Metric | Real `7.bmp` | Real median | Simulation |
| --- | ---: | ---: | ---: |
| Edge rise 10-90 (um) | 3.57 | 5.36 | 2.53 |
| Rim FWHM (um) | 14.5 | 14.0 | 7.0 |
| Rim p95 (DN) | 255 | 255 | 255 |
| Fixture texture corr length (px) | 13 | 14 | 8 |
| Fixture contrast | 0.59 | 0.53 | 0.51 |
| Interior median (DN) | 2.0 | 3.0 | 5.8 |

The simulation is no longer blurrier than the real capture; its edge is now slightly sharper than the single-frame real edge. Honest remaining gaps: the real seam band is about twice as wide as the simulated ring and carries its own speckle texture; the fixture has a large-scale illumination gradient (angular anisotropy) that the model still lacks; per-lens defects are absent by design in the clean reference; and the absolute profile RMSE (about `41 DN`) is dominated by the fixture-region radial pattern, not by the lens aperture.

### Fourth pass: engine-level partial coherence, seam width, illumination dipole

The remaining sim-real gap items were moved into the engine and re-swept:

1. `reflectance.speckle_realizations`: intensities are averaged over `K` independent rough-phase realizations inside the engine (bundle frames share the same realization set, since the fixture is a shared physical object). This replaces the comparison-level averaging of the third pass.
2. `reflectance.rim_width_px`: the seam scattering annulus widened from `3` to `10 px`; the simulated rim FWHM moves from `7.0` to `10.1 um` against the real `14.5 um`.
3. `reflectance.illumination_tilt_strength/angle`: a deterministic illumination dipole applied to the scattering regions only (the smooth lens return sits below the camera floor, so the dipole must not inject an amplitude gradient into the lens interior). With strength `0.25` at `180 deg`, the simulated rim dipole amplitude is `0.29` against `0.25` for the real median; the single-frame dipole (`0.72` for `7.bmp`) is stronger and frame-specific, consistent with the per-frame drift seen in the acquisition inference (`h1` concentration `0.54`).

Measured against the single-frame reference after this pass: edge rise `2.1` vs `3.6 um`, rim FWHM `10.1` vs `14.5 um`, rim p95 `255` vs `255`, fixture correlation length `9` vs `13 px`, fixture contrast `0.49` vs `0.59`, rim dipole `0.29` vs `0.25` (median). The rim angular CV remains lower than the single-frame value (`0.10` vs `0.62`) because per-frame CV is dominated by frame-specific seam speckle and defects, which the clean reference intentionally omits; the noisy config carries the frame-to-frame ranges instead.

`reflection_microlens520_noisy.json` has been rewritten in dark-port terms: ranges centred on the fitted actual-config values for reflectance/dark-port/blur parameters, per-frame illumination dipole angle spread `120-240 deg` (matching the observed `h1` phase concentration), and a camera block consistent with the measured photon transfer (`saturation_level 45-90` intensity units mapping to `255 DN`, `photon_gain 6-12` photoelectrons per unit, i.e. about `0.46 DN/e-`). The stale bright-field-fitted `coherent_ghost` block was removed; the localized ghost must be re-fitted under the dark-port forward model before it returns to the noisy config.

### Fifth pass: asymmetric seam band

The real seam is a sharp inner edge with a scattering skirt extending outward over the substrate. `reflectance.rim_inner_width_px` / `rim_outer_width_px` implement this asymmetry (defaulting to the symmetric `rim_width_px`). With inner `4 px` and outer `24 px`, the simulated rim FWHM reaches `14.48 um` against the real `14.5 um`, profile RMSE drops from `32.6` to `25.7 DN`, and fixture contrast reaches `0.57` against `0.59`. Remaining residuals: interior median `5.6` vs `3.0 DN`, rim dipole `0.38` vs `0.25` (median reference), rim angular CV still below the single-frame value for the reasons above.

## Defect visibility under calibrated radiometry (2026-07-21)

`scripts/render_reflection_defect_visibility.py` injects Gaussian pits (`sigma 2 um`, depth swept `13-260 nm`) into the clean dark-port model with the exposure calibrated against the 24-frame median, and compares the differential response with the shot+quantization floor implied by the measured `0.46 DN/e-`:

- the response is linear at about `15.5 DN/um` of pit depth; even a full-wrap `260 nm` pit peaks at `4.0 DN`, below the `4.9 DN` (3-sigma) floor;
- extrapolated minimum visible smooth-pit depth under the current acquisition protocol: about `0.32 um` at `sigma 2 um`;
- the real anomalous frames show interior deviations of `120-140 DN` (`5.bmp`, `19.bmp`, `20.bmp`), i.e. 25-35x larger than any smooth pit could produce at this exposure.

Interpretation: at this port and exposure, defect visibility is dominated by local scattering (roughness, contamination, chips) that breaks the weak specular return, not by the height-induced phase gradient. Two consequences:

1. simulation-side: realistic synthetic defects for this instrument must modulate the scattering amplitude map, not only the height map; a pure height perturbation underestimates real defect contrast by more than an order of magnitude;
2. acquisition-side: quantitative reconstruction of smooth height defects from the dark port requires raising the interior exposure (or HDR/frame averaging) by roughly `30x`, since the current frames expose for the clipped rim while the lens interior sits at `2-6 DN`.

Artifacts are under `external_data/processed/wechat_2026-07_15-34/reflection_defect_visibility/`.

## Scattering-defect calibration (2026-07-21)

Following the visibility result, the engine gained an optional `extra_field_modifier` per frame (`simulate_capture` / `simulate_bundle`), a complex amplitude/phase map for defects that a pure height perturbation cannot represent. `scripts/render_reflection_scatter_defect_comparison.py` sweeps localized scattering defects (Gaussian support, `sigma 3 um`, correlated rough phase plus local amplitude change) through the calibrated dark-port radiometry and compares the differential peak with the real anomaly contrast (`122-140 DN` for `5/19/20.bmp`):

| Local amplitude factor | Rough phase (rad) | Peak diff (DN) |
| ---: | ---: | ---: |
| 4 | 0.3-2.4 | 18-45 |
| 8 | 0.6 | 100 |
| 8 | 1.2 | 150 |
| 16 | any | 251 (clipped) |

Reading: with the lens specular amplitude at `0.1` (fixture `4.0`), real-contrast defects correspond to local scattering amplitudes around `0.8-1.6` absolute (factor `8-16` over the smooth lens surface, i.e. `20-40%` of the fixture's scattering strength) with rough phase near `0.6-1.2 rad`. The saturation of the `x16` row shows the real `140 DN`-class anomalies sit below the clipping regime, bounding the plausible defect scattering amplitude from above. These ranges are the recommended defect-injection parameters for future dark-port synthetic datasets: a scattering component in this range must accompany any height component, whose own contribution is bounded at about `15.5 DN/um`.

Artifacts are under `external_data/processed/wechat_2026-07_15-34/reflection_scatter_defect_comparison/`.

## Sixth pass: lens interior micro-structure (2026-07-21)

A realism-ordered review of the remaining appearance gaps started with the faint "concentric ripples" in the real lens interiors. Three measurements settled their identity:

1. Common-mode test: single-frame interior radial-profile fluctuation is `0.43 DN`, the 24-frame median retains only `0.05 DN`, and pairwise frame correlation is `0.03`. The texture is per-lens random, not a shared mold signature or an instrument artifact.
2. Height bound: injecting sinusoidal concentric height ripples through the calibrated forward model shows the real fluctuation level admits at most `~5-10 nm` ripple amplitude at the observed `~10 um` period; the texture carries essentially no reconstructable height information.
3. Sector-coherence test (the reversal): detrended radial profiles of eight angular sectors correlate at only `0.02-0.11` in real frames - the real texture is isotropic, not concentric. The simulation scored `0.989`: the rings were in the model, not in the data, caused by the too-strong smooth specular residual (`lens_amplitude 0.1`) diffracting through the pupil.

Engine consequences, all under `reflectance`:

- `lens_amplitude` tightened to `0.02` (the specular ring pattern must stay below the floor);
- `lens_scatter_amplitude`/`lens_scatter_phase_rad`/`lens_texture_sigma_px`: a weak additive rough-phase scattering component of the cap surface (micro-roughness / contamination film) that coexists with the specular return;
- `lens_point_scatter_count`/`lens_point_scatter_amplitude`: sparse point scatterers (dust, micro-pits); positions are drawn once per capture (a fixed property of the lens under test) while their speckle phase varies per coherence realization;
- `rim_phase_rough_rad`: the seam roughness is now independent of the background roughness (a chipped seam is many radians rough).

Calibrated point (`lens 0.02 / scatter 0.05 @ 4 px / 240 points @ 1.0`): interior median `3.8 DN` (real `2-3`), interior p99.9 `14 DN` (real clean frames `8-11`), sector coherence `0.05` (real `0.02-0.11`), radial-profile fluctuation `0.12-0.24 DN` (real `0.43`, which includes real dust/defects). The refreshed full comparison gives aperture corr `0.788`, rim FWHM `11.8 um`, fixture contrast `0.54`. The former artificial ring structure and most of the dipole seam are gone from the log view; the remaining texture-density difference between simulated point specks and the real finer-grained interior is the next candidate, together with the fixture two-component texture.

## Generalization check (2026-07-21)

`scripts/validate_reflection_generalization.py` upgrades the in-sample fits to held-out validation, using a denser defocus/blur grid (9 x 7 candidates).

Global defocus/blur selection:

| Protocol | Held-out radial RMSE | Notes |
| --- | ---: | --- |
| In-sample, 24-frame median | 0.102 | reference |
| Random 12/12 splits (12 seeds) | 0.105 mean | selection regret `0.001` RMSE |
| Leave-one-out (per frame) | 0.115 mean | profile corr mean `0.977`, min `0.934` |

The selected candidate is `defocus=-100` with `blur=10` in 32 of 36 selections and `blur=8` in the remaining 4; the optimum is a flat plateau between blur 8 and 10 at `defocus=-100`. The selection is a stable instrument-level property of this capture batch, not a per-frame artifact. Per-frame leave-one-out residuals are dominated by frame idiosyncrasy (worst: `15.bmp`, `16.bmp`, `21.bmp`), consistent with the empirical anomaly ranking rather than with fit overfitting.

Localized ghost cross-frame transfer (fit on one frame, evaluate on the other):

| Direction | Energy-map corr | Localized spectrum corr |
| --- | ---: | ---: |
| `13 -> 13` (in-sample) | 0.759 | 0.533 |
| `13 -> 14` (transfer) | 0.746 | 0.506 |
| `14 -> 14` (in-sample) | 0.767 | 0.513 |
| `14 -> 13` (transfer) | 0.698 | 0.514 |

Transfer degradation is small (energy corr drops by at most `0.06`, spectrum corr is essentially unchanged). The fitted parasitic path is shared between the two affected frames, which supports an instrument-level ghost interpretation rather than per-lens coincidence. Since these are 24 distinct lenses, the two-frame agreement cannot yet distinguish a fixed instrument path from a batch-level surface feature; more affected frames or a port/illumination confirmation would settle this.

Artifacts are under `external_data/processed/wechat_2026-07_15-34/reflection_generalization_check/`.

## Data-driven acquisition inference (2026-07-21)

`scripts/infer_acquisition_parameters.py` estimates acquisition parameters directly from the 24 raw BMP frames, reducing the lab calibration checklist. Findings:

1. Exposure protocol: the fixture background level is stable across all 24 frames (median `84 DN`, CV `6.4%`) while lens interiors vary strongly (CV `49%`, sample-dependent). Exposure/gain were effectively fixed for the whole session.
2. Saturation: the bright rim clips at `255 DN` (p95) in every frame; lens interiors sit at only `2-6 DN`. The rim level used by the robust normalisation is therefore clipping-limited, and rim radiometry is not usable beyond shape.
3. Sensor response: patch-level noise variance rises linearly with patch mean (`R^2 = 0.988`), so the BMP encoding is linear (no gamma), shot-noise dominated, with an approximate conversion gain of `0.46 DN/e-`, i.e. only about `550` photoelectrons at full scale. The hand-set `photon_gain` range `[8000, 20000]` in `reflection_microlens520_noisy.json` is likely 1-2 orders of magnitude too optimistic and must be recalibrated in the nuisance-calibration step.
4. Illumination anisotropy: the rim angular profile has a strong dipole component (mean relative amplitude `0.42`) whose phase is partially fixed in camera coordinates (circular concentration `0.54` versus about `0.20` for random, mean direction about `180 deg`, dark notch near `200-230 deg`). A fixed instrument/fixture illumination asymmetry coexists with per-frame sample variation; this shared mean profile is the fitting target for directional-illumination modelling.
5. Shear doubling: autocorrelation of high-passed central crops shows no secondary peak anywhere in `8-450 px`. With the dark port now confirmed, this non-detection constrains the shear: an object-plane `10 um` split (`38-57` native px depending on the magnification reading) should double every sharp defect visibly, which is not observed; a detector-plane `10 um` split (about `1.1 um` object plane, `4-6` native px) would be unresolvable. The evidence therefore favours the detector-plane reading of the stated shear, pending lab confirmation.
6. Coherence bound: the localized stripes (period about `4.4 um`) persist over about `160 um` of support, implying an optical path difference span of about `19 um` and hence source coherence length of at least `19 um`, i.e. bandwidth at most about `14 nm` at `520 nm`. The source is laser/SLD-class or narrowband-filtered, not a broadband LED.
7. Capture timeline (file mtimes): frames were taken in numbered groups of four over about 3.5 hours (`21-24, 13-16, 9/5/10-12/6-8, 17-20, 1-4`). The two stripe-anomaly frames `13.bmp` and `14.bmp` are consecutive captures, further supporting a shared instrument state rather than per-lens coincidence for the ghost.
8. Aperture radius spread: detected radii `612-636 px` give a relative spread of `3.9%`, an upper bound mixing batch tolerance with detection error.

Artifacts are under `external_data/processed/wechat_2026-07_15-34/acquisition_inference/`.

## Common/individual decomposition and the seam-band mechanism (2026-07-21)

`scripts/analyze_reflection_common_individual.py` splits the 24 registered frames into a common component (deterministic-model target) and per-frame individual components (nuisance-distribution target). Zone-wise (DN): interior common level `3.0` with structured std `0.6` versus individual RMS `5.9` (per-lens dust/defects dominate); arc zone common structure `17.5` versus individual `3.9` (the common-mode bands dominate, sector coherence `0.95`); seam common `63.6` and individual `48.4` (shared ring plus per-frame seam speckle, residual sector coherence `0.11`); fixture common `22.1` versus individual `46.5` (each frame images a different fixture patch). Per-frame dipole concentration `0.51`. The four frames with the largest individual energy are `21/24/23/22` - exactly the first four captures of the session, indicating an instrument settling period; a warm-up phase is recommended for future differential acquisitions.

The arc characterisation was corrected: 2-D spectra of the high-passed seam neighbourhood show the dominant band period is about `13 um` with the wavevector perpendicular to the seam; the earlier `2-3 um` quasi-period was a detrending artifact.

`scripts/render_reflection_seam_band_test.py` then tested the band mechanism and settled the shear-plane question. A pure x-shear doubling of the coherent seam ring (the object-plane `10 um` reading, `21.3` simulation px) predicts strongly anisotropic bands - strength peaking where the seam normal is parallel to the shear axis and vanishing at the top/bottom sectors. Measured sector anisotropy: real median `0.39` (nearly isotropic, no nulls), simulated object-plane shear `1.05-1.06` with `8-9.5 DN` peaks at the left/right sectors versus `0.3 DN` nulls. The object-plane reading is rejected; the camera-plane reading (`shear_px 1.573`, sector anisotropy `0.68-0.70`, band levels `0.1-0.6 DN` versus real `0.4-1.3 DN`) remains the working value. A first attempt at a focused coherent seam ring (`rim_coherent_amplitude 2.0`) did not reproduce the extra real fringe cycle, which points to the one depth ingredient the engine still lacks: the seam/fixture plane lies out of focus relative to the cap (the same mechanism that should produce the smooth out-of-focus character of the real fixture texture). A two-depth-plane propagation (lens field at focus, seam/fixture field defocused, coherent sum before the dark port) is the next realism item.

Artifacts are under `external_data/processed/wechat_2026-07_15-34/reflection_common_individual/` and `reflection_seam_band_test/`.

## Seam-band mechanism resolved: seam relief, not interference (2026-07-21)

Two internal falsification tests settled the mechanism of the common-mode bands:

1. Point scatterers (dust specks, four above `15 DN` in `7.bmp`) carry no flanking rings (`0.17 DN` oscillation in the `8-30 um` neighbourhood), which eliminates every generic optical origin (PSF sidelobes, coherent edge waves, partial-coherence effects) - those would decorate every bright feature equally.
2. The large moulded pit in `5.bmp` does carry flanking bands (`4.2 DN`, same `10-15 um` scale as the seam bands): features with surface relief have bands, features without relief have none.

Quantitative eliminations recorded for the alternatives: large-defocus Fresnel fringes need `~325 um` of seam defocus, contradicting the measured `3.6 um` edge sharpness; in-focus ring interference and PSF ringing predict `lambda/NA`-scale (`0.6-1.2 um`) spacing, an order of magnitude too fine; the x-shear double image fails the isotropy test (previous section); per-lens tooling marks fail manufacturing logic and two-sidedness.

Conclusion: the bands are the dark-port slope rendering of the real seam relief - a moulding fillet/meniscus with an annular shoulder, `10-15 um` wide, shared by all lenses of the batch; pit-type defects likewise carry raised rims. This is deterministic object geometry, not an instrument nuisance and not interference.

Implementation: `SimulationConfig` gained `seam_fillet_width_um`, `seam_shoulder_height_um/offset/width` and `seam_trench_depth/offset/width`; `spherical_cap` applies the fillet by radially localized smoothing of the profile plus the shoulder/trench annuli (defaults keep the sharp-corner behaviour bit-identical). A first render (`scripts/render_reflection_seam_fillet_test.py`) confirms the mechanism: with `fillet 2 um + shoulder 0.35 um at +6 um`, the two-sided band layout appears in the high-passed seam patch and the seam profile correlation to the real median improves from `0.68` to `0.75`. The precise profile is not yet calibrated - the current model saturates a `20 um` wide skirt where the real profile peaks at `204 DN` and decays smoothly, and hand-tuning six seam parameters plus the rim scattering amplitude is inefficient; a small joint optimisation of the seam-relief and rim-amplitude parameters against the median seam profile is the designated next step.

Artifacts are under `external_data/processed/wechat_2026-07_15-34/reflection_seam_fillet_test/`.

### Joint seam-profile fit

`scripts/fit_reflection_seam_profile.py` fits the ten seam parameters (relief and rim scattering) against the real median radial profile with fixed exposure calibration (random search plus Nelder-Mead, 151 evaluations). Result: profile RMSE `59.1 -> 48.9 DN`, correlation `0.864`; the fitted relief is physically plausible (fillet `4.3 um`, shoulder `0.10 um` at `+7.8 um`, trench `0.26 um` at `-4.9 um`, sharp inner rim edge, `16 um` outer skirt) and the decay flank `+10..+33 um` now tracks the real curve. Both configs carry the fitted values.

The dominant remaining seam residual is now diagnostic rather than radial: the simulated profile rail-clips at `255` over `0..+8 um` while the real median peaks at `204` - because the real seam is angularly nonuniform (only part of the circle saturates at a given radius, so the angular median stays below clipping), whereas the simulated seam is angularly uniform. The missing piece is the angular variation of the seam scattering (manufacturing nonuniformity coupled with the illumination dipole), which is the same known gap as the rim angular CV deficit.

## Apparatus correspondence (PRA 106, 023518)

The instrument follows the general weak-measurement differential imaging scheme: preselection, a weak shear coupling `U = exp(-i k_d A delta/2)`, and a postselection on the ancilla. The mapping to the current simulator is:

| Apparatus concept | Simulator counterpart | Status |
| --- | --- | --- |
| Input field `phi(r)` carrying sample phase | `amplitude * exp(i * phase_scale * h)` in `optical_leakage_lite` | implemented |
| Weak shear `delta` (Wollaston split approx. 20 um class) | `shear_px` sheared-difference channels | implemented, unit conversion pending pixel-scale confirmation |
| Dark-port output `E_o ~ delta * dE/dr_d` (postselection orthogonal) | `I_x`/`I_y = |U(r+d)-U(r-d)|^2` | matches the ideal dark-port limit |
| Bright-port / preselected output | `I_raw = |U|^2` | assumed source of the 24 BMP frames |
| 4f relay (L3, L4) and finite pupil | coherent circular pupil with cutoff `NA/lambda` | implemented |
| Non-ideal BS split / imperfect postselection extinction | not modelled; candidate nuisance for the left-right asymmetry | open |
| Interferometer parasitic paths (Sagnac-type double pass, prism secondary reflections) | localized coherent ghost components | phenomenological stand-in |

Two model consequences follow. First, the dark-port differential output is quadratic in the shear (`(g k)^2` intensity compression), so the DIC channels are low-intensity and noise-sensitive; camera noise ranges must be calibrated per port, not shared. Second, imperfect postselection extinction adds a coherent leakage of the undifferentiated field into the dark port, which is a physically motivated nuisance term that the current engine does not yet expose.

## Calibration checklist for the lab

Confirmed by the lab (2026-07-21): sag `15.34 um`, curvature radius `400 um`, source centre wavelength `520 nm`, `NA 0.42`; semiconductor laser with `2 nm` bandwidth; camera pixel pitch `2.4 um`; system magnification `9.09x`; shear magnitude `10 um` (reference plane unconfirmed); captures at the postselected dark port; 24 frames are 24 distinct lenses; two-axis `I_x`/`I_y` captures planned next.

Estimated from the data (see the acquisition inference section): fixed exposure protocol, linear sensor encoding with conversion gain about `0.46 DN/e-`, no detectable shear doubling (favouring the detector-plane shear reading), coherence bound consistent with the `2 nm` bandwidth, a partially fixed illumination dipole near `180 deg`, capture timeline, aperture-radius spread upper bound `3.9%`.

Open items, ordered by how hard they block the modelling:

1. confirm the magnification bookkeeping externally (tube-lens focal length or array pitch); internal evidence has settled the working scale at `0.177 um/px` / `13.56x` (see the magnification paragraph in the scope section), so this is verification, not a blocker.
2. shear reference plane for the stated `10 um` (object plane: `56.5` native px, ruled out by the absent defect doubling; camera plane: `0.74 um` object, `1.57` sim px, currently adopted) and the shear direction convention relative to the camera axes.
3. postselection extinction ratio and any intentional offset from exact orthogonality (sets the dark-port coherent leakage floor; candidate origin of the fixed `180 deg` dipole).
4. illumination geometry (coaxial epi / ring / oblique) and uniformity.
5. lens material, front-surface reflectance and substrate/fixture materials, needed for the amplitude/reflectance map that the dark-port model requires.
6. the planned `reference / standard / test` grouping for the differential experiments (same lens before/after, or cross-sample as now).
7. exposure time and analog gain (optional, for absolute photon calibration of the `0.46 DN/e-` estimate).

Cheap high-value extras when convenient: a few flat-field frames without sample and dark frames; several repeat captures of one lens without remounting; one or two lenses with independent WLI/profilometer topography.
