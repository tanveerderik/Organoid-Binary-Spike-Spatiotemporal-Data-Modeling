# Interpretable diagnostics -- appendix

Supporting detail for `diagnostics.md`: full paired-test listings, the conditioning ladders behind the collapsed generation table, and measurement provenance. Same runs, same numbers.

Pinned evaluation protocol. task axis: 70 batches, 8 MC samples, seed 20260822; everything else: 70 batches, seed 20260821. The test loader is not shuffled, so a batch count is a prefix of the split and every model within a budget sees identical clips. Never carry a number between budgets.

† **3D U-Net (det.) is not a representation learner, and its reconstruction numbers are not comparable to a tokenizer's.** It is a U-Net: `forward` concatenates the full-resolution stem output back in on the way up, so an uncompressed path runs from input to output and the decoder reads AROUND the compressed layer. Per clip it carries 54.8M floats across its skips against an input of 1.29M binary voxels -- the full-resolution skip alone holds 32x more floats than the volume has voxels. It produces **no codebook, no discrete index, and no reusable latent**: nothing is shared across clips, nothing is indexable, and there is no bottleneck a prior could be trained over. Our alphabet is 1024 tokens over V=961, i.e. 1.24 KB per clip. The skips are what win it the ranking columns and are precisely what disqualify it as a tokenizer; the two cannot be had together, because a tokenizer's value comes from forcing everything through the code.

## Task axis -- provenance

**`recon (0)` is not a completion task.** Its mask spec yields an all-TRUE ROI, so nothing is visible and there is nothing to complete. Verified, not assumed: on a shared batch and seed the pipeline's task-0 output is BITWISE identical to the shipped free-generation path, as is MaskGIT-flat's, and with a full ROI both the GLM's and MaskGIT-flat's `complete()` return the same volume for the true clip and for a random one -- clip-independent, i.e. free generation. Dich. Gaussian has no completion mechanism, so all four of its columns are free generation.

The null rows are one canonical block (`task_eval_nulls.json`), computed once with no model loaded. They contain no model and must not vary by model -- they did, by up to 5%, because the shared train loader takes its shuffle order from the global RNG and reuses persistent workers whose per-item crop RNG had already been advanced by whatever was built first. `train_site_maps` now draws on its own single-process seeded loader, verified byte-identical across processes. Scored clips were never affected: the test split is a `DeterministicSubset`.


### Site-level AP ↑

**Which ELECTRODE fires at all, ignoring when.** Time is collapsed inside the hole: a candidate is one (y, x) electrode touched by the ROI, a positive is an electrode that spikes in at least one hole frame, and an arm's score for it is its mean field over the hole frames. Together with the table above this splits an arm's performance into WHERE and WHEN -- an arm that finds the right electrodes but the wrong frames scores well here and badly above, which is the signature of a sample being graded by a metric that wants a posterior. Not redundant with the spatial-support violation in the battery: that statistic is MARGINAL -- it asks only whether spikes land on electrodes that are ever active, and saturates at 1.0000 for any arm whose output IS a site map. Only AP is paired to the clip being completed.

| model ↑ | recon (0) = free gen | causal (1) | noncausal (2) | spatial (3) |
|---|---|---|---|---|
| Ours (4C+soft) | 0.2244 | 0.2170 | 0.1889 | 0.3092 |
| MaskGIT-flat | 0.1319 | 0.1299 | 0.1406 | 0.1195 |
| 3D U-Net (det.)† | **0.3412** | **0.3297** | **0.3057** | **0.3342** |
| 3D CVAE | 0.2709 | 0.2368 | 0.2127 | 0.2417 |
| | | | | |
| _ref_ Dich. Gaussian | 0.6997 | 0.5875 | 0.5302 | 0.7392 |
| _ref_ Coupled GLM | 0.6984 | 0.5938 | 0.5375 | 0.7402 |
| _ceiling_ Ours (4C+soft) true tokens | 0.2296 | 0.2584 | 0.2359 | **0.3193** |
| _ceiling_ MaskGIT-flat true tokens | **0.2619** | **0.2724** | **0.2777** | 0.2783 |
| | | | | |
| _null_ assay site map, SEEN | **0.6796** | **0.5714** | **0.5184** | **0.7224** |
| _null_ assay site map, UNSEEN | 0.0284 | 0.0230 | 0.0215 | 0.0350 |
| _null_ visible profile x site map | -- | 0.5653 | **0.5184** | 0.6873 |
| _null_ persistence | -- | 0.1567 | 0.2031 | -- |

**Rows.** Bold marks the best MODEL in each column; within the ceiling and null groups it marks the highest value, i.e. the hardest bar in that column, not a compliment to the arm.

- `_chance_` -- expected AP of a uniformly random ranking, which equals the fraction of candidates that are positive. One candidate is one electrode touched by the ROI. Every model number is given as a multiple of this.
- `_ceiling_ ... true tokens` -- that model's OWN tokenizer handed the TRUE codes for the hole and asked to decode them. It is what the model would score if its prior named every hidden token correctly, so the gap below it is the PRIOR's failure and the gap above it is the ALPHABET's. Per model, not shared: the two tokenizers differ and the distance between their ceilings is itself a result. DG and the GLM are not tokenizers and have none.
- `_null_ assay site map, SEEN` -- the clip's own assay's per-site firing rate, measured on that assay's TRAIN clips and held constant in time. No model, no completion -- pure lookup. This is what DG and the GLM reproduce.
- `_null_ assay site map, UNSEEN` -- the same lookup with the clip's own assay WITHHELD, averaged over the other 30. What a site map is worth without having seen this preparation; the gap to the row above it is how much of the SEEN row is memorisation.
- `_null_ visible profile x site map` -- rank-1 separable: the clip's own VISIBLE frames' activity profile times the assay train site map. The honest 'marginals plus whatever you can see' competitor -- it uses no hidden voxel. n/a on `recon`, where nothing is visible.
- `_null_ persistence` -- copy the visible frames adjacent to the hole. Defined only for the two temporal tasks; `spatial`'s hole spans every frame and `recon` has no visible frame, so both are n/a rather than faked.


### Model vs null, paired

Wilcoxon signed-rank on per-clip differences, every clip under every task. `delta` is mean(model_mc - null); positive means the model wins. q is Benjamini-Hochberg across all 80 tests in this table.

| metric | model | vs null | task | delta | q |
|---|---|---|---|---|---|
| voxel | Ours (4C+soft) | SEEN | recon (0) = free gen | -0.0544 | 1.9e-46 |
| voxel | Ours (4C+soft) | SEEN | causal (1) | -0.0558 | 6.5e-44 |
| voxel | Ours (4C+soft) | SEEN | noncausal (2) | -0.0565 | 1.7e-44 |
| voxel | Ours (4C+soft) | SEEN | spatial (3) | -0.0634 | 1.4e-37 |
| voxel | Ours (4C+soft) | UNSEEN | recon (0) = free gen | **+0.0105** | 8.3e-46 |
| voxel | Ours (4C+soft) | UNSEEN | causal (1) | **+0.0116** | 1.6e-46 |
| voxel | Ours (4C+soft) | UNSEEN | noncausal (2) | **+0.0124** | 1.1e-45 |
| voxel | Ours (4C+soft) | UNSEEN | spatial (3) | **+0.0167** | 1.2e-37 |
| voxel | 3D U-Net (det.)† | SEEN | recon (0) = free gen | -0.0402 | 1.5e-46 |
| voxel | 3D U-Net (det.)† | SEEN | causal (1) | -0.0336 | 5.9e-45 |
| voxel | 3D U-Net (det.)† | SEEN | noncausal (2) | -0.0352 | 1.1e-40 |
| voxel | 3D U-Net (det.)† | SEEN | spatial (3) | -0.0579 | 1.6e-38 |
| voxel | 3D U-Net (det.)† | UNSEEN | recon (0) = free gen | **+0.0246** | 6.3e-46 |
| voxel | 3D U-Net (det.)† | UNSEEN | causal (1) | **+0.0338** | 1.5e-46 |
| voxel | 3D U-Net (det.)† | UNSEEN | noncausal (2) | **+0.0337** | 9.8e-46 |
| voxel | 3D U-Net (det.)† | UNSEEN | spatial (3) | **+0.0222** | 4e-41 |
| voxel | 3D CVAE | SEEN | recon (0) = free gen | -0.0420 | 5.7e-46 |
| voxel | 3D CVAE | SEEN | causal (1) | -0.0442 | 8.3e-46 |
| voxel | 3D CVAE | SEEN | noncausal (2) | -0.0462 | 3.8e-45 |
| voxel | 3D CVAE | SEEN | spatial (3) | -0.0625 | 6.4e-41 |
| voxel | 3D CVAE | UNSEEN | recon (0) = free gen | **+0.0228** | 8.3e-46 |
| voxel | 3D CVAE | UNSEEN | causal (1) | **+0.0231** | 1.9e-46 |
| voxel | 3D CVAE | UNSEEN | noncausal (2) | **+0.0226** | 9.8e-46 |
| voxel | 3D CVAE | UNSEEN | spatial (3) | **+0.0175** | 6.4e-41 |
| voxel | Dich. Gaussian | SEEN | recon (0) = free gen | -0.0126 | 1.6e-36 |
| voxel | Dich. Gaussian | SEEN | causal (1) | -0.0121 | 6.6e-26 |
| voxel | Dich. Gaussian | SEEN | noncausal (2) | -0.0108 | 5.1e-22 |
| voxel | Dich. Gaussian | SEEN | spatial (3) | -0.0152 | 4.1e-20 |
| voxel | Dich. Gaussian | UNSEEN | recon (0) = free gen | **+0.0522** | 1.5e-46 |
| voxel | Dich. Gaussian | UNSEEN | causal (1) | **+0.0553** | 1.5e-46 |
| voxel | Dich. Gaussian | UNSEEN | noncausal (2) | **+0.0581** | 9.8e-46 |
| voxel | Dich. Gaussian | UNSEEN | spatial (3) | **+0.0648** | 2.4e-41 |
| voxel | Coupled GLM | SEEN | recon (0) = free gen | **+0.0018** | 0.00047 |
| voxel | Coupled GLM | SEEN | causal (1) | **+0.0043** | 3.8e-05 |
| voxel | Coupled GLM | SEEN | noncausal (2) | **+0.0058** | 6.4e-08 |
| voxel | Coupled GLM | SEEN | spatial (3) | -0.0029 | 0.42 _(n.s.)_ |
| voxel | Coupled GLM | UNSEEN | recon (0) = free gen | **+0.0666** | 2.3e-46 |
| voxel | Coupled GLM | UNSEEN | causal (1) | **+0.0716** | 1.5e-46 |
| voxel | Coupled GLM | UNSEEN | noncausal (2) | **+0.0746** | 9.8e-46 |
| voxel | Coupled GLM | UNSEEN | spatial (3) | **+0.0772** | 2.2e-41 |
| site | Ours (4C+soft) | SEEN | recon (0) = free gen | -0.4552 | 1.5e-46 |
| site | Ours (4C+soft) | SEEN | causal (1) | -0.3543 | 1.3e-45 |
| site | Ours (4C+soft) | SEEN | noncausal (2) | -0.3295 | 1.1e-45 |
| site | Ours (4C+soft) | SEEN | spatial (3) | -0.4132 | 1.1e-35 |
| site | Ours (4C+soft) | UNSEEN | recon (0) = free gen | **+0.1960** | 1.5e-46 |
| site | Ours (4C+soft) | UNSEEN | causal (1) | **+0.1940** | 1.5e-46 |
| site | Ours (4C+soft) | UNSEEN | noncausal (2) | **+0.1673** | 9.8e-46 |
| site | Ours (4C+soft) | UNSEEN | spatial (3) | **+0.2743** | 2.4e-41 |
| site | 3D U-Net (det.)† | SEEN | recon (0) = free gen | -0.3384 | 1.6e-46 |
| site | 3D U-Net (det.)† | SEEN | causal (1) | -0.2417 | 8.6e-46 |
| site | 3D U-Net (det.)† | SEEN | noncausal (2) | -0.2127 | 2.9e-45 |
| site | 3D U-Net (det.)† | SEEN | spatial (3) | -0.3882 | 1.4e-39 |
| site | 3D U-Net (det.)† | UNSEEN | recon (0) = free gen | **+0.3128** | 1.5e-46 |
| site | 3D U-Net (det.)† | UNSEEN | causal (1) | **+0.3066** | 1.5e-46 |
| site | 3D U-Net (det.)† | UNSEEN | noncausal (2) | **+0.2842** | 6.3e-46 |
| site | 3D U-Net (det.)† | UNSEEN | spatial (3) | **+0.2992** | 2.4e-41 |
| site | 3D CVAE | SEEN | recon (0) = free gen | -0.4087 | 1.5e-46 |
| site | 3D CVAE | SEEN | causal (1) | -0.3345 | 2.5e-46 |
| site | 3D CVAE | SEEN | noncausal (2) | -0.3057 | 1.5e-46 |
| site | 3D CVAE | SEEN | spatial (3) | -0.4807 | 2.6e-41 |
| site | 3D CVAE | UNSEEN | recon (0) = free gen | **+0.2425** | 1.5e-46 |
| site | 3D CVAE | UNSEEN | causal (1) | **+0.2138** | 2.8e-46 |
| site | 3D CVAE | UNSEEN | noncausal (2) | **+0.1911** | 9.8e-46 |
| site | 3D CVAE | UNSEEN | spatial (3) | **+0.2067** | 4e-41 |
| site | Dich. Gaussian | SEEN | recon (0) = free gen | **+0.0201** | 2.5e-09 |
| site | Dich. Gaussian | SEEN | causal (1) | **+0.0161** | 1.3e-06 |
| site | Dich. Gaussian | SEEN | noncausal (2) | **+0.0119** | 0.00096 |
| site | Dich. Gaussian | SEEN | spatial (3) | **+0.0168** | 7.7e-05 |
| site | Dich. Gaussian | UNSEEN | recon (0) = free gen | **+0.6713** | 1.5e-46 |
| site | Dich. Gaussian | UNSEEN | causal (1) | **+0.5645** | 1.5e-46 |
| site | Dich. Gaussian | UNSEEN | noncausal (2) | **+0.5087** | 2.3e-46 |
| site | Dich. Gaussian | UNSEEN | spatial (3) | **+0.7042** | 2.2e-41 |
| site | Coupled GLM | SEEN | recon (0) = free gen | **+0.0187** | 1.7e-08 |
| site | Coupled GLM | SEEN | causal (1) | **+0.0225** | 1.7e-08 |
| site | Coupled GLM | SEEN | noncausal (2) | **+0.0191** | 3.4e-07 |
| site | Coupled GLM | SEEN | spatial (3) | **+0.0178** | 0.00035 |
| site | Coupled GLM | UNSEEN | recon (0) = free gen | **+0.6700** | 1.5e-46 |
| site | Coupled GLM | UNSEEN | causal (1) | **+0.5708** | 1.5e-46 |
| site | Coupled GLM | UNSEEN | noncausal (2) | **+0.5159** | 2.3e-46 |
| site | Coupled GLM | UNSEEN | spatial (3) | **+0.7053** | 2.2e-41 |

### Per-task battery, canonical schema

Same three tables for every task. Bins are `DEFAULT_GAP_BINS` and features are `ACTIVITY_CTX_NAMES`, both from `utils/constants.py`; the adjacency rate is `_hard_gap_rates` and the violation fraction is `evaluate_generation_global_metrics`, the same functions training reports, imported rather than reimplemented.

**Readout: rate-matched.** These three statistics are defined on BINARY spike trains, but every arm emits a continuous field, so a binarisation is required before any of them can be computed. Two choices exist and both are reported:

- **Rate-matched** (this section). Every arm is given the same spike budget -- the top `round(assay_train_rate x |ROI|)` voxels of its own ranking -- so all arms emit an equal number of spikes and the only thing that can differ is WHERE they go. This isolates placement from count, and it is what makes arms with incompatible score scales (our probabilities, MaskGIT's logits, the GLM's intensities) comparable at all. The budget is taken from TRAIN clips of the same assay, never from the held-out clip.

- **Own-count** (`Count calibration` above, and the appendix). Each arm sets its own budget from its own field. This measures the deployed system, count errors included, and is where a count head can earn credit.

Rate-matching is the control, not the headline claim: an arm that wins here wins on placement, and its count accuracy is reported separately rather than being allowed to leak into every distribution statistic at once.

Two consequences of rate-matching, true in all four blocks and so stated once here:

1. **`log_mean_firing_density` is not scored here.** It is `log(mean + 1e-6)` over a fixed volume, so it depends only on the spike COUNT -- and the readout sets the count, identically for every model. Measured directly on one batch (a separate check, not a number this report re-renders): the arms shared as little as 2% of their spikes while writing exactly the same number of them. The per-block figure below is the readout's count error, not a model's. See **Count calibration** for what the models themselves predict.

2. **A spatial consistency of 1.0000 is not a win.** An arm whose output IS a per-assay site map can only place mass where that map is already active, so it cannot violate the support by construction. REAL itself does not score 1.0.


#### recon (0) = free gen

Short-gap adjacency rate by gap bin ≈REAL

| gap (frames) | 1 | 2 | 3 | 4-6 | 7-12 | 13-24 | 25-48 |
|---|---|---|---|---|---|---|---|
| REAL | 0.03135 | 0.04928 | 0.05979 | 0.06333 | 0.05641 | 0.04956 | 0.04408 |
| Ours (4C+soft) | **0.22388** (7.14x) | **0.19429** (3.94x) | **0.22789** (3.81x) | 0.30295 (4.78x) | 0.20792 (3.69x) | 0.17654 (3.56x) | 0.15192 (3.45x) |
| MaskGIT-flat | 0.39631 (12.64x) | 0.35990 (7.30x) | 0.32455 (5.43x) | **0.28848** (4.56x) | **0.18307** (3.25x) | **0.11181** (2.26x) | **0.05582** (1.27x) |
| 3D U-Net (det.)† | 0.63975 (20.41x) | 0.59443 (12.06x) | 0.56588 (9.46x) | 0.55389 (8.75x) | 0.52298 (9.27x) | 0.38231 (7.71x) | 0.23383 (5.30x) |
| 3D CVAE | 0.77996 (24.88x) | 0.70842 (14.38x) | 0.71672 (11.99x) | 0.67546 (10.67x) | 0.58389 (10.35x) | 0.43149 (8.71x) | 0.19059 (4.32x) |
| | | | | | | | |
| _ref_ Dich. Gaussian | 0.20478 (6.53x) | 0.22133 (4.49x) | 0.23239 (3.89x) | 0.23865 (3.77x) | 0.21714 (3.85x) | 0.21097 (4.26x) | 0.21371 (4.85x) |
| _ref_ Coupled GLM | 0.81626 (26.04x) | 0.80050 (16.24x) | 0.78357 (13.10x) | 0.77484 (12.23x) | 0.77952 (13.82x) | 0.78394 (15.82x) | 0.81638 (18.52x) |

Spatial consistency ≈REAL  (1 - hard spatial-support violation fraction)

| model | spatial consistency | violation fraction |
|---|---|---|
| REAL | 0.9996 | 0.0004 |
| Ours (4C+soft) | 0.4732 | 0.5268 |
| MaskGIT-flat | 0.5539 | 0.4461 |
| 3D U-Net (det.)† | **0.6839** | 0.3161 |
| 3D CVAE | 0.5475 | 0.4525 |
| | | |
| _ref_ Dich. Gaussian | 1.0000 | 0.0000 |
| _ref_ Coupled GLM | 1.0000 | 0.0000 |

Local context, MAE against the clip's own lct ↓

| feature | Ours (4C+soft) | MaskGIT-flat | 3D U-Net (det.)† | 3D CVAE | _ref_ Dich. Gaussian | _ref_ Coupled GLM |
|---|---|---|---|---|---|---|
| log_mean_firing_density | **0.3440** | 0.3697 | **0.3440** | **0.3440** | 0.3440 | 0.3440 |
| var_x | **0.0175** | 0.0575 | 0.0542 | 0.0519 | 0.0253 | 0.0375 |
| var_y | **0.0241** | 0.0768 | 0.0572 | 0.0711 | 0.0308 | 0.0518 |
| var_t | **0.0476** | 0.1474 | 0.0800 | 0.0899 | 0.0770 | 0.0698 |
| cov_xy | **0.0109** | 0.0334 | 0.0521 | 0.0482 | 0.0179 | 0.0335 |
| cov_xt | **0.0174** | 0.0204 | 0.0239 | 0.0286 | 0.0175 | 0.0153 |
| cov_yt | **0.0187** | 0.0241 | 0.0278 | 0.0233 | 0.0200 | 0.0169 |
| active_site_ratio | 0.0359 | **0.0333** | 0.0557 | 0.0592 | 0.0397 | 0.0646 |
| temporal_trend | **0.1527** | 0.3139 | 0.4323 | 0.5441 | 0.2984 | 0.5479 |

#### causal (1)

Short-gap adjacency rate by gap bin ≈REAL

| gap (frames) | 1 | 2 | 3 | 4-6 | 7-12 | 13-24 | 25-48 |
|---|---|---|---|---|---|---|---|
| REAL | 0.03135 | 0.04928 | 0.05979 | 0.06333 | 0.05641 | 0.04956 | 0.04408 |
| Ours (4C+soft) | **0.12783** (4.08x) | **0.11728** (2.38x) | **0.14357** (2.40x) | 0.18257 (2.88x) | 0.11614 (2.06x) | 0.07807 (1.58x) | **0.03602** (0.82x) |
| MaskGIT-flat | 0.27063 (8.63x) | 0.24790 (5.03x) | 0.22100 (3.70x) | **0.18099** (2.86x) | **0.10162** (1.80x) | **0.05478** (1.11x) | 0.01508 (0.34x) |
| 3D U-Net (det.)† | 0.35465 (11.31x) | 0.31444 (6.38x) | 0.28873 (4.83x) | 0.25843 (4.08x) | 0.21410 (3.80x) | 0.12915 (2.61x) | 0.05394 (1.22x) |
| 3D CVAE | 0.43086 (13.74x) | 0.36485 (7.40x) | 0.33504 (5.60x) | 0.28415 (4.49x) | 0.20094 (3.56x) | 0.09540 (1.92x) | 0.02684 (0.61x) |
| | | | | | | | |
| _ref_ Dich. Gaussian | 0.13797 (4.40x) | 0.15096 (3.06x) | 0.15867 (2.65x) | 0.15993 (2.53x) | 0.14010 (2.48x) | 0.12188 (2.46x) | 0.09651 (2.19x) |
| _ref_ Coupled GLM | 0.48171 (15.37x) | 0.47167 (9.57x) | 0.45954 (7.69x) | 0.44211 (6.98x) | 0.39766 (7.05x) | 0.31097 (6.27x) | 0.17496 (3.97x) |

Spatial consistency ≈REAL  (1 - hard spatial-support violation fraction)

| model | spatial consistency | violation fraction |
|---|---|---|
| REAL | 0.9996 | 0.0004 |
| Ours (4C+soft) | 0.6953 | 0.3047 |
| MaskGIT-flat | 0.7082 | 0.2918 |
| 3D U-Net (det.)† | **0.8615** | 0.1385 |
| 3D CVAE | 0.7425 | 0.2575 |
| | | |
| _ref_ Dich. Gaussian | 0.9997 | 0.0003 |
| _ref_ Coupled GLM | 0.9997 | 0.0003 |

Local context, MAE against the clip's own lct ↓

| feature | Ours (4C+soft) | MaskGIT-flat | 3D U-Net (det.)† | 3D CVAE | _ref_ Dich. Gaussian | _ref_ Coupled GLM |
|---|---|---|---|---|---|---|
| log_mean_firing_density | **0.2491** | 0.3127 | **0.2491** | **0.2491** | 0.2491 | 0.2491 |
| var_x | **0.0104** | 0.0352 | 0.0323 | 0.0499 | 0.0163 | 0.0232 |
| var_y | **0.0172** | 0.0474 | 0.0388 | 0.0511 | 0.0194 | 0.0290 |
| var_t | **0.0441** | 0.0930 | 0.0778 | 0.1239 | 0.0509 | 0.0528 |
| cov_xy | **0.0075** | 0.0237 | 0.0282 | 0.0378 | 0.0120 | 0.0187 |
| cov_xt | **0.0156** | 0.0327 | 0.0308 | 0.0354 | 0.0154 | 0.0248 |
| cov_yt | **0.0187** | 0.0331 | 0.0336 | 0.0283 | 0.0211 | 0.0307 |
| active_site_ratio | **0.0151** | 0.0180 | 0.0184 | 0.0188 | 0.0179 | 0.0260 |
| temporal_trend | **0.1740** | 0.2239 | 0.1826 | 0.2365 | 0.2263 | 0.2569 |

#### noncausal (2)

Short-gap adjacency rate by gap bin ≈REAL

| gap (frames) | 1 | 2 | 3 | 4-6 | 7-12 | 13-24 | 25-48 |
|---|---|---|---|---|---|---|---|
| REAL | 0.03135 | 0.04928 | 0.05979 | 0.06333 | 0.05641 | 0.04956 | 0.04408 |
| Ours (4C+soft) | **0.10390** (3.31x) | **0.09962** (2.02x) | **0.12078** (2.02x) | 0.14062 (2.22x) | 0.08214 (1.46x) | 0.04029 (0.81x) | 0.03079 (0.70x) |
| MaskGIT-flat | 0.20282 (6.47x) | 0.18811 (3.82x) | 0.18355 (3.07x) | **0.13834** (2.18x) | **0.07814** (1.39x) | 0.04492 (0.91x) | 0.03678 (0.83x) |
| 3D U-Net (det.)† | 0.23518 (7.50x) | 0.19553 (3.97x) | 0.17615 (2.95x) | 0.14239 (2.25x) | 0.10718 (1.90x) | 0.06244 (1.26x) | **0.04473** (1.01x) |
| 3D CVAE | 0.27179 (8.67x) | 0.22750 (4.62x) | 0.20032 (3.35x) | 0.16372 (2.59x) | 0.11194 (1.98x) | **0.05288** (1.07x) | 0.03398 (0.77x) |
| | | | | | | | |
| _ref_ Dich. Gaussian | 0.10252 (3.27x) | 0.11730 (2.38x) | 0.12508 (2.09x) | 0.12419 (1.96x) | 0.10499 (1.86x) | 0.07872 (1.59x) | 0.06432 (1.46x) |
| _ref_ Coupled GLM | 0.34487 (11.00x) | 0.33772 (6.85x) | 0.32697 (5.47x) | 0.30073 (4.75x) | 0.23111 (4.10x) | 0.11851 (2.39x) | 0.07591 (1.72x) |

Spatial consistency ≈REAL  (1 - hard spatial-support violation fraction)

| model | spatial consistency | violation fraction |
|---|---|---|
| REAL | 0.9996 | 0.0004 |
| Ours (4C+soft) | 0.7812 | 0.2188 |
| MaskGIT-flat | 0.8158 | 0.1842 |
| 3D U-Net (det.)† | **0.9110** | 0.0890 |
| 3D CVAE | 0.8194 | 0.1806 |
| | | |
| _ref_ Dich. Gaussian | 0.9998 | 0.0002 |
| _ref_ Coupled GLM | 0.9998 | 0.0002 |

Local context, MAE against the clip's own lct ↓

| feature | Ours (4C+soft) | MaskGIT-flat | 3D U-Net (det.)† | 3D CVAE | _ref_ Dich. Gaussian | _ref_ Coupled GLM |
|---|---|---|---|---|---|---|
| log_mean_firing_density | **0.1710** | 0.2068 | **0.1710** | **0.1710** | 0.1710 | 0.1710 |
| var_x | **0.0079** | 0.0175 | 0.0216 | 0.0426 | 0.0125 | 0.0172 |
| var_y | **0.0129** | 0.0310 | 0.0278 | 0.0414 | 0.0143 | 0.0195 |
| var_t | **0.0338** | 0.0624 | 0.0471 | 0.0544 | 0.0388 | 0.0407 |
| cov_xy | **0.0076** | 0.0151 | 0.0209 | 0.0317 | 0.0091 | 0.0129 |
| cov_xt | **0.0116** | 0.0217 | 0.0191 | 0.0284 | 0.0115 | 0.0159 |
| cov_yt | **0.0141** | 0.0175 | 0.0229 | 0.0211 | 0.0149 | 0.0210 |
| active_site_ratio | **0.0102** | 0.0112 | 0.0103 | 0.0108 | 0.0126 | 0.0167 |
| temporal_trend | **0.1131** | 0.1546 | 0.1325 | 0.1292 | 0.1186 | 0.1305 |

#### spatial (3)

Short-gap adjacency rate by gap bin ≈REAL

| gap (frames) | 1 | 2 | 3 | 4-6 | 7-12 | 13-24 | 25-48 |
|---|---|---|---|---|---|---|---|
| REAL | 0.03135 | 0.04928 | 0.05979 | 0.06333 | 0.05641 | 0.04956 | 0.04408 |
| Ours (4C+soft) | **0.08430** (2.69x) | **0.07832** (1.59x) | **0.10379** (1.74x) | **0.13573** (2.14x) | **0.09085** (1.61x) | 0.07708 (1.56x) | 0.06278 (1.42x) |
| MaskGIT-flat | 0.17735 (5.66x) | 0.17211 (3.49x) | 0.16979 (2.84x) | 0.15066 (2.38x) | 0.09944 (1.76x) | **0.05989** (1.21x) | **0.03540** (0.80x) |
| 3D U-Net (det.)† | 0.29307 (9.35x) | 0.27811 (5.64x) | 0.27132 (4.54x) | 0.27292 (4.31x) | 0.25909 (4.59x) | 0.20614 (4.16x) | 0.15488 (3.51x) |
| 3D CVAE | 0.35874 (11.44x) | 0.33285 (6.75x) | 0.34197 (5.72x) | 0.32849 (5.19x) | 0.28771 (5.10x) | 0.22101 (4.46x) | 0.11680 (2.65x) |
| | | | | | | | |
| _ref_ Dich. Gaussian | 0.10486 (3.35x) | 0.11630 (2.36x) | 0.13040 (2.18x) | 0.13063 (2.06x) | 0.11926 (2.11x) | 0.11117 (2.24x) | 0.11758 (2.67x) |
| _ref_ Coupled GLM | 0.33024 (10.53x) | 0.32805 (6.66x) | 0.31678 (5.30x) | 0.30397 (4.80x) | 0.29439 (5.22x) | 0.28677 (5.79x) | 0.28689 (6.51x) |

Spatial consistency ≈REAL  (1 - hard spatial-support violation fraction)

| model | spatial consistency | violation fraction |
|---|---|---|
| REAL | 0.9996 | 0.0004 |
| Ours (4C+soft) | 0.7586 | 0.2414 |
| MaskGIT-flat | 0.7666 | 0.2334 |
| 3D U-Net (det.)† | **0.8165** | 0.1835 |
| 3D CVAE | 0.7383 | 0.2617 |
| | | |
| _ref_ Dich. Gaussian | 0.9773 | 0.0227 |
| _ref_ Coupled GLM | 0.9785 | 0.0215 |

Local context, MAE against the clip's own lct ↓

| feature | Ours (4C+soft) | MaskGIT-flat | 3D U-Net (det.)† | 3D CVAE | _ref_ Dich. Gaussian | _ref_ Coupled GLM |
|---|---|---|---|---|---|---|
| log_mean_firing_density | 0.3284 | **0.2568** | 0.3284 | 0.3284 | 0.3284 | 0.3284 |
| var_x | 0.0281 | **0.0267** | 0.0489 | 0.0449 | 0.0302 | 0.0321 |
| var_y | 0.0282 | **0.0282** | 0.0437 | 0.0551 | 0.0212 | 0.0257 |
| var_t | **0.0356** | 0.0658 | 0.0457 | 0.0563 | 0.0447 | 0.0483 |
| cov_xy | 0.0202 | **0.0192** | 0.0420 | 0.0448 | 0.0184 | 0.0210 |
| cov_xt | 0.0156 | **0.0141** | 0.0197 | 0.0298 | 0.0162 | 0.0229 |
| cov_yt | **0.0161** | 0.0256 | 0.0191 | 0.0292 | 0.0171 | 0.0223 |
| active_site_ratio | 0.0213 | **0.0194** | 0.0249 | 0.0261 | 0.0209 | 0.0274 |
| temporal_trend | **0.1025** | 0.1606 | 0.1403 | 0.2388 | 0.1221 | 0.1894 |

## Per-task battery under each model's OWN count

Readout `roi_topN_at_model_own_expected_count`: k is the arm's own ROI probability sum instead of the assay train rate. Same three tables as the headline battery, same canonical schema.

This is reported for completeness, not as the primary result. The probability sum is not a calibrated count -- it over-produces several-fold -- so these volumes are too dense and `lct` MAEs are WORSE here than under the shared readout even though the same model's count RANKING is near-perfect within an assay. See the count-calibration table in the headline report.


### recon (0) = free gen

Local context, MAE against the clip's own lct ↓  (own-count readout)

| feature | Ours (4C+soft) | MaskGIT-flat | 3D U-Net (det.)† | 3D CVAE | _ref_ Dich. Gaussian | _ref_ Coupled GLM |
|---|---|---|---|---|---|---|
| log_mean_firing_density | 1.9781 | 3.1868 | 0.7758 | **0.5177** | 4.8111 | 0.3378 |
| var_x | **0.0204** | 0.0657 | 0.0612 | 0.0544 | 0.1881 | 0.0378 |
| var_y | **0.0205** | 0.1099 | 0.0547 | 0.0642 | 0.1526 | 0.0524 |
| var_t | **0.0425** | 0.2202 | 0.0777 | 0.0857 | 0.0704 | 0.0706 |
| cov_xy | **0.0106** | 0.0344 | 0.0494 | 0.0490 | 0.0296 | 0.0340 |
| cov_xt | **0.0161** | 0.0174 | 0.0225 | 0.0258 | 0.0143 | 0.0160 |
| cov_yt | **0.0170** | 0.0248 | 0.0253 | 0.0220 | 0.0156 | 0.0178 |
| active_site_ratio | 0.1582 | 0.0785 | **0.0466** | 0.0543 | 0.9169 | 0.0647 |
| temporal_trend | **0.2599** | 0.4488 | 0.4610 | 0.5465 | 0.3242 | 0.5404 |

Spatial consistency ≈REAL | Ours (4C+soft) 0.4042 | MaskGIT-flat 0.7298 | 3D U-Net (det.)† 0.6677 | 3D CVAE 0.5570 | Dich. Gaussian 0.4687 | Coupled GLM 1.0000 | REAL 0.9996


### causal (1)

Local context, MAE against the clip's own lct ↓  (own-count readout)

| feature | Ours (4C+soft) | MaskGIT-flat | 3D U-Net (det.)† | 3D CVAE | _ref_ Dich. Gaussian | _ref_ Coupled GLM |
|---|---|---|---|---|---|---|
| log_mean_firing_density | 1.4100 | 0.9863 | 0.5434 | **0.4305** | 4.2497 | 0.2461 |
| var_x | **0.0157** | 0.0318 | 0.0340 | 0.0531 | 0.1865 | 0.0244 |
| var_y | **0.0194** | 0.0306 | 0.0427 | 0.0508 | 0.1519 | 0.0291 |
| var_t | 0.0958 | 0.1857 | **0.0824** | 0.1186 | 0.1624 | 0.0510 |
| cov_xy | **0.0091** | 0.0146 | 0.0272 | 0.0410 | 0.0294 | 0.0189 |
| cov_xt | **0.0136** | 0.0143 | 0.0260 | 0.0349 | 0.0144 | 0.0246 |
| cov_yt | **0.0156** | 0.0204 | 0.0307 | 0.0269 | 0.0151 | 0.0309 |
| active_site_ratio | 0.1111 | 0.0290 | 0.0191 | **0.0183** | 0.9114 | 0.0259 |
| temporal_trend | 0.6759 | 0.3429 | 0.2404 | **0.2301** | 0.9527 | 0.2450 |

Spatial consistency ≈REAL | Ours (4C+soft) 0.5045 | MaskGIT-flat 0.9817 | 3D U-Net (det.)† 0.8298 | 3D CVAE 0.6946 | Dich. Gaussian 0.4729 | Coupled GLM 0.9998 | REAL 0.9996


### noncausal (2)

Local context, MAE against the clip's own lct ↓  (own-count readout)

| feature | Ours (4C+soft) | MaskGIT-flat | 3D U-Net (det.)† | 3D CVAE | _ref_ Dich. Gaussian | _ref_ Coupled GLM |
|---|---|---|---|---|---|---|
| log_mean_firing_density | 1.2260 | 0.5292 | 0.6122 | **0.4584** | 3.8947 | 0.1648 |
| var_x | 0.0159 | **0.0071** | 0.0256 | 0.0543 | 0.1856 | 0.0175 |
| var_y | 0.0177 | **0.0111** | 0.0318 | 0.0451 | 0.1513 | 0.0192 |
| var_t | 0.1386 | 0.1084 | 0.0809 | **0.0754** | 0.2294 | 0.0390 |
| cov_xy | 0.0106 | **0.0054** | 0.0228 | 0.0391 | 0.0295 | 0.0128 |
| cov_xt | **0.0118** | 0.0121 | 0.0170 | 0.0273 | 0.0140 | 0.0162 |
| cov_yt | 0.0154 | **0.0138** | 0.0227 | 0.0190 | 0.0154 | 0.0212 |
| active_site_ratio | 0.1046 | 0.0215 | 0.0188 | **0.0117** | 0.9106 | 0.0167 |
| temporal_trend | 0.4396 | 0.3177 | 0.2396 | **0.2002** | 0.5641 | 0.1274 |

Spatial consistency ≈REAL | Ours (4C+soft) 0.5255 | MaskGIT-flat 0.9921 | 3D U-Net (det.)† 0.8595 | 3D CVAE 0.7297 | Dich. Gaussian 0.4758 | Coupled GLM 0.9998 | REAL 0.9996


### spatial (3)

Local context, MAE against the clip's own lct ↓  (own-count readout)

| feature | Ours (4C+soft) | MaskGIT-flat | 3D U-Net (det.)† | 3D CVAE | _ref_ Dich. Gaussian | _ref_ Coupled GLM |
|---|---|---|---|---|---|---|
| log_mean_firing_density | 1.0765 | **0.6668** | 0.8451 | 0.7453 | 3.7506 | 0.1452 |
| var_x | 0.0318 | **0.0244** | 0.0486 | 0.0384 | 0.0800 | 0.0135 |
| var_y | **0.0391** | 0.0649 | 0.0520 | 0.0531 | 0.0829 | 0.0178 |
| var_t | 0.0356 | **0.0245** | 0.0471 | 0.0567 | 0.0707 | 0.0342 |
| cov_xy | **0.0188** | 0.0334 | 0.0338 | 0.0359 | 0.0326 | 0.0104 |
| cov_xt | 0.0126 | **0.0124** | 0.0172 | 0.0221 | 0.0138 | 0.0123 |
| cov_yt | 0.0142 | **0.0122** | 0.0188 | 0.0240 | 0.0154 | 0.0149 |
| active_site_ratio | 0.0629 | 0.0345 | **0.0161** | 0.0212 | 0.7829 | 0.0258 |
| temporal_trend | 0.1803 | **0.1004** | 0.1862 | 0.2965 | 0.2630 | 0.1874 |

Spatial consistency ≈REAL | Ours (4C+soft) 0.5972 | MaskGIT-flat 0.9889 | 3D U-Net (det.)† 0.7194 | 3D CVAE 0.6271 | Dich. Gaussian 0.5011 | Coupled GLM 0.9997 | REAL 0.9996


The oracle row decodes each model's TRUE tokens, so it is that tokenizer's ceiling rather than a competitor: the gap between a model and its own oracle is what the PRIOR fails to predict, and the gap between the oracle and the data is what the ALPHABET cannot represent.

`marginal SEEN` reads the test clip's own assay off a table built from that assay's train clips. The split is temporal within assay, so no model here ever faces an unseen assay and that lookup is never charged for it -- which is exactly what DG and the GLM memorise. `marginal UNSEEN` is the same null with the clip's own assay withheld. Neither row establishes transfer for a learned model; that needs a retrain with assays held out.

`separable` collapses onto `marginal` on the two temporal tasks by construction: their holes span whole frames, so no visible voxel lies inside the hole and the clip-derived temporal profile is constant there. At SITE level it collapses onto `marginal` on every task, including `spatial`: collapsing time turns the per-frame profile into one positive per-clip scalar, which cannot reorder sites.

**Reading the winners.** DG and the GLM top both tables. Their margin over the SEEN site map -- the lookup they reproduce -- is:

| margin over `marginal SEEN` | voxel | site |
|---|---|---|
| Dich. Gaussian | -0.015 to -0.011 **(below the null)** | +0.012 to +0.020 |
| Coupled GLM | -0.003 to +0.006 | +0.018 to +0.022 |

The nulls they are beating sit at 0.07-0.08 (voxel) and 0.52-0.72 (site), so these are small margins on a very strong lookup, and several are not significant (see the table above). They are not completing the hole; they reproduce a per-assay lookup table and add a small temporal correction on the tasks where coupling filters can act. Section C withholds that table and both fall below their own null.


## Generation, full conditioning ladders

`local_only` is a control, not a rung: it hands the model the true lct with a MISMATCHED gct, so it is contradictory information rather than less of it.


### A. Conditional accuracy ↓

Per-clip z-scored MAE between the lct recomputed from the sample and the TRUE clip's lct, each feature divided by its spread across test clips. Per-clip, so a paired Wilcoxon applies.

| model ↓ | random | LOCAL only | GLOBAL only | glob+partial | glob+full |
|---|---|---|---|---|---|
| Ours (4C+soft) | **1.0623** | 0.8473 | **0.6738** | **0.6172** | **0.5350** |
| MaskGIT-flat | 1.1892 | 0.9595 | 0.8347 | 0.7528 | 0.7362 |
| 3D U-Net (det.)† | 1.0741 | **0.8415** | 0.7711 | 0.6740 | 0.6803 |
| 3D CVAE | 1.1388 | 0.9099 | 0.8373 | 0.7581 | 0.7441 |
| | | | | | |
| _ref_ Dich. Gaussian | 0.7243 | 0.5183 | 0.6009 | 0.5298 | 0.5137 |
| _ref_ Coupled GLM | 0.5447 | 0.5584 | 0.5560 | 0.5648 | 0.5610 |

### B. Adherence ↑

Mean over the 9 features of r(realised, **requested**) -- against the lct handed to the model, not the true one. A property of the model, not of how much context it got, so an obedient model is flat across the ladder.

| model ↑ | random | LOCAL only | GLOBAL only | glob+partial | glob+full |
|---|---|---|---|---|---|
| Ours (4C+soft) | **0.7183** | 0.3081 | **0.6945** | **0.7212** | **0.6992** |
| MaskGIT-flat | 0.5608 | 0.2535 | 0.5457 | 0.4971 | 0.5429 |
| 3D U-Net (det.)† | 0.5047 | 0.3280 | 0.5042 | 0.5120 | 0.5165 |
| 3D CVAE | 0.4872 | **0.3364** | 0.4885 | 0.4751 | 0.4925 |
| | | | | | |
| _ref_ Dich. Gaussian | 0.1779 | 0.5150 | 0.5210 | 0.5416 | 0.5472 |
| _ref_ Coupled GLM | 0.0057 | 0.4973 | 0.5094 | 0.5039 | 0.5012 |

### C. Spatial placement, lookup-proof ↑

Map correlation against the clip's own electrodes MINUS the same generated map scored against a DIFFERENT clip of the same assay. `assay_idx` bypasses the ladder, so raw map r is mostly a per-assay lookup for DG and the GLM; this difference is the part a fixed site map cannot fake.

| model ↑ | random | LOCAL only | GLOBAL only | glob+partial | glob+full |
|---|---|---|---|---|---|
| Ours (4C+soft) | **0.0018** | -0.0012 | -0.0058 | **0.0104** | **0.0101** |
| MaskGIT-flat | -0.0008 | 0.0003 | -0.0040 | 0.0039 | 0.0027 |
| 3D U-Net (det.)† | -0.0014 | -0.0004 | **0.0010** | 0.0047 | 0.0035 |
| 3D CVAE | -0.0009 | **0.0010** | -0.0014 | -0.0003 | 0.0052 |
| | | | | | |
| _ref_ Dich. Gaussian | -0.0009 | 0.0049 | 0.0018 | 0.0014 | 0.0058 |
| _ref_ Coupled GLM | 0.0048 | -0.0052 | 0.0066 | -0.0028 | -0.0049 |

### D. Marginal realism ↓

Mean symmetric relative error `|gen-real|/(gen+real)` of the short-gap rate over the seven canonical gap bins (`DEFAULT_GAP_BINS`). Bounded in [0,1]: 0 matches real exactly, 1 is a total miss. Bounded on purpose -- a log-ratio explodes when a model emits exactly zero rate in some bin. Does not depend on conditioning.

| model ↓ | random | LOCAL only | GLOBAL only | glob+partial | glob+full |
|---|---|---|---|---|---|
| Ours (4C+soft) | 0.2467 | 0.2285 | 0.2523 | 0.2145 | 0.2242 |
| MaskGIT-flat | 0.3684 | 0.3808 | 0.3704 | 0.3842 | 0.3661 |
| 3D U-Net (det.)† | **0.0762** | **0.1088** | **0.0981** | **0.0881** | **0.1025** |
| 3D CVAE | 0.3450 | 0.4169 | 0.3454 | 0.3311 | 0.3484 |
| | | | | | |
| _ref_ Dich. Gaussian | 0.1455 | 0.0661 | 0.0681 | 0.0619 | 0.0640 |
| _ref_ Coupled GLM | 0.0311 | 0.0321 | 0.0382 | 0.0334 | 0.0268 |

## Short-gap adjacency at full context

Rate P(spike at t+g | spike at t) by canonical gap bin (`DEFAULT_GAP_BINS`), free generation at the `global_full_local` rung.

Note this is NOT the same measurement as the `recon` block of the per-task battery, though both report the same quantity on the same bins: this one is the shipped free-generation path at a conditioning rung, that one is the task harness with MC averaging and a top-N readout. Expect the ratios to differ; quote whichever matches the claim being made, and say which.

Best = closest to REAL, not largest or smallest.

| offset ≈REAL | REAL | Ours (4C+soft) | MaskGIT-flat | 3D U-Net (det.)† | 3D CVAE | Dich. Gaussian | Coupled GLM |
|---|---|---|---|---|---|---|---|
| 1 | 0.03135 | 0.04812 (1.54x) | 0.12795 (4.08x) | 0.05792 (1.85x) | **0.02706** (0.86x) | 0.04733 (1.51x) | 0.02784 (0.89x) |
| 2 | 0.04928 | **0.04841** (0.98x) | 0.11108 (2.25x) | 0.05486 (1.11x) | 0.02568 (0.52x) | 0.05560 (1.13x) | 0.05348 (1.09x) |
| 3 | 0.05979 | 0.10212 (1.71x) | 0.09135 (1.53x) | **0.05263** (0.88x) | 0.02552 (0.43x) | 0.06646 (1.11x) | 0.05945 (0.99x) |
| 4-6 | 0.06333 | 0.15316 (2.42x) | **0.06272** (0.99x) | 0.05354 (0.85x) | 0.02532 (0.40x) | 0.06580 (1.04x) | 0.06139 (0.97x) |
| 7-12 | 0.05641 | 0.09389 (1.66x) | 0.03203 (0.57x) | **0.05042** (0.89x) | 0.02485 (0.44x) | 0.05300 (0.94x) | 0.05394 (0.96x) |
| 13-24 | 0.04956 | 0.07992 (1.61x) | 0.01943 (0.39x) | **0.04489** (0.91x) | 0.02222 (0.45x) | 0.04818 (0.97x) | 0.04945 (1.00x) |
| 25-48 | 0.04408 | 0.06470 (1.47x) | 0.00952 (0.22x) | **0.03512** (0.80x) | 0.01668 (0.38x) | 0.05044 (1.14x) | 0.04825 (1.09x) |

Shape of the curve, not its level: correlation with the REAL profile after mean-centring and scaling. REAL rises +91% from gap 1 to gap 3.

| | Ours (4C+soft) | MaskGIT-flat | 3D U-Net (det.)† | 3D CVAE | Dich. Gaussian | Coupled GLM |
|---|---|---|---|---|---|---|
| shape r vs REAL  ↑ | **0.828** | -0.280 | 0.030 | 0.106 | 0.809 | 0.960 |
| gap 1 -> gap 3  ≈REAL | **+112%** | -29% | -9% | -6% | +40% | +114% |

An arm can match the LEVEL by being featureless. Read this with the table above, not instead of it.


## Per-feature lct, ours

| feature ↑ | random | LOCAL only | GLOBAL only | glob+partial | glob+full |
|---|---|---|---|---|---|
| log_mean_firing_density | +0.95 | +0.90 | +0.95 | +0.96 | **+0.96** |
| var_x | +0.99 | +0.04 | +0.98 | **+0.99** | +0.96 |
| var_y | +0.98 | +0.14 | **+0.98** | +0.97 | +0.97 |
| var_t | +0.63 | +0.39 | +0.68 | +0.61 | **+0.72** |
| cov_xy | +0.85 | +0.03 | +0.86 | **+0.87** | +0.87 |
| cov_xt | +0.35 | -0.02 | +0.20 | **+0.37** | +0.09 |
| cov_yt | +0.26 | +0.07 | +0.17 | +0.20 | **+0.21** |
| active_site_ratio | +0.62 | +0.52 | +0.62 | **+0.67** | +0.66 |
| temporal_trend | +0.84 | +0.70 | +0.83 | +0.85 | **+0.86** |

r against the REQUESTED lct. The spatial-shape features (var_x, var_y, cov_xy) collapse under `LOCAL only` while rate and trend survive: the electrode layout arrives through gct, so a mismatched gct makes a requested spatial variance physically unrealisable.

