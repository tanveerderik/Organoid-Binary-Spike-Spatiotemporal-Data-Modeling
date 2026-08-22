# Interpretable diagnostics

8 test batches (32 clips), seed 20260821, identical clips for every model, each model's own shipped readout. `local_only` is a control, not a rung: it hands the model the true lct with a MISMATCHED gct, so it is not 'less information' than `random` but contradictory information.

## Reconstruction

Step-wise average precision, not the trapezoid AUPRC in `utils/metrics.py` -- trapezoid interpolates the PR curve linearly, which is invalid (Davis & Goadrich 2006) and inflated the saturating MaskGIT tokenizer by +0.22 off a single voxel.

| | Ours (4C+soft) | MaskGIT-flat | Dich. Gaussian | Coupled GLM |
|---|---|---|---|---|
| AP step-wise, exact | **0.2564** | 0.0269 | -- | -- |
| AP step-wise, tolerant | **0.2680** | 0.0492 | -- | -- |
| best F1, exact | **0.3055** | 0.0678 | -- | -- |
| best F1, tolerant | **0.3074** | 0.1046 | -- | -- |
| trapezoid inflation | **+0.0015** | +0.2239 | -- | -- |
| codebook used | 619 | 844 | -- | -- |
| codebook perplexity | 436.9 | 312.2 | -- | -- |

DG and the GLM are point processes with no tokenizer.

## Generation


### A. Conditional accuracy  (the headline)

Per-clip z-scored MAE between the lct recomputed from the sample and the TRUE clip's lct, each feature divided by its spread across test clips. **Lower is better.** Per-clip, so a paired Wilcoxon applies. The `random` column is the null.

| model | random | LOCAL only | GLOBAL only | glob+partial | glob+full |
|---|---|---|---|---|---|
| Ours (4C+soft) | 1.2413 | 1.2302 | 0.8652 | 0.6753 | **0.5557** |
| MaskGIT-flat | 1.3985 | 1.2298 | 0.9227 | 0.8122 | 0.7403 |
| Dich. Gaussian | 0.9108 | **0.6328** | 0.7485 | **0.6261** | 0.6571 |
| Coupled GLM | **0.7105** | 0.7045 | **0.6855** | 0.6652 | 0.6666 |

### B. Adherence  (does it do what it is told)

Mean over the 9 features of r(realised, **requested**) -- against the lct handed to the model, not the true one. Higher is better. This is a property of the model, not of how much context it got, so a model that obeys should be flat across the ladder.

| model | random | LOCAL only | GLOBAL only | glob+partial | glob+full |
|---|---|---|---|---|---|
| Ours (4C+soft) | **0.6708** | 0.2305 | **0.7198** | **0.7698** | **0.7923** |
| MaskGIT-flat | 0.4175 | 0.2030 | 0.5630 | 0.5494 | 0.5490 |
| Dich. Gaussian | 0.2502 | **0.4931** | 0.4552 | 0.5280 | 0.5314 |
| Coupled GLM | 0.0944 | 0.4399 | 0.4911 | 0.5542 | 0.4524 |

### C. Spatial placement, lookup-proof

Map correlation against the clip's own electrodes MINUS the same generated map scored against a different clip of the SAME assay. `assay_idx` bypasses the ladder, so raw map r is mostly a per-assay lookup for DG and the GLM; this difference is the part a fixed site map cannot fake. Higher is better.

| model | random | LOCAL only | GLOBAL only | glob+partial | glob+full |
|---|---|---|---|---|---|
| Ours (4C+soft) | 0.0027 | 0.0009 | -0.0159 | **0.0274** | 0.0138 |
| MaskGIT-flat | 0.0024 | 0.0008 | 0.0052 | -0.0002 | 0.0036 |
| Dich. Gaussian | **0.0091** | **0.0086** | **0.0154** | 0.0138 | 0.0186 |
| Coupled GLM | -0.0013 | 0.0000 | -0.0114 | 0.0065 | **0.0272** |

### D. Marginal realism

Mean symmetric relative error `|gen-real|/(gen+real)` of the co-firing profile over 3 spatial displacements and 7 temporal lags. Bounded in [0,1]: 0 matches real exactly, 1 is a total miss. Bounded on purpose -- a log-ratio explodes when a model emits exactly zero co-firing at some offset, which ours does at d=3. **Lower is better**, and this one does not depend on conditioning.

| model | random | LOCAL only | GLOBAL only | glob+partial | glob+full |
|---|---|---|---|---|---|
| Ours (4C+soft) | 0.3056 | 0.2338 | 0.4348 | 0.3512 | 0.3600 |
| MaskGIT-flat | 0.3608 | 0.4023 | 0.3684 | 0.3761 | 0.3726 |
| Dich. Gaussian | **0.0679** | **0.0524** | **0.0527** | **0.0349** | **0.0482** |
| Coupled GLM | 0.1035 | 0.0891 | 0.0838 | 0.1240 | 0.0940 |

### Adjacency profile at full context  P(spike at neighbour | spike)

Best = closest to REAL, not largest or smallest.

| offset | REAL | Ours (4C+soft) | MaskGIT-flat | Dich. Gaussian | Coupled GLM |
|---|---|---|---|---|---|
| space_d1 | 0.00125 | 0.00026 (0.21x) | 0.01718 (13.74x) | **0.00098** (0.79x) | 0.00157 (1.25x) |
| space_d2 | 0.00095 | 0.00053 (0.55x) | 0.01155 (12.19x) | 0.00080 (0.85x) | **0.00092** (0.97x) |
| space_d3 | 0.00208 | 0.00000 (0.00x) | 0.00843 (4.05x) | **0.00190** (0.91x) | 0.00126 (0.60x) |
| time_lag1 | 0.04680 | 0.08134 (1.74x) | 0.12486 (2.67x) | **0.04502** (0.96x) | 0.03111 (0.66x) |
| time_lag2 | 0.04930 | 0.06722 (1.36x) | 0.10973 (2.23x) | 0.05486 (1.11x) | **0.05476** (1.11x) |
| time_lag3 | 0.06662 | 0.09189 (1.38x) | 0.08653 (1.30x) | **0.06345** (0.95x) | 0.05875 (0.88x) |
| time_lag4 | 0.07170 | 0.10195 (1.42x) | **0.06678** (0.93x) | 0.06114 (0.85x) | 0.05582 (0.78x) |
| time_lag5 | 0.06140 | 0.08521 (1.39x) | 0.05248 (0.85x) | **0.05706** (0.93x) | 0.05234 (0.85x) |
| time_lag6 | 0.06011 | 0.27015 (4.49x) | 0.05550 (0.92x) | **0.05867** (0.98x) | 0.05763 (0.96x) |
| time_lag7 | 0.05663 | 0.06896 (1.22x) | 0.03111 (0.55x) | **0.05784** (1.02x) | 0.05422 (0.96x) |

### Per-feature lct, ours

| feature | random | LOCAL only | GLOBAL only | glob+partial | glob+full |
|---|---|---|---|---|---|
| log_mean_firing_density | +0.94 | +0.86 | +0.96 | +0.97 | +0.97 |
| var_x | +0.99 | +0.14 | +0.91 | +0.95 | +0.89 |
| var_y | +0.98 | +0.33 | +0.97 | +0.97 | +0.98 |
| var_t | +0.67 | +0.21 | +0.81 | +0.77 | +0.56 |
| cov_xy | +0.88 | -0.25 | +0.91 | +0.91 | +0.88 |
| cov_xt | +0.02 | -0.22 | -0.06 | +0.20 | +0.67 |
| cov_yt | +0.33 | -0.11 | +0.38 | +0.48 | +0.54 |
| active_site_ratio | +0.54 | +0.51 | +0.67 | +0.74 | +0.68 |
| temporal_trend | +0.69 | +0.60 | +0.93 | +0.93 | +0.95 |

r against the REQUESTED lct. The spatial-shape features (var_x, var_y, cov_xy) collapse under `LOCAL only` while rate and trend survive: the electrode layout arrives through gct, so a mismatched gct makes a requested spatial variance physically unrealisable.

