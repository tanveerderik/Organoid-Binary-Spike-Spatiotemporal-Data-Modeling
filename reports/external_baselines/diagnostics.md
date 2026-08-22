# Interpretable diagnostics

8 test batches (32 clips), seed 20260821, identical clips for every model. Generated volumes come from each model's own shipped readout.

## 1. Tokenizer reconstruction  (encode -> quantize -> decode)

Step-wise average precision, NOT the trapezoid AUPRC in `utils/metrics.py`. Trapezoid linearly interpolates the PR curve, which is invalid (Davis & Goadrich 2006) and inflated the saturating MaskGIT tokenizer by +0.22 off a single voxel. The `interp. inflation` row shows how much each model was affected.

| | Ours (4C+soft) | MaskGIT-flat | Dich. Gaussian | Coupled GLM |
|---|---|---|---|---|
| AP step-wise, exact | 0.2564 | 0.0269 | -- | -- |
| AP step-wise, tolerant (1,1,1) | 0.2680 | 0.0492 | -- | -- |
| best F1, exact | 0.3055 | 0.0678 | -- | -- |
| best F1, tolerant | 0.3074 | 0.1046 | -- | -- |
| chance (base rate) | 1.53e-04 | 1.53e-04 | -- | -- |
| x chance (exact) | 1,678x | 176x | -- | -- |
| interp. inflation (exact) | +0.0015 | +0.2239 | -- | -- |
| codebook used | 619 | 844 | -- | -- |
| codebook perplexity | 436.9 | 312.2 | -- | -- |

DG and the GLM have no tokenizer; they are point-process models and this section does not apply to them.

## 2. Is the generated field informative?  (the all-blank check)

| | Ours (4C+soft) | MaskGIT-flat | Dich. Gaussian | Coupled GLM |
|---|---|---|---|---|
| all-blank samples | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| AUPRC vs its OWN clip | 0.0407 | 0.0084 | 0.0879 | -- |
| AUPRC vs a DIFFERENT clip | 0.0206 | 0.0053 | 0.0624 | -- |
| **clip-specific margin** | 0.0201 | 0.0031 | 0.0254 | -- |
| field std (flat would be ~0) | 5.2692 | 4.2069 | 1.0186 | -- |

## 3. Local context adherence  (9 lct features recomputed from the sample, r vs the vector the model was GIVEN)

| feature | Ours (4C+soft) rnd -> full | MaskGIT-flat rnd -> full | Dich. Gaussian rnd -> full | Coupled GLM rnd -> full |
|---|---|---|---|---|
| log_mean_firing_density | +0.94 -> +0.97 | +0.99 -> +0.99 | +0.99 -> +0.99 | -0.08 -> +0.72 |
| var_x | +0.99 -> +0.85 | +0.94 -> +0.78 | +0.40 -> +0.93 | +0.40 -> +0.89 |
| var_y | +0.99 -> +0.97 | +0.97 -> +0.93 | +0.04 -> +0.97 | -0.00 -> +0.98 |
| var_t | +0.76 -> +0.77 | +0.38 -> -0.01 | +0.20 -> +0.03 | -0.28 -> -0.23 |
| cov_xy | +0.91 -> +0.91 | +0.56 -> +0.72 | -0.04 -> +0.92 | +0.09 -> +0.96 |
| cov_xt | +0.69 -> +0.26 | +0.18 -> +0.08 | +0.04 -> +0.16 | -0.09 -> -0.29 |
| cov_yt | +0.64 -> +0.46 | +0.37 -> +0.22 | +0.01 -> -0.00 | -0.01 -> +0.01 |
| active_site_ratio | +0.62 -> +0.68 | +0.88 -> +0.93 | +0.66 -> +0.99 | -0.03 -> +0.79 |
| temporal_trend | +0.82 -> +0.92 | +0.16 -> +0.45 | -0.27 -> +0.00 | -0.20 -> -0.16 |

`log_mean_firing_density` is CIRCULAR for DG and MaskGIT-flat -- their firing rate is regressed directly from lct, so a high r there measures the regression, not the model.

## 4. Spatial map  (per-electrode counts vs the true clip)

| | Ours (4C+soft) | MaskGIT-flat | Dich. Gaussian | Coupled GLM |
|---|---|---|---|---|
| pearson r, random ctx | 0.0009 | 0.0111 | 0.6056 | 0.6454 |
| pearson r, full ctx | 0.3379 | 0.1423 | 0.6316 | 0.6458 |
| vs other clip, SAME assay (full) | 0.3270 | 0.1418 | 0.6189 | 0.6412 |
| vs a different assay (full) | -0.0002 | 0.0091 | -0.0009 | -0.0010 |
| **within-assay gap** | 0.0109 | 0.0005 | 0.0126 | 0.0046 |
| assay-identity component | 0.3272 | 0.1327 | 0.6198 | 0.6422 |
| active-site IoU, full ctx | 0.2029 | 0.0555 | 0.4187 | 0.4167 |

`assay_idx` reaches every model unchanged in EVERY regime -- the ladder randomises gct/lct, not assay identity. DG and the GLM key their train-fitted site maps on it, so their `random` rung is not a control and their pearson r is mostly a per-assay lookup. The **within-assay gap** -- own clip minus a different clip from the same assay -- is the only row a fixed site map cannot fake.

## 5. Adjacency  P(spike at neighbour | spike), full context

| | REAL | Ours (4C+soft) | MaskGIT-flat | Dich. Gaussian | Coupled GLM |
|---|---|---|---|---|---|
| space_d1 | 0.00125 | 0.00021 (0.2x) | 0.01318 (10.5x) | 0.00152 (1.2x) | 0.00095 (0.8x) |
| space_d2 | 0.00095 | 0.00038 (0.4x) | 0.01033 (10.9x) | 0.00119 (1.3x) | 0.00143 (1.5x) |
| space_d3 | 0.00208 | 0.00000 (0.0x) | 0.00676 (3.2x) | 0.00119 (0.6x) | 0.00277 (1.3x) |
| time_lag1 | 0.04680 | 0.08776 (1.9x) | 0.11976 (2.6x) | 0.04705 (1.0x) | 0.02724 (0.6x) |
| time_lag2 | 0.04930 | 0.07535 (1.5x) | 0.09985 (2.0x) | 0.05414 (1.1x) | 0.05329 (1.1x) |
| time_lag3 | 0.06662 | 0.10080 (1.5x) | 0.08600 (1.3x) | 0.06139 (0.9x) | 0.05993 (0.9x) |
| time_lag4 | 0.07170 | 0.10814 (1.5x) | 0.06676 (0.9x) | 0.06458 (0.9x) | 0.05512 (0.8x) |
| time_lag5 | 0.06140 | 0.08857 (1.4x) | 0.05067 (0.8x) | 0.06067 (1.0x) | 0.05641 (0.9x) |
| time_lag6 | 0.06011 | 0.28274 (4.7x) | 0.05886 (1.0x) | 0.05891 (1.0x) | 0.05944 (1.0x) |
| time_lag7 | 0.05663 | 0.07702 (1.4x) | 0.03251 (0.6x) | 0.05859 (1.0x) | 0.05247 (0.9x) |
