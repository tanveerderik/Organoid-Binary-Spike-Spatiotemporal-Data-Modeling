# v2 baseline — measured 2026-08-14, before the v3 rebuild

Checkpoints for these numbers are in `ckpts/v2_baseline/`. The v3 run overwrites
`vqvae_stage1_balanced_*` and `motif_prior_best.pt` in place, which is why they
were copied here.

## The defect v3 targets: within-token probability profile

Decoder output inside active tokens, one token = 6 frames x 15 rows x 14 cols.

| statistic | decoder | real | flat ceiling |
|---|---|---|---|
| temporal entropy (6 bins) | **1.6988** | 0.3121 | 1.7918 = log 6 |
| spatial entropy (210 bins) | 3.1921 | 0.2794 | 5.3471 = log 210 |
| temporal peak fraction | **0.2646** | 0.8107 | 0.1667 = 1/6 |
| corr(decoder, truth) temporal | +0.3942 | — | 0 |
| corr(decoder, truth) spatial | +0.6597 | — | 0 |

Temporal sits at the no-information ceiling while spatial does not — the peak term
had spatial reach (`radius_h/w=1`) and no temporal reach (`radius_t=0`).

**v3 succeeds if** temporal entropy moves 1.6988 -> toward 0.3121, peak fraction
0.2646 -> toward 0.8107, and corr stays at or above 0.3942. Entropy falling while
corr also falls means it is peaking confidently in the wrong frame — worse than
hedging.

## Stage 3B activity prior (MaskGIT), validation

| | dense | per-assay marginal | ridge <- global_ctx |
|---|---|---|---|
| NLL | **0.1190** | 0.1315 | 0.1282 |
| AUPRC | **0.7305** | 0.7145 | 0.7142 |
| exact F1 | 0.5798 | 0.5978 | 0.5984 |

F1 is recorded but never selects: it is a reconstruction metric, maximised by
emitting the mode.

## Generation, token level (val-selected sampler, test measured once, predicted counts)

steps=10, temperature=1.5, gumbel=4.0

| statistic | real | MaskGIT | mean-field |
|---|---|---|---|
| rate | 0.0835 | 0.0861 | 0.0861 |
| persistence lag 1 | 0.611 | 0.548 | 0.458 |
| persistence lag 4 | 0.524 | 0.525 | 0.439 |
| persistence lag 7 | 0.450 | 0.453 | 0.392 |
| spatial co-activation | 0.374 | 0.326 | 0.291 |
| **total abs error** | — | **0.2535** | 0.6907 |
| diversity (Jaccard, lower = more diverse) | — | 0.369 | 0.296 |

## Stage 3 generative evaluation, by context regime

floor (real vs real split-half) 0.0012 | anchor (real vs recon) 0.0377

| regime | Frechet | retr@1 | chance | KS avalanche |
|---|---|---|---|---|
| full_random | 0.3984 | **0.029** | 0.032 | 0.678 |
| global_only | 0.3250 | 0.115 | 0.032 | 0.247 |
| global_local | 0.0493 | 0.326 | 0.032 | 0.035 |
| inpaint | 0.0403 | **0.505** | 0.032 | 0.038 |

full_random sits at chance: the negative control passes, so retrieval measures
conditioning rather than leakage.

## Voxel-level statistics (the smear)

| statistic | real | generated (global_local) | recon(real codes) |
|---|---|---|---|
| rate | 0.00012 | 0.00014 | 0.00023 |
| persist1 | **0.0374** | 0.3558 | **0.5089** |
| isi_mean | 8.664 | 3.599 | 2.828 |
| avalanche_mean | 31.5 | 41.2 | 82.4 |

recon(real codes) is worse than full generation, which is what proved the smear is
the decoder's and not the priors'.

## Stage 3C

No accepted improvement. Baseline composite 0.670444; best post-gate 0.669447.
The 0.680885 at epoch 7 was rejected as pre-ramp. Saved checkpoint is a fallback
byte-identical to 3B.

## Held-out-assay diagnostic (NOT the pipeline split — 25 train / 6 unseen)

dense 0.5084 | ridge <- global_ctx 0.2890 | global marginal 0.2836

global_ctx carries a generalizable assay->map function, but it is nonlinear: the
ridge collapses on unseen assays while the model transfers. Do not use
in-distribution linear probes as transfer evidence.
