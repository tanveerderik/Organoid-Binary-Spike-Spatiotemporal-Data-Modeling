# external_baselines/

Published, non-ours generative models, run on the same data, the same split and
the same metrics as the shipped pipeline. Everything in here exists to answer a
reviewer asking "compared to what?".

Named `external_baselines` and not `baselines` on purpose: `training/baselines.py`
already holds the **internal** null ladder (surrogate, swapped-context, marginal,
oracle), which stays part of the main code. Internal nulls bound what the task
allows; these bound what other people's methods achieve. Different claims,
different directories.

## The boundary rule

    external_baselines/  ->  may import the main pipeline
    main pipeline        ->  may NEVER import external_baselines

Enforced, not merely intended:

    python external_baselines/tests/test_import_boundary.py

The shipped numbers must not be able to move because a baseline changed. The
test also pins a second rule -- only `common/data.py` may import `main` -- so a
refactor of the 4.7k-line monolith breaks one import site instead of five. And
it `ast.parse`s every file in both trees, because a bulk rename here once
produced a syntactically invalid identifier that `import` did not catch (the
module was imported lazily).

## Layout

    common/protocol.py    the contract: ConditioningBatch in, binary volume out
    common/data.py        the ONLY bridge to main.py; split-faithful loaders
    common/evaluate.py    scoring; reuses the pipeline's own GenerationWriter
    registry.py           name -> constructor, lazily imported
    run_baseline.py       CLI: fit / sample / score
    <baseline>/           one directory per method

## Why the contract looks like that

`sample()` receives a `ConditioningBatch` and nothing else. It carries gct (64-d,
assay level), lct (9-d, clip level) and the output shape -- **not** the target
volume. Leakage is the first thing a reviewer suspects in a generative
comparison, and here a baseline is structurally incapable of seeing the clip it
is scored against.

Output is a binary `(B,T,H,W)` volume, which is exactly what
`inference/metrics_generative.py::mea_statistics` consumes. So scoring is
model-agnostic: no baseline gets a bespoke metric path, and the voxel-family
metrics are open to any generator. The token-family metrics (MRR, medRank, CE)
need the pipeline's tokenizer, so only token-space baselines implement
`tokenize()`; the rest are marked voxel-only in the results table rather than
being quietly excluded.

Scoring writes the **same tree** as the pipeline's own `generation_regimes_*`
sample sets, so `analysis/compare_regimes.py` can run its paired cross-set test
and BH-FDR against a baseline with no new statistics code.

## One protocol, verified

The deliverable is a comparative table, so every row is produced identically.
`common/evaluate.py::REFERENCE_PROTOCOL` mirrors the defaults the pipeline's own
`reports/generation_regimes_*` sets were generated under -- test split, 8
batches, 4 samples per clip, seed 20260821, the same context bank, the same
pinned partial-local features -- and the sampling loop replicates
`generate_regimes.py`'s `batch -> regime -> rep` ordering, because sample indices
are what the cross-set paired test joins on.

This is checked rather than assumed. `verify_against_reference()` compares the
produced manifest against a real model manifest clip by clip and raises on
mismatch; `compare_table.py` refuses to build a table from unaligned sets. The
check earned its place immediately: the first DG run used 128 distinct clips
while the model's sets used 32 clips x 4 draws. Same n, different clips.

Baselines also run the full conditioning ladder (`common/ladder.py`), so every
method is asked the same monotone sequence of context questions.

## The rate-calibration row

The shipped decode path binarises with an F1-selected threshold (`best_thr_tol`,
which is off-limits) and under-produces spikes by ~25pp. A point-process
baseline emits spikes directly and pays no such tax, so a raw `rate` comparison
measures a threshold rather than a model.

Every baseline that can expose a continuous score implements
`sample_intensity()`, and `common/evaluate.py` emits a second row in which every
model is re-binarised at a per-clip threshold hitting the **train** assay mean
rate. That uses no held-out information and is available to all. Report both
rows; the raw one is honest about the deployed system, the calibrated one is
honest about the model.

**The invariant held fixed is the rate, not the mechanism.** Rank-thresholding
suits a model whose score is a noisy field (the DG, the MaskGIT decoder) but
destroys a point process: top-k over the GLM's log-intensity is dominated by its
static per-electrode baseline and returns time-columns rather than spike trains
(stat_error 2.98 and ks_isi 0.93, against 0.87 and 0.37 for its own samples).
The GLM therefore returns `None` from `sample_intensity` and matches its rate
where a point process should -- a free-running DC offset fitted on train data
inside `fit()`. Each baseline's mechanism is recorded in its `run_config.json`.

## Data regime, and why it dictates the baselines

Measured on the test split: **~200 spikes per 48x120x224 clip, voxel rate
1.55e-4**. Train carries ~186k spikes total across 26,880 channels.

Most channel pairs therefore never co-fire, which rules out the textbook form of
two of the three baselines. A 26,880^2 empirical covariance is not merely large
(2.9 GB), it is unestimable; a per-electrode GLM has no data per electrode. Both
statistical baselines are consequently specified as **stationary** models --
translation-invariant kernels pooled over displacements -- which is both
estimable here and standard practice. This is a property of the data, and it is
stated in the paper rather than buried.

## The three baselines

| dir | method | citation | family | attacks |
|---|---|---|---|---|
| `maskgit_flat/` | single-level 3D VQ + vanilla MaskGIT | Chang et al. CVPR 2022; Yu et al. CVPR 2023 | voxel+token | "your hierarchy and factorisation are unnecessary" |
| `dichotomized_gaussian/` | stationary DG field, FFT-sampled | Macke et al., Neural Comput. 2009 | voxel | "you don't beat classical spike statistics" |
| `coupled_glm/` | conv Poisson GLM, history + local coupling | Pillow et al., Nature 2008 | voxel | "where is the standard point-process model?" |

Cited but deliberately not run, with reasons, so the omissions are arguments
rather than gaps:

* **LFADS** (Pandarinath et al. 2018) -- trialised, ~100 neurons; wrong regime.
* **Pairwise maxent / Ising** (Schneidman et al. 2006) -- intractable at 26,880
  channels; the DG is its tractable cousin and is run instead.
* **LDNS** (Kapoor & Schulz et al., NeurIPS 2024) -- closest recent work, but
  designed for ~100-200 neurons. Running it here needs either heavy coarsening
  (changes the task) or our own latent space (stops being external).
* **VideoGPT** (Yan et al. 2021) -- differs from MaskGIT only in decoding order;
  that is an ablation, not an external baseline.
* **NDT2 / POYO / SpikeProphecy** -- representation learning and forecasting,
  not conditional generation.

## Running

    python external_baselines/run_baseline.py --list
    python external_baselines/run_baseline.py --baseline dg --fit --score --batches 40

CWD is pinned to the repo root by `common/data.py::ensure_repo_cwd()`: assay
discovery globs `../output_data/...` relatively and the dataset stores the
relative paths it found, so DataLoader workers need the same CWD. One chdir,
announced, rather than editing `dataset.py` on the baselines' behalf.
