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

## Readout: what each model does to produce a binary volume

Every method must emit spikes, and how it gets there is not a free choice --
using the wrong operation makes a correctly-fitted model look broken. Both
mistakes below were made here first and caught by measurement.

| baseline | model output | readout | why |
|---|---|---|---|
| DG | latent Gaussian field `U`, per-voxel threshold `theta` | `U > theta` | the DG construction itself |
| GLM | conditional intensity per bin | Bernoulli, sequential | it *is* a point process |
| MaskGIT-flat | per-voxel decoder probability | Bernoulli + scalar shift | decoder trained with BCE |

The rule: **threshold a noisy score, sample a probability.** Rank-thresholding a
smooth probability field picks contiguous voxels inside the hottest regions, and
because each MaskGIT token expands to a 6x15x14 block that yields blobs instead
of isolated spike events -- measured at 11x the real spatial co-activation.
Rank-thresholding a point-process intensity is worse still, since the static
per-electrode baseline dominates and the result is time-columns.

Every baseline matches its rate **in expectation** and none is handed the
realised count of the clip it is scored against. Forcing an exact count by top-k
produces an artificially small `rel_rate` that is a property of the readout, not
the model.

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

## The five baselines

Two groups, and the difference decides how a column may be read.

**Video models** -- the peer group. Same conditioning, same holes, same readout,
nothing stored per assay, so they belong in the same scalability class we do.

| dir | method | citation | family | attacks |
|---|---|---|---|---|
| `maskgit_flat/` | single-level 3D VQ + vanilla MaskGIT | Chang et al. CVPR 2022; Yu et al. CVPR 2023 | voxel+token | "your hierarchy and factorisation are unnecessary" |
| `unet3d/` | conditional 3D U-Net inpainter; FiLM context + learned spatial embedding | Cicek et al. MICCAI 2016 (cf. V-Net, Milletari et al. 3DV 2016); Perez et al. AAAI 2018 | voxel | "the tasks are inpainting -- why not just supervise a conv net?" |
| `cvae3d/` | same backbone plus a conditional latent grid | Sohn et al. NIPS 2015 | voxel | "does the code need to be quantized?" |

**Per-assay lookup tables** -- reference ceilings, not peers. Each stores a full
(H,W) site map per preparation (26,880 floats), against 26 and 5,194 fitted
values respectively, so their footprint grows linearly with the number of
assays and the report renders them as `_ref_` rows.

| dir | method | citation | family | attacks |
|---|---|---|---|---|
| `dichotomized_gaussian/` | stationary DG field, FFT-sampled | Macke et al., Neural Comput. 2009 | voxel | "you don't beat classical spike statistics" |
| `coupled_glm/` | conv Poisson GLM, history + local coupling | Pillow et al., Nature 2008 | voxel | "where is the standard point-process model?" |

`unet3d` and `cvae3d` are a matched pair: identical backbone, conditioning, hole
distribution, optimiser and readout, differing only in whether a latent variable
is present. So the gap between their columns isolates stochasticity from
everything else. Read them together -- `unet3d` predicts the conditional mean
and is therefore *expected* to win the ranking metrics (AP, F1) and to have
nothing to say on the distributional ones, because a point estimate has no
spread. Neither is a fair peer alone.

Neither has a tokenizer, so neither appears in the reconstruction section: there
is no autoencoding path to measure. That is a capability fact, and the report
labels it rather than leaving a blank.

**Both need the spatial embedding, and the first run proves it.** FiLM is
per-channel and spatially uniform, and under free generation the input is
constant, so without a position basis a global assay code has no way to say
"this preparation fires at THESE electrodes". Correlation between the generated
site map and the real one was 0.0389 without it, against 0.1414 for MaskGIT-flat
(which gets the same ability from per-token positional embeddings) and 0.3553
for the pipeline. Adding a learned (H,W) embedding, shared across every
preparation and broadcast over time, took the U-Net's validation completion loss
from 0.6593 to 0.2310. The handicapped checkpoints are kept as `*_nopos.pt` with
their reports, because "no positional basis" is a measured ablation worth one
line rather than a bug to be quietly deleted.

That embedding IS a 120x224 spatial map, which is the thing the lookup arms are
criticised for. The distinction is one map for ALL 31 preparations versus one
PER preparation, so per-assay storage stays zero -- and the parameter census
gives it its own line rather than letting it hide inside the network total.

**The CVAE's latent collapses, and that is the correct answer.** At generation
time z is drawn from the conditional prior, which cannot see the hole, so any
information the posterior packs into z is unusable at test time and `KL(q||p)`
correctly drives it to zero. Both conv arms therefore take their stochasticity
from the Bernoulli readout, exactly as MaskGIT-flat does, and the matched pair
answers "does a continuous latent help here?" with a measured no. The fit report
carries `latent_collapsed` so no column is read as evidence about sampling.

Cited but deliberately not run, with reasons, so the omissions are arguments
rather than gaps:

* **Video diffusion** (Ho et al. 2022; Blattmann et al. 2023) -- the one
  genuinely expected omission, and it is a compute argument rather than a
  principled one: iterative denoising over a 1.29M-voxel binary volume at 480
  training clips on one 16 GB card is a project, not a baseline. `cvae3d`
  covers the continuous-latent generative slot in the meantime. State this in
  the paper; do not let a reviewer find it first.

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
