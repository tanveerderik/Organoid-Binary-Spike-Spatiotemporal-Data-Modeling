# Organoid-Binary-Spike-Spatiotemporal-Data-Modeling

Context-conditioned hierarchical VQ-VAE plus a MaskGIT-style discrete prior for
ultra-sparse binary neural spike data (x, y, t). The VQ-VAE learns discrete
spatiotemporal motifs on a three-level codebook ladder; the prior learns to
generate those motifs conditioned on assay and activity context.

---

## Overview

The data comes from electrophysiological recordings of neural cultures and
organoids, represented as **binary spatiotemporal volumes** where a voxel is 1
if a spike occurred at that electrode and time bin. Spike density is on the
order of **1e-4**, so almost every voxel is zero.

The pipeline has two halves:

1. **A hierarchical VQ-VAE** that compresses a volume into a grid of discrete
   motif tokens and reconstructs it under biological constraints.
2. **A MaskGIT-style prior** over those tokens, which makes the model
   generative rather than merely compressive.

---

## Data representation

| | |
|---|---|
| Input volume | `48 x 120 x 224` (t, h, w) after temporal pooling of a 6000-sample crop by 120 |
| Patch size | `6 x 15 x 14` |
| Token grid | `8 x 8 x 16` = **1024 token positions** |
| Spike voxel probability | ~1.2e-4 |

Splits are **temporal within assay** (a clip's later time range is held out from
its own earlier range), not random, so held-out clips are never interleaved with
training clips from the same recording.

---

## Architecture

**Encoder / decoder** — transformer, embed dim 64, depth 2, 4 heads, code dim
64. The decoder is **dense**: it reconstructs the full volume from `z_q` plus a
blank-token embedding, with temporal-causal decoder attention.

**Quantizer** — a **three-level residual ladder** with `(32, 8, 4)` codes per
level. Level 1 picks a coarse motif, levels 2 and 3 refine it. The ladder spans
`32 x 8 x 4 = 1024` nominal combinations. (This equals the token-grid size by
coincidence; they are unrelated quantities.)

**Context** enters in two distinct ways:

- **Global context (gct)** — a 64-dim assay descriptor mapped to 32 dims,
  pretrained against memory-bank priors (tokenwise and pixelwise spatial support,
  temporal adjacency statistics).
- **Local context (lct)** — 9 clip-level activity features mapped to 32 dims by
  a trained MLP trunk.

Inside the VQ-VAE, context adherence is produced by `ctx_loss_soft`, an
**output-space** loss that reads context features back out of the reconstructed
logits volume. In the prior, gct and lct enter as **prefix tokens**.

---

## Training pipeline

Stages are selected by `TRAIN_STAGES` / `STAGE2_PHASES` / `STAGE4_PHASES` in
`main.py`.

### Stage 1 — Global context pretraining
Learns the assay embedding against memory banks (spatial support maps, temporal
adjacency distributions). Writes `ckpts/spatial_bias_pretrain.pt`, which is the
sole source of the gct mapper downstream.

### Stage 2A — Hierarchical VQ-VAE
Trains encoder, three-level codebook ladder, and decoder. Levels are activated
by a **staged loss-weight warm-up** rather than by scaling the levels, because a
level scale != 1 biases the EMA target.

### Stage 2B — Flatten the ladder
Sums each `(z1, z2, z3)` triple into a single embedding so the prior can predict
**one categorical** instead of three coupled ones, then **merges duplicates**.

The merge criterion is pairwise-relative, matching what Stage 2A already uses
for duplicate restarts:

```
rel = ||e_i - e_j|| / (0.5 * (||e_i|| + ||e_j||))  <  0.05
```

Relative rather than absolute because `tree_embed = ema_weight / ema_count`, so
rarely-used entries have small denominators and inflated norms — a single global
distance scale would be meaningless. Result: **1024 nominal -> 961 rows, 63
merges**, with no frequency filtering and no out-of-alphabet bin.

### Stage 3 — Local context mapper
Trains the lct trunk (9 -> 256 -> 256 -> 32) on frozen codes, scored as dNLL
against the marginal in nats/token. gct dominates on flat-code identity while
**lct dominates on textons** — that inversion is what justifies lct as a separate
stream rather than a weaker copy of gct.

### Stage 4 — Discrete prior
- **4A — motif prior.** MaskGIT-style masked prediction over the 961-entry flat
  alphabet, conditioned on gct, lct, and a task token.
- **4B — activity prior.** Predicts which token positions are active and their
  counts. Selected on **NLL, not F1**, because F1 is maximised by emitting the
  mode. This is also where the end-to-end generation validation runs
  (`iterative_unmask_motif_given_activity` -> `decode_flat_ids_to_xgen`).
- **4C — activity-map adaptation.** 4A is teacher-forced on the *true* activity
  map but is handed 4B's prediction at generation time, and that mismatch costs
  free-generation motif MRR 0.228 -> 0.155 (median rank 8 -> 21). 4C adapts 4A to
  the maps 4B actually emits, at a tenth of 4A's learning rate. **This is the
  shipped motif prior**, and it must be paired with a soft activity field at
  generation time — fed a hard 0/1 map it is worse than the unadapted 4A.
  Selected on val MRR under the model arm; the oracle arm is logged every epoch
  and never selected on.
- **4b_refine — ablation, rejected.** Attacked the same mismatch from the other
  side, moving 4B with gradients from a frozen 4A. All variation sat inside the
  0.92 seed sd and motif-MRR *declined* (t = -10.45). Retained only to show why
  4C moves the motif prior instead.

---

## Evaluation discipline

Accuracy against a ~1000-way alphabet means nothing on its own, so Stage 4A is
always scored against a **null ladder** fitted on the training split only:

| level | what it predicts |
|---|---|
| `uniform` | uniform over the alphabet |
| `global` | the global marginal code frequency |
| `assay` | the per-assay marginal |
| `assay_position` | per (assay, token position), hierarchically smoothed |

`assay_position` is the bar: it is a memorised lookup table of "what motif
usually occupies this latent location in this preparation". Smoothing toward the
assay and global levels is load-bearing — the per-cell table averages ~2.5
observations, so unsmoothed counts are noise.

Model and nulls are always scored **in a single process on the identical
supervised token set**. Null tables from different runs are not comparable,
because the supervised token set differs.

Other conventions:

- **Report tolerant and exact AUPRC together.** Tolerant matching grants a
  27-voxel slack, which systematically favours blurred predictions.
- **Do not weight the classification loss by class frequency.** The flat
  marginal has perplexity ~525 of 961 and a most/median ratio of ~12x — it is
  not long-tailed, and weighting moves the minimiser to `q ~ w*p`, which is
  wrong for a sampled prior.

---

## Repository layout

```
main.py               stage dispatch, model/prior construction, evaluation
model/                vqvae.py, prior.py, base.py
training/             per-stage training loops, stage2b_flatten.py, baselines.py
inference/            sample_prior.py, decode.py, metrics_gen.py
utils/                losses, memory banks
dataset.py            assay discovery, splits, loaders
ckpts/ reports/       checkpoints and JSON training/evaluation reports
```

---

## Project status

Active research project. Stages 1 through 3 are complete; Stage 4 is in
progress. Architecture and training strategy are under continuous development.

---

## Keywords

VQ-VAE, MaskGIT, neural spikes, sparse binary data, electrophysiology, organoid
intelligence, discrete representation learning, generative priors,
computational neuroscience
