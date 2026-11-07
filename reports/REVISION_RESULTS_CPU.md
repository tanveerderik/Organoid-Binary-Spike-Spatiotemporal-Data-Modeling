# Revision results computed on CPU from stored outputs

Three analyses that need no model, data or GPU — everything is read from per-clip
results, probability-field dumps and code counts already in `reports/`. The
experiments that need retraining are in `docs/REVIEW_EXPERIMENTS_RUNBOOK.md`.

## 1. Paired comparisons with the recording as the unit

`analysis/recording_level_stats.py` → `reports/recording_level_stats.{json,md}`

Every paired test in `reports/external_baselines` treated each of the 279 test
clips as independent. Clips are 9 per recording, and recordings 10, 11, 13 are
byte-identical copies of 4, 3, 6, so there are 28 independent recordings. Clip→
recording labels were recovered from the loader's construction (assay block
`i // 9`, sorted ids; other tasks aligned through `clip_keys`) and verified on 24
dumped clips: AP recomputed from each dumped field matched exactly one stored
clip, carrying the predicted recording (8 of them decisive two-field matches,
0 mismatches). The same 88 comparisons were re-run as a Wilcoxon on per-recording
means (n = 28) with a hierarchical bootstrap CI (recordings within tissue, then
clips), BH-corrected jointly.

| comparison | level | first better | arm better | not significant |
|---|---|---|---|---|
| ours vs learned arm (voxel + site AP, 4 tasks) | clip | 7 | 23 | 2 |
| | **recording** | **6** | **16** | **10** |
| static site map vs every arm | clip | 44 | 11 | 1 |
| | **recording** | **44** | **8** | **4** |

11 of 88 verdicts change, all from significant to not significant. The
qualitative picture holds: the 3D U-Net still beats our model on voxel AP and on
site AP on three of four tasks each, and the static site map still beats every
learned model. What changes, once recordings rather than clips are counted:

* 7 of our losses are no longer supported — 4 against the CVAE (voxel spatial;
  site recon, causal, noncausal) and 3 on the spatial task against the U-Net
  (voxel; site, both seeds);
* 1 of our wins is no longer supported (voxel AP vs MaskGIT-flat on `recon`);
* 3 site-map losses to the per-assay lookup references (DG, GLM) on site AP
  become not significant.

Also found: `task_eval_maskgit_flat_cal.json`, which `diagnose_table.py` prefers
for the MaskGIT-flat row, is a 12-batch run (48 clips), not the 279-clip
protocol. Its voxel AP (.018–.021) differs from the 279-clip run
(`task_eval_maskgit_flat.json`, .012–.016), which is the one used here.

## 2. Is the ~6× spike over-count a calibration offset?

`analysis/posthoc_calibration.py` → `reports/posthoc_calibration.json` (pilot)

* Own-count / true-count on the first test clip of 8 recordings × 4 tasks:
  geometric mean **6.8×**, SD of log ratio **0.29** — a near-constant factor with
  roughly ±30% per-clip spread.
* One scalar logit offset (≈ −2), fitted on one recording's decoded fields and
  applied to the other recording: count bias **+323…+727% → −34…+28%**, voxel
  ECE down **~13×**, AP unchanged (largest change 4e-5, tie-breaking at float
  precision).

So most of the excess is a fixable scale offset of the decoded probabilities.
Two recordings is a pilot; runbook step 4 repeats it on all 31.

## 3. Motif reuse without duplicates, and what motif usage predicts

`analysis/motif_usage_probe.py` → `reports/motif_usage_probe.json`

* Removing the three duplicate recordings barely changes reuse: recording
  identity explains 9.1% → **9.4%** of motif-usage entropy (Miller–Madow), and
  rarefied Jaccard is unchanged (within organoid .109 → .108, within slice .124 →
  .123, cross-tissue .103). The published numbers reproduce exactly with the
  duplicates kept.
* Tissue type from per-recording motif-usage profiles (n = 28, logistic on 5
  PCs of the Hellinger embedding, permutation p from 1000 label shuffles):

  | cross-validation | motif usage | rate only |
  |---|---|---|
  | hold out a whole slice **patient** (or one organoid recording) | bal. acc **0.54**, p = 0.17 | 0.59, p = 0.13 |
  | hold out one **recording** (same-patient slices stay in training) | bal. acc **0.89**, p = 0.001 | 0.71, p = 0.01 |

  Settings sweep (2/5/10 PCs × C 0.1/1/10) under patient hold-out: 0.51–0.54.
* Slice patient (10F vs 11G) from motif usage: 0.50 (p = 1.0); from rate alone:
  0.82 (p = 0.004).

Motif usage identifies a preparation when other recordings of the same
preparation are in training, but does not generalize to an unseen patient, and
it carries no patient information beyond firing rate. Tissue and patient are
both confounded with electrode routing. This is a negative result for the
"usefulness" claim at this sample size and should be reported as such.
