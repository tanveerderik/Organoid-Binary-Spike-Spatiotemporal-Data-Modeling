# Runbook: descriptor-source, count-control and calibration experiments

Everything here needs the GPU machine: the burst-window data under
`../output_data`, the cache `../_cache_spike_thw_run1`, and the full `ckpts/`.
Steps are ordered by cost. Each step lists what it writes; commit those files
(not checkpoints) when it finishes.

Already done on CPU from stored results, so it is **not** repeated here:

| analysis | script | output |
|---|---|---|
| paired comparisons re-tested per recording (duplicates merged), hierarchical bootstrap | `analysis/recording_level_stats.py` | `reports/recording_level_stats.{json,md}` |
| post-hoc count calibration, pilot on 2 recordings | `analysis/posthoc_calibration.py` | `reports/posthoc_calibration.json` |
| motif reuse without duplicate recordings; tissue / patient probes | `analysis/motif_usage_probe.py` | `reports/motif_usage_probe.json` |

## What changed in the code

* `CTX_SOURCE` environment variable (`full` | `visible` | `none`, default `full`),
  read by `main.py`. It sets which voxels the 9-d descriptor `lct` is computed
  from, in training (`NpzBurstDataset(ctx_source=...)`, and the Stage 4B loop
  after its scattered-mask substitution) and in evaluation
  (`task_eval.py --ctx-source`, which defaults to `$CTX_SOURCE` and refuses to
  disagree with it). Under `visible`/`none` every context-dependent checkpoint and
  report gets a `_ctx<source>` suffix, so the shipped files are never overwritten.
  The tokenizer (Stage 2A/2B) only uses `lct` as a loss target and is shared.
* `TRAIN_STAGES`, `EVAL_STAGES`, `STAGE4_PHASES` can be set from the environment.
* `canonical_nulls.py` count control has three model-free arms:
  `lct_density_x_roi` (the old one), `lct_exact` (= total from the whole-clip
  descriptor − visible spikes; recovers the hidden count exactly) and
  `visible_density_x_roi` (the fair control for a visible-only descriptor).
* Stage 3 now early-stops and selects on the **validation** loader (it used the
  test loader).
* `common/evaluate.py` ladder flags come from `ladder.CTX_FLAGS` (`local_only` was
  mislabelled).
* `analysis/generation_metric_reference.py` gains an `electrode circular shift`
  rung (per-electrode ISI kept, synchrony broken), on its own RNG so the other
  rungs reproduce; the docstring now says what the temporal shuffle actually does.

All commands below run from the repository root on the desktop:

```bash
cd "/media/derik/Seagate Desktop Drive/organoid_data/MAGVIT_project"
```

```bash
git fetch origin claude/dazzling-cannon-oceifh && git checkout claude/dazzling-cannon-oceifh
```

## 0. Preflight (minutes)

```bash
nvidia-smi
```

```bash
ls ../output_data ../_cache_spike_thw_run1 | head && ls ckpts/*.pt | wc -l
```

```bash
PYTHONPATH=.. python -m pytest -q tools/tests/test_ctx_source.py tools/tests/test_stats.py tools/tests/test_extract_hparams.py
```

## 1. Reproduction gate (about the length of one task_eval run)

Nothing below is trusted until the unchanged default path reproduces the shipped
per-clip numbers.

```bash
python external_baselines/task_eval.py --model pipeline --batches 70 --mc 8 --out /tmp/repro_pipeline.json
```

```bash
python - <<'EOF'
import json, numpy as np
a = json.load(open("/tmp/repro_pipeline.json")); b = json.load(open("reports/external_baselines/task_eval_pipeline.json"))
for t in ("recon", "causal", "noncausal", "spatial"):
    x, y = (np.array(d["tasks"][t]["per_clip"]["model_mc"], float) for d in (a, b))
    print(t, x.shape == y.shape and np.allclose(x, y, atol=1e-6, equal_nan=True), np.nanmax(np.abs(x - y)))
EOF
```

If any task prints `False`, stop: sampling is not reproducing and every later
comparison would mix two effects.

## 2. Exact count control (one pass over the test split)

```bash
python external_baselines/canonical_nulls.py --batches 70
```

Writes `reports/external_baselines/task_eval_nulls.json`. The null arms are
model-free and must come out identical to the committed file (check with
`git diff --stat`; only `count_control_lct_only` should change). Expected:
`lct_exact` r ≈ +1.000 on every task. Re-render the diagnostics tables
afterwards so the new rows appear: `python external_baselines/diagnose_table.py`.

## 3. Dose-response of the descriptor (no training)

```bash
CTX_SOURCE=full python analysis/context_dose_response.py --batches 24
```

Writes `reports/context_dose_response.json`. Slope ≈ 1 means the generated
statistic follows the requested one; ≈ 0 means the model ignores it.

## 4. Calibration on every recording (one task_eval run)

Back up the committed dump first; `--dump` overwrites it.

```bash
cp reports/external_baselines/dumps/pipeline.npz /tmp/pipeline_dump_shipped.npz
```

```bash
python external_baselines/task_eval.py --model pipeline --batches 70 --mc 8 --dump 31 --dump-field 31 --out /tmp/task_eval_pipeline_dump31.json
```

```bash
mv reports/external_baselines/dumps/pipeline.npz /tmp/pipeline_dump31.npz && cp /tmp/pipeline_dump_shipped.npz reports/external_baselines/dumps/pipeline.npz
```

```bash
python analysis/posthoc_calibration.py --dump /tmp/pipeline_dump31.npz --out reports/posthoc_calibration_31rec.json
```

The 31-recording dump is ~300 MB; keep it out of git.

## 5. Visible-only descriptor regime (≈ a day of GPU)

Retrain the context-dependent stages; Stage 1 and the tokenizer are reused.

```bash
CTX_SOURCE=visible TRAIN_STAGES=3,4 STAGE4_PHASES=4a,4b,4c python main.py
```

Refit every baseline under the same descriptor, then score all arms:

```bash
for b in unet3d cvae3d maskgit_flat dg glm; do CTX_SOURCE=visible python external_baselines/run_baseline.py --baseline $b --fit --tag ctxvisible; done
```

```bash
CTX_SOURCE=visible python external_baselines/task_eval.py --model pipeline --batches 70 --mc 8
```

```bash
for b in unet3d cvae3d maskgit_flat dg glm; do CTX_SOURCE=visible python external_baselines/task_eval.py --model $b --batches 70 --mc 8 --ckpt ckpts/external_baselines/${b}_ctxvisible.pt; done
```

```bash
python analysis/recording_level_stats.py --suffix _ctxvisible
```

Writes `reports/external_baselines/task_eval_*_ctxvisible.json`,
`reports/recording_level_stats_ctxvisible.{json,md}`, and the suffixed training
reports under `reports/`.

Note: free generation (task 0) hides every voxel, so under `visible` it receives
a constant descriptor — the R-vis row for task 0 is an unconditional baseline.
The generation context bank (`ckpts/context_prior.pkl`) is not rebuilt.

## 6. No-descriptor regime

Same as step 5 with `CTX_SOURCE=none` and `--tag ctxnone` / `--suffix _ctxnone`.

## 7. Ablation ladder, rung 0: budget-matched MaskGIT-flat

The shipped MaskGIT-flat trained 12 tokenizer / 30 prior epochs against the
pipeline's 300 / 600. Time one epoch first and scale if needed.

```bash
python external_baselines/run_baseline.py --baseline maskgit_flat --fit --tag budget --config '{"tok_epochs": 300, "prior_epochs": 600}'
```

```bash
python external_baselines/task_eval.py --model maskgit_flat --batches 70 --mc 8 --ckpt ckpts/external_baselines/maskgit_flat_budget.pt --out reports/external_baselines/task_eval_maskgit_flat_budget.json
```

Rungs 2 (blank routing without the blank losses) and 4 (single joint prior on
our tokenizer's 961 + blank ids) need new code and are not in this branch yet.

## 8. Commit

```bash
git add reports/external_baselines/task_eval_nulls.json reports/context_dose_response.json reports/posthoc_calibration_31rec.json reports/external_baselines/task_eval_*_ctx*.json reports/recording_level_stats_ctx*.json reports/recording_level_stats_ctx*.md reports/*_ctxvisible.json reports/*_ctxnone.json reports/external_baselines/task_eval_maskgit_flat_budget.json
```

```bash
git commit -m "Add descriptor-source, count-control and calibration results" && git push -u origin claude/dazzling-cannon-oceifh
```
