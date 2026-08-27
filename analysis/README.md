# analysis/

Post-hoc artifacts for the Stage 4 generation composite. Nothing here trains
anything.

The composite is **no longer the paper metric and no longer selects epochs** --
4B selects on val NLL, 4C on val MRR. It remains the **acceptance gate**, and
`reports/generation_metric_reference.json` is consumed by
`analysis/generate_regimes.py` and `external_baselines/common/pipeline.py`, so
these scripts stay live.

| script | needs GPU | answers |
|---|---|---|
| `generation_metric_reference.py` | no | what does a score of 0.63 mean? |
| `generation_weight_sensitivity.py` | no | do the hand-set weights matter? |
| `generation_seed_spread.py` | **yes** | how much of a difference is just sampling noise? |

The first two are model-free and safe to run against a busy GPU. The third runs
the real sampling path.

## generation_metric_reference.py

Establishes what the composite's scale means. The score is a number in [0,1]
with no intrinsic interpretation, which is the first thing a reviewer asks
about. This scores a ladder of controlled degradations through the same
`mea_statistics` / `compare_statistics` path the composite uses:

| rung | what it keeps | what it destroys |
|---|---|---|
| real-vs-real | everything (ceiling) | nothing -- finite-sample noise only |
| temporal shuffle | rate, per-electrode marginals | persistence, avalanches, ISI |
| spatial shuffle | rate, temporal structure | spatial co-activation |
| mean-field | rate only (floor) | everything else |

Measured on the test split, 96 clips, 2026-08-20:

| rung | stat_error | ks_av | ks_isi | subscore |
|---|---|---|---|---|
| CEILING real-vs-real | 0.1060 +- 0.0239 | 0.0930 | 0.0379 | **0.9150** |
| temporal shuffle | 0.2429 | 0.2088 | 0.0836 | 0.8199 |
| spatial shuffle | 0.7728 | 0.0957 | 0.4455 | 0.6475 |
| FLOOR mean-field | 0.9659 | 0.7229 | 0.5254 | **0.4451** |

Only the 0.60 of the composite that compares generated statistics to real ones
is scored -- the remaining 0.40 are model-vs-target consistency terms with no
real-vs-real analogue, and are excluded rather than assumed perfect. The
reported subscore is that 0.60 renormalised to [0,1].

The ceiling's spread across splits (sd 0.0239) is the resolution limit: two
checkpoints closer than that on the distribution terms are not distinguishable.

    python analysis/generation_metric_reference.py --split test --batches 24

## generation_weight_sensitivity.py

The composite's nine weights are hand-set and there is no derivation for 0.35
over 0.30. The defence is evidence, not argument: show the selected checkpoint
does not depend on the exact weights.

1. **Recompute check** -- rebuilds the composite from its logged components and
   requires an exact match against the logged value. If the term table drifts
   out of sync with `training/stage4_activity.py`, everything downstream is
   measuring a different metric, so this aborts rather than warns.
2. **Dirichlet perturbation** -- resamples the weight vector and records which
   epoch each perturbation selects, plus the *regret*: what choosing that epoch
   would have cost measured under the nominal weights. Regret is the number
   that matters. A wandering argmax among epochs that are all within noise is
   not a problem; a small shift that costs real composite is.
3. **Leave-one-out** -- drops each term, renormalises, re-selects.

    python analysis/generation_weight_sensitivity.py \
        --report reports/training_report_prior_3B_activity.json

Run both once 4B and 4C have finished.

## generation_seed_spread.py

The composite is scored by *sampling* over a capped number of validation
batches, so it carries stochastic noise of its own. Two checkpoints separated by
less than that noise are indistinguishable, and any claim that one beat another
needs this number attached.

Scores one fixed checkpoint repeatedly, varying only the sampling seed.
Validation masks stay deterministic, so the measured spread is MaskGIT decoding
noise alone -- not data variation, not mask variation.

    python analysis/generation_seed_spread.py --phase 4b --seeds 8

`--activity-ckpt` / `--motif-ckpt` override checkpoint selection, for scoring a
specific candidate rather than whichever one the phase would pick.

Judge the result against the reference ladder above: if the seed spread is
comparable to the real-vs-real ceiling spread (sd 0.0239 on the distribution
subscore), the metric is at its resolution limit and finer comparisons are not
supportable. Report both numbers together.

**Needs the GPU** -- it runs activity -> MaskGIT -> VQ-VAE for real. Defaults are
deliberately small; raise `--batches` only when the GPU is free.

## Suggested order, once 4B and 4C have finished

    python analysis/generation_metric_reference.py --split test
    python analysis/generation_weight_sensitivity.py --report reports/training_report_prior_3B_activity.json
    python analysis/generation_seed_spread.py --phase 4b --seeds 8
    python analysis/generation_seed_spread.py --phase 4b_refine --seeds 8

The first is already run; its output is in
`reports/generation_metric_reference.json`.
