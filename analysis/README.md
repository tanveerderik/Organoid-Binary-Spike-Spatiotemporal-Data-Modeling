# analysis/

Post-hoc artifacts for defending the Stage 4 generation composite. Neither
script trains anything or touches the GPU; both run against real data or
against a training report that already exists.

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
        --report reports/training_report_prior_4B.json

Run both once 4B and 4C have finished.
