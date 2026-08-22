#!/usr/bin/env python3
"""Merge the per-model diagnose_*.json into one side-by-side report.

`compare_table.py` answers "which model scores better on the pooled generation
statistics". That question turned out to be nearly unanswerable -- the pooled
statistics cannot see conditioning at all, so a model that ignores its context
and emits the dataset average scores well on them.

This answers the two questions that ARE answerable, and keeps them apart
because no model wins both:

    CONDITIONAL   given this much context, does the sample match THIS clip?
    MARGINAL      does the sample look like real data at all?

Four families, no composite. Each is one number with a stated null, and the
conditional ones are per-clip so they admit a paired test.

    python external_baselines/diagnose_table.py > reports/external_baselines/diagnostics.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

DIR = Path("reports/external_baselines")
MODELS = [("pipeline", "Ours (4C+soft)"), ("maskgit_flat", "MaskGIT-flat"),
          ("dg", "Dich. Gaussian"), ("glm", "Coupled GLM")]
RUNGS = [("random", "random"), ("local_only", "LOCAL only"),
         ("global_only", "GLOBAL only"), ("global_partial_local", "glob+partial"),
         ("global_full_local", "glob+full")]
LCT = ["log_mean_firing_density", "var_x", "var_y", "var_t", "cov_xy",
       "cov_xt", "cov_yt", "active_site_ratio", "temporal_trend"]



def _mark(vals, fmt, better, ref=None):
    """Format a list of numbers, bolding the best.

    `better` is "high", "low", or "near" -- "near" means closest to `ref`, which
    is what "best" means for a co-firing probability: matching the real value,
    not maximising or minimising it. Ties are all bolded, and a row where "best"
    is meaningless (codebook size, perplexity) passes better=None.
    """
    out, keys = [], []
    for v in vals:
        ok = isinstance(v, (int, float)) and v == v
        out.append(fmt.format(v) if ok else "--")
        if not ok or better is None:
            keys.append(None)
        elif better == "high":
            keys.append(-float(v))
        elif better == "low":
            keys.append(float(v))
        else:
            keys.append(abs(float(v) - ref))
    live = [k for k in keys if k is not None]
    if live:
        best = min(live)
        out = [f"**{t}**" if k is not None and k == best else t
               for t, k in zip(out, keys)]
    return out


def main() -> int:
    J, have = {}, []
    for k, lab in MODELS:
        p = DIR / f"diagnose_{k}.json"
        if p.exists():
            J[k] = json.loads(p.read_text()); have.append((k, lab))
    rungs = [(r, s) for r, s in RUNGS
             if r in J[have[0][0]]["context_and_space"]["regimes"]]

    def reg(k, r):
        return J[k]["context_and_space"]["regimes"][r]

    real = np.array(J[have[0][0]]["context_and_space"]["adjacency_real"], float)
    labels = J[have[0][0]]["context_and_space"]["adjacency_labels"]

    def ladder(title, note, fn, better, fmt="{:.4f}"):
        """Bold the best MODEL in each rung, i.e. down each column."""
        print(f"\n### {title}\n\n{note}\n")
        print("| model | " + " | ".join(s for _, s in rungs) + " |")
        print("|---|" + "---|" * len(rungs))
        cols = [_mark([fn(k, r) for k, _ in have], fmt, better) for r, _ in rungs]
        for i, (k, lab) in enumerate(have):
            print(f"| {lab} | " + " | ".join(c[i] for c in cols) + " |")

    print("# Interpretable diagnostics\n")
    print("8 test batches (32 clips), seed 20260821, identical clips for every "
          "model, each model's own shipped readout. `local_only` is a control, "
          "not a rung: it hands the model the true lct with a MISMATCHED gct, so "
          "it is not 'less information' than `random` but contradictory "
          "information.\n")

    # ---- reconstruction ------------------------------------------------
    print("## Reconstruction\n")
    print("Step-wise average precision, not the trapezoid AUPRC in "
          "`utils/metrics.py` -- trapezoid interpolates the PR curve linearly, "
          "which is invalid (Davis & Goadrich 2006) and inflated the saturating "
          "MaskGIT tokenizer by +0.22 off a single voxel.\n")
    R = {k: (J[k].get("reconstruction") or {}) for k, _ in have}
    print("| | " + " | ".join(lab for _, lab in have) + " |")
    print("|---|" + "---|" * len(have))
    for lab, key, f, better in (
            ("AP step-wise, exact", "ap_step_exact", "{:.4f}", "high"),
            ("AP step-wise, tolerant", "ap_step_tol111", "{:.4f}", "high"),
            ("best F1, exact", "best_f1_exact", "{:.4f}", "high"),
            ("best F1, tolerant", "best_f1_tol111", "{:.4f}", "high"),
            # Not a score: how far each model's trapezoid AUPRC sits above its
            # own step-wise AP. Nearest zero is the trustworthy one.
            ("trapezoid inflation", None, "{:+.4f}", "low"),
            # Descriptive, not better-or-worse -- a smaller codebook that
            # reconstructs as well is not losing.
            ("codebook used", "codes_used", "{:.0f}", None),
            ("codebook perplexity", "codebook_perplexity", "{:.1f}", None)):
        vals = [(R[k].get("auprc_exact", float("nan")) - R[k].get("ap_step_exact", float("nan"))
                 if key is None else R[k].get(key, float("nan"))) for k, _ in have]
        print(f"| {lab} | " + " | ".join(_mark(vals, f, better)) + " |")
    print("\nDG and the GLM are point processes with no tokenizer.\n")

    print("## Generation\n")
    ladder(
        "A. Conditional accuracy  (the headline)",
        "Per-clip z-scored MAE between the lct recomputed from the sample and "
        "the TRUE clip's lct, each feature divided by its spread across test "
        "clips. **Lower is better.** Per-clip, so a paired Wilcoxon applies. "
        "The `random` column is the null.",
        lambda k, r: np.mean(reg(k, r)["lct_z_mae_per_clip"]), "low")
    ladder(
        "B. Adherence  (does it do what it is told)",
        "Mean over the 9 features of r(realised, **requested**) -- against the "
        "lct handed to the model, not the true one. Higher is better. This is a "
        "property of the model, not of how much context it got, so a model that "
        "obeys should be flat across the ladder.",
        lambda k, r: reg(k, r)["lct_r_mean_vs_used"], "high")
    ladder(
        "C. Spatial placement, lookup-proof",
        "Map correlation against the clip's own electrodes MINUS the same "
        "generated map scored against a different clip of the SAME assay. "
        "`assay_idx` bypasses the ladder, so raw map r is mostly a per-assay "
        "lookup for DG and the GLM; this difference is the part a fixed site "
        "map cannot fake. Higher is better.",
        lambda k, r: reg(k, r)["within_assay_gap"], "high")

    def sre(k, r):
        a = np.array(reg(k, r)["adjacency"], float)
        return float(np.nanmean(np.abs(a - real) / (a + real)))

    ladder(
        "D. Marginal realism",
        "Mean symmetric relative error `|gen-real|/(gen+real)` of the "
        "co-firing profile over 3 spatial displacements and 7 temporal lags. "
        "Bounded in [0,1]: 0 matches real exactly, 1 is a total miss. Bounded "
        "on purpose -- a log-ratio explodes when a model emits exactly zero "
        "co-firing at some offset, which ours does at d=3. **Lower is better**, "
        "and this one does not depend on conditioning.",
        sre, "low")

    print("\n### Adjacency profile at full context  P(spike at neighbour | spike)\n")
    print("Best = closest to REAL, not largest or smallest.\n")
    print("| offset | REAL | " + " | ".join(lab for _, lab in have) + " |")
    print("|---|---|" + "---|" * len(have))
    for i, lab in enumerate(labels):
        vals = [reg(k, "global_full_local")["adjacency"][i] for k, _ in have]
        marked = _mark(vals, "{:.5f}", "near", ref=real[i])
        cells = [m if not real[i] else f"{m} ({v/real[i]:.2f}x)"
                 for m, v in zip(marked, vals)]
        print(f"| {lab} | {real[i]:.5f} | " + " | ".join(cells) + " |")

    print("\n### Per-feature lct, ours\n")
    print("| feature | " + " | ".join(s for _, s in rungs) + " |")
    print("|---|" + "---|" * len(rungs))
    for f in LCT:
        print(f"| {f} | " + " | ".join(
            f"{reg('pipeline', r)['lct_per_feature'][f]['r']:+.2f}"
            for r, _ in rungs) + " |")
    print("\nr against the REQUESTED lct. The spatial-shape features (var_x, "
          "var_y, cov_xy) collapse under `LOCAL only` while rate and trend "
          "survive: the electrode layout arrives through gct, so a mismatched "
          "gct makes a requested spatial variance physically unrealisable.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
