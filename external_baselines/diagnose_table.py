#!/usr/bin/env python3
"""Merge the per-model diagnose_*.json into one side-by-side table.

`compare_table.py` answers "which model scores better on the generation
statistics". This answers "does the model work, and how", which is the question
a reader asks first and the one a summary statistic cannot settle. Every row is
either against chance, against the `random`-context control, or against the real
value, so no number needs the reader to already know the scale.

    python external_baselines/diagnose_table.py > reports/external_baselines/diagnostics.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DIR = Path("reports/external_baselines")
MODELS = [("pipeline", "Ours (4C+soft)"), ("maskgit_flat", "MaskGIT-flat"),
          ("dg", "Dich. Gaussian"), ("glm", "Coupled GLM")]
LCT = ["log_mean_firing_density", "var_x", "var_y", "var_t", "cov_xy",
       "cov_xt", "cov_yt", "active_site_ratio", "temporal_trend"]


def load():
    out = {}
    for key, label in MODELS:
        p = DIR / f"diagnose_{key}.json"
        if p.exists():
            out[key] = json.loads(p.read_text())
    return out


def row(label, vals, fmt="{:.4f}", note=""):
    cells = " | ".join(fmt.format(v) if isinstance(v, (int, float)) and v == v
                       else "--" for v in vals)
    return f"| {label} | {cells} |" + (f" {note}" if note else "")


def main() -> int:
    D = load()
    have = [(k, lab) for k, lab in MODELS if k in D]
    head = "| | " + " | ".join(lab for _, lab in have) + " |"
    rule = "|---|" + "---|" * len(have)

    print("# Interpretable diagnostics\n")
    print("8 test batches (32 clips), seed 20260821, identical clips for every "
          "model. Generated volumes come from each model's own shipped readout.\n")

    print("## 1. Tokenizer reconstruction  (encode -> quantize -> decode)\n")
    print("Step-wise average precision, NOT the trapezoid AUPRC in "
          "`utils/metrics.py`. Trapezoid linearly interpolates the PR curve, "
          "which is invalid (Davis & Goadrich 2006) and inflated the saturating "
          "MaskGIT tokenizer by +0.22 off a single voxel. The `interp. "
          "inflation` row shows how much each model was affected.\n")
    print(head); print(rule)
    R = {k: (D[k].get("reconstruction") or {}) for k, _ in have}
    for lab, key in (("AP step-wise, exact", "ap_step_exact"),
                     ("AP step-wise, tolerant (1,1,1)", "ap_step_tol111"),
                     ("best F1, exact", "best_f1_exact"),
                     ("best F1, tolerant", "best_f1_tol111")):
        print(row(lab, [R[k].get(key, float("nan")) for k, _ in have]))
    print(row("chance (base rate)",
              [R[k].get("base_rate", float("nan")) for k, _ in have], "{:.2e}"))
    print(row("x chance (exact)",
              [R[k].get("ap_step_exact", float("nan")) /
               max(R[k].get("base_rate", 1) or 1, 1e-12) for k, _ in have],
              "{:,.0f}x"))
    print(row("interp. inflation (exact)",
              [R[k].get("auprc_exact", float("nan")) -
               R[k].get("ap_step_exact", float("nan")) for k, _ in have], "{:+.4f}"))
    print(row("codebook used",
              [R[k].get("codes_used", float("nan")) for k, _ in have], "{:.0f}"))
    print(row("codebook perplexity",
              [R[k].get("codebook_perplexity", float("nan")) for k, _ in have], "{:.1f}"))
    print("\nDG and the GLM have no tokenizer; they are point-process models "
          "and this section does not apply to them.\n")

    print("## 2. Is the generated field informative?  (the all-blank check)\n")
    print(head); print(rule)
    G = {k: D[k]["degeneracy"] for k, _ in have}
    for lab, key in (("all-blank samples", "all_blank_fraction"),
                     ("AUPRC vs its OWN clip", "auprc_vs_own_clip"),
                     ("AUPRC vs a DIFFERENT clip", "auprc_vs_other_clip"),
                     ("**clip-specific margin**", "clip_specific_margin"),
                     ("field std (flat would be ~0)", "field_std")):
        print(row(lab, [G[k].get(key, float("nan")) for k, _ in have]))

    print("\n## 3. Local context adherence  (9 lct features recomputed "
          "from the sample, r vs the vector the model was GIVEN)\n")
    print("| feature | " + " | ".join(f"{lab} rnd -> full" for _, lab in have) + " |")
    print("|---|" + "---|" * len(have))
    for f in LCT:
        cells = []
        for k, _ in have:
            reg = D[k]["context_and_space"]["regimes"]
            a = reg["random"]["lct_per_feature"][f]["r"]
            b = reg["global_full_local"]["lct_per_feature"][f]["r"]
            cells.append(f"{a:+.2f} -> {b:+.2f}")
        print(f"| {f} | " + " | ".join(cells) + " |")
    print("\n`log_mean_firing_density` is CIRCULAR for DG and MaskGIT-flat -- "
          "their firing rate is regressed directly from lct, so a high r there "
          "measures the regression, not the model.\n")

    print("## 4. Spatial map  (per-electrode counts vs the true clip)\n")
    print(head); print(rule)
    for lab, key in (("pearson r, random ctx", "map_pearson_r"),
                     ("pearson r, full ctx", "map_pearson_r"),
                     ("vs other clip, SAME assay (full)", "map_r_other_clip_same_assay"),
                     ("vs a different assay (full)", "map_r_other_assay"),
                     ("**within-assay gap**", "within_assay_gap"),
                     ("assay-identity component", "assay_identity_component"),
                     ("active-site IoU, full ctx", "active_site_iou")):
        regime = "random" if "random" in lab else "global_full_local"
        print(row(lab, [D[k]["context_and_space"]["regimes"][regime].get(key, float("nan"))
                        for k, _ in have]))
    print("\n`assay_idx` reaches every model unchanged in EVERY regime -- the "
          "ladder randomises gct/lct, not assay identity. DG and the GLM key "
          "their train-fitted site maps on it, so their `random` rung is not a "
          "control and their pearson r is mostly a per-assay lookup. The "
          "**within-assay gap** -- own clip minus a different clip from the "
          "same assay -- is the only row a fixed site map cannot fake.\n")

    print("## 5. Adjacency  P(spike at neighbour | spike), full context\n")
    print("| | REAL | " + " | ".join(lab for _, lab in have) + " |")
    print("|---|---|" + "---|" * len(have))
    labels = D[have[0][0]]["context_and_space"]["adjacency_labels"]
    real = D[have[0][0]]["context_and_space"]["adjacency_real"]
    for i, lab in enumerate(labels):
        cells = []
        for k, _ in have:
            v = D[k]["context_and_space"]["regimes"]["global_full_local"]["adjacency"][i]
            ratio = v / real[i] if real[i] else float("nan")
            cells.append(f"{v:.5f} ({ratio:.1f}x)")
        print(f"| {lab} | {real[i]:.5f} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
