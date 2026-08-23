#!/usr/bin/env python3
"""Paired tests BETWEEN models on the task axis. Everything else here is within.

`task_eval.py` already pairs each model against its own nulls. Nothing paired
one MODEL against another, so the comparison a reader actually makes -- "does
the U-Net beat the pipeline?" -- was being read off two independent point
estimates. With n=47-48 clips and AP spreads this wide, that is not a
comparison, it is two numbers next to each other.

    python external_baselines/cross_model_tests.py

Writes `reports/external_baselines/cross_model_tests.json`.

## Pairing is checked, not assumed

Index-pairing across two JSONs is only valid if both runs saw the same clips in
the same order. There is no clip identifier in the payload to join on, so this
verifies it a different way: `per_clip["marginal"]` is a MODEL-FREE arm computed
from the ground truth of whichever clips were scored. If two runs agree on it
bitwise, they saw the same clips in the same order. If they disagree, the pair
is dropped rather than tested -- a wrong pairing produces a confident and
meaningless p-value, which is worse than a missing row.

## Multiplicity

Two metric families x four tasks x every learned arm against the pipeline, all
corrected together with Benjamini-Hochberg. Correcting each family separately
would be choosing the denominator after seeing the numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

DIR = Path("reports/external_baselines")
REF = "pipeline"
ARMS = [("maskgit_flat", "MaskGIT-flat"), ("unet3d", "3D U-Net (det.)"),
        ("cvae3d", "3D CVAE"), ("unet3d_seed2", "3D U-Net seed 2")]
TASKS = ("recon", "causal", "noncausal", "spatial")
FAMILIES = (("per_clip", "voxel AP"), ("per_clip_site", "site AP"))
ARM_KEY = "model_mc"


def _load(name):
    f = DIR / f"task_eval_{name}.json"
    return json.loads(f.read_text()) if f.is_file() else None


def _bh(ps):
    """Benjamini-Hochberg q-values, order preserved."""
    p = np.asarray(ps, float)
    n = p.size
    o = np.argsort(p)
    q = np.empty(n)
    run = 1.0
    for rank, i in enumerate(o[::-1]):
        run = min(run, p[i] * n / (n - rank))
        q[i] = run
    return q


def main() -> int:
    J = {REF: _load(REF)}
    if J[REF] is None:
        raise SystemExit(f"no {REF} task_eval payload")
    for k, _ in ARMS:
        d = _load(k)
        if d is not None:
            J[k] = d

    rows, skipped = [], []
    for fam, famlab in FAMILIES:
        for t in TASKS:
            ref = J[REF]["tasks"][t][fam]
            a = np.asarray(ref[ARM_KEY], float)
            base = np.asarray(ref["marginal"], float)
            for k, lab in ARMS:
                if k not in J:
                    continue
                cur = J[k]["tasks"][t][fam]
                b = np.asarray(cur[ARM_KEY], float)
                chk = np.asarray(cur["marginal"], float)
                if chk.shape != base.shape or not np.array_equal(chk, base):
                    skipped.append({"family": famlab, "task": t, "arm": lab,
                                    "reason": "clip sets differ -- the "
                                              "model-free marginal arm does "
                                              "not match, so index pairing "
                                              "would be wrong"})
                    continue
                d = a - b                       # positive = pipeline wins
                if np.allclose(d, 0):
                    stat, p = float("nan"), 1.0
                else:
                    stat, p = stats.wilcoxon(a, b)
                rows.append({
                    "family": famlab, "task": t, "arm": lab, "n": int(a.size),
                    "pipeline": float(a.mean()), "arm_value": float(b.mean()),
                    "median_paired_delta": float(np.median(d)),
                    "mean_paired_delta": float(d.mean()),
                    "pipeline_wins_frac": float((d > 0).mean()),
                    "ratio_arm_over_pipeline": (float(b.mean() / a.mean())
                                                if a.mean() else float("nan")),
                    "p": float(p),
                })

    for r, q in zip(rows, _bh([r["p"] for r in rows])):
        r["q"] = float(q)
        r["verdict"] = ("pipeline" if r["q"] < 0.05 and r["median_paired_delta"] > 0
                        else "arm" if r["q"] < 0.05 and r["median_paired_delta"] < 0
                        else "ns")

    hdr = (f"{'family':10s}{'task':11s}{'arm':18s}{'ours':>8s}{'arm':>8s}"
           f"{'medDelta':>10s}{'win%':>7s}{'q':>10s}  verdict")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['family']:10s}{r['task']:11s}{r['arm']:18s}"
              f"{r['pipeline']:8.4f}{r['arm_value']:8.4f}"
              f"{r['median_paired_delta']:+10.4f}"
              f"{100*r['pipeline_wins_frac']:6.0f}%{r['q']:10.2e}  "
              f"{r['verdict']}")
    for s in skipped:
        print(f"  [skipped] {s['family']} {s['task']} {s['arm']}: {s['reason']}")

    n_arm = sum(r["verdict"] == "arm" for r in rows)
    n_us = sum(r["verdict"] == "pipeline" for r in rows)
    n_ns = sum(r["verdict"] == "ns" for r in rows)
    print(f"\nafter BH-FDR over {len(rows)} tests: arm wins {n_arm}, "
          f"pipeline wins {n_us}, not significant {n_ns}")

    f = DIR / "cross_model_tests.json"
    f.write_text(json.dumps(
        {"reference": REF, "arm_key": ARM_KEY, "rows": rows,
         "skipped": skipped,
         "pairing_check": "per_clip['marginal'] is model-free; bitwise equality "
                          "across two runs establishes the same clips in the "
                          "same order",
         "multiplicity": f"Benjamini-Hochberg over all {len(rows)} tests jointly",
         "summary": {"arm_wins": n_arm, "pipeline_wins": n_us,
                     "not_significant": n_ns}}, indent=1))
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
