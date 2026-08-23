#!/usr/bin/env python3
"""Two derived quantities the tables need, computed once and written to JSON.

`diagnose_table.py` never computes and never carries a typed number, so
anything that is not already a field in a run's report has to be derived here
and read from disk.

1. STATIC LOOKUP vs each learned arm, paired, per task.
   The voxel-AP column is topped by the 3D U-Net, and the fact that matters for
   reading it is that a model-free per-assay site map -- no clip-specific
   information whatsoever -- beats that arm on nearly every clip. Rendering it
   as a table row turns an argument in prose into something checkable.
   The null comes from the CANONICAL `task_eval_nulls.json`, never from a
   model run's own null columns, which are RNG-contaminated.

2. GAP-SHAPE correlation against the real short-gap profile.
   The adjacency table reports each arm's rate per gap bin, and an arm can sit
   close to REAL on every row by being FLAT while the real curve has a definite
   shape (a dip at gap 1, a peak at 3-6). Correlating the mean-centred,
   scaled curves separates "matches the level" from "matches the shape", and
   the two are different claims.

    python external_baselines/derived_stats.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

DIR = Path("reports/external_baselines")
OUT = DIR / "derived_stats.json"
ARMS = [("pipeline", "task_eval.json"),
        ("maskgit_flat", "task_eval_maskgit_flat_cal.json"),
        ("unet3d", "task_eval_unet3d.json"),
        ("cvae3d", "task_eval_cvae3d.json")]
MODES = ("recon", "causal", "noncausal", "spatial")


def _bh(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps); q = np.empty_like(ps); n = len(ps); prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, ps[o[i]] * n / (i + 1)); q[o[i]] = prev
    return q


def lookup_vs_arms() -> dict:
    NJ = json.loads((DIR / "task_eval_nulls.json").read_text())
    rows, ps = [], []
    for mode in MODES:
        nul = np.array(NJ["tasks"][mode]["per_clip"]["marginal"], float)
        for key, fn in ARMS:
            p = DIR / fn
            if not p.is_file():
                continue
            J = json.loads(p.read_text())
            arm = np.array(J["tasks"][mode]["per_clip"]["model_mc"], float)
            if arm.size != nul.size:
                rows.append({"task": mode, "arm": key,
                             "skipped": "clip count differs"})
                continue
            # Pairing check. The model reports carry no clip_keys, so alignment
            # is verified on clip-SPECIFIC properties instead of assumed: AP is
            # NaN exactly where a clip has no spike in the hole, and that
            # pattern is a property of the clip, not of the model.
            if not np.array_equal(np.isnan(arm), np.isnan(nul)):
                rows.append({"task": mode, "arm": key,
                             "skipped": "NaN masks differ; clips not aligned"})
                continue
            m = np.isfinite(arm) & np.isfinite(nul)
            d = nul[m] - arm[m]                       # >0 => lookup wins
            st, pv = wilcoxon(nul[m], arm[m])
            ps.append(pv)
            rows.append({"task": mode, "arm": key, "n": int(m.sum()),
                         "lookup_mean": float(nul[m].mean()),
                         "arm_mean": float(arm[m].mean()),
                         "median_delta": float(np.median(d)),
                         "lookup_win_frac": float((d > 0).mean()),
                         "p": float(pv)})
    q = _bh(ps); i = 0
    for r in rows:
        if "p" in r:
            r["q"] = float(q[i]); i += 1
    return {"rows": rows,
            "null_source": "task_eval_nulls.json (canonical, model-free)",
            "pairing_check": "NaN mask equality per task"}


def gap_shape() -> dict:
    """Shape correlation of each arm's gap profile with REAL, at full context."""
    from diagnose_table import MODELS, RUNGS          # labels, never restated
    out = {"rung": "global_full_local", "arms": {}}
    ref = None
    for key, lab in MODELS:
        p = DIR / f"diagnose_{key}.json"
        if not p.is_file():
            continue
        J = json.loads(p.read_text())
        cs = J["context_and_space"]
        if ref is None:
            ref = np.array(cs["adjacency_real"], float)
            out["labels"] = cs["adjacency_labels"]
            out["real"] = ref.tolist()
        r = cs["regimes"].get(out["rung"])
        if r is None:
            continue
        v = np.array(r["adjacency"], float)
        if v.shape != ref.shape or v.std() == 0 or ref.std() == 0:
            continue
        zs = lambda a: (a - a.mean()) / a.std()
        out["arms"][key] = {
            "label": lab,
            "shape_r": float(np.corrcoef(zs(v), zs(ref))[0, 1]),
            "relief": float((v.max() - v.min()) / max(v.mean(), 1e-12)),
            "gap1_to_gap3_pct": float((v[2] / max(v[0], 1e-12) - 1) * 100),
            "level_abs_err": float(abs(v.mean() - ref.mean()) / ref.mean()),
        }
    out["real_relief"] = float((ref.max() - ref.min()) / ref.mean())
    out["real_gap1_to_gap3_pct"] = float((ref[2] / ref[0] - 1) * 100)
    return out


def main() -> int:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    d = {"lookup_vs_arms": lookup_vs_arms(), "gap_shape": gap_shape()}
    OUT.write_text(json.dumps(d, indent=1))

    print("STATIC per-assay lookup vs each learned arm (positive = lookup wins)")
    print(f"{'task':<11}{'arm':<14}{'lookup':>9}{'arm':>9}{'medDelta':>10}"
          f"{'win%':>7}{'q':>11}")
    print("-" * 71)
    for r in d["lookup_vs_arms"]["rows"]:
        if "skipped" in r:
            print(f"{r['task']:<11}{r['arm']:<14}  SKIPPED: {r['skipped']}")
            continue
        print(f"{r['task']:<11}{r['arm']:<14}{r['lookup_mean']:>9.4f}"
              f"{r['arm_mean']:>9.4f}{r['median_delta']:>+10.4f}"
              f"{100*r['lookup_win_frac']:>6.0f}%{r['q']:>11.2e}")

    g = d["gap_shape"]
    print(f"\nGAP-SHAPE vs REAL at {g['rung']}  "
          f"(real relief {g['real_relief']:.3f}, "
          f"gap1->gap3 {g['real_gap1_to_gap3_pct']:+.0f}%)")
    print(f"{'arm':<16}{'shape r':>9}{'relief':>9}{'gap1->3':>10}{'level err':>11}")
    print("-" * 55)
    for k, v in sorted(g["arms"].items(), key=lambda z: -z[1]["shape_r"]):
        print(f"{v['label']:<16}{v['shape_r']:>9.3f}{v['relief']:>9.3f}"
              f"{v['gap1_to_gap3_pct']:>9.0f}%{v['level_abs_err']:>11.3f}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
