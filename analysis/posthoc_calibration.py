#!/usr/bin/env python3
"""Is the ~6x over-count of our decoded field a calibration offset or a modelling failure?

The pipeline's own expected spike count -- the ROI sum of the decoded probability
field -- over-predicts by +613% to +652% on every task (diagnostics.md, count
calibration). If the excess is a near-constant factor across recordings and tasks,
a single post-hoc correction fitted on held-out data removes it without touching
any ranking metric (a monotone transform leaves AP unchanged). If the factor varies
by recording, the problem is in the model and needs retraining.

This is a feasibility check on what the repository already stores, with no model,
data or GPU:

* Ratio stability: `dumps/pipeline.npz` holds, for the first test clip of 8
  recordings and all 4 tasks, the ground-truth volume, the ROI, and the volume
  completed at the model's OWN predicted count. The spread of log(own / true)
  across recordings says whether one scale factor could serve all of them.
* Held-out logit offset: the same file holds the full decoded probability field
  for 2 of those recordings (assays 0 and 1) per task. A scalar logit offset b,
  p' = sigmoid(logit(p) + b), is fitted on one recording's fields so the expected
  count matches, then applied to the other recording. Count bias, voxel-level
  expected calibration error (ECE) and AP are reported before and after.

Two recordings is a pilot, not an estimate; the runbook re-runs this on all 279
test clips through task_eval.py on the GPU machine.

    python analysis/posthoc_calibration.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.recording_level_stats import _clip_ap  # noqa: E402

DUMP = ROOT / "reports" / "external_baselines" / "dumps" / "pipeline.npz"
OUT = ROOT / "reports" / "posthoc_calibration.json"
TASKS = ("recon", "causal", "noncausal", "spatial")
EPS = 1e-6


def _vol(z, key, shape):
    v = np.zeros(shape, bool)
    idx = z[key].astype(np.int64)
    if idx.size:
        v[tuple(idx.T)] = True
    return v


def _roi(z, key, shape):
    return np.unpackbits(z[key])[:int(np.prod(shape))].reshape(shape).astype(bool)


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-MASS bins.

    Equal-width bins would put >99.9% of these voxels in the lowest bin, so the
    bins are quantiles of the predicted probability instead."""
    order = np.argsort(p)
    p, y = p[order], y[order].astype(float)
    edges = np.linspace(0, p.size, n_bins + 1).astype(int)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > lo:
            err += (hi - lo) * abs(p[lo:hi].mean() - y[lo:hi].mean())
    return float(err / p.size)


def fit_offset(fields, targets) -> float:
    """Logit offset b with sum sigmoid(logit(p)+b) == number of true spikes."""
    lg = [logit(np.clip(f, EPS, 1 - EPS)) for f in fields]
    n_true = float(sum(t.sum() for t in targets))
    g = lambda b: sum(expit(l + b).sum() for l in lg) - n_true  # noqa: E731
    return float(brentq(g, -30.0, 30.0))


def apply_offset(f, b):
    return expit(logit(np.clip(f, EPS, 1 - EPS)) + b)


def ratio_stability(z, shape) -> dict:
    out = {}
    for t in TASKS:
        tags = sorted({n.split("/")[1] for n in z.files if n.startswith(t + "/")})
        rows = []
        for tag in tags:
            roi = _roi(z, f"{t}/{tag}/roi_packed", shape)
            real = _vol(z, f"{t}/{tag}/real", shape)
            own = _vol(z, f"{t}/{tag}/gen_own_count", shape)
            k_true, k_own = int(real[roi].sum()), int(own[roi].sum())
            if k_true == 0:
                continue
            rows.append({"assay": int(z[f"{t}/{tag}/assay"][0]),
                         "true": k_true, "own": k_own,
                         "ratio": k_own / k_true})
        lr = np.log([r["ratio"] for r in rows if r["own"] > 0])
        out[t] = {"clips": rows, "n": len(rows),
                  "geo_mean_ratio": float(np.exp(lr.mean())),
                  "sd_log_ratio": float(lr.std(ddof=1)),
                  "min_ratio": float(np.min(np.exp(lr))),
                  "max_ratio": float(np.max(np.exp(lr)))}
    allr = np.log([r["ratio"] for t in TASKS for r in out[t]["clips"]
                   if r["own"] > 0])
    out["pooled"] = {"n": int(allr.size),
                     "geo_mean_ratio": float(np.exp(allr.mean())),
                     "sd_log_ratio": float(allr.std(ddof=1)),
                     "note": "sd_log_ratio = 0.1 means a single scale factor "
                             "leaves about +/-10% per-clip spread"}
    return out


def heldout_offset(z, shape, seed: int = 20260822) -> dict:
    data = {}
    for t in TASKS:
        for tag in sorted({n.split("/")[1] for n in z.files
                           if n.startswith(t + "/") and n.endswith("/field")}):
            a = int(z[f"{t}/{tag}/assay"][0])
            roi = _roi(z, f"{t}/{tag}/roi_packed", shape)
            y = _vol(z, f"{t}/{tag}/real", shape)[roi]
            if y.sum() == 0:
                continue
            p = z[f"{t}/{tag}/field"].astype(np.float64)[roi]
            data.setdefault(a, []).append((t, p, y))
    assays = sorted(data)
    folds = []
    for test in assays:
        train = [d for a in assays if a != test for d in data[a]]
        b = fit_offset([d[1] for d in train], [d[2] for d in train])
        for t, p, y in data[test]:
            q = apply_offset(p, b)
            folds.append({
                "fit_on_assays": [a for a in assays if a != test],
                "test_assay": test, "task": t, "offset": b,
                "scale_at_median_p": float(np.median(q) / max(np.median(p), 1e-12)),
                "true": int(y.sum()),
                "expected_before": float(p.sum()), "expected_after": float(q.sum()),
                "bias_before": float(p.sum() / y.sum() - 1),
                "bias_after": float(q.sum() / y.sum() - 1),
                "ece_before": ece(p, y), "ece_after": ece(q, y),
                "ap_before": _clip_ap(p, y, seed), "ap_after": _clip_ap(q, y, seed)})
    return {"recordings": assays, "folds": folds,
            "max_abs_ap_change": float(max(abs(f["ap_after"] - f["ap_before"])
                                           for f in folds))}


def main() -> int:
    import argparse
    global DUMP, OUT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", default=str(DUMP),
                    help="task_eval --dump output; with --dump 31 --dump-field 31 "
                         "every recording contributes a field")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    DUMP, OUT = Path(a.dump).resolve(), Path(a.out).resolve()
    z = np.load(DUMP)
    shape = tuple(int(v) for v in z["_shape"])
    res = {"source": str(DUMP.relative_to(ROOT)) if DUMP.is_relative_to(ROOT) else str(DUMP),
           "field": "model_mc (mean decoded probability over 8 draws), the "
                    "field whose ROI sum is the own-count",
           "ratio_stability": ratio_stability(z, shape),
           "heldout_logit_offset": heldout_offset(z, shape),
           "scope": "pilot: 8 recordings for the ratio, 2 for the fitted offset; "
                    "re-run on all 279 clips on the GPU machine"}
    OUT.write_text(json.dumps(res, indent=2) + "\n")

    rs = res["ratio_stability"]
    print("own / true spike count, first test clip of each of 8 recordings")
    print(f"{'task':<11}{'n':>3}{'geo-mean':>10}{'sd log':>9}{'min':>7}{'max':>7}")
    for t in TASKS:
        r = rs[t]
        print(f"{t:<11}{r['n']:>3}{r['geo_mean_ratio']:>10.2f}"
              f"{r['sd_log_ratio']:>9.2f}{r['min_ratio']:>7.2f}{r['max_ratio']:>7.2f}")
    print(f"{'pooled':<11}{rs['pooled']['n']:>3}{rs['pooled']['geo_mean_ratio']:>10.2f}"
          f"{rs['pooled']['sd_log_ratio']:>9.2f}")
    print("\nheld-out logit offset (fit on one recording, test on the other)")
    print(f"{'test':>5} {'task':<10}{'offset':>8}{'bias before':>13}{'bias after':>12}"
          f"{'ECE before':>12}{'ECE after':>11}{'AP before':>11}{'AP after':>10}")
    for f in res["heldout_logit_offset"]["folds"]:
        print(f"{f['test_assay']:>5} {f['task']:<10}{f['offset']:>8.2f}"
              f"{f['bias_before']:>+13.1%}{f['bias_after']:>+12.1%}"
              f"{f['ece_before']:>12.2e}{f['ece_after']:>11.2e}"
              f"{f['ap_before']:>11.4f}{f['ap_after']:>10.4f}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
