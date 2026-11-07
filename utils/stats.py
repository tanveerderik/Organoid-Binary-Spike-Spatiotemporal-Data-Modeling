"""Cluster-aware statistics for per-clip results.

Every evaluation in this repository scores clips, and clips are not independent:
nine test clips come from each recording, they are sampled from that recording's
burst windows with replacement, and three recordings are byte-identical copies of
three others. A clip-level test treats 279 clips as 279 independent observations
when the number of independent units is closer to 28 recordings.

This module puts the recording at the centre:

* `merge_duplicates` maps the duplicate recordings onto one cluster each.
* `paired_recording_test` averages the paired difference within each recording
  and runs a Wilcoxon signed-rank test on the recording means (n = recordings).
* `hierarchical_bootstrap` resamples recordings (stratified by tissue), then
  clips within each sampled recording, and returns a percentile CI for the mean
  of recording-mean differences, so every recording carries equal weight.
* `bh` and `holm` correct a family of p-values.

Only numpy and scipy are required, so this runs without torch or data.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy import stats as _st

# Byte-identical recordings (same acquisition converted twice), keyed
# duplicate -> canonical. Source: reports/data_provenance.json
# `biological_n_note`; pairs named in tools/tests/test_paper_cross_references.py.
DUPLICATE_OF: Dict[int, int] = {10: 4, 11: 3, 13: 6}


def merge_duplicates(labels: Iterable[int],
                     duplicate_of: Mapping[int, int] = DUPLICATE_OF) -> np.ndarray:
    """Recording label per clip with duplicate recordings folded onto one id."""
    return np.asarray([int(duplicate_of.get(int(a), int(a))) for a in labels])


def recording_means(values: Sequence[float], labels: Sequence[int]):
    """(ids, means): NaN-aware mean of `values` within each label, ids sorted."""
    v = np.asarray(values, float)
    lab = np.asarray(labels)
    ids = np.unique(lab)
    means = np.array([np.nanmean(v[lab == i]) if np.any(~np.isnan(v[lab == i]))
                      else np.nan for i in ids])
    return ids, means


def _wilcoxon(d: np.ndarray) -> float:
    d = d[~np.isnan(d)]
    if d.size == 0 or np.allclose(d, 0):
        return 1.0
    return float(_st.wilcoxon(d).pvalue)


def paired_clip_test(a: Sequence[float], b: Sequence[float]) -> dict:
    """The original clip-level test, kept for side-by-side comparison."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    ok = ~np.isnan(d)
    return {"n": int(ok.sum()), "mean_delta": float(np.nanmean(d)),
            "median_delta": float(np.nanmedian(d)),
            "wins_frac": float((d[ok] > 0).mean()) if ok.any() else float("nan"),
            "p": _wilcoxon(d)}


def paired_recording_test(a: Sequence[float], b: Sequence[float],
                          labels: Sequence[int]) -> dict:
    """Wilcoxon signed-rank on per-recording mean paired differences."""
    d = np.asarray(a, float) - np.asarray(b, float)
    ids, md = recording_means(d, labels)
    ok = ~np.isnan(md)
    return {"n_recordings": int(ok.sum()),
            "mean_delta": float(np.nanmean(md)),
            "median_delta": float(np.nanmedian(md)),
            "recordings_won_frac": float((md[ok] > 0).mean()) if ok.any() else float("nan"),
            "p": _wilcoxon(md)}


def hierarchical_bootstrap(diff: Sequence[float], labels: Sequence[int],
                           strata: Optional[Mapping[int, str]] = None,
                           n_boot: int = 10000, seed: int = 0,
                           ci: float = 0.95) -> dict:
    """Two-level bootstrap of the mean of recording-mean differences.

    Level 1 resamples recordings with replacement (within each stratum, so the
    tissue mix is held fixed); level 2 resamples clips with replacement inside
    each drawn recording. The statistic weights recordings equally.
    """
    d = np.asarray(diff, float)
    lab = np.asarray(labels)
    keep = ~np.isnan(d)
    d, lab = d[keep], lab[keep]
    ids = np.unique(lab)
    clips = {i: d[lab == i] for i in ids}
    groups: Dict[str, list] = {}
    for i in ids:
        groups.setdefault((strata or {}).get(int(i), "all"), []).append(i)
    rng = np.random.default_rng(seed)
    point = float(np.mean([clips[i].mean() for i in ids]))
    boots = np.empty(n_boot)
    for k in range(n_boot):
        means = []
        for members in groups.values():
            for i in rng.choice(members, size=len(members), replace=True):
                c = clips[i]
                means.append(c[rng.integers(0, c.size, c.size)].mean())
        boots[k] = np.mean(means)
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"estimate": point, "ci_low": float(lo), "ci_high": float(hi),
            "ci": ci, "n_boot": n_boot,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def bh(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg q-values, input order preserved."""
    p = np.asarray(pvalues, float)
    n = p.size
    order = np.argsort(p)
    q = np.empty(n)
    run = 1.0
    for rank, i in enumerate(order[::-1]):
        run = min(run, p[i] * n / (n - rank))
        q[i] = run
    return q


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, input order preserved."""
    p = np.asarray(pvalues, float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, p[i] * (n - rank)))
        adj[i] = run
    return adj
