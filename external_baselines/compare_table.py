#!/usr/bin/env python3
"""The comparative table: pipeline variants against external baselines.

Every row must come from the same protocol, so the first thing this does is
refuse to build a table out of sets that do not describe the same held-out
clips in the same order. Everything after that assumes alignment.

Two claims are reported, and they are different claims:

  POOLED (`vs_real` in summary.json)
      Generated population statistics against real population statistics.
      A model that ignores its context and emits the dataset average scores
      well here. The Dichotomized Gaussian nearly saturates it. Reported for
      completeness and because reviewers expect it -- not as the headline.

  PER-CLIP (`vs_truth` in manifest.json)
      Each sample against the specific clip whose context it was given. This is
      the only paired measurement in the output, and the only one that can see
      conditioning.

and the ladder itself:

  CONDITIONING GAIN
      per-clip error at `random` minus per-clip error at `global_full_local`,
      paired by clip. Positive means context helped. A model that cannot use
      context is flat here by construction, which is the point of including
      baselines that structurally cannot.

Significance is by Wilcoxon signed-rank across clips, Benjamini-Hochberg
corrected within each family. Medians, not means: several `rel_*` terms divide
by a real statistic that can be near zero, and the mean is then dominated by a
handful of clips (observed means above 1e5 against a median of 0.84).

    python external_baselines/compare_table.py
    python external_baselines/compare_table.py --metrics stat_error,ks_isi --markdown table.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, "/media/derik/Seagate Desktop Drive/organoid_data")

REGIMES = ("random", "global_only", "global_partial_local", "global_full_local")

DEFAULT_METRICS = ("stat_error", "ks_avalanche", "ks_isi", "rel_rate",
                   "rel_spatial_coact", "rel_persist1", "rel_persist4",
                   "rel_isi_mean", "rel_avalanche_mean", "rel_burst_rate")

DEFAULT_SETS = {
    # pipeline
    "4C+soft (ship)": "reports/generation_regimes_4c_soft",
    "4A+soft":        "reports/generation_regimes_4a_soft",
    "4B":             "reports/generation_regimes_4b",
    "4C+hard":        "reports/generation_regimes_4c_hard",
    # external
    "DG (Macke'09)":  "reports/external_baselines/dg",
    "GLM (Pillow'08)": "reports/external_baselines/glm",
    "MaskGIT-flat":   "reports/external_baselines/maskgit_flat",
}


# ----------------------------------------------------------------------

def load_set(path: Path) -> Optional[Dict]:
    mp = path / "manifest.json"
    if not mp.exists():
        return None
    rows = json.loads(mp.read_text())
    by = {}
    for r in rows:
        by.setdefault(r["regime"], []).append(r)
    summ = {}
    sp = path / "summary.json"
    if sp.exists():
        summ = json.loads(sp.read_text())
    return {"rows": by, "summary": summ, "path": str(path)}


def clip_keys(rows: List[Dict]) -> List[tuple]:
    return [(r["assay"], round(float(r.get("real_spikes", float("nan"))), 3))
            for r in rows]


def check_alignment(sets: Dict[str, Dict]) -> List[str]:
    """Every set must describe the same clips in the same order, per regime."""
    problems, ref_name, ref = [], None, None
    for name, s in sets.items():
        if ref is None:
            ref_name, ref = name, s
            continue
        for reg in REGIMES:
            a = clip_keys(ref["rows"].get(reg, []))
            b = clip_keys(s["rows"].get(reg, []))
            if not a or not b:
                continue
            if len(a) != len(b):
                problems.append(f"{name}/{reg}: n={len(b)} vs {ref_name} n={len(a)}")
            elif a != b:
                i = next(k for k in range(len(a)) if a[k] != b[k])
                problems.append(f"{name}/{reg}: clip mismatch at {i} "
                                f"({b[i]} vs {a[i]} in {ref_name})")
    return problems


# ----------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------

def wilcoxon(d: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value, normal approximation with ties."""
    d = d[np.isfinite(d) & (d != 0)]
    n = d.size
    if n < 6:
        return float("nan")
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    # average ranks for ties
    a = np.abs(d)
    order = np.argsort(a)
    sa = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    w = float(r[d > 0].sum())
    mu = n * (n + 1) / 4.0
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sd == 0:
        return float("nan")
    z = (w - mu) / sd
    from math import erfc, sqrt
    return float(erfc(abs(z) / sqrt(2.0)))


def bh_fdr(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg step-up q-values, returned in input order."""
    idx = [i for i, p in enumerate(pvals) if np.isfinite(p)]
    n = len(idx)
    q = [float("nan")] * len(pvals)
    if not n:
        return q
    order = sorted(idx, key=lambda i: pvals[i])
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = n - rank + 1
        prev = min(prev, pvals[i] * n / k)
        q[i] = prev
    return q


def clean_stat_error(rows: List[Dict], keep: List[str]) -> np.ndarray:
    """Per-clip mean relative error over the NON-degenerate rel_ terms only.

    `compare_statistics` builds `stat_error` as the mean over every rel_ term it
    computed, degenerate ones included. At this sparsity `spatial_coact` is
    identically zero on 99% of clips, so its rel_ term is |gen|/1e-8 whenever the
    model emits any adjacent pair at all -- which pushed one baseline's median
    per-clip stat_error to 5.4e4 while its pooled stat_error was 0.81.

    That is a property of the metric, not of the model, and it silently favours
    whichever models happen to emit zeros in the same places the real clips do.
    Recomputing over the surviving terms makes the headline comparable across
    methods; the raw value is still reported alongside it.
    """
    out = []
    for r in rows:
        vt = r.get("vs_truth", {})
        vals = [vt[k] for k in keep if k in vt and np.isfinite(vt[k])]
        out.append(float(np.mean(vals)) if vals else np.nan)
    return np.array(out, dtype=float)


def series(rows: List[Dict], metric: str) -> np.ndarray:
    return np.array([r.get("vs_truth", {}).get(metric, np.nan) for r in rows],
                    dtype=float)


def degeneracy(sets: Dict[str, Dict], metric: str, regime: str) -> float:
    """Fraction of clips on which this per-clip metric carries no information.

    `compare_statistics` forms rel_X = |gen - real| / max(|real|, 1e-8). When a
    clip's real statistic is 0 the ratio is meaningless -- it is either exactly
    1.0 (generated also 0) or ~1e8 (generated nonzero), and the median over
    clips then reports the mixing proportion rather than any model property.

    Measured here rather than assumed: `spatial_coact` is identically zero on
    98% of clips at this sparsity, and `persist1` on a substantial minority, so
    both are excluded from the headline instead of quietly reporting noise.
    """
    # Only the rel_ family divides by a real statistic. stat_error and the KS
    # terms are comparison outputs in their own right and have no `stats` entry
    # to look up -- checking them here would flag them as 100% degenerate,
    # which is a bug in the check, not a property of the metric.
    if not metric.startswith("rel_"):
        return 0.0
    key = metric[4:]
    frac = []
    for s in sets.values():
        rows = s["rows"].get(regime, [])
        v = np.array([r.get("stats", {}).get(key, np.nan) for r in rows], float)
        if v.size:
            frac.append(float(np.mean(~np.isfinite(v) | (v == 0))))
    return max(frac) if frac else 0.0


# ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default=None,
                    help="name=path,name=path ... (default: the standard set list)")
    ap.add_argument("--reference", default="4C+soft (ship)",
                    help="set every other row is paired against")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--regime", default="global_full_local",
                    help="regime for the head-to-head table")
    ap.add_argument("--markdown", default=None, help="also write a markdown table here")
    ap.add_argument("--json", default="reports/external_baselines/comparison.json")
    ap.add_argument("--allow-unaligned", action="store_true")
    ap.add_argument("--degeneracy-max", type=float, default=0.25,
                    help="drop a per-clip metric whose real statistic is zero "
                         "on more than this fraction of clips")
    args = ap.parse_args()

    spec = DEFAULT_SETS
    if args.sets:
        spec = dict(kv.split("=", 1) for kv in args.sets.split(","))

    sets, missing = {}, []
    for name, path in spec.items():
        s = load_set(Path(path))
        (sets.__setitem__(name, s) if s else missing.append(f"{name} ({path})"))
    if missing:
        print("not yet available: " + ", ".join(missing))
    if len(sets) < 2:
        print("need at least two sample sets to compare")
        return 1

    print(f"\nsets: {', '.join(sets)}")
    problems = check_alignment(sets)
    if problems:
        print(f"\n!! {len(problems)} ALIGNMENT PROBLEM(S):")
        for p in problems[:12]:
            print("   " + p)
        if not args.allow_unaligned:
            print("\nRefusing to build a table from unaligned sets. Re-run the "
                  "offending baseline under the reference protocol, or pass "
                  "--allow-unaligned to inspect anyway.")
            return 1
    else:
        n = len(sets[list(sets)[0]]["rows"].get(args.regime, []))
        print(f"alignment OK -- {n} samples per regime, identical clips in every set")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    ref = args.reference if args.reference in sets else list(sets)[0]

    deg = {m: degeneracy(sets, m, args.regime) for m in metrics}
    dropped = [m for m in metrics if deg[m] > args.degeneracy_max]
    if dropped:
        print("\nexcluded from the per-clip head-to-head (real statistic is "
              f"zero on >{args.degeneracy_max:.0%} of clips, so rel_ is "
              "uninformative):")
        for m in dropped:
            print(f"   {m:22s} degenerate on {deg[m]:.1%} of clips")
    metrics = [m for m in metrics if m not in dropped]
    keep_rel = [m for m in metrics if m.startswith("rel_")]

    def _series(rows, m):
        return clean_stat_error(rows, keep_rel) if m == "stat_error_clean" \
            else series(rows, m)

    # Headline replaces the contaminated aggregate; the raw one stays visible.
    if "stat_error" in metrics:
        metrics = ["stat_error_clean"] + metrics
    out: Dict = {"reference": ref, "regime": args.regime, "sets": list(sets),
                 "aligned": not problems, "degeneracy": deg,
                 "excluded_degenerate": dropped}

    # ---------------- 1. conditioning gain (the headline) -------------
    print(f"\n{'='*78}\nCONDITIONING GAIN  (per-clip error: random -> global_full_local)")
    print(f"{'':22s} {'random':>9s} {'full':>9s} {'gain':>9s} {'better':>10s} {'q':>9s}")
    gain_rows, praw = [], []
    for name, s in sets.items():
        a = _series(s["rows"].get("random", []), "stat_error_clean")
        b = _series(s["rows"].get("global_full_local", []), "stat_error_clean")
        n = min(a.size, b.size)
        if n == 0:
            continue
        a, b = a[:n], b[:n]
        ok = np.isfinite(a) & np.isfinite(b)
        d = a[ok] - b[ok]                       # positive => context helped
        p = wilcoxon(d)
        gain_rows.append((name, float(np.median(a[ok])), float(np.median(b[ok])),
                          float(np.median(d)), int((d > 0).sum()), int(d.size)))
        praw.append(p)
    qs = bh_fdr(praw)
    out["conditioning_gain"] = []
    for (name, ma, mb, md, w, tot), p, q in zip(gain_rows, praw, qs):
        star = " *" if np.isfinite(q) and q < 0.05 else ""
        print(f"{name:22s} {ma:9.4f} {mb:9.4f} {md:+9.4f} {w:5d}/{tot:<4d} {q:9.2e}{star}")
        out["conditioning_gain"].append(
            {"set": name, "median_random": ma, "median_full": mb,
             "median_gain": md, "n_better": w, "n": tot, "p": p, "q": q})

    # ---------------- 2. head-to-head, per clip -----------------------
    print(f"\n{'='*78}\nPER-CLIP vs_truth  regime={args.regime}   (paired against {ref})")
    hdr = f"{'metric':20s}" + "".join(f"{n[:13]:>14s}" for n in sets)
    print(hdr)
    out["per_clip"] = {}
    allp, keys = [], []
    for m in metrics:
        cells, row = [], {}
        base = _series(sets[ref]["rows"].get(args.regime, []), m)
        for name, s in sets.items():
            v = _series(s["rows"].get(args.regime, []), m)
            med = float(np.nanmedian(v)) if v.size else float("nan")
            row[name] = {"median": med}
            cells.append(f"{med:14.4f}")
            if name != ref and v.size and base.size:
                n = min(v.size, base.size)
                d = base[:n] - v[:n]            # negative => reference better
                d = d[np.isfinite(d)]
                p = wilcoxon(d)
                row[name]["median_diff_vs_ref"] = float(np.median(d))
                allp.append(p); keys.append((m, name))
        print(f"{m:20s}" + "".join(cells))
        out["per_clip"][m] = row
    qs = bh_fdr(allp)
    for (m, name), p, q in zip(keys, allp, qs):
        out["per_clip"][m][name].update({"p": p, "q": q})

    print(f"\nsignificant differences vs {ref} (BH-FDR q<0.05, {len(allp)} tests):")
    any_sig = False
    for (m, name), q in zip(keys, qs):
        if np.isfinite(q) and q < 0.05:
            d = out["per_clip"][m][name]["median_diff_vs_ref"]
            who = ref if d < 0 else name
            print(f"  {m:20s} vs {name:18s} q={q:.2e}  better: {who}")
            any_sig = True
    if not any_sig:
        print("  none")

    # ---------------- 3. pooled, for completeness ---------------------
    print(f"\n{'='*78}\nPOOLED vs_real  regime={args.regime}   "
          f"(cannot see conditioning -- context)")
    print(f"{'set':22s} {'stat_error':>12s} {'ks_aval':>10s} {'ks_isi':>10s}")
    out["pooled"] = {}
    for name, s in sets.items():
        e = s["summary"].get(args.regime, {}).get("vs_real", {})
        out["pooled"][name] = e
        print(f"{name:22s} {e.get('stat_error', float('nan')):12.4f} "
              f"{e.get('ks_avalanche', float('nan')):10.4f} "
              f"{e.get('ks_isi', float('nan')):10.4f}")
    print(f"{'-- ceiling (real/real)':22s} {0.1060:12.4f} {0.0930:10.4f} {0.0379:10.4f}")
    print(f"{'-- floor (mean-field)':22s} {0.9659:12.4f} {0.7229:10.4f} {0.5254:10.4f}")

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {args.json}")

    if args.markdown:
        _write_markdown(Path(args.markdown), out, sets, metrics, args.regime, ref)
        print(f"wrote {args.markdown}")
    return 0


def _write_markdown(path: Path, out, sets, metrics, regime, ref) -> None:
    L = ["# External baseline comparison", "",
         f"Regime `{regime}`, paired against **{ref}**. "
         f"Medians across clips; q-values are BH-FDR within family.", "",
         "## Conditioning gain (per-clip error, random -> full context)", "",
         "| set | random | full | gain | clips better | q |", "|---|---|---|---|---|---|"]
    for r in out["conditioning_gain"]:
        L.append(f"| {r['set']} | {r['median_random']:.4f} | {r['median_full']:.4f} "
                 f"| {r['median_gain']:+.4f} | {r['n_better']}/{r['n']} | {r['q']:.2e} |")
    L += ["", "## Per-clip vs_truth", "",
          "| metric | " + " | ".join(sets) + " |",
          "|---" * (len(sets) + 1) + "|"]
    for m in metrics:
        row = out["per_clip"].get(m, {})
        L.append(f"| {m} | " + " | ".join(
            f"{row.get(n, {}).get('median', float('nan')):.4f}" for n in sets) + " |")
    L += ["", "## Pooled vs_real (reported for completeness; blind to conditioning)", "",
          "| set | stat_error | ks_avalanche | ks_isi |", "|---|---|---|---|"]
    for n in sets:
        e = out["pooled"].get(n, {})
        L.append(f"| {n} | {e.get('stat_error', float('nan')):.4f} | "
                 f"{e.get('ks_avalanche', float('nan')):.4f} | "
                 f"{e.get('ks_isi', float('nan')):.4f} |")
    L += ["| *ceiling (real vs real)* | 0.1060 | 0.0930 | 0.0379 |",
          "| *floor (mean-field)* | 0.9659 | 0.7229 | 0.5254 |", ""]
    path.write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
