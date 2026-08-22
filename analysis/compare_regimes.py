"""Compare the 4b and 4c four-regime sample sets, and grade each against real data.

Two questions:

  1. Does 4B-refine change the samples at all? Both sets were generated from the same
     clips, the same regimes and the same seeds, so any per-regime difference is
     attributable to 4B-refine's calibration. The Stage 4B-refine training run says it should
     be ~nothing (all epoch variation inside 0.92 seed sd, motif MRR declining
     t=-10.45), and this is the end-product check of that claim.

  2. Does conditioning do what it is supposed to? The ladder
     random -> global_only -> global_partial_local -> global_full_local adds
     information at each rung, so the distance to the real held-out clips should
     fall monotonically. That is the paper's core conditioning result and it does
     not depend on question 1 at all.
"""
import json, sys, os, argparse
from pathlib import Path

ROOT = Path("/media/derik/Seagate Desktop Drive/organoid_data/MAGVIT_project")
REGIMES = ["random", "global_only", "global_partial_local", "global_full_local"]

_ap = argparse.ArgumentParser()
_ap.add_argument("--sets", default=None,
                 help="comma-separated name=path list. Default: every "
                      "reports/generation_regimes_* directory, in name order.")
_ap.add_argument("--baseline", default=None,
                 help="set name every other set is differenced against. "
                      "Default: the first set.")
_args = _ap.parse_args()

if _args.sets:
    SETS = {}
    for _item in _args.sets.split(","):
        _n, _, _p = _item.partition("=")
        SETS[_n.strip()] = ROOT / _p.strip() if _p else ROOT / f"reports/generation_regimes_{_n.strip()}"
else:
    SETS = {d.name.replace("generation_regimes_", ""): d
            for d in sorted((ROOT / "reports").glob("generation_regimes_*"))
            if d.is_dir()}
PHASES = list(SETS)
BASE = _args.baseline or (PHASES[0] if PHASES else None)
print(f"sets: {', '.join(PHASES)}   baseline: {BASE}")

def load(p):
    f = p / "summary.json"
    return json.loads(f.read_text()) if f.exists() else None

sums = {k: load(v) for k, v in SETS.items()}
for k, v in sums.items():
    print(f"{k}: {'loaded' if v else 'MISSING ' + str(SETS[k])}")
if not any(sums.values()):
    sys.exit("no summaries found")

def regime_rows(s):
    """summary.json layout varies; find the per-regime dict wherever it lives."""
    if s is None:
        return {}
    for key in ("regimes", "per_regime", "by_regime"):
        if isinstance(s.get(key), dict):
            return s[key]
    return {k: v for k, v in s.items() if k in REGIMES and isinstance(v, dict)}

rows = {k: regime_rows(v) for k, v in sums.items()}
def flat(reg):
    """summary.json nests the numbers under stats/ and vs_real/."""
    out = {}
    for blk in ("stats", "vs_real"):
        for k, v in (reg.get(blk) or {}).items():
            if isinstance(v, (int, float)):
                out[k] = v
    return out

for r in rows.values():
    for k in list(r):
        r[k] = {**flat(r[k]), "n_samples": r[k].get("n_samples", 0)}

allkeys = set()
for r in rows.values():
    for reg in r.values():
        allkeys |= {k for k, v in reg.items() if isinstance(v, (int, float))}
METRICS = [m for m in ("stat_error", "statErr", "ks_avalanche", "KSaval",
                       "ks_isi", "KSisi", "rate", "burst_rate", "isi_mean",
                       "avalanche_mean", "persist1", "spatial_coactivity")
           if m in allkeys] or sorted(allkeys)[:8]

for phase in PHASES:
    if not rows.get(phase):
        continue
    print(f"\n=== {phase}: conditioning ladder ===")
    print(f"{'regime':<24}" + "".join(f"{m[:12]:>14}" for m in METRICS))
    for reg in REGIMES:
        d = rows[phase].get(reg)
        if not d:
            continue
        print(f"{reg:<24}" + "".join(
            f"{d[m]:>14.5f}" if isinstance(d.get(m), (int, float)) else f"{'-':>14}"
            for m in METRICS))

for _other in [p_ for p_ in PHASES if p_ != BASE]:
    if not (rows.get(BASE) and rows.get(_other)):
        continue
    print(f"\n=== {_other} - {BASE} (same clips, regimes, seeds) ===")
    print(f"{'regime':<24}" + "".join(f"{m[:12]:>14}" for m in METRICS))
    for reg in REGIMES:
        a, b = rows[BASE].get(reg), rows[_other].get(reg)
        if not a or not b:
            continue
        cells = []
        for m in METRICS:
            if isinstance(a.get(m), (int, float)) and isinstance(b.get(m), (int, float)):
                cells.append(f"{b[m]-a[m]:>+14.5f}")
            else:
                cells.append(f"{'-':>14}")
        print(f"{reg:<24}" + "".join(cells))

for phase, p in SETS.items():
    mf = p / "manifest.json"
    if mf.exists():
        m = json.loads(mf.read_text())
        rowsm = m if isinstance(m, list) else m.get("samples", m.get("rows", []))
        print(f"\n{phase}: {len(rowsm)} samples in manifest")


# ----------------------------------------------------------------------------
# The paired conditioning test.
#
# Within a regime the writer numbers samples in generation order:
#   for batch: for rep: for i in range(B)
# so sample n decodes to batch = n // (reps*B), rep, and clip index i. The clip
# identity (batch, i) is shared across regimes and reps, which is what makes
# this paired: the same held-out clip is generated four ways.
#
# vs_truth is the distance from a sample to the clip it was aligned with.
# "random" is the null -- context drawn unconditionally, so its vs_truth is what
# you get with no clip information. Every conditioned rung should beat it.
# ----------------------------------------------------------------------------
import statistics

def paired_truth(phase):
    p = SETS[phase] / "manifest.json"
    if not p.exists():
        return
    rows = json.loads(p.read_text())
    rows = rows if isinstance(rows, list) else rows.get("samples", [])
    if not any("vs_truth" in r for r in rows):
        print(f"\n{phase}: no vs_truth in manifest (pre-patch run)")
        return
    # Relative errors whose denominator is a real-clip statistic that is ~0 blow
    # up per clip: rel_persist2..7 and rel_spatial_coact land in the 1e4-1e6
    # range, and stat_error is their mean so it inherits the blow-up. They are
    # fine POOLED (summary.json) because pooling keeps the denominator away from
    # zero, but per clip they are noise with a huge magnitude and would dominate
    # any average. Excluded here rather than silently averaged.
    DEGENERATE = {"stat_error", "stat_error_within", "rel_spatial_coact"} | {
        f"rel_persist{i}" for i in range(1, 8)}
    keys = sorted({k for r in rows for k in r.get("vs_truth", {})
                   if isinstance(r["vs_truth"][k], (int, float))} - DEGENERATE)
    dropped = sorted({k for r in rows for k in r.get("vs_truth", {})} & DEGENERATE)
    if dropped:
        print(f"\n[{phase}] excluded near-zero-denominator fields: {', '.join(dropped)}")
    if not keys:
        return

    # Pair on (sample index) -- the writer numbers each regime independently in
    # the SAME generation order, so index n is the same clip and the same rep in
    # every regime. Averaging over reps first gives one value per clip.
    by = {}   # regime -> sample_index -> {metric: value}
    reps = max(int(r.get("rep", 0)) for r in rows) + 1
    for r in rows:
        vt = r.get("vs_truth")
        if not vt:
            continue
        by.setdefault(r["regime"], {})[int(r["sample"])] = vt

    print(f"\n=== {phase}: vs_truth, paired per clip "
          f"({reps} reps/clip, lower = closer to the aligned clip) ===")
    print(f"{'regime':<24}" + "".join(f"{k[:13]:>15}" for k in keys))
    for reg in REGIMES:
        d = by.get(reg)
        if not d:
            continue
        print(f"{reg:<24}" + "".join(
            f"{statistics.fmean([v[k] for v in d.values() if k in v]):>15.5f}"
            for k in keys))

    if "random" not in by:
        return
    print(f"\npaired difference vs the random null (t over shared samples)")
    print(f"{'regime':<24}" + "".join(f"{k[:13]:>15}" for k in keys))
    for reg in REGIMES:
        if reg == "random" or reg not in by:
            continue
        cells = []
        for k in keys:
            shared = [n for n in by[reg] if n in by["random"]
                      and k in by[reg][n] and k in by["random"][n]]
            diffs = [by[reg][n][k] - by["random"][n][k] for n in shared]
            if len(diffs) < 3:
                cells.append(f"{'-':>15}"); continue
            m = statistics.fmean(diffs)
            sd = statistics.stdev(diffs)
            t = m / (sd / len(diffs) ** 0.5) if sd > 1e-12 else float("inf")
            cells.append(f"{m:>+10.5f}/{t:>+4.1f}")
        print(f"{reg:<24}" + "".join(cells))

for ph in PHASES:
    paired_truth(ph)


# ----------------------------------------------------------------------------
# Robust version of the same test.
#
# Dropping rel_persist*/rel_spatial_coact/stat_error* because their MEAN blows up
# throws away usable data: the metric is well behaved for most clips and only
# explodes where the real-clip denominator is ~0. The median is immune to that,
# and the sign test asks the question we actually care about -- for how many
# individual clips does conditioning move the sample CLOSER to its own clip? --
# without assuming anything about the tails.
# ----------------------------------------------------------------------------
from math import comb

# Every sign test computed anywhere in this script registers itself here so the
# final section can apply a multiple-comparison correction. Without it the
# tables invite the reader to do the correction themselves and discover that
# some of the starred rows do not survive: each conditioning ladder is ~19
# metrics x 3 regimes = 57 simultaneous tests against the same null.
_PVALS = []   # (family, regime, metric, p, better, worse, median_diff)


def register(family, regime, metric, p, better, worse, med):
    _PVALS.append((family, regime, metric, p, better, worse, med))


def bh_fdr(pvals):
    """Benjamini-Hochberg step-up q-values. Returns them in input order."""
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    q = [0.0] * n
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = n - rank + 1                       # 1-based rank of this p-value
        prev = min(prev, pvals[i] * n / k)
        q[i] = prev
    return q


def sign_p(better, worse):
    """Two-sided exact binomial test against p=0.5."""
    n = better + worse
    if n == 0:
        return float("nan")
    k = min(better, worse)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)

def robust_truth(phase):
    p = SETS[phase] / "manifest.json"
    if not p.exists():
        return
    rows = json.loads(p.read_text())
    rows = rows if isinstance(rows, list) else rows.get("samples", [])
    by = {}
    for r in rows:
        if r.get("vs_truth"):
            by.setdefault(r["regime"], {})[int(r["sample"])] = r["vs_truth"]
    if "random" not in by:
        return
    keys = sorted({k for d in by.values() for v in d.values() for k in v})

    print(f"\n=== {phase}: ROBUST paired test vs the random null ===")
    print("     median of per-clip differences, and the sign test "
          "(how many clips improve)")
    for reg in REGIMES:
        if reg == "random" or reg not in by:
            continue
        print(f"\n  {reg}")
        print(f"    {'metric':<26}{'med diff':>12}{'better':>8}{'worse':>7}{'p':>10}")
        for k in keys:
            shared = [n for n in by[reg] if n in by["random"]
                      and k in by[reg][n] and k in by["random"][n]]
            d = [by[reg][n][k] - by["random"][n][k] for n in shared]
            d = [x for x in d if x == x]
            if len(d) < 5:
                continue
            better = sum(1 for x in d if x < 0)
            worse = sum(1 for x in d if x > 0)
            pv = sign_p(better, worse)
            register(f"conditioning[{phase}]", reg, k, pv, better, worse,
                     statistics.median(d))
            star = "***" if pv < 1e-3 else "**" if pv < 1e-2 else "*" if pv < 0.05 else ""
            print(f"    {k:<26}{statistics.median(d):>+12.4f}{better:>8}{worse:>7}"
                  f"{pv:>10.2e} {star}")

for ph in PHASES:
    robust_truth(ph)


# ----------------------------------------------------------------------------
# Cross-set paired test: same clip, same regime, same seed, different model.
#
# The sections above ask "does conditioning help?" within one set. This asks
# "does the model change help?" -- which is the question the soft activity
# field and Stage 4C exist to answer, and it cannot be read off the pooled
# summary (see the conditioning-ladder result: pooled stat_error inverted the
# sign of an effect the paired test found clearly).
#
# Sample index n is the same clip and rep in every set, because every run used
# the same split, the same seed and the same generation order.
# ----------------------------------------------------------------------------

def _truth_by_regime(phase):
    p = SETS[phase] / "manifest.json"
    if not p.exists():
        return {}
    rws = json.loads(p.read_text())
    rws = rws if isinstance(rws, list) else rws.get("samples", [])
    by = {}
    for r in rws:
        if r.get("vs_truth"):
            by.setdefault(r["regime"], {})[int(r["sample"])] = r["vs_truth"]
    return by

_TB = {ph: _truth_by_regime(ph) for ph in PHASES}


def _identity(phase):
    """(regime, sample) -> (assay, real_spikes). Cross-set pairing is only valid
    if index n means the same held-out clip in every set; these two fields come
    from the data, not the model, so they must agree exactly."""
    p = SETS[phase] / "manifest.json"
    if not p.exists():
        return {}
    rws = json.loads(p.read_text())
    rws = rws if isinstance(rws, list) else rws.get("samples", [])
    return {(r["regime"], int(r["sample"])): (r.get("assay"), r.get("real_spikes"))
            for r in rws}

_ID = {ph: _identity(ph) for ph in PHASES}
for _other in [p_ for p_ in PHASES if p_ != BASE]:
    _a, _b = _ID.get(BASE, {}), _ID.get(_other, {})
    _shared = set(_a) & set(_b)
    _bad = [k for k in _shared if _a[k] != _b[k]]
    if not _shared:
        print(f"\n[pairing] {_other} vs {BASE}: NO shared sample keys")
    elif _bad:
        print(f"\n[pairing] *** {_other} vs {BASE}: {len(_bad)}/{len(_shared)} "
              f"sample indices point at DIFFERENT clips -- the paired test below "
              f"is INVALID. e.g. {_bad[:3]}")
    else:
        print(f"\n[pairing] {_other} vs {BASE}: {len(_shared)} sample indices "
              f"verified to be the same clip (assay + real_spikes match)")

for _other in [p_ for p_ in PHASES if p_ != BASE]:
    A, B_ = _TB.get(BASE, {}), _TB.get(_other, {})
    if not A or not B_:
        continue
    print(f"\n=== vs_truth, {_other} vs {BASE}, PAIRED per clip "
          f"(negative = {_other} is closer to the real clip) ===")
    for reg in REGIMES:
        if reg not in A or reg not in B_:
            continue
        keys = sorted({k for v in A[reg].values() for k in v}
                      & {k for v in B_[reg].values() for k in v})
        print(f"\n  {reg}")
        print(f"    {'metric':<26}{'med diff':>12}{'better':>8}{'worse':>7}{'p':>10}")
        for k in keys:
            shared = [n for n in B_[reg] if n in A[reg]
                      and k in B_[reg][n] and k in A[reg][n]]
            d = [B_[reg][n][k] - A[reg][n][k] for n in shared]
            d = [x for x in d if x == x]
            if len(d) < 5:
                continue
            better = sum(1 for x in d if x < 0)
            worse = sum(1 for x in d if x > 0)
            pv = sign_p(better, worse)
            register(f"crossset[{_other} vs {BASE}]", reg, k, pv, better, worse,
                     statistics.median(d))
            star = "***" if pv < 1e-3 else "**" if pv < 1e-2 else "*" if pv < 0.05 else ""
            print(f"    {k:<26}{statistics.median(d):>+12.4f}{better:>8}{worse:>7}"
                  f"{pv:>10.2e} {star}")


# ----------------------------------------------------------------------------
# Multiple-comparison correction.
#
# Benjamini-Hochberg within each family, where a family is one test type on one
# set: a conditioning ladder is ~19 metrics x 3 regimes tested against the same
# random null, and a cross-set comparison is the same metrics x regimes against
# the same baseline. Correcting across families would be over-conservative --
# they answer different questions and are reported separately.
#
# Report q, not p. A row that survives q<0.05 is one you can defend; a row that
# only had p<0.05 is one a reviewer will delete for you.
# ----------------------------------------------------------------------------
if _PVALS:
    fams = {}
    for row in _PVALS:
        fams.setdefault(row[0], []).append(row)

    print("\n\n" + "=" * 78)
    print("MULTIPLE-COMPARISON CORRECTION (Benjamini-Hochberg, within family)")
    print("=" * 78)
    for fam, rows in fams.items():
        qs = bh_fdr([r[3] for r in rows])
        keep = [(r, q) for r, q in zip(rows, qs) if q < 0.05]
        keep.sort(key=lambda rq: rq[1])
        lost = [(r, q) for r, q in zip(rows, qs) if r[3] < 0.05 <= q]
        print(f"\n{fam}   {len(rows)} tests, {len(keep)} survive q<0.05")
        if keep:
            print(f"    {'regime':<24}{'metric':<26}{'med':>10}{'p':>10}{'q':>10}")
            for (f_, reg, met, pv, b, w, med), q in keep:
                print(f"    {reg:<24}{met:<26}{med:>+10.4f}{pv:>10.2e}{q:>10.2e}")
        if lost:
            print(f"  -- had p<0.05 but did NOT survive correction "
                  f"({len(lost)}): "
                  + ", ".join(f"{reg}/{met}" for (f_, reg, met, *_), q in lost))
