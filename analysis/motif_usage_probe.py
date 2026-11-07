#!/usr/bin/env python3
"""Motif reuse without duplicate recordings, and whether motif usage carries
information about the preparation beyond firing rate.

Two questions about the shared vocabulary, answered from the per-recording code
counts that analysis/motif_reuse.py already stored (reports/analysis_motif_reuse.npz,
31 recordings x 961 flat codes, blank token excluded). No model, data or GPU.

1. Reuse statistics with the three byte-identical duplicate recordings removed
   (10, 11, 13 are copies of 4, 3, 6). The published numbers counted each copy
   as a separate recording, which inflates within-organoid overlap: a copy
   shares its whole vocabulary distribution with its twin.

2. A downstream probe (reviewer request: show the representation is useful).
   Tissue type (organoid vs hippocampal slice) is predicted from each
   recording's motif-usage profile and, as the control a reviewer would ask
   for, from firing rate alone (log non-blank token count per recording, which
   is proportional to activity because every recording contributes 9 clips).
   A second probe asks which of the two slice patients a slice came from.
   Cross-validation holds out whole groups: a slice patient, or one organoid
   recording, so a fold never trains on a recording it is tested on.

Confound stated with the result: tissue and patient are both confounded with
electrode routing (each patient's slices share 1-2 routing configurations, and
organoid configurations differ from slice ones). Motif usage can reflect
which electrodes are routed inside a patch, so a positive probe shows the
profile is informative, not that it is free of geometry. The V2 plan's
routing-mask conditioning and leave-preparation-out splits address this.

    python analysis/motif_usage_probe.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.stats import DUPLICATE_OF  # noqa: E402

NPZ = ROOT / "reports" / "analysis_motif_reuse.npz"
PROV = ROOT / "reports" / "data_provenance.json"
PUBLISHED = ROOT / "reports" / "analysis_motif_reuse.json"
OUT = ROOT / "reports" / "motif_usage_probe.json"


# Same estimators as analysis/motif_reuse.py, restated so this script runs
# without torch (motif_reuse imports the training stack at module level).
def _entropy(c):
    t = c.sum()
    if t <= 0:
        return 0.0
    p = c[c > 0] / t
    return float(-(p * np.log(p)).sum())


def _entropy_mm(c):
    n = c.sum()
    return 0.0 if n <= 0 else _entropy(c) + (int((c > 0).sum()) - 1) / (2.0 * n)


def _rarefied_sets(counts, n_sub, rng):
    out = []
    for row in counts:
        if int(row.sum()) <= n_sub:
            out.append(set(np.nonzero(row)[0].tolist()))
            continue
        toks = np.repeat(np.arange(row.size), row)
        out.append(set(np.unique(rng.choice(toks, size=n_sub, replace=False)).tolist()))
    return out


def _jaccard(sets):
    n = len(sets)
    J = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            u = len(sets[i] | sets[j])
            J[i, j] = J[j, i] = len(sets[i] & sets[j]) / u if u else 0.0
    return J


def _block(J, rows, cols, same):
    v = [J[i, j] for i in rows for j in cols if (i < j if same else True)]
    return float(np.mean(v)) if v else float("nan")


def reuse_stats(counts, tissue, draws=200, seed=20261107):
    tot = counts.sum()
    w = counts.sum(1) / tot
    H = _entropy_mm(counts.sum(0))
    Hc = float(sum(w[i] * _entropy_mm(counts[i]) for i in range(len(counts))))
    org = [i for i, t in enumerate(tissue) if t == "organoid"]
    sli = [i for i, t in enumerate(tissue) if t == "slice"]
    allr = list(range(len(counts)))
    n_sub = int(counts.sum(1).min())
    rng = np.random.default_rng(seed)
    Jr = np.zeros((len(counts), len(counts)))
    for _ in range(draws):
        Jr += _jaccard(_rarefied_sets(counts, n_sub, rng))
    Jr /= draws
    return {"n_recordings": len(counts), "rarefy_budget": n_sub, "draws": draws,
            "H_code_mm": H, "H_code_given_recording_mm": Hc,
            "retained_fraction_mm": Hc / H,
            "recording_explains_frac_mm": 1 - Hc / H,
            "rarefied_jaccard": {"all": _block(Jr, allr, allr, True),
                                 "within_organoid": _block(Jr, org, org, True),
                                 "within_slice": _block(Jr, sli, sli, True),
                                 "cross_tissue": _block(Jr, org, sli, False)}}


def _cv_probe(X, y, groups, make_model, n_perm, seed):
    """Leave-one-group-out predictions, balanced accuracy, AUC, permutation p."""
    def run(yy):
        pred = np.empty(len(yy), float)
        for g in np.unique(groups):
            te = groups == g
            tr = ~te
            if len(np.unique(yy[tr])) < 2:
                pred[te] = yy[tr].mean()
                continue
            m = make_model().fit(X[tr], yy[tr])
            pred[te] = m.predict_proba(X[te])[:, 1]
        return pred
    p = run(y)
    bacc = balanced_accuracy_score(y, p > 0.5)
    auc = roc_auc_score(y, p)
    rng = np.random.default_rng(seed)
    null = np.array([balanced_accuracy_score(yp := rng.permutation(y), run(yp) > 0.5)
                     for _ in range(n_perm)])
    if n_perm == 0:
        return {"balanced_accuracy": float(bacc), "auc": float(auc)}
    return {"balanced_accuracy": float(bacc), "auc": float(auc),
            "perm_p": float((1 + (null >= bacc).sum()) / (1 + n_perm)),
            "null_mean_bacc": float(null.mean()), "n": int(len(y)),
            "n_groups": int(len(np.unique(groups)))}


def probes(counts, tissue, patient, n_perm=1000, seed=20261107):
    freq = counts / counts.sum(1, keepdims=True)
    motif = np.sqrt(freq)                                # Hellinger embedding
    rate = np.log(counts.sum(1, keepdims=True).astype(float))

    def motif_model():
        return make_pipeline(PCA(n_components=5, random_state=0),
                             StandardScaler(), LogisticRegression(C=1.0))

    def rate_model():
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0))

    def both_model():
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0))

    out = {}
    # tissue: hold out one slice patient, or one organoid recording, per fold
    y = np.array([t == "slice" for t in tissue], int)
    groups = np.array([f"pat:{patient[i]}" if tissue[i] == "slice" else f"rec:{i}"
                       for i in range(len(tissue))])
    pcs = []
    out["tissue"] = {
        "target": "slice (1) vs organoid (0)",
        "cv": "leave-one-group-out; group = slice patient or organoid recording",
        "motif_usage": _cv_probe(motif, y, groups, motif_model, n_perm, seed),
        "rate_only": _cv_probe(rate, y, groups, rate_model, n_perm, seed + 1)}
    # combined: motif PCs fitted inside each fold would be cleaner; PCA here is
    # unsupervised, fitted on all recordings without labels.
    pcs = PCA(n_components=5, random_state=0).fit_transform(motif)
    out["tissue"]["motif_plus_rate"] = _cv_probe(
        np.hstack([pcs, rate]), y, groups, both_model, n_perm, seed + 2)
    # Same probe, but holding out single recordings: other slices of the SAME
    # patient stay in training. The gap between the two schemes is the point --
    # it separates "tissue type" from "this patient / this routing".
    loro = np.arange(len(y))
    out["tissue_leave_one_recording_out"] = {
        "target": out["tissue"]["target"],
        "cv": "leave-one-recording-out (same-patient slices remain in training)",
        "motif_usage": _cv_probe(motif, y, loro, motif_model, n_perm, seed + 5),
        "rate_only": _cv_probe(rate, y, loro, rate_model, n_perm, seed + 6)}
    # settings sweep for the main (leave-patient-out) probe, reported in full
    sweep = {}
    for k in (2, 5, 10):
        for C in (0.1, 1.0, 10.0):
            def mk(k=k, C=C):
                return make_pipeline(PCA(n_components=k, random_state=0),
                                     StandardScaler(), LogisticRegression(C=C))
            sweep[f"pca{k}_C{C:g}"] = _cv_probe(motif, y, groups, mk, 0, seed)[
                "balanced_accuracy"]
    out["tissue"]["motif_usage_settings_sweep_bacc"] = sweep

    # patient, slices only: hold out one recording per fold
    s = np.array([t == "slice" for t in tissue])
    ys = np.array([patient[i].startswith("10F") for i in np.where(s)[0]], int)
    gs = np.arange(int(s.sum()))
    out["slice_patient"] = {
        "target": "patient 10F (1) vs 11G (0), slices only",
        "cv": "leave-one-recording-out",
        "confound": "each patient's slices use their own routing configurations",
        "motif_usage": _cv_probe(motif[s], ys, gs, motif_model, n_perm, seed + 3),
        "rate_only": _cv_probe(rate[s], ys, gs, rate_model, n_perm, seed + 4)}
    return out


def main() -> int:
    warnings.filterwarnings("ignore", category=UserWarning)
    z = np.load(NPZ)
    counts, ids = z["counts"].astype(np.int64), [int(a) for a in z["assay_ids"]]
    prov = {int(r["assay_idx"]): r for r in json.loads(PROV.read_text())["used"]}
    tissue_all = ["slice" if "hippocampal" in prov[a]["preparation"] else "organoid"
                  for a in ids]
    keep = [i for i, a in enumerate(ids) if a not in DUPLICATE_OF]
    c_dedup = counts[keep]
    t_dedup = [tissue_all[i] for i in keep]
    patient = [prov[ids[i]]["subject_id"].split("-")[0] for i in keep]

    pub = json.loads(PUBLISHED.read_text()) if PUBLISHED.is_file() else {}
    res = {
        "source": str(NPZ.relative_to(ROOT)),
        "duplicates_removed": {str(k): v for k, v in DUPLICATE_OF.items()},
        "reuse_with_duplicates": reuse_stats(counts, tissue_all),
        "reuse_without_duplicates": reuse_stats(c_dedup, t_dedup),
        "published_reference_keys": sorted(pub.keys())[:40],
        "probes": probes(c_dedup, t_dedup, patient),
        "caveats": [
            "n = 28 recordings (15 organoid, 13 slice from 2 patients); every "
            "number carries that sample size",
            "tissue and slice patient are confounded with electrode routing",
            "counts come from the 9 test clips per recording on the temporal split",
        ],
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n")

    for k in ("reuse_with_duplicates", "reuse_without_duplicates"):
        r = res[k]
        j = r["rarefied_jaccard"]
        print(f"{k:<26} n={r['n_recordings']:>2} budget={r['rarefy_budget']:>3} "
              f"recording explains {100 * r['recording_explains_frac_mm']:.1f}% | "
              f"Jaccard all {j['all']:.3f} org {j['within_organoid']:.3f} "
              f"slice {j['within_slice']:.3f} cross {j['cross_tissue']:.3f}")
    print("tissue, leave-patient-out, settings sweep (bal.acc):",
          res["probes"]["tissue"]["motif_usage_settings_sweep_bacc"])
    for name, pr in res["probes"].items():
        print(f"\n{name}: {pr['target']} ({pr['cv']})")
        for feat in ("motif_usage", "rate_only", "motif_plus_rate"):
            if feat in pr:
                m = pr[feat]
                print(f"  {feat:<16} bal.acc {m['balanced_accuracy']:.2f}  "
                      f"AUC {m['auc']:.2f}  perm p {m['perm_p']:.3f}  "
                      f"(null bal.acc {m['null_mean_bacc']:.2f}, n={m['n']})")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
