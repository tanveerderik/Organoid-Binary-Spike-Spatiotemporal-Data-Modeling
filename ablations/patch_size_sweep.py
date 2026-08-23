#!/usr/bin/env python3
"""Why (6,15,14), measured without training anything.

The patch size is a standing constraint and reads like an unexamined default
unless the trade-off behind it is on the page. It is not a default: shrinking
the patch improves reconstruction and destroys the prior, and the mechanism is
arithmetic over the data rather than a property of any model, so it can be
measured with one pass and no fitting.

The quantity that matters is the BLANK FRACTION of the token population. At
~1.6e-4 voxel rate a patch is empty unless it happens to catch a spike, so as
the patch shrinks the token grid grows much faster than the number of tokens
that contain anything. The prior is then trained on a target distribution that
is almost entirely blank, where predicting blank is close to optimal -- which
is the collapse, and it coexists with BETTER reconstruction because finer
patches decode more sharply.

Reported per candidate patch size:
  blank_frac        fraction of tokens containing no spike
  active_tokens     mean tokens per clip that contain at least one spike
  spikes_per_active mean spikes inside a non-blank patch -- how much there is
                    for an alphabet entry to actually distinguish
  spikes_sd         spread of that, i.e. how heterogeneous non-blank patches are
  capture_k32       fraction of active-patch variance a 32-entry alphabet can
                    describe, from k-means at K=32 on the raw patches. This is
                    the LEARNABILITY side of the trade: the parent codebook has
                    32 entries, so if 32 centroids cannot describe the patch
                    distribution then no amount of training fixes it. Note the
                    quantizer is EMA -- the codebook is a moving average of its
                    assignments, with no gradient and no sparsity penalty on the
                    code vectors. The only usage term is
                    `loss_usage = max_entropy - entropy` at weight 1e-3
                    (model/base.py:424), which pushes usage TOWARD UNIFORM. That
                    is an anti-collapse term, the opposite of a sparsity prior,
                    and it does nothing to help a small codebook cover a
                    high-diversity patch distribution.

Same measurement underlies the sparse-encoder ablation, which is the other
place blank fraction binds. No model is loaded.

    python ablations/patch_size_sweep.py --batches 12
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent))
import MAGVIT_project.main as M                                  # noqa: E402

OUT = ROOT / "reports" / "ablation_patch_size.json"

# Every candidate must divide 48x120x224 exactly; non-dividing sizes are
# reported as skipped rather than silently dropped.
# Both directions from the shipped size. FEWER tokens is not the safe end: the
# patch gets larger and more heterogeneous, so a fixed 32-entry parent codebook
# has to cover far more distinct content, which is what `capture_k32` measures.
CANDIDATES = [(24, 15, 14), (12, 30, 14), (6, 30, 28),      # 256 tokens
              (12, 15, 14), (6, 30, 14), (24, 15, 7),       # 512 tokens
              (6, 15, 14),                                  # 1024, shipped
              (3, 15, 14), (6, 15, 7),                      # 2048
              (3, 15, 7), (6, 10, 7), (3, 10, 7), (2, 8, 7)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", type=int, default=12)
    a = ap.parse_args()

    ad = M.find_assays()
    _tr, val, _te, _m = M.make_loaders(assay_dict=ad,
                                       assay_indices=list(ad.keys()),
                                       per_assay_quota=M.per_assay_quota_stage12)
    vols = []
    for i, b in enumerate(val):
        if i >= a.batches:
            break
        x = b["x"]
        vols.append((x if x.dim() == 5 else x.unsqueeze(1)).bool())
    X = torch.cat(vols)
    B, _, T, H, W = X.shape
    shipped = tuple(int(v) for v in M.patch_size)

    rows = []
    for ps in CANDIDATES:
        pT, pH, pW = ps
        if T % pT or H % pH or W % pW:
            rows.append({"patch": list(ps), "skipped": "does not divide volume"})
            continue
        t, h, w = T // pT, H // pH, W // pW
        pat = (X.view(B, 1, t, pT, h, pH, w, pW)
                .permute(0, 2, 4, 6, 3, 5, 7, 1)
                .reshape(B, t * h * w, pT * pH * pW))
        cnt = pat.sum(-1)
        blank = cnt == 0
        act = pat[~blank].float()
        cap = float("nan")
        if act.shape[0] >= 64:
            # Subsample for cost; the statistic is a variance ratio and is
            # stable well below the full set.
            g = torch.Generator().manual_seed(0)
            idx = torch.randperm(act.shape[0], generator=g)[:4000]
            A = act[idx].numpy()
            km = KMeans(n_clusters=32, n_init=4, random_state=0).fit(A)
            ss_tot = float(((A - A.mean(0)) ** 2).sum())
            ss_res = float(km.inertia_)
            cap = 1.0 - ss_res / max(ss_tot, 1e-12)
        rows.append({
            "patch": list(ps), "voxels_per_patch": int(pT * pH * pW),
            "n_tokens": int(t * h * w),
            "blank_frac": float(blank.float().mean()),
            "active_tokens": float((~blank).float().sum(1).mean()),
            "spikes_per_active": float(cnt[~blank].float().mean()),
            "spikes_sd": float(cnt[~blank].float().std()),
            "capture_k32": cap,
            "is_shipped": ps == shipped})
    out = {"n_clips": int(B), "shape": [T, H, W],
           "voxel_rate": float(X.float().mean()),
           "shipped_patch": list(shipped), "rows": rows}
    OUT.write_text(json.dumps(out, indent=1))

    print(f"{B} val clips, {T}x{H}x{W}, rate {out['voxel_rate']:.3e}\n")
    hdr = (f"{'patch':<13}{'vox':>6}{'tokens':>8}{'blank%':>9}"
           f"{'active tok':>12}{'spk/act':>9}{'spk sd':>8}{'capture@32':>12}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        if "skipped" in r:
            print(f"{str(tuple(r['patch'])):<13}  {r['skipped']}"); continue
        print(f"{str(tuple(r['patch'])):<13}{r['voxels_per_patch']:>6}"
              f"{r['n_tokens']:>8}{100*r['blank_frac']:>8.1f}%"
              f"{r['active_tokens']:>12.0f}{r['spikes_per_active']:>9.2f}"
              f"{r['spikes_sd']:>8.2f}{r['capture_k32']:>12.3f}"
              + ("  <- shipped" if r["is_shipped"] else ""))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
