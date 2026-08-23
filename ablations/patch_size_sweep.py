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

Same measurement underlies the sparse-encoder ablation, which is the other
place blank fraction binds. No model is loaded.

    python ablations/patch_size_sweep.py --batches 12
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent))
import MAGVIT_project.main as M                                  # noqa: E402

OUT = ROOT / "reports" / "ablation_patch_size.json"

# Every candidate must divide 48x120x224 exactly; non-dividing sizes are
# reported as skipped rather than silently dropped.
CANDIDATES = [(12, 15, 14), (6, 15, 14), (3, 15, 14), (6, 15, 7),
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
        rows.append({
            "patch": list(ps), "voxels_per_patch": int(pT * pH * pW),
            "n_tokens": int(t * h * w),
            "blank_frac": float(blank.float().mean()),
            "active_tokens": float((~blank).float().sum(1).mean()),
            "spikes_per_active": float(cnt[~blank].float().mean()),
            "is_shipped": ps == shipped})
    out = {"n_clips": int(B), "shape": [T, H, W],
           "voxel_rate": float(X.float().mean()),
           "shipped_patch": list(shipped), "rows": rows}
    OUT.write_text(json.dumps(out, indent=1))

    print(f"{B} val clips, {T}x{H}x{W}, rate {out['voxel_rate']:.3e}\n")
    hdr = (f"{'patch':<13}{'vox':>6}{'tokens':>8}{'blank%':>9}"
           f"{'active tok':>12}{'spikes/active':>15}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        if "skipped" in r:
            print(f"{str(tuple(r['patch'])):<13}  {r['skipped']}"); continue
        print(f"{str(tuple(r['patch'])):<13}{r['voxels_per_patch']:>6}"
              f"{r['n_tokens']:>8}{100*r['blank_frac']:>8.1f}%"
              f"{r['active_tokens']:>12.0f}{r['spikes_per_active']:>15.2f}"
              + ("  <- shipped" if r["is_shipped"] else ""))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
