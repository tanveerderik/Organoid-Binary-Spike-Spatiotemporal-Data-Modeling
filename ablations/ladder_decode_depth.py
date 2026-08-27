#!/usr/bin/env python3
"""Does z3 change what the DECODER produces, or only the code index?

The content analysis found z3 near-inert: on ladder paths it adds -0.009 excess
eta^2 on spike count and +0.019 on temporal spread, for 550 extra paths. But
that target is three summary statistics, and z3 could carry exact within-patch
placement they cannot resolve. This is the confirming test, and it is the one
that matters, because it asks the question in the decoder's own terms.

Encode ONCE with all three levels, then decode the same codes at each
cumulative depth -- z1, z1+z2, z1+z2+z3 -- via
`decode_from_codes(..., return_all_refinements=True)`, which returns exactly
that ladder. Nothing is retrained and nothing is re-encoded, so the only thing
that varies is how much of the residual sum reaches the decoder.

Metric is `task_eval.clip_ap`, the same exact step-wise AP the diagnostics use,
imported rather than restated (the trapezoid form in utils/metrics.py
interpolates PR space and is invalid -- Davis & Goadrich 2006). Paired per clip,
so the comparison is a paired Wilcoxon and not a difference of means.

    python ablations/ladder_decode_depth.py --batches 12
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent))
import MAGVIT_project.main as M                                    # noqa: E402
from MAGVIT_project.ablations import resolve_out                   # noqa: E402
from MAGVIT_project.ablations.sparse_encoder import build          # noqa: E402
from MAGVIT_project.external_baselines.task_eval import clip_ap    # noqa: E402

OUT = ROOT / "reports" / "ablation_ladder_decode_depth.json"


CKPT_DEFAULT = "ckpts/vqvae_stage2a_best.pt"


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--out", default=None,
                    help="explicit output path; overrides --tag")
    ap.add_argument("--tag", default=None,
                    help="suffix for the output filename; REQUIRED when --ckpt "
                         "is not the shipped checkpoint, so a non-shipped "
                         "result cannot overwrite the shipped one")
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260823)
    a = ap.parse_args()
    out_path = resolve_out(OUT, ckpt_overridden=(a.ckpt != CKPT_DEFAULT),
                           out=a.out, tag=a.tag)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ad = M.find_assays()
    _tr, val, _te, _m = M.make_loaders(assay_dict=ad,
                                       assay_indices=list(ad.keys()),
                                       per_assay_quota=M.per_assay_quota_stage12)
    b0 = next(iter(val))
    model = build(tuple(b0["x"].shape[-3:]), tuple(map(int, b0["full_hw"][0])),
                  dev, dense=False, seed=0)
    sd = torch.load(a.ckpt, map_location="cpu")
    model.load_state_dict(sd.get("model", sd.get("state_dict", sd)), strict=False)
    model.eval()
    L = int(model.vq.active_quantizers)
    print(f"active_quantizers = {L}", flush=True)

    per = {l: [] for l in range(1, L + 1)}
    for i, batch in enumerate(val):
        if i >= a.batches: break
        x = batch["x"].to(dev).float()
        if x.dim() == 4: x = x.unsqueeze(1)
        gct = batch["global_ctx"].to(dev).float()
        lct = batch["local_ctx"].to(dev).float()
        # NOTE: `decode_from_codes(..., return_all_refinements=True)` cannot be
        # used. Its refinement branch calls `_apply_hole` (vqvae.py:555), which
        # is a closure defined inside `forward` (vqvae.py:1139) and is not in
        # scope there -- that branch raises NameError on every call and has
        # evidently never been exercised. `forward` has the same option and IS
        # exercised (train_vqvae.py:360, eval_vqvae.py:314), so it is the route
        # with test coverage behind it.
        out = model(x, local_ctx=lct, global_ctx=gct,
                    return_all_refinements=True)
        refs = out["refinements"]
        if len(refs) != L:
            raise RuntimeError(f"expected {L} cumulative decodes, got {len(refs)}")
        for lvl, r in enumerate(refs, start=1):
            v = r["logits_vol"]
            if v.dim() == 4: v = v.unsqueeze(1)
            for b in range(x.shape[0]):
                # Same clip, same seed at every depth, so the random tie-break
                # is identical across levels and cannot contribute a difference.
                per[lvl].append(clip_ap(v[b, 0], x[b, 0], seed=a.seed + i * 100 + b))
        print(f"  batch {i}  clips={len(per[1])}", flush=True)

    arr = {l: np.array(v, float) for l, v in per.items()}
    ok = np.ones(len(arr[1]), bool)
    for v in arr.values(): ok &= np.isfinite(v)
    res = {"ckpt": a.ckpt, "batches": a.batches, "n_clips": int(ok.sum()),
           "levels": {}, "increments": {}}

    print(f"\nexact step-wise AP by decode depth, n={int(ok.sum())} paired clips")
    print(f"{'depth':<14}{'mean AP':>10}{'sd':>9}")
    print("-" * 33)
    names = {1: "z1", 2: "z1+z2", 3: "z1+z2+z3"}
    for l in sorted(arr):
        v = arr[l][ok]
        res["levels"][names.get(l, str(l))] = {
            "mean": float(v.mean()), "sd": float(v.std()),
            "per_clip": v.tolist()}
        print(f"{names.get(l, l):<14}{v.mean():>10.4f}{v.std():>9.4f}")

    print(f"\npaired increments (Wilcoxon over the same clips)")
    print(f"{'step':<20}{'delta':>10}{'median':>10}{'win%':>8}{'p':>12}")
    print("-" * 60)
    for l in sorted(arr)[1:]:
        x_, y_ = arr[l - 1][ok], arr[l][ok]
        d = y_ - x_
        try: p = float(wilcoxon(x_, y_).pvalue)
        except Exception: p = float("nan")
        step = f"{names.get(l-1, l-1)} -> {names.get(l, l)}"
        res["increments"][step] = {
            "mean_delta": float(d.mean()), "median_delta": float(np.median(d)),
            "win_frac": float((d > 0).mean()), "wilcoxon_p": p}
        print(f"{step:<20}{d.mean():>+10.4f}{np.median(d):>+10.4f}"
              f"{100*(d>0).mean():>7.0f}%{p:>12.3e}")

    out_path.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
