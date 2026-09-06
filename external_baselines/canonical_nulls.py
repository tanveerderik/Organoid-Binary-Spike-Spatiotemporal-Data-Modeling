#!/usr/bin/env python3
"""The canonical model-free null block for the completion battery.

The null arms -- separable / marginal / marginal_xa / copy -- contain no model,
so there must be exactly ONE set of numbers for them, shared by every row of
the table. They previously varied by up to 5% between runs because the
train-map sampler inherited the global RNG that model construction had already
perturbed; `task_eval.train_site_maps` fixes that, and this script produces the
single canonical block, building only the VQ-VAE (for ROI geometry) and neither
a prior nor a baseline.

It also writes the arithmetic count control. Every arm is handed the true
clip's `lct` at generation time and its first feature is log mean firing
density, so "predicts the ROI count at r = +0.95" may be no more than
exp(lct_0) x |ROI|. That product, with no model at all, is the floor any count
head must clear, and it belongs in the same file as the other model-free
references.

    python external_baselines/canonical_nulls.py --batches 70

`--batches 0` scores the whole test split. The battery and this script must run
at the SAME budget: the nulls are paired per clip against every model arm, and
a null computed on a different clip set is not a null.

Writes reports/external_baselines/task_eval_nulls.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MAGVIT_project.external_baselines import task_eval as TE
from MAGVIT_project.external_baselines.common import data as bdata
from MAGVIT_project.external_baselines.common.pipeline import build_pipeline
from MAGVIT_project.utils.constants import ACTIVITY_CTX_NAMES

NULLS = ("separable", "marginal", "marginal_xa", "copy")
OUT_PATH = Path("reports/external_baselines/task_eval_nulls.json")


def _fmt(v) -> str:
    return f"{'n/a' if v is None else format(v, '.4f'):>13}"


def canonical_nulls(seed: int, batches: int, train_batches: int) -> dict:
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    P = build_pipeline("cuda", phase="4b", quiet=True)
    vq = P["vqvae"]

    print(f"train site maps from {train_batches} batches (seed-pinned) ...",
          flush=True)
    maps, n_train = TE.train_site_maps(train_batches, seed=seed)
    xa_maps = TE.cross_assay_maps(maps)
    print(f"  {len(maps)} assays from {n_train} clips", flush=True)

    res = {m: {a: [] for a in NULLS} for m in TE.MODES}
    res_site = {m: {a: [] for a in NULLS} for m in TE.MODES}
    roi_frac = {m: [] for m in TE.MODES}
    clip_keys = {m: [] for m in TE.MODES}

    for mode in TE.MODES:
        print(f"\n=== {mode} ===", flush=True)
        for bi, batch in enumerate(bdata.iter_raw("test", batches=(batches or None))):
            x = batch["x"].to(dev).float()
            if x.dim() == 4:
                x = x.unsqueeze(1)
            assays = batch["assay_idx"].tolist()
            B = x.shape[0]
            specs = TE.masks_for_task(batch, x, mode)

            pm0 = vq.predict_mask_from_spec(specs, vq.token_grid, device=dev)
            roi_vox = TE.roi_voxels(vq, pm0, vq.token_grid, x.shape[-3:])
            roi_frac[mode].append(float((pm0 > 0.5).float().mean()))

            sp = TE.spatial_prior_field(maps, assays, x.shape[-3:], dev)
            sp_xa = TE.spatial_prior_field(xa_maps, assays, x.shape[-3:], dev)
            has_vis = bool(((1.0 - roi_vox).sum() > 0).item())
            p_sep = (sp * TE.visible_time_profile(x, roi_vox).view(B, 1, -1, 1, 1)
                     if has_vis else None)
            p_copy = TE.copy_field(x, roi_vox, mode)

            fields = {"separable": p_sep, "marginal": sp,
                      "marginal_xa": sp_xa, "copy": p_copy}
            for b in range(B):
                m = roi_vox[b, 0] > 0.5
                if not bool(m.any()):
                    continue
                tgt = x[b, 0][m]
                if float(tgt.sum()) < 1.0:
                    continue
                k = TE._clip_seed(batch, b)
                clip_keys[mode].append(k)
                sd = seed + int(k % 100003)
                for arm, f in fields.items():
                    res[mode][arm].append(
                        float("nan") if f is None
                        else TE.clip_ap(f[b, 0][m], tgt, seed=sd))
                    res_site[mode][arm].append(
                        float("nan") if f is None
                        else TE.site_ap(f, x, roi_vox, b, sd))
            print(f"  batch {bi}  clips={len(clip_keys[mode])}", flush=True)

    out = {"seed": seed, "batches": batches, "train_batches": train_batches,
           "n_train_clips": n_train, "tasks": {}}
    print("\n" + "=" * 78)
    print("CANONICAL NULLS -- spatiotemporal AP")
    print("=" * 78)
    hdr = f"{'task':<12}" + "".join(f"{a:>13}" for a in NULLS) + f"{'clips':>8}"
    print(hdr)
    for mode in TE.MODES:
        row = {}
        for a in NULLS:
            v = np.array(res[mode][a], dtype=float)
            vs = np.array(res_site[mode][a], dtype=float)
            row[a] = {
                "ap": (float(np.nanmean(v)) if np.isfinite(v).any() else None),
                "site_ap": (float(np.nanmean(vs)) if np.isfinite(vs).any() else None),
                "n": int(np.isfinite(v).sum()),
            }
        out["tasks"][mode] = {
            "arms": row,
            "n_clips": len(clip_keys[mode]),
            "roi_frac": float(np.mean(roi_frac[mode])),
            "clip_keys": [str(k) for k in clip_keys[mode]],
            "per_clip": {a: res[mode][a] for a in NULLS},
            "per_clip_site": {a: res_site[mode][a] for a in NULLS},
        }
        cells = "".join(_fmt(row[a]["ap"]) for a in NULLS)
        print(f"{mode:<12}{cells}{len(clip_keys[mode]):>8}")

    print("\n" + "=" * 78)
    print("CANONICAL NULLS -- site-level AP")
    print("=" * 78)
    print(f"{'task':<12}" + "".join(f"{a:>13}" for a in NULLS))
    for mode in TE.MODES:
        row = out["tasks"][mode]["arms"]
        cells = "".join(_fmt(row[a]["site_ap"]) for a in NULLS)
        print(f"{mode:<12}{cells}")

    return out


def _within_assay_r(pred, true, assays) -> float | None:
    """Correlation after removing each recording's mean from both series.

    Pooled across recordings the correlation is dominated by between-recording
    rate differences, which no arm has to work for -- the recording index alone
    supplies them. Within-recording is the part that requires reading the clip.
    """
    u = np.asarray(pred, float).copy()
    v = np.asarray(true, float).copy()
    a = np.asarray(assays)
    for x in np.unique(a):
        i = a == x
        if i.sum() > 1:
            u[i] -= u[i].mean()
            v[i] -= v[i].mean()
        else:
            u[i] = v[i] = 0.0
    if u.std() <= 0 or v.std() <= 0:
        return None
    return float(np.corrcoef(u, v)[0, 1])


def count_control(vq, seed: int, batches: int, dev) -> dict:
    """ROI count predicted from the conditioning input by arithmetic alone."""
    i0 = ACTIVITY_CTX_NAMES.index("log_mean_firing_density")
    out = {}
    print(f"\n{'task':<11}{'n':>4}{'lct-only r':>13}{'lct-only bias':>15}")
    for mode in TE.MODES:
        pred, true, assays = [], [], []
        for batch in bdata.iter_raw("test", batches=(batches or None)):
            x = batch["x"].to(dev).float()
            if x.dim() == 4:
                x = x.unsqueeze(1)
            lct = batch["local_ctx"].float()
            specs = TE.masks_for_task(batch, x, mode)
            pm0 = vq.predict_mask_from_spec(specs, vq.token_grid, device=dev)
            roi_vox = TE.roi_voxels(vq, pm0, vq.token_grid, x.shape[-3:])
            for b in range(x.shape[0]):
                m = roi_vox[b, 0] > 0.5
                n = int(m.sum())
                if n == 0:
                    continue
                tgt = x[b, 0][m]
                if float(tgt.sum()) < 1.0:
                    continue
                # the entire "model": clip density from lct, times the ROI size
                dens = float(np.exp(float(lct[b, i0])) - 1e-6)
                pred.append(dens * n)
                true.append(float(tgt.sum()))
                assays.append(int(batch["assay_idx"][b]))
        r = _within_assay_r(pred, true, assays)
        bias = (np.mean(pred) - np.mean(true)) / np.mean(true)
        print(f"{mode:<11}{len(true):>4}"
              f"{('--' if r is None else f'{r:+.4f}'):>13}{bias:>+15.1%}")
        out[mode] = {"n": len(true), "r_within_assay": r, "bias": float(bias),
                     "mae": float(np.mean(np.abs(np.array(pred)
                                                 - np.array(true))))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--batches", type=int, default=70,
                    help="test batches to score; 0 = the whole test split")
    ap.add_argument("--train-batches", type=int, default=60,
                    help="train batches the per-recording site maps are built "
                         "from; must cover every recording")
    ap.add_argument("--out", default=str(OUT_PATH))
    a = ap.parse_args()

    out = canonical_nulls(a.seed, a.batches, a.train_batches)

    dev = torch.device("cuda")
    vq = build_pipeline("cuda", phase="4b", quiet=True)["vqvae"]
    out["count_control_lct_only"] = {
        "seed": a.seed, "batches": a.batches,
        "tasks": count_control(vq, a.seed, a.batches, dev),
        "definition": "ROI count predicted as exp(lct[log_mean_firing_density])"
                      " * |ROI|, i.e. arithmetic on the conditioning input with"
                      " no model. Every arm is handed the true clip lct at"
                      " generation time, so this is the floor any count head"
                      " must clear."}

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
