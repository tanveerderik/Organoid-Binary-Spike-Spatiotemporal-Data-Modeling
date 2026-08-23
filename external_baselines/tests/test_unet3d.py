#!/usr/bin/env python3
"""The two things that can silently invalidate the U-Net baseline.

1. **Hole geometry.** The U-Net builds its own training holes, because it has no
   tokenizer to build them from. If `unet3d/masks.py` and the pipeline's
   `VQVAE.predict_mask_from_spec` disagree by even one patch, the baseline
   trains on one problem and is scored on another, and the resulting number is
   uninterpretable in either direction. Checked exhaustively on the causal
   grid, on every noncausal span, and on random spatial boxes -- not on a
   handful of examples.

2. **Free generation.** Task 0 is a completion with an all-ones ROI
   (`recon-task0-is-free-generation`). If `sample()` took a different path from
   `complete()` the recon column would measure two different models. Checked as
   a bitwise identity, not by eye.

Runs without a fitted checkpoint for part 1 (geometry needs no weights) and
skips part 2 if `ckpts/external_baselines/unet3d.pt` is absent.

    python external_baselines/tests/test_unet3d.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/media/derik/Seagate Desktop Drive/organoid_data")
from MAGVIT_project.external_baselines.common import data as bdata          # noqa: E402
from MAGVIT_project.external_baselines.common.pipeline import build_pipeline  # noqa: E402

bdata.ensure_repo_cwd()
from MAGVIT_project.external_baselines import registry                      # noqa: E402
from MAGVIT_project.external_baselines.task_eval import roi_voxels          # noqa: E402
from MAGVIT_project.external_baselines.unet3d.masks import (                # noqa: E402
    PATCH, sample_spec, spec_to_roi)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = Path("ckpts/external_baselines/unet3d.pt")

n_pass = n_fail = 0


def check(name, ok, detail=""):
    global n_pass, n_fail
    if ok:
        n_pass += 1
        print(f"  [PASS] {name}  {detail}")
    else:
        n_fail += 1
        print(f"  [FAIL] {name}  {detail}")


def _specs_to_check(T, H, W):
    """Every causal prefix and every noncausal start, plus 200 random boxes and
    the draws `sample_spec` itself produces. Enumerating the two temporal
    families is cheap (T=48) and removes the possibility that a boundary case
    was simply never sampled."""
    out = [{"type": "recon"}]
    out += [{"type": "causal", "prefix_frames": pf} for pf in range(1, T)]
    L = max(1, int(round(0.30 * T)))
    out += [{"type": "noncausal", "time_spans": [(s, s + L)]}
            for s in range(0, T - L + 1)]
    rng = np.random.default_rng(7)
    for _ in range(200):
        bh = int(rng.integers(1, H + 1)); bw = int(rng.integers(1, W + 1))
        y0 = int(rng.integers(0, H - bh + 1)); x0 = int(rng.integers(0, W - bw + 1))
        out.append({"type": "spatial", "spatial_box": (y0, y0 + bh, x0, x0 + bw)})
    out += [sample_spec(rng, T, H, W) for _ in range(400)]
    return out


def main() -> int:
    P = build_pipeline(DEV, quiet=True)
    vq = P["vqvae"]
    T, H, W = P["shape"]

    check("patch constant matches the shipped VQ-VAE",
          tuple(int(v) for v in vq.patch_size) == PATCH,
          f"{tuple(int(v) for v in vq.patch_size)} vs {PATCH}")

    specs = _specs_to_check(T, H, W)
    # In chunks: one ROI is 1.29M voxels, and the comparison holds two of them.
    pt, ph, pw = PATCH
    n_bad, n_const, fmin, fmax = 0, True, 1.0, 0.0
    for i in range(0, len(specs), 32):
        ch = specs[i:i + 32]
        mine = spec_to_roi(ch, (T, H, W), DEV)
        theirs = roi_voxels(
            vq, vq.predict_mask_from_spec(ch, vq.token_grid, device=DEV),
            vq.token_grid, (T, H, W))
        n_bad += int((mine != theirs).flatten(1).any(dim=1).sum())
        frac = mine.flatten(1).mean(1)
        fmin, fmax = min(fmin, float(frac.min())), max(fmax, float(frac.max()))
        blk = mine.reshape(len(ch), T // pt, pt, H // ph, ph, W // pw, pw)
        n_const &= bool((blk.amax(dim=(2, 4, 6)) == blk.amin(dim=(2, 4, 6))).all())
    kinds = sorted({s["type"] for s in specs})
    check(f"voxel ROI matches predict_mask_from_spec on {len(specs)} specs "
          f"({', '.join(kinds)})",
          n_bad == 0, f"mismatched specs = {n_bad}/{len(specs)}")

    # A vacuous pass would be all-ones everywhere; require real variation.
    check("ROIs are non-degenerate", fmin < 0.5 < fmax,
          f"fraction of volume in [{fmin:.3f}, {fmax:.3f}]")

    check("every ROI is constant within a patch", n_const)

    check("sample_spec is reproducible from its seed",
          [sample_spec(np.random.default_rng(3), T, H, W) for _ in range(20)]
          == [sample_spec(np.random.default_rng(3), T, H, W) for _ in range(20)])

    if not CKPT.exists():
        print(f"\n  [skip] free-generation identity: no {CKPT}")
    else:
        b = registry.build("unet3d")
        b.load(CKPT, device=DEV)
        batch = next(iter(bdata.iter_raw("test", batches=1)))
        cond, _ = bdata.split_batch(batch)
        cond = cond.to(DEV)
        x = batch["x"].to(DEV).float()
        if x.dim() == 4:
            x = x.unsqueeze(1)
        ones = torch.ones(x.shape[0], 1, T, H, W, device=DEV)
        free = b.sample_intensity(cond)
        comp = b.complete(cond, x, ones)
        check("free generation == completion with an all-ones ROI (bitwise)",
              bool(torch.equal(free, comp)),
              f"max |diff| = {float((free - comp).abs().max()):.3g}")
        check("free-generation field is non-degenerate",
              float(free.std()) > 0.0, f"sd={float(free.std()):.4g}")

    print(f"\n{n_pass}/{n_pass + n_fail} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
