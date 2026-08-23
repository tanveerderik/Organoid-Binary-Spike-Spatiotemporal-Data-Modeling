#!/usr/bin/env python3
"""Completion must not read the held-out region.

`SpikeVolumeBaseline.complete` promises it reads `real` only where `roi == 0`.
This is the enforcement: re-run every completion with the ROI voxels replaced
by noise and require a BITWISE-identical result. A model that peeks produces a
different field and fails here rather than in the paper.

Also checks the mechanics that would silently produce a plausible wrong number:
the visible tokens really are pinned, and the token ROI really is patch-aligned.

    python external_baselines/tests/test_task_completion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, "/media/derik/Seagate Desktop Drive/organoid_data")
from MAGVIT_project.external_baselines.common import data as bdata     # noqa: E402
from MAGVIT_project.external_baselines.common.pipeline import build_pipeline  # noqa: E402

bdata.ensure_repo_cwd()
from MAGVIT_project.external_baselines import registry                 # noqa: E402
from MAGVIT_project.external_baselines.task_eval import (              # noqa: E402
    masks_for_task, roi_voxels)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 20260822
CKPTS = Path("ckpts/external_baselines")

n_pass = n_fail = 0


def check(name, ok, detail=""):
    global n_pass, n_fail
    if ok:
        n_pass += 1
        print(f"  [PASS] {name}  {detail}")
    else:
        n_fail += 1
        print(f"  [FAIL] {name}  {detail}")


def main() -> int:
    P = build_pipeline(DEV, phase="4b", quiet=True)
    vq = P["vqvae"]

    batch = next(iter(bdata.iter_raw("test", batches=1)))
    x = batch["x"].to(DEV).float()
    if x.dim() == 4:
        x = x.unsqueeze(1)
    cond, _ = bdata.split_batch(batch)
    cond = cond.to(DEV)

    for mode in ("causal", "noncausal", "spatial"):
        specs = masks_for_task(batch, x, mode)
        pm = vq.predict_mask_from_spec(specs, vq.token_grid, device=DEV)
        roi = roi_voxels(vq, pm, vq.token_grid, x.shape[-3:])
        print(f"\n--- {mode}  (ROI {float(roi.mean()):.3f} of voxels) ---")

        # A corrupted clip that differs ONLY inside the hole.
        g = torch.Generator(device=DEV).manual_seed(1)
        noise = (torch.rand(x.shape, generator=g, device=DEV) < 0.5).float()
        x_bad = x * (1.0 - roi) + noise * roi
        check(f"{mode}: corruption touches only the ROI",
              bool(torch.equal(x * (1 - roi), x_bad * (1 - roi))) and
              not bool(torch.equal(x, x_bad)))

        for name in ("maskgit_flat", "unet3d", "glm", "dg"):
            ck = CKPTS / f"{name}.pt"
            if not ck.exists():
                print(f"  [skip] {name}: no checkpoint")
                continue
            b = registry.build(name)
            b.load(ck, device=DEV)

            def draw(vol):
                gg = torch.Generator().manual_seed(SEED)
                torch.manual_seed(SEED)
                if DEV == "cuda":
                    torch.cuda.manual_seed_all(SEED)
                return b.complete(cond, vol, roi, generator=gg)

            a = draw(x)
            if a is None:
                check(f"{mode}/{name}: declares no completion mechanism", True,
                      "(free-generation fallback, labelled in the report)")
                continue
            c = draw(x_bad)
            same = bool(torch.equal(a, c))
            check(f"{mode}/{name}: completion ignores held-out voxels (bitwise)",
                  same, f"differing voxels = {int((a != c).sum())}")

            # ...and it must not be trivially constant, which would also pass.
            check(f"{mode}/{name}: field is non-degenerate",
                  float(a.float().std()) > 0.0, f"sd={float(a.float().std()):.4g}")

        # MaskGIT mechanics: pinned tokens and patch alignment.
        ck = CKPTS / "maskgit_flat.pt"
        if ck.exists():
            import torch.nn.functional as F
            from MAGVIT_project.external_baselines.maskgit_flat.tokenizer import PATCH
            b = registry.build("maskgit_flat")
            b.load(ck, device=DEV)
            rt = F.max_pool3d(roi[:, :1], kernel_size=PATCH, stride=PATCH)
            rt = rt.reshape(roi.shape[0], -1) > 0.5
            up = F.interpolate(rt.float().reshape(roi.shape[0], 1, *b.tokenizer.grid),
                               scale_factor=PATCH, mode="nearest")
            check(f"{mode}: token ROI is patch-aligned (round-trips exactly)",
                  bool(torch.equal(up, roi[:, :1])))

            ids = b.tokenizer.tokens(x * (1.0 - roi)).reshape(roi.shape[0], -1)
            gg = torch.Generator().manual_seed(SEED)
            out = b.prior.complete(ids, rt, cond.global_ctx, cond.local_ctx,
                                   steps=4, generator=gg)
            check(f"{mode}: visible tokens are pinned",
                  bool(torch.equal(out[~rt], ids[~rt])),
                  f"changed = {int((out[~rt] != ids[~rt]).sum())}")
            check(f"{mode}: no MASK token survives decoding",
                  bool((out != b.prior.mask_id).all()))

    print(f"\n{n_pass}/{n_pass + n_fail} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
