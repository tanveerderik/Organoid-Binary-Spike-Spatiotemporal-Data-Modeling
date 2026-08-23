"""Hole geometry for the four completion tasks, reproduced outside the pipeline.

The U-Net has no tokenizer, so nothing in it needs a token grid -- but the
*evaluation* ROI comes from `VQVAE.predict_mask_from_spec`, which snaps every
hole to the (6,15,14) patch lattice. A model trained on ragged voxel-aligned
holes and scored on patch-aligned ones is being asked a slightly different
question at test time, and the difference is not small: a patch is 210
electrodes wide, so snapping moves the boundary by up to 7 rows and 13 columns.

So the same snapping is applied here, by the same index arithmetic rather than
by a pooling trick that happens to agree. `tests/test_unet3d.py` checks this
function against the pipeline's on the specs the loader actually emits.

`sample_spec` reproduces `NpzBurstDataset.__getitem__`'s draw (dataset.py:661-680
and `_sample_large_spatial_box`, :540) so that training-time holes come from the
distribution the shipped model trained on. It is re-drawn every epoch instead of
being cached with the clip -- the clips are cached for speed, and freezing one
hole per clip for the whole run would hand the U-Net 480 fixed puzzles.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

PATCH: Tuple[int, int, int] = (6, 15, 14)

# The four task names and their ids, in `dataset.py`'s order. 25% each.
TASKS: Tuple[str, ...] = ("recon", "causal", "noncausal", "spatial")
TASK_ID = {k: i for i, k in enumerate(TASKS)}


def sample_spec(rng: np.random.Generator, T: int, H: int, W: int) -> Dict[str, Any]:
    """One mask spec, drawn as `dataset.py` draws it."""
    mode = TASKS[int(rng.integers(0, len(TASKS)))]

    if mode == "causal":
        pf = int(round(T * float(rng.uniform(0.25, 0.75))))
        return {"type": "causal", "prefix_frames": int(np.clip(pf, 1, T - 1))}

    if mode == "noncausal":
        L = max(1, int(round(0.30 * T)))
        s = int(rng.integers(0, max(1, T - L + 1)))
        return {"type": "noncausal", "time_spans": [(s, s + L)]}

    if mode == "spatial":
        a = float(rng.uniform(0.25, 0.60))
        r = float(rng.uniform(0.5, 2.0))
        A = max(1.0, a * H * W)
        bh = int(np.clip(round(np.sqrt(A * r)), 1, H))
        bw = int(np.clip(round(np.sqrt(A / r)), 1, W))
        y0 = int(rng.integers(0, max(1, H - bh + 1)))
        x0 = int(rng.integers(0, max(1, W - bw + 1)))
        return {"type": "spatial", "spatial_box": (y0, y0 + bh, x0, x0 + bw)}

    return {"type": "recon"}


def _token_mask(spec: Dict[str, Any], grid: Tuple[int, int, int]) -> np.ndarray:
    """(t,h,w) float 0/1 token mask -- `VQVAE.predict_mask_from_spec` restated.

    Every branch below mirrors one branch of that method, including its two
    guards: the causal prefix is clamped into [1, t-1], and an all-zero mask
    falls back to all-ones. Both are reachable only for holes outside the
    distribution `sample_spec` draws from, but a baseline that diverges from the
    pipeline exactly where the pipeline defends itself is not a baseline.
    """
    tt, th, tw = grid
    pt, ph, pw = PATCH
    stype = spec.get("type", "recon")

    if stype == "causal" and tt > 1:
        pm = np.zeros((tt, th, tw), np.float32)
        k = int(np.clip(int(spec.get("prefix_frames", 0)) // pt, 1, tt - 1))
        pm[k:] = 1.0
    elif stype == "noncausal":
        pm = np.zeros((tt, th, tw), np.float32)
        for a, b in spec.get("time_spans", []):
            a, b = int(a), int(b)
            if b <= a:
                continue
            ka = int(np.clip(a // pt, 0, tt - 1))
            kb = int(np.clip((b - 1) // pt, 0, tt - 1))
            if kb >= ka:
                pm[ka:kb + 1] = 1.0
    elif stype == "spatial" and spec.get("spatial_box") is not None:
        y0, y1, x0, x1 = map(int, spec["spatial_box"])
        ay = int(np.clip(y0 // ph, 0, th - 1))
        by = int(np.clip((y1 - 1) // ph, 0, th - 1))
        ax = int(np.clip(x0 // pw, 0, tw - 1))
        bx = int(np.clip((x1 - 1) // pw, 0, tw - 1))
        pm = np.zeros((tt, th, tw), np.float32)
        if by >= ay and bx >= ax:
            pm[:, ay:by + 1, ax:bx + 1] = 1.0
    else:
        pm = np.ones((tt, th, tw), np.float32)

    if pm.sum() <= 0.0:
        pm = np.ones((tt, th, tw), np.float32)
    return pm


def spec_to_roi(specs: Sequence[Dict[str, Any]], shape: Tuple[int, int, int],
                device) -> torch.Tensor:
    """(B,1,T,H,W) float 0/1 voxel ROI, snapped to the patch lattice.

    1 = to be predicted (the hole). The volume dimensions must be exact
    multiples of PATCH; they are, by construction -- the dataset pads 220 -> 224
    for precisely this reason -- and an assertion here beats a silent crop.
    """
    T, H, W = map(int, shape)
    pt, ph, pw = PATCH
    if (T % pt, H % ph, W % pw) != (0, 0, 0):
        raise ValueError(f"volume {(T, H, W)} is not a multiple of patch {PATCH}")
    grid = (T // pt, H // ph, W // pw)

    tok = np.stack([_token_mask(s, grid) for s in specs])          # (B,t,h,w)
    v = torch.from_numpy(tok).to(device)
    v = v.repeat_interleave(pt, 1).repeat_interleave(ph, 2).repeat_interleave(pw, 3)
    return v.unsqueeze(1).contiguous()


def task_ids(specs: Sequence[Dict[str, Any]]) -> List[int]:
    return [TASK_ID.get(s.get("type", "recon"), 0) for s in specs]
