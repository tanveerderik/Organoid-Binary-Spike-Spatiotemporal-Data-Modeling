#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 13:21:29 2025

@author: derik
"""

# utils/metrics.py
import numpy as np
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


@torch.no_grad()
def confusion_at_threshold(
    prob: torch.Tensor,       # (B,1,T,H,W)
    target: torch.Tensor,     # (B,1,T,H,W)
    mask_vol: Optional[torch.Tensor],  # (B,1,T,H,W) or None
    thr: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = (prob >= thr).to(target.dtype)
    if mask_vol is not None:
        pred = pred * mask_vol
        tgt  = target * mask_vol
    else:
        tgt = target
    TP = (pred * tgt).sum()
    FP = (pred * (1.0 - tgt)).sum()
    FN = ((1.0 - pred) * tgt).sum()
    return TP, FP, FN


def pr_curve(prob, target, mask_vol, num_thr, device):
    thr_grid = torch.linspace(0, 1, steps=num_thr, device=device)
    TP = torch.zeros_like(thr_grid)
    FP = torch.zeros_like(thr_grid)
    FN = torch.zeros_like(thr_grid)
    for i, thr in enumerate(thr_grid):
        tp, fp, fn = confusion_at_threshold(prob, target, mask_vol, thr)
        TP[i], FP[i], FN[i] = tp, fp, fn
    precision = TP / (TP + FP).clamp(min=1.0)
    recall    = TP / (TP + FN).clamp(min=1.0)
    rec, idx  = torch.sort(recall)
    prec_sorted = precision[idx]
    auprc = torch.trapz(prec_sorted, rec)
    f1 = (2 * precision * recall) / (precision + recall).clamp(min=1e-8)
    j = torch.argmax(f1)
    return float(auprc.item()), float(f1[j].item()), float(thr_grid[j].item())



def f1_from_bool(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """y_true/y_pred: boolean arrays with the same shape."""
    tp = np.logical_and(y_true, y_pred).sum(dtype=np.int64)
    fp = np.logical_and(~y_true, y_pred).sum(dtype=np.int64)
    fn = np.logical_and(y_true, ~y_pred).sum(dtype=np.int64)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    return (2.0 * precision * recall) / (precision + recall + eps)



@torch.no_grad()
def confusion_at_threshold_tolerant(
    prob: torch.Tensor,       # (B,1,T,H,W)
    target: torch.Tensor,     # (B,1,T,H,W)
    mask_vol: Optional[torch.Tensor],
    thr: float,
    radius_t: int = 1,
    radius_h: int = 0,
    radius_w: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = (prob >= thr).to(target.dtype)

    # Dilate target only: prediction gets credit if it lands near a true spike.
    tgt_tol = F.max_pool3d(
        target.float(),
        kernel_size=(2 * radius_t + 1, 2 * radius_h + 1, 2 * radius_w + 1),
        stride=1,
        padding=(radius_t, radius_h, radius_w),
    ).to(target.dtype)

    if mask_vol is not None:
        pred = pred * mask_vol
        tgt_exact = target * mask_vol
        tgt_tol = tgt_tol * mask_vol
    else:
        tgt_exact = target

    TP = (pred * tgt_tol).sum()
    FP = (pred * (1.0 - tgt_tol)).sum()
    FN = ((1.0 - pred) * tgt_exact).sum()

    return TP, FP, FN


def pr_curve_tolerant(
    prob,
    target,
    mask_vol,
    num_thr,
    device,
    radius_t: int = 1,
    radius_h: int = 0,
    radius_w: int = 0,
):
    thr_grid = torch.linspace(0, 1, steps=num_thr, device=device)
    TP = torch.zeros_like(thr_grid)
    FP = torch.zeros_like(thr_grid)
    FN = torch.zeros_like(thr_grid)

    for i, thr in enumerate(thr_grid):
        tp, fp, fn = confusion_at_threshold_tolerant(
            prob, target, mask_vol, thr,
            radius_t=radius_t, radius_h=radius_h, radius_w=radius_w
        )
        TP[i], FP[i], FN[i] = tp, fp, fn

    precision = TP / (TP + FP).clamp(min=1.0)
    recall    = TP / (TP + FN).clamp(min=1.0)

    rec, idx = torch.sort(recall)
    prec_sorted = precision[idx]
    auprc = torch.trapz(prec_sorted, rec)

    f1 = (2 * precision * recall) / (precision + recall).clamp(min=1e-8)
    j = torch.argmax(f1)

    return float(auprc.item()), float(f1[j].item()), float(thr_grid[j].item())



@torch.no_grad()
def get_vq_codebook_stats(model, near_zero_thresh: float = 1e-6):
    stats = {}

    embeds = getattr(model.vq, "embeds", [])
    num_levels = len(embeds)

    stats["vq_num_levels"] = int(num_levels)
    stats["vq_near_zero_thresh"] = float(near_zero_thresh)

    total_alive = 0
    total_dead = 0
    total_codes = 0

    for lvl, emb in enumerate(embeds):
        cb = emb.weight.detach().float()
        K, D = cb.shape
        norms = cb.norm(dim=1)
        alive = norms >= near_zero_thresh
        n_alive = int(alive.sum().item())
        n_dead = int((~alive).sum().item())

        total_alive += n_alive
        total_dead += n_dead
        total_codes += int(K)

        prefix = f"vq_l{lvl+1}"

        stats[f"{prefix}_num_codes"] = int(K)
        stats[f"{prefix}_code_dim"] = int(D)

        stats[f"{prefix}_alive"] = n_alive
        stats[f"{prefix}_dead"] = n_dead
        stats[f"{prefix}_alive_frac"] = float(n_alive / max(1, K))
        stats[f"{prefix}_dead_frac"] = float(n_dead / max(1, K))

        stats[f"{prefix}_norm_min"] = float(norms.min().item())
        stats[f"{prefix}_norm_max"] = float(norms.max().item())
        stats[f"{prefix}_norm_mean"] = float(norms.mean().item())
        stats[f"{prefix}_norm_std"] = float(norms.std(unbiased=False).item())

        if n_alive > 1:
            cb_alive = F.normalize(cb[alive], dim=1)
            sim = cb_alive @ cb_alive.T
            A = sim.size(0)
            eye = torch.eye(A, device=sim.device, dtype=torch.bool)
            offdiag = sim[~eye]

            stats[f"{prefix}_cos_max"] = float(offdiag.max().item())
            stats[f"{prefix}_cos_min"] = float(offdiag.min().item())
            stats[f"{prefix}_cos_mean"] = float(offdiag.mean().item())
            stats[f"{prefix}_cos_std"] = float(offdiag.std(unbiased=False).item())
        else:
            stats[f"{prefix}_cos_max"] = 0.0
            stats[f"{prefix}_cos_min"] = 0.0
            stats[f"{prefix}_cos_mean"] = 0.0
            stats[f"{prefix}_cos_std"] = 0.0

    stats["vq_total_codes"] = int(total_codes)
    stats["vq_total_alive"] = int(total_alive)
    stats["vq_total_dead"] = int(total_dead)
    stats["vq_total_alive_frac"] = float(total_alive / max(1, total_codes))
    stats["vq_total_dead_frac"] = float(total_dead / max(1, total_codes))

    return stats