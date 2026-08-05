#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 13:21:29 2025

@author: derik
"""

# utils/metrics.py
import numpy as np
from typing import Optional
import torch
import torch.nn.functional as F


def f1_from_bool(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """y_true/y_pred: boolean arrays with the same shape."""
    tp = np.logical_and(y_true, y_pred).sum(dtype=np.int64)
    fp = np.logical_and(~y_true, y_pred).sum(dtype=np.int64)
    fn = np.logical_and(y_true, ~y_pred).sum(dtype=np.int64)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    return (2.0 * precision * recall) / (precision + recall + eps)


class PRCurveAccumulator:
    """
    One PR curve for the complete evaluation population.

    TP, FP, and FN are summed across every batch before AUPRC, Best-F1,
    and its operating threshold are calculated. ``radius_t=None`` selects
    exact evaluation; otherwise predictions receive tolerant credit inside
    the configured spatiotemporal radius.
    """

    def __init__(
        self,
        num_thr: int,
        device: torch.device,
        radius_t: Optional[int] = None,
        radius_h: int = 0,
        radius_w: int = 0,
    ):
        self.thr_grid = torch.linspace(
            0.0,
            1.0,
            steps=num_thr,
            device=device,
            dtype=torch.float32,
        )

        # Float64 preserves integer count precision across the full loader.
        self.tp = torch.zeros(num_thr, device=device, dtype=torch.float64)
        self.fp = torch.zeros(num_thr, device=device, dtype=torch.float64)
        self.fn = torch.zeros(num_thr, device=device, dtype=torch.float64)

        self.radius_t = radius_t
        self.radius_h = int(radius_h)
        self.radius_w = int(radius_w)
        self.num_updates = 0

    @staticmethod
    def _summarize(tp, fp, fn, thr_grid):
        precision = tp / (tp + fp).clamp(min=1.0)
        recall = tp / (tp + fn).clamp(min=1.0)

        # Thresholds run low -> high, so reverse them to integrate recall
        # from low -> high without reordering equal-recall points.
        auprc = torch.trapezoid(precision.flip(0), recall.flip(0))

        f1 = 2.0 * tp / (2.0 * tp + fp + fn).clamp(min=1.0)
        best_f1 = f1.max()

        # Prefer the highest threshold when several thresholds have the
        # same Best-F1. This is the conservative choice for sparse data.
        j = torch.where(f1 == best_f1)[0][-1]

        return (
            float(auprc.item()),
            float(best_f1.item()),
            float(thr_grid[j].item()),
        )

    @torch.no_grad()
    def update(
        self,
        prob: torch.Tensor,
        target: torch.Tensor,
        mask_vol: Optional[torch.Tensor],
    ) -> None:
        valid = (
            torch.ones_like(target, dtype=torch.bool)
            if mask_vol is None
            else mask_vol > 0
        )
    
        # Targets outside the evaluated region must not provide tolerant credit.
        target_valid = (target > 0.5) & valid
    
        if self.radius_t is None:
            # Exact evaluation.
            radius_t = 0
            radius_h = 0
            radius_w = 0
    
            target_near = target_valid
            prob_local_max = prob.masked_fill(~valid, -1.0)
    
        else:
            # Tolerant evaluation.
            radius_t = int(self.radius_t)
            radius_h = int(self.radius_h)
            radius_w = int(self.radius_w)
    
            kernel_size = (
                2 * radius_t + 1,
                2 * radius_h + 1,
                2 * radius_w + 1,
            )
            padding = (
                radius_t,
                radius_h,
                radius_w,
            )
    
            # Used to determine whether each prediction is near a target.
            target_near = F.max_pool3d(
                target_valid.float(),
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ) > 0
    
            # Used to determine whether each target has a nearby prediction.
            #
            # Pool probabilities once rather than dilating binary predictions
            # separately at every threshold:
            #
            # max(probability neighborhood) >= threshold
            #
            # is equivalent to:
            #
            # any(binary prediction neighborhood)
            prob_for_pool = prob.masked_fill(~valid, -1.0)
    
            prob_local_max = F.max_pool3d(
                prob_for_pool,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            )
    
        target_total = target_valid.sum().double()
    
        for i, thr in enumerate(self.thr_grid):
            pred_valid = (prob >= thr) & valid
    
            # Predictions that fall within tolerance of a target.
            pred_hits = pred_valid & target_near
    
            # Targets that have at least one prediction within tolerance.
            target_hits = target_valid & (prob_local_max >= thr)
    
            # With radius_t=0, matching is independent between frames.
            # Count-capping per frame prevents extra predictions in one frame
            # from compensating for missed targets in another frame.
            if radius_t == 0:
                reduce_dims = (1, 3, 4)  # retain B and T
            else:
                # Temporal neighborhoods can cross frames, so cap per sample.
                reduce_dims = (1, 2, 3, 4)  # retain B
    
            pred_hit_count = pred_hits.sum(dim=reduce_dims)
            target_hit_count = target_hits.sum(dim=reduce_dims)
    
            # One covered target can provide credit for at most one prediction,
            # and one prediction can recover at most one target.
            matched = torch.minimum(
                pred_hit_count,
                target_hit_count,
            ).sum().double()
    
            pred_total = pred_valid.sum().double()
    
            self.tp[i].add_(matched)
            self.fp[i].add_(pred_total - matched)
            self.fn[i].add_(target_total - matched)
    
        self.num_updates += 1

    @torch.no_grad()
    def compute(self):
        if self.num_updates == 0:
            return 0.0, 0.0, 0.5

        return self._summarize(
            self.tp,
            self.fp,
            self.fn,
            self.thr_grid,
        )

@torch.no_grad()
def get_vq_codebook_stats(model, near_zero_thresh: float = 1e-6):
    stats = {}

    num_levels = int(getattr(model.vq, "num_quantizers", 1))

    stats["vq_num_levels"] = int(num_levels)
    stats["vq_near_zero_thresh"] = float(near_zero_thresh)

    total_alive = 0
    total_dead = 0
    total_codes = 0

    for lvl in range(num_levels):
        if hasattr(model.vq, "get_effective_codebook_weight"):
            cb = model.vq.get_effective_codebook_weight(lvl).detach().float()
        else:
            cb = model.vq.embeds[lvl].weight.detach().float()
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
