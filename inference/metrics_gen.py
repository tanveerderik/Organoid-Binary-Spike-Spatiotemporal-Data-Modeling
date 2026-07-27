#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:29:07 2026

@author: derik
"""

from typing import Optional, Sequence

import torch

from ..utils.constants import normalize_gap_bins, gap_bin_label
from ..utils.losses import (
    spatial_support_violation_loss,
    soft_binary_from_logits,
)

from ..utils.recon import (
    dilate_spatial_support_hw,
    sample_full_pixel_map_to_crop,
    sample_full_token_map_to_crop,
    soft_spatial_pixel_map_from_logits,
    soft_spatial_token_map_from_logits,
    spatial_pixel_map_from_input,
    spatial_token_map_from_input,
)


def _optional_float(x):
    if x is None:
        return None
    x = float(x)
    return x if torch.isfinite(torch.tensor(x)) else None


def _correlation(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)

    if a.numel() < 2 or b.numel() != a.numel():
        return None

    a = a - a.mean()
    b = b - b.mean()
    den = a.square().sum().sqrt() * b.square().sum().sqrt()

    if den <= 1e-12:
        return None

    return float((a * b).sum().div(den).item())


def _overlap_metrics(generated, target, allowed):
    generated = generated.bool()
    target = target.bool()
    allowed = allowed.bool()

    n_gen = int(generated.sum().item())
    n_target = int(target.sum().item())
    n_allowed = int(allowed.sum().item())

    overlap_target = int((generated & target).sum().item())
    overlap_allowed = int((generated & allowed).sum().item())
    union_target = int((generated | target).sum().item())

    return {
        "spatial_target_support_size": n_target,
        "spatial_generated_support_size": n_gen,
        "spatial_allowed_support_size": n_allowed,
        "spatial_allowed_region_coverage": (
            overlap_allowed / n_allowed if n_allowed > 0 else None
        ),
        "spatial_target_recall": (
            overlap_target / n_target if n_target > 0 else None
        ),
        "spatial_generated_precision": (
            overlap_allowed / n_gen if n_gen > 0 else None
        ),
        "spatial_iou": (
            overlap_target / union_target if union_target > 0 else None
        ),
        "spatial_dice": (
            2.0 * overlap_target / (n_gen + n_target)
            if n_gen + n_target > 0 else None
        ),
        "spatial_precision_defined": bool(n_gen > 0),
        "spatial_recall_defined": bool(n_target > 0),
        "spatial_iou_defined": bool(union_target > 0),
        "spatial_dice_defined": bool(n_gen + n_target > 0),
    }


def _gap_rates(values_b1thw, gap_bins, eps=1e-8):
    values = values_b1thw.float()
    B, _, T, _, _ = values.shape

    rates = []
    denominators = []
    valid = []

    for lo, hi in gap_bins:
        num = values.new_zeros((B,))
        den = values.new_zeros((B,))
        has_temporal_pair = False

        for gap in range(lo, hi + 1):
            if T <= gap:
                continue

            has_temporal_pair = True
            x0 = values[:, :, :-gap]
            xg = values[:, :, gap:]

            num += (x0 * xg).sum(dim=(1, 2, 3, 4))
            den += x0.sum(dim=(1, 2, 3, 4))

        rates.append(num / den.clamp_min(eps))
        denominators.append(den)
        valid.append(
            torch.full(
                (B,),
                has_temporal_pair,
                device=values.device,
                dtype=torch.bool,
            ) & den.gt(eps)
        )

    return (
        torch.stack(rates, dim=1),
        torch.stack(denominators, dim=1),
        torch.stack(valid, dim=1),
    )


def _adjacency_metrics(
    *,
    logits,
    x_hard,
    target,
    confidence,
    gap_bins,
    tau,
    prob_threshold,
    margin,
):
    soft_values = soft_binary_from_logits(
        logits.float().clamp(-20.0, 20.0),
        tau=tau,
        prob_threshold=prob_threshold,
    )
    hard_values = x_hard.float()

    soft_rate, soft_den, soft_valid = _gap_rates(soft_values, gap_bins)
    hard_rate, hard_den, hard_valid = _gap_rates(hard_values, gap_bins)

    allowed = (1.0 + margin) * target
    rows = []

    for b in range(x_hard.shape[0]):
        per_bin = []

        for j, gap_bin in enumerate(gap_bins):
            target_j = float(target[b, j].item())
            allowed_j = float(allowed[b, j].item())
            conf_j = float(confidence[b, j].item())

            soft_defined = bool(soft_valid[b, j].item())
            hard_defined = bool(hard_valid[b, j].item())

            soft_j = float(soft_rate[b, j].item()) if soft_defined else None
            hard_j = float(hard_rate[b, j].item()) if hard_defined else None

            per_bin.append({
                "gap_bin": [int(gap_bin[0]), int(gap_bin[1])],
                "gap_label": gap_bin_label(gap_bin),
                "soft_generated_gap_rate": soft_j,
                "hard_generated_gap_rate": hard_j,
                "target_gap_rate": target_j,
                "allowed_gap_rate": allowed_j,
                "confidence": conf_j,
                "soft_valid_denominator": float(soft_den[b, j].item()),
                "hard_valid_denominator": float(hard_den[b, j].item()),
                "soft_defined": soft_defined,
                "hard_defined": hard_defined,
                "soft_minus_target": (
                    soft_j - target_j if soft_defined else None
                ),
                "hard_minus_target": (
                    hard_j - target_j if hard_defined else None
                ),
                "soft_absolute_error": (
                    abs(soft_j - target_j) if soft_defined else None
                ),
                "hard_absolute_error": (
                    abs(hard_j - target_j) if hard_defined else None
                ),
                "soft_ratio_to_target": (
                    soft_j / target_j
                    if soft_defined and target_j > 0 else None
                ),
                "hard_ratio_to_target": (
                    hard_j / target_j
                    if hard_defined and target_j > 0 else None
                ),
                "soft_excess_over_allowed": (
                    max(0.0, soft_j - allowed_j)
                    if soft_defined else None
                ),
                "hard_excess_over_allowed": (
                    max(0.0, hard_j - allowed_j)
                    if hard_defined else None
                ),
                "soft_below_target": (
                    max(0.0, target_j - soft_j)
                    if soft_defined else None
                ),
                "hard_below_target": (
                    max(0.0, target_j - hard_j)
                    if hard_defined else None
                ),
            })

        def aggregate(which):
            valid_key = f"{which}_defined"
            rate_key = f"{which}_generated_gap_rate"
            valid_rows = [r for r in per_bin if r[valid_key]]

            if not valid_rows:
                return {
                    "defined": False,
                    "excess_loss": None,
                    "weighted_mae": None,
                    "weighted_rmse": None,
                    "mean_signed_error": None,
                    "bins_above_allowed": None,
                    "bins_below_target": None,
                    "target_match_fraction": None,
                }

            weights = torch.tensor(
                [r["confidence"] for r in valid_rows],
                dtype=torch.float64,
            )
            errors = torch.tensor(
                [
                    r[rate_key] - r["target_gap_rate"]
                    for r in valid_rows
                ],
                dtype=torch.float64,
            )
            excess = torch.tensor(
                [
                    max(0.0, r[rate_key] - r["allowed_gap_rate"])
                    for r in valid_rows
                ],
                dtype=torch.float64,
            )

            weight_sum = float(weights.sum().item())
            weighted_defined = weight_sum > 0

            matches = [
                abs(r[rate_key] - r["target_gap_rate"])
                <= max(1e-8, margin * abs(r["target_gap_rate"]))
                for r in valid_rows
            ]

            return {
                "defined": True,
                "excess_loss": (
                    float((weights * excess.square()).sum().item() / weight_sum)
                    if weighted_defined else None
                ),
                "weighted_mae": (
                    float((weights * errors.abs()).sum().item() / weight_sum)
                    if weighted_defined else None
                ),
                "weighted_rmse": (
                    float(
                        (
                            (weights * errors.square()).sum() / weight_sum
                        ).sqrt().item()
                    )
                    if weighted_defined else None
                ),
                "mean_signed_error": float(errors.mean().item()),
                "bins_above_allowed": int(excess.gt(0).sum().item()),
                "bins_below_target": int(errors.lt(0).sum().item()),
                "target_match_fraction": (
                    float(sum(matches) / len(matches))
                ),
            }

        soft_agg = aggregate("soft")
        hard_agg = aggregate("hard")

        rows.append({
            "adjacency_metric_available": True,
            "adjacency_metric_defined": {
                "soft": soft_agg["defined"],
                "hard": hard_agg["defined"],
            },
            "adjacency_status": "ok",
            "adjacency_status_reason": None,
            "adjacency_valid_denominator": {
                "soft": [r["soft_valid_denominator"] for r in per_bin],
                "hard": [r["hard_valid_denominator"] for r in per_bin],
            },
            "adjacency_gap_bins": per_bin,
            "adjacency_soft_excess_loss": soft_agg["excess_loss"],
            "adjacency_hard_excess_loss": hard_agg["excess_loss"],
            "adjacency_soft_weighted_mae": soft_agg["weighted_mae"],
            "adjacency_hard_weighted_mae": hard_agg["weighted_mae"],
            "adjacency_soft_weighted_rmse": soft_agg["weighted_rmse"],
            "adjacency_hard_weighted_rmse": hard_agg["weighted_rmse"],
            "adjacency_soft_mean_signed_error": soft_agg["mean_signed_error"],
            "adjacency_hard_mean_signed_error": hard_agg["mean_signed_error"],
            "adjacency_bins_above_allowed": {
                "soft": soft_agg["bins_above_allowed"],
                "hard": hard_agg["bins_above_allowed"],
            },
            "adjacency_bins_below_target": {
                "soft": soft_agg["bins_below_target"],
                "hard": hard_agg["bins_below_target"],
            },
            "adjacency_target_match_fraction": {
                "soft": soft_agg["target_match_fraction"],
                "hard": hard_agg["target_match_fraction"],
            },
        })

    return rows


def evaluate_generation_global_metrics(
    *,
    model,
    logits,
    x_hard,
    global_ctx,
    roi_hw,
    pad_hw,
    gap_bins,
    prob_threshold=None,
    spatial_allowed_threshold=0.20,
    spatial_tolerance_radius=1,
    adjacency_tau=0.25,
    adjacency_margin=0.25,
    adjacency_conf_den_scale=100.0,
):
    gap_bins = normalize_gap_bins(gap_bins)

    if prob_threshold is None:
        prob_threshold = float(model.best_thr_tol.item())
    else:
        prob_threshold = float(prob_threshold)

    if tuple(model.memory_adj.gap_bins) != tuple(gap_bins):
        raise RuntimeError(
            "Stage 0 memory_adj gap bins do not match inference gap bins: "
            f"{model.memory_adj.gap_bins} != {gap_bins}"
        )

    B, _, _, H, W = x_hard.shape
    _, pH, pW = model.patch_size
    out_tok_hw = (H // pH, W // pW)

    results = []

    for b in range(B):
        g = global_ctx[b:b + 1]
        logit_b = logits[b:b + 1]
        hard_b = x_hard[b:b + 1]
        roi_b = None if roi_hw is None else [roi_hw[b]]
        pad_b = None if pad_hw is None else [pad_hw[b]]

        row = {
            "global_spatial_metrics": {
                "spatial_metric_available": False,
                "spatial_metric_defined": False,
                "spatial_status": "error",
                "spatial_status_reason": None,
            },
            "global_adjacency_metrics": {
                "adjacency_metric_available": False,
                "adjacency_metric_defined": {"soft": False, "hard": False},
                "adjacency_status": "error",
                "adjacency_status_reason": None,
            },
        }

        try:
            full_tok = model.memory_tok.get(
                g, device=hard_b.device, dtype=torch.float32
            )
            full_pix = model.memory_pix.get(
                g, device=hard_b.device, dtype=torch.float32
            )

            target_tok = sample_full_token_map_to_crop(
                full_tok_bhw=full_tok,
                out_tok_hw=out_tok_hw,
                patch_size=model.patch_size,
                roi_hw=roi_b,
                pad_hw=pad_b,
            )
            target_pix = sample_full_pixel_map_to_crop(
                full_pix_bhw=full_pix,
                out_hw=(H, W),
                roi_hw=roi_b,
                pad_hw=pad_b,
            )

            allowed_tok = dilate_spatial_support_hw(
                target_tok,
                radius_h=spatial_tolerance_radius,
                radius_w=spatial_tolerance_radius,
            )
            allowed_pix = dilate_spatial_support_hw(
                target_pix,
                radius_h=spatial_tolerance_radius,
                radius_w=spatial_tolerance_radius,
            )

            soft_tok = soft_spatial_token_map_from_logits(
                logit_b,
                model.patch_size,
                tau=0.25,
                prob_threshold=prob_threshold,
            )
            soft_pix = soft_spatial_pixel_map_from_logits(
                logit_b,
                tau=0.25,
                prob_threshold=prob_threshold,
            )
            hard_tok = spatial_token_map_from_input(
                hard_b, model.patch_size
            )
            hard_pix = spatial_pixel_map_from_input(hard_b)

            target_pix_bool = target_pix.ge(spatial_allowed_threshold)
            allowed_pix_bool = allowed_pix.ge(spatial_allowed_threshold)
            hard_pix_bool = hard_pix.bool()

            spike_mask = hard_b.bool()
            forbidden_spike_mask = (
                spike_mask
                & ~allowed_pix_bool[:, None, None, :, :]
            )
            total_spikes = int(spike_mask.sum().item())
            forbidden_spikes = int(forbidden_spike_mask.sum().item())
            allowed_spikes = total_spikes - forbidden_spikes

            spatial = _overlap_metrics(
                hard_pix_bool,
                target_pix_bool,
                allowed_pix_bool,
            )
            spatial.update({
                "spatial_metric_available": True,
                "spatial_metric_defined": True,
                "spatial_status": "ok",
                "spatial_status_reason": None,
                "spatial_target_source": "memory_tok+memory_pix",
                "spatial_allowed_threshold": float(spatial_allowed_threshold),
                "spatial_tolerance_radius": int(spatial_tolerance_radius),
                "spatial_total_generated_spikes": total_spikes,
                "spatial_forbidden_spike_count": forbidden_spikes,
                "spatial_allowed_spike_count": allowed_spikes,
                "spatial_forbidden_spike_fraction": (
                    forbidden_spikes / total_spikes
                    if total_spikes > 0 else None
                ),
                "spatial_allowed_spike_fraction": (
                    allowed_spikes / total_spikes
                    if total_spikes > 0 else None
                ),
                "spatial_soft_violation_loss": float(
                    spatial_support_violation_loss(
                        pred_support=soft_tok,
                        allowed_support=allowed_tok,
                        neg_thresh=spatial_allowed_threshold,
                    ).item()
                ),
                "spatial_hard_violation_fraction": (
                    forbidden_spikes / total_spikes
                    if total_spikes > 0 else None
                ),
                "spatial_support_correlation": _correlation(
                    soft_pix, target_pix
                ),
                "spatial_blank_video": bool(total_spikes == 0),
                "spatial_training_consistent_space": "token_hw",
                "spatial_interpretable_space": "pixel_hw",
            })
            row["global_spatial_metrics"] = spatial

        except KeyError as exc:
            row["global_spatial_metrics"]["spatial_status_reason"] = str(exc)

        try:
            target = model.memory_adj.get(
                g, device=hard_b.device, dtype=torch.float32
            )
            confidence = model.memory_adj.get_confidence(
                g,
                device=hard_b.device,
                dtype=torch.float32,
                den_scale=adjacency_conf_den_scale,
            )
            row["global_adjacency_metrics"] = _adjacency_metrics(
                logits=logit_b,
                x_hard=hard_b,
                target=target,
                confidence=confidence,
                gap_bins=gap_bins,
                tau=adjacency_tau,
                prob_threshold=prob_threshold,
                margin=adjacency_margin,
            )[0]

        except KeyError as exc:
            row["global_adjacency_metrics"][
                "adjacency_status_reason"
            ] = str(exc)

        results.append(row)

    return results