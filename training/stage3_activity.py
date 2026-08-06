#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inference-aligned Stage-3B activity training and Stage-3C calibration."""

import copy
import hashlib
import os
import time
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..model.prior import build_activity_targets_from_codes, detr_activity_loss
from ..inference.decode import (
    decode_codes_to_xgen,
    decode_motif_logits_soft_given_activity,
)
from ..inference.metrics_gen import evaluate_generation_global_metrics
from ..inference.sample_prior import iterative_unmask_motif_given_activity
from ..utils.constants import normalize_gap_bins
from ..utils.losses import (
    ctx_loss_soft,
    local_moment_field_loss,
    short_gap_excess_loss_from_logits_batch_targets,
    spatial_support_violation_loss,
)
from ..utils.recon import (
    compute_activity_ctx,
    dilate_spatial_support_hw,
    sample_full_token_map_to_crop,
    soft_spatial_token_map_from_logits,
)
from .train_prior import (
    _batch_to_device,
    _freeze_module,
    _make_activity_in_from_codes,
    _vq_codes_alpha_and_pmask,
    _vq_codes_and_pmask,
)


def stage3c_trainable_modules(activity_prior) -> Tuple[str, ...]:
    if activity_prior.coordinate_mode == "joint_dense":
        return (
            "count_event_film",
            "event_head",
            "grid_head",
        )
    return (
        "count_event_film",
        "event_head",
        "t_head",
        "h_head",
        "w_head",
    )

HARD_ACTIVITY_COMPOSITE_FORMULA = (
    "0.45*tolerance_f1 + 0.25*exact_f1 + 0.20*count_consistency "
    "+ 0.05*context_consistency + 0.03*adjacency_consistency "
    "+ 0.02*spatial_consistency - 0.05*duplicate_rate"
)

STAGE3C_GENERATION_COMPOSITE_FORMULA = (
    "0.30*activity_tolerance_f1 + 0.15*activity_exact_f1 "
    "+ 0.15*activity_count_consistency + 0.15*decoded_count_consistency "
    "+ 0.10*local_context_consistency + 0.05*local_field_consistency "
    "+ 0.05*short_gap_consistency + 0.05*spatial_consistency "
    "- 0.05*duplicate_rate"
)


def configure_stage3c_event_calibration(activity_prior) -> list[str]:
    """Freeze Stage 3B except the conservative event-placement allowlist."""
    for parameter in activity_prior.parameters():
        parameter.requires_grad_(False)

    missing = []
    for module_name in stage3c_trainable_modules(activity_prior):
        module = getattr(activity_prior, module_name, None)
        if module is None:
            missing.append(module_name)
            continue
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    if missing:
        raise AttributeError(
            "Stage 3C activity prior is missing required modules: "
            + ", ".join(missing)
        )
    return [
        name
        for name, parameter in activity_prior.named_parameters()
        if parameter.requires_grad
    ]


def _validate_optimizer_scope(module, optimizer, expected_names: Sequence[str]) -> None:
    named = dict(module.named_parameters())
    expected_ids = {id(named[name]) for name in expected_names}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != expected_ids:
        unexpected = [
            name
            for name, parameter in named.items()
            if id(parameter) in optimizer_ids and name not in expected_names
        ]
        missing = [
            name for name in expected_names if id(named[name]) not in optimizer_ids
        ]
        raise RuntimeError(
            "Stage 3C optimizer does not match the event-side allowlist. "
            f"Unexpected={unexpected}; missing={missing}."
        )


def _stable_validation_key(batch, index: int) -> str:
    # Deliberately exclude the dataset's random t0/ROI crop so a recording gets
    # the same validation task/mask family on every epoch. Training retains the
    # original random task and mask_spec from the dataset.
    parts = []
    for key in ("path", "assay_name", "assay_idx"):
        value = batch.get(key, None)
        if torch.is_tensor(value) and value.ndim > 0:
            value = value[index].detach().cpu().item()
        elif isinstance(value, np.ndarray) and value.ndim > 0:
            value = value[index].item()
        elif isinstance(value, (list, tuple)) and len(value) > index:
            value = value[index]
        parts.append(str(value))
    return "|".join(parts)


def deterministic_validation_task_and_masks(batch, x, device):
    """Return the same task and mask for the same sampled crop every epoch."""
    batch_size, _, frames, height, width = x.shape
    task_ids = []
    mask_specs = []
    modes = ("recon", "causal", "noncausal", "spatial")

    for index in range(batch_size):
        digest = hashlib.sha256(
            _stable_validation_key(batch, index).encode("utf-8")
        ).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        task_id = seed % len(modes)
        mode = modes[task_id]
        task_ids.append(task_id)

        if mode == "recon" or frames < 2:
            mask_specs.append({"type": "recon"})
        elif mode == "causal":
            unit = ((seed >> 8) % 10000) / 9999.0
            fraction = 0.25 + 0.50 * unit
            prefix = int(np.clip(round(frames * fraction), 1, frames - 1))
            mask_specs.append({"type": "causal", "prefix_frames": prefix})
        elif mode == "noncausal":
            length = max(1, int(round(0.30 * frames)))
            max_start = max(0, frames - length)
            start = int((seed >> 20) % (max_start + 1))
            mask_specs.append({
                "type": "noncausal",
                "time_spans": [(start, start + length)],
            })
        else:
            box_h = max(1, int(round(0.50 * height)))
            box_w = max(1, int(round(0.50 * width)))
            y0 = int((seed >> 32) % (max(0, height - box_h) + 1))
            x0 = int((seed >> 44) % (max(0, width - box_w) + 1))
            mask_specs.append({
                "type": "spatial",
                "spatial_box": (y0, y0 + box_h, x0, x0 + box_w),
            })

    return (
        torch.tensor(task_ids, device=device, dtype=torch.long),
        mask_specs,
    )


def _count_teacher_probability(
    epoch: int,
    teacher_epochs: int,
    transition_epochs: int,
) -> float:
    teacher_epochs = max(0, int(teacher_epochs))
    transition_epochs = max(0, int(transition_epochs))
    if int(epoch) <= teacher_epochs:
        return 1.0
    if transition_epochs == 0 or int(epoch) >= teacher_epochs + transition_epochs:
        return 0.0
    progress = (int(epoch) - teacher_epochs) / float(transition_epochs)
    return float(max(0.0, 1.0 - progress))


def _binary_prf(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    precision = float(tp / max(tp + fp, 1.0))
    recall = float(tp / max(tp + fn, 1.0))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    return precision, recall, f1


def _activity_ctx_torch(activity_bthw: torch.Tensor, eps: float = 1e-8):
    """Compute the repository's nine activity features on token activity."""
    x = activity_bthw.float()
    batch_size, frames, height, width = x.shape
    density = x.mean(dim=(1, 2, 3))
    log_density = torch.log(density + 1e-6).clamp(-15.0, 0.0)
    mass = x.sum(dim=(1, 2, 3))
    safe_mass = mass.clamp_min(eps)

    tt = torch.linspace(-1.0, 1.0, frames, device=x.device, dtype=x.dtype).view(
        1, frames, 1, 1
    )
    yy = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype).view(
        1, 1, height, 1
    )
    xx = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype).view(
        1, 1, 1, width
    )
    mean_x = (x * xx).sum(dim=(1, 2, 3)) / safe_mass
    mean_y = (x * yy).sum(dim=(1, 2, 3)) / safe_mass
    mean_t = (x * tt).sum(dim=(1, 2, 3)) / safe_mass
    dx = xx - mean_x.view(batch_size, 1, 1, 1)
    dy = yy - mean_y.view(batch_size, 1, 1, 1)
    dt = tt - mean_t.view(batch_size, 1, 1, 1)
    var_x = (x * dx.square()).sum(dim=(1, 2, 3)) / safe_mass
    var_y = (x * dy.square()).sum(dim=(1, 2, 3)) / safe_mass
    var_t = (x * dt.square()).sum(dim=(1, 2, 3)) / safe_mass
    cov_xy = (x * dx * dy).sum(dim=(1, 2, 3)) / safe_mass
    cov_xt = (x * dx * dt).sum(dim=(1, 2, 3)) / safe_mass
    cov_yt = (x * dy * dt).sum(dim=(1, 2, 3)) / safe_mass

    active_ratio = x.amax(dim=1).gt(0).float().sum(dim=(1, 2))
    active_ratio = active_ratio / float(max(1, min(height * width, 1024)))
    active_ratio = active_ratio.clamp(0.0, 1.0)

    frame_means = x.mean(dim=(2, 3))
    time_axis = torch.arange(frames, device=x.device, dtype=x.dtype)
    time_axis = (time_axis - time_axis.mean()) / time_axis.std(
        unbiased=False
    ).clamp_min(eps)
    centered = frame_means - frame_means.mean(dim=1, keepdim=True)
    trend = (centered * time_axis.view(1, frames)).mean(dim=1)
    trend = trend / centered.std(dim=1, unbiased=False).clamp_min(eps)

    features = torch.stack((
        log_density,
        var_x,
        var_y,
        var_t,
        cov_xy,
        cov_xt,
        cov_yt,
        active_ratio,
        trend,
    ), dim=1)
    blank = mass.le(eps)
    features[blank, 1:] = 0.0
    return features


def _hard_activity_batch_stats(
    activity_prior,
    output,
    targets,
    *,
    roi_mask,
    tolerance=(1, 1, 1),
    memory_tok=None,
    global_ctx=None,
    roi_hw=None,
    pad_hw=None,
    patch_size=(1, 1, 1),
) -> Dict[str, float]:
    hard = activity_prior.sample_hard_activity_gridtopk(
        output,
        count_mode="expected",
        count_temperature=1.0,
        count_stochastic_round=False,
        roi_mask=roi_mask,
    ).bool()
    target = targets["activity_flat"].bool()
    roi = roi_mask.squeeze(-1) if roi_mask.dim() == 3 else roi_mask
    roi = roi.to(device=hard.device).bool()
    hard &= roi
    target &= roi

    batch_size = hard.shape[0]
    hard_grid = hard.view(
        batch_size,
        activity_prior.Ttok,
        activity_prior.Htok,
        activity_prior.Wtok,
    )
    target_grid = target.view_as(hard_grid)
    hard_5d = hard_grid[:, None].float()
    target_5d = target_grid[:, None].float()

    radius_t, radius_h, radius_w = map(int, tolerance)
    target_dilated = F.max_pool3d(
        target_5d,
        kernel_size=(2 * radius_t + 1, 2 * radius_h + 1, 2 * radius_w + 1),
        stride=1,
        padding=(radius_t, radius_h, radius_w),
    ).bool()
    hard_dilated = F.max_pool3d(
        hard_5d,
        kernel_size=(2 * radius_t + 1, 2 * radius_h + 1, 2 * radius_w + 1),
        stride=1,
        padding=(radius_t, radius_h, radius_w),
    ).bool()

    selected_counts = activity_prior.select_counts(
        output["count_logits"],
        mode="expected",
        temperature=1.0,
        stochastic_round=False,
    )
    selected_counts = torch.minimum(selected_counts, roi.sum(dim=1).long())
    hard_counts = hard.sum(dim=1).float()
    target_counts = target.sum(dim=1).float()
    duplicate_counts = (selected_counts.float() - hard_counts).clamp_min(0.0)

    pred_ctx = _activity_ctx_torch(hard_grid)
    target_ctx = _activity_ctx_torch(target_grid)
    context_abs = (pred_ctx - target_ctx).abs()

    gap_errors = []
    for gap in (1, 2, 3):
        if activity_prior.Ttok <= gap:
            continue
        pred_num = (
            hard_grid[:, :-gap] & hard_grid[:, gap:]
        ).float().sum(dim=(1, 2, 3))
        pred_den = hard_grid[:, :-gap].float().sum(dim=(1, 2, 3)).clamp_min(1.0)
        target_num = (
            target_grid[:, :-gap] & target_grid[:, gap:]
        ).float().sum(dim=(1, 2, 3))
        target_den = target_grid[:, :-gap].float().sum(
            dim=(1, 2, 3)
        ).clamp_min(1.0)
        gap_errors.append((pred_num / pred_den - target_num / target_den).abs())
    if gap_errors:
        gap_error = torch.stack(gap_errors, dim=1).mean(dim=1)
    else:
        gap_error = hard_counts.new_zeros((batch_size,))

    pred_support = hard_grid.any(dim=1)
    allowed = None
    if memory_tok is not None and global_ctx is not None:
        try:
            full_allowed = memory_tok.get(
                global_ctx, device=hard.device, dtype=torch.float32
            )
            allowed = sample_full_token_map_to_crop(
                full_tok_bhw=full_allowed,
                out_tok_hw=(activity_prior.Htok, activity_prior.Wtok),
                patch_size=patch_size,
                roi_hw=roi_hw,
                pad_hw=pad_hw,
            )
            allowed = dilate_spatial_support_hw(
                allowed, radius_h=1, radius_w=1
            ).ge(0.20)
        except (KeyError, RuntimeError, ValueError):
            allowed = None
    if allowed is None:
        allowed = F.max_pool2d(
            target_grid.any(dim=1).float()[:, None],
            kernel_size=3,
            stride=1,
            padding=1,
        )[:, 0].bool()
    violation_count = (pred_support & ~allowed).sum(dim=(1, 2)).float()
    support_count = pred_support.sum(dim=(1, 2)).float()

    return {
        "exact_tp": float((hard & target).sum().item()),
        "exact_fp": float((hard & ~target).sum().item()),
        "exact_fn": float((target & ~hard).sum().item()),
        "tol_tp_pred": float((hard_5d.bool() & target_dilated).sum().item()),
        "tol_fp": float((hard_5d.bool() & ~target_dilated).sum().item()),
        "tol_tp_target": float((target_5d.bool() & hard_dilated).sum().item()),
        "tol_fn": float((target_5d.bool() & ~hard_dilated).sum().item()),
        "hard_count_abs_error_sum": float(
            (hard_counts - target_counts).abs().sum().item()
        ),
        "hard_count_sum": float(hard_counts.sum().item()),
        "target_count_sum": float(target_counts.sum().item()),
        "duplicate_count_sum": float(duplicate_counts.sum().item()),
        "context_abs_sum": float(context_abs.sum().item()),
        "context_elements": float(context_abs.numel()),
        "gap_abs_error_sum": float(gap_error.sum().item()),
        "support_violation_sum": float(violation_count.sum().item()),
        "support_den_sum": float(support_count.sum().item()),
        "samples": float(batch_size),
    }


def _accumulate(total: Dict[str, float], values: Dict[str, float]) -> None:
    for key, value in values.items():
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0.0) + float(value)


def _finalize_hard_activity_stats(total: Dict[str, float]) -> Dict[str, float]:
    exact_p, exact_r, exact_f1 = _binary_prf(
        total.get("exact_tp", 0.0),
        total.get("exact_fp", 0.0),
        total.get("exact_fn", 0.0),
    )
    tolerance_p = total.get("tol_tp_pred", 0.0) / max(
        total.get("tol_tp_pred", 0.0) + total.get("tol_fp", 0.0), 1.0
    )
    tolerance_r = total.get("tol_tp_target", 0.0) / max(
        total.get("tol_tp_target", 0.0) + total.get("tol_fn", 0.0), 1.0
    )
    tolerance_f1 = 2.0 * tolerance_p * tolerance_r / max(
        tolerance_p + tolerance_r, 1e-12
    )
    samples = max(total.get("samples", 0.0), 1.0)
    target_count_mean = total.get("target_count_sum", 0.0) / samples
    count_mae = total.get("hard_count_abs_error_sum", 0.0) / samples
    count_consistency = max(
        0.0, 1.0 - count_mae / max(target_count_mean, 1.0)
    )
    context_mae = total.get("context_abs_sum", 0.0) / max(
        total.get("context_elements", 0.0), 1.0
    )
    gap_mae = total.get("gap_abs_error_sum", 0.0) / samples
    spatial_violation = total.get("support_violation_sum", 0.0) / max(
        total.get("support_den_sum", 0.0), 1.0
    )
    duplicate_rate = total.get("duplicate_count_sum", 0.0) / max(
        total.get("hard_count_sum", 0.0), 1.0
    )
    metric = (
        0.45 * tolerance_f1
        + 0.25 * exact_f1
        + 0.20 * count_consistency
        + 0.05 / (1.0 + context_mae)
        + 0.03 / (1.0 + gap_mae)
        + 0.02 * max(0.0, 1.0 - spatial_violation)
        - 0.05 * duplicate_rate
    )
    return {
        "hard_predicted_count_mean": total.get("hard_count_sum", 0.0) / samples,
        "hard_target_count_mean": target_count_mean,
        "hard_count_mae": count_mae,
        "hard_exact_precision": exact_p,
        "hard_exact_recall": exact_r,
        "hard_exact_f1": exact_f1,
        "hard_tolerance_precision": float(tolerance_p),
        "hard_tolerance_recall": float(tolerance_r),
        "hard_tolerance_f1": float(tolerance_f1),
        "hard_duplicate_token_count": total.get("duplicate_count_sum", 0.0),
        "hard_duplicate_token_rate": duplicate_rate,
        "hard_token_context_mae": context_mae,
        "hard_temporal_gap_mae": gap_mae,
        "hard_spatial_support_violation": spatial_violation,
        "hard_metric": float(metric),
        "hard_metric_formula": HARD_ACTIVITY_COMPOSITE_FORMULA,
    }


def _save_activity_checkpoint(
    path: str,
    activity_prior,
    *,
    epoch: int,
    token_grid,
    selection: str,
    validation_metrics: Dict[str, float],
    hyperparameters: Dict,
) -> None:
    coordinate_metadata = activity_prior.coordinate_metadata()
    torch.save({
        "model": activity_prior.state_dict(),
        "epoch": int(epoch),
        "selection": selection,
        "best_val_loss": float(validation_metrics["loss"]),
        "best_hard_metric": float(validation_metrics.get("hard_metric", float("nan"))),
        "hard_metrics": {
            key: value
            for key, value in validation_metrics.items()
            if key.startswith("hard_")
        },
        "hard_metric_formula": HARD_ACTIVITY_COMPOSITE_FORMULA,
        "token_grid": tuple(map(int, token_grid)),
        "Kmax": int(activity_prior.Kmax),
        **coordinate_metadata,
        "hyperparameters": hyperparameters,
        "deterministic_validation_masks": True,
    }, path)


def train_activity_prior_detr(
    activity_prior,
    vqvae,
    opt,
    train_loader,
    val_loader=None,
    *,
    epochs: int = 20,
    grad_clip: float = 1.0,
    ckpt_out: str = "ckpts/activity_prior_best.pt",
    ckpt_loss_out: Optional[str] = None,
    ckpt_hard_out: Optional[str] = None,
    early_stop_patience: int = 5,
    min_delta: float = 0.0,
    use_amp: bool = True,
    grad_accum_steps: int = 1,
    log_every: int = 50,
    blank_code: Optional[int] = None,
    lambda_count: float = 1.0,
    lambda_count_neighbor: float = 0.25,
    lambda_count_distance: float = 0.05,
    lambda_obj: float = 1.0,
    lambda_coord: float = 1.0,
    lambda_soft_count: float = 0.10,
    lambda_soft_grid: float = 1.0,
    lambda_dup: float = 0.10,
    no_object_weight: float = 0.10,
    count_neighbor_k: int = 11,
    count_neighbor_tau: float = 2.0,
    count_distance_scale: float = 5.0,
    soft_count_beta: float = 5.0,
    count_teacher_epochs: int = 15,
    count_transition_epochs: int = 45,
    hard_tolerance=(1, 1, 1),
    memory_tok=None,
    deterministic_val_masks: bool = True,
):
    """Train Stage 3B and save loss-best plus hard-generation-best states."""
    device = next(activity_prior.parameters()).device
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    _freeze_module(vqvae)

    directory = os.path.dirname(ckpt_out) or "."
    ckpt_loss_out = ckpt_loss_out or os.path.join(
        directory, "activity_prior_best_loss.pt"
    )
    ckpt_hard_out = ckpt_hard_out or os.path.join(
        directory, "activity_prior_best_hard_metric.pt"
    )
    for path in (ckpt_out, ckpt_loss_out, ckpt_hard_out):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if blank_code is None:
        blank_code = getattr(vqvae.vq, "blank_code", -1)
    token_grid = (
        activity_prior.Ttok,
        activity_prior.Htok,
        activity_prior.Wtok,
    )
    hyperparameters = {
        "coordinate_mode": activity_prior.coordinate_mode,
        "token_grid": list(map(int, token_grid)),
        "Ntok": int(activity_prior.Ntok),
        "lambda_count": float(lambda_count),
        "lambda_count_neighbor": float(lambda_count_neighbor),
        "lambda_count_distance": float(lambda_count_distance),
        "lambda_obj": float(lambda_obj),
        "lambda_coord": float(lambda_coord),
        "lambda_soft_count": float(lambda_soft_count),
        "lambda_soft_grid": float(lambda_soft_grid),
        "lambda_dup": float(lambda_dup),
        "no_object_weight": float(no_object_weight),
        "count_teacher_epochs": int(count_teacher_epochs),
        "count_transition_epochs": int(count_transition_epochs),
        "hard_count_mode": "expected",
        "hard_count_temperature": 1.0,
        "hard_count_stochastic_round": False,
        "deterministic_validation_masks": bool(deterministic_val_masks),
    }

    def run_epoch(loader, *, train: bool, teacher_probability: float):
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        activity_prior.train(train)
        totals: Dict[str, float] = {}
        hard_totals: Dict[str, float] = {}
        sample_count = 0.0

        for iteration, batch in enumerate(loader, start=1):
            x, global_ctx, local_ctx, task_id, mask_spec = _batch_to_device(
                batch, device
            )
            if not train and deterministic_val_masks:
                task_id, mask_spec = deterministic_validation_task_and_masks(
                    batch, x, device
                )
            with torch.no_grad():
                codes, predict_mask, _ = _vq_codes_and_pmask(
                    vqvae,
                    x,
                    global_ctx,
                    local_ctx,
                    mask_spec,
                    device,
                )
                targets = build_activity_targets_from_codes(
                    codes=codes,
                    token_grid=token_grid,
                    Kmax=activity_prior.Kmax,
                    blank_code=blank_code,
                    predict_mask=predict_mask,
                )
                activity_input = _make_activity_in_from_codes(
                    codes,
                    predict_mask,
                    blank_code=blank_code,
                    a_mask_id=activity_prior.a_mask_id,
                )

            if train and (iteration - 1) % grad_accum_steps == 0:
                opt.zero_grad(set_to_none=True)
            context = torch.enable_grad() if train else torch.no_grad()
            with context:
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    output = activity_prior(
                        global_ctx=global_ctx,
                        local_ctx=local_ctx,
                        task_id=task_id,
                        a_in=activity_input,
                        roi_mask=predict_mask,
                        count_target=(targets["count_target"] if train else None),
                        count_teacher_prob=(
                            teacher_probability if train else 0.0
                        ),
                    )
                    loss, auxiliary = detr_activity_loss(
                        output,
                        targets,
                        activity_prior,
                        lambda_count=lambda_count,
                        lambda_count_neighbor=lambda_count_neighbor,
                        lambda_count_distance=lambda_count_distance,
                        lambda_obj=lambda_obj,
                        lambda_coord=lambda_coord,
                        lambda_soft_count=lambda_soft_count,
                        lambda_soft_grid=lambda_soft_grid,
                        lambda_dup=lambda_dup,
                        no_object_weight=no_object_weight,
                        count_neighbor_k=count_neighbor_k,
                        count_neighbor_tau=count_neighbor_tau,
                        count_distance_scale=count_distance_scale,
                        soft_count_beta=soft_count_beta,
                    )

            if train:
                scaler.scale(loss / float(grad_accum_steps)).backward()
                should_step = (
                    iteration % grad_accum_steps == 0
                    or iteration == len(loader)
                )
                if should_step:
                    if grad_clip is not None and grad_clip > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(
                            [
                                parameter
                                for parameter in activity_prior.parameters()
                                if parameter.requires_grad
                            ],
                            float(grad_clip),
                        )
                    scaler.step(opt)
                    scaler.update()

            batch_size = x.shape[0]
            sample_count += float(batch_size)
            for key, value in auxiliary.items():
                if torch.is_tensor(value):
                    value = float(value.detach().item())
                totals[key] = totals.get(key, 0.0) + float(value) * batch_size

            if not train:
                _accumulate(
                    hard_totals,
                    _hard_activity_batch_stats(
                        activity_prior,
                        output,
                        targets,
                        roi_mask=predict_mask,
                        tolerance=hard_tolerance,
                        memory_tok=memory_tok,
                        global_ctx=global_ctx,
                        roi_hw=batch.get("roi_hw", None),
                        pad_hw=batch.get("pad_hw", None),
                        patch_size=vqvae.patch_size,
                    ),
                )

            if train and log_every and iteration % log_every == 0:
                denominator = max(sample_count, 1.0)
                print(
                    f"  [3B Activity] it {iteration:05d}: "
                    f"loss={totals['loss'] / denominator:.4f} "
                    f"count={totals['loss_count'] / denominator:.4f} "
                    f"obj={totals['loss_obj'] / denominator:.4f} "
                    f"coord={totals['loss_coord'] / denominator:.4f} "
                    f"grid={totals['loss_soft_grid'] / denominator:.4f} "
                    f"dup={totals['loss_dup'] / denominator:.4f} "
                    f"countE={totals['count_expected_mean'] / denominator:.2f} "
                    f"targetK={totals['target_count_mean'] / denominator:.2f} "
                    f"teacherP={teacher_probability:.3f}"
                )

        denominator = max(sample_count, 1.0)
        result = {key: value / denominator for key, value in totals.items()}
        result["epoch_duration_seconds"] = float(time.perf_counter() - epoch_start)
        result["peak_gpu_memory_mb"] = (
            float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            if device.type == "cuda"
            else 0.0
        )
        result["count_teacher_probability"] = float(
            teacher_probability if train else 0.0
        )
        if not train:
            result.update(_finalize_hard_activity_stats(hard_totals))
        return result

    history = {
        "train": [],
        "val": [],
        "hyperparameters": hyperparameters,
        "hard_metric_formula": HARD_ACTIVITY_COMPOSITE_FORMULA,
    }
    best_loss = float("inf")
    best_hard_metric = -float("inf")
    patience = 0

    for epoch in range(1, int(epochs) + 1):
        teacher_probability = _count_teacher_probability(
            epoch,
            count_teacher_epochs,
            count_transition_epochs,
        )
        train_metrics = run_epoch(
            train_loader,
            train=True,
            teacher_probability=teacher_probability,
        )
        validation_metrics = run_epoch(
            val_loader if val_loader is not None else train_loader,
            train=False,
            teacher_probability=0.0,
        )
        history["train"].append(train_metrics)
        history["val"].append(validation_metrics)

        if activity_prior.coordinate_mode == "factorized":
            coordinate_diagnostics = (
                f"axisAcc=({validation_metrics['matched_t_acc']:.4f},"
                f"{validation_metrics['matched_h_acc']:.4f},"
                f"{validation_metrics['matched_w_acc']:.4f}) "
            )
        else:
            coordinate_diagnostics = ""

        print(
            f"[activity epoch {epoch:03d}] "
            f"train={train_metrics['loss']:.4f} "
            f"val={validation_metrics['loss']:.4f} "
            f"hardK={validation_metrics['hard_predicted_count_mean']:.2f}/"
            f"{validation_metrics['hard_target_count_mean']:.2f} "
            f"hardMAE={validation_metrics['hard_count_mae']:.2f} "
            f"exactF1={validation_metrics['hard_exact_f1']:.4f} "
            f"tolF1={validation_metrics['hard_tolerance_f1']:.4f} "
            f"flatTop1={validation_metrics['matched_flat_top1_acc']:.4f} "
            f"flatTop5={validation_metrics['matched_flat_top5_acc']:.4f} "
            f"gridH={validation_metrics['joint_grid_entropy']:.3f} "
            f"gridPmax={validation_metrics['joint_grid_max_probability']:.4f} "
            f"{coordinate_diagnostics}"
            f"ctxMAE={validation_metrics['hard_token_context_mae']:.4f} "
            f"gapMAE={validation_metrics['hard_temporal_gap_mae']:.4f} "
            f"supportV={validation_metrics['hard_spatial_support_violation']:.4f} "
            f"duplicates={validation_metrics['hard_duplicate_token_count']:.1f} "
            f"hardMetric={validation_metrics['hard_metric']:.6f} "
            f"peakMB={validation_metrics['peak_gpu_memory_mb']:.1f} "
            f"seconds={validation_metrics['epoch_duration_seconds']:.1f} "
            f"teacherP={teacher_probability:.3f}"
        )

        improved = False
        if validation_metrics["loss"] < best_loss - float(min_delta):
            best_loss = validation_metrics["loss"]
            _save_activity_checkpoint(
                ckpt_loss_out,
                activity_prior,
                epoch=epoch,
                token_grid=token_grid,
                selection="validation_loss",
                validation_metrics=validation_metrics,
                hyperparameters=hyperparameters,
            )
            print(f"  saved {ckpt_loss_out}  best_loss={best_loss:.6f}")
            improved = True

        if validation_metrics["hard_metric"] > best_hard_metric + float(min_delta):
            best_hard_metric = validation_metrics["hard_metric"]
            _save_activity_checkpoint(
                ckpt_hard_out,
                activity_prior,
                epoch=epoch,
                token_grid=token_grid,
                selection="generation_aligned_hard_metric",
                validation_metrics=validation_metrics,
                hyperparameters=hyperparameters,
            )
            if os.path.abspath(ckpt_out) != os.path.abspath(ckpt_hard_out):
                _save_activity_checkpoint(
                    ckpt_out,
                    activity_prior,
                    epoch=epoch,
                    token_grid=token_grid,
                    selection="generation_aligned_hard_metric",
                    validation_metrics=validation_metrics,
                    hyperparameters=hyperparameters,
                )
            print(
                f"  saved {ckpt_hard_out}  "
                f"best_hard_metric={best_hard_metric:.6f}"
            )
            improved = True

        patience = 0 if improved else patience + 1
        if patience >= int(early_stop_patience):
            print(
                f"Early stopping Stage 3B at epoch {epoch}; "
                f"best_hard_metric={best_hard_metric:.6f}."
            )
            break

    return history


def _build_inference_aligned_motif_inputs(
    motif_prior,
    motif_targets: Dict[str, torch.Tensor],
    hard_roi: torch.Tensor,
):
    """Construct the exact motif state used by partial MaskGIT inference."""
    roi = motif_targets["predict_mask"].bool()
    gt_active = motif_targets["active"].bool()
    hard_roi = hard_roi.to(device=roi.device).bool() & roi

    pred_active = roi & hard_roi
    visible_active = (~roi) & gt_active
    batch_size, token_count = roi.shape

    z1_in = torch.full(
        (batch_size, token_count),
        int(motif_prior.z1_null_id),
        device=roi.device,
        dtype=torch.long,
    )
    z2_in = torch.full(
        (batch_size, token_count),
        int(motif_prior.z2_null_id),
        device=roi.device,
        dtype=torch.long,
    )
    alpha_in = torch.zeros(
        (batch_size, token_count, int(motif_prior.K2)),
        device=roi.device,
        dtype=motif_targets["alpha"].dtype,
    )

    z1_in[visible_active] = motif_targets["z1"][visible_active].long()
    z2_in[visible_active] = motif_targets["z2"][visible_active].long()
    alpha_in[visible_active] = motif_targets["alpha"][visible_active]

    z1_in[pred_active] = int(motif_prior.z1_mask_id)
    z2_in[pred_active] = int(motif_prior.z2_mask_id)
    alpha_in[pred_active] = 0.0

    decode_targets = dict(motif_targets)
    decode_targets["decode_motif_mask"] = pred_active
    decode_targets["visible_motif_mask"] = visible_active
    decode_targets["z1_loss_mask"] = roi & gt_active
    decode_targets["z2_loss_mask"] = roi & gt_active
    decode_targets["alpha_loss_mask"] = roi & gt_active
    decode_targets["z_loss_mask"] = roi & gt_active

    return {
        "z1_in": z1_in,
        "z2_in": z2_in,
        "alpha_in": alpha_in,
        "pred_active": pred_active,
        "visible_active": visible_active,
        "targets": decode_targets,
    }


def _hard_gap_rates(x_b1thw: torch.Tensor, gap_bins) -> torch.Tensor:
    x = x_b1thw.bool()
    if x.dim() != 5 or x.shape[1] != 1:
        raise ValueError(f"Expected (B,1,T,H,W), got {tuple(x.shape)}")
    rates = []
    for lo, hi in normalize_gap_bins(gap_bins):
        numerator = x.new_zeros((x.shape[0],), dtype=torch.float32)
        denominator = x.new_zeros((x.shape[0],), dtype=torch.float32)
        for gap in range(int(lo), int(hi) + 1):
            if x.shape[2] <= gap:
                continue
            x0 = x[:, :, :-gap]
            xg = x[:, :, gap:]
            numerator += (x0 & xg).float().sum(dim=(1, 2, 3, 4))
            denominator += x0.float().sum(dim=(1, 2, 3, 4))
        rates.append(numerator / denominator.clamp_min(1.0))
    if not rates:
        return torch.zeros((x.shape[0], 0), device=x.device)
    return torch.stack(rates, dim=1)


def _safe_generation_global_rows(
    vqvae,
    *,
    logits,
    x_hard,
    global_ctx,
    roi_hw,
    pad_hw,
    gap_bins,
):
    required = ("memory_tok", "memory_pix", "memory_adj")
    if any(getattr(vqvae, name, None) is None for name in required):
        return []
    try:
        return evaluate_generation_global_metrics(
            model=vqvae,
            logits=logits,
            x_hard=x_hard,
            global_ctx=global_ctx,
            roi_hw=roi_hw,
            pad_hw=pad_hw,
            gap_bins=gap_bins,
            prob_threshold=float(vqvae.best_thr_tol.item()),
        )
    except (KeyError, RuntimeError, ValueError) as exc:
        print(f"  [3C generation metrics] global-memory metrics unavailable: {exc}")
        return []


@torch.no_grad()
def evaluate_true_stage3_generation(
    activity_prior,
    motif_prior,
    vqvae,
    loader,
    *,
    blank_code: int,
    max_batches: int = 4,
    motif_steps: int = 12,
    motif_temperature: float = 1.0,
    motif_alpha_temperature: float = 1.0,
    motif_z1_top_k: int = 5,
    hard_tolerance=(1, 1, 1),
    gap_bins=((1, 1), (2, 2), (3, 3)),
    deterministic_masks: bool = True,
    seed: int = 314159,
) -> Dict[str, float]:
    """Evaluate the exact hard activity -> MaskGIT -> VQ-VAE path."""
    device = next(activity_prior.parameters()).device
    activity_prior.eval()
    motif_prior.eval()
    vqvae.eval()

    hard_totals: Dict[str, float] = {}
    decoded_totals: Dict[str, float] = {}
    processed_batches = 0

    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else 0]

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

        for batch_index, batch in enumerate(loader):
            if batch_index >= int(max_batches):
                break
            x, global_ctx, local_ctx, task_id, mask_spec = _batch_to_device(
                batch, device
            )
            if deterministic_masks:
                task_id, mask_spec = deterministic_validation_task_and_masks(
                    batch, x, device
                )

            codes, alpha_target, predict_mask, grid = _vq_codes_alpha_and_pmask(
                vqvae,
                x,
                global_ctx,
                local_ctx,
                mask_spec,
                device,
            )
            if grid is None:
                grid = (
                    activity_prior.Ttok,
                    activity_prior.Htok,
                    activity_prior.Wtok,
                )
            token_grid = tuple(map(int, grid))
            targets = build_activity_targets_from_codes(
                codes=codes,
                token_grid=token_grid,
                Kmax=activity_prior.Kmax,
                blank_code=blank_code,
                predict_mask=predict_mask,
            )
            activity_input = _make_activity_in_from_codes(
                codes,
                predict_mask,
                blank_code=blank_code,
                a_mask_id=activity_prior.a_mask_id,
            )
            activity_output = activity_prior(
                global_ctx=global_ctx,
                local_ctx=local_ctx,
                task_id=task_id,
                a_in=activity_input,
                roi_mask=predict_mask,
                count_target=None,
                count_teacher_prob=0.0,
            )
            hard_roi = activity_prior.sample_hard_activity_gridtopk(
                activity_output,
                count_mode="expected",
                count_temperature=1.0,
                count_stochastic_round=False,
                roi_mask=predict_mask,
            ).long()
            roi = predict_mask.squeeze(-1) if predict_mask.dim() == 3 else predict_mask
            roi = roi.bool()
            visible_activity = codes[..., 0].ne(blank_code).long()
            full_activity = torch.where(roi, hard_roi, visible_activity)

            motif_sample = iterative_unmask_motif_given_activity(
                motif_prior,
                activity=full_activity,
                global_ctx=global_ctx,
                local_ctx=local_ctx,
                task_id=task_id,
                roi_mask=roi,
                visible_codes=codes,
                visible_alpha=alpha_target,
                steps=int(motif_steps),
                temperature=float(motif_temperature),
                alpha_temperature=float(motif_alpha_temperature),
                z1_top_k=int(motif_z1_top_k),
            )
            generated = decode_codes_to_xgen(
                vqvae,
                motif_sample["codes"],
                alpha=motif_sample["alpha"],
                grid=token_grid,
                global_ctx=global_ctx,
                local_ctx=local_ctx,
                roi_hw=batch.get("roi_hw", None),
                pad_hw=batch.get("pad_hw", None),
            )

            _accumulate(
                hard_totals,
                _hard_activity_batch_stats(
                    activity_prior,
                    activity_output,
                    targets,
                    roi_mask=predict_mask,
                    tolerance=hard_tolerance,
                    memory_tok=getattr(vqvae, "memory_tok", None),
                    global_ctx=global_ctx,
                    roi_hw=batch.get("roi_hw", None),
                    pad_hw=batch.get("pad_hw", None),
                    patch_size=vqvae.patch_size,
                ),
            )

            x_generated = generated["x_gen"]
            logits_generated = generated["logits"]
            _, _, decoded_t, decoded_h, decoded_w = x_generated.shape
            x_target = x[:, :1, :decoded_t, :decoded_h, :decoded_w]
            generated_count = x_generated.sum(dim=(1, 2, 3, 4)).float()
            target_count = x_target.sum(dim=(1, 2, 3, 4)).float()
            count_abs = (generated_count - target_count).abs()
            count_rel = count_abs / target_count.clamp_min(1.0)
            density_abs = (
                x_generated.float().mean(dim=(1, 2, 3, 4))
                - x_target.float().mean(dim=(1, 2, 3, 4))
            ).abs()

            generated_ctx = _activity_ctx_torch(x_generated[:, 0])
            intended_ctx = local_ctx[:, :9].to(generated_ctx)
            ctx_abs = (generated_ctx - intended_ctx).abs()
            field_error = local_moment_field_loss(
                logits_b1thw=logits_generated,
                target_b1thw=x_target,
                patch_size=vqvae.patch_size,
                tau=0.25,
                min_active_spikes=1,
                min_shape_spikes=5,
                min_trend_spikes=6,
                min_trend_frames=3,
                prob_threshold=float(vqvae.best_thr_tol.item()),
            )
            generated_gap = _hard_gap_rates(x_generated, gap_bins)
            target_gap = _hard_gap_rates(x_target, gap_bins)
            short_gap_mae = (
                (generated_gap - target_gap).abs().mean(dim=1)
                if generated_gap.numel() > 0
                else generated_count.new_zeros(generated_count.shape)
            )

            global_rows = _safe_generation_global_rows(
                vqvae,
                logits=logits_generated,
                x_hard=x_generated,
                global_ctx=global_ctx,
                roi_hw=batch.get("roi_hw", None),
                pad_hw=batch.get("pad_hw", None),
                gap_bins=gap_bins,
            )
            spatial_values = []
            global_adjacency_values = []
            for row in global_rows:
                spatial = row.get("global_spatial_metrics", {})
                adjacency = row.get("global_adjacency_metrics", {})
                value = spatial.get("spatial_hard_violation_fraction", None)
                if value is not None:
                    spatial_values.append(float(value))
                value = adjacency.get("adjacency_hard_weighted_mae", None)
                if value is not None:
                    global_adjacency_values.append(float(value))

            batch_size = x.shape[0]
            decoded_totals["samples"] = decoded_totals.get("samples", 0.0) + batch_size
            decoded_totals["generated_spike_count_sum"] = (
                decoded_totals.get("generated_spike_count_sum", 0.0)
                + float(generated_count.sum().item())
            )
            decoded_totals["target_spike_count_sum"] = (
                decoded_totals.get("target_spike_count_sum", 0.0)
                + float(target_count.sum().item())
            )
            decoded_totals["decoded_spike_count_abs_error_sum"] = (
                decoded_totals.get("decoded_spike_count_abs_error_sum", 0.0)
                + float(count_abs.sum().item())
            )
            decoded_totals["decoded_spike_count_relative_error_sum"] = (
                decoded_totals.get("decoded_spike_count_relative_error_sum", 0.0)
                + float(count_rel.sum().item())
            )
            decoded_totals["decoded_density_abs_error_sum"] = (
                decoded_totals.get("decoded_density_abs_error_sum", 0.0)
                + float(density_abs.sum().item())
            )
            decoded_totals["local_context_abs_sum"] = (
                decoded_totals.get("local_context_abs_sum", 0.0)
                + float(ctx_abs.sum().item())
            )
            decoded_totals["local_context_elements"] = (
                decoded_totals.get("local_context_elements", 0.0)
                + float(ctx_abs.numel())
            )
            decoded_totals["local_field_error_sum"] = (
                decoded_totals.get("local_field_error_sum", 0.0)
                + float(field_error.item()) * batch_size
            )
            decoded_totals["short_gap_error_sum"] = (
                decoded_totals.get("short_gap_error_sum", 0.0)
                + float(short_gap_mae.sum().item())
            )
            decoded_totals["spatial_violation_sum"] = (
                decoded_totals.get("spatial_violation_sum", 0.0)
                + sum(spatial_values)
            )
            decoded_totals["spatial_violation_samples"] = (
                decoded_totals.get("spatial_violation_samples", 0.0)
                + len(spatial_values)
            )
            decoded_totals["global_adjacency_error_sum"] = (
                decoded_totals.get("global_adjacency_error_sum", 0.0)
                + sum(global_adjacency_values)
            )
            decoded_totals["global_adjacency_samples"] = (
                decoded_totals.get("global_adjacency_samples", 0.0)
                + len(global_adjacency_values)
            )
            processed_batches += 1

    hard_metrics = _finalize_hard_activity_stats(hard_totals)
    samples = max(decoded_totals.get("samples", 0.0), 1.0)
    generated_mean = decoded_totals.get("generated_spike_count_sum", 0.0) / samples
    target_mean = decoded_totals.get("target_spike_count_sum", 0.0) / samples
    decoded_count_mae = (
        decoded_totals.get("decoded_spike_count_abs_error_sum", 0.0) / samples
    )
    decoded_relative = (
        decoded_totals.get("decoded_spike_count_relative_error_sum", 0.0) / samples
    )
    local_context_mae = decoded_totals.get("local_context_abs_sum", 0.0) / max(
        decoded_totals.get("local_context_elements", 0.0), 1.0
    )
    local_field_error = decoded_totals.get("local_field_error_sum", 0.0) / samples
    short_gap_error = decoded_totals.get("short_gap_error_sum", 0.0) / samples
    spatial_sample_count = decoded_totals.get("spatial_violation_samples", 0.0)
    if spatial_sample_count > 0:
        spatial_violation = (
            decoded_totals.get("spatial_violation_sum", 0.0)
            / spatial_sample_count
        )
        spatial_source = "decoded_global_memory"
    else:
        spatial_violation = hard_metrics["hard_spatial_support_violation"]
        spatial_source = "hard_activity_target_fallback"
    adjacency_samples = max(decoded_totals.get("global_adjacency_samples", 0.0), 1.0)
    global_adjacency_error = (
        decoded_totals.get("global_adjacency_error_sum", 0.0) / adjacency_samples
    )

    decoded_count_consistency = max(0.0, 1.0 - decoded_relative)
    generation_metric = (
        0.30 * hard_metrics["hard_tolerance_f1"]
        + 0.15 * hard_metrics["hard_exact_f1"]
        + 0.15 * max(
            0.0,
            1.0
            - hard_metrics["hard_count_mae"]
            / max(hard_metrics["hard_target_count_mean"], 1.0),
        )
        + 0.15 * decoded_count_consistency
        + 0.10 / (1.0 + local_context_mae)
        + 0.05 / (1.0 + local_field_error)
        + 0.05 / (1.0 + short_gap_error)
        + 0.05 * max(0.0, 1.0 - spatial_violation)
        - 0.05 * hard_metrics["hard_duplicate_token_rate"]
    )

    result = dict(hard_metrics)
    result.update({
        "generation_batches": int(processed_batches),
        "generated_spike_count_mean": generated_mean,
        "target_spike_count_mean": target_mean,
        "decoded_spike_count_mae": decoded_count_mae,
        "decoded_spike_count_relative_error": decoded_relative,
        "decoded_density_mae": (
            decoded_totals.get("decoded_density_abs_error_sum", 0.0) / samples
        ),
        "generated_local_context_mae": local_context_mae,
        "local_moment_field_error": local_field_error,
        "decoded_short_gap_mae": short_gap_error,
        "global_adjacency_mae": global_adjacency_error,
        "decoded_spatial_support_violation": spatial_violation,
        "spatial_support_metric_source": spatial_source,
        "generation_metric": float(generation_metric),
        "generation_metric_formula": STAGE3C_GENERATION_COMPOSITE_FORMULA,
        "deterministic_validation_masks": bool(deterministic_masks),
        "hard_activity_mode": {
            "count_mode": "expected",
            "count_temperature": 1.0,
            "count_stochastic_round": False,
        },
    })
    return result


def _stage3c_accepts_candidate(
    candidate: Dict[str, float],
    baseline: Dict[str, float],
    best_score: float,
    *,
    count_relative_tolerance: float = 0.10,
    count_absolute_tolerance: float = 1.0,
    activity_f1_tolerance: float = 0.02,
    decoded_relative_tolerance: float = 0.05,
) -> Tuple[bool, list[str]]:
    reasons = []
    if candidate["generation_metric"] <= float(best_score):
        reasons.append("generation composite did not improve")
    count_limit = (
        baseline["hard_count_mae"] * (1.0 + float(count_relative_tolerance))
        + float(count_absolute_tolerance)
    )
    if candidate["hard_count_mae"] > count_limit:
        reasons.append(
            f"hard count MAE {candidate['hard_count_mae']:.4f} exceeded {count_limit:.4f}"
        )
    if candidate["hard_tolerance_f1"] < (
        baseline["hard_tolerance_f1"] - float(activity_f1_tolerance)
    ):
        reasons.append("activity tolerance F1 materially worsened")
    if candidate["decoded_spike_count_relative_error"] > (
        baseline["decoded_spike_count_relative_error"]
        + float(decoded_relative_tolerance)
    ):
        reasons.append("decoded spike-count consistency materially worsened")
    return len(reasons) == 0, reasons


def train_activity_prior_with_frozen_motif(
    activity_prior,
    motif_prior,
    vqvae,
    opt,
    train_loader,
    val_loader=None,
    *,
    epochs: int = 20,
    grad_clip: float = 1.0,
    ckpt_out: str = "ckpts/activity_prior_refined_best.pt",
    early_stop_patience: int = 5,
    min_delta: float = 0.0,
    use_amp: bool = True,
    grad_accum_steps: int = 1,
    log_every: int = 50,
    blank_code: Optional[int] = None,
    freeze_motif: bool = True,
    lambda_detr: float = 1.0,
    lambda_count: float = 1.0,
    lambda_count_neighbor: float = 0.25,
    lambda_count_distance: float = 0.05,
    lambda_obj: float = 1.0,
    lambda_coord: float = 1.0,
    lambda_soft_count: float = 0.10,
    lambda_soft_grid: float = 1.0,
    lambda_dup: float = 0.10,
    no_object_weight: float = 0.10,
    count_neighbor_k: int = 11,
    count_neighbor_tau: float = 2.0,
    count_distance_scale: float = 5.0,
    soft_count_beta: float = 5.0,
    lambda_ctx: float = 0.25,
    lambda_ctx_field: float = 0.05,
    lambda_adj: float = 0.25,
    lambda_spatial: float = 0.25,
    auxiliary_ramp_epochs: int = 10,
    ctx_tau: float = 0.25,
    ctx_field_tau: float = 0.25,
    motif_tau_z: float = 0.25,
    isi_tau: float = 0.25,
    isi_margin: float = 0.25,
    isi_lower_margin: float = 0.20,
    isi_lower_weight: float = 0.50,
    isi_max_gap: int = 3,
    isi_gap_bins=None,
    memory_adj=None,
    memory_tok=None,
    memory_adj_conf_den_scale: float = 100.0,
    generation_val_max_batches: int = 4,
    generation_motif_steps: int = 12,
    generation_seed: int = 314159,
    hard_tolerance=(1, 1, 1),
    deterministic_val_masks: bool = True,
):
    """Inference-aligned Stage 3C event-placement calibration."""
    if not freeze_motif:
        raise ValueError(
            "Stage 3C requires the complete Stage 3A motif prior to remain frozen."
        )
    if val_loader is None:
        raise ValueError(
            "Stage 3C checkpoint gating requires a validation loader."
        )

    device = next(activity_prior.parameters()).device
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    _freeze_module(vqvae)
    _freeze_module(motif_prior)

    trainable_names = configure_stage3c_event_calibration(activity_prior)
    _validate_optimizer_scope(activity_prior, opt, trainable_names)
    trainable_parameters = [
        parameter
        for name, parameter in activity_prior.named_parameters()
        if name in trainable_names
    ]

    if blank_code is None:
        blank_code = int(getattr(vqvae.vq, "blank_code", -1))
    token_grid = (
        activity_prior.Ttok,
        activity_prior.Htok,
        activity_prior.Wtok,
    )
    gap_bins = normalize_gap_bins(
        isi_gap_bins
        if isi_gap_bins is not None
        else tuple((gap, gap) for gap in range(1, int(isi_max_gap) + 1))
    )
    os.makedirs(os.path.dirname(ckpt_out) or ".", exist_ok=True)

    hyperparameters = {
        "coordinate_mode": activity_prior.coordinate_mode,
        "token_grid": list(map(int, token_grid)),
        "Ntok": int(activity_prior.Ntok),
        "lambda_detr": float(lambda_detr),
        "lambda_count": float(lambda_count),
        "lambda_count_neighbor": float(lambda_count_neighbor),
        "lambda_count_distance": float(lambda_count_distance),
        "lambda_obj": float(lambda_obj),
        "lambda_coord": float(lambda_coord),
        "lambda_soft_count": float(lambda_soft_count),
        "lambda_soft_grid": float(lambda_soft_grid),
        "lambda_dup": float(lambda_dup),
        "no_object_weight": float(no_object_weight),
        "lambda_ctx": float(lambda_ctx),
        "lambda_ctx_field": float(lambda_ctx_field),
        "lambda_adj": float(lambda_adj),
        "lambda_spatial": float(lambda_spatial),
        "auxiliary_ramp_epochs": int(auxiliary_ramp_epochs),
        "generation_val_max_batches": int(generation_val_max_batches),
        "generation_motif_steps": int(generation_motif_steps),
        "hard_activity_mode": "expected-count unique grid top-K with straight-through soft occupancy gradients",
        "deterministic_validation_masks": bool(deterministic_val_masks),
        "trainable_parameter_names": trainable_names,
    }

    baseline_state = {
        key: value.detach().cpu().clone()
        for key, value in activity_prior.state_dict().items()
    }
    baseline_metrics = evaluate_true_stage3_generation(
        activity_prior,
        motif_prior,
        vqvae,
        val_loader,
        blank_code=blank_code,
        max_batches=generation_val_max_batches,
        motif_steps=generation_motif_steps,
        hard_tolerance=hard_tolerance,
        gap_bins=gap_bins,
        deterministic_masks=deterministic_val_masks,
        seed=generation_seed,
    )
    print(
        "[3C baseline hard generation] "
        f"score={baseline_metrics['generation_metric']:.6f} "
        f"activityK={baseline_metrics['hard_predicted_count_mean']:.2f}/"
        f"{baseline_metrics['hard_target_count_mean']:.2f} "
        f"tolF1={baseline_metrics['hard_tolerance_f1']:.4f} "
        f"decodedRelErr={baseline_metrics['decoded_spike_count_relative_error']:.4f}"
    )

    auxiliary_gradient_audited = False

    def _auxiliary_scale(epoch: int) -> float:
        ramp_epochs = max(1, int(auxiliary_ramp_epochs))
        if ramp_epochs == 1:
            return 1.0
        return float(np.clip((int(epoch) - 1) / float(ramp_epochs - 1), 0.0, 1.0))

    trainable_module_names = stage3c_trainable_modules(activity_prior)

    def _set_activity_mode(train: bool) -> None:
        activity_prior.eval()
        for module_name in trainable_module_names:
            getattr(activity_prior, module_name).train(train)

    def _run_epoch(loader, *, train: bool, epoch: int):
        nonlocal auxiliary_gradient_audited
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _set_activity_mode(train)
        motif_prior.eval()
        vqvae.eval()
        totals: Dict[str, float] = {}
        hard_totals: Dict[str, float] = {}
        total_samples = 0.0
        aux_scale = _auxiliary_scale(epoch)

        for iteration, batch in enumerate(loader, start=1):
            x, global_ctx, local_ctx, task_id, mask_spec = _batch_to_device(
                batch, device
            )
            if not train and deterministic_val_masks:
                task_id, mask_spec = deterministic_validation_task_and_masks(
                    batch, x, device
                )

            with torch.no_grad():
                codes, alpha_target, predict_mask, grid = _vq_codes_alpha_and_pmask(
                    vqvae,
                    x,
                    global_ctx,
                    local_ctx,
                    mask_spec,
                    device,
                )
                activity_targets = build_activity_targets_from_codes(
                    codes=codes,
                    token_grid=token_grid,
                    Kmax=activity_prior.Kmax,
                    blank_code=blank_code,
                    predict_mask=predict_mask,
                )
                motif_targets = motif_prior.make_targets_from_codes(
                    codes=codes,
                    predict_mask=predict_mask,
                    blank_code=blank_code,
                    alpha=alpha_target,
                )
                activity_input = _make_activity_in_from_codes(
                    codes,
                    predict_mask,
                    blank_code=blank_code,
                    a_mask_id=activity_prior.a_mask_id,
                )

            if train and (iteration - 1) % grad_accum_steps == 0:
                opt.zero_grad(set_to_none=True)
            context = torch.enable_grad() if train else torch.no_grad()
            with context:
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    activity_output = activity_prior(
                        global_ctx=global_ctx,
                        local_ctx=local_ctx,
                        task_id=task_id,
                        a_in=activity_input,
                        roi_mask=predict_mask,
                        count_target=None,
                        count_teacher_prob=0.0,
                    )
                    loss_detr, auxiliary_detr = detr_activity_loss(
                        activity_output,
                        activity_targets,
                        activity_prior,
                        lambda_count=lambda_count,
                        lambda_count_neighbor=lambda_count_neighbor,
                        lambda_count_distance=lambda_count_distance,
                        lambda_obj=lambda_obj,
                        lambda_coord=lambda_coord,
                        lambda_soft_count=lambda_soft_count,
                        lambda_soft_grid=lambda_soft_grid,
                        lambda_dup=lambda_dup,
                        no_object_weight=no_object_weight,
                        count_neighbor_k=count_neighbor_k,
                        count_neighbor_tau=count_neighbor_tau,
                        count_distance_scale=count_distance_scale,
                        soft_count_beta=soft_count_beta,
                    )

                    soft_roi = activity_prior.soft_activity_flat(
                        activity_output,
                        roi_mask=predict_mask,
                    ).clamp(0.0, 1.0)
                    with torch.no_grad():
                        hard_roi = activity_prior.sample_hard_activity_gridtopk(
                            activity_output,
                            count_mode="expected",
                            count_temperature=1.0,
                            count_stochastic_round=False,
                            roi_mask=predict_mask,
                        ).float()
                    roi = (
                        predict_mask.squeeze(-1)
                        if predict_mask.dim() == 3
                        else predict_mask
                    ).bool()
                    soft_roi = soft_roi * roi.to(soft_roi.dtype)
                    hard_roi = hard_roi * roi.to(hard_roi.dtype)
                    activity_roi_st = hard_roi + soft_roi - soft_roi.detach()
                    if not torch.equal(
                        activity_roi_st.detach().bool(), hard_roi.detach().bool()
                    ):
                        raise RuntimeError(
                            "Stage 3C straight-through activity lost its hard binary forward values."
                        )
                    gt_activity = motif_targets["active"].to(
                        device=device, dtype=activity_roi_st.dtype
                    )
                    activity_full_st = torch.where(roi, activity_roi_st, gt_activity)

                    motif_inputs = _build_inference_aligned_motif_inputs(
                        motif_prior,
                        motif_targets,
                        hard_roi,
                    )
                    motif_logits = motif_prior.forward_with_activity_prob(
                        z1_in=motif_inputs["z1_in"],
                        z2_in=motif_inputs["z2_in"],
                        activity_prob=activity_full_st,
                        global_ctx=global_ctx,
                        local_ctx=local_ctx,
                        task_id=task_id,
                        roi_mask=predict_mask,
                        alpha_in=motif_inputs["alpha_in"],
                    )
                    decoded = decode_motif_logits_soft_given_activity(
                        model=vqvae,
                        logits=motif_logits,
                        targets=motif_inputs["targets"],
                        activity_prob=activity_full_st,
                        grid=grid,
                        global_ctx=global_ctx,
                        local_ctx=local_ctx,
                        tau_z=motif_tau_z,
                        roi_hw=batch.get("roi_hw", None),
                        pad_hw=batch.get("pad_hw", None),
                    )
                    logits_volume = decoded["logits_vol"]
                    zero = loss_detr.new_zeros(())
                    loss_ctx = zero
                    loss_ctx_field = zero
                    loss_adj = zero
                    loss_spatial = zero

                    if lambda_ctx > 0.0 and local_ctx is not None:
                        loss_ctx = ctx_loss_soft(
                            logits_b1thw=logits_volume,
                            ctx_tgt_b9=local_ctx,
                            dims=tuple(range(9)),
                            tau=ctx_tau,
                            prob_threshold=float(vqvae.best_thr_tol.item()),
                        )
                    if lambda_ctx_field > 0.0:
                        _, _, dec_t, dec_h, dec_w = logits_volume.shape
                        target_volume = x[:, :1, :dec_t, :dec_h, :dec_w]
                        if target_volume.shape != logits_volume.shape:
                            raise RuntimeError(
                                "Stage 3C local-field target/decoder shape mismatch: "
                                f"target={tuple(target_volume.shape)}, "
                                f"logits={tuple(logits_volume.shape)}"
                            )
                        loss_ctx_field = local_moment_field_loss(
                            logits_b1thw=logits_volume,
                            target_b1thw=target_volume,
                            patch_size=vqvae.patch_size,
                            tau=ctx_field_tau,
                            min_active_spikes=1,
                            min_shape_spikes=5,
                            min_trend_spikes=6,
                            min_trend_frames=3,
                            prob_threshold=float(vqvae.best_thr_tol.item()),
                        )
                    if lambda_adj > 0.0:
                        if memory_adj is None:
                            raise RuntimeError("lambda_adj > 0 but memory_adj is None.")
                        adjacency_target = memory_adj.get(
                            global_ctx, device=device, dtype=torch.float32
                        )
                        adjacency_confidence = memory_adj.get_confidence(
                            global_ctx,
                            device=device,
                            dtype=torch.float32,
                            den_scale=memory_adj_conf_den_scale,
                        )
                        adjacency_parts = short_gap_excess_loss_from_logits_batch_targets(
                            logits_b1thw=logits_volume.float(),
                            target_gap_rates_bg=adjacency_target.float(),
                            max_gap=isi_max_gap,
                            gap_bins=gap_bins,
                            tau=isi_tau,
                            prob_threshold=float(vqvae.best_thr_tol.item()),
                            margin=isi_margin,
                            lower_margin=isi_lower_margin,
                            lower_weight=isi_lower_weight,
                            confidence_bg=adjacency_confidence.float(),
                            return_parts=True,
                        )
                        loss_adj = torch.nan_to_num(
                            adjacency_parts["loss"], nan=0.0, posinf=1e3, neginf=0.0
                        )
                    if lambda_spatial > 0.0:
                        if memory_tok is None:
                            raise RuntimeError("lambda_spatial > 0 but memory_tok is None.")
                        full_teacher = memory_tok.get(
                            global_ctx, device=device, dtype=logits_volume.dtype
                        )
                        _, _, _, height, width = logits_volume.shape
                        _, patch_h, patch_w = vqvae.patch_size
                        teacher = sample_full_token_map_to_crop(
                            full_tok_bhw=full_teacher,
                            out_tok_hw=(height // patch_h, width // patch_w),
                            patch_size=vqvae.patch_size,
                            roi_hw=batch.get("roi_hw", None),
                            pad_hw=batch.get("pad_hw", None),
                        )
                        student = soft_spatial_token_map_from_logits(
                            logits_b1thw=logits_volume,
                            patch_size=vqvae.patch_size,
                            tau=0.25,
                            prob_threshold=float(vqvae.best_thr_tol.item()),
                        )
                        allowed = dilate_spatial_support_hw(
                            teacher.detach(), radius_h=1, radius_w=1
                        )
                        loss_spatial = spatial_support_violation_loss(
                            pred_support=student,
                            allowed_support=allowed,
                            neg_thresh=0.20,
                        )

                    auxiliary_raw = (
                        float(lambda_ctx) * loss_ctx
                        + float(lambda_ctx_field) * loss_ctx_field
                        + float(lambda_adj) * loss_adj
                        + float(lambda_spatial) * loss_spatial
                    )
                    loss = float(lambda_detr) * loss_detr + aux_scale * auxiliary_raw

            if train:
                if (
                    not auxiliary_gradient_audited
                    and aux_scale > 0.0
                    and auxiliary_raw.requires_grad
                ):
                    gradients = torch.autograd.grad(
                        auxiliary_raw,
                        trainable_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    connected = sum(gradient is not None for gradient in gradients)
                    nonzero = sum(
                        gradient is not None
                        and bool(torch.isfinite(gradient).all())
                        and float(gradient.detach().abs().sum().item()) > 0.0
                        for gradient in gradients
                    )
                    if connected == 0:
                        raise RuntimeError(
                            "Stage 3C decoder/context losses are disconnected from all event-side parameters."
                        )
                    gradient_status = {
                        name: (
                            gradient is not None,
                            bool(torch.isfinite(gradient).all()) if gradient is not None else False,
                            float(gradient.detach().abs().sum().item()) if gradient is not None else 0.0,
                        )
                        for name, gradient in zip(trainable_names, gradients)
                    }
                    if activity_prior.coordinate_mode == "joint_dense":
                        failed_joint = [
                            name
                            for name, (is_connected, is_finite, magnitude) in gradient_status.items()
                            if name.startswith("grid_head.")
                            and (not is_connected or not is_finite or magnitude <= 0.0)
                        ]
                        if failed_joint:
                            raise RuntimeError(
                                "Stage 3C auxiliary losses do not provide non-zero finite "
                                f"gradients to joint-head parameters: {failed_joint}"
                            )
                    print(
                        "  [3C gradient audit] decoder/context losses connect to "
                        f"{connected}/{len(trainable_parameters)} trainable tensors; "
                        f"{nonzero} have non-zero finite gradients on this batch. "
                        f"Details={gradient_status}"
                    )
                    auxiliary_gradient_audited = True

                scaler.scale(loss / float(grad_accum_steps)).backward()
                should_step = (
                    iteration % grad_accum_steps == 0
                    or iteration == len(loader)
                )
                if should_step:
                    if grad_clip is not None and grad_clip > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(
                            trainable_parameters, float(grad_clip)
                        )
                    scaler.step(opt)
                    scaler.update()

            batch_size = x.shape[0]
            total_samples += float(batch_size)
            predicted_roi_soft_k = (soft_roi * roi).sum(dim=1).mean()
            predicted_roi_hard_k = (hard_roi * roi).sum(dim=1).mean()
            visible_gt_k = ((~roi) & motif_targets["active"].bool()).sum(dim=1).float().mean()
            full_activity_k = activity_full_st.detach().sum(dim=1).mean()
            target_roi_k = (roi & motif_targets["active"].bool()).sum(dim=1).float().mean()
            metrics = {
                "loss": loss,
                "loss_detr": loss_detr,
                "loss_ctx": loss_ctx,
                "loss_ctx_field": loss_ctx_field,
                "loss_adj": loss_adj,
                "loss_spatial": loss_spatial,
                "auxiliary_scale": aux_scale,
                "predicted_roi_softK": predicted_roi_soft_k,
                "predicted_roi_hardK": predicted_roi_hard_k,
                "visible_gtK": visible_gt_k,
                "full_activityK": full_activity_k,
                "target_roiK": target_roi_k,
                "countE": auxiliary_detr["count_expected_mean"],
                "predicted_active_motif_tokens": motif_inputs["pred_active"].sum(dim=1).float().mean(),
                "visible_active_motif_tokens": motif_inputs["visible_active"].sum(dim=1).float().mean(),
            }
            metrics.update({f"detr_{key}": value for key, value in auxiliary_detr.items()})
            for key, value in metrics.items():
                if torch.is_tensor(value):
                    value = float(value.detach().item())
                totals[key] = totals.get(key, 0.0) + float(value) * batch_size

            if not train:
                _accumulate(
                    hard_totals,
                    _hard_activity_batch_stats(
                        activity_prior,
                        activity_output,
                        activity_targets,
                        roi_mask=predict_mask,
                        tolerance=hard_tolerance,
                        memory_tok=memory_tok,
                        global_ctx=global_ctx,
                        roi_hw=batch.get("roi_hw", None),
                        pad_hw=batch.get("pad_hw", None),
                        patch_size=vqvae.patch_size,
                    ),
                )

            if train and log_every and iteration % log_every == 0:
                denominator = max(total_samples, 1.0)
                print(
                    f"  [3C calibration] it {iteration:05d}: "
                    f"loss={totals['loss'] / denominator:.4f} "
                    f"detr={totals['loss_detr'] / denominator:.4f} "
                    f"ctx={totals['loss_ctx'] / denominator:.4f} "
                    f"field={totals['loss_ctx_field'] / denominator:.4f} "
                    f"adj={totals['loss_adj'] / denominator:.4f} "
                    f"sp={totals['loss_spatial'] / denominator:.4f} "
                    f"roiSoftK={totals['predicted_roi_softK'] / denominator:.2f} "
                    f"roiHardK={totals['predicted_roi_hardK'] / denominator:.2f} "
                    f"visibleK={totals['visible_gtK'] / denominator:.2f} "
                    f"fullK={totals['full_activityK'] / denominator:.2f} "
                    f"countE={totals['countE'] / denominator:.2f} "
                    f"targetRoiK={totals['target_roiK'] / denominator:.2f} "
                    f"auxRamp={aux_scale:.3f}"
                )

        denominator = max(total_samples, 1.0)
        result = {key: value / denominator for key, value in totals.items()}
        result["epoch_duration_seconds"] = float(time.perf_counter() - epoch_start)
        result["peak_gpu_memory_mb"] = (
            float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            if device.type == "cuda"
            else 0.0
        )
        if not train:
            result.update(_finalize_hard_activity_stats(hard_totals))
        return result

    history = {
        "train": [],
        "val": [],
        "generation_val": [],
        "baseline_generation": baseline_metrics,
        "hyperparameters": hyperparameters,
        "generation_metric_formula": STAGE3C_GENERATION_COMPOSITE_FORMULA,
    }
    best_score = float(baseline_metrics["generation_metric"])
    best_epoch = 0
    best_state = None
    best_metrics = baseline_metrics
    patience = 0

    for epoch in range(1, int(epochs) + 1):
        train_metrics = _run_epoch(train_loader, train=True, epoch=epoch)
        validation_metrics = _run_epoch(val_loader, train=False, epoch=epoch)
        generation_metrics = evaluate_true_stage3_generation(
            activity_prior,
            motif_prior,
            vqvae,
            val_loader,
            blank_code=blank_code,
            max_batches=generation_val_max_batches,
            motif_steps=generation_motif_steps,
            hard_tolerance=hard_tolerance,
            gap_bins=gap_bins,
            deterministic_masks=deterministic_val_masks,
            seed=generation_seed,
        )
        history["train"].append(train_metrics)
        history["val"].append(validation_metrics)
        history["generation_val"].append(generation_metrics)

        accepted, rejection_reasons = _stage3c_accepts_candidate(
            generation_metrics,
            baseline_metrics,
            best_score + float(min_delta),
        )
        print(
            f"[activity-refine epoch {epoch:03d}] "
            f"train={train_metrics['loss']:.4f} "
            f"val={validation_metrics['loss']:.4f} "
            f"roiSoftK={validation_metrics['predicted_roi_softK']:.2f} "
            f"roiHardK={validation_metrics['predicted_roi_hardK']:.2f} "
            f"visibleK={validation_metrics['visible_gtK']:.2f} "
            f"fullK={validation_metrics['full_activityK']:.2f} "
            f"countE={validation_metrics['countE']:.2f} "
            f"targetRoiK={validation_metrics['target_roiK']:.2f} "
            f"exactF1={generation_metrics['hard_exact_f1']:.4f} "
            f"tolF1={generation_metrics['hard_tolerance_f1']:.4f} "
            f"decodedRelErr={generation_metrics['decoded_spike_count_relative_error']:.4f} "
            f"genScore={generation_metrics['generation_metric']:.6f} "
            f"peakMB={validation_metrics['peak_gpu_memory_mb']:.1f} "
            f"seconds={validation_metrics['epoch_duration_seconds']:.1f} "
            f"accepted={accepted}"
        )
        if rejection_reasons:
            print("  Stage 3C checkpoint gate: " + "; ".join(rejection_reasons))

        if accepted:
            best_score = float(generation_metrics["generation_metric"])
            best_epoch = int(epoch)
            best_metrics = generation_metrics
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in activity_prior.state_dict().items()
            }
            patience = 0
            torch.save({
                "activity_prior": best_state,
                "epoch": best_epoch,
                "accepted": True,
                "selection": "true_hard_generation_composite",
                "generation_metrics": best_metrics,
                "baseline_generation_metrics": baseline_metrics,
                "generation_metric_formula": STAGE3C_GENERATION_COMPOSITE_FORMULA,
                "token_grid": token_grid,
                "Kmax": int(activity_prior.Kmax),
                **activity_prior.coordinate_metadata(),
                "hyperparameters": hyperparameters,
                "trainable_parameter_names": trainable_names,
                "frozen_motif_prior": True,
                "deterministic_validation_masks": bool(deterministic_val_masks),
            }, ckpt_out)
            print(f"  saved accepted Stage 3C checkpoint: {ckpt_out}")
        else:
            patience += 1
            if patience >= int(early_stop_patience):
                print(
                    f"Early stopping Stage 3C at epoch {epoch}; no accepted "
                    f"hard-generation improvement for {patience} epochs."
                )
                break

    if best_state is None:
        activity_prior.load_state_dict(baseline_state, strict=True)
        torch.save({
            "activity_prior": baseline_state,
            "epoch": 0,
            "accepted": False,
            "selection": "fallback_to_unrefined_stage3b",
            "generation_metrics": baseline_metrics,
            "baseline_generation_metrics": baseline_metrics,
            "generation_metric_formula": STAGE3C_GENERATION_COMPOSITE_FORMULA,
            "token_grid": token_grid,
            "Kmax": int(activity_prior.Kmax),
            **activity_prior.coordinate_metadata(),
            "hyperparameters": hyperparameters,
            "trainable_parameter_names": trainable_names,
            "frozen_motif_prior": True,
            "deterministic_validation_masks": bool(deterministic_val_masks),
            "reason": "No Stage 3C epoch passed the hard-generation checkpoint gate.",
        }, ckpt_out)
        print(
            "Stage 3C did not outperform the unrefined Stage 3B checkpoint; "
            "restored Stage 3B event heads."
        )
    else:
        activity_prior.load_state_dict(best_state, strict=True)
        print(
            f"Restored accepted Stage 3C epoch {best_epoch} with "
            f"generation_metric={best_score:.6f}."
        )

    history["accepted"] = best_state is not None
    history["best_epoch"] = int(best_epoch)
    history["best_generation"] = best_metrics
    return history
