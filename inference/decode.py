#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:28:50 2026

@author: derik
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..utils.recon import compute_activity_ctx
from ..utils.constants import (
    ACTIVITY_CTX_NAMES,
    ACTIVITY_CTX_DIM,
)
from ..visualization.video import save_volume_as_mp4_imageio


LOCAL_CTX_NAMES = list(ACTIVITY_CTX_NAMES)


def xgen_to_thw_np(x_b1thw_i):
    # x_b1thw_i: (1,T,H,W)
    return x_b1thw_i.detach().cpu().numpy()[0].astype(np.uint8)


def save_xgen_video(x_b1thw_i, path, fps=30):
    thw = xgen_to_thw_np(x_b1thw_i)          # (T,H,W)
    hwt = np.transpose(thw, (1, 2, 0)) * 255 # (H,W,T)
    save_volume_as_mp4_imageio(hwt.astype(np.uint8), str(path), fps=fps)


def measure_xgen_local_ctx(x_b1thw_i):
    thw = xgen_to_thw_np(x_b1thw_i)
    return compute_activity_ctx(thw)


def compare_local_ctx(
    measured,
    intended=None,
    *,
    local_min=None,
    local_max=None,
    local_mean=None,
    local_std=None,
    partial_local=None,
    atol=None,
):
    measured = np.asarray(measured, dtype=np.float32)
    
    measured = measured.reshape(-1)

    if measured.shape != (len(LOCAL_CTX_NAMES),):
        raise ValueError(
            f"Measured local context must have "
            f"{len(LOCAL_CTX_NAMES)} values, "
            f"got shape {measured.shape}"
        )

    if atol is None:
        atol = np.array(
            [
                0.50,  # log mean density
                0.10,  # var_x
                0.10,  # var_y
                0.10,  # var_t
                0.10,  # cov_xy
                0.10,  # cov_xt
                0.10,  # cov_yt
                0.05,  # active-site ratio
                0.25,  # temporal trend
            ],
            dtype=np.float32,
        )

    if intended is not None:
        intended = np.asarray(intended, dtype=np.float32)
        
        intended = intended.reshape(-1)

        if intended.shape != (len(LOCAL_CTX_NAMES),):
            raise ValueError(
                f"Intended local context must have "
                f"{len(LOCAL_CTX_NAMES)} values, "
                f"got shape {intended.shape}"
            )

    row = {}
    all_match = True
    all_range = True

    partial_target = None
    if partial_local is not None:
        partial_target = np.full((len(LOCAL_CTX_NAMES),), np.nan, dtype=np.float32)
        if isinstance(partial_local, dict):
            for k, v in partial_local.items():
                j = LOCAL_CTX_NAMES.index(k) if isinstance(k, str) else int(k)
                partial_target[j] = float(v)
        else:
            arr = np.asarray(partial_local, dtype=np.float32).reshape(-1)
            partial_target[:len(arr)] = arr

    for j, name in enumerate(LOCAL_CTX_NAMES):
        row[f"gen_{name}"] = float(measured[j])

        if intended is not None:
            err = float(measured[j] - intended[j])
            ok = bool(abs(err) <= float(atol[j]))
            row[f"target_{name}"] = float(intended[j])
            row[f"err_{name}"] = err
            row[f"match_{name}"] = ok
            all_match = all_match and ok

        if partial_target is not None and np.isfinite(partial_target[j]):
            err = float(measured[j] - partial_target[j])
            ok = bool(abs(err) <= float(atol[j]))
            row[f"partial_target_{name}"] = float(partial_target[j])
            row[f"partial_err_{name}"] = err
            row[f"partial_match_{name}"] = ok
            all_match = all_match and ok

        if local_min is not None and local_max is not None:
            ok = bool(float(local_min[j]) <= float(measured[j]) <= float(local_max[j]))
            row[f"in_bank_range_{name}"] = ok
            all_range = all_range and ok

        if local_mean is not None and local_std is not None:
            row[f"z_from_bank_{name}"] = float(
                (measured[j] - local_mean[j]) / (local_std[j] + 1e-8)
            )

    row["all_context_matches"] = bool(all_match)
    row["all_features_in_bank_range"] = bool(all_range)
    return row





def decode_motif_logits_soft_given_activity(
    model,
    logits,
    targets,
    *,
    motif_prior,
    activity_prob=None,
    activity_ids=None,
    grid,
    global_ctx,
    local_ctx,
    tau_z=0.25,
    roi_hw=None,
    pad_hw=None,
):
    """Decode current flat-code outputs with a hard-forward argmax.

    Generated positions use a straight-through argmax over the flat Stage-2B
    codebook. Visible active positions retain their exact encoded entry. All
    other positions use the VQ-VAE blank latent.
    """
    f_t = targets["f"].long()
    B, N = f_t.shape
    device = logits["flat"].device
    dtype = logits["flat"].dtype
    V = int(motif_prior.V)

    if activity_prob is not None:
        p_active = activity_prob.to(device=device, dtype=dtype)
    elif activity_ids is not None:
        p_active = activity_ids.to(device=device, dtype=dtype).clamp(0, 1)
    else:
        p_active = targets["a"].to(device=device, dtype=dtype).clamp(0, 1)
    if p_active.shape != (B, N):
        raise ValueError(f"p_active must have shape {(B, N)}, got {tuple(p_active.shape)}")

    fallback = targets.get("f_loss_mask", targets["z_loss_mask"]).bool()
    predicted_motif = targets.get("decode_motif_mask", fallback).to(device).bool()
    default_visible = (
        targets.get("active", targets["a"].bool()).bool()
        & ~targets.get("predict_mask", predicted_motif).bool()
    )
    visible_motif = targets.get("visible_motif_mask", default_visible).to(device).bool()
    if predicted_motif.shape != (B, N) or visible_motif.shape != (B, N):
        raise ValueError("Motif source masks must have shape (B,N).")
    if bool((predicted_motif & visible_motif).any()):
        raise ValueError("Predicted and visible motif masks overlap.")

    # Flat codebook lookup. There is no parent/child split at prior time:
    # Stage 2B already summed the ladder into one entry per token.
    E = motif_prior.flat_codebook.detach().to(device=device, dtype=dtype)   # (V,D)

    temperature = max(float(tau_z), 1e-6)
    # The OOV bin has no codebook entry, so it can never be decoded.
    flat_logits = logits["flat"][..., :V]
    f_prob = F.softmax(flat_logits / temperature, dim=-1)
    f_hard = F.one_hot(f_prob.argmax(dim=-1), num_classes=V).to(f_prob.dtype)
    f_prob_st = f_hard + f_prob - f_prob.detach()

    z_pred = torch.einsum("bnv,vd->bnd", f_prob_st, E)
    z_gt = E[f_t.clamp(0, V - 1)]

    blank_token = model.vq.blank_token.detach().to(device=device, dtype=dtype)
    z_blank = blank_token.view(1, 1, -1).expand(B, N, -1)
    z_active = torch.where(visible_motif.unsqueeze(-1), z_gt, z_blank)
    z_active = torch.where(predicted_motif.unsqueeze(-1), z_pred, z_active)
    z_q = (1.0 - p_active.unsqueeze(-1)) * z_blank + p_active.unsqueeze(-1) * z_active

    _, pos_dec = model._get_pos_embed(grid, z_q.device, z_q.dtype)
    z_dec_base_no_pos = model.code_to_dec(z_q)
    type_offset = model.activity_type_offset.to(z_dec_base_no_pos)
    signed_type_offset = (2.0 * p_active.unsqueeze(-1) - 1.0) * type_offset.view(1, 1, -1)
    z_d = z_dec_base_no_pos + model.offset_scale * signed_type_offset + pos_dec
    model._ensure_dec_masks(grid, device=z_d.device)
    for blk in model.dec_blocks:
        z_d = blk(z_d)
    z_d = model.dec_norm(z_d)
    logits_vol_raw, pred_patches_raw = model.patch_renderer(z_d, grid, return_patches=True)
    pred_patches, spatial_diag = model._apply_output_biases(
        pred_patches_raw, grid=grid, global_ctx=global_ctx, roi_hw=roi_hw, pad_hw=pad_hw
    )
    logits_vol = model.unpatchify(pred_patches, grid)
    return {
        "z_q": z_q, "p_active": p_active, "flat_prob_st": f_prob_st,
        "predicted_motif_mask": predicted_motif,
        "visible_motif_mask": visible_motif, "logits_vol": logits_vol,
        "logits_vol_raw": logits_vol_raw, "pred_patches": pred_patches,
        "pred_patches_raw": pred_patches_raw, "spatial_diag": spatial_diag,
    }







@torch.no_grad()
def decode_flat_ids_to_xgen(
    model,
    flat_ids,
    *,
    flat_codebook,
    grid,
    global_ctx,
    local_ctx,
    threshold: Optional[float] = None,
    roi_hw=None,
    pad_hw=None,
):
    """Decode generated flat Stage-2B ids straight to a spike volume.

    There is no ladder to walk at inference: the flat entry already IS
    s0*z1 + s1*z2 + s2*z3. Inactive positions (-1) take the VQ blank token.
    """
    device = next(model.parameters()).device
    flat_ids = flat_ids.to(device=device, dtype=torch.long)
    B, N = flat_ids.shape
    V = int(flat_codebook.shape[0])

    E = flat_codebook.detach().to(device=device, dtype=torch.float32)
    blank = model.vq.blank_token.detach().to(device=device, dtype=torch.float32)

    active = flat_ids.ge(0)
    z_q = blank.view(1, 1, -1).expand(B, N, -1).clone()
    z_q[active] = E[flat_ids[active].clamp(0, V - 1)]

    dec = model._decode_quantized_latent(
        z_q=z_q,
        active_mask=active,
        grid=grid,
        global_ctx=global_ctx,
        roi_hw=roi_hw,
        pad_hw=pad_hw,
    )

    logits = dec["logits_vol"]
    prob = torch.sigmoid(logits)
    if threshold is None:
        threshold = float(model.best_thr_tol.item())

    return {
        "logits": logits,
        "prob": prob,
        "x_gen": (prob >= threshold).float(),
        "threshold": float(threshold),
        "z_q": z_q,
        "active": active,
    }


@torch.no_grad()
def decode_codes_to_xgen(
    model,
    codes,
    *,
    grid,
    global_ctx,
    local_ctx,
    threshold: Optional[float] = None,
    roi_hw=None,
    pad_hw=None,
):
    dec = model.decode_from_codes(
        codes,
        grid=grid,
        global_ctx=global_ctx,
        local_ctx=local_ctx,
        roi_hw=roi_hw,
        pad_hw=pad_hw,
    )

    logits = dec["logits_vol"]
    prob = torch.sigmoid(logits)
    if threshold is None:
        threshold = float(model.best_thr_tol.item())
    
    x_gen = (prob >= threshold).float()

    return {
        "logits": logits,
        "prob": prob,
        "x_gen": x_gen,
        "threshold": float(threshold),
    }


@torch.no_grad()
def save_generated_batch_outputs(
    *,
    x_gen,
    out_dir,
    prefix,
    intended_local_ctx=None,
    fps=30,
    local_min=None,
    local_max=None,
    local_mean=None,
    local_std=None,
    partial_local=None,
):
    out_dir = Path(out_dir)
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for i in range(x_gen.shape[0]):
        video_path = video_dir / f"{prefix}_s{i:03d}.mp4"
        save_xgen_video(x_gen[i], video_path, fps=fps)

        measured = measure_xgen_local_ctx(x_gen[i])
        intended = None
        if intended_local_ctx is not None:
            intended = intended_local_ctx[i].detach().cpu().numpy()

        row = {
            "sample": int(i),
            "video_path": str(video_path),
        }
        row.update(compare_local_ctx(
            measured,
            intended,
            local_min=local_min,
            local_max=local_max,
            local_mean=local_mean,
            local_std=local_std,
            partial_local=partial_local,
        ))
        rows.append(row)

    return rows


def save_generation_metrics_json(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)