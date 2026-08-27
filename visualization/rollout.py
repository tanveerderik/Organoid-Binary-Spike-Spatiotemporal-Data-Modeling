#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generation utilities.

Public API
----------
1) generate_full_video(...)
   Fully masked conditional generation from global/local context.

2) generate_masked_video(...)
   Partial generation / infill / inpainting from an observed video + mask_spec.

3) rollout_causal_long(...)
   Long free-running causal rollout by repeated causal suffix generation.
"""

import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from .video import save_volume_as_mp4_imageio


# ---------------------------------------------------------------------
# minimal rollout-specific helpers
# ---------------------------------------------------------------------

def _to_b1thw(
    x: Union[np.ndarray, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Accepts:
      - (H,W,T)
      - (T,H,W)
      - (1,1,T,H,W)
      - torch.Tensor in any of the above
    Returns:
      - (1,1,T,H,W)
    """
    if isinstance(x, torch.Tensor):
        if x.ndim == 5:
            return x.to(device=device, dtype=dtype)
        x = x.detach().cpu().numpy()

    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D or 5D input, got shape={x.shape}")

    x = x.astype(np.float32, copy=False)

    # Heuristic: if first axis is clearly time, interpret as (T,H,W)
    if x.shape[0] <= 32 and x.shape[1] > 32 and x.shape[2] > 32:
        x = x.transpose(1, 2, 0)  # -> (H,W,T)

    if x.max() > 1.0:
        x = (x > 0).astype(np.float32)

    x = x.transpose(2, 0, 1)[None, None, ...]  # -> (1,1,T,H,W)
    return torch.from_numpy(x).to(device=device, dtype=dtype)


def _to_hw_t(x_b1thw: torch.Tensor) -> np.ndarray:
    """
    (1,1,T,H,W) -> (H,W,T)
    """
    if x_b1thw.ndim != 5 or x_b1thw.shape[0] != 1 or x_b1thw.shape[1] != 1:
        raise ValueError(f"Expected (1,1,T,H,W), got {tuple(x_b1thw.shape)}")
    return x_b1thw[0, 0].detach().float().cpu().numpy().transpose(1, 2, 0)


def _ctx_np(
    ctx: Union[np.ndarray, torch.Tensor, None],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if ctx is None:
        return None
    if isinstance(ctx, np.ndarray):
        ctx = torch.from_numpy(ctx)
    ctx = ctx.to(device=device, dtype=torch.float32)
    if ctx.ndim == 1:
        ctx = ctx.unsqueeze(0)
    return ctx


def _task_np(task_id: Union[int, np.ndarray, torch.Tensor], device: torch.device) -> torch.Tensor:
    if isinstance(task_id, torch.Tensor):
        t = task_id.to(device=device, dtype=torch.long)
    elif isinstance(task_id, np.ndarray):
        t = torch.from_numpy(task_id).to(device=device, dtype=torch.long)
    else:
        t = torch.tensor([int(task_id)], device=device, dtype=torch.long)
    if t.ndim == 0:
        t = t.unsqueeze(0)
    return t


def _save_generation_videos(
    prob_hw_t: np.ndarray,
    bin_hw_t: np.ndarray,
    out_dir: Optional[str],
    *,
    fps: int = 30,
    pool_t: Optional[int] = None,
    save_mask_hw_t: Optional[np.ndarray] = None,
    prefix: str = "",
) -> None:
    if out_dir is None:
        return

    os.makedirs(out_dir, exist_ok=True)
    pre = f"{prefix}_" if prefix else ""

    save_volume_as_mp4_imageio(
        prob_hw_t,
        os.path.join(out_dir, f"{pre}prob.mp4"),
        fps=fps,
        pool_t=pool_t,
    )
    save_volume_as_mp4_imageio(
        bin_hw_t,
        os.path.join(out_dir, f"{pre}bin.mp4"),
        fps=fps,
        pool_t=pool_t,
    )
    if save_mask_hw_t is not None:
        save_volume_as_mp4_imageio(
            save_mask_hw_t.astype(np.float32),
            os.path.join(out_dir, f"{pre}mask.mp4"),
            fps=fps,
            pool_t=pool_t,
        )


@torch.no_grad()
def _encode_decode_sample(
    vqvae: torch.nn.Module,
    prior: torch.nn.Module,
    x_b1thw: torch.Tensor,
    *,
    global_ctx: Optional[torch.Tensor],
    local_ctx: Optional[torch.Tensor],
    task_id: torch.Tensor,
    predict_mask_tok: torch.Tensor,
    steps: int,
    top_k: Optional[int],
    thr: Optional[float],
) -> Dict[str, Any]:
    enc = vqvae(x_b1thw, global_ctx=global_ctx, local_ctx=local_ctx)
    gt_codes = enc["codes"].long()
    grid = tuple(enc["grid"])

    sampled_codes = prior.sample_from_prior(
        gt_codes=gt_codes,
        global_ctx=global_ctx,
        local_ctx=local_ctx,
        task_id=task_id,
        steps=int(steps),
        top_k=top_k,
        predict_mask=predict_mask_tok,
    )

    out = vqvae.decode_from_codes(
        sampled_codes,
        grid=grid,
        global_ctx=global_ctx,
        local_ctx=local_ctx,
    )

    logits_vol = out["logits_vol"]
    prob_vol = torch.sigmoid(logits_vol)
    thr = float(getattr(vqvae, "best_thr", 0.5) if thr is None else thr)
    bin_vol = (prob_vol >= thr).float()

    return {
        "grid": grid,
        "codes_gt": gt_codes,
        "codes_sampled": sampled_codes,
        "predict_mask_tok": predict_mask_tok,
        "logits_vol": logits_vol,
        "prob_vol": prob_vol,
        "bin_vol": bin_vol,
        "thr": thr,
        "assay_spatial_pix2d_support": out.get("assay_spatial_pix2d_support", None),
    }


# ---------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------

@torch.no_grad()
def generate_full_video(
    vqvae: torch.nn.Module,
    prior: torch.nn.Module,
    *,
    out_shape_thw: Tuple[int, int, int],
    global_ctx: Union[np.ndarray, torch.Tensor],
    local_ctx: Union[np.ndarray, torch.Tensor],
    task_id: int = 0,
    steps: int = 24,
    top_k: Optional[int] = None,
    thr: Optional[float] = None,
    out_dir: Optional[str] = None,
    fps: int = 30,
    pool_t: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fully masked conditional generation.
    out_shape_thw = (T,H,W), must be divisible by patch_size.
    """
    device = next(vqvae.parameters()).device
    T, H, W = map(int, out_shape_thw)
    pT, pH, pW = map(int, vqvae.patch_size)

    if (T % pT) != 0 or (H % pH) != 0 or (W % pW) != 0:
        raise ValueError(
            f"Requested output shape {(T, H, W)} must be divisible by patch_size {tuple(vqvae.patch_size)}"
        )

    grid = (T // pT, H // pH, W // pW)
    N = int(np.prod(grid))

    global_ctx_t = _ctx_np(global_ctx, device)
    local_ctx_t = _ctx_np(local_ctx, device)
    task_id_t = _task_np(task_id, device)

    dummy_x = torch.zeros((1, 1, T, H, W), device=device, dtype=torch.float32)
    predict_mask_tok = torch.ones((1, N), device=device, dtype=torch.float32)

    out = _encode_decode_sample(
        vqvae,
        prior,
        dummy_x,
        global_ctx=global_ctx_t,
        local_ctx=local_ctx_t,
        task_id=task_id_t,
        predict_mask_tok=predict_mask_tok,
        steps=steps,
        top_k=top_k,
        thr=thr,
    )

    prob_hw_t = _to_hw_t(out["prob_vol"])
    bin_hw_t = _to_hw_t(out["bin_vol"])

    _save_generation_videos(prob_hw_t, bin_hw_t, out_dir, fps=fps, pool_t=pool_t)

    return {
        "mode": "full_generation",
        "task_id": int(task_id_t[0].item()),
        "grid": grid,
        "prob_hw_t": prob_hw_t,
        "bin_hw_t": bin_hw_t,
        **out,
    }


@torch.no_grad()
def generate_masked_video(
    vqvae: torch.nn.Module,
    prior: torch.nn.Module,
    x: Union[np.ndarray, torch.Tensor],
    *,
    global_ctx: Optional[Union[np.ndarray, torch.Tensor]],
    local_ctx: Optional[Union[np.ndarray, torch.Tensor]],
    task_id: Union[int, np.ndarray, torch.Tensor],
    mask_spec: Union[Dict[str, Any], list[Dict[str, Any]]],
    steps: int = 24,
    top_k: Optional[int] = None,
    thr: Optional[float] = None,
    out_dir: Optional[str] = None,
    fps: int = 30,
    pool_t: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Partial generation / infill / inpainting.
    Supports whatever vqvae.predict_mask_from_spec(...) supports.
    """
    device = next(vqvae.parameters()).device
    x_b1thw = _to_b1thw(x, device)
    global_ctx_t = _ctx_np(global_ctx, device)
    local_ctx_t = _ctx_np(local_ctx, device)
    task_id_t = _task_np(task_id, device)

    enc = vqvae(x_b1thw, global_ctx=global_ctx_t, local_ctx=local_ctx_t)
    grid = tuple(enc["grid"])

    spec = [mask_spec] if isinstance(mask_spec, dict) else list(mask_spec)
    predict_mask_tok = vqvae.predict_mask_from_spec(
        spec,
        grid,
        device=device,
        dtype=torch.float32,
    )
    if predict_mask_tok.dim() == 3:
        predict_mask_tok = predict_mask_tok.squeeze(-1)

    out = _encode_decode_sample(
        vqvae,
        prior,
        x_b1thw,
        global_ctx=global_ctx_t,
        local_ctx=local_ctx_t,
        task_id=task_id_t,
        predict_mask_tok=predict_mask_tok,
        steps=steps,
        top_k=top_k,
        thr=thr,
    )

    prob_hw_t = _to_hw_t(out["prob_vol"])
    bin_hw_t = _to_hw_t(out["bin_vol"])
    ref_hw_t = _to_hw_t(x_b1thw)

    pm = predict_mask_tok
    if pm.dim() == 2:
        pm = pm.unsqueeze(-1)

    P = int(np.prod(vqvae.patch_size)) * int(getattr(vqvae, "out_chans", 1))
    pm_vol = vqvae.unpatchify(pm.expand(pm.shape[0], pm.shape[1], P), grid)
    mask_hw_t = _to_hw_t(pm_vol) > 0.5

    _save_generation_videos(
        prob_hw_t,
        bin_hw_t,
        out_dir,
        fps=fps,
        pool_t=pool_t,
        save_mask_hw_t=mask_hw_t.astype(np.float32),
    )

    return {
        "mode": "masked_generation",
        "task_id": int(task_id_t[0].item()),
        "grid": grid,
        "mask_spec": mask_spec,
        "prob_hw_t": prob_hw_t,
        "bin_hw_t": bin_hw_t,
        "ref_hw_t": ref_hw_t,
        "predict_mask_tok": predict_mask_tok,
        "predict_mask_hw_t": mask_hw_t.astype(np.float32),
        **out,
    }


@torch.no_grad()
def rollout_causal_long(
    vqvae: torch.nn.Module,
    prior: torch.nn.Module,
    start_hw_t: np.ndarray,
    *,
    window_frames: int,
    prefix_frames: int,
    rollout_steps: int,
    global_ctx: Optional[Union[np.ndarray, torch.Tensor]],
    local_ctx: Optional[Union[np.ndarray, torch.Tensor]] = None,
    task_id: int = 1,
    steps_per_window: int = 24,
    top_k: Optional[int] = None,
    thr: Optional[float] = None,
    feed_mode: str = "bin",               # "bin" or "prob"
    ctx_mode: str = "prefix_recomputed",  # "prefix_recomputed" or "fixed"
    out_dir: Optional[str] = None,
    fps: int = 30,
    pool_t: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Repeated causal rollout by sliding-window suffix generation.
    """
    device = next(vqvae.parameters()).device
    thr = float(getattr(vqvae, "best_thr", 0.5) if thr is None else thr)

    start_hw_t = np.asarray(start_hw_t, dtype=np.float32)
    if start_hw_t.max() > 1.0:
        start_hw_t = (start_hw_t > 0).astype(np.float32)

    H, W, T0 = start_hw_t.shape
    if T0 < window_frames:
        raise ValueError(f"Need at least {window_frames} seed frames, got {T0}")
    if not (0 < prefix_frames < window_frames):
        raise ValueError(f"Need 0 < prefix_frames < window_frames, got {prefix_frames} vs {window_frames}")

    pT, pH, pW = map(int, vqvae.patch_size)
    if (window_frames % pT) != 0 or (H % pH) != 0 or (W % pW) != 0:
        raise ValueError(
            f"(window_frames,H,W)=({window_frames},{H},{W}) must be divisible by patch_size={tuple(vqvae.patch_size)}"
        )

    if feed_mode not in ("bin", "prob"):
        raise ValueError(f"Unknown feed_mode={feed_mode}")
    if ctx_mode not in ("prefix_recomputed", "fixed"):
        raise ValueError(f"Unknown ctx_mode={ctx_mode}")

    global_ctx_t = _ctx_np(global_ctx, device) if global_ctx is not None else None
    fixed_local_ctx_t = _ctx_np(local_ctx, device) if local_ctx is not None else None
    task_id_t = _task_np(task_id, device)

    current_hw_t = start_hw_t[:, :, -window_frames:].copy()
    suffix_frames = window_frames - prefix_frames

    prob_segments = [start_hw_t.copy()]
    bin_segments = [(start_hw_t > thr).astype(np.float32)]

    for _step_idx in range(int(rollout_steps)):
        prefix_hw_t = current_hw_t[:, :, :prefix_frames]

        if ctx_mode == "prefix_recomputed":
            # local ctx is recomputed from the currently visible prefix
            from ..utils.recon import compute_activity_ctx
            local_ctx_np = compute_activity_ctx((prefix_hw_t > thr).astype(np.uint8).transpose(2, 0, 1))
            local_ctx_t = torch.from_numpy(local_ctx_np).unsqueeze(0).to(device=device, dtype=torch.float32)
        else:
            if fixed_local_ctx_t is None:
                raise ValueError("ctx_mode='fixed' requires local_ctx")
            local_ctx_t = fixed_local_ctx_t

        step_out = generate_masked_video(
            vqvae,
            prior,
            current_hw_t,
            global_ctx=global_ctx_t,
            local_ctx=local_ctx_t,
            task_id=task_id_t,
            mask_spec={"type": "causal", "prefix_frames": int(prefix_frames)},
            steps=steps_per_window,
            top_k=top_k,
            thr=thr,
            out_dir=None,
        )

        full_prob_hw_t = step_out["prob_hw_t"]
        full_bin_hw_t = step_out["bin_hw_t"]

        new_prob_hw_t = full_prob_hw_t[:, :, prefix_frames:]
        new_bin_hw_t = full_bin_hw_t[:, :, prefix_frames:]

        prob_segments.append(new_prob_hw_t)
        bin_segments.append(new_bin_hw_t)

        feedback_suffix = new_prob_hw_t if feed_mode == "prob" else new_bin_hw_t
        current_hw_t = np.concatenate(
            [current_hw_t[:, :, suffix_frames:], feedback_suffix],
            axis=2,
        )

    rollout_prob_hw_t = np.concatenate(prob_segments, axis=2)
    rollout_bin_hw_t = np.concatenate(bin_segments, axis=2)

    _save_generation_videos(
        rollout_prob_hw_t,
        rollout_bin_hw_t,
        out_dir,
        fps=fps,
        pool_t=pool_t,
        prefix="rollout",
    )

    return {
        "mode": "causal_long_rollout",
        "task_id": int(task_id_t[0].item()),
        "window_frames": int(window_frames),
        "prefix_frames": int(prefix_frames),
        "suffix_frames": int(suffix_frames),
        "rollout_steps": int(rollout_steps),
        "feed_mode": str(feed_mode),
        "ctx_mode": str(ctx_mode),
        "thr": float(thr),
        "rollout_prob_hw_t": rollout_prob_hw_t,
        "rollout_bin_hw_t": rollout_bin_hw_t,
    }