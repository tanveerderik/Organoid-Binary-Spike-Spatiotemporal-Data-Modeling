#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 13:21:55 2025

@author: derik
"""

# utils/recon.py
from typing import Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np

from .constants import (
    ACTIVITY_CTX_NAMES,
    ACTIVITY_CTX_DIM,
)

from .losses import soft_binary_from_logits

def tokens_to_voxel_masks(model, pred_patches: torch.Tensor, grid: Tuple[int,int,int], predict_mask: Optional[torch.Tensor]):
    """
    pred_patches: (B,N,P); grid: (t_tok,h_tok,w_tok); predict_mask: (B,N) or (B,N,1) or None
    Returns: logits_vol (B,1,Tp,Hp,Wp), pmask_vol (B,1,Tp,Hp,Wp)
    """
    pmask_tok = predict_mask
    if pmask_tok is None:
        pmask_tok = torch.ones(pred_patches.shape[0], pred_patches.shape[1], 1,
                               device=pred_patches.device, dtype=pred_patches.dtype)
    elif pmask_tok.dim() == 2:
        pmask_tok = pmask_tok.unsqueeze(-1)

    logits_vol = model.unpatchify(pred_patches, grid)       # (B,1,Tp,Hp,Wp)
    P = pred_patches.shape[-1]
    pmask_tok_exp = pmask_tok.expand(-1, -1, P)             # (B,N,P)
    pmask_vol = model.unpatchify(pmask_tok_exp, grid)       # (B,1,Tp,Hp,Wp)
    return logits_vol, pmask_vol



def lift_crop_token_map_to_full(
    crop_tok_map_bhw: torch.Tensor,   # (B,h_tok,w_tok)
    full_hw: tuple[int, int],         # full pooled pixel size
    patch_size: tuple[int, int, int],
    roi_hw,
    pad_hw,
) -> torch.Tensor:
    """
    Lift current crop token maps into full assay token coordinates.

    roi_hw: length-B list of (y0, x0, y1, x1) in PRE-PAD pooled pixel coords
    pad_hw: length-B list of (top, bottom, left, right) padding added AFTER crop
    returns: (B, full_h_tok, full_w_tok)
    """
    import math
    import torch

    B, h_tok, w_tok = crop_tok_map_bhw.shape
    _, pH, pW = patch_size
    full_H, full_W = map(int, full_hw)

    full_h_tok = math.ceil(full_H / pH)
    full_w_tok = math.ceil(full_W / pW)

    out = crop_tok_map_bhw.new_zeros((B, full_h_tok, full_w_tok))

    if roi_hw is None:
        if h_tok != full_h_tok or w_tok != full_w_tok:
            raise ValueError(
                f"roi_hw is None but crop token size {(h_tok, w_tok)} "
                f"!= full token size {(full_h_tok, full_w_tok)}"
            )
        out.copy_(crop_tok_map_bhw)
        return out

    if pad_hw is None:
        pad_hw = [(0, 0, 0, 0)] * B

    for b in range(B):
        y0, x0, y1, x1 = map(float, roi_hw[b])
        ptop, pbot, pleft, pright = map(float, pad_hw[b])

        y_start = y0 - ptop
        x_start = x0 - pleft

        # nearest token anchor in full map
        y_tok0 = int(round(y_start / pH))
        x_tok0 = int(round(x_start / pW))

        y_tok1 = y_tok0 + h_tok
        x_tok1 = x_tok0 + w_tok

        fy0 = max(0, y_tok0)
        fx0 = max(0, x_tok0)
        fy1 = min(full_h_tok, y_tok1)
        fx1 = min(full_w_tok, x_tok1)

        cy0 = fy0 - y_tok0
        cx0 = fx0 - x_tok0
        cy1 = cy0 + (fy1 - fy0)
        cx1 = cx0 + (fx1 - fx0)

        if fy1 > fy0 and fx1 > fx0:
            out[b, fy0:fy1, fx0:fx1] = torch.maximum(
                out[b, fy0:fy1, fx0:fx1],
                crop_tok_map_bhw[b, cy0:cy1, cx0:cx1]
            )

    return out



def spatial_token_map_from_input(
    x_b1thw: torch.Tensor,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """
    x_b1thw: (B,1,T,H,W)
    returns: (B,h_tok,w_tok) in {0,1}
    A token is 1 if any pixel inside that spatial block fired at least once.
    """
    if x_b1thw.dim() != 5 or x_b1thw.size(1) != 1:
        raise ValueError(f"Expected (B,1,T,H,W), got {tuple(x_b1thw.shape)}")

    tgt_site_map = (x_b1thw > 0).any(dim=2).float()  # (B,1,H,W)
    pT, pH, pW = patch_size

    tgt_tok_map = F.max_pool2d(
        tgt_site_map,
        kernel_size=(pH, pW),
        stride=(pH, pW),
    )  # (B,1,h_tok,w_tok)

    return tgt_tok_map[:, 0]  # (B,h_tok,w_tok)

def sample_full_token_map_to_crop(
    full_tok_bhw: torch.Tensor,
    out_tok_hw: tuple[int, int],
    patch_size: tuple[int, int, int],
    roi_hw=None,
    pad_hw=None,
) -> torch.Tensor:
    """
    Convert full assay token memory map (B,full_h_tok,full_w_tok)
    to current runtime crop token coordinates (B,h_tok,w_tok).
    """
    if full_tok_bhw.dim() != 3:
        raise ValueError(f"Expected (B,h,w), got {tuple(full_tok_bhw.shape)}")

    B, full_h_tok, full_w_tok = full_tok_bhw.shape
    out_h_tok, out_w_tok = map(int, out_tok_hw)
    _, pH, pW = patch_size

    if roi_hw is None:
        if (full_h_tok, full_w_tok) == (out_h_tok, out_w_tok):
            return full_tok_bhw
        raise ValueError(
            f"roi_hw is None but full token map size {(full_h_tok, full_w_tok)} "
            f"!= runtime token size {(out_h_tok, out_w_tok)}"
        )

    if pad_hw is None:
        pad_hw = [(0, 0, 0, 0)] * B

    out = full_tok_bhw.new_zeros((B, out_h_tok, out_w_tok))

    for b in range(B):
        y0, x0, y1, x1 = map(float, roi_hw[b])
        ptop, pbot, pleft, pright = map(float, pad_hw[b])

        y_start = y0 - ptop
        x_start = x0 - pleft

        y_tok0 = int(round(y_start / pH))
        x_tok0 = int(round(x_start / pW))

        y_tok1 = y_tok0 + out_h_tok
        x_tok1 = x_tok0 + out_w_tok

        fy0 = max(0, y_tok0)
        fx0 = max(0, x_tok0)
        fy1 = min(full_h_tok, y_tok1)
        fx1 = min(full_w_tok, x_tok1)

        cy0 = fy0 - y_tok0
        cx0 = fx0 - x_tok0
        cy1 = cy0 + (fy1 - fy0)
        cx1 = cx0 + (fx1 - fx0)

        if fy1 > fy0 and fx1 > fx0:
            out[b, cy0:cy1, cx0:cx1] = full_tok_bhw[b, fy0:fy1, fx0:fx1]

    return out

def soft_spatial_token_map_from_logits(
    logits_b1thw: torch.Tensor,
    patch_size: tuple[int, int, int],
    full_hw: Optional[tuple[int, int]] = None,
    roi_hw=None,
    pad_hw=None,
    eps: float = 1e-6,
    tau: float = 0.25,
    prob_threshold: Optional[float] = None,
) -> torch.Tensor:
    """
    Build a soft spatial support map from decoder logits.

    Parameters
    ----------
    logits_b1thw : torch.Tensor
        (B,1,T,H,W) raw decoder logits, typically logits_vol_raw.
    patch_size : tuple[int, int, int]
        Model patch size (pT, pH, pW).
    full_hw : Optional[tuple[int, int]]
        Full pooled assay spatial size (H_full, W_full).
        If provided together with roi_hw/pad_hw, the crop token map is lifted
        into full assay token coordinates.
    roi_hw : optional
        Per-sample ROI boxes in pooled pixel coordinates, same convention as
        your existing lift_crop_token_map_to_full(...).
    pad_hw : optional
        Per-sample padding metadata, same convention as
        lift_crop_token_map_to_full(...).
    eps : float
        Numerical stability clamp.

    Returns
    -------
    torch.Tensor
        If full lifting is not requested:
            (B, h_tok, w_tok) soft token support in [0,1]
        If full lifting is requested:
            (B, full_h_tok, full_w_tok) soft token support in [0,1]

    Meaning
    -------
    Each spatial token value is the soft probability that at least one voxel
    inside that spatial patch fired at least once over time.
    """
    if logits_b1thw.dim() != 5 or logits_b1thw.size(1) != 1:
        raise ValueError(f"Expected (B,1,T,H,W), got {tuple(logits_b1thw.shape)}")

    B, _, T, H, W = logits_b1thw.shape
    _, pH, pW = patch_size

    if (H % pH) != 0 or (W % pW) != 0:
        raise ValueError(
            f"Spatial size {(H, W)} must be divisible by patch_size_hw {(pH, pW)}"
        )

    # Threshold-aware soft approximation of the final binary video.
    p = soft_binary_from_logits(
        logits_b1thw,
        tau=tau,
        prob_threshold=prob_threshold,
        eps=eps,
    ).clamp(eps, 1.0 - eps)

    # soft probability each spatial site was active at least once over time
    # P(active at least once) = 1 - prod_t (1 - p_t)
    site_support = 1.0 - torch.prod(1.0 - p, dim=2)         # (B,1,H,W)

    # pool spatially to token grid
    tok_support = F.max_pool2d(
        site_support,
        kernel_size=(pH, pW),
        stride=(pH, pW),
    )  # (B,1,h_tok,w_tok)

    tok_support = tok_support[:, 0].clamp(eps, 1.0 - eps)   # (B,h_tok,w_tok)

    # Optional: lift crop token map into full assay token coordinates
    # For VQVAE spatial consistency loss, keep full_hw=None so gradients stay
    # in crop coordinates only. Full lifting is mainly for assay-level alignment
    # in spatial-map pretraining / visualization.
    if full_hw is not None:
        tok_support = lift_crop_token_map_to_full(
            crop_tok_map_bhw=tok_support,
            full_hw=full_hw,
            patch_size=patch_size,
            roi_hw=roi_hw,
            pad_hw=pad_hw,
        ).clamp(eps, 1.0 - eps)

    return tok_support


# Pixelwise

def spatial_pixel_map_from_input(x_b1thw: torch.Tensor) -> torch.Tensor:
    """
    x_b1thw: (B,1,T,H,W)
    returns: (B,H,W), binary pixel/site support.
    """
    if x_b1thw.dim() != 5 or x_b1thw.size(1) != 1:
        raise ValueError(f"Expected (B,1,T,H,W), got {tuple(x_b1thw.shape)}")

    return (x_b1thw > 0).any(dim=2).float()[:, 0]


def soft_spatial_pixel_map_from_logits(
    logits_b1thw: torch.Tensor,
    eps: float = 1e-6,
    tau: float = 0.25,
    prob_threshold: Optional[float] = None,
) -> torch.Tensor:
    """
    logits_b1thw: (B,1,T,H,W)
    returns: (B,H,W), soft temporal-union pixel support.
    """
    if logits_b1thw.dim() != 5 or logits_b1thw.size(1) != 1:
        raise ValueError(f"Expected (B,1,T,H,W), got {tuple(logits_b1thw.shape)}")

    p = soft_binary_from_logits(
        logits_b1thw,
        tau=tau,
        prob_threshold=prob_threshold,
        eps=eps,
    ).clamp(eps, 1.0 - eps)
    site_support = 1.0 - torch.prod(1.0 - p, dim=2)
    return site_support[:, 0].clamp(eps, 1.0 - eps)


def dilate_spatial_support_hw(
    support_bhw: torch.Tensor,
    radius_h: int = 1,
    radius_w: int = 1,
) -> torch.Tensor:
    if radius_h <= 0 and radius_w <= 0:
        return support_bhw

    x = support_bhw.unsqueeze(1)
    x = F.max_pool2d(
        x,
        kernel_size=(2 * radius_h + 1, 2 * radius_w + 1),
        stride=1,
        padding=(radius_h, radius_w),
    )
    return x[:, 0]


def lift_crop_pixel_map_to_full(
    crop_pix_map_bhw: torch.Tensor,
    full_hw: tuple[int, int],
    roi_hw,
    pad_hw,
) -> torch.Tensor:
    """
    Lift current crop pixel maps into full assay pixel coordinates.
    """
    B, Hc, Wc = crop_pix_map_bhw.shape
    full_H, full_W = map(int, full_hw)

    out = crop_pix_map_bhw.new_zeros((B, full_H, full_W))

    if roi_hw is None:
        if (Hc, Wc) != (full_H, full_W):
            raise ValueError(
                f"roi_hw is None but crop pixel size {(Hc, Wc)} != full size {(full_H, full_W)}"
            )
        out.copy_(crop_pix_map_bhw)
        return out

    if pad_hw is None:
        pad_hw = [(0, 0, 0, 0)] * B

    for b in range(B):
        y0, x0, y1, x1 = map(int, roi_hw[b])
        ptop, pbot, pleft, pright = map(int, pad_hw[b])

        y_a = ptop
        y_b = Hc - pbot if pbot > 0 else Hc
        x_a = pleft
        x_b = Wc - pright if pright > 0 else Wc

        crop_valid = crop_pix_map_bhw[b, y_a:y_b, x_a:x_b]

        valid_h = min(crop_valid.shape[0], y1 - y0)
        valid_w = min(crop_valid.shape[1], x1 - x0)

        fy0 = max(0, y0)
        fx0 = max(0, x0)
        fy1 = min(full_H, y0 + valid_h)
        fx1 = min(full_W, x0 + valid_w)

        cy0 = fy0 - y0
        cx0 = fx0 - x0
        cy1 = cy0 + (fy1 - fy0)
        cx1 = cx0 + (fx1 - fx0)

        if fy1 > fy0 and fx1 > fx0:
            out[b, fy0:fy1, fx0:fx1] = torch.maximum(
                out[b, fy0:fy1, fx0:fx1],
                crop_valid[cy0:cy1, cx0:cx1],
            )

    return out


def sample_full_pixel_map_to_crop(
    full_pix_bhw: torch.Tensor,
    out_hw: tuple[int, int],
    roi_hw=None,
    pad_hw=None,
) -> torch.Tensor:
    """
    Convert full assay pixel memory map (B,full_H,full_W)
    to current runtime crop/padded coordinates (B,H,W).

    This mirrors the crop/pad convention used by lift_crop_pixel_map_to_full.
    """
    if full_pix_bhw.dim() != 3:
        raise ValueError(f"Expected (B,H,W), got {tuple(full_pix_bhw.shape)}")

    B, full_H, full_W = full_pix_bhw.shape
    out_H, out_W = map(int, out_hw)

    if roi_hw is None:
        if (full_H, full_W) == (out_H, out_W):
            return full_pix_bhw
        raise ValueError(
            f"roi_hw is None but full map size {(full_H, full_W)} != runtime size {(out_H, out_W)}"
        )

    if pad_hw is None:
        pad_hw = [(0, 0, 0, 0)] * B

    crops = []

    for b in range(B):
        y0, x0, y1, x1 = map(int, roi_hw[b])
        ptop, pbot, pleft, pright = map(int, pad_hw[b])

        y0_clip = max(0, y0)
        x0_clip = max(0, x0)
        y1_clip = min(full_H, y1)
        x1_clip = min(full_W, x1)

        crop = full_pix_bhw[b:b+1, None, y0_clip:y1_clip, x0_clip:x1_clip]

        extra_top = y0_clip - y0
        extra_left = x0_clip - x0
        extra_bottom = y1 - y1_clip
        extra_right = x1 - x1_clip

        crop = torch.nn.functional.pad(
            crop,
            (
                pleft + extra_left,
                pright + extra_right,
                ptop + extra_top,
                pbot + extra_bottom,
            ),
            value=0.0,
        )

        if crop.shape[-2:] != (out_H, out_W):
            crop = torch.nn.functional.interpolate(
                crop,
                size=(out_H, out_W),
                mode="nearest",
            )

        crops.append(crop[:, 0])

    return torch.cat(crops, dim=0)


# ---- Local context stuff ----
def temporal_trend_score_np(frame_means: np.ndarray, eps: float = 1e-8) -> float:
    """
    Scale-free temporal trend score in roughly [-1, 1].

    Positive  -> activity tends to increase over time
    Negative  -> activity tends to decrease over time
    Near zero -> no consistent monotonic trend

    Uses correlation with standardized time, so it is:
      - independent of temporal window length
      - independent of H/W, because frame_means is already averaged spatially
      - much less sensitive to absolute firing magnitude
    """
    frame_means = np.asarray(frame_means, dtype=np.float32)
    T = frame_means.shape[0]

    if T <= 1:
        return 0.0

    t = np.arange(T, dtype=np.float32)
    t = (t - t.mean()) / (t.std() + eps)

    fm = frame_means - frame_means.mean()
    fm_std = fm.std()

    if fm_std < eps:
        return 0.0

    return float((fm * t).mean() / (fm_std + eps))


def compute_activity_ctx(thw: np.ndarray, max_num_units: float = 1024.0, eps: float = 1e-8) -> np.ndarray:
    """
    Compute 9D local context from a (T,H,W) binary spike volume.

    Features:
      0) log_mean_firing_density
      1) var_x
      2) var_y
      3) var_t
      4) cov_xy
      5) cov_xt
      6) cov_yt
      7) active_site_ratio
      8) temporal_trend_score
    """
    x = thw.astype(np.float32)
    T, H, W = x.shape

    mean_firing_density = float(x.mean())
    log_mean_firing_density = float(np.clip(np.log(mean_firing_density + 1e-6), -15.0, 0.0))

    mass = float(x.sum())

    if mass <= eps:
        var_x = var_y = var_t = 0.0
        cov_xy = cov_xt = cov_yt = 0.0
    else:
        tt = np.linspace(-1.0, 1.0, T, dtype=np.float32)[:, None, None]
        yy = np.linspace(-1.0, 1.0, H, dtype=np.float32)[None, :, None]
        xx = np.linspace(-1.0, 1.0, W, dtype=np.float32)[None, None, :]

        mx = float((x * xx).sum() / mass)
        my = float((x * yy).sum() / mass)
        mt = float((x * tt).sum() / mass)

        dx = xx - mx
        dy = yy - my
        dt = tt - mt

        var_x = float((x * dx * dx).sum() / mass)
        var_y = float((x * dy * dy).sum() / mass)
        var_t = float((x * dt * dt).sum() / mass)

        cov_xy = float((x * dx * dy).sum() / mass)
        cov_xt = float((x * dx * dt).sum() / mass)
        cov_yt = float((x * dy * dt).sum() / mass)

    denom = float(min(H * W, max_num_units))
    active_site_count = float((x.max(axis=0) > 0).sum())
    active_site_ratio = float(min(active_site_count / max(denom, 1.0), 1.0))

    frame_means = x.reshape(T, -1).mean(axis=1)
    temporal_trend = temporal_trend_score_np(frame_means)

    return np.array([
        log_mean_firing_density,
        var_x,
        var_y,
        var_t,
        cov_xy,
        cov_xt,
        cov_yt,
        active_site_ratio,
        temporal_trend,
    ], dtype=np.float32)

def perturb_local_ctx(
    local_ctx,
    per_dim_std,
    *,
    dims,
    global_ctx=None,
    bank=None,
    assay_idx=None,
    pair_p: float = 0.5,
    sigma_scale: float = 0.25,
    generator=None,
    return_stats: bool = False,
):
    """
    Draw a counterfactual context request.

    Stage 2C's context head cannot learn from a reconstruction objective:
    `local_ctx` is measured from x, and the reconstruction loss already drives
    the output toward x, so argmin(recon) == argmin(ctx_loss) and the context
    term contributes no gradient of its own.  Training it needs a request that
    DISAGREES with the codes; then the only way to satisfy `ctx_loss_soft` is
    to use the control signal.

    WHAT MAKES A REQUEST VALID.  Not "the local context matches the assay" --
    the real constraint is that the (global, local) PAIR is jointly plausible.
    sample_context.py supports both ways of honouring that, and both are used
    here:

      * paired swap (probability `pair_p`): take another assay's global
        context together with one of ITS local contexts.  Both move, the pair
        stays consistent, and the displacement is large -- across assays
        rather than within one.
      * within-assay resample: hold `global_ctx` fixed and draw a different
        local context from the same assay's pool.  Smaller displacement, tests
        fine-grained steering.

    What is never produced is assay A's global paired with assay B's local.

    Drawing real observed vectors rather than perturbing numerically also
    keeps requests on-manifold: the nine dimensions are not independent (the
    second-moment entries must form a valid covariance, density and
    active-site ratio co-vary), so independent Gaussian noise can synthesise
    combinations no clip could exhibit.  Jitter scaled by the within-assay std
    survives only as the fallback for assays with fewer than two entries.

    Args:
      local_ctx:   (B, L) true local context
      per_dim_std: (L,) within-assay per-dimension std, for the fallback
      dims:        iterable of local dimensions currently supervised
      global_ctx:  (B, G) true global context; required for paired swaps
      bank:        {assay_id: {"local": (n_a, L), "global": (G,)}}
      assay_idx:   (B,) assay ids
      pair_p:      probability of a full paired (global, local) swap
      sigma_scale: fallback jitter magnitude, in within-assay std units

    Returns:
      (local_cf, global_cf) or ((local_cf, global_cf), stats).
      global_cf is `global_ctx` unchanged where no paired swap occurred.
      stats["bank_frac"]    fraction of rows served from the bank
      stats["pair_frac"]    fraction that were paired swaps
      stats["displacement"] mean |request - truth| over `dims`, in
                            within-assay std units
    """
    import torch

    if local_ctx.dim() != 2:
        raise ValueError(f"local_ctx must be (B,L), got {tuple(local_ctx.shape)}")

    B, L = local_ctx.shape
    device = local_ctx.device

    dim_index = torch.as_tensor(sorted(set(int(d) for d in dims)), device=device)
    if dim_index.numel() == 0:
        result = (local_ctx.clone(), None if global_ctx is None else global_ctx.clone())
        return (result, {}) if return_stats else result

    std = torch.as_tensor(per_dim_std, device=device, dtype=local_ctx.dtype)
    if std.numel() != L:
        raise ValueError(f"per_dim_std must have {L} entries, got {std.numel()}")

    noise = torch.randn(
        local_ctx.shape, device=device, dtype=local_ctx.dtype, generator=generator
    )
    proposal = local_ctx + float(sigma_scale) * std.view(1, L) * noise
    global_cf = None if global_ctx is None else global_ctx.clone()

    from_bank = torch.zeros((B,), device=device, dtype=torch.bool)
    paired = torch.zeros((B,), device=device, dtype=torch.bool)

    if bank is not None and assay_idx is not None:
        servable = [
            a for a, entry in bank.items() if entry["local"].shape[0] >= 2
        ]
        ids = torch.as_tensor(assay_idx).reshape(-1).tolist()

        def _draw(pool):
            k = int(
                torch.randint(
                    pool.shape[0], (1,), device=pool.device, generator=generator
                )
            )
            return pool[k]

        for row, assay in enumerate(ids):
            want_pair = (
                global_cf is not None
                and len(servable) > 1
                and float(
                    torch.rand((), device=device, generator=generator)
                ) < float(pair_p)
            )

            if want_pair:
                # Another assay entirely: its global and one of its locals.
                choices = [a for a in servable if a != int(assay)] or servable
                pick = choices[
                    int(
                        torch.randint(
                            len(choices), (1,), device=device, generator=generator
                        )
                    )
                ]
                entry = bank[pick]
                proposal[row] = _draw(entry["local"]).to(
                    device=device, dtype=local_ctx.dtype
                )
                global_cf[row] = entry["global"].to(
                    device=device, dtype=global_cf.dtype
                )
                from_bank[row] = True
                paired[row] = True
                continue

            entry = bank.get(int(assay), None)
            if entry is None or entry["local"].shape[0] < 2:
                continue
            proposal[row] = _draw(entry["local"]).to(
                device=device, dtype=local_ctx.dtype
            )
            from_bank[row] = True

    out = local_ctx.clone()
    out[:, dim_index] = proposal[:, dim_index]
    result = (out, global_cf)

    if not return_stats:
        return result

    delta = (out[:, dim_index] - local_ctx[:, dim_index]).abs()
    scale = std[dim_index].clamp_min(1e-8).view(1, -1)
    stats = {
        "bank_frac": float(from_bank.float().mean()),
        "pair_frac": float(paired.float().mean()),
        "displacement": float((delta / scale).mean()),
    }
    return result, stats
