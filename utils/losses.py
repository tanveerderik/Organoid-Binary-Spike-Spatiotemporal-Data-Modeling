#%%
# --------------------------- Losses, Eval, Train ---------------------------
from typing import Sequence, Optional
import math
import torch
import torch.nn.functional as F



# ---------- Main reconstruction loss (weighted logit based BCE, can also take in mask) ----------


def masked_bce_with_logits_weighted(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask_vol: Optional[torch.Tensor] = None,
    pos_weight: Optional[float | torch.Tensor | str] = None,
    fp_weight: float = 0.01,
    fp_margin: float = 0.01,
):
    """
    Weighted BCE over the optional voxel mask, with optional hallucination penalty.

    Args:
        logits, target, mask_vol: (B,1,T,H,W)
        pos_weight: None | "auto" | float | Tensor broadcastable per-sample
        fp_weight: weight of false-positive penalty on target==0 voxels
        fp_margin: only penalize background probs above this margin

    Returns:
        scalar loss
    """
    B = logits.shape[0]
    dtype = logits.dtype
    device = logits.device

    if mask_vol is None:
        mask_bool = torch.ones_like(logits, dtype=torch.bool, device=device)
    else:
        mask_bool = (mask_vol > 0)

    mask_f = mask_bool.to(dtype=dtype)

    with torch.no_grad():
        tgt_masked = target * mask_f
        pos = tgt_masked.sum(dim=(1, 2, 3, 4))            # (B,)
        tot = mask_f.sum(dim=(1, 2, 3, 4)).clamp(min=1.0) # (B,)
        neg = (tot - pos).clamp(min=0.0)                  # (B,)

        if pos_weight == "auto":
            pw = (neg / pos.clamp(min=1.0)).to(dtype=dtype)
            pw = pw.clamp(max=100.0)
        elif isinstance(pos_weight, (float, int)):
            pw = torch.full((B,), float(pos_weight), dtype=dtype, device=device)
        elif torch.is_tensor(pos_weight):
            pw = pos_weight.to(device=device, dtype=dtype)
            if pw.ndim == 0:
                pw = pw.expand(B)
        else:
            pw = None

    logits_f = logits[mask_bool]
    target_f = target[mask_bool]

    if pw is not None:
        with torch.no_grad():
            mask_counts = mask_bool.view(B, -1).sum(dim=1).to(torch.long)
            idxs = torch.repeat_interleave(torch.arange(B, device=device), mask_counts)
            pos_weight_flat = pw[idxs]

        bce_sum = F.binary_cross_entropy_with_logits(
            logits_f,
            target_f,
            pos_weight=pos_weight_flat,
            reduction="sum",
        )
        bce_denom = (neg + pw * pos).sum().clamp(min=1.0)
    else:
        bce_sum = F.binary_cross_entropy_with_logits(
            logits_f,
            target_f,
            reduction="sum",
        )
        bce_denom = mask_f.sum().clamp(min=1.0)

    bce_loss = bce_sum / bce_denom

    fp_loss = torch.zeros((), dtype=dtype, device=device)
    if fp_weight > 0.0:
        neg_mask_f = (target_f < 0.5)
        if neg_mask_f.any():
            logits_neg = logits_f[neg_mask_f]
            p_neg = torch.sigmoid(logits_neg)
            if fp_margin > 0.0:
                p_neg = F.relu(p_neg - fp_margin)
            fp_loss = -torch.log(1.0 - p_neg + 1e-8).mean()

    return bce_loss + fp_weight * fp_loss


def tolerant_spike_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask_vol: Optional[torch.Tensor] = None,
    pos_weight: Optional[float | torch.Tensor | str] = None,
    radius_t: int = 0,
    radius_h: int = 1,
    radius_w: int = 1,
    alpha_exact: float = 0.35,
    beta_hit: float = 0.55,
    gamma_peak: float = 0.05,
    delta_multi: float = 0.10,
    peak_margin: float = 0.5,
    multi_margin: float = 1.0,
    max_peak_sites: Optional[int] = None,
    max_multi_sites: Optional[int] = None,
    eps: float = 1e-6,
    return_parts: bool = False,
):
    """
    Tolerant reconstruction loss for sparse binary spikes.

    Terms:
      - exact BCE: preserves exact supervision
      - tolerant hit: allows small local misalignment
      - peak term: exact positive should beat its neighbors, preventing plateaus
      - multi term: penalizes excess local mass around a target spike,
                    discouraging several nearby spikes for one true event
    """
    if logits.shape != target.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} != target shape {tuple(target.shape)}")
    if logits.dim() != 5 or logits.size(1) != 1:
        raise ValueError(f"Expected logits of shape (B,1,T,H,W), got {tuple(logits.shape)}")

    if mask_vol is None:
        mask_bool = torch.ones_like(target, dtype=torch.bool)
    else:
        mask_bool = (mask_vol > 0)

    # 1) Exact supervision
    loss_exact = masked_bce_with_logits_weighted(
        logits=logits,
        target=target,
        mask_vol=mask_vol,
        pos_weight=pos_weight,
        fp_weight=0.0,   # you may later turn this on mildly if needed
    )

    # neighborhood size
    kT = 2 * radius_t + 1
    kH = 2 * radius_h + 1
    kW = 2 * radius_w + 1

    probs = torch.sigmoid(logits).clamp(eps, 1.0 - eps)

    # 2) Tolerant local-hit supervision
    probs_local_max = F.max_pool3d(
        probs,
        kernel_size=(kT, kH, kW),
        stride=1,
        padding=(radius_t, radius_h, radius_w),
    )

    pos_mask = (target > 0.5) & mask_bool
    if pos_mask.any():
        loss_hit = -torch.log(probs_local_max[pos_mask]).mean()
    else:
        loss_hit = logits.new_zeros(())

    # 3) Peakness term: center should beat neighbors EXCLUDING center
    loss_peak = logits.new_zeros(())
    if gamma_peak > 0.0 and pos_mask.any():
        pos_idx = pos_mask.nonzero(as_tuple=False)  # (M,5)

        if max_peak_sites is not None and pos_idx.size(0) > max_peak_sites:
            perm = torch.randperm(pos_idx.size(0), device=pos_idx.device)[:max_peak_sites]
            pos_idx = pos_idx[perm]

        B, _, T, H, W = logits.shape
        peak_terms = []

        for idx in pos_idx:
            b, _, t, h, w = idx.tolist()

            t0 = max(0, t - radius_t)
            t1 = min(T, t + radius_t + 1)
            h0 = max(0, h - radius_h)
            h1 = min(H, h + radius_h + 1)
            w0 = max(0, w - radius_w)
            w1 = min(W, w + radius_w + 1)

            patch_logits = logits[b, 0, t0:t1, h0:h1, w0:w1]
            patch_valid = mask_bool[b, 0, t0:t1, h0:h1, w0:w1]

            ct = t - t0
            ch = h - h0
            cw = w - w0

            neigh_valid = patch_valid.clone()
            neigh_valid[ct, ch, cw] = False

            neigh_logits = patch_logits[neigh_valid]
            if neigh_logits.numel() == 0:
                continue

            center_logit = logits[b, 0, t, h, w]
            max_neighbor = neigh_logits.max()

            peak_terms.append(F.relu(max_neighbor - center_logit + peak_margin))

        if len(peak_terms) > 0:
            loss_peak = torch.stack(peak_terms).mean()

    # 4) NEW: prevent multiple nearby spikes
    # Sum probability mass in the tolerance neighborhood around each true spike.
    # Penalize only the excess above multi_margin (~1 spike).
    loss_multi = logits.new_zeros(())
    if delta_multi > 0.0 and pos_mask.any():
        pos_idx = pos_mask.nonzero(as_tuple=False)

        if max_multi_sites is not None and pos_idx.size(0) > max_multi_sites:
            perm = torch.randperm(pos_idx.size(0), device=pos_idx.device)[:max_multi_sites]
            pos_idx = pos_idx[perm]

        B, _, T, H, W = logits.shape
        multi_terms = []

        for idx in pos_idx:
            b, _, t, h, w = idx.tolist()

            t0 = max(0, t - radius_t)
            t1 = min(T, t + radius_t + 1)
            h0 = max(0, h - radius_h)
            h1 = min(H, h + radius_h + 1)
            w0 = max(0, w - radius_w)
            w1 = min(W, w + radius_w + 1)

            patch_probs = probs[b, 0, t0:t1, h0:h1, w0:w1]
            patch_valid = mask_bool[b, 0, t0:t1, h0:h1, w0:w1]

            # only count valid region
            local_mass = patch_probs[patch_valid].sum()

            # allow about one spike worth of mass; penalize extra
            multi_terms.append(F.relu(local_mass - multi_margin))

        if len(multi_terms) > 0:
            loss_multi = torch.stack(multi_terms).mean()


    total = (
        alpha_exact * loss_exact
        + beta_hit * loss_hit
        + gamma_peak * loss_peak
        + delta_multi * loss_multi
    )

    if not return_parts:
        return total

    parts = {
        "total": total,
        "loss_exact": loss_exact,
        "loss_hit": loss_hit,
        "loss_peak": loss_peak,
        "loss_multi": loss_multi,
        "weighted_exact": alpha_exact * loss_exact,
        "weighted_tol": (
            beta_hit * loss_hit
            + gamma_peak * loss_peak
            + delta_multi * loss_multi
        ),
        "alpha_exact": float(alpha_exact),
        "beta_hit": float(beta_hit),
        "gamma_peak": float(gamma_peak),
        "delta_multi": float(delta_multi),
    }
    return parts



# --- Encoder and codebook stability ---

def variance_floor_loss(x, target_std=0.25, eps=1e-4):
    if x.numel() == 0 or x.shape[0] < 2:
        return x.new_zeros(())
    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    return torch.relu(target_std - std).mean()


def soft_code_usage_loss(
    code_logits: torch.Tensor,   # (B, N, K)
    active_mask: torch.Tensor,   # (B, N) bool
    tau: float = 0.5,
    eps: float = 1e-8,
):
    """
    Differentiable batch-level usage balancing loss.

    Encourages the soft average code usage across active tokens
    to be closer to uniform, without changing hard quantization.
    """
    B, N, K = code_logits.shape

    act = active_mask.reshape(B * N)
    logits_act = code_logits.reshape(B * N, K)[act]   # (M, K)

    if logits_act.numel() == 0:
        return code_logits.new_zeros(()), {
            "soft_usage_entropy": 0.0,
            "soft_usage_perplexity": 1.0,
        }

    # Soft assignments over codes
    probs = F.softmax(logits_act / tau, dim=-1)       # (M, K)

    # Average usage over active tokens
    usage = probs.mean(dim=0)                         # (K,)
    usage = usage / usage.sum().clamp_min(eps)

    # Max entropy = uniform usage
    entropy = -(usage * (usage + eps).log()).sum()
    max_entropy = torch.log(torch.tensor(float(K), device=usage.device, dtype=usage.dtype))

    # Minimize this
    loss = (max_entropy - entropy)

    aux = {
        "soft_usage_entropy": float(entropy.detach().item()),
        "soft_usage_perplexity": float(entropy.detach().exp().item()),
    }
    return loss, aux

# -------- blank patch enforcing loss for proper blank embedding ----------

def blank_patch_logit_hinge_loss(
    pred_patches_raw,
    blank_mask,
    margin: float = -6.0,
    detach_mask: bool = True,
):
    """
    Enforce blank-routed patches to decode as near-zero.

    Args:
        pred_patches_raw: (B, N, P) raw renderer logits before output biases
        blank_mask: (B, N) bool, True = blank patch
        margin: desired upper bound for blank logits.
                margin=-6 means sigmoid(logit) ~= 0.0025.
    """
    if blank_mask is None:
        return pred_patches_raw.new_zeros(())

    if detach_mask:
        blank_mask = blank_mask.detach()

    blank_mask = blank_mask.to(device=pred_patches_raw.device, dtype=torch.bool)

    if blank_mask.sum() == 0:
        return pred_patches_raw.new_zeros(())

    blank_logits = pred_patches_raw[blank_mask]  # (num_blank, P)

    # penalize only logits above margin
    loss = torch.nn.functional.softplus(blank_logits - margin).mean()

    return loss

# -------- blank-active decoder latent separation ----------

def blank_active_decoder_separation_loss(
    z_blank_dec,
    z_active_dec,
    margin: float = 0.0,
):
    """
    Encourage decoder-side blank representation to be separated from
    active VQ representations.

    z_blank_dec:  (1, D) or (B, 1, D)
    z_active_dec: (M, D) active decoder-side latents
    margin: cosine upper bound. margin=0 means orthogonal-or-less.
    """

    if z_active_dec is None or z_active_dec.numel() == 0:
        return z_blank_dec.new_tensor(0.0)

    z_blank = z_blank_dec.reshape(1, -1)
    z_active = z_active_dec.reshape(-1, z_blank.shape[-1])

    z_blank = F.normalize(z_blank, dim=-1)
    z_active = F.normalize(z_active, dim=-1)

    cos = (z_active * z_blank).sum(dim=-1)

    return F.relu(cos - margin).mean()

# ---------- ISI / refractory constraint helpers ----------
def short_gap_excess_loss_from_logits_batch_targets(
    logits_b1thw,
    target_gap_rates_bg,
    max_gap=3,
    tau=0.25,
    margin=0.25,
    confidence_bg=None,
    eps=1e-8,
    return_parts=False,
):
    """
    Batchwise one-sided conditional short-gap excess loss.

    pred_rate[b,g] =
        sum p_b(t)*p_b(t+g) / sum p_b(t)

    allowed[b,g] =
        (1 + margin) * target_gap_rates_bg[b,g]
    """
    assert logits_b1thw.dim() == 5 and logits_b1thw.size(1) == 1

    logits_b1thw = logits_b1thw.float().clamp(-20.0, 20.0)
    p = torch.sigmoid(logits_b1thw / tau)

    target_gap_rates_bg = target_gap_rates_bg.to(
        device=logits_b1thw.device,
        dtype=logits_b1thw.dtype,
    )

    B = p.shape[0]
    rates = []

    for g in range(1, max_gap + 1):
        if p.shape[2] <= g:
            rates.append(p.new_zeros((B,)))
        else:
            p0 = p[:, :, :-g]
            pg = p[:, :, g:]

            num = (p0 * pg).sum(dim=(1, 2, 3, 4))
            den = p0.sum(dim=(1, 2, 3, 4)).clamp_min(eps)

            rates.append(num / den)

    pred_rates = torch.stack(rates, dim=1)  # (B,G)

    tgt_rates = target_gap_rates_bg[:, :max_gap]
    allowed = (1.0 + margin) * tgt_rates

    excess = torch.relu(pred_rates - allowed).pow(2)

    if confidence_bg is not None:
        conf = confidence_bg[:, :max_gap].to(device=excess.device, dtype=excess.dtype)
        excess = excess * conf
        denom = conf.sum().clamp_min(1.0)
        loss = excess.sum() / denom
    else:
        loss = excess.mean()

    if return_parts:
        return {
            "loss": loss,
            "pred_gap_rates": pred_rates.detach(),
            "allowed_gap_rates": allowed.detach(),
            "target_gap_rates": tgt_rates.detach(),
            "confidence": None if confidence_bg is None else confidence_bg.detach(),
        }

    return loss


# ---------- Local context losses ----------

import torch

def soft_active_site_ratio_from_logits(
    logits_b1thw: torch.Tensor,
    tau_prob: float = 0.25,
    tau_time: float = 0.5,
    site_threshold: float = 0.0,
    max_active_site: float = 1024.0,
    clamp_max: bool = True,
) -> torch.Tensor:
    """
    logits_b1thw: (B,1,T,H,W)

    Returns:
      (B,) soft active-site ratio normalized by effective max active sites,
      matching the hard version based on active-site COUNT / max_active_site.

    Notes:
    - soft temporal OR is approximated by logsumexp over time in logit space
    - site_active is a soft indicator in [0,1] for whether each spatial site
      was active at least once
    - final ratio is SUM over sites divided by min(H*W, max_active_site)
    """

    if logits_b1thw.dim() != 5:
        raise ValueError(f"Expected (B,1,T,H,W), got {tuple(logits_b1thw.shape)}")

    B, C, T, H, W = logits_b1thw.shape
    if C != 1:
        raise ValueError(f"Expected channel dim = 1, got {C}")

    # soft temporal max in logit space
    site_logit = tau_time * torch.logsumexp(
        logits_b1thw / max(tau_time, 1e-6),
        dim=2
    )  # (B,1,H,W)

    # soft site activation indicator
    site_active = torch.sigmoid(
        (site_logit - site_threshold) / max(tau_prob, 1e-6)
    )  # (B,1,H,W)

    # soft active-site count
    soft_active_count = site_active.sum(dim=(1, 2, 3))  # (B,)

    # match hard-version denominator
    denom = float(min(H * W, max_active_site))
    d3 = soft_active_count / max(denom, 1.0)

    if clamp_max:
        d3 = d3.clamp(max=1.0)

    return d3

def temporal_trend_score_torch(
    frame_mean_bt: torch.Tensor,
    eps_time: float = 1e-6,
) -> torch.Tensor:
    """
    frame_mean_bt: (B,T)

    Returns a scale-free trend score in roughly [-1, 1].
    """
    if frame_mean_bt.dim() != 2:
        raise ValueError(f"Expected (B,T), got {tuple(frame_mean_bt.shape)}")

    B, T = frame_mean_bt.shape
    if T <= 1:
        return frame_mean_bt.new_zeros((B,))

    t = torch.arange(T, device=frame_mean_bt.device, dtype=frame_mean_bt.dtype)
    t = (t - t.mean()) / (t.std(unbiased=False) + eps_time)

    fm = frame_mean_bt - frame_mean_bt.mean(dim=1, keepdim=True)
    fm_std = fm.std(dim=1, unbiased=False)

    corr = (fm * t.unsqueeze(0)).mean(dim=1) / (fm_std + eps_time)

    # avoid noisy explosion on nearly-flat clips
    corr = torch.where(fm_std > eps_time, corr, torch.zeros_like(corr))
    return corr


def ctx_features_soft_from_logits(
    logits_b1thw: torch.Tensor,
    eps_prob: float = 1e-4,
    eps_time: float = 1e-6,
    tau: float = 0.25,
) -> torch.Tensor:
    """
    returns: (B,5) =
      [mean_firing_density, frame_mean_std, pixel_mean_std, active_site_ratio, temporal_trend_score]
    """
    if logits_b1thw.dim() != 5:
        raise ValueError(f"Expected (B,1,T,H,W), got {tuple(logits_b1thw.shape)}")

    B, C, T, H, W = logits_b1thw.shape
    if C != 1:
        raise ValueError(f"Expected channel=1, got {C}")

    p = torch.sigmoid(logits_b1thw / max(1e-6, float(tau))).clamp(eps_prob, 1.0 - eps_prob)

    # 0) mean firing density (log)
    mfd = p.mean(dim=(1, 2, 3, 4))
    d0 = torch.log(mfd + 1e-6).clamp(min=-15.0, max=0.0)

    # 1) frame-wise temporal variability
    frame_mean = p.mean(dim=(1, 3, 4))  # (B,T)
    d1 = frame_mean.std(dim=1, unbiased=False) * math.sqrt(float(H * W))

    # 2) pixel-wise spatial variability
    pixel_mean = p.mean(dim=2).squeeze(1)  # (B,H,W)
    d2 = pixel_mean.std(dim=(1, 2), unbiased=False) * math.sqrt(float(T))

    # 3) active site ratio
    d3 = soft_active_site_ratio_from_logits(
        logits_b1thw,
        tau_prob=tau,
        tau_time=0.5,
        site_threshold=0.05,
    )

    # 4) scale-free temporal trend score
    d4 = temporal_trend_score_torch(frame_mean, eps_time=eps_time)

    return torch.stack([d0, d1, d2, d3, d4], dim=1)



def ctx_loss_soft(
    logits_b1thw: torch.Tensor,
    ctx_tgt_b5: torch.Tensor,
    dims: Sequence[int] = (0, 1, 2, 3, 4),
    weights: Optional[Sequence[float]] = (1.0, 1.0, 1.0, 1.0, 1.0),
    tau: float = 0.25,
) -> torch.Tensor:

    if ctx_tgt_b5.dim() != 2 or ctx_tgt_b5.size(1) < 5:
        raise ValueError(f"ctx_tgt_b5 must be (B,>=5), got {tuple(ctx_tgt_b5.shape)}")

    pred5 = ctx_features_soft_from_logits(logits_b1thw.float(), tau=tau)

    if weights is None:
        weight_map = {d: 1.0 for d in range(5)}
    else:
        if len(weights) != 5:
            raise ValueError("weights must have length 5")
        weight_map = {d: float(weights[d]) for d in range(5)}

    loss = pred5.new_tensor(0.0)

    for d in dims:
        if d < 0 or d > 4:
            raise ValueError(f"Invalid dim {d}")

        pred = pred5[:, d]
        tgt = ctx_tgt_b5[:, d].to(pred5.dtype)

        # L2 loss for all dims
        diff = pred - tgt
        term = diff.pow(2).mean()

        loss = loss + weight_map[d] * term

    active_weight_sum = sum(weight_map[d] for d in dims)
    return loss / max(1e-8, active_weight_sum)



# --- Spatial consistency (loss indirectly dependent on the global context) ----

def spatial_map_union_sparse_loss(
    support_map_bhw: torch.Tensor,
    target_tok_map_bhw: torch.Tensor,
    lambda_size: float = 0.01,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    support_map_bhw:    (B,h,w) soft support map in [0,1]
    target_tok_map_bhw: (B,h,w) binary target token map

    Goal:
      - cover all target-positive locations  -> union behavior
      - avoid trivial all-ones support       -> weak global size penalty
    """
    if support_map_bhw.shape != target_tok_map_bhw.shape:
        raise ValueError(
            f"Shape mismatch: {tuple(support_map_bhw.shape)} vs {tuple(target_tok_map_bhw.shape)}"
        )

    tgt = target_tok_map_bhw.float()
    pred = support_map_bhw.clamp(eps, 1.0 - eps)

    pos_count = tgt.sum().clamp_min(1.0)
    loss_cover = (-(pred.log()) * tgt).sum() / pos_count

    loss_size = pred.mean()

    return loss_cover + lambda_size * loss_size




def spatial_support_violation_loss(
    pred_support: torch.Tensor,
    allowed_support: torch.Tensor,
    eps: float = 1e-6,
    neg_thresh: float = 0.20,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Penalize predicted spatial support only in regions that are NOT allowed.

    Intended use:
      - VQVAE spatial consistency / suppression during reconstruction training
      - NOT for spatial map pretraining

    Args:
      pred_support:
        Soft student support map in [0, 1], shape (B,H,W) or (B,h,w).

      allowed_support:
        Soft teacher/allowed-support map in [0, 1], same shape as pred_support.
        High values mean the region is allowed.
        Low values mean the region is unsupported / forbidden.

      eps:
        Numerical stability.

      neg_thresh:
        Regions with allowed_support < neg_thresh are treated as forbidden.

      reduction:
        "mean" or "sum".

    Returns:
      Scalar loss.
    """
    if pred_support.shape != allowed_support.shape:
        raise ValueError(
            f"Shape mismatch: pred_support {tuple(pred_support.shape)} "
            f"vs allowed_support {tuple(allowed_support.shape)}"
        )

    pred = pred_support.clamp(min=eps, max=1.0 - eps)
    allow = allowed_support.clamp(min=0.0, max=1.0)

    # Forbidden regions only
    forbidden_mask = (allow < neg_thresh).to(pred.dtype)

    # Penalize support in forbidden regions
    # Large penalty when pred -> 1 in forbidden area
    per_pixel = -torch.log(1.0 - pred) * forbidden_mask

    denom = forbidden_mask.sum().clamp_min(1.0)

    if reduction == "sum":
        return per_pixel.sum()
    if reduction == "mean":
        return per_pixel.sum() / denom

    raise ValueError(f"Unsupported reduction: {reduction}")



def spatial_support_separation_loss(
    pred_support_map_bhw: torch.Tensor,
    target_support_map_bhw: torch.Tensor,
    ctx_key: torch.Tensor | None = None,
    round_decimals: int = 6,
    eps: float = 1e-8,
    mode: str = "margin",
    margin: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Target-aware separation for spatial support maps.

    - Samples sharing the same rounded ctx_key are grouped together.
    - Predicted maps are encouraged to stay similar when their grouped target
      memory maps are similar, and to separate only when grouped target maps
      are dissimilar.

    Parameters
    ----------
    pred_support_map_bhw : (B,H,W)
        Predicted full support maps in [0,1].
    target_support_map_bhw : (B,H,W)
        Target full support maps in [0,1], ideally from the memory bank.
    ctx_key : (B,D) or None
        Stable global context vectors for grouping.
    round_decimals : int
        Rounding used to define equality of ctx keys.
    eps : float
        Numerical stability constant.
    mode : str
        "match"  -> match pairwise cosine similarity structure
        "margin" -> repel only pairs whose target maps are dissimilar
    margin : float
        Similarity tolerance used when mode == "margin".

    Returns
    -------
    loss_raw : torch.Tensor
        Separation loss.
    sim_pred_mean : torch.Tensor
        Mean predicted cosine similarity over valid unordered pairs.
    num_pairs : int
        Number of compared unordered context pairs.
    """
    if pred_support_map_bhw.dim() != 3:
        raise ValueError(
            f"pred_support_map_bhw must have shape (B,H,W), got {tuple(pred_support_map_bhw.shape)}"
        )
    if target_support_map_bhw.dim() != 3:
        raise ValueError(
            f"target_support_map_bhw must have shape (B,H,W), got {tuple(target_support_map_bhw.shape)}"
        )
    if pred_support_map_bhw.shape != target_support_map_bhw.shape:
        raise ValueError(
            f"Shape mismatch: {tuple(pred_support_map_bhw.shape)} vs {tuple(target_support_map_bhw.shape)}"
        )

    B = pred_support_map_bhw.size(0)
    device = pred_support_map_bhw.device
    dtype = pred_support_map_bhw.dtype

    if B <= 1:
        z = torch.zeros((), device=device, dtype=dtype)
        return z, z, 0

    def _group_mean(map_bhw: torch.Tensor, key: torch.Tensor | None) -> torch.Tensor:
        flat = map_bhw.flatten(1)  # (B,HW)

        if key is None:
            return flat

        if key.dim() != 2 or key.size(0) != B:
            raise ValueError(
                f"ctx_key must have shape (B,D), got {tuple(key.shape)} for B={B}"
            )

        scale = float(10 ** round_decimals)
        keyq = torch.round(key.to(device) * scale).to(torch.int64)

        _, inverse = torch.unique(keyq, dim=0, return_inverse=True)
        K = int(inverse.max().item()) + 1

        if K <= 1:
            return flat.new_zeros((1, flat.size(1)))

        grouped = torch.zeros((K, flat.size(1)), device=device, dtype=dtype)
        grouped.index_add_(0, inverse, flat)

        counts = torch.bincount(inverse, minlength=K).to(device=device, dtype=dtype)
        grouped = grouped / counts.unsqueeze(1).clamp_min(1.0)
        return grouped

    pred_group = _group_mean(pred_support_map_bhw, ctx_key)    # (N,HW)
    tgt_group  = _group_mean(target_support_map_bhw, ctx_key)  # (N,HW)

    N = pred_group.size(0)
    if N <= 1:
        z = torch.zeros((), device=device, dtype=dtype)
        return z, z, 0

    def _cosine_sim(x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=1, keepdim=True)
        x = F.normalize(x, p=2, dim=1, eps=eps)
        return x @ x.t()

    sim_pred = _cosine_sim(pred_group)
    sim_tgt  = _cosine_sim(tgt_group)

    upper = torch.triu(torch.ones((N, N), device=device, dtype=torch.bool), diagonal=1)
    if not upper.any():
        z = torch.zeros((), device=device, dtype=dtype)
        return z, z, 0

    sim_pred_valid = sim_pred[upper]
    sim_tgt_valid  = sim_tgt[upper]

    if mode == "match":
        # exact pairwise-geometry matching
        loss_raw = (sim_pred_valid - sim_tgt_valid).pow(2).mean()
    elif mode == "margin":
        # repel only where targets are dissimilar
        pair_weight = (1.0 - sim_tgt_valid).clamp_min(0.0)
        loss_raw = (pair_weight * F.relu(sim_pred_valid - margin).pow(2)).mean()
    else:
        raise ValueError(f"Unknown mode={mode!r}; expected 'match' or 'margin'")

    sim_pred_mean = sim_pred_valid.mean()
    num_pairs = int(upper.sum().item())

    return loss_raw, sim_pred_mean, num_pairs

# --- Codebook separation loss --- 
def codebook_separation_loss(codebook, eps=1e-8):
    # codebook: (K, D)
    w = F.normalize(codebook, dim=1, eps=eps)
    sim = w @ w.t()  # (K,K)

    K = sim.size(0)
    eye = torch.eye(K, device=sim.device, dtype=sim.dtype)
    offdiag = sim * (1.0 - eye)

    return offdiag.pow(2).sum() / (K * (K - 1))