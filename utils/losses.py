#%%
# --------------------------- Losses, Eval, Train ---------------------------
from typing import Sequence, Optional
import math
import torch
import torch.nn.functional as F

from .constants import (
    normalize_gap_bins,
    max_gap_from_bins,
)

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


def soft_code_norm_ceiling_loss(
    x: torch.Tensor,
    rms_ceiling: float = 10.0,
    token_ceiling: float = 14.0,
    huber_beta: float = 0.10,
    tail_weight: float = 0.25,
    eps: float = 1e-8,
    return_parts: bool = False,
):
    """
    Stabilize code-space scale without forcing every token onto one sphere.

    The RMS term controls the aggregate scale of the active-token population.
    The tail term prevents a small number of tokens from becoming extreme.

    Both are one-sided: vectors below the ceilings receive no penalty.
    """
    if x.numel() == 0:
        zero = x.new_zeros(())
        if return_parts:
            return zero, {
                "rms": zero,
                "mean": zero,
                "std": zero,
                "p95": zero,
                "max": zero,
                "tail_fraction": zero,
                "rms_loss": zero,
                "tail_loss": zero,
            }
        return zero

    norms = x.float().norm(dim=-1)  # (M,)

    rms = torch.sqrt(norms.square().mean() + eps)

    # Dimensionless excesses make the loss less dependent on code_dim.
    rms_excess = torch.relu(
        rms / float(rms_ceiling) - 1.0
    )

    token_excess = torch.relu(
        norms / float(token_ceiling) - 1.0
    )

    # Huber growth is quadratic near the boundary but only linear for
    # extremely large norms. This avoids an enormous loss if a transient
    # instability produces norms in the hundreds or thousands.
    loss_rms = F.smooth_l1_loss(
        rms_excess,
        torch.zeros_like(rms_excess),
        beta=float(huber_beta),
    )

    loss_tail = F.smooth_l1_loss(
        token_excess,
        torch.zeros_like(token_excess),
        beta=float(huber_beta),
    )

    loss = loss_rms + float(tail_weight) * loss_tail

    if not return_parts:
        return loss.to(dtype=x.dtype)

    p95 = torch.quantile(norms.detach(), 0.95)

    return loss.to(dtype=x.dtype), {
        "rms": rms.detach(),
        "mean": norms.mean().detach(),
        "std": norms.std(unbiased=False).detach(),
        "p95": p95,
        "max": norms.max().detach(),
        "tail_fraction": (
            norms > float(token_ceiling)
        ).float().mean().detach(),
        "rms_loss": loss_rms.detach(),
        "tail_loss": loss_tail.detach(),
    }

def encoder_isotropy_loss(
    x,
    target_std=0.10,
    mean_weight=1e-4,
    cov_weight=1e-4,
    cov_margin=0.50,
    eps=1e-4,
    max_tokens=512,
    return_parts=False,
):
    if x.numel() == 0 or x.shape[0] < 2:
        zero = x.new_zeros(())
        if return_parts:
            return zero, {"mean": zero, "var": zero, "cov": zero}
        return zero

    # Subsample active tokens for speed
    if x.shape[0] > max_tokens:
        idx = torch.randperm(x.shape[0], device=x.device)[:max_tokens]
        # idx = torch.randint(
        #     low=0,
        #     high=x.shape[0],
        #     size=(max_tokens,),
        #     device=x.device,
        # )
        x = x[idx]

    x = x.float()

    mean = x.mean(dim=0, keepdim=True)
    loss_mean = mean.pow(2).mean()

    xc = x - mean

    std = torch.sqrt(xc.var(dim=0, unbiased=False) + eps)
    loss_var = torch.relu(target_std - std).mean()

    x_norm = xc / (std.unsqueeze(0) + eps)

    corr = (x_norm.T @ x_norm) / max(1, x_norm.shape[0])

    # Only off-diagonal correlation.
    # Do not force correlations to zero; only penalize excessive correlation.
    d = corr.shape[0]
    eye = torch.eye(d, device=corr.device, dtype=torch.bool)
    offdiag = corr.masked_select(~eye)

    loss_cov = torch.relu(offdiag.abs() - cov_margin).pow(2).mean()

    loss = loss_var + mean_weight * loss_mean + cov_weight * loss_cov

    if return_parts:
        return loss, {
            "mean": loss_mean.detach(),
            "var": loss_var.detach(),
            "cov": loss_cov.detach(),
        }

    return loss

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
    margin: float = -9.0,
    tau: float = 0.25,
    sharpness: float = 10.0,
    bad_buffer: float = 0.0,
    detach_mask: bool = True,
):
    import torch
    import torch.nn.functional as F

    if blank_mask is None:
        return pred_patches_raw.new_zeros(())

    if detach_mask:
        blank_mask = blank_mask.detach()

    blank_mask = blank_mask.to(device=pred_patches_raw.device, dtype=torch.bool)

    if blank_mask.sum() == 0:
        return pred_patches_raw.new_zeros(())

    blank_logits = pred_patches_raw[blank_mask]  # (M_blank, P)

    patch_score = tau * torch.logsumexp(blank_logits / tau, dim=1)

    bad = patch_score > (margin + bad_buffer)

    if bad.sum() == 0:
        return pred_patches_raw.new_zeros(())

    bad_scores = patch_score[bad]

    loss = F.softplus(sharpness * (bad_scores - margin)) / sharpness
    return loss.mean()

# -------- blank-active decoder latent separation ----------

def blank_active_decoder_separation_loss(
    z_blank_base: torch.Tensor,        # (1,D) or (Nb,D), detached outside or inside
    z_active_base: torch.Tensor,       # (Na,D), detached outside or inside
    offset: torch.Tensor,              # (D,)
    margin: float = 1.0,
    offset_scale: float = 0.5,
    norm_reg_weight: float = 1e-5,
    detach_base: bool = True,
):
    """
    Separation loss where only the signed decoder-side offset is intended to
    carry the separation pressure.

    blank  = base_blank  - offset_scale * offset
    active = base_active + offset_scale * offset
    """

    if detach_base:
        z_blank_base = z_blank_base.detach()
        z_active_base = z_active_base.detach()

    if z_blank_base.dim() == 1:
        z_blank_base = z_blank_base.view(1, -1)
    if z_active_base.dim() == 1:
        z_active_base = z_active_base.view(1, -1)

    offset = offset.view(1, -1).to(
        device=z_blank_base.device,
        dtype=z_blank_base.dtype,
    )

    z_blank = z_blank_base - offset_scale * offset
    z_active = z_active_base + offset_scale * offset

    # Pairwise blank-active distance.
    dist = torch.cdist(z_blank.float(), z_active.float(), p=2)

    sep_loss = torch.relu(float(margin) - dist).mean()

    # Weak norm regularization so the offset does not solve everything by exploding.
    offset_norm_reg = offset.float().pow(2).mean()

    total = sep_loss + float(norm_reg_weight) * offset_norm_reg

    return {
        "total": total,
        "sep": sep_loss,
        "offset_norm_reg": offset_norm_reg,
        "mean_dist": dist.detach().mean(),
        "min_dist": dist.detach().min(),
        "offset_norm": offset.detach().float().norm(),
    }


# ---------- Threshold-aware soft binary approximation ----------

def soft_binary_from_logits(
    logits: torch.Tensor,
    *,
    tau: float = 0.25,
    prob_threshold: Optional[float] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Smooth approximation of:

        sigmoid(logits) >= prob_threshold

    If prob_threshold is None, preserve the historical 0.5 boundary.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")

    threshold_value = (
        0.5
        if prob_threshold is None
        else float(prob_threshold)
    )

    threshold = torch.as_tensor(
        threshold_value,
        device=logits.device,
        dtype=logits.dtype,
    ).clamp(eps, 1.0 - eps)

    threshold_logit = torch.logit(threshold)

    return torch.sigmoid(
        (logits - threshold_logit) / float(tau)
    )


# ---------- ISI / refractory constraint helpers ----------
def short_gap_excess_loss_from_logits_batch_targets(
    logits_b1thw,
    target_gap_rates_bg,
    max_gap=3,
    gap_bins=None,
    tau=0.25,
    margin=0.25,
    lower_margin=0.20,
    lower_weight=0.50,
    prob_threshold=None,
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

    p = soft_binary_from_logits(
        logits_b1thw,
        tau=tau,
        prob_threshold=prob_threshold,
        eps=eps,
    )

    target_gap_rates_bg = target_gap_rates_bg.to(
        device=logits_b1thw.device,
        dtype=logits_b1thw.dtype,
    )

    B = p.shape[0]
    rates = []
    
    if gap_bins is None:
        if max_gap is None:
            raise ValueError(
                "Either gap_bins or max_gap must be provided."
            )
    
        gap_bins = tuple(
            (gap, gap)
            for gap in range(
                1,
                int(max_gap) + 1,
            )
        )
    
    gap_bins = normalize_gap_bins(
        gap_bins
    )
    
    max_gap = max_gap_from_bins(
        gap_bins
    )
    
    if target_gap_rates_bg.size(1) < len(gap_bins):
        raise ValueError(
            f"target_gap_rates_bg has {target_gap_rates_bg.size(1)} bins, "
            f"but gap_bins has {len(gap_bins)} bins: {gap_bins}"
        )
    
    if confidence_bg is not None and confidence_bg.size(1) < len(gap_bins):
        raise ValueError(
            f"confidence_bg has {confidence_bg.size(1)} bins, "
            f"but gap_bins has {len(gap_bins)} bins: {gap_bins}"
        )

    for lo, hi in gap_bins:
        num_bin = p.new_zeros((B,))
        den_bin = p.new_zeros((B,))
    
        for g in range(lo, hi + 1):
            if p.shape[2] <= g:
                continue
    
            p0 = p[:, :, :-g]
            pg = p[:, :, g:]
    
            num_bin = num_bin + (p0 * pg).sum(dim=(1, 2, 3, 4))
            den_bin = den_bin + p0.sum(dim=(1, 2, 3, 4))
    
        rates.append(num_bin / den_bin.clamp_min(eps))

    pred_rates = torch.stack(rates, dim=1)  # (B,G)

    tgt_rates = target_gap_rates_bg[:, :len(gap_bins)]

    upper_allowed = (1.0 + margin) * tgt_rates
    lower_allowed = (1.0 - lower_margin) * tgt_rates
    
    upper_error = torch.relu(pred_rates - upper_allowed).pow(2)
    lower_error = torch.relu(lower_allowed - pred_rates).pow(2)
    
    error = upper_error + float(lower_weight) * lower_error
    
    if confidence_bg is not None:
        conf = confidence_bg[:, :len(gap_bins)].to(
            device=error.device,
            dtype=error.dtype,
        )
        error = error * conf
        denom = conf.sum().clamp_min(1.0)
        loss = error.sum() / denom
    else:
        conf = None
        loss = error.mean()

    if return_parts:
        return {
            "loss": loss,
            "pred_gap_rates": pred_rates.detach(),
            "allowed_gap_rates": upper_allowed.detach(),
            "lower_allowed_gap_rates": lower_allowed.detach(),
            "target_gap_rates": tgt_rates.detach(),
            "upper_error": upper_error.detach(),
            "lower_error": lower_error.detach(),
            "confidence": None if conf is None else conf.detach(),
            "gap_bins": gap_bins,
        }

    return loss


# ---------- Local context losses ----------
def soft_active_site_ratio_from_logits(
    logits_b1thw: torch.Tensor,
    tau_prob: float = 0.25,
    prob_threshold: Optional[float] = None,
    max_active_site: float = 1024.0,
    clamp_max: bool = True,
) -> torch.Tensor:
    """
    logits_b1thw: (B,1,T,H,W)

    Returns:
      (B,) soft active-site ratio normalized by effective max active sites,
      matching the hard version based on active-site COUNT / max_active_site.

    Notes:
    - site_active is a soft indicator in [0,1] for whether each spatial site
      was active at least once
    - final ratio is SUM over sites divided by min(H*W, max_active_site)
    """

    if logits_b1thw.dim() != 5:
        raise ValueError(f"Expected (B,1,T,H,W), got {tuple(logits_b1thw.shape)}")

    B, C, T, H, W = logits_b1thw.shape
    if C != 1:
        raise ValueError(f"Expected channel dim = 1, got {C}")

    # A site is hard-active when any temporal logit crosses the decoder
    # threshold. Therefore max(logit_t) has exactly the correct boundary.
    site_logit = logits_b1thw.amax(dim=2)  # (B,1,H,W)
    
    threshold_value = (
        0.5
        if prob_threshold is None
        else float(prob_threshold)
    )
    
    threshold = site_logit.new_tensor(
        threshold_value
    ).clamp(1e-6, 1.0 - 1e-6)
    
    site_boundary = torch.logit(threshold)

    # Soft approximation of whether any time point crosses the
    # decoder's calibrated firing boundary.
    site_active = torch.sigmoid(
        (site_logit - site_boundary)
        / max(tau_prob, 1e-6)
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
    eps_prob: float = 1e-6,
    eps_time: float = 1e-6,
    tau: float = 0.25,
    prob_threshold: Optional[float] = None,
) -> torch.Tensor:
    """
    FP32 soft local context from logits.

    returns: (B,9) =
      [log_mean_firing_density,
       var_x, var_y, var_t,
       cov_xy, cov_xt, cov_yt,
       active_site_ratio,
       temporal_trend]"""
    if logits_b1thw.dim() != 5:
        raise ValueError(f"Expected (B,1,T,H,W), got {tuple(logits_b1thw.shape)}")

    B, C, T, H, W = logits_b1thw.shape
    if C != 1:
        raise ValueError(f"Expected channel=1, got {C}")

    # Critical: do moment math in FP32, outside AMP.
    with torch.cuda.amp.autocast(enabled=False):
        logits_f = logits_b1thw.float()

        p = soft_binary_from_logits(
            logits_f,
            tau=max(1e-6, float(tau)),
            prob_threshold=prob_threshold,
            eps=eps_prob,
        )
        
        p0 = p[:, 0]  # (B,T,H,W)

        # 0) log mean firing density
        mfd = p.mean(dim=(1, 2, 3, 4))
        d0 = torch.log(mfd.clamp_min(eps_prob)).clamp(min=-15.0, max=0.0)

        # fixed normalized coordinate fields in FP32
        tt = torch.linspace(-1.0, 1.0, T, device=p.device, dtype=torch.float32).view(1, T, 1, 1)
        yy = torch.linspace(-1.0, 1.0, H, device=p.device, dtype=torch.float32).view(1, 1, H, 1)
        xx = torch.linspace(-1.0, 1.0, W, device=p.device, dtype=torch.float32).view(1, 1, 1, W)

        # Clamp only total mass. Do NOT clamp every voxel probability.
        mass_raw = p0.sum(dim=(1, 2, 3))
        mass = mass_raw.clamp_min(eps_prob)

        mx = (p0 * xx).sum(dim=(1, 2, 3)) / mass
        my = (p0 * yy).sum(dim=(1, 2, 3)) / mass
        mt = (p0 * tt).sum(dim=(1, 2, 3)) / mass

        dx = xx - mx.view(B, 1, 1, 1)
        dy = yy - my.view(B, 1, 1, 1)
        dt = tt - mt.view(B, 1, 1, 1)

        d1 = (p0 * dx * dx).sum(dim=(1, 2, 3)) / mass
        d2 = (p0 * dy * dy).sum(dim=(1, 2, 3)) / mass
        d3 = (p0 * dt * dt).sum(dim=(1, 2, 3)) / mass

        d4 = (p0 * dx * dy).sum(dim=(1, 2, 3)) / mass
        d5 = (p0 * dx * dt).sum(dim=(1, 2, 3)) / mass
        d6 = (p0 * dy * dt).sum(dim=(1, 2, 3)) / mass
        
        # 7) Soft active spatial-site ratio.
        d7 = soft_active_site_ratio_from_logits(
            logits_b1thw=logits_f,
            tau_prob=max(1e-6, float(tau)),
            prob_threshold=prob_threshold,
            max_active_site=1024.0,
            clamp_max=True,
        )
        
        # 8) Scale-free temporal activity trend.
        frame_mean = p0.mean(dim=(2, 3))  # (B,T)
        
        d8 = temporal_trend_score_torch(
            frame_mean_bt=frame_mean,
            eps_time=eps_time,
        )
        
        out = torch.stack(
            [
                d0,
                d1,
                d2,
                d3,
                d4,
                d5,
                d6,
                d7,
                d8,
            ],
            dim=1,
        )
        
        # Fully blank pathological case: keep shape/trend statistics finite.
        blank = mass_raw <= eps_prob
        
        if blank.any():
            out[blank, 1:7] = 0.0
            out[blank, 7] = 0.0
            out[blank, 8] = 0.0
        
        return out



def _patchify_b1thw(
    x_b1thw: torch.Tensor,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """
    Convert (B,1,T,H,W) into non-overlapping patches (B,N,P).

    N = number of tokens
    P = pT * pH * pW
    """
    if x_b1thw.dim() != 5 or x_b1thw.size(1) != 1:
        raise ValueError(
            f"Expected (B,1,T,H,W), got {tuple(x_b1thw.shape)}"
        )

    B, _, T, H, W = x_b1thw.shape
    pT, pH, pW = map(int, patch_size)

    if T % pT != 0 or H % pH != 0 or W % pW != 0:
        raise ValueError(
            f"Volume shape {(T, H, W)} must be divisible by "
            f"patch_size {patch_size}"
        )

    nT = T // pT
    nH = H // pH
    nW = W // pW

    return (
        x_b1thw
        .reshape(
            B,
            1,
            nT,
            pT,
            nH,
            pH,
            nW,
            pW,
        )
        .permute(
            0,
            2,
            4,
            6,
            3,
            5,
            7,
            1,
        )
        .reshape(
            B,
            nT * nH * nW,
            pT * pH * pW,
        )
    )


def _patch_moments(
    patch_prob_mp: torch.Tensor,
    patch_size: tuple[int, int, int],
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Calculate one statistics vector for each selected patch.

    Input:
        patch_prob_mp: (M, P)

    Output:
        features: (M, 9)

    Feature layout:
        0: log density
        1: variance x
        2: variance y
        3: variance t
        4: covariance xy
        5: covariance xt
        6: covariance yt
        7: active spatial-site ratio
        8: temporal trend

    Coordinates are normalized independently inside each patch to [-1, 1].
    """
    if patch_prob_mp.dim() != 2:
        raise ValueError(
            "Expected selected patches with shape (M,P), "
            f"got {tuple(patch_prob_mp.shape)}"
        )

    pT, pH, pW = map(int, patch_size)
    patch_volume = pT * pH * pW

    if patch_prob_mp.size(1) != patch_volume:
        raise ValueError(
            f"Expected patch volume {patch_volume}, "
            f"got {patch_prob_mp.size(1)}"
        )

    # Keep all moment calculations in FP32.
    p = patch_prob_mp.float().clamp(
        min=0.0,
        max=1.0,
    )

    device = p.device

    # (M, pT, pH, pW)
    p4 = p.reshape(
        -1,
        pT,
        pH,
        pW,
    )

    tt = torch.linspace(
        -1.0,
        1.0,
        pT,
        device=device,
        dtype=torch.float32,
    )

    yy = torch.linspace(
        -1.0,
        1.0,
        pH,
        device=device,
        dtype=torch.float32,
    )

    xx = torch.linspace(
        -1.0,
        1.0,
        pW,
        device=device,
        dtype=torch.float32,
    )

    t3, y3, x3 = torch.meshgrid(
        tt,
        yy,
        xx,
        indexing="ij",
    )

    tf = t3.reshape(
        1,
        patch_volume,
    )

    yf = y3.reshape(
        1,
        patch_volume,
    )

    xf = x3.reshape(
        1,
        patch_volume,
    )

    mass_raw = p.sum(dim=1)
    mass = mass_raw.clamp_min(eps)

    density = p.mean(dim=1).clamp_min(eps)

    mean_x = (
        (p * xf).sum(dim=1)
        / mass
    )

    mean_y = (
        (p * yf).sum(dim=1)
        / mass
    )

    mean_t = (
        (p * tf).sum(dim=1)
        / mass
    )

    ex2 = (
        (p * xf.square()).sum(dim=1)
        / mass
    )

    ey2 = (
        (p * yf.square()).sum(dim=1)
        / mass
    )

    et2 = (
        (p * tf.square()).sum(dim=1)
        / mass
    )

    exy = (
        (p * xf * yf).sum(dim=1)
        / mass
    )

    ext = (
        (p * xf * tf).sum(dim=1)
        / mass
    )

    eyt = (
        (p * yf * tf).sum(dim=1)
        / mass
    )

    var_x = (
        ex2 - mean_x.square()
    ).clamp_min(0.0)

    var_y = (
        ey2 - mean_y.square()
    ).clamp_min(0.0)

    var_t = (
        et2 - mean_t.square()
    ).clamp_min(0.0)

    cov_xy = (
        exy - mean_x * mean_y
    )

    cov_xt = (
        ext - mean_x * mean_t
    )

    cov_yt = (
        eyt - mean_y * mean_t
    )

    log_density = torch.log(
        density
    ).clamp(
        min=-15.0,
        max=0.0,
    )

    # A spatial site is considered softly active when it is active at
    # least once within the temporal extent of the patch.
    #
    # Because patch_prob_mp is already produced by soft_binary_from_logits(),
    # temporal max is a direct soft approximation to temporal OR.
    site_active = p4.amax(
        dim=1,
    )  # (M, pH, pW)

    active_site_ratio = site_active.mean(
        dim=(1, 2),
    ).clamp(
        min=0.0,
        max=1.0,
    )

    # Mean soft activity per frame.
    frame_mean = p4.mean(
        dim=(2, 3),
    )  # (M, pT)

    temporal_trend = temporal_trend_score_torch(
        frame_mean_bt=frame_mean,
        eps_time=eps,
    )

    features = (
        log_density,
        var_x,
        var_y,
        var_t,
        cov_xy,
        cov_xt,
        cov_yt,
        active_site_ratio,
        temporal_trend,
    )

    out = torch.stack(
        features,
        dim=1,
    )

    # Keep pathological blank-patch outputs finite.
    blank = mass_raw <= eps

    if blank.any():
        out[blank, 1:9] = 0.0

    return out


def local_moment_field_loss(
    logits_b1thw: torch.Tensor,
    target_b1thw: torch.Tensor,
    patch_size: tuple[int, int, int],
    tau: float = 0.25,
    prob_threshold: Optional[float] = None,
    min_active_spikes: int = 1,
    min_shape_spikes: int = 5,
    min_trend_spikes: int = 6,
    min_trend_frames: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Match statistics independently inside non-overlapping patches.

    Feature validity rules:

    Patches with at least min_active_spikes:
        - log density
        - active spatial-site ratio

    Patches with at least min_shape_spikes:
        - variance x
        - variance y
        - variance t
        - covariance xy
        - covariance xt
        - covariance yt

    Patches with at least min_trend_spikes and min_trend_frames:
        - temporal trend

    Blank target patches are excluded. Existing reconstruction, outside,
    blank-token, and blank-patch losses constrain false-positive activity
    in blank regions.
    """
    if logits_b1thw.shape != target_b1thw.shape:
        raise ValueError(
            f"logits shape {tuple(logits_b1thw.shape)} "
            f"does not match target shape {tuple(target_b1thw.shape)}"
        )

    if logits_b1thw.dim() != 5 or logits_b1thw.size(1) != 1:
        raise ValueError(
            "Expected logits and target with shape (B,1,T,H,W), "
            f"got {tuple(logits_b1thw.shape)}"
        )

    pT, pH, pW = map(
        int,
        patch_size,
    )

    logit_patches = _patchify_b1thw(
        logits_b1thw,
        patch_size,
    )

    with torch.no_grad():
        target_patches = _patchify_b1thw(
            target_b1thw.float(),
            patch_size,
        )

        target_counts = target_patches.sum(
            dim=-1,
        )

        active_mask = (
            target_counts
            >= int(min_active_spikes)
        )

    # Differentiable zero for an entirely blank batch.
    if not bool(active_mask.any()):
        return logits_b1thw.sum() * 0.0

    # Only calculate predicted statistics for target-active patches.
    pred_prob = soft_binary_from_logits(
        logit_patches[active_mask].float(),
        tau=max(
            float(tau),
            eps,
        ),
        prob_threshold=prob_threshold,
        eps=eps,
    )

    target_prob = target_patches[
        active_mask
    ].float()

    pred_features = _patch_moments(
        pred_prob,
        patch_size=patch_size,
        eps=eps,
    )

    with torch.no_grad():
        target_features = _patch_moments(
            target_prob,
            patch_size=patch_size,
            eps=eps,
        )

        active_counts = target_counts[
            active_mask
        ]

        # Reshape selected target patches to recover temporal support.
        target_active_4d = target_prob.reshape(
            -1,
            pT,
            pH,
            pW,
        )

        # Count frames containing at least one target spike.
        active_frame_counts = (
            target_active_4d.sum(
                dim=(2, 3),
            ) > 0
        ).sum(
            dim=1,
        )

    if pred_features.shape[1] != 9:
        raise RuntimeError(
            "Expected 9 patch-field features, "
            f"got shape {tuple(pred_features.shape)}"
        )

    if target_features.shape != pred_features.shape:
        raise RuntimeError(
            "Predicted and target patch-feature shapes differ: "
            f"{tuple(pred_features.shape)} vs "
            f"{tuple(target_features.shape)}"
        )

    per_feature = F.smooth_l1_loss(
        pred_features,
        target_features,
        reduction="none",
    )

    active_valid = (
        active_counts
        >= int(min_active_spikes)
    ).to(
        dtype=per_feature.dtype,
    )

    shape_valid = (
        active_counts
        >= int(min_shape_spikes)
    ).to(
        dtype=per_feature.dtype,
    )

    trend_valid = (
        (
            active_counts
            >= int(min_trend_spikes)
        )
        & (
            active_frame_counts
            >= int(min_trend_frames)
        )
    ).to(
        dtype=per_feature.dtype,
    )

    feature_mask = torch.zeros_like(
        per_feature,
    )

    # Feature layout:
    #   0   : log density
    #   1:7 : variances and covariances
    #   7   : active spatial-site ratio
    #   8   : temporal trend

    # Reliable for every target-active patch.
    feature_mask[:, 0] = active_valid
    feature_mask[:, 7] = active_valid

    # Require stronger support for variances and covariances.
    feature_mask[:, 1:7] = shape_valid.unsqueeze(
        dim=1,
    )

    # Require both enough spikes and enough occupied frames.
    feature_mask[:, 8] = trend_valid

    denominator = feature_mask.sum().clamp_min(
        1.0,
    )

    return (
        per_feature
        * feature_mask
    ).sum() / denominator

def ctx_loss_soft(
    logits_b1thw: torch.Tensor,
    ctx_tgt_b9: torch.Tensor,
    dims: Sequence[int] = tuple(range(9)),
    weights: Optional[Sequence[float]] = None,
    tau: float = 0.25,
    prob_threshold: Optional[float] = None,
) -> torch.Tensor:

    if ctx_tgt_b9.dim() != 2 or ctx_tgt_b9.size(1) < 9:
        raise ValueError(f"ctx_tgt_b9 must be (B,>=9), got {tuple(ctx_tgt_b9.shape)}")
    
    pred9 = ctx_features_soft_from_logits(
        logits_b1thw.float(),
        tau=tau,
        prob_threshold=prob_threshold,
    )


    if weights is None:
        weight_map = {d: 1.0 for d in range(9)}
    else:
        if len(weights) != 9:
            raise ValueError("weights must have length 9")
        weight_map = {d: float(weights[d]) for d in range(9)}

    loss = pred9.new_tensor(0.0)

    for d in dims:
        if d < 0 or d > 8:
            raise ValueError(f"Invalid dim {d}")

        pred = pred9[:, d]
        tgt = ctx_tgt_b9[:, d].to(pred9.dtype)

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