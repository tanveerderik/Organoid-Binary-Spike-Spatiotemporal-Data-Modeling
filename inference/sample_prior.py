#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import torch


def _sample_from_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: int = None,
):
    """Sample categorical IDs, optionally restricted to predicted top-k."""
    sample_logits = logits

    if top_k is not None:
        k = max(1, min(int(top_k), logits.size(-1)))
        top_values, top_indices = torch.topk(
            logits,
            k=k,
            dim=-1,
        )

        sample_logits = torch.full_like(
            logits,
            float("-inf"),
        )
        sample_logits.scatter_(
            dim=-1,
            index=top_indices,
            src=top_values,
        )

    if temperature <= 0:
        probs = torch.softmax(sample_logits, dim=-1)
        ids = sample_logits.argmax(dim=-1)
    else:
        probs = torch.softmax(
            sample_logits / float(temperature),
            dim=-1,
        )
        ids = torch.multinomial(
            probs.reshape(-1, probs.size(-1)),
            1,
        ).view(logits.shape[:2])

    conf = probs.gather(
        -1,
        ids.unsqueeze(-1),
    ).squeeze(-1)

    return ids, conf

@torch.no_grad()
def iterative_unmask_motif_given_activity(
    prior,
    activity,
    global_ctx,
    local_ctx,
    task_id,
    roi_mask=None,
    visible_codes=None,
    visible_alpha=None,
    steps: int = 12,
    temperature: float = 1.0,
    alpha_temperature: float = 1.0,
    z1_top_k: int = 5,
):
    """MaskGIT z1 sampling plus stochastic logistic-normal alpha sampling."""
    device = global_ctx.device
    activity = activity.to(device).long().clamp(0, 1)
    B, N = activity.shape
    K2 = int(prior.K2)
    steps = int(max(1, steps))
    a = activity

    if roi_mask is None:
        roi = torch.ones((B, N), device=device, dtype=torch.bool)
    else:
        roi = roi_mask.squeeze(-1) if roi_mask.dim() == 3 else roi_mask
        roi = roi.to(device=device, dtype=torch.bool)
        if roi.shape != (B, N):
            raise ValueError(f"roi_mask must have shape {(B, N)}, got {tuple(roi.shape)}")

    z1 = torch.full((B, N), prior.z1_null_id, device=device, dtype=torch.long)
    # z2 IDs are retained only as mask/null state and dominant-child summaries
    # for the transformer input. Final decoding uses the full alpha tensor.
    z2 = torch.full((B, N), prior.z2_null_id, device=device, dtype=torch.long)
    alpha = torch.zeros((B, N, K2), device=device, dtype=torch.float32)

    active = a.eq(prior.a_active_id)
    if visible_codes is not None:
        visible_codes = visible_codes.to(device=device, dtype=torch.long)
        if visible_codes.shape != (B, N, 2):
            raise ValueError(
                f"visible_codes must have shape {(B, N, 2)}, got {tuple(visible_codes.shape)}"
            )
        visible_active = (
            visible_codes[..., 0].ge(0)
            & visible_codes[..., 1].ge(0)
            & (~roi)
        )
        z1[visible_active] = visible_codes[..., 0][visible_active].clamp(0, prior.K1 - 1)
        z2[visible_active] = visible_codes[..., 1][visible_active].clamp(0, prior.K2 - 1)
        if visible_alpha is None:
            alpha[visible_active] = torch.nn.functional.one_hot(
                z2[visible_active], num_classes=K2
            ).float()
        else:
            visible_alpha = visible_alpha.to(
                device=device, dtype=alpha.dtype
            )
            if visible_alpha.shape != (B, N, K2):
                raise ValueError(
                    f"visible_alpha must have shape {(B, N, K2)}, "
                    f"got {tuple(visible_alpha.shape)}"
                )
            normalized_visible_alpha = visible_alpha / visible_alpha.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            alpha[visible_active] = normalized_visible_alpha[visible_active]

    masked = active & roi
    z1[masked] = prior.z1_mask_id
    z2[masked] = prior.z2_mask_id

    for step in range(steps):
        if not masked.any():
            break
        logits, _, _ = prior(
            a, z1, z2,
            alpha_in=alpha,
            global_ctx=global_ctx,
            local_ctx=local_ctx,
            task_id=task_id,
            roi_mask=roi_mask,
            targets=None,
        )

        z1_samp, conf_z1 = _sample_from_logits(
            logits["z1"],
            temperature,
            top_k=z1_top_k,
        )

        mu = logits["alpha_mu"]
        std = logits["alpha_log_std"].exp()
        if alpha_temperature <= 0:
            alpha_samp = torch.softmax(mu, dim=-1)
        else:
            noise = torch.randn_like(mu)
            alpha_samp = torch.softmax(
                mu + float(alpha_temperature) * std * noise,
                dim=-1,
            )
        # Confidence is high when the predicted mean is concentrated and the
        # learned uncertainty is low. It is used only for MaskGIT commit order.
        alpha_mean = torch.softmax(mu, dim=-1)
        conf_alpha = alpha_mean.max(dim=-1).values * torch.exp(-std.mean(dim=-1))
        joint_conf = (conf_z1 * conf_alpha).masked_fill(~masked, -1.0)

        num_left = masked.sum(dim=1)
        keep_ratio = 1.0 - float(step + 1) / float(steps)
        num_keep_masked = torch.ceil(num_left.float() * keep_ratio).long()

        for b in range(B):
            n_unmask = int(num_left[b].item() - num_keep_masked[b].item())
            if n_unmask <= 0:
                continue
            idx = torch.topk(joint_conf[b], k=n_unmask).indices
            z1[b, idx] = z1_samp[b, idx]
            alpha[b, idx] = alpha_samp[b, idx]
            z2[b, idx] = alpha_samp[b, idx].argmax(dim=-1)
            masked[b, idx] = False

    if masked.any():
        logits, _, _ = prior(
            a, z1, z2,
            alpha_in=alpha,
            global_ctx=global_ctx,
            local_ctx=local_ctx,
            task_id=task_id,
            roi_mask=roi_mask,
            targets=None,
        )
        z1_final = logits["z1"].argmax(dim=-1)
        alpha_final = torch.softmax(logits["alpha_mu"], dim=-1)
        z1[masked] = z1_final[masked]
        alpha[masked] = alpha_final[masked]
        z2[masked] = alpha_final[masked].argmax(dim=-1)

    z1[~active] = prior.z1_null_id
    z2[~active] = prior.z2_null_id
    alpha[~active] = 0.0

    codes = torch.full((B, N, 2), -1, device=device, dtype=torch.long)
    codes[..., 0][active] = z1[active].clamp(0, prior.K1 - 1)
    codes[..., 1][active] = z2[active].clamp(0, prior.K2 - 1)
    return {"codes": codes, "alpha": alpha}

@torch.no_grad()
def sample_hierarchical_roi(
    prior,
    global_ctx,
    local_ctx,
    task_id,
    roi_mask,
    *,
    visible_codes=None,
    visible_alpha=None,
    activity_count_temperature=1.0,
    activity_count_mode="expected",
    activity_count_stochastic_round=False,
    activity_coord_temperature=1.0,
    motif_steps=12,
    motif_temperature=1.0,
    motif_alpha_temperature=1.0,
    z1_top_k=5,
):
    """
    Generate activity and motif codes inside roi_mask.

    Full generation:
        roi_mask is all True
        visible_codes is None
        a_in is therefore all mask ID 2

    Partial generation:
        visible_codes contains the real/full VQ codes
        positions outside roi_mask are converted into visible blank/active
        activity states and preserved during motif generation
    """

    device = global_ctx.device
    B = global_ctx.shape[0]
    N = prior.activity_prior.Ntok

    roi = roi_mask
    if roi.dim() == 3:
        roi = roi.squeeze(-1)

    roi = roi.to(
        device=device,
        dtype=torch.bool,
    )

    if roi.shape != (B, N):
        raise ValueError(
            f"roi_mask must have shape {(B, N)}, "
            f"got {tuple(roi.shape)}"
        )

    # Default for full generation: there is no visible activity.
    visible_activity = torch.zeros(
        (B, N),
        device=device,
        dtype=torch.long,
    )

    if visible_codes is not None:
        visible_codes = visible_codes.to(
            device=device,
            dtype=torch.long,
        )

        if visible_codes.shape != (B, N, 2):
            raise ValueError(
                f"visible_codes must have shape {(B, N, 2)}, "
                f"got {tuple(visible_codes.shape)}"
            )

        visible_activity = visible_codes[..., 0].ne(-1).long()

    # Training convention:
    #   0 = visible blank
    #   1 = visible active
    #   2 = masked/predict
    a_in = visible_activity.clone()
    a_in[roi] = int(prior.activity_prior.a_mask_id)

    activity_out = prior.forward_activity(
        global_ctx=global_ctx,
        local_ctx=local_ctx,
        task_id=task_id,
        a_in=a_in,
        roi_mask=roi,
    )

    # Use the aggregated soft activity grid followed by a count-controlled
    # unique top-k selection. This avoids coordinate collisions reducing the
    # final hard activity count below the selected count.
    activity_roi = prior.activity_prior.sample_hard_activity_gridtopk(
        activity_out,
        count_temperature=activity_count_temperature,
        count_mode=activity_count_mode,
        count_stochastic_round=activity_count_stochastic_round,
        roi_mask=roi,
    )

    # Preserve visible activity outside the ROI.
    activity = torch.where(
        roi,
        activity_roi,
        visible_activity,
    )

    motif_sample = iterative_unmask_motif_given_activity(
        prior.motif_prior,
        activity=activity,
        global_ctx=global_ctx,
        local_ctx=local_ctx,
        task_id=task_id,
        roi_mask=roi,
        visible_codes=visible_codes,
        visible_alpha=visible_alpha,
        steps=motif_steps,
        temperature=motif_temperature,
        alpha_temperature=motif_alpha_temperature,
        z1_top_k=z1_top_k,
    )

    return {
        "activity": activity,
        "codes": motif_sample["codes"],
        "alpha": motif_sample["alpha"],
        "activity_input": a_in,
        "activity_out": activity_out,
    }