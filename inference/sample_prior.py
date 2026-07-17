#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import torch


def _sample_from_logits(logits: torch.Tensor, temperature: float):
    """Sample categorical ids and return sampled ids plus their probabilities.

    logits: (B,N,V)
    returns:
        ids:  (B,N)
        conf: (B,N), probability assigned to sampled id
    """
    if temperature <= 0:
        probs = torch.softmax(logits, dim=-1)
        ids = probs.argmax(dim=-1)
    else:
        probs = torch.softmax(logits / float(temperature), dim=-1)
        ids = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).view(logits.shape[:2])
    conf = probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
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
    steps: int = 12,
    temperature: float = 1.0,
):
    device = global_ctx.device
    activity = activity.to(device).long().clamp(0, 1)  # (B,N)

    B, N = activity.shape
    steps = int(max(1, steps))

    a = activity

    if roi_mask is None:
        roi = torch.ones(
            (B, N),
            device=device,
            dtype=torch.bool,
        )
    else:
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

    z1 = torch.full(
        (B, N),
        prior.z1_null_id,
        device=device,
        dtype=torch.long,
    )
    z2 = torch.full(
        (B, N),
        prior.z2_null_id,
        device=device,
        dtype=torch.long,
    )

    active = a.eq(prior.a_active_id)

    # Preserve visible motif codes outside the prediction ROI.
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

        visible_active = (
            visible_codes[..., 0].ge(0)
            & visible_codes[..., 1].ge(0)
        )
        visible_active = visible_active & (~roi)

        z1[visible_active] = visible_codes[..., 0][visible_active].clamp(
            0,
            prior.K1 - 1,
        )
        z2[visible_active] = visible_codes[..., 1][visible_active].clamp(
            0,
            prior.K2 - 1,
        )

    # Only active positions inside the prediction ROI are sampled.
    masked = active & roi

    z1[masked] = prior.z1_mask_id
    z2[masked] = prior.z2_mask_id

    for s in range(steps):
        if not masked.any():
            break

        logits, _, _ = prior(
            a, z1, z2,
            global_ctx=global_ctx,
            local_ctx=local_ctx,
            task_id=task_id,
            roi_mask=roi_mask,
            targets=None,
        )

        z1_samp, conf_z1 = _sample_from_logits(logits["z1"], temperature)
        z2_samp, conf_z2 = _sample_from_logits(logits["z2"], temperature)

        joint_conf = (conf_z1 * conf_z2).masked_fill(~masked, -1.0)

        num_left = masked.sum(dim=1)
        keep_ratio = 1.0 - float(s + 1) / float(steps)
        num_keep_masked = torch.ceil(num_left.float() * keep_ratio).long()

        for b in range(B):
            n_unmask = int(num_left[b].item() - num_keep_masked[b].item())
            if n_unmask <= 0:
                continue

            idx = torch.topk(joint_conf[b], k=n_unmask).indices
            z1[b, idx] = z1_samp[b, idx]
            z2[b, idx] = z2_samp[b, idx]
            masked[b, idx] = False

    if masked.any():
        logits, _, _ = prior(
            a, z1, z2,
            global_ctx=global_ctx,
            local_ctx=local_ctx,
            task_id=task_id,
            roi_mask=roi_mask,
            targets=None,
        )
        z1_final = logits["z1"].argmax(dim=-1)
        z2_final = logits["z2"].argmax(dim=-1)

        z1[masked] = z1_final[masked]
        z2[masked] = z2_final[masked]

    z1[~active] = prior.z1_null_id
    z2[~active] = prior.z2_null_id

    codes = torch.full((B, N, 2), -1, device=device, dtype=torch.long)
    codes[..., 0][active] = z1[active].clamp(0, prior.K1 - 1)
    codes[..., 1][active] = z2[active].clamp(0, prior.K2 - 1)

    return codes

@torch.no_grad()
def sample_hierarchical_roi(
    prior,
    global_ctx,
    local_ctx,
    task_id,
    roi_mask,
    *,
    visible_codes=None,
    activity_count_temperature=1.0,
    activity_coord_temperature=1.0,
    motif_steps=12,
    motif_temperature=1.0,
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

    # The current inference path uses the aggregated soft activity grid
    # followed by a count-controlled top-k selection.
    activity_roi = prior.activity_prior.sample_hard_activity_gridtopk(
        activity_out,
        count_temperature=activity_count_temperature,
        roi_mask=roi,
    )

    # Preserve visible activity outside the ROI.
    activity = torch.where(
        roi,
        activity_roi,
        visible_activity,
    )

    codes = iterative_unmask_motif_given_activity(
        prior.motif_prior,
        activity=activity,
        global_ctx=global_ctx,
        local_ctx=local_ctx,
        task_id=task_id,
        roi_mask=roi,
        visible_codes=visible_codes,
        steps=motif_steps,
        temperature=motif_temperature,
    )

    return {
        "activity": activity,
        "codes": codes,
        "activity_input": a_in,
        "activity_out": activity_out,
    }
