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
    steps: int = 12,
    temperature: float = 1.0,
    top_k: int = 5,
):
    """MaskGIT sampling over the flat Stage-2B alphabet.

    One stream, one pass per step. The old two-pass rule existed only because
    alpha was a distribution over the children of ONE parent, so it had to be
    re-conditioned on the z1 actually sampled. A flat code has no parent, so
    that entire dance disappears.
    """
    device = global_ctx.device
    activity = activity.to(device).long().clamp(0, 1)
    B, N = activity.shape
    V = int(prior.V)
    steps = int(max(1, steps))
    a = activity

    if roi_mask is None:
        roi = torch.ones((B, N), device=device, dtype=torch.bool)
    else:
        roi = roi_mask.squeeze(-1) if roi_mask.dim() == 3 else roi_mask
        roi = roi.to(device=device, dtype=torch.bool)
        if roi.shape != (B, N):
            raise ValueError(f"roi_mask must have shape {(B, N)}, got {tuple(roi.shape)}")

    f = torch.full((B, N), prior.f_null_id, device=device, dtype=torch.long)

    active = a.eq(prior.a_active_id)
    if visible_codes is not None:
        visible_codes = visible_codes.to(device=device, dtype=torch.long)
        if visible_codes.dim() != 3 or visible_codes.size(-1) < 3:
            raise ValueError(
                f"visible_codes must have shape {(B, N, 3)}, "
                f"got {tuple(visible_codes.shape)}"
            )
        vis_f, vis_active = prior.flat_ids_from_codes(visible_codes)
        visible_active = vis_active & (~roi)
        f[visible_active] = vis_f[visible_active]

    masked = active & roi
    f[masked] = prior.f_mask_id

    def _logits(f_state):
        out, _, _ = prior(
            a, f_state,
            global_ctx=global_ctx,
            local_ctx=local_ctx,
            task_id=task_id,
            roi_mask=roi_mask,
            targets=None,
        )
        return out["flat"]

    for step in range(steps):
        if not masked.any():
            break

        f_samp, conf = _sample_from_logits(_logits(f), temperature, top_k=top_k)
        conf = conf.masked_fill(~masked, -1.0)

        num_left = masked.sum(dim=1)
        keep_ratio = 1.0 - float(step + 1) / float(steps)
        num_keep_masked = torch.ceil(num_left.float() * keep_ratio).long()

        for b in range(B):
            n_unmask = int(num_left[b].item() - num_keep_masked[b].item())
            if n_unmask <= 0:
                continue
            idx = torch.topk(conf[b], k=n_unmask).indices
            f[b, idx] = f_samp[b, idx]
            masked[b, idx] = False

    if masked.any():
        f_final = _logits(f).argmax(dim=-1)
        f[masked] = f_final[masked]

    f[~active] = prior.f_null_id

    flat_ids = torch.full((B, N), -1, device=device, dtype=torch.long)
    flat_ids[active] = f[active].clamp(0, V - 1)
    return {"flat_ids": flat_ids, "active": active}


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
    activity_count_mode="expected",
    activity_count_stochastic_round=False,
    activity_coord_temperature=1.0,
    motif_steps=12,
    motif_temperature=1.0,
    motif_top_k=5,
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

        if visible_codes.dim() != 3 or visible_codes.size(-1) < 3:
            raise ValueError(
                f"visible_codes must have shape {(B, N, 3)}, "
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
        steps=motif_steps,
        temperature=motif_temperature,
        top_k=motif_top_k,
    )

    return {
        "activity": activity,
        "flat_ids": motif_sample["flat_ids"],
        "activity_input": a_in,
        "activity_out": activity_out,
    }