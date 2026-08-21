#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F

from typing import Union, Optional

from ..model.prior import (
    build_activity_targets_from_codes,
)

from ..inference.decode import decode_motif_logits_soft_given_activity
from .stage4d_adapt import build_adapted_motif_io, adapt_probability

from ..utils.losses import (
    ctx_loss_soft,
    local_moment_field_loss,
    short_gap_excess_loss_from_logits_batch_targets,
    spatial_support_violation_loss,
)

from ..utils.recon import (
    soft_spatial_token_map_from_logits,
    sample_full_token_map_to_crop,
    dilate_spatial_support_hw,
)


def ramp(ep, start, end, max_val):
    if ep < start:
        return 0.0
    if ep >= end:
        return float(max_val)
    return float(max_val) * (ep - start) / max(1, end - start)

def _freeze_module(m):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)


def _batch_to_device(batch, device):
    x = batch["x"].to(device, non_blocking=True)

    gct = batch.get("global_ctx", None)
    lct = batch.get("local_ctx", None)

    if isinstance(gct, torch.Tensor):
        gct = gct.to(device, non_blocking=True)
    if isinstance(lct, torch.Tensor):
        lct = lct.to(device, non_blocking=True)

    task_id = batch.get("task_id", None)
    if task_id is None:
        raise KeyError('Batch missing "task_id".')
    task_id = task_id.to(device, non_blocking=True).long()

    mask_spec = batch.get("mask_spec", None)

    return x, gct, lct, task_id, mask_spec



@torch.no_grad()
def _make_activity_in_from_codes(
    codes,
    pmask,
    *,
    blank_code: int = -1,
    a_mask_id: int = 2,
):
    """
    Creates masked activity input for the activity prior.

    a_in:
        0 = visible blank
        1 = visible active
        2 = masked / predict token
    """
    if pmask.dim() == 3:
        pmask = pmask.squeeze(-1)

    active = codes[..., 0].ne(blank_code)  # (B,N)

    a_in = active.long()
    a_in[pmask.bool()] = int(a_mask_id)

    return a_in


@torch.no_grad()
def _vq_codes_and_pmask(vqvae, x, gct, lct, mask_spec, device):
    try:
        out = vqvae(
            x,
            global_ctx=gct,
            local_ctx=lct,
            predict_mask_spec=mask_spec,
        )
    except TypeError:
        out = vqvae(
            x,
            global_ctx=gct,
            local_ctx=lct,
        )

    codes = out["codes"].long()  # (B,N,2)

    pmask = out.get("predict_mask", None)
    if pmask is None:
        B, N = codes.shape[:2]
        pmask = torch.ones((B, N), device=device, dtype=torch.float32)
    else:
        if pmask.dim() == 3:
            pmask = pmask.squeeze(-1)
        pmask = pmask.to(device=device, dtype=torch.float32)

    grid = out.get("grid", None)

    return codes, pmask, grid



@torch.no_grad()
def _vq_codes_and_pmask_for_prior(vqvae, x, gct, lct, mask_spec, device):
    """Encode ladder codes and the predict mask. No alpha: the prior speaks the
    flat Stage-2B alphabet, and make_targets_from_codes does the (a,b,c) -> flat
    mapping through merge_map."""
    try:
        out = vqvae(
            x,
            global_ctx=gct,
            local_ctx=lct,
            predict_mask_spec=mask_spec,
        )
    except TypeError:
        out = vqvae(x, global_ctx=gct, local_ctx=lct)

    codes = out["codes"].long()
    pmask = out.get("predict_mask", None)
    if pmask is None:
        B, N = codes.shape[:2]
        pmask = torch.ones((B, N), device=device, dtype=torch.float32)
    else:
        if pmask.dim() == 3:
            pmask = pmask.squeeze(-1)
        pmask = pmask.to(device=device, dtype=torch.float32)

    return codes, pmask, out.get("grid", None)


def expected_code_distance_loss(logits, target, mask, distance_matrix):
    """Expected normalized z1 codebook distance from the target code.

    Computed in fp32. Under AMP the incoming logits are fp16 while softmax is
    promoted to fp32, so mixing them silently relies on type promotion; forcing
    fp32 keeps this small reduction both correct and numerically stable.
    """
    mask = mask.bool()
    if not mask.any():
        return logits.sum() * 0.0
    lm = logits[mask].float()
    ym = target[mask].long()
    p = F.softmax(lm, dim=-1)
    d = distance_matrix.to(device=lm.device, dtype=lm.dtype)[ym]
    return (p * d).sum(dim=-1).mean()


def distance_neighborhood_ce_loss(
    logits, target, mask, distance_matrix, *, k=5, tau=0.25
):
    """Soft CE over the k codebook-nearest alternatives to each target.

    This replaces the old logit-top-k margin. The acceptable top-k set is
    determined by frozen z1 geometry, not by the model's current ranking.
    """
    mask = mask.bool()
    if not mask.any():
        return logits.sum() * 0.0
    # fp32 throughout. Under AMP `logits` is fp16 but F.softmax below is
    # promoted to fp32, and scatter() requires the destination and source to
    # share a dtype, so building `q` from a fp16 zeros_like raised
    # "scatter(): Expected self.dtype to be equal to src.dtype".
    lm = logits[mask].float()
    ym = target[mask].long()
    dist = distance_matrix.to(device=lm.device, dtype=lm.dtype)[ym]
    k = max(1, min(int(k), dist.size(-1)))
    near_d, near_idx = torch.topk(dist, k=k, largest=False, dim=-1)
    q_local = F.softmax(-near_d / max(float(tau), 1e-6), dim=-1)
    q = torch.zeros_like(lm).scatter(1, near_idx, q_local)
    return -(q * F.log_softmax(lm, dim=-1)).sum(dim=-1).mean()

def topk_margin_ce_loss(logits, target, mask, k=5, margin=1.0):
    """
    Penalize only when target logit is not competitive with top-k logits.

    logits: (B,N,C)
    target: (B,N)
    mask:   (B,N) bool
    """
    mask = mask.bool()
    if int(mask.sum().item()) == 0:
        return logits.new_zeros(())

    lm = logits[mask]          # (M,C)
    ym = target[mask].long()   # (M,)

    target_logit = lm.gather(1, ym[:, None]).squeeze(1)  # (M,)

    topk_vals = lm.topk(min(k, lm.size(-1)), dim=-1).values
    kth_logit = topk_vals[:, -1]                         # (M,)

    # zero loss if target_logit >= kth_logit - margin
    return F.relu(kth_logit - target_logit + margin).mean()

@torch.no_grad()
def _masked_cls_metrics(logits, target, mask, num_classes: int, topk: int = 5):
    mask = mask.bool()
    n = int(mask.sum().item())

    if n == 0:
        return {
            "acc": 0.0,
            "topk_acc": 0.0,
            "mrr": 0.0,
            "mean_rank": 0.0,
            "entropy": 0.0,
            "unique": 0.0,
            "n": 0,
            "k": 0,
        }

    y = target[mask].long()
    logit_m = logits[mask]

    pred = logit_m.argmax(dim=-1)
    acc = (pred == y).float().mean().item()

    k = min(topk, logit_m.size(-1))
    topk_pred = logit_m.topk(k, dim=-1).indices
    topk_acc = (topk_pred == y.unsqueeze(-1)).any(dim=-1).float().mean().item()

    # Mean reciprocal rank of the true entry over the whole 961-way ranking.
    # Selection needs a metric that is not a loss but still sees the tail:
    # top-1 here is ~0.08 and top-5 ~0.36, so both discard most of the
    # ordering the model actually learns. MRR reads the full ranking and
    # moves when a target climbs from rank 40 to rank 12, which accuracy
    # cannot see. Strict '>' gives the optimistic rank under ties.
    target_logit = logit_m.gather(1, y.unsqueeze(1))
    rank = (logit_m > target_logit).sum(dim=-1) + 1
    mrr = (1.0 / rank.float()).mean().item()
    mean_rank = rank.float().mean().item()

    counts = torch.bincount(y, minlength=num_classes).float()
    probs = counts / counts.sum().clamp_min(1.0)
    probs = probs[probs > 0]
    entropy = -(probs * probs.log()).sum().item()

    unique = int((counts > 0).sum().item())

    return {
        "acc": acc,
        "topk_acc": topk_acc,
        "mrr": mrr,
        "mean_rank": mean_rank,
        "entropy": entropy,
        "unique": float(unique),
        "n": n,
        "k": k,
    }



def train_motif_prior_mgit(
    motif_prior,
    vqvae,
    opt,
    train_loader,
    val_loader=None,
    *,
    epochs: int = 20,
    grad_clip: float = 1.0,
    ckpt_out: str = "ckpts/motif_prior_best.pt",
    early_stop_patience: int = 5,
    min_delta: float = 0.0,
    scheduler=None,
    # Selection metric. "loss" is kept only so an old call reproduces the old
    # behaviour; the pipeline selects on a task metric, never on a loss.
    select_on: str = "mrr",
    # The dataset already hands validation a fixed crop/task/mask_spec
    # (DeterministicSubset, dataset.py). The MaskGIT corruption in
    # _make_motif_io does not go through it -- it draws from the main-process
    # RNG, which is why supervised tokens/sample swung 26.0-31.6 between epochs
    # and val loss carried an sd of 0.037 with no model change behind it.
    deterministic_val_masks: bool = True,
    val_mask_seed: int = 20260820,
    use_amp: bool = True,
    grad_accum_steps: int = 1,
    ensure_at_least_one_mask: bool = True,
    full_mask_prob: float = 0.15,
    log_every: int = 50,
    blank_code: int = None,
    loss_weights=(1.0, 1.0),
    
    lambda_topk: Union[float, tuple, list] = (1.0, 1.0),
    topk: Union[int, tuple, list] = (5, 2),
    topk_margin: float = 0.25,
    lambda_z1_distance: float = 0.05,
    lambda_z1_neighbor_ce: float = 0.25,
    z1_neighbor_tau: float = 0.25,
    
    
    lambda_ctx: float = 1.0,
    lambda_ctx_field: float = 0.05,
    lambda_adj: float = 1.0,
    lambda_spatial: float = 1.0,

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
    # ---- Stage 4D: adapt to the activity maps Stage 4B actually emits -----
    # None keeps this function bit-identical to Stage 4A. See
    # training/stage4d_adapt.py for why the adaptation runs on 4A rather than
    # on 4B (unbiased gradients over a frozen sampling distribution, versus a
    # straight-through surrogate through a top-K selection).
    adapt_activity_prior=None,
    adapt_ramp_epochs: int = 20,
    adapt_max_p: float = 0.8,
    adapt_readout: str = "gumbel",
    adapt_readout_tau: float = 1.0,
    adapt_soft_field: bool = True,
):
    """
    Stage 4A.

    Trains MaskGITMotifPrior only:
        teacher-forced activity + masked z1/z2 -> z1/z2

    Optionally adds voxel-domain biological losses through the frozen VQVAE
    decoder using teacher-forced activity.
    
    """

    device = next(motif_prior.parameters()).device
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    _freeze_module(vqvae)

    os.makedirs(os.path.dirname(ckpt_out) or ".", exist_ok=True)

    if blank_code is None:
        blank_code = getattr(vqvae.vq, "blank_code", -1)
        
    # ----------------------------------
    # normalize top-k configuration
    # ----------------------------------
    
    if isinstance(lambda_topk, (list, tuple)):
        if len(lambda_topk) != 2:
            raise ValueError(
                "lambda_topk must be scalar or length-2 sequence."
            )
        lambda_topk_z1 = float(lambda_topk[0])
        lambda_topk_z2 = float(lambda_topk[1])
    else:
        lambda_topk_z1 = float(lambda_topk)
        lambda_topk_z2 = float(lambda_topk)
    
    if isinstance(topk, (list, tuple)):
        if len(topk) != 2:
            raise ValueError(
                "topk must be scalar or length-2 sequence."
            )
        topk_z1 = int(topk[0])
        topk_z2 = int(topk[1])
    else:
        topk_z1 = int(topk)
        topk_z2 = int(topk)

    def _make_motif_io(codes, pmask):
        targets = motif_prior.make_targets_from_codes(
            codes=codes,
            predict_mask=pmask,
            blank_code=blank_code,
        )

        a_in, f_in, targets = motif_prior.corrupt_inputs_from_targets(
            targets,
            ensure_at_least_one_mask=ensure_at_least_one_mask,
            full_mask_prob=full_mask_prob,
        )
        return a_in, f_in, targets

    def _run_epoch(loader, train: bool, adapt_p: Optional[float] = None):
        motif_prior.train(train)
        if adapt_activity_prior is not None:
            adapt_activity_prior.eval()

        total_loss = 0.0
        total_cnt = 0.0

        total_z1 = 0.0
        total_z2 = 0.0
        total_z1_cnt = 0.0
        total_z2_cnt = 0.0

        total_z1_acc = 0.0
        total_z1_mrr = 0.0
        total_z1_rank = 0.0
        total_z2_acc = 0.0
        total_z1_topk = 0.0
        total_z2_topk = 0.0
        total_z1_ent = 0.0
        total_z2_ent = 0.0
        total_z1_unique = 0.0
        total_z2_unique = 0.0
        
        total_ce = 0.0
        total_ctx = 0.0
        total_ctx_field = 0.0
        total_adj = 0.0
        total_spatial = 0.0
        total_topk_loss = 0.0

        total_gamma = 0.0
        total_batches = 0.0
        total_samples = 0.0
        total_active_tokens = 0.0
        topk_used_z1 = 0
        topk_used_z2 = 0

        for it, batch in enumerate(loader, start=1):
            x, gct, lct, task_id, mask_spec = _batch_to_device(batch, device)

            with torch.no_grad():
                codes, pmask, grid = _vq_codes_and_pmask_for_prior(
                    vqvae, x, gct, lct, mask_spec, device
                )
                if adapt_activity_prior is None:
                    a_in, f_in, targets = _make_motif_io(codes, pmask)
                    activity_prob = None
                else:
                    a_in, f_in, targets, activity_prob = build_adapted_motif_io(
                        motif_prior=motif_prior,
                        activity_prior=adapt_activity_prior,
                        codes=codes, predict_mask=pmask,
                        global_ctx=gct, local_ctx=lct, task_id=task_id,
                        blank_code=blank_code,
                        p_model=float(adapt_p if adapt_p is not None else 0.0),
                        readout=adapt_readout, readout_tau=adapt_readout_tau,
                        full_mask_prob=full_mask_prob,
                        ensure_at_least_one_mask=ensure_at_least_one_mask,
                        use_soft_field=bool(adapt_soft_field),
                    )

            if train:
                if (it - 1) % grad_accum_steps == 0:
                    opt.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    logits, loss_ce, aux = motif_prior(
                        a_in,
                        f_in,
                        global_ctx=gct,
                        local_ctx=lct,
                        task_id=task_id,
                        targets=targets,
                        loss_weights=loss_weights,
                        activity_prob=activity_prob,
                    )
                    
                    loss_ce_raw = loss_ce

                    # Distance-weighted auxiliaries on the flat alphabet.
                    flat_valid = targets["f_loss_mask"]
                    loss_topk_z1 = distance_neighborhood_ce_loss(
                        logits["flat"],
                        targets["f"],
                        flat_valid,
                        motif_prior.flat_distance_matrix,
                        k=topk_z1,
                        tau=z1_neighbor_tau,
                    )
                    loss_z1_distance = expected_code_distance_loss(
                        logits["flat"],
                        targets["f"],
                        flat_valid,
                        motif_prior.flat_distance_matrix,
                    )
                    loss_topk_z2 = logits["flat"].sum() * 0.0
                    loss_topk = (
                        float(lambda_z1_neighbor_ce) * loss_topk_z1
                        + float(lambda_z1_distance) * loss_z1_distance
                    )
                    
                    loss_ce = loss_ce_raw + loss_topk
                                        

                    loss_ctx = loss_ce.new_zeros(())
                    loss_ctx_field = loss_ce.new_zeros(())
                    loss_adj = loss_ce.new_zeros(())
                    loss_spatial = loss_ce.new_zeros(())
                
                    if lambda_ctx > 0.0 or lambda_ctx_field > 0.0 or lambda_adj > 0.0 or lambda_spatial > 0.0:
                        dec = decode_motif_logits_soft_given_activity(
                            model=vqvae,
                            logits=logits,
                            targets=targets,
                            motif_prior=motif_prior,
                            activity_ids=targets["a"],
                            grid=grid,
                            global_ctx=gct,
                            local_ctx=lct,
                            tau_z=motif_tau_z,
                            roi_hw=batch.get("roi_hw", None),
                            pad_hw=batch.get("pad_hw", None),
                        )
                
                        logits_vol = dec["logits_vol"]
                
                        if lambda_ctx > 0.0 and lct is not None:
                            loss_ctx = ctx_loss_soft(
                                logits_b1thw=logits_vol,
                                ctx_tgt_b9=lct,
                                dims=tuple(range(9)),
                                tau=ctx_tau,
                                prob_threshold=float(
                                    float(vqvae.best_thr_tol.item())
                                ),
                            )
                        
                        if lambda_ctx_field > 0.0:
                            _, _, T_dec, H_dec, W_dec = logits_vol.shape
                        
                            target_vol = x[
                                :,
                                :1,
                                :T_dec,
                                :H_dec,
                                :W_dec,
                            ]
                        
                            if target_vol.shape != logits_vol.shape:
                                raise RuntimeError(
                                    "Stage 4A local field target/decoder shape mismatch: "
                                    f"target={tuple(target_vol.shape)}, "
                                    f"logits={tuple(logits_vol.shape)}"
                                )
                        
                            loss_ctx_field = local_moment_field_loss(
                                logits_b1thw=logits_vol,
                                target_b1thw=target_vol,
                                patch_size=vqvae.patch_size,
                                tau=ctx_field_tau,
                                min_active_spikes=1,
                                min_shape_spikes=5,
                                min_trend_spikes=6,
                                min_trend_frames=3,
                                prob_threshold=float(
                                    float(vqvae.best_thr_tol.item())
                                ),
                            )
                            
                            
                
                        if lambda_adj > 0.0:
                            if memory_adj is None:
                                raise RuntimeError("lambda_adj > 0 but memory_adj is None.")
                
                            adj_target_bg = memory_adj.get(
                                gct,
                                device=device,
                                dtype=torch.float32,
                            )
                
                            adj_conf_bg = memory_adj.get_confidence(
                                gct,
                                device=device,
                                dtype=torch.float32,
                                den_scale=memory_adj_conf_den_scale,
                            )
                
                            adj_parts = short_gap_excess_loss_from_logits_batch_targets(
                                logits_b1thw=logits_vol.float(),
                                target_gap_rates_bg=adj_target_bg.float(),
                                max_gap=isi_max_gap,
                                gap_bins=isi_gap_bins,
                                tau=isi_tau,
                                prob_threshold=float(
                                    float(vqvae.best_thr_tol.item())
                                ),
                                margin=isi_margin,
                                lower_margin=isi_lower_margin,
                                lower_weight=isi_lower_weight,
                                confidence_bg=adj_conf_bg.float(),
                                return_parts=True,
                            )
                
                            loss_adj = torch.nan_to_num(
                                adj_parts["loss"],
                                nan=0.0,
                                posinf=1e3,
                                neginf=0.0,
                            )
                
                        if lambda_spatial > 0.0 and memory_tok is not None and gct is not None:
                            full_teacher_tok = memory_tok.get(
                                gct,
                                device=device,
                                dtype=logits_vol.dtype,
                            )
                
                            _, _, _, Hp, Wp = logits_vol.shape
                            _, pH, pW = vqvae.patch_size
                            h_tok = Hp // pH
                            w_tok = Wp // pW
                
                            teacher_tok = sample_full_token_map_to_crop(
                                full_tok_bhw=full_teacher_tok,
                                out_tok_hw=(h_tok, w_tok),
                                patch_size=vqvae.patch_size,
                                roi_hw=batch.get("roi_hw", None),
                                pad_hw=batch.get("pad_hw", None),
                            )
                
                            student_tok = soft_spatial_token_map_from_logits(
                                logits_b1thw=logits_vol,
                                patch_size=vqvae.patch_size,
                                tau=0.25,
                                prob_threshold=float(
                                    float(vqvae.best_thr_tol.item())
                                ),
                            )
                
                            teacher_tok_tol = dilate_spatial_support_hw(
                                teacher_tok.detach(),
                                radius_h=1,
                                radius_w=1,
                            )
                
                            loss_spatial = spatial_support_violation_loss(
                                pred_support=student_tok,
                                allowed_support=teacher_tok_tol,
                                neg_thresh=0.20,
                            )
                
                    loss = (
                        loss_ce
                        + float(lambda_ctx) * loss_ctx
                        + float(lambda_ctx_field) * loss_ctx_field
                        + float(lambda_adj) * loss_adj
                        + float(lambda_spatial) * loss_spatial
                    )

                scaler.scale(loss / float(grad_accum_steps)).backward()

                do_step = (it % grad_accum_steps == 0) or (it == len(loader))
                if do_step:
                    if grad_clip is not None and grad_clip > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(
                            motif_prior.parameters(),
                            float(grad_clip),
                        )
                    scaler.step(opt)
                    scaler.update()

            else:
                with torch.no_grad():
                    logits, loss_ce, aux = motif_prior(
                        a_in,
                        f_in,
                        global_ctx=gct,
                        local_ctx=lct,
                        task_id=task_id,
                        targets=targets,
                        loss_weights=loss_weights,
                        activity_prob=activity_prob,
                    )
                    
                    loss_ce_raw = loss_ce
                    
                    # Distance-weighted auxiliaries on the flat alphabet.
                    flat_valid = targets["f_loss_mask"]
                    loss_topk_z1 = distance_neighborhood_ce_loss(
                        logits["flat"],
                        targets["f"],
                        flat_valid,
                        motif_prior.flat_distance_matrix,
                        k=topk_z1,
                        tau=z1_neighbor_tau,
                    )
                    loss_z1_distance = expected_code_distance_loss(
                        logits["flat"],
                        targets["f"],
                        flat_valid,
                        motif_prior.flat_distance_matrix,
                    )
                    loss_topk_z2 = logits["flat"].sum() * 0.0
                    loss_topk = (
                        float(lambda_z1_neighbor_ce) * loss_topk_z1
                        + float(lambda_z1_distance) * loss_z1_distance
                    )
                    
                    loss_ce = loss_ce_raw + loss_topk
                
                    loss_ctx = loss_ce.new_zeros(())
                    loss_ctx_field = loss_ce.new_zeros(())
                    loss_adj = loss_ce.new_zeros(())
                    loss_spatial = loss_ce.new_zeros(())
                
                    if lambda_ctx > 0.0 or lambda_ctx_field > 0.0 or lambda_adj > 0.0 or lambda_spatial > 0.0:
                        dec = decode_motif_logits_soft_given_activity(
                            model=vqvae,
                            logits=logits,
                            targets=targets,
                            motif_prior=motif_prior,
                            activity_ids=targets["a"],
                            grid=grid,
                            global_ctx=gct,
                            local_ctx=lct,
                            tau_z=motif_tau_z,
                            roi_hw=batch.get("roi_hw", None),
                            pad_hw=batch.get("pad_hw", None),
                        )
                
                        logits_vol = dec["logits_vol"]
                        
                        if lambda_ctx > 0.0 and lct is not None:
                            loss_ctx = ctx_loss_soft(
                                logits_b1thw=logits_vol,
                                ctx_tgt_b9=lct,
                                dims=tuple(range(9)),
                                tau=ctx_tau,
                                prob_threshold=float(
                                    float(vqvae.best_thr_tol.item())
                                ),
                            )
                        
                        if lambda_ctx_field > 0.0:
                            _, _, T_dec, H_dec, W_dec = logits_vol.shape
                        
                            target_vol = x[
                                :,
                                :1,
                                :T_dec,
                                :H_dec,
                                :W_dec,
                            ]
                        
                            if target_vol.shape != logits_vol.shape:
                                raise RuntimeError(
                                    "Stage 4A validation local field target/decoder "
                                    "shape mismatch: "
                                    f"target={tuple(target_vol.shape)}, "
                                    f"logits={tuple(logits_vol.shape)}"
                                )
                        
                            loss_ctx_field = local_moment_field_loss(
                                logits_b1thw=logits_vol,
                                target_b1thw=target_vol,
                                patch_size=vqvae.patch_size,
                                tau=ctx_field_tau,
                                min_active_spikes=1,
                                min_shape_spikes=5,
                                min_trend_spikes=6,
                                min_trend_frames=3,
                                prob_threshold=float(
                                    float(vqvae.best_thr_tol.item())
                                ),
                            )
                
                        if lambda_adj > 0.0:
                            if memory_adj is None:
                                raise RuntimeError("lambda_adj > 0 but memory_adj is None.")
                
                            adj_target_bg = memory_adj.get(
                                gct,
                                device=device,
                                dtype=torch.float32,
                            )
                
                            adj_conf_bg = memory_adj.get_confidence(
                                gct,
                                device=device,
                                dtype=torch.float32,
                                den_scale=memory_adj_conf_den_scale,
                            )
                
                            adj_parts = short_gap_excess_loss_from_logits_batch_targets(
                                logits_b1thw=logits_vol.float(),
                                target_gap_rates_bg=adj_target_bg.float(),
                                max_gap=isi_max_gap,
                                gap_bins=isi_gap_bins,
                                tau=isi_tau,
                                prob_threshold=float(
                                    float(vqvae.best_thr_tol.item())
                                ),
                                margin=isi_margin,
                                lower_margin=isi_lower_margin,
                                lower_weight=isi_lower_weight,
                                confidence_bg=adj_conf_bg.float(),
                                return_parts=True,
                            )
                
                            loss_adj = torch.nan_to_num(
                                adj_parts["loss"],
                                nan=0.0,
                                posinf=1e3,
                                neginf=0.0,
                            )
                
                        if lambda_spatial > 0.0 and memory_tok is not None and gct is not None:
                            full_teacher_tok = memory_tok.get(
                                gct,
                                device=device,
                                dtype=logits_vol.dtype,
                            )
                
                            _, _, _, Hp, Wp = logits_vol.shape
                            _, pH, pW = vqvae.patch_size
                            h_tok = Hp // pH
                            w_tok = Wp // pW
                
                            teacher_tok = sample_full_token_map_to_crop(
                                full_tok_bhw=full_teacher_tok,
                                out_tok_hw=(h_tok, w_tok),
                                patch_size=vqvae.patch_size,
                                roi_hw=batch.get("roi_hw", None),
                                pad_hw=batch.get("pad_hw", None),
                            )
                
                            student_tok = soft_spatial_token_map_from_logits(
                                logits_b1thw=logits_vol,
                                patch_size=vqvae.patch_size,
                                tau=0.25,
                                prob_threshold=float(
                                    float(vqvae.best_thr_tol.item())
                                ),
                            )
                
                            teacher_tok_tol = dilate_spatial_support_hw(
                                teacher_tok.detach(),
                                radius_h=1,
                                radius_w=1,
                            )
                
                            loss_spatial = spatial_support_violation_loss(
                                pred_support=student_tok,
                                allowed_support=teacher_tok_tol,
                                neg_thresh=0.20,
                            )
                
                    loss = (
                        loss_ce
                        + float(lambda_ctx) * loss_ctx
                        + float(lambda_ctx_field) * loss_ctx_field
                        + float(lambda_adj) * loss_adj
                        + float(lambda_spatial) * loss_spatial
                    )

            z1_mask = targets["f_loss_mask"].bool()
            z2_mask = z1_mask

            n_z1 = float(z1_mask.sum().item())
            n_z2 = n_z1

            # One stream now: both metric slots report the flat alphabet, so
            # existing report keys keep working.
            mz1 = _masked_cls_metrics(
                logits["flat"],
                targets["f"],
                z1_mask,
                motif_prior.V,
                topk=topk_z1,
            )
            mz2 = mz1

            B = x.size(0)

            total_loss += float(loss.item()) * float(B)
            total_ce += float(loss_ce_raw.item()) * float(B)
            total_topk_loss += float(loss_topk.item()) * float(B)

            total_ctx += float(loss_ctx.item()) * float(B)
            total_ctx_field += float(loss_ctx_field.item()) * float(B)
            total_adj += float(loss_adj.item()) * float(B)
            total_spatial += float(loss_spatial.item()) * float(B)
            total_cnt += float(B)

            # One stream: both slots report the flat-alphabet CE so the
            # existing z1/z2 report keys stay populated.
            if n_z1 > 0:
                total_z1 += float(aux["loss_flat"].item()) * n_z1
                total_z1_cnt += n_z1
                total_z2 += float(aux["loss_flat"].item()) * n_z1
                total_z2_cnt += n_z1

            total_z1_mrr += mz1["mrr"] * mz1["n"]
            total_z1_rank += mz1["mean_rank"] * mz1["n"]
            total_z1_acc += mz1["acc"] * mz1["n"]
            total_z2_acc += mz2["acc"] * mz2["n"]
            total_z1_topk += mz1["topk_acc"] * mz1["n"]
            total_z2_topk += mz2["topk_acc"] * mz2["n"]
            total_z1_ent += mz1["entropy"] * mz1["n"]
            total_z2_ent += mz2["entropy"] * mz2["n"]
            total_z1_unique += mz1["unique"]
            total_z2_unique += mz2["unique"]

            topk_used_z1 = max(topk_used_z1, mz1["k"])
            topk_used_z2 = max(topk_used_z2, mz2["k"])

            total_gamma += float(targets["gamma"].mean().item())
            total_batches += 1.0
            total_samples += float(B)
            total_active_tokens += float(targets["active"].sum().item())

            if train and log_every and (it % log_every == 0):
                print(
                    f"  [4A motif] it {it:05d}: "
                    f"loss={total_loss / max(total_cnt, 1.0):.4f} "
                    f"ce={total_ce / max(total_cnt, 1.0):.4f} "
                    f"topk_ce={total_topk_loss / max(total_cnt, 1.0):.4f} "
                    f"ctx={total_ctx / max(total_cnt, 1.0):.4f} "
                    f"ctx_field={total_ctx_field / max(total_cnt, 1.0):.4f} "
                    f"adj={total_adj / max(total_cnt, 1.0):.4f} "
                    f"sp={total_spatial / max(total_cnt, 1.0):.4f} "
                    f"z1={total_z1 / max(total_z1_cnt, 1.0):.4f} "
                    f"z2={total_z2 / max(total_z2_cnt, 1.0):.4f} "
                    f"acc_z1={total_z1_acc / max(total_z1_cnt, 1.0):.3f} "
                    f"acc_z2={total_z2_acc / max(total_z2_cnt, 1.0):.3f} "
                    f"acc_topk_z1={total_z1_topk / max(total_z1_cnt, 1.0):.3f} "
                    f"acc_topk_z2={total_z2_topk / max(total_z2_cnt, 1.0):.3f} "
                    f"gamma={total_gamma / max(total_batches, 1.0):.3f}"
                )

        den = max(total_cnt, 1.0)
        den_z1 = max(total_z1_cnt, 1.0)
        den_z2 = max(total_z2_cnt, 1.0)

        return {
            "loss": total_loss / den,
            "loss_z1": total_z1 / den_z1,
            "loss_z2": total_z2 / den_z2,
            
            "loss_ce": total_ce / den,
            "loss_topk": total_topk_loss / den,
            "loss_ctx": total_ctx / den,
            "loss_ctx_field": total_ctx_field / den,
            "loss_adj": total_adj / den,
            "loss_spatial": total_spatial / den,
            
            "acc_z1": total_z1_acc / den_z1,
            "mrr_z1": total_z1_mrr / den_z1,
            "mean_rank_z1": total_z1_rank / den_z1,
            "acc_z2": total_z2_acc / den_z2,
            "topk_z1": topk_used_z1,
            "topk_z2": topk_used_z2,
            "topk_acc_z1": total_z1_topk / den_z1,
            "topk_acc_z2": total_z2_topk / den_z2,
            "entropy_z1": total_z1_ent / den_z1,
            "entropy_z2": total_z2_ent / den_z2,
            "ce_minus_entropy_z1": (total_z1 / den_z1) - (total_z1_ent / den_z1),
            "ce_minus_entropy_z2": (total_z2 / den_z2) - (total_z2_ent / den_z2),
            "gamma": total_gamma / max(total_batches, 1.0),
            "active_tokens_per_sample": total_active_tokens / max(total_samples, 1.0),
            "supervised_z1_tokens_per_sample": total_z1_cnt / max(total_samples, 1.0),
            "supervised_z2_tokens_per_sample": total_z2_cnt / max(total_samples, 1.0),
            "unique_z1_per_batch": total_z1_unique / max(total_batches, 1.0),
            "unique_z2_per_batch": total_z2_unique / max(total_batches, 1.0),
        }

    # Selection metric. Loss is not used: as the model sharpens, confident
    # errors raise cross-entropy even while the ranking improves, so the
    # loss-best epoch can be materially worse at the actual task.
    #
    #   mrr   -- mean reciprocal rank of the true entry (default)
    #   acc   -- top-1
    #   topk  -- top-k as configured by `topk`
    #   loss  -- legacy, reproduces the pre-2026-08-20 behaviour
    _SELECT_KEYS = {
        "mrr": ("mrr_z1", False),
        "acc": ("acc_z1", False),
        "topk": ("topk_acc_z1", False),
        "loss": ("loss", True),
    }
    if select_on not in _SELECT_KEYS:
        raise ValueError(
            f"select_on must be one of {sorted(_SELECT_KEYS)}, got {select_on!r}"
        )
    select_key, select_lower_is_better = _SELECT_KEYS[select_on]

    best_score = float("inf") if select_lower_is_better else -float("inf")
    patience = 0

    # Secondary checkpoints, saved but never used for early stopping. They cost
    # one write each and make an alternative selection rule available after the
    # fact instead of requiring a retrain to ask the question.
    best_val_loss = float("inf")
    best_val_z1_acc = -1.0
    ckpt_loss_out = os.path.splitext(ckpt_out)[0] + "_best_loss.pt"
    ckpt_acc_out = os.path.splitext(ckpt_out)[0] + "_best_z1acc.pt"
    history = {"train": [], "val": []}

    for ep in range(1, epochs + 1):
        
        # No teacher-forcing schedule: there is no second cascade stage to
        # condition on a committed parent. One flat head, trained against its
        # own predictions throughout.
        # Linear ramp from Stage 4A's own regime (p=0) to adapt_max_p.
        adapt_p = (adapt_probability(ep, adapt_ramp_epochs, adapt_max_p)
                   if adapt_activity_prior is not None else None)
        train_m = _run_epoch(train_loader, train=True, adapt_p=adapt_p)

        if val_loader is None:
            val_m = train_m
        elif deterministic_val_masks:
            # Same corruption pattern every epoch, so consecutive validation
            # numbers measure the model and nothing else. State is saved and
            # restored so training's RNG stream is untouched.
            cpu_state = torch.get_rng_state()
            cuda_state = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
            try:
                torch.manual_seed(int(val_mask_seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(val_mask_seed))
                val_m = _run_epoch(
                    val_loader, train=False,
                    adapt_p=(1.0 if adapt_activity_prior is not None else None))
                if adapt_activity_prior is not None:
                    # Oracle arm, same seed so the corruption pattern matches
                    # and the two arms differ only in the activity map. Logged,
                    # never selected on: it is the guard that adapting to a
                    # noisy map has not destroyed the model's ability to use a
                    # good one, which the inpainting tasks still need.
                    torch.manual_seed(int(val_mask_seed))
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(int(val_mask_seed))
                    oracle_m = _run_epoch(val_loader, train=False, adapt_p=0.0)
                    val_m = dict(val_m)
                    val_m["oracle_mrr_z1"] = oracle_m["mrr_z1"]
                    val_m["oracle_mean_rank_z1"] = oracle_m["mean_rank_z1"]
                    val_m["oracle_loss_ce"] = oracle_m["loss_ce"]
                    val_m["adapt_p"] = float(adapt_p)
            finally:
                torch.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state_all(cuda_state)
        else:
            val_m = _run_epoch(
                val_loader, train=False,
                adapt_p=(1.0 if adapt_activity_prior is not None else None))

        history["train"].append(train_m)
        history["val"].append(val_m)

        topk_z1_used = val_m["topk_z1"]
        topk_z2_used = val_m["topk_z2"]

        print(
            f"[motif epoch {ep:03d}] "
            f"train loss={train_m['loss']:.4f} "
            f"val loss={val_m['loss']:.4f} "
            f"ce={val_m['loss_ce']:.4f} "
            f"topk_ce={val_m['loss_topk']:.4f} "
            f"ctx={val_m['loss_ctx']:.4f} "
            f"ctx_field={val_m['loss_ctx_field']:.4f} "
            f"adj={val_m['loss_adj']:.4f} "
            f"sp={val_m['loss_spatial']:.4f} "
            f"z1={val_m['loss_z1']:.4f} "
            f"z2={val_m['loss_z2']:.4f} "
            f"mrr={val_m['mrr_z1']:.4f} "
            f"rank={val_m['mean_rank_z1']:.1f} "
            f"acc_z1={val_m['acc_z1']:.3f} "
            f"acc_z2={val_m['acc_z2']:.3f} "
            f"top{topk_z1_used}_z1={val_m['topk_acc_z1']:.3f} "
            f"top{topk_z2_used}_z2={val_m['topk_acc_z2']:.3f} "
            f"H_z1={val_m['entropy_z1']:.3f} "
            f"H_z2={val_m['entropy_z2']:.3f} "
            f"CE-H_z1={val_m['ce_minus_entropy_z1']:.3f} "
            f"CE-H_z2={val_m['ce_minus_entropy_z2']:.3f} "
            + (f"p={val_m['adapt_p']:.2f} "
               f"ORACLEmrr={val_m['oracle_mrr_z1']:.4f} "
               if 'oracle_mrr_z1' in val_m else "") +
            f"act/sample={val_m['active_tokens_per_sample']:.1f} "
            f"sup_z1/sample={val_m['supervised_z1_tokens_per_sample']:.1f} "
            f"sup_z2/sample={val_m['supervised_z2_tokens_per_sample']:.1f}"
        )

        # Secondary: best top-1. Key is "acc_z1" -- the metrics dict has never
        # emitted "z1_acc", so the previous read silently returned its default
        # and this checkpoint was never written.
        current_z1_acc = float(val_m.get("acc_z1", -1.0))
        if current_z1_acc > best_val_z1_acc:
            best_val_z1_acc = current_z1_acc
            torch.save(
                {
                    # Key must be "model" to match the primary checkpoint and
                    # main._load_stage4_motif_best, which reads ckpt["model"].
                    "model": motif_prior.state_dict(),
                    "epoch": ep,
                    "best_val_z1_acc": best_val_z1_acc,
                    "val_loss_at_best_acc": float(val_m["loss"]),
                },
                ckpt_acc_out,
            )

        # Secondary: best loss, for comparison against the old selection rule.
        current_loss = float(val_m["loss"])
        if current_loss < best_val_loss:
            best_val_loss = current_loss
            torch.save(
                {
                    "model": motif_prior.state_dict(),
                    "epoch": ep,
                    "best_val_loss": best_val_loss,
                },
                ckpt_loss_out,
            )

        # Primary: the configured task metric drives both the shipped
        # checkpoint and early stopping.
        score = float(val_m[select_key])
        improved = (
            score < best_score - float(min_delta)
            if select_lower_is_better
            else score > best_score + float(min_delta)
        )
        if improved:
            best_score = score
            patience = 0
            torch.save(
                {
                    "model": motif_prior.state_dict(),
                    "epoch": ep,
                    "select_on": select_on,
                    "best_score": best_score,
                    "val_loss_at_best": current_loss,
                    "val_acc_at_best": current_z1_acc,
                    "val_mrr_at_best": float(val_m["mrr_z1"]),
                },
                ckpt_out,
            )
            print(f"  saved {ckpt_out}  best {select_on}={best_score:.5f}")
        else:
            patience += 1
            if patience >= int(early_stop_patience):
                print(
                    f"Early stopping at epoch {ep}; "
                    f"best {select_on}={best_score:.5f}"
                )
                break

        if scheduler is not None:
            scheduler.step()

    return history








# Stage 4A and the shared encoding helpers live in this module. Stage 4B/3C are
# in stage4_activity.py and are imported from there directly; the backward-compat
# shims that used to re-export them here went with the DETR implementations.