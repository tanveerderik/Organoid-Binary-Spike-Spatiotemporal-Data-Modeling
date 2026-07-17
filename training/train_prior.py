#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F

from typing import Union

from ..model.prior import (
    build_activity_targets_from_codes,
    detr_activity_loss,
)

from ..inference.decode import decode_motif_logits_soft_given_activity

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

    counts = torch.bincount(y, minlength=num_classes).float()
    probs = counts / counts.sum().clamp_min(1.0)
    probs = probs[probs > 0]
    entropy = -(probs * probs.log()).sum().item()

    unique = int((counts > 0).sum().item())

    return {
        "acc": acc,
        "topk_acc": topk_acc,
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
    use_amp: bool = True,
    grad_accum_steps: int = 1,
    ensure_at_least_one_mask: bool = True,
    full_mask_prob: float = 0.15,
    log_every: int = 50,
    blank_code: int = None,
    loss_weights=(5.0, 0.5),
    
    lambda_topk: Union[float, tuple, list] = (5, 0.5),
    topk: Union[int, tuple, list] = (5, 2),
    topk_margin: float = 0.25,
    
    z1_teacher_prob_start: float = 1.0,
    z1_teacher_prob_end: float = 0.0,
    z1_teacher_decay_epochs: int = 50,
    
    lambda_ctx: float = 1.0,
    lambda_ctx_field: float = 0.05,
    lambda_adj: float = 1.0,
    lambda_spatial: float = 1.0,

    ctx_tau: float = 0.25,
    ctx_field_tau: float = 0.25,
    
    motif_tau_z: float = 0.25,
    isi_tau: float = 0.25,
    isi_margin: float = 0.25,
    isi_max_gap: int = 3,
    isi_gap_bins=None,
    memory_adj=None,
    memory_tok=None,
    memory_adj_conf_den_scale: float = 100.0,
):
    """
    Stage 3A.

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
        
        a_in, z1_in, z2_in, targets = motif_prior.corrupt_inputs_from_targets(
            targets,
            ensure_at_least_one_mask=ensure_at_least_one_mask,
            full_mask_prob=full_mask_prob,
        )
        targets["z1_teacher_prob"] = z1_teacher_prob

        return a_in, z1_in, z2_in, targets

    def _run_epoch(loader, train: bool):
        motif_prior.train(train)

        total_loss = 0.0
        total_cnt = 0.0

        total_z1 = 0.0
        total_z2 = 0.0
        total_z1_cnt = 0.0
        total_z2_cnt = 0.0

        total_z1_acc = 0.0
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
                codes, pmask, grid = _vq_codes_and_pmask(vqvae, x, gct, lct, mask_spec, device)
                a_in, z1_in, z2_in, targets = _make_motif_io(codes, pmask)
                
                if not train:
                    targets["z1_teacher_prob"] = 0.0

            if train:
                if (it - 1) % grad_accum_steps == 0:
                    opt.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    logits, loss_ce, aux = motif_prior(
                        a_in,
                        z1_in,
                        z2_in,
                        global_ctx=gct,
                        local_ctx=lct,
                        task_id=task_id,
                        targets=targets,
                        loss_weights=loss_weights,
                    )
                    
                    loss_ce_raw = loss_ce

                    loss_topk_z1 = topk_margin_ce_loss(
                        logits["z1"],
                        targets["z1"],
                        targets["z1_loss_mask"],
                        k=topk_z1,
                        margin=topk_margin,
                    )
                    
                    loss_topk_z2 = topk_margin_ce_loss(
                        logits["z2"],
                        targets["z2"],
                        targets["z2_loss_mask"],
                        k=topk_z2,
                        margin=topk_margin,
                    )
                    
                    loss_topk = (
                        float(lambda_topk_z1) * loss_topk_z1
                        + float(lambda_topk_z2) * loss_topk_z2
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
                                    "Stage 3A local field target/decoder shape mismatch: "
                                    f"target={tuple(target_vol.shape)}, "
                                    f"logits={tuple(logits_vol.shape)}"
                                )
                        
                            loss_ctx_field = local_moment_field_loss(
                                logits_b1thw=logits_vol,
                                target_b1thw=target_vol,
                                patch_size=vqvae.patch_size,
                                tau=ctx_field_tau,
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
                                margin=isi_margin,
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
                        z1_in,
                        z2_in,
                        global_ctx=gct,
                        local_ctx=lct,
                        task_id=task_id,
                        targets=targets,
                        loss_weights=loss_weights,
                    )
                    
                    loss_ce_raw = loss_ce
                    
                    loss_topk_z1 = topk_margin_ce_loss(
                        logits["z1"],
                        targets["z1"],
                        targets["z1_loss_mask"],
                        k=topk_z1,
                        margin=topk_margin,
                    )
                    
                    loss_topk_z2 = topk_margin_ce_loss(
                        logits["z2"],
                        targets["z2"],
                        targets["z2_loss_mask"],
                        k=topk_z2,
                        margin=topk_margin,
                    )
                    
                    loss_topk = (
                        float(lambda_topk_z1) * loss_topk_z1
                        + float(lambda_topk_z2) * loss_topk_z2
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
                                    "Stage 3A validation local field target/decoder "
                                    "shape mismatch: "
                                    f"target={tuple(target_vol.shape)}, "
                                    f"logits={tuple(logits_vol.shape)}"
                                )
                        
                            loss_ctx_field = local_moment_field_loss(
                                logits_b1thw=logits_vol,
                                target_b1thw=target_vol,
                                patch_size=vqvae.patch_size,
                                tau=ctx_field_tau,
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
                                margin=isi_margin,
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

            z1_mask = targets["z1_loss_mask"].bool()
            z2_mask = targets["z2_loss_mask"].bool()

            n_z1 = float(z1_mask.sum().item())
            n_z2 = float(z2_mask.sum().item())

            mz1 = _masked_cls_metrics(
                logits["z1"],
                targets["z1"],
                z1_mask,
                motif_prior.K1,
                topk=topk_z1,
            )
            mz2 = _masked_cls_metrics(
                logits["z2"],
                targets["z2"],
                z2_mask,
                motif_prior.K2,
                topk=topk_z2,
            )

            B = x.size(0)

            total_loss += float(loss.item()) * float(B)
            total_ce += float(loss_ce_raw.item()) * float(B)
            total_topk_loss += float(loss_topk.item()) * float(B)

            total_ctx += float(loss_ctx.item()) * float(B)
            total_ctx_field += float(loss_ctx_field.item()) * float(B)
            total_adj += float(loss_adj.item()) * float(B)
            total_spatial += float(loss_spatial.item()) * float(B)
            total_cnt += float(B)

            if n_z1 > 0:
                total_z1 += float(aux["loss_z1"].item()) * n_z1
                total_z1_cnt += n_z1

            if n_z2 > 0:
                total_z2 += float(aux["loss_z2"].item()) * n_z2
                total_z2_cnt += n_z2

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
                    f"  [3A motif] it {it:05d}: "
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

    best_val = float("inf")
    patience = 0
    history = {"train": [], "val": []}

    for ep in range(1, epochs + 1):
        
        if z1_teacher_decay_epochs <= 0:
            z1_teacher_prob = float(z1_teacher_prob_end)
        else:
            alpha = min(1.0, max(0.0, (ep - 1) / float(z1_teacher_decay_epochs)))
            z1_teacher_prob = (
                (1.0 - alpha) * float(z1_teacher_prob_start)
                + alpha * float(z1_teacher_prob_end)
            )
        
        train_m = _run_epoch(train_loader, train=True)
        val_m = _run_epoch(val_loader, train=False) if val_loader is not None else train_m

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
            f"acc_z1={val_m['acc_z1']:.3f} "
            f"acc_z2={val_m['acc_z2']:.3f} "
            f"top{topk_z1_used}_z1={val_m['topk_acc_z1']:.3f} "
            f"top{topk_z2_used}_z2={val_m['topk_acc_z2']:.3f} "
            f"H_z1={val_m['entropy_z1']:.3f} "
            f"H_z2={val_m['entropy_z2']:.3f} "
            f"CE-H_z1={val_m['ce_minus_entropy_z1']:.3f} "
            f"CE-H_z2={val_m['ce_minus_entropy_z2']:.3f} "
            f"act/sample={val_m['active_tokens_per_sample']:.1f} "
            f"sup_z1/sample={val_m['supervised_z1_tokens_per_sample']:.1f} "
            f"sup_z2/sample={val_m['supervised_z2_tokens_per_sample']:.1f}"
        )

        if val_m["loss"] < best_val - float(min_delta):
            best_val = val_m["loss"]
            patience = 0
            torch.save(
                {
                    "model": motif_prior.state_dict(),
                    "epoch": ep,
                    "best_val_loss": best_val,
                },
                ckpt_out,
            )
            print(f"  saved {ckpt_out}  best={best_val:.4f}")
        else:
            patience += 1
            if patience >= int(early_stop_patience):
                print(f"Early stopping at epoch {ep}; best={best_val:.4f}")
                break

    return history



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
    early_stop_patience: int = 5,
    min_delta: float = 0.0,
    use_amp: bool = True,
    grad_accum_steps: int = 1,
    log_every: int = 50,
    blank_code: int = None,
    lambda_count: float = 1.0,
    lambda_obj: float = 1.0,
    lambda_coord: float = 1.0,
    lambda_soft_count: float = 0.1,
    lambda_soft_grid: float = 1.0,
    lambda_dup: float = 0.01,
    no_object_weight: float = 0.1,
):
    """
    Stage 3B.

    Trains DETRActivityPrior only:
        global_ctx + local_ctx + task_id -> count + sparse activity coordinates

    VQVAE is frozen and only provides target codes.
    """

    device = next(activity_prior.parameters()).device
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    _freeze_module(vqvae)

    os.makedirs(os.path.dirname(ckpt_out) or ".", exist_ok=True)

    if blank_code is None:
        blank_code = getattr(vqvae.vq, "blank_code", -1)

    token_grid = (
        activity_prior.Ttok,
        activity_prior.Htok,
        activity_prior.Wtok,
    )

    def _run_epoch(loader, train: bool):
        activity_prior.train(train)

        total = {}
        total_samples = 0.0

        for it, batch in enumerate(loader, start=1):
            x, gct, lct, task_id, mask_spec = _batch_to_device(batch, device)

            with torch.no_grad():
                codes, pmask, _ = _vq_codes_and_pmask(vqvae, x, gct, lct, mask_spec, device)
                targets = build_activity_targets_from_codes(
                    codes=codes,
                    token_grid=token_grid,
                    Kmax=activity_prior.Kmax,
                    blank_code=blank_code,
                    predict_mask=pmask,
                )
                
                a_in = _make_activity_in_from_codes(
                    codes,
                    pmask,
                    blank_code=blank_code,
                    a_mask_id=activity_prior.a_mask_id,
                )

            if train:
                if (it - 1) % grad_accum_steps == 0:
                    opt.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    out = activity_prior(
                        global_ctx=gct,
                        local_ctx=lct,
                        task_id=task_id,
                        a_in=a_in,
                        roi_mask=pmask,
                    )

                    loss, aux = detr_activity_loss(
                        out,
                        targets,
                        activity_prior,
                        lambda_count=lambda_count,
                        lambda_obj=lambda_obj,
                        lambda_coord=lambda_coord,
                        lambda_soft_count=lambda_soft_count,
                        lambda_soft_grid=lambda_soft_grid,
                        lambda_dup=lambda_dup,
                        no_object_weight=no_object_weight,
                    )

                scaler.scale(loss / float(grad_accum_steps)).backward()

                do_step = (it % grad_accum_steps == 0) or (it == len(loader))
                if do_step:
                    if grad_clip is not None and grad_clip > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(
                            activity_prior.parameters(),
                            float(grad_clip),
                        )
                    scaler.step(opt)
                    scaler.update()

            else:
                with torch.no_grad():
                    out = activity_prior(
                        global_ctx=gct,
                        local_ctx=lct,
                        task_id=task_id,
                        a_in=a_in,
                        roi_mask=pmask,
                    )
                    loss, aux = detr_activity_loss(
                        out,
                        targets,
                        activity_prior,
                        lambda_count=lambda_count,
                        lambda_obj=lambda_obj,
                        lambda_coord=lambda_coord,
                        lambda_soft_count=lambda_soft_count,
                        lambda_soft_grid=lambda_soft_grid,
                        lambda_dup=lambda_dup,
                        no_object_weight=no_object_weight,
                    )

            B = x.size(0)
            total_samples += float(B)

            for k, v in aux.items():
                if torch.is_tensor(v):
                    v = float(v.detach().item())
                total[k] = total.get(k, 0.0) + float(v) * float(B)

            if train and log_every and (it % log_every == 0):
                den = max(total_samples, 1.0)
                print(
                    f"  [3B Activity] it {it:05d}: "
                    f"loss={total['loss'] / den:.4f} "
                    f"count={total['loss_count'] / den:.4f} "
                    f"obj={total['loss_obj'] / den:.4f} "
                    f"coord={total['loss_coord'] / den:.4f} "
                    f"soft_count={total['loss_soft_count'] / den:.4f} "
                    f"dup={total['loss_dup'] / den:.4f} "
                    f"predK={total['pred_count_mean'] / den:.2f} "
                    f"tgtK={total['target_count_mean'] / den:.2f} "
                    f"count_acc={total['count_acc'] / den:.3f}"
                )

        den = max(total_samples, 1.0)
        return {k: v / den for k, v in total.items()}

    best_val = float("inf")
    patience = 0
    history = {"train": [], "val": []}

    for ep in range(1, epochs + 1):
        train_m = _run_epoch(train_loader, train=True)
        val_m = _run_epoch(val_loader, train=False) if val_loader is not None else train_m

        history["train"].append(train_m)
        history["val"].append(val_m)

        print(
            f"[activity epoch {ep:03d}] "
            f"train loss={train_m['loss']:.4f} "
            f"val loss={val_m['loss']:.4f} "
            f"count={val_m['loss_count']:.4f} "
            f"obj={val_m['loss_obj']:.4f} "
            f"coord={val_m['loss_coord']:.4f} "
            f"soft_count={val_m['loss_soft_count']:.4f} "
            f"dup={val_m['loss_dup']:.4f} "
            f"predK={val_m['pred_count_mean']:.2f} "
            f"hardK={val_m['hard_count_mean']:.2f} "
            f"tgtK={val_m['target_count_mean']:.2f} "
            f"rawK={val_m['target_raw_count_mean']:.2f} "
            f"count_acc={val_m['count_acc']:.3f}"
        )

        if val_m["loss"] < best_val - float(min_delta):
            best_val = val_m["loss"]
            patience = 0
            torch.save(
                {
                    "model": activity_prior.state_dict(),
                    "epoch": ep,
                    "best_val_loss": best_val,
                    "token_grid": token_grid,
                    "Kmax": activity_prior.Kmax,
                },
                ckpt_out,
            )
            print(f"  saved {ckpt_out}  best={best_val:.4f}")
        else:
            patience += 1
            if patience >= int(early_stop_patience):
                print(f"Early stopping at epoch {ep}; best={best_val:.4f}")
                break

    return history




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
    blank_code: int = None,
    freeze_motif: bool = True,
    lambda_detr: float = 1.0,
    lambda_count: float = 1.0,
    lambda_obj: float = 1.0,
    lambda_coord: float = 1.0,
    lambda_soft_count: float = 0.1,
    lambda_soft_grid: float = 1.0,
    lambda_dup: float = 0.01,
    no_object_weight: float = 0.1,
    
    lambda_ctx: float = 1.0,
    lambda_ctx_field: float = 0.05,
    lambda_adj: float = 1.0,
    lambda_spatial: float = 1.0,
    
    ctx_tau: float = 0.25,
    ctx_field_tau: float = 0.25,
    
    motif_tau_z: float = 0.25,
    isi_tau: float = 0.25,
    isi_margin: float = 0.25,
    isi_max_gap: int = 3,
    isi_gap_bins=None,
    memory_adj=None,
    memory_tok=None,
    memory_adj_conf_den_scale: float = 100.0,
):
    """
    Stage 3C.

    Refines DETRActivityPrior through frozen/low-LR motif prior and frozen VQVAE.

    Path:
        activity_prior(ctx) -> soft activity_prob
        motif_prior.forward_with_activity_prob(...)
        decode_motif_logits_soft_given_activity(...)
        voxel biological losses
        gradients update activity_prior
    """

    device = next(activity_prior.parameters()).device
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    _freeze_module(vqvae)

    if freeze_motif:
        _freeze_module(motif_prior)
    else:
        motif_prior.train()

    os.makedirs(os.path.dirname(ckpt_out) or ".", exist_ok=True)

    if blank_code is None:
        blank_code = getattr(vqvae.vq, "blank_code", -1)

    token_grid = (
        activity_prior.Ttok,
        activity_prior.Htok,
        activity_prior.Wtok,
    )

    def _make_roi_mask_inputs(motif_targets):
        z1_in = motif_targets["z1"].clone()
        z2_in = motif_targets["z2"].clone()
    
        active = motif_targets["active"].bool()
        pmask = motif_targets["predict_mask"].bool()
    
        z1_in[~active] = motif_prior.z1_null_id
        z2_in[~active] = motif_prior.z2_null_id
    
        motif_mask = pmask & active
        z1_in[motif_mask] = motif_prior.z1_mask_id
        z2_in[motif_mask] = motif_prior.z2_mask_id
    
        return z1_in, z2_in

    def _run_epoch(loader, train: bool):
        activity_prior.train(train)

        if freeze_motif:
            motif_prior.eval()
        else:
            motif_prior.train(train)

        total = {}
        total_samples = 0.0

        for it, batch in enumerate(loader, start=1):
            x, gct, lct, task_id, mask_spec = _batch_to_device(batch, device)

            with torch.no_grad():
                codes, pmask, grid = _vq_codes_and_pmask(
                    vqvae, x, gct, lct, mask_spec, device
                )

                activity_targets = build_activity_targets_from_codes(
                    codes=codes,
                    token_grid=token_grid,
                    Kmax=activity_prior.Kmax,
                    blank_code=blank_code,
                    predict_mask=pmask,
                )

                motif_targets = motif_prior.make_targets_from_codes(
                    codes=codes,
                    predict_mask=pmask,
                    blank_code=blank_code,
                )

                # Predict only ROI tokens; outside ROI remains teacher-forced.
                
                active = motif_targets["a"].bool()
                roi_mask = motif_targets["predict_mask"].bool()

                motif_targets["z1_loss_mask"] = roi_mask & active
                motif_targets["z2_loss_mask"] = roi_mask & active
                motif_targets["z_loss_mask"] = roi_mask & active
                
                z1_in, z2_in = _make_roi_mask_inputs(motif_targets)
                
                a_in = _make_activity_in_from_codes(
                    codes,
                    pmask,
                    blank_code=blank_code,
                    a_mask_id=activity_prior.a_mask_id,
                )

            if train:
                if (it - 1) % grad_accum_steps == 0:
                    opt.zero_grad(set_to_none=True)

                context = torch.enable_grad()
            else:
                context = torch.no_grad()

            with context:
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    activity_out = activity_prior(
                        global_ctx=gct,
                        local_ctx=lct,
                        task_id=task_id,
                        a_in=a_in,
                        roi_mask=pmask,
                    )

                    loss_detr, aux_detr = detr_activity_loss(
                        activity_out,
                        activity_targets,
                        activity_prior,
                        lambda_count=lambda_count,
                        lambda_obj=lambda_obj,
                        lambda_coord=lambda_coord,
                        lambda_soft_count=lambda_soft_count,
                        lambda_soft_grid=lambda_soft_grid,
                        lambda_dup=lambda_dup,
                        no_object_weight=no_object_weight,
                    )

                    activity_prob_roi = activity_prior.soft_activity_flat(activity_out, roi_mask=pmask)

                    pmask_bool = motif_targets["predict_mask"].bool()
                    gt_activity = motif_targets["a"].to(
                        device=activity_prob_roi.device,
                        dtype=activity_prob_roi.dtype,
                    )
                    
                    activity_prob = torch.where(
                        pmask_bool,
                        activity_prob_roi,
                        gt_activity,
                    )

                    motif_logits = motif_prior.forward_with_activity_prob(
                        z1_in=z1_in,
                        z2_in=z2_in,
                        activity_prob=activity_prob,
                        global_ctx=gct,
                        local_ctx=lct,
                        task_id=task_id,
                        roi_mask=pmask,
                    )

                    dec = decode_motif_logits_soft_given_activity(
                        model=vqvae,
                        logits=motif_logits,
                        targets=motif_targets,
                        activity_prob=activity_prob,
                        grid=grid,
                        global_ctx=gct,
                        local_ctx=lct,
                        tau_z=motif_tau_z,
                        roi_hw=batch.get("roi_hw", None),
                        pad_hw=batch.get("pad_hw", None),
                    )

                    logits_vol = dec["logits_vol"]

                    loss_ctx = loss_detr.new_zeros(())
                    loss_ctx_field = loss_detr.new_zeros(())
                    loss_adj = loss_detr.new_zeros(())
                    loss_spatial = loss_detr.new_zeros(())

                    if lambda_ctx > 0.0 and lct is not None:
                        loss_ctx = ctx_loss_soft(
                            logits_b1thw=logits_vol,
                            ctx_tgt_b9=lct,
                            dims=tuple(range(9)),
                            tau=ctx_tau,
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
                                "Stage 3C local field target/decoder shape mismatch: "
                                f"target={tuple(target_vol.shape)}, "
                                f"logits={tuple(logits_vol.shape)}"
                            )

                        loss_ctx_field = local_moment_field_loss(
                            logits_b1thw=logits_vol,
                            target_b1thw=target_vol,
                            patch_size=vqvae.patch_size,
                            tau=ctx_field_tau,
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
                            margin=isi_margin,
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
                        float(lambda_detr) * loss_detr
                        + float(lambda_ctx) * loss_ctx
                        + float(lambda_ctx_field) * loss_ctx_field
                        + float(lambda_adj) * loss_adj
                        + float(lambda_spatial) * loss_spatial
                    )

            if train:
                scaler.scale(loss / float(grad_accum_steps)).backward()

                do_step = (it % grad_accum_steps == 0) or (it == len(loader))
                if do_step:
                    if grad_clip is not None and grad_clip > 0:
                        scaler.unscale_(opt)
                        params = list(activity_prior.parameters())
                        if not freeze_motif:
                            params += list(motif_prior.parameters())
                        torch.nn.utils.clip_grad_norm_(params, float(grad_clip))

                    scaler.step(opt)
                    scaler.update()

            B = x.size(0)
            total_samples += float(B)

            metrics = {
                "loss": loss,
                "loss_detr": loss_detr,
                "loss_ctx": loss_ctx,
                "loss_ctx_field": loss_ctx_field,
                "loss_adj": loss_adj,
                "loss_spatial": loss_spatial,
                "activity_prob_mean": activity_prob.mean(),
                "activity_prob_sum": activity_prob.sum(dim=1).mean(),
            }
            metrics.update({f"detr_{k}": v for k, v in aux_detr.items()})

            for k, v in metrics.items():
                if torch.is_tensor(v):
                    v = float(v.detach().item())
                total[k] = total.get(k, 0.0) + float(v) * float(B)

            if train and log_every and (it % log_every == 0):
                den = max(total_samples, 1.0)
                print(
                    f"  [3C Joint] it {it:05d}: "
                    f"loss={total['loss'] / den:.4f} "
                    f"detr={total['loss_detr'] / den:.4f} "
                    f"ctx={total['loss_ctx'] / den:.4f} "
                    f"field={total['loss_ctx_field'] / den:.4f} "
                    f"adj={total['loss_adj'] / den:.4f} "
                    f"sp={total['loss_spatial'] / den:.4f} "
                    f"softK={total['activity_prob_sum'] / den:.2f} "
                    f"tgtK={total['detr_target_count_mean'] / den:.2f}"
                )

        den = max(total_samples, 1.0)
        return {k: v / den for k, v in total.items()}

    best_val = float("inf")
    patience = 0
    history = {"train": [], "val": []}

    for ep in range(1, epochs + 1):
        train_m = _run_epoch(train_loader, train=True)
        val_m = _run_epoch(val_loader, train=False) if val_loader is not None else train_m

        history["train"].append(train_m)
        history["val"].append(val_m)

        print(
            f"[activity-refine epoch {ep:03d}] "
            f"train loss={train_m['loss']:.4f} "
            f"val loss={val_m['loss']:.4f} "
            f"detr={val_m['loss_detr']:.4f} "
            f"ctx={val_m['loss_ctx']:.4f} "
            f"ctx_field={val_m['loss_ctx_field']:.4f} "
            f"adj={val_m['loss_adj']:.4f} "
            f"sp={val_m['loss_spatial']:.4f} "
            f"softK={val_m['activity_prob_sum']:.2f} "
            f"tgtK={val_m['detr_target_count_mean']:.2f} "
            f"count_acc={val_m['detr_count_acc']:.3f}"
        )

        if val_m["loss"] < best_val - float(min_delta):
            best_val = val_m["loss"]
            patience = 0
            torch.save(
                {
                    "activity_prior": activity_prior.state_dict(),
                    "motif_prior": motif_prior.state_dict() if not freeze_motif else None,
                    "epoch": ep,
                    "best_val_loss": best_val,
                    "token_grid": token_grid,
                    "Kmax": activity_prior.Kmax,
                },
                ckpt_out,
            )
            print(f"  saved {ckpt_out}  best={best_val:.4f}")
        else:
            patience += 1
            if patience >= int(early_stop_patience):
                print(f"Early stopping at epoch {ep}; best={best_val:.4f}")
                break

    return history