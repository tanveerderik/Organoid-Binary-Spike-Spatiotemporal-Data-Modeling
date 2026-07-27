#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:10:22 2026

@author: derik
"""
import torch
from contextlib import nullcontext


from ..utils.metrics import PRCurveAccumulator
from ..utils.losses import tolerant_spike_loss


@torch.no_grad()
def evaluate_vqvae(
    model,
    loader,
    use_amp: bool = True,
    num_thr: int = 101,
    use_ROI_mask: bool = True,
    pos_weight: float = 1.0,
    recon_tolerance: tuple[int, int, int] = (2, 2, 2),
    metric_tolerance: tuple[int, int, int] = (2, 2, 2),
    # If True: compute both conditioned (FiLM on) and unconditioned (FiLM off) metrics.
    eval_cfg_modes: bool = True,
):
    """
    Deterministic evaluation.

    - "Conditional" = FiLM always ON  (cfg_ctx_force_unc=False)
    - "Unconditional" = FiLM always OFF (cfg_ctx_force_unc=True)

    PR metrics are calculated once from TP/FP/FN accumulated over the complete
    loader. Conditional keys end in ``_cond``; optional unconditional keys end
    in ``_uncond``.
    """
    model.eval()

    device = next(model.parameters()).device
    
    if use_amp and device.type == "cuda":
        autocast_ctx = torch.cuda.amp.autocast
    else:
        autocast_ctx = nullcontext
    # autocast_ctx = torch.cuda.amp.autocast if (use_amp and device.type == "cuda") else torch.cpu.amp.autocast
    
    # ---- accumulators (conditional) ----
    total_bce_c, total_bce_full_c, sum_vq_c = 0.0, 0.0, 0.0
    
    # ---- accumulators (unconditional) ----
    total_bce_u, total_bce_full_u, sum_vq_u = 0.0, 0.0, 0.0
    
    mt, mh, mw = metric_tolerance

    exact_c = PRCurveAccumulator(
        num_thr=num_thr,
        device=device,
    )
    tolerant_c = PRCurveAccumulator(
        num_thr=num_thr,
        device=device,
        radius_t=mt,
        radius_h=mh,
        radius_w=mw,
    )
    
    exact_u = None
    tolerant_u = None
    if eval_cfg_modes:
        exact_u = PRCurveAccumulator(
            num_thr=num_thr,
            device=device,
        )
        tolerant_u = PRCurveAccumulator(
            num_thr=num_thr,
            device=device,
            radius_t=mt,
            radius_h=mh,
            radius_w=mw,
        )
    
    total_frames = 0
    total_eval_voxels = 0
    total_full_voxels = 0
    eval_passes = 0

    sb_sum = 0.0
    sb_abs_sum = 0.0
    sb_sum_sq = 0.0
    sb_count = 0

    
    max_ref_levels = int(getattr(model.vq, "num_quantizers", 1))
    refinement_sums_c = {}
    refinement_sums_u = {}
    
    for ridx in range(1, max_ref_levels + 1):
        refinement_sums_c[f"ref{ridx}"] = 0.0
        refinement_sums_c[f"ref{ridx}_exact"] = 0.0
        refinement_sums_c[f"ref{ridx}_tol"] = 0.0
    
        refinement_sums_u[f"ref{ridx}"] = 0.0
        refinement_sums_u[f"ref{ridx}_exact"] = 0.0
        refinement_sums_u[f"ref{ridx}_tol"] = 0.0

    def _metrics_from_out(
        out,
        x,
        exact_accumulator,
        tolerant_accumulator,
    ):
        """
        Compute eval metrics for one output dict.
    
        Always computes:
          - full-volume BCE
    
        If use_ROI_mask=True, also computes:
          - ROI-masked BCE
          - PR metrics on ROI only
    
        If use_ROI_mask=False:
          - val_loss_BCE == val_loss_BCE_full
          - PR metrics use full volume
        """        
        grid = out["grid"]
        logits_vol = out["logits_vol"]          # final refinement, already decoded
        logits_vol_raw = out.get("logits_vol_raw", logits_vol)
        prob_vol = torch.sigmoid(logits_vol)
        
        rt, rh, rw = recon_tolerance
        mt, mh, mw = metric_tolerance
    
        _, _, Tp, Hp, Wp = logits_vol.shape
        tgt_vol = x[:, :1, :Tp, :Hp, :Wp]
        
        # ---- full mask always ----
        full_mask_vol = torch.ones_like(tgt_vol, dtype=torch.float32, device=logits_vol.device)

        refinements = out.get("refinements", None)
        refinement_losses = []
        refinement_exact = []
        refinement_tol = []
        
        if refinements is not None:
            for ref in refinements:
                ref_logits = ref.get("logits_vol_raw", ref["logits_vol"])[..., :Tp, :Hp, :Wp]
                  
                # use train-matched settings by level
                if ref["level"] < len(refinements):
                    ref_parts = tolerant_spike_loss(
                        logits=ref_logits.float(),
                        target=tgt_vol.float(),
                        mask_vol=full_mask_vol,
                        pos_weight=pos_weight,
                        radius_t=rt,
                        radius_h=rh,
                        radius_w=rw,
                        alpha_exact=0.20,
                        beta_hit=0.70,
                        gamma_peak=0.05,
                        delta_multi=0.05,
                        return_parts=True,
                    )
                else:
                    ref_parts = tolerant_spike_loss(
                        logits=ref_logits.float(),
                        target=tgt_vol.float(),
                        mask_vol=full_mask_vol,
                        pos_weight=pos_weight,
                        radius_t=rt,
                        radius_h=rh,
                        radius_w=rw,
                        alpha_exact=0.90,
                        beta_hit=0.08,
                        gamma_peak=0.02,
                        delta_multi=0.00,
                        return_parts=True,
                    )
        
                refinement_losses.append(float(ref_parts["total"].item()))
                refinement_exact.append(float(ref_parts["weighted_exact"].item()))
                refinement_tol.append(float(ref_parts["weighted_tol"].item()))
    


        bce_full = tolerant_spike_loss(
            logits=logits_vol.float(),
            target=tgt_vol.float(),
            mask_vol=full_mask_vol,
            pos_weight=pos_weight,
            radius_t=rt,
            radius_h=rh,
            radius_w=rw,
            alpha_exact=0.90,
            beta_hit=0.08,
            gamma_peak=0.02,
            delta_multi=0.00,
        )
    
        # ---- active mask depends on use_ROI_mask ----
        if use_ROI_mask:
            pmask_tok = out.get("predict_mask", None)
            
            if pmask_tok is None:
                pmask_vol = full_mask_vol
            else:
                if pmask_tok.dim() == 2:
                    pmask_tok = pmask_tok.unsqueeze(-1)   # (B,N,1)
            
                patch_dim = model.patch_size[0] * model.patch_size[1] * model.patch_size[2] * model.out_chans
                pmask_patch = pmask_tok.float().expand(-1, -1, patch_dim)   # (B,N,PD)
            
                pmask_vol = model.unpatchify(pmask_patch, grid)
                pmask_vol = pmask_vol[:, :1, :Tp, :Hp, :Wp]
                pmask_vol = (pmask_vol > 0.5).float()
    
            # bce_active = masked_bce_with_logits_weighted(
            #     logits_vol.float(),
            #     tgt_vol.float(),
            #     mask_vol=pmask_vol,
            #     pos_weight=pos_weight,
            # )
            bce_active = tolerant_spike_loss(
                logits=logits_vol.float(),
                target=tgt_vol.float(),
                mask_vol=pmask_vol,
                pos_weight=pos_weight,
                radius_t=rt,
                radius_h=rh,
                radius_w=rw,
                alpha_exact=0.90,
                beta_hit=0.08,
                gamma_peak=0.02,
                delta_multi=0.00,
            )
        else:
            pmask_vol = full_mask_vol
            bce_active = bce_full
    
        # Update the one validation-population PR curve once per batch.
        exact_accumulator.update(prob_vol, tgt_vol, pmask_vol)
        tolerant_accumulator.update(prob_vol, tgt_vol, pmask_vol)

        B = x.shape[0]
        eval_voxels = int((pmask_vol > 0).sum().item())
        full_voxels = int(full_mask_vol.sum().item())
        eval_frames = int(Tp * B)
    
        return (
            float(bce_active.item()),
            float(bce_full.item()),
            float(out["vq_loss"].detach().cpu()),
            eval_frames,
            eval_voxels,
            full_voxels,
            B,
            refinement_losses,
            refinement_exact,
            refinement_tol,
        )

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)

        gct = batch.get("global_ctx", None)
        lct = batch.get("local_ctx", None)
        mask_spec = batch.get("mask_spec", None)
        roi_hw = batch.get("roi_hw", None)
        pad_hw = batch.get("pad_hw", None)

        task_id = batch.get("task_id", None)
        if task_id is None:
            raise KeyError('Batch missing "task_id" (required for MGIT conditioning).')
        task_id = task_id.to(device, non_blocking=True).long()

        if isinstance(gct, torch.Tensor):
            gct = gct.to(device, non_blocking=True)
        if isinstance(lct, torch.Tensor):
            lct = lct.to(device, non_blocking=True)

        # ---- forward passes (deterministic) ----
        with autocast_ctx():
            out_c = model(
                x,
                global_ctx=gct,
                local_ctx=lct,
                predict_mask_spec=mask_spec,
                cfg_ctx_force_unc=False,   # conditioned
                roi_hw=roi_hw,
                pad_hw=pad_hw,
            )
            
            sp_bias = out_c.get("assay_spatial_full_pix2d_support", None)
            
            
            if sp_bias is not None:
                sb = sp_bias.detach().float()
                sb_sum += float(sb.sum().item())
                sb_abs_sum += float(sb.abs().sum().item())
                sb_sum_sq += float((sb * sb).sum().item())
                sb_count += int(sb.numel())
            
            out_u = None
            if eval_cfg_modes:
                out_u = model(
                    x,
                    global_ctx=gct,
                    local_ctx=lct,
                    predict_mask_spec=mask_spec,
                    cfg_ctx_force_unc=True,  # unconditional
                    roi_hw=roi_hw,
                    pad_hw=pad_hw,
                )

        # ---- metrics: conditional ----
        bce_c, bce_full_c, vq_c, frames_c, vox_c, full_vox_c, samples_c, \
            ref_losses_c, ref_exact_c, ref_tol_c = _metrics_from_out(
                out_c,
                x,
                exact_c,
                tolerant_c,
            )
            
        for ridx, val in enumerate(ref_losses_c, start=1):
            refinement_sums_c[f"ref{ridx}"] += val
        for ridx, val in enumerate(ref_exact_c, start=1):
            refinement_sums_c[f"ref{ridx}_exact"] += val
        for ridx, val in enumerate(ref_tol_c, start=1):
            refinement_sums_c[f"ref{ridx}_tol"] += val
            
        total_bce_c += bce_c
        total_bce_full_c += bce_full_c
        sum_vq_c += vq_c
        total_frames += frames_c
        total_eval_voxels += vox_c
        total_full_voxels += full_vox_c
        eval_passes += samples_c

        # ---- metrics: unconditional ----
        if eval_cfg_modes and out_u is not None:
            bce_u, bce_full_u, vq_u, _, _, _, _, \
                ref_losses_u, ref_exact_u, ref_tol_u = _metrics_from_out(
                    out_u,
                    x,
                    exact_u,
                    tolerant_u,
                )
            total_bce_u += bce_u
            total_bce_full_u += bce_full_u
            sum_vq_u += vq_u
            
            for ridx, val in enumerate(ref_losses_u, start=1):
                refinement_sums_u[f"ref{ridx}"] += val
            for ridx, val in enumerate(ref_exact_u, start=1):
                refinement_sums_u[f"ref{ridx}_exact"] += val
            for ridx, val in enumerate(ref_tol_u, start=1):
                refinement_sums_u[f"ref{ridx}_tol"] += val


    (
        auprc_c,
        bestf1_c,
        bestthr_c,
    ) = exact_c.compute()
    
    (
        auprc_tol_c,
        bestf1_tol_c,
        bestthr_tol_c,
    ) = tolerant_c.compute()
    
    if eval_cfg_modes:
        (
            auprc_u,
            bestf1_u,
            bestthr_u,
        ) = exact_u.compute()
    
        (
            auprc_tol_u,
            bestf1_tol_u,
            bestthr_tol_u,
        ) = tolerant_u.compute()

    report = {
        "val_loss_BCE": total_bce_c / max(1, len(loader)),
        "val_loss_BCE_full": total_bce_full_c / max(1, len(loader)),
        "val_loss_vq": sum_vq_c / max(1, len(loader)),
        "val_loss_BCE_cond": total_bce_c / max(1, len(loader)),
        "val_loss_BCE_full_cond": total_bce_full_c / max(1, len(loader)),
        "val_loss_vq_cond": sum_vq_c / max(1, len(loader)),
        "AUPRC_cond": auprc_c,
        "BestF1_cond": bestf1_c,
        "BestF1_threshold_cond": bestthr_c,
        "AUPRC_tol_cond": auprc_tol_c,
        "BestF1_tol_cond": bestf1_tol_c,
        "BestF1_threshold_tol_cond": bestthr_tol_c,
        "eval_count": eval_passes,
        "eval_density": float(total_eval_voxels) / max(1.0, float(total_frames)),
        "eval_mask_frac": float(total_eval_voxels) / max(1.0, float(total_full_voxels)),
    }
    
    for ridx in range(1, max_ref_levels + 1):
        report[f"val_loss_recon_ref{ridx}_cond"] = refinement_sums_c[f"ref{ridx}"] / max(1, len(loader))
        report[f"val_loss_recon_ref{ridx}_exact_cond"] = refinement_sums_c[f"ref{ridx}_exact"] / max(1, len(loader))
        report[f"val_loss_recon_ref{ridx}_tol_cond"] = refinement_sums_c[f"ref{ridx}_tol"] / max(1, len(loader))

    if sb_count > 0:
        sb_mean = sb_sum / sb_count
        sb_abs_mean = sb_abs_sum / sb_count
        sb_var = sb_sum_sq / sb_count - sb_mean ** 2
        sb_var = max(sb_var, 0.0)
        sb_std = sb_var ** 0.5
    
        report.update({
            "val_spatial_bias_mean": sb_mean,
            "val_spatial_bias_abs_mean": sb_abs_mean,
            "val_spatial_bias_std": sb_std,
            "val_num_bias_values": sb_count,
        })
    
    if eval_cfg_modes:
        report.update({
            "val_loss_BCE_uncond": total_bce_u / max(1, len(loader)),
            "val_loss_BCE_full_uncond": total_bce_full_u / max(1, len(loader)),
            "val_loss_vq_uncond": sum_vq_u / max(1, len(loader)),
            "AUPRC_uncond": auprc_u,
            "BestF1_uncond": bestf1_u,
            "BestF1_threshold_uncond": bestthr_u,
            "AUPRC_tol_uncond": auprc_tol_u,
            "BestF1_tol_uncond": bestf1_tol_u,
            "BestF1_threshold_tol_uncond": bestthr_tol_u,
        })
        
        for ridx in range(1, max_ref_levels + 1):
            report[f"val_loss_recon_ref{ridx}_uncond"] = refinement_sums_u[f"ref{ridx}"] / max(1, len(loader))
            report[f"val_loss_recon_ref{ridx}_exact_uncond"] = refinement_sums_u[f"ref{ridx}_exact"] / max(1, len(loader))
            report[f"val_loss_recon_ref{ridx}_tol_uncond"] = refinement_sums_u[f"ref{ridx}_tol"] / max(1, len(loader))


    return report