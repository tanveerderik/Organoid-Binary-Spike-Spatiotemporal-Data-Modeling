#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:08:36 2026

@author: derik
"""
import math
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable
import time

from ..utils.recon import (
    soft_spatial_pixel_map_from_logits,
    soft_spatial_token_map_from_logits,
    dilate_spatial_support_hw,
    sample_full_pixel_map_to_crop,
    sample_full_token_map_to_crop,
)

from ..utils.losses import (
    tolerant_spike_loss,
    short_gap_excess_loss_from_logits_batch_targets,
    ctx_loss_soft,
    ctx_features_soft_from_logits,
    local_moment_field_loss,
    encoder_isotropy_loss,
    variance_floor_loss,
    soft_code_norm_ceiling_loss,
    spatial_support_violation_loss,
    blank_patch_logit_hinge_loss,
    blank_active_decoder_separation_loss,
)

from ..utils.metrics import get_vq_codebook_stats

from .eval_vqvae import evaluate_vqvae




def fit_vqvae(
    model: nn.Module,
    train_loader,
    val_loader=None,
    optimizer: torch.optim.Optimizer = None,
    scheduler: Optional[Any] = None,
    epochs: int = 50,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
    grad_accum_steps: int = 1,
    
    recon_tolerance: tuple[int, int, int] = (2, 2, 2),
    metric_tolerance: tuple[int, int, int] = (2, 2, 2),
    
    # ---- training parameters ----
    pos_weight_start: float = 50.0,
    pos_weight_end: float = 20.0,
    pos_decay_epochs: int = 50,        # reach pos_weight by epoch 50

    # Peak/multi neighbourhood for tolerant_spike_loss, decoupled from the hit
    # tolerance. The hit term's max_pool REWARDS a spike anywhere within
    # radius_*, so widening that permits smear; peak/multi PUNISH mass away from
    # the centre, so widening these suppresses it. They were sharing one radius,
    # and with radius_t=0 at the final refinement nothing ever compared across
    # frames -- which is why the within-token temporal profile sits at entropy
    # 1.699 against a 1.792 flat ceiling while the spatial profile, where
    # radius_h/w=1 gave the peak term reach, reaches 3.19 against 5.35.
    peak_radius_t: int = 3,          # spans the 6-frame token extent
    peak_radius_h: int = 1,
    peak_radius_w: int = 1,
    gamma_peak_mid: float = 0.05,
    gamma_peak_final: float = 0.15,
    delta_multi_mid: float = 0.05,
    delta_multi_final: float = 0.05,
    lambda_isi: float = 1e-2,
    lambda_vq: float = 1.0,
    lambda_ctx: float = 1e-3,
    lambda_ctx_field: float = 1e-4,
    
    ctx_start_epoch: int = 30,         # start ctx after 20
    ctx_warmup_epochs: int = 50,       # ramp duration (or shorter, like 10–15)

    # ---- latent masking (Stage 2C context conditioning) ----
    # Hide a subset of latent tokens from the decoder so the clip-level
    # context has something to contribute.  Under full autoencoding the
    # context is redundant with the codes and the branch cannot train.
    mask_latents: bool = False,
    latent_recon_drop_p: float = 0.5,
    log_ctx_diagnostics: bool = False,

    # Per-dimension weights for ctx_loss_soft, w_d = 1 / var_d.  Without these
    # the nine activity features enter the loss as raw squared error, and
    # log_mean_firing_density (std 0.85) contributes ~1500x more than cov_xt
    # (std 0.02) -- so only density and active_site_ratio ever get optimized.
    # w_d = 1/var_d makes the objective a plain mean of squared errors
    # measured in each dimension's own std units.
    ctx_dim_weights: Optional[Any] = None,
    
    # ---- hierarchical refinement supervision ----
    refinement_loss_weights: Optional[list[float]] = None,
    refinement_use_raw_logits: bool = True,
    
    # ---- hierarchical VQ schedule ----
    level2_start_epoch: int = 1,
    level3_start_epoch: int = 1,
    level2_full_loss_epoch: int = 20,
    level3_full_loss_epoch: int = 50,
    
    # ---- CFG FiLM conditioning dropout schedule (per-sample) ----
    cfg_ctx_drop_start: float = 0.0,     # start p(drop conditioning)
    cfg_ctx_drop_end: float = 0.0,       # end p(drop conditioning)
    cfg_ctx_start_epoch: int = 1,        # when to start ramp
    cfg_ctx_warmup_epochs: int = 0,      # ramp length (0 = jump to end at start_epoch)    
    ctx_epoch_schedule: Optional[dict] = None,
    
    lambda_sp_token: float = 1e-3,
    lambda_sp_pixel: float = 1e-4,
    sp_pixel_start_epoch: int = 40,
    sp_pixel_warmup_epochs: int = 60,
    lambda_enc_var: float = 1.0,
    
    lambda_code_norm: float = 0.0,
    code_norm_rms_ceiling: float = 10.0,
    code_norm_token_ceiling: float = 14.0,

    # ---- Stage-2 continuous residual relaxation ----
    
    # --- Blank embedding and patch enforcement ---
    lambda_blank: float = 0.05,
    blank_logit_margin: float = -9.0,
    blank_start_epoch: int = 1,
    blank_warmup_epochs: int = 20,
    lambda_blank_sep: float = 0.01,
    blank_sep_margin: float = 1.0,
    blank_sep_start_epoch: int = 1,
    blank_sep_warmup_epochs: int = 20,
    
    # ---- logging / callbacks ----
    on_step: Optional[Callable[[Dict[str, float]], None]] = None,
    on_epoch: Optional[Callable[[Dict[str, float]], None]] = None,
    # ---- checkpoints (optional) ----
    ckpt_best_path: Optional[str] = None,
    ckpt_last_path: Optional[str] = None,
    # ---- early stop ----
    save_start_epoch: int = 0,
    early_stop_patience: Optional[int] = None,
    val_metric_name: str = "AUPRC_tol",
    val_metric_goal: str = "max",  # or "min"
    use_ROI_mask: bool = False,
    
    # ---- ISI maintenance loss terms ----
    isi_max_gap: int = 3,
    isi_gap_bins=None,
    isi_tau: float = 0.25,
    isi_margin: float = 0.25,
    isi_lower_margin: float = 0.20,
    isi_lower_weight: float = 0.50,
    
    memory_tok = None,
    memory_pix = None,
    memory_adj = None,
    memory_adj_conf_den_scale: float = 100.0,
) -> Dict[str, Any]:

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_val = -math.inf if val_metric_goal == "max" else math.inf
    best_epoch = -1
    best_thr_exact_for_best_model = 0.5
    best_thr_tol_for_best_model = 0.5
    no_improve = 0
    eval_report = None
    history = {"train_log": [], "val_metrics": []}
    
    target_std_start = 1.0
    target_std_end = 0.25
    tau_logit = 0.25
    
    if ctx_epoch_schedule is None:
        ctx_epoch_schedule = {0: ctx_start_epoch}
    
    if refinement_loss_weights is None:
        if getattr(model.vq, "num_quantizers", 1) == 1:
            refinement_loss_weights = [1.0]
        else:
            L = int(model.vq.num_quantizers)
            refinement_loss_weights = [1.0] * L
    
    def _linear_ramp(epoch_idx: int, start_epoch: int, warmup_epochs: int, v0: float, v1: float) -> float:
        """
        Returns value ramped linearly from v0 to v1 starting at start_epoch over warmup_epochs.
        If warmup_epochs <= 0: jumps to v1 at start_epoch.
        """
        if epoch_idx < start_epoch:
            return float(v0)
        if warmup_epochs <= 0:
            return float(v1)
        t = (epoch_idx - start_epoch) / float(warmup_epochs)
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        return float(v0 + (v1 - v0) * t)
    

    def _cosine_ramp(epoch_idx, start_epoch, warmup_epochs, v0, v1):
        if epoch_idx < start_epoch:
            return v0
        if warmup_epochs <= 0:
            return v1
    
        t = min(1.0, (epoch_idx - start_epoch) / warmup_epochs)
        cos_t = 0.5 * (1 + math.cos(math.pi * t))
        return v1 + (v0 - v1) * cos_t

    for epoch in range(1, epochs + 1):
        active_prob_threshold = float(
            model.training_prob_threshold.item()
        )
        model.train()
        t0 = time.time()
        accum = 0
        total_loss = 0.0
        
        sums = {
            "loss_recon_total": 0.0,
            "loss_recon_exact_total": 0.0,
            "loss_recon_tol_total": 0.0,
            
            "loss_vq": 0.0,
            "loss_isi": 0.0,
            
            "loss_ctx": 0.0,
            "loss_ctx_raw": 0.0,
            "loss_ctx_field": 0.0,
            "latent_hidden_frac": 0.0,
            "latent_hidden_active_frac": 0.0,
            "loss_sp_cons": 0.0,
            "loss_sp_token": 0.0,
            "loss_sp_pixel": 0.0,
            
            "sp_memory_used": 0.0,
            "adj_memory_used": 0.0,
            "adj_pred_mean": 0.0,
            "adj_upper_allowed_mean": 0.0,
            "adj_lower_allowed_mean": 0.0,
            "adj_tgt_mean": 0.0,
            "adj_conf_mean": 0.0,
                        
            "loss_enc_var": 0.0,
            "loss_code_norm": 0.0,
            "loss_blank": 0.0,
            "loss_blank_sep": 0.0,
            
        }
        
        # ---- hierarchy schedule ----
        # Was hardcoded to 1-or-2, which silently capped a 3-level model at two
        # levels every epoch, overriding the constructor. Level 3 was allocated
        # (1024 entries) but never assigned: perplexity_nonblank_l3 = 0.
        _L_max = int(getattr(model.vq, "num_quantizers", 1))
        if epoch < level2_start_epoch:
            _n_active = 1
        elif _L_max < 3 or epoch < level3_start_epoch:
            _n_active = min(2, _L_max)
        else:
            _n_active = min(3, _L_max)
        model.vq.set_active_quantizers(_n_active)
        
        max_ref_levels = int(getattr(model.vq, "num_quantizers", 1))
        
        
        for ridx in range(1, max_ref_levels + 1):
            sums[f"loss_recon_l{ridx}"] = 0.0
            sums[f"loss_recon_l{ridx}_exact"] = 0.0
            sums[f"loss_recon_l{ridx}_tol"] = 0.0
        
        vq_metric_sums = {
            "blank_frac": 0.0,
            "num_nonblank": 0.0,
            "residual_norm": 0.0,
            "commit_loss": 0.0,
            "usage_loss": 0.0,
        }
        
        sum_pred_mean_p = 0.0
        sum_tgt_mean    = 0.0
        num_batches = 0
        ctx_diag_keys: set = set()

        cfg_p = _cosine_ramp(
            epoch_idx=epoch,
            start_epoch=cfg_ctx_start_epoch,
            warmup_epochs=cfg_ctx_warmup_epochs,
            v0=cfg_ctx_drop_start,
            v1=cfg_ctx_drop_end,
        )

        # Counterfactual conditioning: the loss weight switches on at
        # ---- dynamic refinement supervision ----
        L_active = int(getattr(model.vq, "active_quantizers", model.vq.num_quantizers))
        
        if L_active == 1:
            refinement_loss_weights_eff = [1.0]
        elif L_active == 2:
            if epoch < level2_full_loss_epoch:
                refinement_loss_weights_eff = [1.0, 0.5]
            else:
                refinement_loss_weights_eff = [0.75, 1.0]
        elif L_active == 3:
            # Two handovers instead of one, walking the tolerance ladder
            # (1,1,1) -> (0,1,1) -> (0,0,0). level3_full_loss_epoch defaults to
            # 50 rather than 40 so the second handover does not land on top of
            # the ctx dim schedule completing at 40, which would confound two
            # transitions in the same epochs.
            if epoch < level2_full_loss_epoch:
                refinement_loss_weights_eff = [1.0, 0.5, 0.25]
            elif epoch < level3_full_loss_epoch:
                refinement_loss_weights_eff = [0.75, 1.0, 0.5]
            else:
                refinement_loss_weights_eff = [0.5, 0.75, 1.0]
        else:
            raise ValueError(f"Unsupported active quantizer count: {L_active}")
        
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            gct = batch.get("global_ctx", None)
            lct  = batch.get("local_ctx", None)
            predict_mask_spec = batch.get("mask_spec", None)
            roi_hw = batch.get("roi_hw", None)
            pad_hw = batch.get("pad_hw", None)

            if isinstance(gct, torch.Tensor): gct = gct.to(device, non_blocking=True)
            if isinstance(lct, torch.Tensor): lct = lct.to(device, non_blocking=True)

            task_id = batch.get("task_id", None)
            if task_id is None:
                raise KeyError('Batch missing "task_id".')
            task_id = task_id.to(device, non_blocking=True).long()
            
            _last_handover = (
                level3_full_loss_epoch if L_active >= 3 else level2_full_loss_epoch
            )
            do_refinement_supervision = (
                (epoch < _last_handover)
                or (num_batches % 100 == 0)
            )
            num_batches += 1
            
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = model(
                    x,
                    global_ctx=gct,
                    local_ctx=lct,
                    predict_mask_spec=predict_mask_spec,
                    roi_hw=roi_hw,
                    pad_hw=pad_hw,
                    return_all_refinements=do_refinement_supervision,
                    mask_latents=mask_latents,
                    latent_recon_drop_p=latent_recon_drop_p,
                )                
                vq_loss = out["vq_loss"]
                z_e_active = out["z_e_active"]
                
                if lambda_code_norm > 0.0:
                    loss_code_norm, code_norm_parts = soft_code_norm_ceiling_loss(
                        z_e_active,
                        rms_ceiling=code_norm_rms_ceiling,
                        token_ceiling=code_norm_token_ceiling,
                        huber_beta=0.10,
                        tail_weight=0.25,
                        return_parts=True,
                    )
                else:
                    loss_code_norm = z_e_active.new_zeros(())
                
                    with torch.no_grad():
                        code_norms = z_e_active.float().norm(dim=-1)
                
                        code_norm_parts = {
                            "rms": torch.sqrt(
                                code_norms.square().mean() + 1e-8
                            ),
                            "mean": code_norms.mean(),
                            "std": code_norms.std(unbiased=False),
                            "p95": torch.quantile(code_norms, 0.95),
                            "max": code_norms.max(),
                            "tail_fraction": (
                                code_norms > float(code_norm_token_ceiling)
                            ).float().mean(),
                            "rms_loss": z_e_active.new_zeros(()),
                            "tail_loss": z_e_active.new_zeros(()),
                        }
                
                
                logits_patches = out["pred_patches"]
                grid = out["grid"]
                predict_mask = out["predict_mask"]
                logits_vol = out["logits_vol"]
                logits_vol_raw = out["logits_vol_raw"]
                prob_vol_raw = torch.sigmoid(logits_vol_raw)
                
                refinements = out.get("refinements", None)
                if refinements is None or len(refinements) == 0:
                    raise RuntimeError("Model output is missing non-empty 'refinements'.")
                
                vq_aux = out.get("vq_aux", {})
                
                if vq_aux:
                    vq_metric_sums["blank_frac"] += float(vq_aux.get("blank_frac", 0.0))
                    vq_metric_sums["num_nonblank"] += float(vq_aux.get("num_nonblank", 0.0))
                    vq_metric_sums["residual_norm"] += float(vq_aux.get("residual_norm", 0.0))
                    vq_metric_sums["commit_loss"] += float(vq_aux.get("commit_loss", 0.0))
                    vq_metric_sums["usage_loss"] += float(vq_aux.get("usage_loss", 0.0))
                
                levels_aux = vq_aux.get("levels", []) if vq_aux else []
                for lvl, lvl_aux in enumerate(levels_aux):
                    lvl_id = lvl + 1

                    metric_keys = [
                        f"perplexity_nonblank_l{lvl_id}",
                        f"active_codes_nonblank_l{lvl_id}",
                        f"child_per_active_parent_l{lvl_id}",
                        f"commit_loss_l{lvl_id}",
                        f"usage_loss_l{lvl_id}",
                        f"soft_usage_loss_l{lvl_id}",
                    ]

                    for k in metric_keys:
                        if k not in vq_metric_sums:
                            vq_metric_sums[k] = 0.0

                    vq_metric_sums[f"perplexity_nonblank_l{lvl_id}"] += float(lvl_aux.get("perplexity_nonblank", 0.0))
                    vq_metric_sums[f"active_codes_nonblank_l{lvl_id}"] += float(lvl_aux.get("active_codes_nonblank", 0.0))
                    vq_metric_sums[f"child_per_active_parent_l{lvl_id}"] += float(lvl_aux.get("child_per_active_parent", 0.0))
                    vq_metric_sums[f"commit_loss_l{lvl_id}"] += float(lvl_aux.get("commit_loss", 0.0))
                    vq_metric_sums[f"usage_loss_l{lvl_id}"] += float(lvl_aux.get("usage_loss", 0.0))
                    vq_metric_sums[f"soft_usage_loss_l{lvl_id}"] += float(lvl_aux.get("soft_usage_loss", 0.0))     
                                

                _, _, Tp, Hp, Wp = logits_vol.shape
                tgt_vol = x[:, :1, :Tp, :Hp, :Wp]
                
                if not use_ROI_mask:
                    mask_vol = torch.ones_like(tgt_vol, dtype=torch.float32, device=device)
                else:
                    predict_mask = out.get("predict_mask", None)
                    if predict_mask is None:
                        mask_vol = torch.ones_like(tgt_vol, dtype=torch.float32, device=device)
                    else:
                        if predict_mask.dim() == 2:
                            predict_mask = predict_mask.unsqueeze(-1)   # (B,N,1)
                
                        P = logits_patches.shape[-1]                    # patch volume, e.g. 4096
                        predict_mask_exp = predict_mask.float().expand(-1, -1, P)   # (B,N,P)
                
                        mask_vol = model.unpatchify(predict_mask_exp, grid)
                        mask_vol = mask_vol[:, :1, :Tp, :Hp, :Wp]
                        mask_vol = (mask_vol > 0.5).float()
                
                with torch.no_grad():
                    # prob_vol: (B,1,T,H,W) = sigmoid(logits_vol)
                    # tgt_vol : (B,1,T,H,W) binary {0,1}
                    # mask_vol: same shape, 0/1 mask (if you use one)
                
                    if mask_vol is None:
                        pred_mean_p_b = prob_vol_raw.mean()
                        tgt_mean_b    = tgt_vol.float().mean()
                    else:
                        m = mask_vol.float()
                        denom = m.sum().clamp_min(1.0)
                        pred_mean_p_b = (prob_vol_raw * m).sum() / denom
                        tgt_mean_b    = (tgt_vol.float() * m).sum() / denom
                
                    sum_pred_mean_p += float(pred_mean_p_b.item())
                    sum_tgt_mean    += float(tgt_mean_b.item())
                
                                

                # --- scheduled loss function ---
                
                # --- pos_weight schedule (cosine down from 50 -> 20 over first 20 epochs) ---
                t_pos = min(1.0, max(0.0, (epoch - 1) / max(1, pos_decay_epochs)))
                pos_weight_eff = pos_weight_end + 0.5 * (pos_weight_start - pos_weight_end) * (1 + math.cos(math.pi * t_pos))

                
                # --- losses ---
                rt, rh, rw = recon_tolerance

                # choose raw-vs-biased supervision source for refinement losses
                def _get_ref_logits(ref):
                    return ref["logits_vol_raw"] if refinement_use_raw_logits else ref["logits_vol"]
                 
                num_ref = len(refinements)
                
                if num_ref == 1:
                    # final-only fast path
                    refinement_loss_weights_batch = [1.0]
                else:
                    refinement_loss_weights_batch = refinement_loss_weights_eff[:num_ref]
                
                if len(refinement_loss_weights_batch) != num_ref:
                    raise ValueError(
                        f"refinement weights length {len(refinement_loss_weights_batch)} "
                        f"does not match num refinements {num_ref}"
                    )

                 
                # ---- early/intermediate refinements: tolerant ----
                
                recon_total = logits_vol_raw.new_zeros(())

                recon_level_vals = []
                recon_level_exact_vals = []
                recon_level_tol_vals = []
                recon_level_ids = []
                
                for ridx, (ref, w_ref) in enumerate(
                    zip(refinements, refinement_loss_weights_batch),
                    start=1,
                ):
                    logits_ref = _get_ref_logits(ref)
                    level_id = int(ref.get("level", ridx))
                    is_final_refinement = ridx == num_ref

                    # Single-quantizer models have no coarse companion decode,
                    # so their sole pass is is_final_refinement from epoch 1 and
                    # never sees the tolerant curriculum that shapes z1 over the
                    # first level2_full_loss_epoch epochs of a two-level run.
                    # Without this the z1-size sweep confounds "one pass instead
                    # of two" with "exact objective from scratch", while still
                    # being scored on the tolerant metric. Give the single pass
                    # the same curriculum: tolerant first, exact from
                    # level2_full_loss_epoch on. L_active == 2 is untouched.
                    if L_active == 1 and epoch < level2_full_loss_epoch:
                        is_final_refinement = False
                
                    # ---- tolerance ladder ----
                    # L=3 walks (1,1,1) -> (0,1,1) -> (0,0,0) keyed on level_id.
                    # L<=2 reproduces the historical two-branch behaviour exactly.
                    #
                    # beta_hit STAYS NON-ZERO at zero tolerance. At radius
                    # (0,0,0) the hit term's max_pool3d is the identity, so
                    # loss_hit = -log(p[pos]).mean() -- an UNWEIGHTED,
                    # POSITIVE-ONLY log-likelihood with no false-positive
                    # counterbalance. That is NOT a duplicate of alpha_exact's
                    # BCE, which carries pos_weight (decaying 100 -> 1 over the
                    # first 100 epochs) and the negative term. beta_hit is the
                    # only anchor holding probability up at true spikes once
                    # pos_weight has decayed.
                    # Setting it to 0 collapsed the model: pred_mean_p halved
                    # (0.00179 -> 0.00088) and AUPRC turned over
                    # (0.03848 -> 0.02377) exactly at level3_full_loss_epoch=65,
                    # when this tier took full weight. Target density is 0.00012,
                    # so 99.99% zeros dominate the BCE without this term.
                    #
                    # peak/multi keep peak_radius_* at EVERY tier. Those are
                    # decoupled from radius_* precisely so the profile is still
                    # shaped once the hit tolerance vanishes (utils/losses.py:118);
                    # without it the exact tier is per-voxel BCE with nothing
                    # penalising a flat profile.
                    if L_active >= 3:
                        if level_id <= 1:
                            tier = (rt, rh, rw, 0.10, gamma_peak_mid, delta_multi_mid)
                        elif level_id == 2:
                            tier = (0, 1, 1, 0.05, gamma_peak_mid, delta_multi_mid)
                        else:
                            tier = (0, 0, 0, 0.05, gamma_peak_final, delta_multi_final)
                    elif not is_final_refinement:
                        tier = (rt, rh, rw, 0.10, gamma_peak_mid, delta_multi_mid)
                    else:
                        tier = (0, 1, 1, 0.05, gamma_peak_final, delta_multi_final)
                    t_rt, t_rh, t_rw, t_beta, t_gamma, t_delta = tier

                    ref_parts = tolerant_spike_loss(
                        logits=logits_ref,
                        target=tgt_vol,
                        mask_vol=mask_vol,
                        pos_weight=pos_weight_eff,
                        radius_t=t_rt,
                        radius_h=t_rh,
                        radius_w=t_rw,
                        alpha_exact=1,
                        beta_hit=t_beta,
                        gamma_peak=t_gamma,
                        delta_multi=t_delta,
                        peak_radius_t=peak_radius_t,
                        peak_radius_h=peak_radius_h,
                        peak_radius_w=peak_radius_w,
                        return_parts=True,
                    )

                    ref_total = ref_parts["total"]
                    recon_total = recon_total + float(w_ref) * ref_total
                    
                    recon_level_vals.append(ref_total.detach())

                    recon_level_exact_vals.append(ref_parts["weighted_exact"].detach())
                    recon_level_tol_vals.append(ref_parts["weighted_tol"].detach())
                    recon_level_ids.append(level_id)

                isi_source = "none"
                adj_memory_used = 0.0
                adj_parts = None
                loss_isi = logits_vol_raw.new_zeros(())
                
                if lambda_isi > 0.0:
                    if memory_adj is None:
                        raise RuntimeError(
                            "lambda_isi > 0 but memory_adj is None. "
                            "Run/load Stage 0 GCT pretraining checkpoint with memory_adj."
                        )
                
                    if gct is None:
                        raise RuntimeError(
                            "lambda_isi > 0 but gct is None. "
                            "Cannot retrieve assaywise adjacency targets."
                        )
                
                    try:
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
                
                    except KeyError as e:
                        raise RuntimeError(
                            f"memory_adj lookup failed for this batch. "
                            f"Keys are missing or unstable. Original error: {e}"
                        )
                
                    with torch.cuda.amp.autocast(enabled=False):
                        adj_parts = short_gap_excess_loss_from_logits_batch_targets(
                            logits_b1thw=logits_vol_raw.float(),
                            target_gap_rates_bg=adj_target_bg.float(),
                            max_gap=isi_max_gap,
                            gap_bins=isi_gap_bins,
                            tau=isi_tau,
                            margin=isi_margin,
                            lower_margin=isi_lower_margin,
                            lower_weight=isi_lower_weight,
                            confidence_bg=adj_conf_bg.float(),
                            return_parts=True,
                            prob_threshold=active_prob_threshold,
                        )
                
                        loss_isi = adj_parts["loss"]
                        loss_isi = torch.nan_to_num(
                            loss_isi,
                            nan=0.0,
                            posinf=1e3,
                            neginf=0.0,
                        )
                
                    adj_memory_used = 1.0
                
                target_std_eff = target_std_end + 0.5 * (target_std_start - target_std_end) * (1 + math.cos(math.pi * t_pos))
                # loss_enc_var = variance_floor_loss(z_e_active, target_std=target_std_eff)
                loss_enc_var, enc_iso_parts = encoder_isotropy_loss(
                    z_e_active,
                    target_std=target_std_eff,
                    mean_weight=1e-2,
                    cov_weight=5e-2,
                    cov_margin=0.35,
                    max_tokens=1024,
                    return_parts=True,
                )
                
                # --- ctx schedule (0 until ctx_start_epoch, then cosine up to 1) ---
                t_ctx = (epoch - ctx_start_epoch) / max(1, ctx_warmup_epochs)
                t_ctx = min(1.0, max(0.0, t_ctx))
                ctx_ramp = 0.5 * (1 - math.cos(math.pi * t_ctx))
                lambda_ctx_eff = lambda_ctx * ctx_ramp
                lambda_sp_token_eff = lambda_sp_token * ctx_ramp

                t_pix = (epoch - sp_pixel_start_epoch) / max(1, sp_pixel_warmup_epochs)
                t_pix = min(1.0, max(0.0, t_pix))
                pixel_ramp = 0.5 * (1 - math.cos(math.pi * t_pix))
                
                lambda_sp_pixel_eff = lambda_sp_pixel * ctx_ramp * pixel_ramp
                lambda_isi_eff = lambda_isi * ctx_ramp
                # --- choose ctx dims (curriculum) ---

                
                ctx_dims = tuple(
                    d for d, ep0 in sorted(ctx_epoch_schedule.items())
                    if epoch >= ep0
                )
                
                loss_ctx = ctx_loss_soft(
                    logits_b1thw=logits_vol_raw,
                    ctx_tgt_b9=lct,
                    dims=ctx_dims,
                    weights=ctx_dim_weights,
                    tau=0.25,
                    prob_threshold=active_prob_threshold,
                )
                # Unweighted mirror of loss_ctx, logged only.  The stored
                # Stage 1 report recorded the unweighted quantity, so this is
                # what makes a rebalanced run comparable to it at matched
                # epochs.
                loss_ctx_raw = loss_ctx
                if ctx_dim_weights is not None:
                    with torch.no_grad():
                        loss_ctx_raw = ctx_loss_soft(
                            logits_b1thw=logits_vol_raw,
                            ctx_tgt_b9=lct,
                            dims=ctx_dims,
                            tau=0.25,
                            prob_threshold=active_prob_threshold,
                        )

                # Per-dimension ctx error, raw units.  This is the direct test
                # of whether rebalancing actually moved the under-supervised
                # moments, rather than inferring it from an aggregate.
                if log_ctx_diagnostics and len(ctx_dims) > 0:
                    with torch.no_grad():
                        _f = ctx_features_soft_from_logits(
                            logits_vol_raw.float(), tau=0.25,
                            prob_threshold=active_prob_threshold,
                        )
                        _e = (_f - lct.to(_f)).pow(2).mean(0)
                        for _d in ctx_dims:
                            _k = f"ctxerr_d{_d}"
                            sums[_k] = sums.get(_k, 0.0) + float(_e[_d].cpu())
                            ctx_diag_keys.add(_k)

                lambda_ctx_field_eff = (
                    float(lambda_ctx_field)
                    * float(ctx_ramp)
                )
                
                loss_ctx_field = logits_vol_raw.new_zeros(())
                
                if lambda_ctx_field_eff > 0.0:
                    loss_ctx_field = local_moment_field_loss(
                        logits_b1thw=logits_vol_raw,
                        target_b1thw=tgt_vol,
                        patch_size=model.patch_size,
                        tau=0.25,
                        min_active_spikes=1,
                        min_shape_spikes=5,
                        min_trend_spikes=6,
                        min_trend_frames=3,
                        prob_threshold=active_prob_threshold,
                    )
                
                # --- blank patch enforcing loss ---
                t_blank = (epoch - blank_start_epoch) / max(1, blank_warmup_epochs)
                t_blank = min(1.0, max(0.0, t_blank))
                blank_ramp = 0.5 * (1 - math.cos(math.pi * t_blank))
                lambda_blank_eff = lambda_blank * blank_ramp
                
                loss_blank = blank_patch_logit_hinge_loss(
                    pred_patches_raw=out["pred_patches_raw"],
                    blank_mask=out["blank_mask"],
                    margin=blank_logit_margin,
                    tau=0.25,
                    sharpness=10.0,
                )
                
                # --- blank-active decoder latent separation ---
                t_blank_sep = (epoch - blank_sep_start_epoch) / max(1, blank_sep_warmup_epochs)
                t_blank_sep = min(1.0, max(0.0, t_blank_sep))
                blank_sep_ramp = 0.5 * (1 - math.cos(math.pi * t_blank_sep))
                lambda_blank_sep_eff = lambda_blank_sep * blank_sep_ramp
                              
                lambda_offset_reg = 1e-4
                active_mask_sep = out["active_mask"].detach()

                if active_mask_sep.any() and (~active_mask_sep).any():
                    z_active_base = out["z_dec_base_no_pos"][active_mask_sep]
                    z_blank_base = out["z_dec_base_no_pos"][~active_mask_sep].mean(dim=0, keepdim=True)
                
                    sep_parts = blank_active_decoder_separation_loss(
                        z_blank_base=z_blank_base,
                        z_active_base=z_active_base,
                        offset=out["activity_type_offset"],
                        margin=blank_sep_margin,
                        offset_scale=model.offset_scale,
                        norm_reg_weight=lambda_offset_reg,
                        detach_base=True,
                    )
                
                    loss_blank_sep = sep_parts["total"]
                else:
                    loss_blank_sep = torch.tensor(0.0, device=device)

                # ---- bias-aware regularizer schedule ----
                sp_memory_used = 0.0
                loss_sp_token = torch.tensor(0.0, device=device)
                loss_sp_pixel = torch.tensor(0.0, device=device)
                
                # ----------------------------
                # Tokenwise spatial violation
                # ----------------------------
                if memory_tok is not None and gct is not None and lambda_sp_token_eff > 0:
                    try:
                        full_teacher_tok = memory_tok.get(
                            gct,
                            device=device,
                            dtype=logits_vol_raw.dtype,
                        )
                
                        _, _, _, Hp, Wp = logits_vol_raw.shape
                        _, pH, pW = model.patch_size
                        h_tok = Hp // pH
                        w_tok = Wp // pW
                
                        teacher_tok = sample_full_token_map_to_crop(
                            full_tok_bhw=full_teacher_tok,
                            out_tok_hw=(h_tok, w_tok),
                            patch_size=model.patch_size,
                            roi_hw=roi_hw,
                            pad_hw=pad_hw,
                        )
                
                        student_tok = soft_spatial_token_map_from_logits(
                            logits_b1thw=logits_vol_raw,
                            patch_size=model.patch_size,
                            tau=0.25,
                            prob_threshold=active_prob_threshold,
                        )
                
                        teacher_tok_tol = dilate_spatial_support_hw(
                            teacher_tok.detach(),
                            radius_h=1,
                            radius_w=1,
                        )
                
                        loss_sp_token = spatial_support_violation_loss(
                            pred_support=student_tok,
                            allowed_support=teacher_tok_tol,
                            neg_thresh=0.20,
                        )
                
                        sp_memory_used = 1.0
                
                    except KeyError:
                        loss_sp_token = torch.tensor(0.0, device=device)
                
                
                # ----------------------------
                # Pixelwise spatial violation
                # ----------------------------
                if memory_pix is not None and gct is not None and lambda_sp_pixel_eff > 0:
                    try:
                        full_teacher_pix = memory_pix.get(
                            gct,
                            device=device,
                            dtype=logits_vol_raw.dtype,
                        )
                
                        _, _, _, Hp, Wp = logits_vol_raw.shape
                
                        teacher_pix = sample_full_pixel_map_to_crop(
                            full_pix_bhw=full_teacher_pix,
                            out_hw=(Hp, Wp),
                            roi_hw=roi_hw,
                            pad_hw=pad_hw,
                        )
                
                        student_pix = soft_spatial_pixel_map_from_logits(
                            logits_b1thw=logits_vol_raw,
                            tau=0.25,
                            prob_threshold=active_prob_threshold,
                        )
                
                        teacher_pix_tol = dilate_spatial_support_hw(
                            teacher_pix.detach(),
                            radius_h=1,
                            radius_w=1,
                        )
                
                        loss_sp_pixel = spatial_support_violation_loss(
                            pred_support=student_pix,
                            allowed_support=teacher_pix_tol,
                            neg_thresh=0.20,
                        )
                
                        sp_memory_used = 1.0
                
                    except KeyError:
                        loss_sp_pixel = torch.tensor(0.0, device=device)
                
                loss_sp_cons = lambda_sp_token_eff * loss_sp_token + lambda_sp_pixel_eff * loss_sp_pixel
            # --- Total loss      
            loss = (
                recon_total
                + lambda_vq * vq_loss
                + lambda_isi_eff * loss_isi
                + lambda_enc_var * loss_enc_var
                + lambda_code_norm * loss_code_norm
                + lambda_ctx_eff * loss_ctx
                + lambda_ctx_field_eff * loss_ctx_field
                + loss_sp_cons
                + lambda_blank_eff * loss_blank + lambda_blank_sep_eff * loss_blank_sep
            )
            
            scaler.scale(loss / grad_accum_steps).backward()
            accum += 1
            
            # --- scalar logging accumulators
            total_loss += float(loss.detach().cpu())
            
            sums["loss_recon_total"] += float(recon_total.detach().cpu())
            
            sums["loss_vq"] += float(vq_loss.detach().cpu())
            sums["loss_enc_var"] += float(loss_enc_var.detach().cpu())
            sums["loss_code_norm"] += float(loss_code_norm.detach().cpu())
            
            sums["loss_blank"] += float(loss_blank.detach().cpu())
            sums["loss_blank_sep"] += float(loss_blank_sep.detach().cpu())
            
            sums["loss_isi"] += float(loss_isi.detach().cpu())

            latent_hidden_mask = out.get("latent_hidden_mask", None)
            if latent_hidden_mask is not None:
                sums["latent_hidden_frac"] += float(
                    latent_hidden_mask.float().mean().detach().cpu()
                )
                # ~92% of tokens are blank at this spike density, so the
                # overall hidden fraction badly overstates how much
                # information the hole actually removes.  What matters is the
                # share of ACTIVE tokens hidden.
                active_mask_enc = out.get("active_mask", None)
                if active_mask_enc is not None:
                    n_active = active_mask_enc.sum()
                    if int(n_active) > 0:
                        sums["latent_hidden_active_frac"] += float(
                            (
                                (latent_hidden_mask & active_mask_enc).sum()
                                / n_active
                            ).detach().cpu()
                        )

            sums["loss_ctx"] += float(loss_ctx.detach().cpu())
            sums["loss_ctx_raw"] += float(loss_ctx_raw.detach().cpu())

            sums["loss_ctx_field"] += float(loss_ctx_field.detach().cpu())
            sums["loss_sp_cons"] += float(loss_sp_cons.detach().cpu())
            sums["loss_sp_token"] += float(loss_sp_token.detach().cpu())
            sums["loss_sp_pixel"] += float(loss_sp_pixel.detach().cpu())
            
            sums["sp_memory_used"] += float(sp_memory_used)
            sums["adj_memory_used"] += float(adj_memory_used)
            
            # --- adjacency diagnostics
            if adj_parts is not None:
                sums["adj_pred_mean"] += float(
                    adj_parts["pred_gap_rates"].mean().detach().cpu()
                )
                sums["adj_upper_allowed_mean"] += float(
                    adj_parts["allowed_gap_rates"].mean().detach().cpu()
                )
                sums["adj_lower_allowed_mean"] += float(
                    adj_parts["lower_allowed_gap_rates"].mean().detach().cpu()
                )
                sums["adj_tgt_mean"] += float(
                    adj_parts["target_gap_rates"].mean().detach().cpu()
                )
            
                if adj_parts.get("confidence", None) is not None:
                    sums["adj_conf_mean"] += float(
                        adj_parts["confidence"].mean().detach().cpu()
                    )
            
            # --- per-refinement reconstruction diagnostics
            for level_id, val in zip(recon_level_ids, recon_level_vals):
                sums[f"loss_recon_l{level_id}"] += float(val.detach().cpu())
            
            for level_id, val in zip(recon_level_ids, recon_level_exact_vals):
                sums[f"loss_recon_l{level_id}_exact"] += float(val.detach().cpu())
                
            # --- weighted totals (match recon_total weighting)
            for ridx, (exact_val, tol_val, w_ref) in enumerate(
                zip(recon_level_exact_vals, recon_level_tol_vals, refinement_loss_weights_batch),
                start=1,
            ):
                sums["loss_recon_exact_total"] += float(w_ref * exact_val.detach().cpu())
                sums["loss_recon_tol_total"] += float(w_ref * tol_val.detach().cpu())
            
            for level_id, val in zip(recon_level_ids, recon_level_tol_vals):
                sums[f"loss_recon_l{level_id}_tol"] += float(val.detach().cpu())
                        
            pred_mean_p_epoch = sum_pred_mean_p / max(1, num_batches)
            tgt_mean_epoch    = sum_tgt_mean    / max(1, num_batches)


            if accum == grad_accum_steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum = 0

        if accum > 0:
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

        lrs = [pg["lr"] for pg in optimizer.param_groups]
        lr = max(lrs)  # or keep all of them for logging
        
        with torch.no_grad():
            blank_token_norm = (
                float(model.vq.blank_token.detach().norm().item())
                if hasattr(model, "vq") and hasattr(model.vq, "blank_token")
                else 0.0
            )
        
            activity_offset_norm = (
                float(model.activity_type_offset.detach().norm().item())
                if hasattr(model, "activity_type_offset")
                else 0.0
            )
                
        
        train_log = {
            "epoch": epoch,
            "time_sec": time.time() - t0,
            "lr": lr,
        
            # total + components
            "loss": total_loss / max(1, num_batches),
        
            # reconstruction naming
            "loss_recon_total": sums["loss_recon_total"] / max(1, num_batches),
            "loss_recon_exact_total": sums["loss_recon_exact_total"] / max(1, num_batches),
            "loss_recon_tol_total": sums["loss_recon_tol_total"] / max(1, num_batches),
            
            "loss_vq": sums["loss_vq"] / max(1, num_batches),
            "loss_enc_var": sums["loss_enc_var"] / max(1, num_batches),
            "loss_code_norm": sums["loss_code_norm"] / max(1, num_batches),
            "blank": sums["loss_blank"] / max(1, num_batches),
            "blank_sep": sums["loss_blank_sep"] / max(1, num_batches),
            "loss_isi": sums["loss_isi"] / max(1, num_batches),
            
            "loss_ctx": sums["loss_ctx"] / max(1, num_batches),
            "loss_ctx_raw": sums["loss_ctx_raw"] / max(1, num_batches),
            "loss_ctx_field": sums["loss_ctx_field"] / max(1, num_batches),
            "latent_hidden_frac": sums["latent_hidden_frac"] / max(1, num_batches),
            "latent_hidden_active_frac": (
                sums["latent_hidden_active_frac"] / max(1, num_batches)
            ),
            "loss_sp_cons": sums["loss_sp_cons"] / max(1, num_batches),
            "loss_sp_token": sums["loss_sp_token"] / max(1, num_batches),
            "loss_sp_pixel": sums["loss_sp_pixel"] / max(1, num_batches),
            "lambda_sp_token_eff": lambda_sp_token_eff,
            "lambda_sp_pixel_eff": lambda_sp_pixel_eff,
            
            "sp_memory_used": sums["sp_memory_used"] / max(1, num_batches),
            "adj_memory_used": sums["adj_memory_used"] / max(1, num_batches),
            "adj_pred_mean": sums["adj_pred_mean"] / max(1, num_batches),
            "adj_upper_allowed_mean": sums["adj_upper_allowed_mean"] / max(1, num_batches),
            "adj_lower_allowed_mean": sums["adj_lower_allowed_mean"] / max(1, num_batches),
            "adj_tgt_mean": sums["adj_tgt_mean"] / max(1, num_batches),
            "adj_conf_mean": sums["adj_conf_mean"] / max(1, num_batches),
                        
            # vq information
            "avg_vq_blank_frac": vq_metric_sums["blank_frac"] / max(1, num_batches),
            "avg_vq_residual_norm": vq_metric_sums["residual_norm"] / max(1, num_batches),
            "avg_vq_commit": vq_metric_sums["commit_loss"] / max(1, num_batches),
            "avg_vq_usage": vq_metric_sums["usage_loss"] / max(1, num_batches),
            
            "blank_token_norm": blank_token_norm,
            "activity_offset_norm": activity_offset_norm,

            # schedules
            "pos_weight_eff": float(pos_weight_eff),
            "lambda_ctx_eff": float(lambda_ctx_eff),
            "lambda_isi": float(lambda_isi),
            "lambda_vq": float(lambda_vq),
            "cfg_drop_prob": float(cfg_p),
            "training_prob_threshold_used": active_prob_threshold,
            
            # activities
            "pred_mean_p": pred_mean_p_epoch,
            "tgt_mean": tgt_mean_epoch,

        }
        
        for ridx in range(1, max_ref_levels + 1):
            train_log[f"loss_recon_l{ridx}"] = sums[f"loss_recon_l{ridx}"] / max(1, num_batches)
            train_log[f"loss_recon_l{ridx}_exact"] = sums[f"loss_recon_l{ridx}_exact"] / max(1, num_batches)
            train_log[f"loss_recon_l{ridx}_tol"] = sums[f"loss_recon_l{ridx}_tol"] / max(1, num_batches)
            
        for lvl in range(1, max_ref_levels + 1):
            train_log[f"perplexity_nonblank_l{lvl}"] = (
                vq_metric_sums.get(f"perplexity_nonblank_l{lvl}", 0.0) / max(1, num_batches)
            )
            train_log[f"active_codes_nonblank_l{lvl}"] = (
                vq_metric_sums.get(f"active_codes_nonblank_l{lvl}", 0.0) / max(1, num_batches)
            )
            train_log[f"child_per_active_parent_l{lvl}"] = (
                vq_metric_sums.get(f"child_per_active_parent_l{lvl}", 0.0) / max(1, num_batches)
            )
            train_log[f"vq_commit_l{lvl}"] = (
                vq_metric_sums.get(f"commit_loss_l{lvl}", 0.0) / max(1, num_batches)
            )
            train_log[f"vq_usage_l{lvl}"] = (
                vq_metric_sums.get(f"usage_loss_l{lvl}", 0.0) / max(1, num_batches)
            )
            train_log[f"vq_soft_usage_raw_l{lvl}"] = (
                vq_metric_sums.get(f"soft_usage_loss_l{lvl}", 0.0) / max(1, num_batches)
            )
        cb_stats = get_vq_codebook_stats(model)
        train_log.update(cb_stats)
    
        val_metric = None

        if val_loader is not None:
            eval_report = evaluate_vqvae(
                model,
                val_loader,
                use_amp=use_amp,
                pos_weight=1,
                use_ROI_mask=use_ROI_mask,
                recon_tolerance=recon_tolerance,
                metric_tolerance=metric_tolerance,
                mask_latents=mask_latents,
                latent_recon_drop_p=latent_recon_drop_p,
            )

            if val_metric_name in eval_report:
                val_metric = eval_report[val_metric_name]
            else:
                val_metric = (
                    -eval_report["val_loss_BCE"]
                    if val_metric_goal == "max"
                    else eval_report["val_loss_BCE"]
                )

            train_log[val_metric_name] = val_metric
            # Log BOTH AUPRC variants regardless of which one selects. Only the
            # selected metric was stored, so every comparison ran on the tolerant
            # number alone -- which structurally favours a diffuse field, since
            # radius (1,1,1) is a 27-voxel neighbourhood a blurry model harvests
            # cheaply. A sharper model looked worse with no way to see otherwise.
            for _k in ("AUPRC", "AUPRC_tol"):
                if eval_report is not None and _k in eval_report:
                    train_log[_k] = float(eval_report[_k])

            # These remain raw evaluation operating points.
            exact_thr = eval_report["BestF1_threshold"]
            tolerant_thr = eval_report["BestF1_threshold_tol"]

            model._set_best_thresholds(
                exact=exact_thr,
                tolerant=tolerant_thr,
            )

            history["val_metrics"].append(eval_report)

        for key in sorted(ctx_diag_keys):
            train_log[key] = sums[key] / max(1, num_batches)

        history["train_log"].append(train_log)

        if on_epoch:
            on_epoch(train_log)


        log_str = (
            f"[Epoch {epoch}] lr={lr:.6g} "
            f"train_loss={train_log['loss']:.5f}\n"
            f"\n"
            
            f"l1={train_log.get('loss_recon_l1', 0):.5f} "
            f"l1_exact={train_log.get('loss_recon_l1_exact', 0):.5f} "
            f"l1_tol={train_log.get('loss_recon_l1_tol', 0):.5f} "
            f"l2={train_log.get('loss_recon_l2', 0):.5f} "
            f"l2_exact={train_log.get('loss_recon_l2_exact', 0):.5f} "
            f"l2_tol={train_log.get('loss_recon_l2_tol', 0):.5f}\n"
            
            f"recon_total={train_log.get('loss_recon_total', 0):.5f} "
            f"exact_total={train_log.get('loss_recon_exact_total', 0):.5f} "
            f"tol_total={train_log.get('loss_recon_tol_total', 0):.5f}\n"
            f"\n"
            
            f"vq={train_log.get('loss_vq', 0):.5f} "
            f"enc_var={train_log.get('loss_enc_var', 0):.5f} "
            f"code_norm={train_log.get('loss_code_norm', 0):.5f} "
            f"ctx={train_log.get('loss_ctx', 0):.5f} "
            f"ctx_field={train_log.get('loss_ctx_field', 0):.5f} "
            f"sp_cons={train_log.get('loss_sp_cons', 0):.6e} "
            f"isi={train_log.get('loss_isi', 0):.5f} "
            f"sp_mem={train_log.get('sp_memory_used', 0):.2f} "
            f"adj_mem={train_log.get('adj_memory_used', 0):.2f} "
            f"adj_pred={train_log.get('adj_pred_mean', 0):.5e} "
            f"adj_allowed=[{train_log.get('adj_lower_allowed_mean', 0):.5e}, {train_log.get('adj_upper_allowed_mean', 0):.5e}]\n"
            f"adj_conf={train_log.get('adj_conf_mean', 0):.3f}\n"
            f"blank={train_log.get('blank', 0):.5f} "
            f"blank_sep={train_log.get('blank_sep', 0):.5f}\n"            
            f"\n"
            f"\n"
            
            f"pred_mean_p={train_log.get('pred_mean_p', 0):.6e}, "
            f"tgt_mean={train_log.get('tgt_mean', 0):.6e}\n"            
            f"\n"
            f"---"
            f"\n\n"
            
            f"avg_vq_blank_frac={train_log.get('avg_vq_blank_frac', 0):.3f} "
            f"avg_vq_resid={train_log.get('avg_vq_residual_norm', 0):.5f} "
            f"vq_commit={train_log.get('avg_vq_commit', 0):.5f} "
            f"vq_usage={train_log.get('avg_vq_usage', 0):.5f}\n"
            f"\n"
            
            f"vq_l1_ppl={train_log.get('perplexity_nonblank_l1', 0):.2f} "
            f"vq_l2_ppl={train_log.get('perplexity_nonblank_l2', 0):.2f} "
            f"vq_l1_act={train_log.get('active_codes_nonblank_l1', 0):.1f} "
            f"vq_l2_flat_act={train_log.get('active_codes_nonblank_l2', 0):.1f} "
            f"child/parent={train_log.get('child_per_active_parent_l2', 0):.2f}\n"
            f"\n"
            
            f"vq_commit_l1={train_log.get('vq_commit_l1', 0):.5f} "
            f"vq_commit_l2={train_log.get('vq_commit_l2', 0):.5f} "
            f"vq_usage_l1={train_log.get('vq_usage_l1', 0):.5f} "
            f"vq_usage_l2={train_log.get('vq_usage_l2', 0):.5f} "
            f"vq_usage_raw_l1={train_log.get('vq_soft_usage_raw_l1', 0):.5f} "
            f"vq_usage_raw_l2={train_log.get('vq_soft_usage_raw_l2', 0):.5f}\n"
            f"\n"

            f"cb_total_alive={train_log.get('vq_total_alive', 0)}/{train_log.get('vq_total_codes', 0)} "
            f"({train_log.get('vq_total_alive_frac', 0):.3f}) "
            f"dead={train_log.get('vq_total_dead', 0)}\n"
            f"\n"

            f"cb_l1: alive={train_log.get('vq_l1_alive', 0)}/{train_log.get('vq_l1_num_codes', 0)} "
            f"dead={train_log.get('vq_l1_dead', 0)} "
            f"norm=({train_log.get('vq_l1_norm_min', 0):.4f},"
            f"{train_log.get('vq_l1_norm_mean', 0):.4f}±{train_log.get('vq_l1_norm_std', 0):.4f},"
            f"{train_log.get('vq_l1_norm_max', 0):.4f}) "
            f"cos=({train_log.get('vq_l1_cos_min', 0):.4f},"
            f"{train_log.get('vq_l1_cos_mean', 0):.4f}±{train_log.get('vq_l1_cos_std', 0):.4f},"
            f"{train_log.get('vq_l1_cos_max', 0):.4f})\n"
            f"\n"

            f"cb_l2: alive={train_log.get('vq_l2_alive', 0)}/{train_log.get('vq_l2_num_codes', 0)} "
            f"dead={train_log.get('vq_l2_dead', 0)} "
            f"norm=({train_log.get('vq_l2_norm_min', 0):.4f},"
            f"{train_log.get('vq_l2_norm_mean', 0):.4f}±{train_log.get('vq_l2_norm_std', 0):.4f},"
            f"{train_log.get('vq_l2_norm_max', 0):.4f}) "
            f"cos=({train_log.get('vq_l2_cos_min', 0):.4f},"
            f"{train_log.get('vq_l2_cos_mean', 0):.4f}±{train_log.get('vq_l2_cos_std', 0):.4f},"
            f"{train_log.get('vq_l2_cos_max', 0):.4f})\n"
            f"\n"
            
            f"blank_norm={train_log.get('blank_token_norm', 0):.5f} "
            f"offset_norm={train_log.get('activity_offset_norm', 0):.5f}\n"
            f"---"
            f"\n\n"

            f"val_{val_metric_name}={(val_metric if val_metric is not None else 0.0):.5f}"
        )

        if val_loader is not None and eval_report is not None:
            if "val_loss_BCE" in eval_report:
                log_str += f"  val_BCE_tol={eval_report['val_loss_BCE']:.5f}"
            if "val_loss_BCE_full" in eval_report:
                log_str += f"  val_BCE_full_tol={eval_report['val_loss_BCE_full']:.5f}"

            # exact metrics
            if "AUPRC" in eval_report:
                log_str += f"\nAUPRC_exact={eval_report['AUPRC']:.6f}"
            if "BestF1" in eval_report:
                log_str += f"  BestF1_exact={eval_report['BestF1']:.4f}"
            if "BestF1_threshold" in eval_report:
                log_str += f"  Thr_exact={eval_report['BestF1_threshold']:.3f}"

            # tolerant metrics
            if "AUPRC_tol" in eval_report:
                log_str += f"\nAUPRC_tol={eval_report['AUPRC_tol']:.6f}"
            if "BestF1_tol" in eval_report:
                log_str += f"  BestF1_tol={eval_report['BestF1_tol']:.4f}"
            if "BestF1_threshold_tol" in eval_report:
                log_str += f"  Thr_tol={eval_report['BestF1_threshold_tol']:.3f}"

        print(log_str)

        if log_ctx_diagnostics:
            ctx_lines = [
                "  [ctx] hidden_frac=%.3f hidden_active_frac=%.3f"
                % (
                    train_log.get("latent_hidden_frac", 0.0),
                    train_log.get("latent_hidden_active_frac", 0.0),
                )
            ]
            if val_loader is not None and len(history["val_metrics"]) > 0:
                last_val = history["val_metrics"][-1]
                ctx_lines.append(
                    "  [ctx] specificity_bce=%+.5f"
                    % (last_val.get("ctx_specificity_bce", float("nan")),)
                )
            print("\n".join(ctx_lines))
        print(" ")

        
        print("===========================================")
        print(" ")


        improved = False
        checkpointing_active = (epoch >= save_start_epoch)
        
        if val_loader is not None and val_metric is not None:
            if checkpointing_active:
                if ((val_metric_goal == "max" and val_metric > best_val) or (val_metric_goal == "min" and val_metric < best_val)):
                    improved = True
                    best_val = val_metric
                    best_epoch = epoch
                    best_thr_exact_for_best_model = float(
                        model.best_thr_exact.item()
                    )
                    
                    best_thr_tol_for_best_model = float(
                        model.best_thr_tol.item()
                    )
                    no_improve = 0
                else:
                    no_improve += 1



        # ---- always save last checkpoint with current threshold ----
        if ckpt_last_path:
            model.save_checkpoint(ckpt_last_path)

        # ---- save best checkpoint only when validation metric improves ----
        if improved and ckpt_best_path :
            model.save_checkpoint(ckpt_best_path)
            print(
                f"✔️  Saved new best model at epoch {epoch} to '{ckpt_best_path}' "
                f"(val_{val_metric_name} = {val_metric:.5f}, "
                f"thr_exact = {best_thr_exact_for_best_model:.4f}, "
                f"thr_tol = {best_thr_tol_for_best_model:.4f})"
            )

        if early_stop_patience is not None and no_improve >= early_stop_patience:
            print(f"⏹️  Early stopping triggered. No improvement for {early_stop_patience} consecutive epochs.")
            break

    return {
        "best_epoch": best_epoch,
        "best_val": best_val,
        "best_thr_exact": best_thr_exact_for_best_model,
        "best_thr_tol": best_thr_tol_for_best_model,
        "final_training_prob_threshold": float(model.training_prob_threshold.item()),
        "history": history,
    }
