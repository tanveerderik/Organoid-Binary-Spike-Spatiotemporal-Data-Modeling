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
    variance_floor_loss,
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
    
    # ---- training parameters ---
    use_logit_bias_schedule: bool = True,
    logit_bias_start: float = -2.5,
    logit_bias_end: float = 0.0,
    logit_bias_decay_epochs: int = 15,

    pos_weight_start: float = 50.0,
    pos_weight_end: float = 20.0,
    pos_decay_epochs: int = 50,        # reach pos_weight by epoch 50

    lambda_isi: float = 1e-2,
    lambda_ctx: float = 1e-3,
    
    ctx_start_epoch: int = 30,         # start ctx after 20
    ctx_warmup_epochs: int = 50,       # ramp duration (or shorter, like 10–15)
    
    # ---- hierarchical refinement supervision ----
    refinement_loss_weights: Optional[list[float]] = None,
    refinement_use_raw_logits: bool = True,
    
    # ---- hierarchical VQ schedule ----
    level2_start_epoch: int = 20,
    level2_full_loss_epoch: int = 40,
    
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
    
    # --- Blank embedding and patch enforcement ---
    lambda_blank: float = 0.05,
    blank_logit_margin: float = -6.0,
    blank_start_epoch: int = 1,
    blank_warmup_epochs: int = 20,
    lambda_blank_sep: float = 0.01,
    blank_sep_margin: float = 0.0,
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
    val_metric_name: str = "AUPRC_tol_cond",
    val_metric_goal: str = "max",  # or "min"
    use_ROI_mask: bool = False,
    
    # ---- ISI maintenance loss terms ----
    isi_max_gap: int = 3,
    isi_tau: float = 0.25,
    isi_margin: float = 0.25,
    
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
    best_thr_for_best_model = 0.5
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

    def _scheduled_logit_bias(epoch_idx: int) -> float:
        if not use_logit_bias_schedule:
            return float(logit_bias_end)

        if logit_bias_decay_epochs <= 0:
            return float(logit_bias_end)

        frac = min(1.0, max(0.0, (epoch_idx - 1) / float(logit_bias_decay_epochs)))
        return float(logit_bias_start + frac * (logit_bias_end - logit_bias_start))

    for epoch in range(1, epochs + 1):
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
            "loss_sp_cons": 0.0,
            "loss_sp_token": 0.0,
            "loss_sp_pixel": 0.0,
            
            "sp_memory_used": 0.0,
            "adj_memory_used": 0.0,
            "adj_pred_mean": 0.0,
            "adj_allowed_mean": 0.0,
            "adj_tgt_mean": 0.0,
            "adj_conf_mean": 0.0,
                        
            "loss_enc_var": 0.0,
            "loss_blank": 0.0,
            "loss_blank_sep": 0.0,
            
        }
        
        # ---- hierarchy schedule ----
        if epoch < level2_start_epoch:
            model.vq.set_active_quantizers(1)
        else:
            model.vq.set_active_quantizers(2)
        
        max_ref_levels = int(getattr(model.vq, "num_quantizers", 1))
        
        
        for ridx in range(1, max_ref_levels + 1):
            sums[f"loss_recon_l{ridx}"] = 0.0
            sums[f"loss_recon_l{ridx}_exact"] = 0.0
            sums[f"loss_recon_l{ridx}_tol"] = 0.0
        
        vq_metric_sums = {
            "perplexity_nonblank": 0.0,
            "entropy_nonblank": 0.0,
            "active_codes_nonblank": 0.0,
            "active_frac_nonblank": 0.0,
            "blank_frac": 0.0,
            "num_nonblank": 0.0,
            "commit_loss": 0.0,
            "residual_norm": 0.0,
        }
        
        sum_pred_mean_p = 0.0
        sum_tgt_mean    = 0.0
        num_batches = 0
        
        def _encode_code_logits_only(x_in: torch.Tensor, global_ctx=None, local_ctx=None):
            """
            Encoder + VQ only (no decoder).

            Returns:
              codes_hat: (B,N,L) long
              code_logits_hat: (B,N,L,K) float
              grid: (t_tok,h_tok,w_tok)

            Important:
              - no VQ EMA update in this path
              - x_in is expected to be logits-like / continuous, not binary
            """
            x_stem = model.stem(x_in)
            tokens, grid, blank_mask, active_mask = model.patch_embed(x_stem)
            pos_enc, _ = model._get_pos_embed(grid, tokens.device, tokens.dtype)

            x_enc_full, _, _, _ = model.sparse_encoder(
                tokens=tokens,
                active_mask=active_mask,
                pos_embed=pos_enc,
                fill_value=0.0,
            )

            z_e_full = model.to_code(x_enc_full)

            B2, N2, D2 = z_e_full.shape
            active_flat2 = active_mask.reshape(B2 * N2)
            z_e_flat2 = z_e_full.reshape(B2 * N2, D2)
            z_e_active2 = z_e_flat2[active_flat2]

            was_train = model.vq.training
            model.vq.eval()
            try:
                _, _, codes_flat_hat, code_logits_flat_hat, _ = model.vq.quantize_active_only(
                    z_e_active=z_e_active2,
                    active_flat=active_flat2,
                    num_total_tokens=B2 * N2,
                    return_logits=True,
                    return_aux=True,
                )
            finally:
                model.vq.train(was_train)

            codes_hat = codes_flat_hat.view(B2, N2, model.vq.num_quantizers)
            code_logits_hat = code_logits_flat_hat.view(
                B2, N2, model.vq.num_quantizers, model.vq.max_num_codes
            )

            return codes_hat, code_logits_hat, grid
        

        cfg_p = _cosine_ramp(
            epoch_idx=epoch,
            start_epoch=cfg_ctx_start_epoch,
            warmup_epochs=cfg_ctx_warmup_epochs,
            v0=cfg_ctx_drop_start,
            v1=cfg_ctx_drop_end,
        )
        
        # ---- dynamic refinement supervision ----
        L_active = int(getattr(model.vq, "active_quantizers", model.vq.num_quantizers))
        
        if L_active == 1:
            refinement_loss_weights_eff = [1.0]
        elif L_active == 2:
            if epoch < level2_full_loss_epoch:
                refinement_loss_weights_eff = [1.0, 0.5]
            else:
                refinement_loss_weights_eff = [0.75, 1.0]
        else:
            raise ValueError(f"Unsupported active quantizer count: {L_active}")
        
        current_logit_bias = _scheduled_logit_bias(epoch)
        if hasattr(model, "set_logit_bias"):
            model.set_logit_bias(current_logit_bias)
        
        for batch in train_loader:
            num_batches += 1
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
            
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = model(
                    x,
                    global_ctx=gct,
                    local_ctx=lct,
                    predict_mask_spec=predict_mask_spec,
                    cfg_ctx_drop_p=cfg_p,
                    roi_hw=roi_hw,
                    pad_hw=pad_hw,
                )                
                vq_loss = out["vq_loss"]
                z_e_active = out["z_e_active"]
                logits_patches = out["pred_patches"]
                grid = out["grid"]
                predict_mask = out["predict_mask"]
                logits_vol = out["logits_vol"]
                logits_vol_raw = out["logits_vol_raw"]
                prob_vol_raw = torch.sigmoid(logits_vol_raw/tau_logit)
                
                refinements = out.get("refinements", None)
                if refinements is None or len(refinements) == 0:
                    raise RuntimeError("Model output is missing non-empty 'refinements'.")
                
                vq_aux = out.get("vq_aux", {})
                
                if vq_aux:
                    vq_metric_sums["perplexity_nonblank"] += float(vq_aux.get("perplexity_nonblank", 0.0))
                    vq_metric_sums["entropy_nonblank"] += float(vq_aux.get("entropy_nonblank", 0.0))
                    vq_metric_sums["active_codes_nonblank"] += float(vq_aux.get("active_codes_nonblank", 0.0))
                    vq_metric_sums["active_frac_nonblank"] += float(vq_aux.get("active_frac_nonblank", 0.0))
                    vq_metric_sums["blank_frac"] += float(vq_aux.get("blank_frac", 0.0))
                    vq_metric_sums["num_nonblank"] += float(vq_aux.get("num_nonblank", 0.0))
                    vq_metric_sums["commit_loss"] += float(vq_aux.get("commit_loss", 0.0))
                
                levels_aux = vq_aux.get("levels", []) if vq_aux else []
                for lvl, lvl_aux in enumerate(levels_aux):
                    key_ppl = f"perplexity_nonblank_l{lvl+1}"
                    key_act = f"active_codes_nonblank_l{lvl+1}"
                    key_commit = f"commit_loss_l{lvl+1}"
                
                    if key_ppl not in vq_metric_sums:
                        vq_metric_sums[key_ppl] = 0.0
                        vq_metric_sums[key_act] = 0.0
                        vq_metric_sums[key_commit] = 0.0
                
                    vq_metric_sums[key_ppl] += float(lvl_aux.get("perplexity_nonblank", 0.0))
                    vq_metric_sums[key_act] += float(lvl_aux.get("active_codes_nonblank", 0.0))
                    vq_metric_sums[key_commit] += float(lvl_aux.get("commit_loss", 0.0))
                vq_metric_sums["residual_norm"] += float(vq_aux.get("residual_norm", 0.0))
                
            

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
                if len(refinement_loss_weights_eff) != num_ref:
                    raise ValueError(
                        f"refinement_loss_weights_eff length {len(refinement_loss_weights_eff)} "
                        f"does not match num refinements {num_ref}"
                    )

                 
                # ---- early/intermediate refinements: tolerant ----
                
                recon_total = logits_vol_raw.new_zeros(())

                recon_level_vals = []
                recon_level_exact_vals = []
                recon_level_tol_vals = []
                
                for ridx, (ref, w_ref) in enumerate(
                    zip(refinements, refinement_loss_weights_eff),
                    start=1,
                ):
                    logits_ref = _get_ref_logits(ref)
                
                    is_final_refinement = ridx == num_ref
                
                    if not is_final_refinement:
                        # ---- early/intermediate refinements: tolerant coarse supervision ----
                        ref_parts = tolerant_spike_loss(
                            logits=logits_ref,
                            target=tgt_vol,
                            mask_vol=mask_vol,
                            pos_weight=pos_weight_eff,
                            radius_t=rt,
                            radius_h=rh,
                            radius_w=rw,
                            alpha_exact=1,
                            beta_hit=0.1,
                            gamma_peak=0.05,
                            delta_multi=0.05,
                            return_parts=True,
                        )
                    else:
                        # ---- final refinement: mostly exact, weak local-tolerance auxiliary ----
                        ref_parts = tolerant_spike_loss(
                            logits=logits_ref,
                            target=tgt_vol,
                            mask_vol=mask_vol,
                            pos_weight=pos_weight_end,
                            radius_t=0,
                            radius_h=1,
                            radius_w=1,
                            alpha_exact=1,
                            beta_hit=0.05,
                            gamma_peak=0.01,
                            delta_multi=0.01,
                            return_parts=True,
                        )
                
                    ref_total = ref_parts["total"]
                    recon_total = recon_total + float(w_ref) * ref_total
                    
                    recon_level_vals.append(ref_total.detach())

                    recon_level_exact_vals.append(ref_parts["weighted_exact"].detach())
                    recon_level_tol_vals.append(ref_parts["weighted_tol"].detach())


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
                            tau=isi_tau,
                            margin=isi_margin,
                            confidence_bg=adj_conf_bg.float(),
                            return_parts=True,
                        )
                
                        loss_isi = adj_parts["loss"]
                        loss_isi = torch.nan_to_num(
                            loss_isi,
                            nan=0.0,
                            posinf=1e3,
                            neginf=0.0,
                        )
                
                    adj_memory_used = 1.0
                    isi_source = "memory_adj"

                
                target_std_eff = target_std_end + 0.5 * (target_std_start - target_std_end) * (1 + math.cos(math.pi * t_pos))
                loss_enc_var = variance_floor_loss(z_e_active, target_std=target_std_eff)
                
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
                    ctx_tgt_b5=lct,
                    dims=ctx_dims,
                    tau=0.25,
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
                
                active_mask_sep = out["active_mask"].detach()
                
                if active_mask_sep.any():
                    z_active_dec = out["z_dec_typed"][active_mask_sep].detach()
                
                    blank_type = model.token_type_embed.weight[0:1].to(
                        device=out["z_dec_typed"].device,
                        dtype=out["z_dec_typed"].dtype,
                    )
                
                    z_blank_core = model.code_to_dec(
                        model.vq.blank_token.to(
                            device=out["z_dec_typed"].device,
                            dtype=out["z_dec_typed"].dtype,
                        ).view(1, -1)
                    ).detach()
                
                    z_blank_dec = z_blank_core + model.token_type_scale * blank_type
                
                    loss_blank_sep = blank_active_decoder_separation_loss(
                        z_blank_dec=z_blank_dec,
                        z_active_dec=z_active_dec,
                        margin=blank_sep_margin,
                    )
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
                + vq_loss
                + lambda_isi_eff * loss_isi
                + lambda_enc_var * loss_enc_var
                + lambda_ctx_eff * loss_ctx
                + loss_sp_cons
                + lambda_blank_eff * loss_blank + + lambda_blank_sep_eff * loss_blank_sep
            )
            
            scaler.scale(loss / grad_accum_steps).backward()
            accum += 1
            
            # --- scalar logging accumulators
            total_loss += float(loss.detach().cpu())
            
            sums["loss_recon_total"] += float(recon_total.detach().cpu())
            
            sums["loss_vq"] += float(vq_loss.detach().cpu())
            sums["loss_enc_var"] += float(loss_enc_var.detach().cpu())
            
            sums["loss_blank"] += float(loss_blank.detach().cpu())
            sums["loss_blank_sep"] += float(loss_blank_sep.detach().cpu())
            
            sums["loss_isi"] += float(loss_isi.detach().cpu())
            sums["loss_ctx"] += float(loss_ctx.detach().cpu())
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
                sums["adj_allowed_mean"] += float(
                    adj_parts["allowed_gap_rates"].mean().detach().cpu()
                )
                sums["adj_tgt_mean"] += float(
                    adj_parts["target_gap_rates"].mean().detach().cpu()
                )
            
                if adj_parts.get("confidence", None) is not None:
                    sums["adj_conf_mean"] += float(
                        adj_parts["confidence"].mean().detach().cpu()
                    )
            
            # --- per-refinement reconstruction diagnostics
            for ridx, val in enumerate(recon_level_vals, start=1):
                sums[f"loss_recon_l{ridx}"] += float(val.detach().cpu())
            
            for ridx, val in enumerate(recon_level_exact_vals, start=1):
                sums[f"loss_recon_l{ridx}_exact"] += float(val.detach().cpu())
                
            # --- weighted totals (match recon_total weighting)
            for ridx, (exact_val, tol_val, w_ref) in enumerate(
                zip(recon_level_exact_vals, recon_level_tol_vals, refinement_loss_weights_eff),
                start=1,
            ):
                sums["loss_recon_exact_total"] += float(w_ref * exact_val.detach().cpu())
                sums["loss_recon_tol_total"] += float(w_ref * tol_val.detach().cpu())
            
            for ridx, val in enumerate(recon_level_tol_vals, start=1):
                sums[f"loss_recon_l{ridx}_tol"] += float(val.detach().cpu())
                        
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

        lb = float(model.global_logit_bias.item())
        
                
        
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
            "blank": sums["loss_blank"] / max(1, num_batches),
            "blank_sep": sums["loss_blank_sep"] / max(1, num_batches),
            "loss_isi": sums["loss_isi"] / max(1, num_batches),
            
            "loss_ctx": sums["loss_ctx"] / max(1, num_batches),
            "loss_sp_cons": sums["loss_sp_cons"] / max(1, num_batches),
            "loss_sp_token": sums["loss_sp_token"] / max(1, num_batches),
            "loss_sp_pixel": sums["loss_sp_pixel"] / max(1, num_batches),
            "lambda_sp_token_eff": lambda_sp_token_eff,
            "lambda_sp_pixel_eff": lambda_sp_pixel_eff,
            
            "sp_memory_used": sums["sp_memory_used"] / max(1, num_batches),
            "adj_memory_used": sums["adj_memory_used"] / max(1, num_batches),
            "adj_pred_mean": sums["adj_pred_mean"] / max(1, num_batches),
            "adj_allowed_mean": sums["adj_allowed_mean"] / max(1, num_batches),
            "adj_tgt_mean": sums["adj_tgt_mean"] / max(1, num_batches),
            "adj_conf_mean": sums["adj_conf_mean"] / max(1, num_batches),
                        
            # vq information
            "avg_vq_perp_nb": vq_metric_sums["perplexity_nonblank"] / max(1, num_batches),
            "avg_vq_ent_nb": vq_metric_sums["entropy_nonblank"] / max(1, num_batches),
            "avg_vq_active_nb": vq_metric_sums["active_codes_nonblank"] / max(1, num_batches),
            "avg_vq_blank_frac": vq_metric_sums["blank_frac"] / max(1, num_batches),
            "avg_vq_commit_loss": vq_metric_sums["commit_loss"] / max(1, num_batches),
            "active_quantizers": int(model.vq.active_quantizers),
            "avg_vq_residual_norm": vq_metric_sums["residual_norm"] / max(1, num_batches),

            # schedules
            "pos_weight_eff": float(pos_weight_eff),
            "lambda_ctx_eff": float(lambda_ctx_eff),
            "lambda_isi": float(lambda_isi),
            "cfg_drop_prob": float(cfg_p),

            
            # activities
            "pred_mean_p": pred_mean_p_epoch,
            "tgt_mean": tgt_mean_epoch,
            "logit_bias": lb,

        }
        
        for ridx in range(1, max_ref_levels + 1):
            train_log[f"loss_recon_l{ridx}"] = sums[f"loss_recon_l{ridx}"] / max(1, num_batches)
            train_log[f"loss_recon_l{ridx}_exact"] = sums[f"loss_recon_l{ridx}_exact"] / max(1, num_batches)
            train_log[f"loss_recon_l{ridx}_tol"] = sums[f"loss_recon_l{ridx}_tol"] / max(1, num_batches)
            
        for lvl in range(1, max_ref_levels + 1):
            train_log[f"perplexity_nonblank_l{lvl}"] = vq_metric_sums.get(f"perplexity_nonblank_l{lvl}", 0.0) / max(1, num_batches)
            train_log[f"active_codes_nonblank_l{lvl}"] = vq_metric_sums.get(f"active_codes_nonblank_l{lvl}", 0.0) / max(1, num_batches)
            train_log[f"commit_loss_l{lvl}"] = vq_metric_sums.get(f"commit_loss_l{lvl}", 0.0) / max(1, num_batches)
        
        
        cb_stats = get_vq_codebook_stats(model)
        train_log.update(cb_stats)
    
            
        if on_epoch: on_epoch(train_log)

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
            )
            # train_log.update(eval_report)
            if val_metric_name in eval_report:
                val_metric = eval_report[val_metric_name]
            else:
                val_metric = -eval_report["val_loss_BCE"] if val_metric_goal == "max" else eval_report["val_loss_BCE"]
            train_log[val_metric_name] = val_metric
            
            history["val_metrics"].append(eval_report)

        history["train_log"].append(train_log)

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
            f"ctx={train_log.get('loss_ctx', 0):.5f} "
            f"sp_cons={train_log.get('loss_sp_cons', 0):.6e} "
            f"isi={train_log.get('loss_isi', 0):.5f} "
            f"sp_mem={train_log.get('sp_memory_used', 0):.2f} "
            f"adj_mem={train_log.get('adj_memory_used', 0):.2f} "
            f"adj_pred={train_log.get('adj_pred_mean', 0):.5e} "
            f"adj_allowed={train_log.get('adj_allowed_mean', 0):.5e} "
            f"adj_conf={train_log.get('adj_conf_mean', 0):.3f}\n"
            f"blank={train_log.get('blank', 0):.5f} "
            f"blank_sep={train_log.get('blank_sep', 0):.5f}\n"
            
            f"\n"
            
            f"logit_bias={train_log.get('logit_bias', 0):.5f} "
            f"pred_mean_p={train_log.get('pred_mean_p', 0):.6e}, "
            f"tgt_mean={train_log.get('tgt_mean', 0):.6e}\n"
            f"\n"
            
            f"\n"
            f"---"
            f"\n\n"
            
            f"avg_vq_perp_nb={train_log.get('avg_vq_perp_nb', 0):.2f} "
            f"avg_vq_ent_nb={train_log.get('avg_vq_ent_nb', 0):.2f} "
            f"avg_vq_blank_frac={train_log.get('avg_vq_blank_frac', 0):.3f} "
            f"avg_vq_resid={train_log.get('avg_vq_residual_norm', 0):.5f}\n"
            f"\n"
            
            f"vq_l1_ppl={train_log.get('perplexity_nonblank_l1', 0):.2f} "
            f"vq_l2_ppl={train_log.get('perplexity_nonblank_l2', 0):.2f} "
            f"(ratio={train_log.get('perplexity_nonblank_l2', 0) / (train_log.get('perplexity_nonblank_l1', 1) + 1e-6):.2f}) "
            f"vq_l1_act={train_log.get('active_codes_nonblank_l1', 0):.1f} "
            f"vq_l2_act={train_log.get('active_codes_nonblank_l2', 0):.1f}\n"
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
            if "AUPRC_cond" in eval_report:
                log_str += f"\nAUPRC_exact={eval_report['AUPRC_cond']:.6f}"
            if "BestF1_cond" in eval_report:
                log_str += f"  BestF1_exact={eval_report['BestF1_cond']:.4f}"
            if "BestF1_threshold_cond" in eval_report:
                log_str += f"  Thr_exact={eval_report['BestF1_threshold_cond']:.3f}"

            # tolerant metrics
            if "AUPRC_tol_cond" in eval_report:
                log_str += f"\nAUPRC_tol={eval_report['AUPRC_tol_cond']:.6f}"
            if "BestF1_tol_cond" in eval_report:
                log_str += f"  BestF1_tol={eval_report['BestF1_tol_cond']:.4f}"
            if "BestF1_threshold_tol_cond" in eval_report:
                log_str += f"  Thr_tol={eval_report['BestF1_threshold_tol_cond']:.3f}"

        print(log_str)
        print(" ")

        
        print("===========================================")
        print(" ")

        
        # ---- update model.best_thr every epoch from current eval_report ----
        if eval_report is not None:
            if val_metric_name == "AUPRC_uncond":
                thr_key = "BestF1_threshold_uncond"
            elif val_metric_name == "AUPRC_tol_uncond":
                thr_key = "BestF1_threshold_tol_uncond"
            elif val_metric_name == "AUPRC_tol_cond":
                thr_key = "BestF1_threshold_tol_cond"
            else:
                thr_key = "BestF1_threshold"

            if thr_key in eval_report:
                best_thr = float(eval_report[thr_key])
                model._set_best_thr(best_thr)

        improved = False
        checkpointing_active = (epoch >= save_start_epoch)
        
        if val_loader is not None and val_metric is not None:
            if checkpointing_active:
                if ((val_metric_goal == "max" and val_metric > best_val) or (val_metric_goal == "min" and val_metric < best_val)):
                    improved = True
                    best_val = val_metric
                    best_epoch = epoch
                    best_thr_for_best_model = float(getattr(model, "best_thr", 0.5))
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
                f"(val_{val_metric_name} = {val_metric:.5f}, best_thr = {best_thr_for_best_model:.4f})"
            )

        if early_stop_patience is not None and no_improve >= early_stop_patience:
            print(f"⏹️  Early stopping triggered. No improvement for {early_stop_patience} consecutive epochs.")
            break

    return {
        "best_epoch": best_epoch,
        "best_val": best_val,
        "best_thr": best_thr_for_best_model,
        "history": history,
    }