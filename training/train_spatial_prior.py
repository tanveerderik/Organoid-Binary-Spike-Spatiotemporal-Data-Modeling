#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:16:09 2026

@author: derik
"""

import time
from typing import Optional

import torch
import torch.nn as nn

from ..utils.recon import (
    spatial_token_map_from_input,
    lift_crop_token_map_to_full,
    spatial_pixel_map_from_input,
    lift_crop_pixel_map_to_full,
)

from ..utils.losses import (
    spatial_support_separation_loss,
)

from ..model.spatial_map import GlobalContextSpatialBank, GlobalContextAdjacencyBank


def support_map_loss(pred, tgt, outside_w=3.0, mass_w=2.0, eps=1e-6):
    pred = pred.clamp(eps, 1.0 - eps)
    tgt = tgt.float()

    pos_count = tgt.sum().clamp_min(1.0)
    neg_count = (1.0 - tgt).sum().clamp_min(1.0)

    loss_cover = (-(pred.log()) * tgt).sum() / pos_count
    loss_outside = (pred * (1.0 - tgt)).sum() / neg_count
    loss_mass = (pred.mean() - tgt.mean()).pow(2)

    loss = loss_cover + outside_w * loss_outside + mass_w * loss_mass
    return loss, loss_cover, loss_outside, loss_mass

def fit_spatial_prior_pretrain(
    model: nn.Module,
    train_loader,
    spatial_ckpt_path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    epochs: int = 10,
    device: Optional[torch.device] = None,
    memory_momentum: float = 0.95,
    memory_mode: str = "max",   # "max" or "ema"
    lambda_sep: float = 1e-4,
    adj_max_gap: int = 3,
    lambda_adj: float = 0.5,
    early_stop_patience: Optional[int] = None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if optimizer is None:
        raise ValueError("optimizer must not be None")

    if model.spatial_map_prior is None:
        raise ValueError("model.spatial_map_prior is None")

    memory_tok = GlobalContextSpatialBank(
        momentum=memory_momentum,
        union_mode=memory_mode,
        round_decimals=6,
    )
    
    memory_pix = GlobalContextSpatialBank(
        momentum=memory_momentum,
        union_mode=memory_mode,
        round_decimals=6,
    )
    
    max_gap = adj_max_gap
    memory_adj = GlobalContextAdjacencyBank(
        max_gap=max_gap,
        alpha=1.0,
        beta=20.0,
        round_decimals=6,
    )


    loss_best = 99999999
    no_improve = 0
    history = {"train_log": []}
    
    sep_warmup_epochs = 15

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()

        total_loss = 0.0
        
        total_loss_cover = 0.0
        total_loss_outside = 0.0
        total_loss_mass = 0.0

        
        total_sep = 0.0
        total_sim = 0.0
        total_pred_mean = 0.0
        total_pred_min = 0.0
        total_pred_max = 0.0
        total_pos_cov = 0.0
        total_mem_mean = 0.0
        total_adj_loss = 0.0
        total_adj_pred = 0.0
        total_adj_tgt = 0.0
        total_adj_den = 0.0
        total_pairs = 0
        num_batches = 0
        
        sep_scale = min(1.0, float(epoch) / float(sep_warmup_epochs))


        for batch in train_loader:
            num_batches += 1

            x = batch["x"].to(device, non_blocking=True)
            gct = batch["global_ctx"].to(device, non_blocking=True)
            roi_hw = batch.get("roi_hw", None)
            pad_hw = batch.get("pad_hw", None)
            
            # Current sample -> token support map in crop coordinates
            # ---- Tokenwise target: coarse support ----
            crop_tok = spatial_token_map_from_input(x, model.patch_size)
            
            full_tok_target = lift_crop_token_map_to_full(
                crop_tok_map_bhw=crop_tok,
                full_hw=model.full_spatial_size,
                patch_size=model.patch_size,
                roi_hw=roi_hw,
                pad_hw=pad_hw,
            )
            
            # ---- Pixelwise target: fine support ----
            crop_pix = spatial_pixel_map_from_input(x)
            
            full_pix_target = lift_crop_pixel_map_to_full(
                crop_pix_map_bhw=crop_pix,
                full_hw=model.full_spatial_size,
                roi_hw=roi_hw,
                pad_hw=pad_hw,
            )

            # Update per-context memory bank from observed data            
            memory_tok.update(gct=gct, target_full_bhw=full_tok_target)
            memory_pix.update(gct=gct, target_full_bhw=full_pix_target)
            
            memory_adj.update_from_x(gct=gct, x=x)
            
            mem_tok_target = memory_tok.get(gct=gct, device=device, dtype=x.dtype)
            mem_pix_target = memory_pix.get(gct=gct, device=device, dtype=x.dtype)

            # Retrieve accumulated support target for current context
            # Global-context embedding
            g_emb = model._global_emb_only(gct)

            sp = model.spatial_map_prior(
                g_emb,
                grid=model.token_grid,
                roi_hw=roi_hw,
                pad_hw=pad_hw,
            )

            pred_tok = sp["token_support"]       # (B, full_h_tok, full_w_tok)
            pred_pix = sp["full_hw_support"]     # (B, full_H, full_W)
            adj_logits = sp["adjacency_logits"]  # (B,G)
            adj_probs = sp["adjacency_probs"]    # (B,G), diagnostics only

            loss_tok, loss_tok_cover, loss_tok_outside, loss_tok_mass = support_map_loss(
                pred_tok,
                mem_tok_target,
                outside_w=2.0,
                mass_w=1.0,
            )

            loss_pix, loss_pix_cover, loss_pix_outside, loss_pix_mass = support_map_loss(
                pred_pix,
                mem_pix_target,
                outside_w=3.0,
                mass_w=2.0,
            )

            pix_scale = min(1.0, float(epoch) / 10.0)

            loss_sep, sim_mean, num_pairs = spatial_support_separation_loss(
                pred_support_map_bhw=pred_pix,
                target_support_map_bhw=mem_pix_target,
                ctx_key=gct,
                mode="margin",
                margin=0.25,
            )


            tok_scale = max(0.15, 1.0 - float(epoch) / 40.0)
            pix_scale = min(1.0, float(epoch) / 10.0)
            

            adj_target = memory_adj.get(gct=gct, device=device, dtype=x.dtype).clamp(0.0, 1.0)

            loss_adj = nn.functional.binary_cross_entropy_with_logits(
                adj_logits,
                adj_target,
            )
            
            
            loss = tok_scale * loss_tok + pix_scale * loss_pix + (lambda_sep * sep_scale) * loss_sep
            loss = loss + float(lambda_adj) * loss_adj


            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Logging stats
            with torch.no_grad():
                pred_mean = pred_pix.mean()
                pred_min = pred_pix.min()
                pred_max = pred_pix.max()
                mem_mean = mem_pix_target.mean()
                adj_den = memory_adj.get_den(gct=gct, device=device, dtype=x.dtype)

                pos_mask = mem_pix_target > 0
                if pos_mask.any():
                    pos_coverage = pred_pix[pos_mask].mean()
                else:
                    pos_coverage = pred_pix.new_tensor(0.0)

            total_loss += float(loss.detach().cpu())
            
            total_loss_cover += float((loss_tok_cover + pix_scale * loss_pix_cover).detach().cpu())
            total_loss_outside += float((loss_tok_outside + pix_scale * loss_pix_outside).detach().cpu())
            total_loss_mass += float((loss_tok_mass + pix_scale * loss_pix_mass).detach().cpu())
            total_sep += float(loss_sep.detach().cpu())
            
            total_adj_loss += float(loss_adj.detach().cpu())
            total_adj_pred += float(adj_probs.mean().detach().cpu())
            total_adj_tgt += float(adj_target.mean().detach().cpu())
            total_adj_den += float(adj_den.mean().detach().cpu())
            
            total_sim += float(sim_mean.detach().cpu())
            total_pred_mean += float(pred_mean.detach().cpu())
            total_pred_min += float(pred_min.detach().cpu())
            total_pred_max += float(pred_max.detach().cpu())
            total_pos_cov += float(pos_coverage.detach().cpu())
            total_mem_mean += float(mem_mean.detach().cpu())
            total_pairs += int(num_pairs)

        if scheduler is not None:
            scheduler.step()

        denom = max(1, num_batches)

        train_log = {
            "epoch": epoch,
            "loss": total_loss / denom,
            
            "loss_cover": total_loss_cover / denom,
            "loss_outside": total_loss_outside / denom,
            "loss_mass": total_loss_mass / denom,
            "loss_sep": total_sep / denom,
            
            "loss_adj": total_adj_loss / denom,
            "adj_pred_mean": total_adj_pred / denom,
            "adj_tgt_mean": total_adj_tgt / denom,
            "adj_den_mean": total_adj_den / denom,
            
            "sim_mean": total_sim / denom,
            "pred_mean": total_pred_mean / denom,
            "pred_min": total_pred_min / denom,
            "pred_max": total_pred_max / denom,
            "pos_coverage": total_pos_cov / denom,
            "mem_mean": total_mem_mean / denom,
            "pairs": total_pairs / denom,
            "time_sec": time.time() - t0,
            "lr": optimizer.param_groups[0]["lr"],
        }

        history["train_log"].append(train_log)

        print(
            f"[Epoch {epoch}] "
            f"loss={train_log['loss']:.5f} "
            
            f"cover={train_log['loss_cover']:.5f} "
            f"outside={train_log['loss_outside']:.5f} "
            f"mass={train_log['loss_mass']:.5f} "

            f"sep={train_log['loss_sep']:.5f} "
            
            f"sim={train_log['sim_mean']:.4f} "
            f"pred_mean={train_log['pred_mean']:.5f} "
            f"pred_min={train_log['pred_min']:.5f} "
            f"pred_max={train_log['pred_max']:.5f} "
            f"pos_cov={train_log['pos_coverage']:.5f} "
            f"mem_mean={train_log['mem_mean']:.5f} "
            f"pairs={train_log['pairs']} "
            
            f"adj={train_log['loss_adj']:.5f} "
            f"adj_pred={train_log['adj_pred_mean']:.5e} "
            f"adj_tgt={train_log['adj_tgt_mean']:.5e} "
            f"adj_den={train_log['adj_den_mean']:.1f} "

            f"lr={train_log['lr']:.6g}"
        )
        
        if total_loss / denom < loss_best:
            print(f"Loss improved from {loss_best:.5f} to {(total_loss / denom):.5f}. Saving map to {spatial_ckpt_path}.\n")
            loss_best = total_loss / denom
            no_improve = 0
            torch.save({
                "global_embedder": model.global_embedder.state_dict(),
                "spatial_map_prior": model.spatial_map_prior.state_dict(),
                "memory_tok": memory_tok.state_dict(),
                "memory_pix": memory_pix.state_dict(),
                "memory_adj": memory_adj.state_dict()   # NEW
            }, spatial_ckpt_path)
        else:
            no_improve += 1
            
        if early_stop_patience is not None and no_improve >= early_stop_patience:
            print(f"⏹️  Early stopping triggered. No improvement for {early_stop_patience} consecutive epochs.\n")
            break

    return history