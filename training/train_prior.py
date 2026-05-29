#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:14:37 2026

@author: derik
"""
import os
import math
import torch



@torch.no_grad()
def iterative_unmask_hierarchical(
    prior,
    global_ctx,
    local_ctx,
    task_id,
    N,
    steps=12,
    temperature=1.0,
):
    B = global_ctx.shape[0]

    a = torch.full((B, N), prior.a_mask_id, device=global_ctx.device, dtype=torch.long)
    z1 = torch.full((B, N), prior.z1_mask_id, device=global_ctx.device, dtype=torch.long)
    z2 = torch.full((B, N), prior.z2_mask_id, device=global_ctx.device, dtype=torch.long)

    still_masked = torch.ones((B, N), device=global_ctx.device, dtype=torch.bool)

    for s in range(steps):
        logits, _, _ = prior(
            a, z1, z2,
            global_ctx=global_ctx,
            local_ctx=local_ctx,
            task_id=task_id,
            targets=None,
        )

        pa = torch.softmax(logits["a"] / temperature, dim=-1)
        pz1 = torch.softmax(logits["z1"] / temperature, dim=-1)
        pz2 = torch.softmax(logits["z2"] / temperature, dim=-1)

        a_samp = torch.multinomial(pa.reshape(-1, 2), 1).view(B, N)
        z1_samp = torch.multinomial(pz1.reshape(-1, prior.K1), 1).view(B, N)
        z2_samp = torch.multinomial(pz2.reshape(-1, prior.K2), 1).view(B, N)

        conf_a = pa.gather(-1, a_samp.unsqueeze(-1)).squeeze(-1)
        conf_z1 = pz1.gather(-1, z1_samp.unsqueeze(-1)).squeeze(-1)
        conf_z2 = pz2.gather(-1, z2_samp.unsqueeze(-1)).squeeze(-1)

        active = a_samp.eq(prior.a_active_id)

        # blank confidence only uses activity confidence
        # active confidence uses joint confidence
        conf = torch.where(
            active,
            conf_a * conf_z1 * conf_z2,
            conf_a,
        )

        conf = conf.masked_fill(~still_masked, -1.0)

        num_left = still_masked.sum(dim=1)
        num_keep_masked = torch.ceil(
            num_left.float() * (1.0 - (s + 1) / steps)
        ).long()

        for b in range(B):
            n_unmask = int(num_left[b] - num_keep_masked[b])
            if n_unmask <= 0:
                continue

            idx = torch.topk(conf[b], k=n_unmask).indices

            a[b, idx] = a_samp[b, idx]

            active_idx = a_samp[b, idx].eq(prior.a_active_id)

            z1[b, idx] = prior.z1_null_id
            z2[b, idx] = prior.z2_null_id

            if active_idx.any():
                active_pos = idx[active_idx]
                z1[b, active_pos] = z1_samp[b, active_pos]
                z2[b, active_pos] = z2_samp[b, active_pos]

            still_masked[b, idx] = False

    # Convert final token triplet back to VQVAE code format.
    codes = torch.full((B, N, 2), -1, device=global_ctx.device, dtype=torch.long)
    active = a.eq(prior.a_active_id)
    codes[..., 0][active] = z1[active]
    codes[..., 1][active] = z2[active]

    return codes



def train_prior_mgit(
    prior,                 # HierarchicalTokenMGITTransformer (bidirectional MaskGIT prior)
    vqvae,                 # your VQVAE tokenizer (frozen inside)
    opt,
    train_loader,
    val_loader=None,
    epochs: int = 20,
    grad_clip: float = 1.0,
    ckpt_out: str = "ckpts/prior_mgit_best.pt",
    early_stop_patience: int = 5,
    min_delta: float = 0.0,
    use_amp: bool = True,
    # --- MaskGIT knobs ---
    recon_task_id: int = 0,           # your mapping: recon=0
    ensure_at_least_one_mask: bool = True,
    # --- logging ---
    log_every: int = 50,
    activity_pos_weight: float = 20.0,
):
    """
    MaskGIT-style (masked token modeling) prior training.

    Assumptions about batch dict (from your dataset/collate):
      batch["x"]             : (B,1,T,H,W)
      batch["global_ctx_ids"]: (B,2) where [:,0]=assay_id, [:,1]=task_id
      batch.get("global_ctx"): optional (B,G) float (passed to vqvae, safe if your vqvae expects it)
      batch.get("local_ctx") : optional (B,L) float (passed to vqvae)
      batch.get("mask_spec") : optional (list of dicts) (passed to vqvae forward as predict_mask_spec)

    Assumptions about vqvae output:
      out["codes"]           : (B,N) long
      out["predict_mask"]    : (B,N,1) float, 1 = "predict/supervise this token"
                               NOTE: for your current spec, recon usually returns all-ones; we override it here.

    Assumptions about prior (TokenMGITTransformer):
      prior.mask_id          : int, the in-vocab MASK token id
      prior(input_ids, targets, global_ctx, local_ctx, task_id) returns (logits, loss)
        - input_ids : (B,N) with masked positions set to mask_id
        - targets   : (B,N) with -100 for ignore positions
        - loss is CE over masked positions only

    Checkpoint metric:
      - validation perplexity over masked positions (exp(avg_nll)).
    """
    device = next(prior.parameters()).device
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)


    # --- freeze vqvae ---
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)

    os.makedirs(os.path.dirname(ckpt_out) or ".", exist_ok=True)

    best_val = float("inf")
    patience = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_loss_a": [],
        "train_loss_z1": [],
        "train_loss_z2": [],
        "val_loss_a": [],
        "val_loss_z1": [],
        "val_loss_z2": [],
    }


    def _get_batch(batch):
        x = batch["x"].to(device, non_blocking=True)
    
        gct = batch.get("global_ctx", None)
        lct = batch.get("local_ctx", None)
        if isinstance(gct, torch.Tensor):
            gct = gct.to(device, non_blocking=True)
        if isinstance(lct, torch.Tensor):
            lct = lct.to(device, non_blocking=True)
    
        task_id = batch.get("task_id", None)
        if task_id is None:
            raise KeyError('Batch missing "task_id" (required for FiLM-conditioned prior).')
        task_id = task_id.to(device, non_blocking=True).long()
    
        mask_spec = batch.get("mask_spec", None)
        return x, gct, lct, task_id, mask_spec


    @torch.no_grad()
    def _vq_codes_and_pmask(x, gct, lct, mask_spec):
        # prefer new signature if present
        try:
            out = vqvae(x, global_ctx=gct, local_ctx=lct, predict_mask_spec=mask_spec)
        except TypeError:
            out = vqvae(x, global_ctx=gct, local_ctx=lct)

        codes = out["codes"].long()  # expected (B,N,2): z1,z2; blank usually -1
        pmask = out.get("predict_mask", None)  # (B,N,1) float

        if pmask is None:
            B, N = codes.shape[:2]
        
            mask_ratio = torch.empty((B, 1), device=device).uniform_(0.3, 0.95)
        
            pmask = (
                torch.rand((B, N), device=device) < mask_ratio
            ).float().unsqueeze(-1)

        if pmask.dim() == 2:
            pmask = pmask.unsqueeze(-1)
        pmask = pmask.float().squeeze(-1)  # (B,N)
        return codes, pmask

    def _override_recon_mask(pmask, task_id):
        """
        For MaskGIT, recon batches should NOT mean "mask all"; they should mean "random mask ratio".
        Also ensure each sample has at least one masked token so loss isn't empty.
        """
        B, N = pmask.shape

        # Override recon rows
        is_recon = (task_id == recon_task_id)
        if is_recon.any():
            
            mask_ratio = torch.empty(
                (int(is_recon.sum().item()), 1),
                device=device,
            ).uniform_(0.3, 0.95)
            
            rnd = (
                torch.rand((int(is_recon.sum().item()), N), device=device)
                < mask_ratio
            ).float()
            
            if ensure_at_least_one_mask:
                rnd[:, 0] = 1.0
            pmask[is_recon] = rnd

        if ensure_at_least_one_mask:
            none_masked = (pmask.sum(dim=1) < 0.5)
            if none_masked.any():
                pmask[none_masked, 0] = 1.0

        return pmask

    def _make_hierarchical_mgit_io(prior, codes, pmask):
        targets = prior.make_targets_from_codes(
            codes=codes,
            predict_mask=pmask,
            blank_code=getattr(vqvae.vq, "blank_code", -1),
        )
        a_in, z1_in, z2_in, targets = prior.corrupt_inputs_from_targets(
            targets,
            mode_probs=(0.50, 0.25, 0.25),
        )
        return a_in, z1_in, z2_in, targets

    def _run_epoch(loader, train: bool):
        prior.train(train)
        total_loss = 0.0
        total_cnt = 0.0

        total_a = 0.0
        total_a_cnt = 0.0

        total_z1 = 0.0
        total_z2 = 0.0
        total_z1_cnt = 0.0
        total_z2_cnt = 0.0

        for it, batch in enumerate(loader, start=1):
            x, gct, lct, task_id, mask_spec = _get_batch(batch)
            with torch.no_grad():
                codes, pmask = _vq_codes_and_pmask(x, gct, lct, mask_spec)
                pmask = _override_recon_mask(pmask, task_id)
                a_in, z1_in, z2_in, targets = _make_hierarchical_mgit_io(prior, codes, pmask)

            if train:
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    _, loss, aux = prior(
                        a_in,
                        z1_in,
                        z2_in,
                        global_ctx=gct,
                        local_ctx=lct,
                        task_id=task_id,
                        targets=targets,
                        loss_weights=(1.0, 1.0, 1.0),
                        activity_pos_weight=activity_pos_weight,
                    )
                scaler.scale(loss).backward()

                if grad_clip is not None and grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(prior.parameters(), float(grad_clip))

                scaler.step(opt)
                scaler.update()
            else:
                with torch.no_grad():
                    _, loss, aux = prior(
                        a_in,
                        z1_in,
                        z2_in,
                        global_ctx=gct,
                        local_ctx=lct,
                        task_id=task_id,
                        targets=targets,
                        loss_weights=(1.0, 1.0, 1.0),
                        activity_pos_weight=activity_pos_weight,
                    )
            # accumulate NLL weighted by how many masked tokens contributed
            n_a = float(targets["a_loss_mask"].sum().item())
            n_z1 = float(targets.get("z1_loss_mask", targets["z_loss_mask"]).sum().item())
            n_z2 = float(targets.get("z2_loss_mask", targets["z_loss_mask"]).sum().item())

            if n_a < 1.0:
                continue

            total_loss += float(loss.item()) * n_a
            total_cnt += n_a

            total_a += float(aux["loss_a"].item()) * n_a
            total_a_cnt += n_a

            if n_z1 > 0:
                total_z1 += float(aux["loss_z1"].item()) * n_z1
                total_z1_cnt += n_z1
            
            if n_z2 > 0:
                total_z2 += float(aux["loss_z2"].item()) * n_z2
                total_z2_cnt += n_z2
            


            if train and log_every and (it % log_every == 0):
                avg = total_loss / max(total_cnt, 1.0)
                print(f"  it {it:05d}: loss={avg:.4f}")

        den = max(total_cnt, 1.0)
        den_a = max(total_a_cnt, 1.0)
        den_z1 = max(total_z1_cnt, 1.0)
        den_z2 = max(total_z2_cnt, 1.0)



        return {
            "loss": total_loss / den,
            "loss_a": total_a / den_a,
            "loss_z1": total_z1 / den_z1,
            "loss_z2": total_z2 / den_z2,
        }

    for ep in range(1, epochs + 1):
        train_m = _run_epoch(train_loader, train=True)
        val_m = _run_epoch(val_loader, train=False) if val_loader is not None else train_m

        for k in ("loss", "loss_a", "loss_z1", "loss_z2"):
            history[f"train_{k}"].append(train_m[k])
            history[f"val_{k}"].append(val_m[k])

        print(
            f"[epoch {ep:03d}] "
            f"train loss={train_m['loss']:.4f} "
            f"val loss={val_m['loss']:.4f} "
            f"val a={val_m['loss_a']:.4f} "
            f"z1={val_m['loss_z1']:.4f} "
            f"z2={val_m['loss_z2']:.4f}"
        )

        # checkpoint on best val loss (lower is better)
        if val_m["loss"] < best_val - float(min_delta):
            best_val = val_m["loss"]
            patience = 0
            torch.save(
                {"model": prior.state_dict(), "epoch": ep, "best_val_loss": best_val},
                ckpt_out,
            )
            print(f"  saved {ckpt_out}  (best val loss {best_val:.2f})")
        else:
            patience += 1
            if patience >= int(early_stop_patience):
                print(f"Early stopping at epoch {ep} (best val loss {best_val:.2f})")
                break

    return history


