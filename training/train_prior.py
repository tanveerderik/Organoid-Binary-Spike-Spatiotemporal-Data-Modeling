#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:14:37 2026

@author: derik
"""
import os
import math
import torch



def train_prior_mgit(
    prior,                 # TokenMGITTransformer (bidirectional MaskGIT prior)
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
    recon_mask_ratio: float = 0.60,   # for recon batches: random masking ratio (MaskGIT needs some visible tokens)
    ensure_at_least_one_mask: bool = True,
    # --- logging ---
    log_every: int = 50,
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

    if not hasattr(prior, "mask_id"):
        raise ValueError("prior must have attribute mask_id (MASK token id).")
    mask_id = int(prior.mask_id)

    os.makedirs(os.path.dirname(ckpt_out) or ".", exist_ok=True)

    best_val = float("inf")
    patience = 0
    history = {"train_ppl": [], "val_ppl": []}


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

        codes = out["codes"].long()  # (B,N)
        pmask = out.get("predict_mask", None)  # (B,N,1) float

        if pmask is None:
            # If vqvae didn't return it, default to random masking for everyone (valid for MGIT)
            B, N = codes.shape
            pmask = (torch.rand((B, N), device=device) < recon_mask_ratio).float().unsqueeze(-1)

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
            rnd = (torch.rand((int(is_recon.sum().item()), N), device=device) < float(recon_mask_ratio)).float()
            if ensure_at_least_one_mask:
                rnd[:, 0] = 1.0
            pmask[is_recon] = rnd

        if ensure_at_least_one_mask:
            none_masked = (pmask.sum(dim=1) < 0.5)
            if none_masked.any():
                pmask[none_masked, 0] = 1.0

        return pmask

    def _make_mgit_io(codes, pmask):
        """
        pmask: (B,N) float/bool, 1 = mask/predict
        """
        inp = codes.clone()
        inp[pmask > 0.5] = mask_id

        tgt = codes.clone()
        tgt[pmask <= 0.5] = -100  # ignore unmasked positions in CE
        return inp, tgt

    def _run_epoch(loader, train: bool):
        prior.train(train)
        total_nll = 0.0
        total_cnt = 0.0

        for it, batch in enumerate(loader, start=1):
            x, gct, lct, task_id, mask_spec = _get_batch(batch)
            with torch.no_grad():
                codes, pmask = _vq_codes_and_pmask(x, gct, lct, mask_spec)
                pmask = _override_recon_mask(pmask, task_id)
                inp, tgt = _make_mgit_io(codes, pmask)

            if train:
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    _, loss = prior(inp, targets=tgt, global_ctx=gct, local_ctx=lct, task_id=task_id)
                scaler.scale(loss).backward()

                if grad_clip is not None and grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(prior.parameters(), float(grad_clip))

                scaler.step(opt)
                scaler.update()
            else:
                with torch.no_grad():
                    _, loss = prior(inp, targets=tgt, global_ctx=gct, local_ctx=lct, task_id=task_id)
            # accumulate NLL weighted by how many masked tokens contributed
            n_mask = float((tgt != -100).sum().item())
            if n_mask < 1.0:
                # should not happen if ensure_at_least_one_mask True, but guard anyway
                continue

            total_nll += float(loss.item()) * n_mask
            total_cnt += n_mask

            if train and log_every and (it % log_every == 0):
                avg_nll = total_nll / max(total_cnt, 1.0)
                ppl = math.exp(avg_nll)
                print(f"  it {it:05d}: ppl={ppl:.2f}")

        avg_nll = total_nll / max(total_cnt, 1.0)
        ppl = math.exp(avg_nll)
        return ppl

    for ep in range(1, epochs + 1):
        train_ppl = _run_epoch(train_loader, train=True)
        val_ppl = _run_epoch(val_loader, train=False) if val_loader is not None else train_ppl

        history["train_ppl"].append(train_ppl)
        history["val_ppl"].append(val_ppl)

        print(f"[epoch {ep:03d}] train ppl={train_ppl:.2f}  val ppl={val_ppl:.2f}")

        # checkpoint on best val ppl (lower is better)
        if val_ppl < best_val - float(min_delta):
            best_val = val_ppl
            patience = 0
            torch.save(
                {"model": prior.state_dict(), "epoch": ep, "best_val_ppl": best_val},
                ckpt_out,
            )
            print(f"  saved {ckpt_out}  (best val ppl {best_val:.2f})")
        else:
            patience += 1
            if patience >= int(early_stop_patience):
                print(f"Early stopping at epoch {ep} (best val ppl {best_val:.2f})")
                break

    # unfreeze vqvae for later stages if you want
    for p in vqvae.parameters():
        p.requires_grad_(True)

    return history


