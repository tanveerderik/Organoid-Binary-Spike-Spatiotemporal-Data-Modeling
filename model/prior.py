#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:27:45 2026

@author: derik
"""

#%%
# ---- Drop-in new prior for MaskGIT-style training/sampling ----
# Save this in model.py (or prior.py) and import it where you build the prior.

from typing import Optional
import torch.nn.functional as F
import math
import torch
import torch.nn as nn


class TokenMGITTransformer(nn.Module):
    """
    MaskGIT-style (bidirectional) transformer prior over VQ code indices,
    conditioned by PREFIX TOKENS:
      [gct_prefix, lct_prefix, task_prefix, code_tokens...]

    - gct_embed (B,A) -> frozen gct_mapper -> (B,G) -> Linear -> (B,d_model) -> prefix token
    - task_id (B,) -> nn.Embedding -> (B,d_model) -> prefix token
    - input_ids are only code tokens (B,N) where masked positions use mask_id
    - returns logits over ONLY the code-token positions (B,N,V)

    Compatible with your existing train_prior_mgit(...) which calls:
        _, loss = prior(inp, targets=tgt, global_ctx=gct, local_ctx=lct, task_id=task_id)
    """

    def __init__(
        self,
        vocab_size: int,                 # typically num_codes + 1 (extra is MASK)
        mask_id: int,                    # typically num_codes
        num_tasks: int,

        gct_mapper: nn.Module,         # frozen copy of vqvae.global_embedder
        gct_dim: Optional[int] = None,   # not used directly; here for clarity
        gct_latent_dim: int = 16,              # output dim of gct_mapper
        
        lct_mapper: nn.Module = None,         # frozen copy of vqvae.local_embedder
        lct_dim: Optional[int] = None,   # not used directly; here for clarity
        lct_latent_dim: int = 16,              # output dim of gct_mapper

        d_model: int = 512,
        n_layer: int = 8,
        n_head: int = 8,
        max_len: int = 4096,             # must be >= N tokens (t'*h'*w')
        dropout: float = 0.1,
        pad_id: Optional[int] = None,    # usually None for your MGIT
    ):
        super().__init__()

        self.vocab_size = int(vocab_size)
        self.mask_id = int(mask_id)
        self.pad_id = pad_id

        self.d_model = int(d_model)
        self.max_len = int(max_len)

        # ---- frozen assay mapper ----
        self.gct_mapper = gct_mapper
        for p in self.gct_mapper.parameters():
            p.requires_grad = False
        self.gct_mapper.eval()
        
        self.lct_mapper = lct_mapper
        for p in self.lct_mapper.parameters():
            p.requires_grad = False
        self.lct_mapper.eval()
        

        # ---- prefix tokens ----
        self.ctx_len = 3  # task + global + local

        self.task_emb = nn.Embedding(int(num_tasks), self.d_model)
        self.gct_proj = nn.Linear(int(gct_latent_dim), self.d_model)
        self.lct_proj = nn.Linear(int(lct_latent_dim), self.d_model)

        self.ctx_drop = nn.Dropout(dropout)

        # ---- token + pos ----
        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)

        # positions include prefix too
        self.pos_emb = nn.Embedding(self.max_len + self.ctx_len, self.d_model)

        # ---- transformer (bidirectional) ----
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_head),
            dim_feedforward=int(4 * self.d_model),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(enc_layer, num_layers=int(n_layer))
        self.ln_f = nn.LayerNorm(self.d_model)

        self.head = nn.Linear(self.d_model, self.vocab_size)

    def _build_prefix(self, global_ctx: torch.Tensor, local_ctx: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        """
        Returns prefix tokens: (B,3,d_model)
        """
        if global_ctx is None:
            raise ValueError("global_ctx is required for prefix conditioning.")
        if local_ctx is None:
            raise ValueError("local_ctx is required for prefix conditioning.")
        if task_id is None:
            raise ValueError("task_id is required for prefix conditioning.")

        # gct_mapper: (B,A) float -> mapper -> (B,G) float, same for lct_mapper
        with torch.no_grad():
            g_param = next(self.gct_mapper.parameters())
            l_param = next(self.lct_mapper.parameters())
        
            g_in = global_ctx.to(device=g_param.device, dtype=g_param.dtype)
            l_in = local_ctx.to(device=l_param.device, dtype=l_param.dtype)
        
            a = self.gct_mapper(g_in)
            b = self.lct_mapper(l_in)
        
        a = a.to(dtype=self.gct_proj.weight.dtype, device=self.gct_proj.weight.device)
        b = b.to(dtype=self.lct_proj.weight.dtype, device=self.lct_proj.weight.device)
        
        # with torch.no_grad():
        #     a = self.gct_mapper(global_ctx)
        #     b = self.lct_mapper(local_ctx)

        # project to token space
        gct_tok = self.gct_proj(a).unsqueeze(1)          # (B,1,D)
        lct_tok = self.lct_proj(b).unsqueeze(1)          # (B,1,D)
        task_tok = self.task_emb(task_id.long()).unsqueeze(1)  # (B,1,D)

        prefix = torch.cat([gct_tok, lct_tok, task_tok], dim=1)     # (B,3,D)
        return self.ctx_drop(prefix)

    def forward(
        self,
        input_ids: torch.LongTensor,                 # (B,N)
        *,
        targets: Optional[torch.LongTensor] = None,  # (B,N), with -100 to ignore
        global_ctx: Optional[torch.Tensor] = None,  # (B,A) float
        local_ctx: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,      # (B,) long
    ):
        B, N = input_ids.shape
        if N > self.max_len:
            raise ValueError(f"Sequence length N={N} exceeds max_len={self.max_len}.")

        prefix = self._build_prefix(global_ctx, local_ctx, task_id)    # (B,3,D)
        x_tok = self.tok_emb(input_ids)                      # (B,N,D)

        x = torch.cat([prefix, x_tok], dim=1)                # (B,3+N,D)

        # positions 0..(2+N-1)
        pos = torch.arange(self.ctx_len + N, device=x.device)
        x = x + self.pos_emb(pos).unsqueeze(0)

        # optional padding mask (rare for your case)
        key_padding_mask = None
        if self.pad_id is not None:
            # only applies to code tokens; prefix is never pad
            pad = (input_ids == self.pad_id)                 # (B,N)
            prefix_pad = torch.zeros((B, self.ctx_len), device=pad.device, dtype=torch.bool)
            key_padding_mask = torch.cat([prefix_pad, pad], dim=1)  # (B,2+N)

        h = self.blocks(x, src_key_padding_mask=key_padding_mask)
        h = self.ln_f(h)
        logits_all = self.head(h)                             # (B,2+N,V)

        # Return logits aligned to code tokens only
        logits = logits_all[:, self.ctx_len:, :]              # (B,N,V)


        loss = None
        if targets is not None:
            if targets.shape != (B, N):
                raise ValueError(f"targets must be (B,N)={(B,N)}, got {tuple(targets.shape)}")
        
            logits_for_loss = logits.clone()
            logits_for_loss[..., self.mask_id] = torch.finfo(logits_for_loss.dtype).min
        
            loss = F.cross_entropy(
                logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def sample_maskgit(
        self,
        init_tokens: torch.LongTensor,              # (B,N) with mask_id in unknown positions
        global_ctx: torch.Tensor,                   # (B,A)
        local_ctx: torch.Tensor,                    # (B,A)
        task_id: torch.LongTensor,                  # (B,)
        *,
        fixed_mask: Optional[torch.Tensor] = None,  # (B,N) True = never change
        steps: int = 12,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        remask_only_originally_masked: bool = True,
    ) -> torch.LongTensor:
        """
        Iterative MaskGIT refinement sampling.
    
        init_tokens:
            Unknown positions should be set to self.mask_id.
    
        fixed_mask:
            True means the token is fixed and will never be changed.
            If None, fixed_mask = (init_tokens != self.mask_id).
    
        Returns:
            (B,N) token IDs in [0, ..., vocab_size-1], with no mask_id remaining.
        """
        self.eval()
        device = init_tokens.device
        B, N = init_tokens.shape
    
        if fixed_mask is None:
            fixed_mask = (init_tokens != self.mask_id)
        fixed_mask = fixed_mask.to(device=device, dtype=torch.bool)
    
        changeable = ~fixed_mask
        if changeable.sum().item() == 0:
            return init_tokens.clone()
    
        tokens = init_tokens.clone()
        originally_changeable = changeable.clone()
        changeable_count = changeable.sum(dim=1)  # (B,)
    
        def remaining_to_mask(step_idx: int) -> torch.LongTensor:
            # Cosine schedule: number of changeable tokens to re-mask after this step
            t = (step_idx + 1) / max(1, int(steps))
            r = math.cos((math.pi / 2.0) * t)
            rem = torch.ceil(changeable_count.float() * r).to(torch.long)
            return rem
    
        def _prepare_logits(logits: torch.Tensor) -> torch.Tensor:
            if not torch.isfinite(logits).all():
                raise RuntimeError("[sample_maskgit] non-finite logits encountered")
    
            if temperature is not None and float(temperature) != 1.0:
                logits = logits / max(1e-6, float(temperature))
    
            if top_k is not None and int(top_k) > 0:
                k = min(int(top_k), logits.size(-1))
                v, idx = torch.topk(logits, k=k, dim=-1)
                filt = torch.full_like(logits, float("-inf"))
                logits = filt.scatter(-1, idx, v)
    
            # Never sample the mask token itself
            logits = logits.clone()
            logits[..., self.mask_id] = float("-inf")
            return logits
    
        def _sample_into_masked_positions(tokens: torch.Tensor, select_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Sample tokens for positions where select_mask == True.
            Returns:
                updated tokens, probs
            """
            logits, _ = self(tokens, global_ctx=global_ctx, local_ctx=local_ctx, task_id=task_id)
            logits = _prepare_logits(logits)
            probs = F.softmax(logits, dim=-1)
    
            if select_mask.any():
                probs_flat = probs[select_mask]  # (M,V)
    
                if not torch.isfinite(probs_flat).all():
                    raise RuntimeError("[sample_maskgit] non-finite probabilities before multinomial")
    
                row_sums = probs_flat.sum(dim=-1)
                if (row_sums <= 0).any() or (~torch.isfinite(row_sums)).any():
                    raise RuntimeError("[sample_maskgit] invalid probability rows before multinomial")
    
                if (probs_flat < 0).any():
                    raise RuntimeError("[sample_maskgit] negative probabilities before multinomial")
    
                sampled = torch.multinomial(probs_flat, 1).squeeze(1)
    
                if (sampled < 0).any() or (sampled >= probs.size(-1)).any():
                    raise RuntimeError("[sample_maskgit] sampled invalid token IDs")
    
                tokens = tokens.clone()
                tokens[select_mask] = sampled
    
            return tokens, probs
    
        for s in range(int(steps)):
            masked_now = (tokens == self.mask_id) & changeable
            tokens, probs = _sample_into_masked_positions(tokens, masked_now)
    
            V = probs.size(-1)
            if (tokens < 0).any() or (tokens >= V).any():
                raise RuntimeError("[sample_maskgit] tokens out of range before confidence gather")
    
            # Confidence of current chosen token
            conf = probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)  # (B,N)
            conf = conf.masked_fill(fixed_mask, 2.0)
    
            rem = remaining_to_mask(s)
            if rem.max().item() == 0:
                break
    
            # Re-mask lowest-confidence eligible tokens
            neg_conf = -conf
            neg_conf = neg_conf.masked_fill(fixed_mask, float("-inf"))
    
            if remask_only_originally_masked:
                neg_conf = neg_conf.masked_fill(~originally_changeable, float("-inf"))
    
            for b in range(B):
                k_rem = int(rem[b].item())
                if k_rem <= 0:
                    continue
    
                valid = torch.isfinite(neg_conf[b])
                n_valid = int(valid.sum().item())
                if n_valid <= 0:
                    continue
    
                k_rem = min(k_rem, n_valid)
                scores = neg_conf[b].masked_fill(~valid, float("-inf"))
                _, idx = torch.topk(scores, k=k_rem, largest=True)
                tokens[b, idx] = self.mask_id
    
            # Restore fixed tokens exactly
            tokens[fixed_mask] = init_tokens[fixed_mask]
    
        # Final cleanup pass: fill any leftover masks once more
        leftover = (tokens == self.mask_id) & changeable
        if leftover.any():
            tokens, _ = _sample_into_masked_positions(tokens, leftover)
    
        # Restore fixed tokens exactly
        tokens[fixed_mask] = init_tokens[fixed_mask]
    
        # Final hard checks
        if (tokens == self.mask_id).any():
            n_left = int((tokens == self.mask_id).sum().item())
            raise RuntimeError(f"[sample_maskgit] returning with {n_left} mask tokens still present")
    
        if (tokens < 0).any() or (tokens >= self.vocab_size).any():
            raise RuntimeError(
                f"[sample_maskgit] returning invalid token IDs: "
                f"min={int(tokens.min().item())}, max={int(tokens.max().item())}, vocab_size={int(self.vocab_size)}"
            )
    
        return tokens
            
    
    @torch.no_grad()
    def sample_from_prior(
        self,
        gt_codes: torch.Tensor,
        global_ctx: torch.Tensor,
        local_ctx: torch.Tensor,
        task_id: torch.Tensor,
        steps: int = 32,
        top_k: Optional[int] = None,
        predict_mask: Optional[torch.Tensor] = None,
        prompt_len: int = 0,
    ) -> torch.LongTensor:
        if not hasattr(self, "sample_maskgit") or not hasattr(self, "mask_id"):
            raise ValueError("sample_from_prior is for MGIT/MaskGIT priors only (needs sample_maskgit + mask_id).")
    
        gt_codes = gt_codes.long()
        B, N = gt_codes.shape
        device = gt_codes.device
    
        if predict_mask is not None:
            pm = predict_mask
            if pm.dim() == 3:
                pm = pm.squeeze(-1)
            pm = pm.to(device=device)
            if pm.shape != (B, N):
                raise ValueError(f"predict_mask must be (B,N) or (B,N,1); got {tuple(predict_mask.shape)}")
    
            fixed_mask = (pm == 0)
            tokens = torch.full((B, N), int(self.mask_id), device=device, dtype=torch.long)
            tokens[fixed_mask] = gt_codes[fixed_mask]
    
            return self.sample_maskgit(
                init_tokens=tokens,
                global_ctx=global_ctx,
                local_ctx=local_ctx,
                task_id=task_id,
                fixed_mask=fixed_mask,
                steps=int(steps),
                top_k=top_k,
            )
    
        prompt_len = int(max(0, min(int(prompt_len), N)))
        fixed_mask = torch.zeros((B, N), device=device, dtype=torch.bool)
        if prompt_len > 0:
            fixed_mask[:, :prompt_len] = True
    
        tokens = torch.full((B, N), int(self.mask_id), device=device, dtype=torch.long)
        if prompt_len > 0:
            tokens[:, :prompt_len] = gt_codes[:, :prompt_len]
    
        return self.sample_maskgit(
            init_tokens=tokens,
            global_ctx=global_ctx,
            local_ctx=local_ctx,
            task_id=task_id,
            fixed_mask=fixed_mask,
            steps=int(steps),
            top_k=top_k,
        )