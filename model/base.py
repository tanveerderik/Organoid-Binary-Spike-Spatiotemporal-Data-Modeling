#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:24:28 2026

@author: derik
"""

#%%
from typing import Optional, Tuple, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


#%% Model
# --------------------------- ViT blocks ---------------------------


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x
    

class CtxEmbed(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int, mlp_ratio: float = 2.0, drop: float = 0.0, alpha_init: float = 0.0):
        super().__init__()
        self.alpha_raw = nn.Parameter(torch.tensor(float(alpha_init)))  # start at 0 => behaves like proj-only
        self.alpha_max = 1.0  # or 0.5 if you want it conservative
        
        self.proj = nn.Linear(in_dim, emb_dim, bias=True)
        self.mlp = MLP(emb_dim, mlp_ratio=mlp_ratio, drop=drop)
        self.ln = nn.LayerNorm(emb_dim)

    def forward(self, x):
        h = self.proj(x)
        h = h + self.alpha_max * torch.tanh(self.alpha_raw) * self.mlp(h)
        return self.ln(h)
    
    
class HierarchicalVectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_codes: Union[int, Sequence[int]] = 512,
        code_dim: int = 128,
        decay: float = 0.95,
        eps: float = 1e-5,
        beta: float = 0.25,
        blank_code: int = -1,
        blank_token_std: float = 0.02,
        usage_loss_weight: float = 1e-3,
        usage_tau: float = 0.5,
        num_quantizers: int = 2,
        level_scale_decay: float = 1.0,

        active_quantizers: Optional[int] = None,

        sep_cos_thresh: float = 0.8,
        sep_margin_weight_l0: float = 5e-2,
        sep_margin_weight_later: float = 5e-3,

        dead_code_restart_every: int = 200,
        dead_code_usage_thresh: float = 1.0,
        restart_noise_std: float = 0.01,
    ):
        super().__init__()

        if num_quantizers < 1:
            raise ValueError("num_quantizers must be >= 1")

        if isinstance(num_codes, int):
            if num_codes < 1:
                raise ValueError("num_codes must be >= 1")
            num_codes_per_level = [int(num_codes)] * int(num_quantizers)
        else:
            num_codes_per_level = [int(k) for k in num_codes]
            if len(num_codes_per_level) != int(num_quantizers):
                raise ValueError(
                    f"len(num_codes) must equal num_quantizers; got "
                    f"{len(num_codes_per_level)} vs {num_quantizers}"
                )
            if any(k < 1 for k in num_codes_per_level):
                raise ValueError("all num_codes entries must be >= 1")

        self.num_quantizers = int(num_quantizers)
        self.num_codes_per_level = num_codes_per_level
        self.max_num_codes = int(max(num_codes_per_level))

        # backward-compat: some old code still reads self.num_codes
        self.num_codes = self.max_num_codes

        self.code_dim = int(code_dim)
        self.decay = float(decay)
        self.eps = float(eps)
        self.beta = float(beta)
        self.blank_code = int(blank_code)
        self.usage_loss_weight = float(usage_loss_weight)
        self.usage_tau = float(usage_tau)
        self.level_scales = [1.0] + [level_scale_decay ** i for i in range(1, self.num_quantizers)]

        self.active_quantizers = int(active_quantizers) if active_quantizers is not None else int(num_quantizers)

        self.sep_cos_thresh = float(sep_cos_thresh)
        self.sep_margin_weight_l0 = float(sep_margin_weight_l0)
        self.sep_margin_weight_later = float(sep_margin_weight_later)

        self.freeze_codebook_updates = False
        self.dead_code_restart_every = int(dead_code_restart_every)
        self.dead_code_usage_thresh = float(dead_code_usage_thresh)
        self.restart_noise_std = float(restart_noise_std)

        self.register_buffer("step_counter", torch.zeros((), dtype=torch.long))

        # Dense hierarchical tree codebooks.
        # Level l has shape (*num_codes_per_level[:l+1], D)
        self.tree_embeds = nn.ParameterList()
        self.level_num_entries = []
        
        for lvl in range(self.num_quantizers):
            shape_l = tuple(self.num_codes_per_level[:lvl + 1]) + (self.code_dim,)
            p = nn.Parameter(torch.empty(*shape_l))
            nn.init.normal_(p, mean=0.0, std=0.1)
            self.tree_embeds.append(p)
        
            n_entries_l = 1
            for k in self.num_codes_per_level[:lvl + 1]:
                n_entries_l *= int(k)
            self.level_num_entries.append(int(n_entries_l))
        
        self.blank_token = nn.Parameter(torch.zeros(self.code_dim))
        nn.init.normal_(self.blank_token, mean=0.0, std=float(blank_token_std))
        
        # EMA buffers, one dense tree per level.
        for lvl, p in enumerate(self.tree_embeds):
            self.register_buffer(f"ema_count_l{lvl}", torch.ones(*p.shape[:-1]))
            self.register_buffer(f"ema_weight_l{lvl}", p.data.clone())
                
    def _ema_count(self, lvl: int):
        return getattr(self, f"ema_count_l{lvl}")
    
    def _ema_weight(self, lvl: int):
        return getattr(self, f"ema_weight_l{lvl}")
    
    def _flat_codebook(self, lvl: int) -> torch.Tensor:
        return self.tree_embeds[lvl].reshape(-1, self.code_dim)
    
    def get_effective_codebook_weight(self, lvl: int) -> torch.Tensor:
        return self._flat_codebook(lvl)
    
    def _prefix_flat_index(self, codes_prefix: torch.Tensor, upto_level: int) -> torch.Tensor:
        """
        codes_prefix: (..., upto_level+1), containing codes [k0,...,k_upto]
        returns flattened index into dense tree level `upto_level`.
    
        Example:
          lvl 0: k0
          lvl 1: k0*K1 + k1
          lvl 2: (k0*K1 + k1)*K2 + k2
        """
        idx = codes_prefix[..., 0].long()
        for j in range(1, upto_level + 1):
            idx = idx * int(self.num_codes_per_level[j]) + codes_prefix[..., j].long()
        return idx
    
    def get_codebook_entry(self, lvl: int, codes_prefix: torch.Tensor) -> torch.Tensor:
        """
        codes_prefix must contain the full path up to `lvl`.
    
        lvl=0:
            codes_prefix shape (...,1), values [k0]
            returns E0[k0]
    
        lvl=1:
            codes_prefix shape (...,2), values [k0,k1]
            returns E1[k0,k1]
    
        lvl=2:
            codes_prefix shape (...,3), values [k0,k1,k2]
            returns E2[k0,k1,k2]
        """
        flat_idx = self._prefix_flat_index(codes_prefix, lvl)
        cb = self._flat_codebook(lvl)
        return cb[flat_idx]

    @property
    def embed(self):
        """
        Backward-compatible level-0 embedding-like object.
        Prefer get_effective_codebook_weight(0) in new code.
        """
        return self.tree_embeds[0]

    @torch.no_grad()
    def _compute_vq_aux(
        self,
        active_codes_levels: torch.Tensor,
        num_total_tokens: int,
        num_blank_tokens: int,
        device: torch.device,
        commit_losses: list[torch.Tensor],
        usage_losses: list[torch.Tensor],
        usage_aux_levels: list[dict],
        residual_norm: torch.Tensor,
        level_norms: list[torch.Tensor],
        loss_cb_norm: torch.Tensor,
    ):
        """
        active_codes_levels: (M, L) long.
    
        Metrics are computed over the flattened effective table at each level:
          lvl 0: k0
          lvl 1: k0*K1 + k1
          lvl 2: (k0*K1 + k1)*K2 + k2
    
        Therefore L2 perplexity/entropy/active codes measure actual used
        flattened branches, not reused child IDs alone.
        """
        L_total = self.num_quantizers
    
        aux = {
            "num_nonblank": int(active_codes_levels.shape[0]),
            "num_blank": int(num_blank_tokens),
            "blank_frac": float(num_blank_tokens / max(1, num_total_tokens)),
            "perplexity_nonblank": 1.0,
            "entropy_nonblank": 0.0,
            "active_codes_nonblank": 0.0,
            "active_frac_nonblank": 0.0,
            "commit_loss": 0.0,
            "usage_loss": 0.0,
            "residual_norm": float(residual_norm.detach().item()),
            "level_norms": [float(n.detach().item()) for n in level_norms],
            "loss_cb_norm": float(loss_cb_norm.detach().item()),
            "levels": [],
        }
    
        if active_codes_levels.numel() == 0:
            aux["level_norms"] = [0.0 for _ in range(L_total)]
            aux["loss_cb_norm"] = 0.0
            for lvl in range(L_total):
                aux["levels"].append({
                    "perplexity_nonblank": 1.0,
                    "entropy_nonblank": 0.0,
                    "active_codes_nonblank": 0,
                    "active_frac_nonblank": 0.0,
                    "commit_loss": 0.0,
                    "usage_loss": 0.0,
                    "soft_usage_entropy": 0.0,
                    "soft_usage_perplexity": 1.0,
                    "soft_usage_loss": 0.0,
                    "active_parents_nonblank": 0,
                    "child_per_active_parent": 0.0,
                })
            return aux
    
        per_level_perplexity = []
        per_level_entropy = []
        per_level_active_codes = []
        per_level_active_frac = []
    
        L_active = active_codes_levels.shape[1]
    
        for lvl in range(L_active):
            n_entries = self.level_num_entries[lvl]
    
            flat_idx = self._prefix_flat_index(
                active_codes_levels[:, :lvl + 1],
                upto_level=lvl,
            )
    
            counts = torch.bincount(
                flat_idx,
                minlength=n_entries,
            ).to(device=device, dtype=torch.float32)
    
            probs = counts / counts.sum().clamp_min(1.0)
            nz = probs > 0
    
            entropy = -(probs[nz] * probs[nz].log()).sum()
            perplexity = entropy.exp()
            active_codes = int(nz.sum().item())
            active_frac = float(active_codes / max(1, n_entries))
    
            if lvl == 0:
                active_parents = active_codes
                child_per_parent = 1.0 if active_codes > 0 else 0.0
            else:
                parent_idx = self._prefix_flat_index(
                    active_codes_levels[:, :lvl],
                    upto_level=lvl - 1,
                )
                parent_counts = torch.bincount(
                    parent_idx,
                    minlength=self.level_num_entries[lvl - 1],
                )
                active_parents = int((parent_counts > 0).sum().item())
                child_per_parent = float(active_codes / max(1, active_parents))
    
            lvl_aux = {
                "perplexity_nonblank": float(perplexity.item()),
                "entropy_nonblank": float(entropy.item()),
                "active_codes_nonblank": active_codes,
                "active_frac_nonblank": active_frac,
                "commit_loss": float(commit_losses[lvl].detach().item()),
                "usage_loss": float((self.usage_loss_weight * usage_losses[lvl]).detach().item()),
                "active_parents_nonblank": active_parents,
                "child_per_active_parent": child_per_parent,
    
                **usage_aux_levels[lvl],
            }
    
            aux["levels"].append(lvl_aux)
    
            per_level_perplexity.append(float(perplexity.item()))
            per_level_entropy.append(float(entropy.item()))
            per_level_active_codes.append(float(active_codes))
            per_level_active_frac.append(float(active_frac))
    
        aux["perplexity_nonblank"] = float(sum(per_level_perplexity) / max(1, L_active))
        aux["entropy_nonblank"] = float(sum(per_level_entropy) / max(1, L_active))
        aux["active_codes_nonblank"] = float(sum(per_level_active_codes) / max(1, L_active))
        aux["active_frac_nonblank"] = float(sum(per_level_active_frac) / max(1, L_active))
        aux["commit_loss"] = float(torch.stack([c.detach() for c in commit_losses]).mean().item())
        aux["usage_loss"] = float(
            (self.usage_loss_weight * torch.stack([u.detach() for u in usage_losses]).mean()).item()
        )
    
        return aux

    def _soft_usage_loss_from_d2(self, d2: torch.Tensor, num_codes_this_level: int):
        """
        d2: (M, K_l)
        """
        if d2.numel() == 0:
            zero = self.tree_embeds[0].new_zeros(())
            return zero, {
                "soft_usage_entropy": 0.0,
                "soft_usage_perplexity": 1.0,
                "soft_usage_loss": 0.0,
            }
    
        logits = -d2
        probs = F.softmax(logits / self.usage_tau, dim=-1)   # (M, K_l)
    
        usage = probs.mean(dim=0)
        usage = usage / usage.sum().clamp_min(1e-8)
    
        entropy = -(usage * usage.clamp_min(1e-8).log()).sum()
        perplexity = entropy.exp()
    
        max_entropy = torch.log(
            torch.tensor(float(num_codes_this_level), device=usage.device, dtype=usage.dtype)
        )
    
        loss_usage = max_entropy - entropy
    
        return loss_usage, {
            "soft_usage_entropy": float(entropy.detach().item()),
            "soft_usage_perplexity": float(perplexity.detach().item()),
            "soft_usage_loss": float(loss_usage.detach().item()),
        }

    def _compute_d2(self, x: torch.Tensor, codebook_weight: torch.Tensor):
        """
        x: (M, D)
        codebook_weight: (K, D)
        returns d2: (M, K)
        """
        x_f = x.float()
        e_f = codebook_weight.float()

        d2 = (
            x_f.pow(2).sum(1, keepdim=True)
            - 2.0 * (x_f @ e_f.T)
            + e_f.pow(2).sum(1, keepdim=True).T
        )
        return d2


    def set_active_quantizers(self, n: int):
        n = int(n)
        n = max(1, min(self.num_quantizers, n))
        self.active_quantizers = n
    
    def _used_code_duplicate_loss(
        self,
        codebook_weight: torch.Tensor,   # (K,D)
        used_counts: torch.Tensor,       # (K,)
        cos_thresh: float,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Penalize both positive and negative near-duplicate cosine among USED codes.
        This is much more aligned with your actual pathology than global sim^2.
        """
        used_mask = used_counts > 0
        if int(used_mask.sum().item()) <= 1:
            return codebook_weight.new_zeros(())
    
        w = F.normalize(codebook_weight[used_mask], dim=1, eps=eps)
        sim = w @ w.t()
    
        K_used = sim.size(0)
        eye = torch.eye(K_used, device=sim.device, dtype=torch.bool)
        offdiag = sim[~eye]
    
        # punish both positive and negative near-duplicates
        return F.relu(offdiag.abs() - cos_thresh).pow(2).mean()
    
    def _per_parent_child_duplicate_loss(
        self,
        codebook_weight: torch.Tensor,   # flattened: (parent_count * K_child, D)
        used_counts: torch.Tensor,       # flattened: (parent_count * K_child,)
        parent_count: int,
        K_child: int,
        cos_thresh: float,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Penalize positive and negative near-duplicate child codes only within the same parent branch.
        Does not compare children across different parents.
        """
        D = codebook_weight.shape[-1]
    
        w = codebook_weight.reshape(parent_count, K_child, D)
        counts = used_counts.reshape(parent_count, K_child)
    
        losses = []
    
        for p in range(parent_count):
            used_mask = counts[p] > 0
            if int(used_mask.sum().item()) <= 1:
                continue
    
            wp = F.normalize(w[p, used_mask], dim=1, eps=eps)
            sim = wp @ wp.t()
    
            n = sim.size(0)
            eye = torch.eye(n, device=sim.device, dtype=torch.bool)
            offdiag = sim[~eye]
    
            # punish both positive and negative angle alignment
            losses.append(F.relu(offdiag.abs() - cos_thresh).pow(2).mean())
    
        if len(losses) == 0:
            return codebook_weight.new_zeros(())
    
        return torch.stack(losses).mean()

    @torch.no_grad()
    def _maybe_restart_dead_codes(
        self,
        lvl: int,
        residual: torch.Tensor,
    ):
        """
        Reinitialize dead flattened tree entries from residual samples.
        For level l, this checks all prod(K0...Kl) entries.
        """
        if residual.numel() == 0:
            return
    
        if self.dead_code_restart_every <= 0:
            return
    
        if int(self.step_counter.item()) % self.dead_code_restart_every != 0:
            return
    
        ema_count = self._ema_count(lvl).reshape(-1)
        ema_weight = self._ema_weight(lvl).reshape(-1, self.code_dim)
        cb = self.tree_embeds[lvl].data.reshape(-1, self.code_dim)
    
        dead_mask = ema_count < self.dead_code_usage_thresh
        dead_idx = torch.nonzero(dead_mask, as_tuple=False).squeeze(1)
    
        if dead_idx.numel() == 0:
            return
    
        M = residual.size(0)
        take = min(int(dead_idx.numel()), int(M))
        if take <= 0:
            return
    
        perm = torch.randperm(M, device=residual.device)[:take]
        samples = residual[perm].detach()
    
        if self.restart_noise_std > 0.0:
            samples = samples + self.restart_noise_std * torch.randn_like(samples)
    
        dead_sel = dead_idx[:take]
    
        cb[dead_sel] = samples.to(cb.dtype)
        ema_weight[dead_sel] = samples.to(ema_weight.dtype)
        ema_count[dead_sel] = torch.ones_like(ema_count[dead_sel])

    def quantize_active_only(
        self,
        z_e_active: torch.Tensor,      # (M, D)
        active_flat: torch.Tensor,     # (B*N,) bool, True = active
        num_total_tokens: int,         # B*N
        return_logits: bool = False,
        return_aux: bool = True,
    ):
        """
        Residual-quantize only active tokens, then scatter back into full flattened token grid.

        Blank positions:
          - z_q_full gets self.blank_token
          - codes_full gets self.blank_code at every residual level

        Active positions:
          - quantized sequentially with EMA codebooks over indices [0..K-1]
        """
        device = active_flat.device
        base_weight = self.tree_embeds[0]
        dtype = z_e_active.dtype if z_e_active.numel() > 0 else base_weight.dtype
        
        if self.training and not self.freeze_codebook_updates:
            self.step_counter.add_(1)
            
        D = self.code_dim
        Kmax = self.max_num_codes
        L_total = self.num_quantizers
        L = min(self.active_quantizers, self.num_quantizers)
        
        level_norms = []

        if active_flat.dtype != torch.bool:
            active_flat = active_flat.to(torch.bool)

        if active_flat.numel() != num_total_tokens:
            raise ValueError("active_flat size must equal num_total_tokens")

        if z_e_active.numel() > 0 and z_e_active.shape[-1] != D:
            raise ValueError(f"Expected code_dim={D}, got {z_e_active.shape[-1]}")

        num_blank_tokens = int((~active_flat).sum().item())

        # Initialize full outputs with blank token / blank code
        blank_embed = self.blank_token.to(device=device, dtype=dtype)
        z_q_full = blank_embed.unsqueeze(0).expand(num_total_tokens, -1).clone()

        codes_full = torch.full(
            (num_total_tokens, L_total),
            fill_value=self.blank_code,
            dtype=torch.long,
            device=device,
        )

        neg_large = torch.finfo(dtype).min
        code_logits_full = torch.full(
            (num_total_tokens, L_total, Kmax),
            neg_large,
            dtype=dtype,
            device=device,
        )

        # No active tokens
        if z_e_active.numel() == 0:
            vq_loss = blank_embed.new_zeros(())
            vq_aux = None
            if return_aux:
                vq_aux = {
                    "num_nonblank": 0,
                    "num_blank": int(num_total_tokens),
                    "blank_frac": 1.0,
                    "perplexity_nonblank": 1.0,
                    "entropy_nonblank": 0.0,
                    "active_codes_nonblank": 0.0,
                    "active_frac_nonblank": 0.0,
                    "commit_loss": 0.0,
                    "usage_loss": 0.0,
                    "residual_norm": 0.0,
                    "levels": [
                        {
                            "perplexity_nonblank": 1.0,
                            "entropy_nonblank": 0.0,
                            "active_codes_nonblank": 0,
                            "active_frac_nonblank": 0.0,
                            "commit_loss": 0.0,
                            "usage_loss": 0.0,
                            "soft_usage_entropy": 0.0,
                            "soft_usage_perplexity": 1.0,
                            "soft_usage_loss": 0.0,
                        }
                        for _ in range(L)
                    ],
                }

            if return_logits:
                return z_q_full, vq_loss, codes_full, code_logits_full, vq_aux
            return z_q_full, vq_loss, codes_full, vq_aux

        residual = z_e_active
        z_q_active_sum = torch.zeros_like(z_e_active)

        level_indices = []
        level_logits = []
        commit_losses = []
        usage_losses = []
        usage_aux_levels = []
        level_path_flat_indices = []

        # Quantize residuals sequentially through dense hierarchical tree.
        # Level 0 searches E0[:].
        # Level l>0 searches only the child table under the previously selected path.
        for lvl in range(L):
            K_l = self.num_codes_per_level[lvl]
        
            if lvl == 0:
                e = self.tree_embeds[0].reshape(K_l, self.code_dim)      # (K0,D)
                d2 = self._compute_d2(residual, e)                       # (M,K0)
                nn_idx = torch.argmin(d2, dim=1)                         # (M,)
                z_q_level = e[nn_idx]                                    # (M,D)
        
                path_flat_idx = nn_idx                                   # flat index into E0
        
            else:
                prev_codes = torch.stack(level_indices, dim=1)           # (M,lvl)
                parent_flat_idx = self._prefix_flat_index(prev_codes, lvl - 1)  # (M,)
        
                # Flatten E_l from (*K[:lvl+1],D) to (prod_prev, K_l, D)
                parent_count = self.level_num_entries[lvl - 1]
                e_l = self.tree_embeds[lvl].reshape(parent_count, K_l, self.code_dim)
        
                # Only compare residual to children under selected parent path.
                e_parent = e_l[parent_flat_idx]                          # (M,K_l,D)
        
                r_f = residual.float()
                e_f = e_parent.float()
                d2 = (
                    r_f.pow(2).sum(dim=1, keepdim=True).unsqueeze(-1)
                    - 2.0 * torch.bmm(e_f, r_f.unsqueeze(-1))
                    + e_f.pow(2).sum(dim=2, keepdim=True)
                ).squeeze(-1)                                            # (M,K_l)
        
                nn_idx = torch.argmin(d2, dim=1)                         # (M,)
                z_q_level = e_parent[
                    torch.arange(e_parent.size(0), device=device),
                    nn_idx,
                ]                                                        # (M,D)
        
                path_flat_idx = parent_flat_idx * K_l + nn_idx           # flat index into E_l
        
            level_indices.append(nn_idx)
            level_path_flat_indices.append(path_flat_idx)
        
            level_logit_l = torch.full(
                (d2.size(0), Kmax),
                fill_value=torch.finfo(dtype).min,
                dtype=dtype,
                device=d2.device,
            )
            level_logit_l[:, :K_l] = (-d2).to(dtype)
            level_logits.append(level_logit_l)
        
            commit_loss_l = self.beta * F.mse_loss(z_q_level.detach(), residual)
            loss_usage_l, usage_aux_l = self._soft_usage_loss_from_d2(d2, num_codes_this_level=K_l)
        
            if self.training and not self.freeze_codebook_updates:
                self._maybe_restart_dead_codes(lvl=lvl, residual=residual.detach())
        
            commit_losses.append(commit_loss_l)
            usage_losses.append(loss_usage_l)
            usage_aux_levels.append(usage_aux_l)
        
            scale = self.level_scales[lvl]
            level_norms.append((scale * z_q_level).pow(2).mean(dim=1).sqrt().mean())
            z_q_active_sum = z_q_active_sum + scale * z_q_level
        
            # EMA update for dense tree level using flattened path indices.
            # This avoids creating one-hot tensors of shape (M, prod(K0...Kl)).
            if self.training and not self.freeze_codebook_updates:
                with torch.no_grad():
                    n_entries = self.level_num_entries[lvl]
        
                    count_flat = torch.zeros(n_entries, device=device, dtype=residual.dtype)
                    count_flat.index_add_(
                        0,
                        path_flat_idx,
                        torch.ones_like(path_flat_idx, dtype=residual.dtype),
                    )
        
                    dw_flat = torch.zeros(n_entries, self.code_dim, device=device, dtype=residual.dtype)
                    dw_flat.index_add_(0, path_flat_idx, residual)
        
                    ema_count = self._ema_count(lvl)
                    ema_weight = self._ema_weight(lvl)
        
                    ema_count_flat = ema_count.reshape(-1)
                    ema_weight_flat = ema_weight.reshape(-1, self.code_dim)
        
                    ema_count_flat.mul_(self.decay).add_(count_flat, alpha=1.0 - self.decay)
                    ema_weight_flat.mul_(self.decay).add_(dw_flat, alpha=1.0 - self.decay)
        
                    smoothed = ema_count_flat.clamp_min(1e-5)
                    new_embed_flat = ema_weight_flat / smoothed.unsqueeze(1)
        
                    self.tree_embeds[lvl].data.reshape(-1, self.code_dim).copy_(
                        new_embed_flat.to(self.tree_embeds[lvl].dtype)
                    )
        
            residual = residual - scale * z_q_level.detach()

        loss_cb_norm = z_q_active_sum.new_zeros(())
        norm_cap = 5.0
        lambda_cb_norm = 1e-3
        
        for lvl in range(L):
            cb_norms = self.get_effective_codebook_weight(lvl).norm(dim=1)
            loss_cb_norm = loss_cb_norm + F.relu(cb_norms - norm_cap).pow(2).mean()
    
        loss_sep = z_q_active_sum.new_zeros(())
        
        for lvl in range(L):
            K_l = self.num_codes_per_level[lvl]
        
            used_counts_l = torch.bincount(
                level_path_flat_indices[lvl],
                minlength=self.level_num_entries[lvl],
            )
        
            cb_l = self.get_effective_codebook_weight(lvl)
        
            if lvl == 0:
                # Global duplicate prevention among coarse L1 codes.
                dup_loss = self._used_code_duplicate_loss(
                    codebook_weight=cb_l,
                    used_counts=used_counts_l,
                    cos_thresh=self.sep_cos_thresh,
                )
                loss_sep = loss_sep + self.sep_margin_weight_l0 * dup_loss
        
            else:
                # Per-parent duplicate prevention among child codes only.
                parent_count = self.level_num_entries[lvl - 1]
        
                dup_loss = self._per_parent_child_duplicate_loss(
                    codebook_weight=cb_l,
                    used_counts=used_counts_l,
                    parent_count=parent_count,
                    K_child=K_l,
                    cos_thresh=self.sep_cos_thresh,
                )
                loss_sep = loss_sep + self.sep_margin_weight_later * dup_loss
        
        vq_loss = (
            torch.stack(commit_losses).mean()
            + self.usage_loss_weight * torch.stack(usage_losses).mean()
            +  lambda_cb_norm * loss_cb_norm
            + loss_sep
        )

        # Scatter active results back into full grid
        active_codes = torch.stack(level_indices, dim=1)   # (M, L)
        codes_full[active_flat, :L] = active_codes
        
        z_q_active_sum = z_q_active_sum.to(device=z_q_full.device, dtype=z_q_full.dtype)
        z_q_full[active_flat] = z_q_active_sum

        if return_logits:
            active_logits = torch.stack(level_logits, dim=1)  # (M, L, K)
            code_logits_full[active_flat, :L, :] = active_logits

        # Straight-through estimator only on active positions, on the FULL SUM
        z_q_full_st = z_q_full.clone()
        z_q_full_st[active_flat] = z_e_active + (z_q_active_sum - z_e_active).detach()

        vq_aux = None
        if return_aux:
            residual_norm = residual.pow(2).mean().sqrt()
            vq_aux = self._compute_vq_aux(
                active_codes_levels=active_codes,
                num_total_tokens=num_total_tokens,
                num_blank_tokens=num_blank_tokens,
                device=device,
                commit_losses=commit_losses,
                usage_losses=usage_losses,
                usage_aux_levels=usage_aux_levels,
                residual_norm=residual_norm,
                level_norms=level_norms,
                loss_cb_norm=loss_cb_norm,
            )

        if return_logits:
            return z_q_full_st, vq_loss, codes_full, code_logits_full, vq_aux
        return z_q_full_st, vq_loss, codes_full, vq_aux

    def forward(
        self,
        z_e: torch.Tensor,                        # (B, N, D)
        blank_mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        return_aux: bool = True,
    ):
        B, N, D = z_e.shape
        if D != self.code_dim:
            raise ValueError(f"code_dim={self.code_dim} but got {D}")

        device = z_e.device

        if blank_mask is None:
            blank_mask_bn = torch.zeros((B, N), dtype=torch.bool, device=device)
        else:
            if blank_mask.shape != (B, N):
                raise ValueError(f"blank_mask must be {(B, N)}, got {tuple(blank_mask.shape)}")
            blank_mask_bn = blank_mask.to(device=device, dtype=torch.bool)

        active_flat = (~blank_mask_bn).reshape(-1)
        z_e_flat = z_e.reshape(B * N, D)
        z_e_active = z_e_flat[active_flat]

        out = self.quantize_active_only(
            z_e_active=z_e_active,
            active_flat=active_flat,
            num_total_tokens=B * N,
            return_logits=return_logits,
            return_aux=return_aux,
        )

        if return_logits:
            z_q_flat, vq_loss, codes_flat, code_logits_flat, vq_aux = out
            z_q = z_q_flat.view(B, N, D)
            codes = codes_flat.view(B, N, self.num_quantizers)
            code_logits = code_logits_flat.view(B, N, self.num_quantizers, self.max_num_codes)
            return z_q, vq_loss, codes, code_logits, vq_aux

        z_q_flat, vq_loss, codes_flat, vq_aux = out
        z_q = z_q_flat.view(B, N, D)
        codes = codes_flat.view(B, N, self.num_quantizers)
        return z_q, vq_loss, codes, vq_aux
    
# --------------------------- VQ-VAE core ---------------------------



# --------------------------- transformer blocks ---------------------------

class ConvStem3D(nn.Module):
    def __init__(self, in_chans=1, out_chans=1, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv3d(
            in_chans,
            out_chans,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,   # IMPORTANT
        )

    def forward(self, x):
        return self.conv(x)
    
class ActivePatchEmbed3D(nn.Module):
    """
    Active-only patch embed.

    - Patchifies raw volume explicitly.
    - Detects blank patches before projection.
    - Projects only active patches with a Linear layer.
    - Scatters back to full (B, N, D) with fill_value for blanks.

    This fully removes blank patches from the patch embedding forward path.
    """
    def __init__(self, in_chans=1, embed_dim=384, patch_size=(16, 16, 16), bias=True):
        super().__init__()
        self.patch_size = tuple(int(v) for v in patch_size)
        self.in_chans = int(in_chans)
        self.embed_dim = int(embed_dim)

        pT, pH, pW = self.patch_size
        self.patch_dim = self.in_chans * pT * pH * pW

        self.proj = nn.Linear(self.patch_dim, self.embed_dim, bias=bias)

    def patchify(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        pT, pH, pW = self.patch_size

        if (T % pT) != 0 or (H % pH) != 0 or (W % pW) != 0:
            raise ValueError(
                f"Input shape {(T,H,W)} must be divisible by patch size {self.patch_size}"
            )

        t = T // pT
        h = H // pH
        w = W // pW

        patches = (
            x.view(B, C, t, pT, h, pH, w, pW)
             .permute(0, 2, 4, 6, 3, 5, 7, 1)
             .contiguous()
             .view(B, t * h * w, pT * pH * pW * C)
        )
        return patches, (t, h, w)

    def forward(self, x: torch.Tensor, blank_mask: torch.Tensor = None, fill_value: float = 0.0):
        """
        Args:
          x:          (B, C, T, H, W)
          blank_mask: optional (B, N) bool computed externally
        Returns:
          tokens_full: (B, N, D)
          grid:        (t_tok, h_tok, w_tok)
          blank_mask:  (B, N) bool
          active_mask: (B, N) bool
        """
        patches, grid = self.patchify(x)  # (B, N, patch_dim)
    
        if blank_mask is None:
            blank_mask = patches.sum(dim=-1).eq(0)
    
        active_mask = ~blank_mask
    
        B, N, PD = patches.shape
        D = self.embed_dim
    
        patches_flat = patches.view(B * N, PD)
        active_flat = active_mask.view(B * N)
    
        # Keep inputs to Linear in parameter dtype/device for stable math setup.
        proj_device = self.proj.weight.device
        proj_in_dtype = self.proj.weight.dtype
    
        if active_flat.any():
            active_patches = patches_flat[active_flat].to(device=proj_device, dtype=proj_in_dtype)
            active_tokens = self.proj(active_patches)   # under AMP this may be fp16/bf16
    
            # IMPORTANT: allocate destination in the ACTUAL output dtype
            out_dtype = active_tokens.dtype
            tokens_full = torch.full(
                (B * N, D),
                fill_value=float(fill_value),
                device=active_tokens.device,
                dtype=out_dtype,
            )
            tokens_full[active_flat] = active_tokens
        else:
            # no active patches: choose a safe downstream dtype
            # keeping this in fp32 is fine because later layers/norms handle it safely
            tokens_full = torch.full(
                (B * N, D),
                fill_value=float(fill_value),
                device=proj_device,
                dtype=proj_in_dtype,
            )
    
        tokens_full = tokens_full.view(B, N, D)
        return tokens_full, grid, blank_mask, active_mask


class PatchRenderer3D(nn.Module):
    """
    Learned token-to-volume renderer.
    
    Takes token grid embeddings (B, N, D), reshapes them to (B, D, t_tok, h_tok, w_tok),
    renders a dense logits volume with ConvTranspose3d, and optionally applies a small
    3D refinement network.
    """

    def __init__(
        self,
        token_dim: int,
        out_chans: int = 1,
        patch_size=(16, 16, 16),
        refine_layers: int = 1,
        refine_hidden: int = 32,
    ):
        super().__init__()

        self.token_dim = int(token_dim)
        self.out_chans = int(out_chans)
        self.patch_size = tuple(int(x) for x in patch_size)

        # Main inverse-like renderer
        self.deproj = nn.ConvTranspose3d(
            in_channels=self.token_dim,
            out_channels=refine_hidden,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        
        if refine_layers > 0:
            layers = []
            in_ch = refine_hidden
            for i in range(refine_layers):
                hidden = int(refine_hidden)
                layers.extend([
                    nn.Conv3d(in_ch, hidden, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.GroupNorm(num_groups=1, num_channels=hidden),
                    nn.GELU(),
                ])
                in_ch = hidden
        
            layers.append(
                nn.Conv3d(in_ch, self.out_chans, kernel_size=1, stride=1, padding=0, bias=True)
            )
            self.refine = nn.Sequential(*layers)
        else:
            self.refine = nn.Conv3d(refine_hidden, self.out_chans, kernel_size=1, stride=1, padding=0, bias=True)

    def _tokens_to_grid(self, tokens: torch.Tensor, grid):
        """
        tokens: (B, N, D)
        grid:   (t_tok, h_tok, w_tok)

        returns:
          x: (B, D, t_tok, h_tok, w_tok)
        """
        B, N, D = tokens.shape
        t_tok, h_tok, w_tok = map(int, grid)

        if D != self.token_dim:
            raise ValueError(f"Expected token dim {self.token_dim}, got {D}")

        if N != t_tok * h_tok * w_tok:
            raise ValueError(
                f"N={N} does not match grid product {t_tok*h_tok*w_tok} "
                f"for grid={grid}"
            )

        # Match PatchEmbed3D flatten order:
        # Conv3d -> (B, D, T', H', W') -> flatten(2).transpose(1, 2)
        x = tokens.transpose(1, 2).contiguous().view(B, D, t_tok, h_tok, w_tok)
        return x

    def _vol_to_patches(self, vol: torch.Tensor, grid):
        """
        vol: (B, C, T, H, W)
        returns patches: (B, N, patch_dim)
        """
        B, C, T, H, W = vol.shape
        t_tok, h_tok, w_tok = map(int, grid)
        pT, pH, pW = self.patch_size

        if T != t_tok * pT or H != h_tok * pH or W != w_tok * pW:
            raise ValueError(
                f"Rendered volume shape {(T,H,W)} does not match grid={grid} "
                f"and patch_size={self.patch_size}"
            )

        patches = (
            vol.view(B, C, t_tok, pT, h_tok, pH, w_tok, pW)
               .permute(0, 2, 4, 6, 3, 5, 7, 1)
               .contiguous()
               .view(B, t_tok * h_tok * w_tok, pT * pH * pW * C)
        )
        return patches

    def forward(self, tokens: torch.Tensor, grid, return_patches: bool = True):
        """
        tokens: (B, N, D)
        grid:   (t_tok, h_tok, w_tok)

        returns:
          logits_vol:     (B, C, T, H, W)
          pred_patches:   (B, N, patch_dim)   if return_patches=True else None
        """
        x = self._tokens_to_grid(tokens, grid)   # (B, D, t_tok, h_tok, w_tok)
        logits_vol = self.deproj(x)              # (B, C, T, H, W)
        logits_vol = self.refine(logits_vol)

        pred_patches = self._vol_to_patches(logits_vol, grid) if return_patches else None
        return logits_vol, pred_patches



# --- Encoder and Decoder blocks 

class SparseSelfAttention(nn.Module):
    """
    Explicit self-attention with manual QKV projections.
    Safer than nn.TransformerEncoder / native fastpath for eval+AMP.

    Input:
      x: (B, L, D)
      key_padding_mask: (B, L) bool, True = PAD

    Output:
      out: (B, L, D)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,                           # (B, L, D)
        key_padding_mask: Optional[torch.Tensor] = None,   # (B, L) bool, True = PAD
    ):
        B, L, D = x.shape
        H = self.num_heads
        Hd = self.head_dim

        q = self.q_proj(x).view(B, L, H, Hd).transpose(1, 2)   # (B, H, L, Hd)
        k = self.k_proj(x).view(B, L, H, Hd).transpose(1, 2)   # (B, H, L, Hd)
        v = self.v_proj(x).view(B, L, H, Hd).transpose(1, 2)   # (B, H, L, Hd)

        # Compute attention scores in fp32 for stability / dtype consistency
        qf = q.float()
        kf = k.float()
        vf = v.float()

        attn_scores = torch.matmul(qf, kf.transpose(-2, -1)) * self.scale   # (B, H, L, L)

        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, L):
                raise ValueError(
                    f"key_padding_mask must have shape {(B, L)}, got {tuple(key_padding_mask.shape)}"
                )
            pad = key_padding_mask[:, None, None, :].to(dtype=torch.bool, device=attn_scores.device)
            fill_value = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(pad, fill_value)

        attn = torch.softmax(attn_scores, dim=-1)   # fp32
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, vf)                # (B, H, L, Hd), fp32
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = out.to(dtype=x.dtype)

        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


class SparseEncoderBlock(nn.Module):
    """
    Pre-norm encoder block:
      x = x + self_attn(ln1(x))
      x = x + mlp(ln2(x))
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = SparseSelfAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop1 = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.drop2 = nn.Dropout(drop)

    def forward(
        self,
        x: torch.Tensor,                           # (B, L, D)
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        x1 = self.norm1(x).to(dtype=x.dtype)
        x = x + self.drop1(self.attn(x1, key_padding_mask=key_padding_mask))
    
        x2 = self.norm2(x).to(dtype=x.dtype)
        x = x + self.drop2(self.mlp(x2))
        return x


class SparseTokenTransformerEncoder(nn.Module):
    """
    Token-sparse Transformer encoder with explicit attention blocks.

    Inputs:
      tokens:      (B, N, D)
      active_mask: (B, N) bool, True = active token, False = blank token
      pos_embed:   optional positional embedding, shape (N,D), (1,N,D), or (B,N,D)

    Behavior:
      - gathers only active tokens per sample
      - pads to max active length within the batch
      - runs explicit encoder blocks with key padding mask
      - scatters encoded active outputs back into the original (B, N, D) layout
      - blank positions are returned as fill_value (default: 0)

    Returns:
      full_out:       (B, N, D)
      padded_out:     (B, Lmax, D)
      key_pad_mask:   (B, Lmax) bool, True = pad
      lengths:        (B,) long
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        activation: str = "gelu",   # kept only for API compatibility
        norm_first: bool = True,    # kept only for API compatibility
    ):
        super().__init__()

        self.dim = int(dim)
        self.depth = int(depth)

        self.blocks = nn.ModuleList([
            SparseEncoderBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def _normalize_active_mask(active_mask: torch.Tensor, B: int, N: int, device: torch.device) -> torch.Tensor:
        if active_mask.shape != (B, N):
            raise ValueError(f"active_mask must have shape {(B, N)}, got {tuple(active_mask.shape)}")
        return active_mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _expand_pos_embed(
        pos_embed: Optional[torch.Tensor],
        B: int,
        N: int,
        D: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if pos_embed is None:
            return None

        if pos_embed.dim() == 2:
            if pos_embed.shape != (N, D):
                raise ValueError(f"pos_embed 2D must have shape {(N, D)}, got {tuple(pos_embed.shape)}")
            pos_embed = pos_embed.unsqueeze(0)

        if pos_embed.dim() != 3:
            raise ValueError("pos_embed must be None, (N,D), (1,N,D), or (B,N,D)")

        if pos_embed.shape == (1, N, D):
            pos_embed = pos_embed.expand(B, N, D)
        elif pos_embed.shape != (B, N, D):
            raise ValueError(f"pos_embed must have shape (1,{N},{D}) or ({B},{N},{D}), got {tuple(pos_embed.shape)}")

        return pos_embed.to(device=device, dtype=dtype)

    def forward(
        self,
        tokens: torch.Tensor,                 # (B, N, D)
        active_mask: torch.Tensor,           # (B, N) bool, True=active
        pos_embed: Optional[torch.Tensor] = None,
        fill_value: Optional[torch.Tensor] = None,   # scalar or (D,)
    ):
        B, N, D = tokens.shape
        device = tokens.device
        dtype = tokens.dtype

        if D != self.dim:
            raise ValueError(f"Expected token dim {self.dim}, got {D}")

        active_mask = self._normalize_active_mask(active_mask, B, N, device=device)
        pos_embed = self._expand_pos_embed(pos_embed, B, N, D, device=device, dtype=dtype)

        tokens_in = tokens if pos_embed is None else (tokens + pos_embed)

        lengths = active_mask.sum(dim=1)  # (B,)
        max_len = int(lengths.max().item()) if B > 0 else 0

        # fill tensor for blank positions in returned full_out
        if fill_value is None:
            fill_vec = torch.zeros(D, device=device, dtype=dtype)
        else:
            if not torch.is_tensor(fill_value):
                fill_value = torch.tensor(fill_value, device=device, dtype=dtype)
            fill_value = fill_value.to(device=device, dtype=dtype)
            if fill_value.numel() == 1:
                fill_vec = fill_value.expand(D)
            else:
                if tuple(fill_value.shape) != (D,):
                    raise ValueError(f"fill_value must be scalar or shape {(D,)}, got {tuple(fill_value.shape)}")
                fill_vec = fill_value

        full_out = fill_vec.view(1, 1, D).expand(B, N, D).clone()

        # Entire batch has zero active tokens
        if max_len == 0:
            padded_out = tokens_in.new_zeros((B, 0, D))
            key_pad_mask = torch.ones((B, 0), dtype=torch.bool, device=device)
            return full_out, padded_out, key_pad_mask, lengths

        # Build padded active-token tensor
        padded = tokens_in.new_zeros((B, max_len, D))
        key_pad_mask = torch.ones((B, max_len), dtype=torch.bool, device=device)  # True = PAD

        # Gather active tokens
        for b in range(B):
            idx = torch.nonzero(active_mask[b], as_tuple=False).squeeze(1)
            Lb = int(idx.numel())
            if Lb == 0:
                continue
            padded[b, :Lb] = tokens_in[b, idx]
            key_pad_mask[b, :Lb] = False

        # Explicit block stack
        x = padded
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_pad_mask)
        x = self.norm(x)

        encoded = x.to(device=full_out.device, dtype=full_out.dtype)

        # Scatter back
        for b in range(B):
            idx = torch.nonzero(active_mask[b], as_tuple=False).squeeze(1)
            Lb = int(idx.numel())
            if Lb == 0:
                continue
            full_out[b, idx] = encoded[b, :Lb]

        return full_out, encoded, key_pad_mask, lengths
    
    

class DecoderCrossAttnBlock(nn.Module):
    """
    Decoder block:
      1) self-attention on decoder tokens
      2) cross-attention: Q = decoder tokens, K/V = context tokens
      3) MLP

    Context is expected as (B, M, D), where M is small
    (e.g. 2 tokens = [local_ctx_token, global_ctx_token]).
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        *,
        attn_mask_kind: str = "none",
        grid: Optional[Tuple[int, int, int]] = None,
        window: Optional[int] = None,
    ):
        super().__init__()
        assert attn_mask_kind in {"none", "temporal_causal", "temporal_band", "temporal_band_bi"}

        # self-attn
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop1 = nn.Dropout(drop)

        # cross-attn
        self.norm2 = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop2 = nn.Dropout(drop)

        # mlp
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
        self.drop3 = nn.Dropout(drop)

        # self-attn mask config
        self._attn_mask_kind: str = attn_mask_kind
        self._grid: Optional[Tuple[int, int, int]] = tuple(grid) if grid is not None else None
        self._window: Optional[int] = int(window) if window is not None else None

        self.register_buffer("_attn_mask", None, persistent=False)

        if self._attn_mask_kind != "none" and self._grid is not None:
            self._rebuild_mask(device=torch.device("cpu"))

    @staticmethod
    def _build_temporal_mask(
        grid: Tuple[int, int, int],
        attn_mask_kind: str,
        window: Optional[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        t, h, w = grid
        N = t * h * w

        tau = torch.arange(t, device=device).repeat_interleave(h * w)
        tauQ = tau.unsqueeze(1).expand(N, N)
        tauK = tau.unsqueeze(0).expand(N, N)
        d = tauK - tauQ

        if attn_mask_kind == "temporal_causal":
            allow = (d <= 0)
        elif attn_mask_kind == "temporal_band":
            if window is None or window <= 0:
                allow = (d <= 0)
            else:
                allow = (d <= 0) & (d >= -(window - 1))
        elif attn_mask_kind == "temporal_band_bi":
            if window is None:
                raise ValueError("temporal_band_bi requires window.")
            wlen = int(window)
            if wlen <= 0 or (wlen % 2) == 0:
                raise ValueError(f"temporal_band_bi window must be positive odd, got {wlen}")
            radius = wlen // 2
            allow = (d.abs() <= radius)
        else:
            return None

        mask = torch.full((N, N), float("-inf"), device=device, dtype=dtype)
        mask[allow] = 0.0
        return mask

    def _rebuild_mask(self, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32):
        if self._attn_mask_kind == "none" or self._grid is None:
            self._attn_mask = None
            return
        dev = device if device is not None else (
            self._attn_mask.device if isinstance(self._attn_mask, torch.Tensor) else torch.device("cpu")
        )
        self._attn_mask = self._build_temporal_mask(self._grid, self._attn_mask_kind, self._window, dev, dtype)

    @torch.no_grad()
    def set_attn_mask(
        self,
        grid: Tuple[int, int, int],
        attn_mask_kind: str,
        window: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        assert attn_mask_kind in {"none", "temporal_causal", "temporal_band", "temporal_band_bi"}
        self._grid = tuple(int(x) for x in grid)
        self._attn_mask_kind = attn_mask_kind
        self._window = int(window) if window is not None else None
        self._rebuild_mask(device=device, dtype=torch.float32 if dtype is None else dtype)

    @torch.no_grad()
    def clear_mask(self):
        self._attn_mask_kind = "none"
        self._attn_mask = None

    def forward(
        self,
        x: torch.Tensor,                     # (B, N, D)
        ctx_tokens: Optional[torch.Tensor] = None,   # (B, M, D)
        key_padding_mask: Optional[torch.Tensor] = None,
        ctx_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        # 1) self-attn
        x1 = self.norm1(x).to(dtype=x.dtype)
        attn_mask = self._attn_mask.to(device=x.device, dtype=x1.dtype) \
            if isinstance(self._attn_mask, torch.Tensor) else None

        sa_out, _ = self.self_attn(
            x1, x1, x1,
            need_weights=False,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.drop1(sa_out)

        # 2) cross-attn
        if ctx_tokens is not None and ctx_tokens.size(1) > 0:
            if ctx_key_padding_mask is None:
                q = self.norm2(x).to(dtype=x.dtype)
                kv = self.norm_ctx(ctx_tokens).to(dtype=ctx_tokens.dtype)

                ca_out, _ = self.cross_attn(
                    q, kv, kv,
                    need_weights=False,
                    key_padding_mask=None,
                )
                x = x + self.drop2(ca_out)
            else:
                keep_rows = ~ctx_key_padding_mask.all(dim=1)

                if keep_rows.any():
                    q = self.norm2(x[keep_rows]).to(dtype=x.dtype)
                    kv = self.norm_ctx(ctx_tokens[keep_rows]).to(dtype=ctx_tokens.dtype)
                    kpm = ctx_key_padding_mask[keep_rows]

                    ca_out, _ = self.cross_attn(
                        q, kv, kv,
                        need_weights=False,
                        key_padding_mask=kpm,
                    )

                    x = x.clone()
                    x[keep_rows] = x[keep_rows] + self.drop2(ca_out)

        # 3) mlp
        x3 = self.norm3(x).to(dtype=x.dtype)
        x = x + self.drop3(self.mlp(x3))
        return x