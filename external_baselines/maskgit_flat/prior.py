"""Vanilla MaskGIT prior over the flat token grid.

    Chang, Zhang, Jiang, Liu & Freeman, "MaskGIT: Masked Generative Image
    Transformer", CVPR 2022
    Yu et al., "MAGVIT: Masked Generative Video Transformer", CVPR 2023

The published recipe, unmodified: a bidirectional transformer trained to fill in
masked tokens under a cosine mask schedule, then sampled by parallel iterative
decoding in which the most confident predictions are frozen at each step.

What it deliberately does NOT have, because these are the pipeline's
contributions and the baseline exists to price them:

  * no hierarchical alphabet -- one flat codebook, no ladder, no dedupe
  * no where/what factorisation -- no separate activity prior, so nothing tells
    it which cells should be active before it decides what goes in them
  * no soft activity field, no adaptation stage
  * no motif-structured attention masking

Conditioning is by prefix token, MaskGIT's own mechanism for class conditioning,
here carrying gct and lct. That is the same information the pipeline gets, so
the comparison isolates architecture rather than context.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_schedule(r: torch.Tensor) -> torch.Tensor:
    """MaskGIT's mask-ratio schedule, in the paper's orientation.

        gamma(r) = cos(pi/2 * r)

    `r` is progress in [0,1] and the return value is the fraction of tokens
    STILL MASKED at that point: gamma(0)=1 (everything masked), gamma(1)=0
    (nothing masked), monotone decreasing in between.

    This file previously defined the complement, cos(pi/2*(1-r)). Both call
    sites compensated, so behaviour was correct, but the function did not match
    the convention in the paper -- which is the first thing anyone checking this
    implementation against MaskGIT will compare.
    """
    return torch.cos(math.pi / 2.0 * r)


class Block(nn.Module):
    def __init__(self, d, heads, drop=0.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class MaskGITPrior(nn.Module):

    def __init__(self, n_codes: int = 1024, n_tokens: int = 1024,
                 d_model: int = 256, layers: int = 6, heads: int = 8,
                 gct_dim: int = 64, lct_dim: int = 9):
        super().__init__()
        self.n_codes, self.n_tokens = n_codes, n_tokens
        self.mask_id = n_codes                      # one extra embedding row
        self.tok = nn.Embedding(n_codes + 1, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.gct = nn.Sequential(nn.Linear(gct_dim, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_model))
        self.lct = nn.Sequential(nn.Linear(lct_dim, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([Block(d_model, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_codes)

    def forward(self, ids, gct, lct):
        """ids (B,N) with mask_id in masked slots -> logits (B,N,n_codes)."""
        h = self.tok(ids) + self.pos
        prefix = torch.stack([self.gct(gct), self.lct(lct)], dim=1)   # (B,2,D)
        h = torch.cat([prefix, h], dim=1)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h[:, 2:]))

    # ------------------------------------------------------------------
    def loss(self, ids, gct, lct, generator=None):
        """Masked-token cross-entropy under the cosine schedule."""
        B, N = ids.shape
        # r ~ U(0,1) so gamma(r) sweeps the full range of mask ratios; at least
        # one token is always masked or the batch contributes no loss term.
        r = torch.rand(B, 1, device=ids.device, generator=generator)
        n_mask = (cosine_schedule(r) * N).clamp(min=1).long()          # (B,1)
        noise = torch.rand(B, N, device=ids.device, generator=generator)
        thresh = noise.sort(dim=1).values.gather(1, n_mask - 1)
        mask = noise <= thresh
        inp = torch.where(mask, torch.full_like(ids, self.mask_id), ids)
        logits = self(inp, gct, lct)
        return F.cross_entropy(logits[mask], ids[mask]), mask.float().mean()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, gct, lct, *, steps: int = 12, temperature: float = 1.0,
                 generator=None, device=None):
        """Parallel iterative decoding, MaskGIT Algorithm 1.

        Every token starts masked -- free generation, nothing about the target
        clip leaks in. At each step all masked slots are predicted, tokens are
        sampled from the logits, and the highest-confidence ones are kept
        according to the cosine schedule; the rest are re-masked for the next
        pass. Confidence carries annealed Gumbel noise, as in the paper, so
        early steps do not lock in a greedy commitment.
        """
        B = gct.shape[0]
        dev = device or gct.device
        ids = torch.full((B, self.n_tokens), self.mask_id, dtype=torch.long, device=dev)
        unknown = torch.ones(B, self.n_tokens, dtype=torch.bool, device=dev)

        for t in range(steps):
            logits = self(ids, gct, lct) / max(temperature, 1e-6)
            probs = logits.softmax(-1)
            flat = probs.reshape(-1, self.n_codes)
            # Draw on CPU and move. The harness hands every baseline the same
            # CPU generator so that one seed means the same thing across
            # methods; a CUDA generator here would silently desynchronise this
            # model's randomness from the others'.
            samp = torch.multinomial(
                flat.cpu(), 1, generator=generator).reshape(B, -1).to(dev)
            conf = probs.gather(-1, samp.unsqueeze(-1)).squeeze(-1)

            ids = torch.where(unknown, samp, ids)
            # already-known tokens must never be reconsidered
            conf = conf.masked_fill(~unknown, float("inf"))

            # Tokens still masked after this step, straight from the schedule.
            r = torch.tensor((t + 1) / steps, device=dev)
            keep_masked = int((cosine_schedule(r) * self.n_tokens).item())
            if t == steps - 1 or keep_masked <= 0:
                unknown = torch.zeros_like(unknown)
                break

            ann = temperature * (1.0 - (t + 1) / steps)
            u = torch.rand(conf.shape, generator=generator).to(dev)
            g = -torch.log(-torch.log(u + 1e-9) + 1e-9)
            score = conf + ann * g
            cut = score.sort(dim=1).values[:, keep_masked - 1:keep_masked]
            unknown = score <= cut
            ids = torch.where(unknown, torch.full_like(ids, self.mask_id), ids)

        return ids

    @torch.no_grad()
    def complete(self, ids_known, roi, gct, lct, *, steps: int = 12,
                 temperature: float = 1.0, generator=None):
        """MaskGIT decoding with the visible tokens PINNED.

        Deliberately a separate method rather than a flag on `generate`: the
        free-generation path produced every number already in the diagnostics
        table, and it must not move because completion was added.

        `roi` is (B,N) bool -- True where the token must be predicted. Known
        slots are never re-masked and never resampled. The cosine schedule is
        applied to each row's OWN unknown count, not to n_tokens, or a row
        whose hole is 30% of the grid would be declared finished after the
        first step.
        """
        dev = ids_known.device
        ids = ids_known.clone()
        ids[roi] = self.mask_id
        unknown = roi.clone()
        n_unk0 = unknown.sum(1, keepdim=True).float()

        for t in range(steps):
            if not bool(unknown.any()):
                break
            logits = self(ids, gct, lct) / max(temperature, 1e-6)
            probs = logits.softmax(-1)
            samp = torch.multinomial(
                probs.reshape(-1, self.n_codes).cpu(), 1,
                generator=generator).reshape(ids.shape).to(dev)
            conf = probs.gather(-1, samp.unsqueeze(-1)).squeeze(-1)

            ids = torch.where(unknown, samp, ids)
            conf = conf.masked_fill(~unknown, float("inf"))

            r = torch.tensor((t + 1) / steps, device=dev)
            keep = (cosine_schedule(r) * n_unk0).long()      # (B,1) per row
            if t == steps - 1:
                unknown = torch.zeros_like(unknown)
                break

            ann = temperature * (1.0 - (t + 1) / steps)
            u = torch.rand(conf.shape, generator=generator).to(dev)
            g = -torch.log(-torch.log(u + 1e-9) + 1e-9)
            score = conf + ann * g
            srt = score.sort(dim=1).values
            idx = (keep - 1).clamp(min=0, max=score.shape[1] - 1)
            cut = srt.gather(1, idx)
            unknown = (score <= cut) & (keep > 0)
            ids = torch.where(unknown, torch.full_like(ids, self.mask_id), ids)

        return ids
