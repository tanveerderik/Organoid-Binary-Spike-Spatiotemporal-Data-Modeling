"""Single-level 3D VQ tokenizer for the MaskGIT-flat baseline.

Deliberately NOT the pipeline's tokenizer with options disabled. This is the
plain thing the MAGVIT / ViT-VQGAN line of work describes: patch-embed the
volume, a few residual blocks, one flat codebook, mirror decoder. No ladder, no
residual level, no context losses, no spatial-bias module. If the paper is going
to claim a hierarchical alphabet earns its keep, the comparison has to be
against the ordinary version rather than against ours in a costume.

    Yu et al., "MAGVIT: Masked Generative Video Transformer", CVPR 2023
    Esser et al., "Taming Transformers" (VQGAN), CVPR 2021
    van den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE), 2017

Geometry matches the pipeline so token budgets are comparable: 48x120x224 voxels
-> an 8x8x16 grid of 1024 tokens via a (6,15,14) patch, the same patch the
pipeline uses. Codebook 1024 entries, the MAGVIT default and close to the
pipeline's V=961, so neither side wins on alphabet size.

Reconstruction is Bernoulli, not MSE: the data is binary and 99.98% zeros, so a
squared error would be minimised by predicting zero everywhere. The decoder
emits per-voxel logits, and binarisation happens downstream at a rate-calibrated
threshold -- the same treatment every other baseline gets.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH = (6, 15, 14)
GRID = (8, 8, 16)


class ResBlock3d(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.n1, self.n2 = nn.GroupNorm(8, c), nn.GroupNorm(8, c)
        self.c1 = nn.Conv3d(c, c, 3, padding=1)
        self.c2 = nn.Conv3d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class VectorQuantizer(nn.Module):
    """Straight-through VQ with EMA codebook updates and dead-code restart.

    EMA rather than a codebook loss because it is markedly more stable at this
    sparsity, and dead-code restart because an unused entry otherwise drifts to
    a large norm and never recovers -- a failure already observed in this
    project's own codebooks.
    """

    def __init__(self, n_codes: int = 1024, dim: int = 64, decay: float = 0.99,
                 eps: float = 1e-5):
        super().__init__()
        self.n_codes, self.dim, self.decay, self.eps = n_codes, dim, decay, eps
        emb = torch.randn(n_codes, dim) * 0.1
        self.register_buffer("embedding", emb)
        self.register_buffer("cluster_size", torch.zeros(n_codes))
        self.register_buffer("ema_w", emb.clone())

    def forward(self, z):                      # z: (B,D,t,h,w)
        B, D, *sp = z.shape
        flat = z.permute(0, 2, 3, 4, 1).reshape(-1, D)
        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ self.embedding.t()
             + self.embedding.pow(2).sum(1))
        idx = d.argmin(1)
        q = self.embedding[idx].view(B, *sp, D).permute(0, 4, 1, 2, 3)

        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.n_codes).type(flat.dtype)
                self.cluster_size.mul_(self.decay).add_(
                    onehot.sum(0), alpha=1 - self.decay)
                self.ema_w.mul_(self.decay).add_(
                    onehot.t() @ flat, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cs = ((self.cluster_size + self.eps)
                      / (n + self.n_codes * self.eps) * n)
                self.embedding.copy_(self.ema_w / cs.unsqueeze(1))
                dead = self.cluster_size < 1e-3
                if dead.any():
                    pick = flat[torch.randint(0, flat.shape[0], (int(dead.sum()),),
                                              device=flat.device)]
                    self.embedding[dead] = pick
                    self.ema_w[dead] = pick
                    self.cluster_size[dead] = 1.0

        commit = F.mse_loss(z, q.detach())
        q = z + (q - z).detach()               # straight-through
        return q, idx.view(B, *sp), commit

    def lookup(self, idx):                     # (B,t,h,w) -> (B,D,t,h,w)
        return self.embedding[idx].permute(0, 4, 1, 2, 3).contiguous()


class FlatVQTokenizer(nn.Module):

    def __init__(self, n_codes: int = 1024, dim: int = 64, width: int = 128,
                 n_blocks: int = 4):
        super().__init__()
        self.patch, self.grid, self.n_codes = PATCH, GRID, n_codes
        self.enc_in = nn.Conv3d(1, width, PATCH, stride=PATCH)
        self.enc = nn.Sequential(*[ResBlock3d(width) for _ in range(n_blocks)])
        self.to_z = nn.Conv3d(width, dim, 1)
        self.vq = VectorQuantizer(n_codes, dim)
        self.from_z = nn.Conv3d(dim, width, 1)
        self.dec = nn.Sequential(*[ResBlock3d(width) for _ in range(n_blocks)])
        self.dec_out = nn.ConvTranspose3d(width, 1, PATCH, stride=PATCH)

    def encode(self, x):                       # x: (B,1,T,H,W)
        z = self.to_z(self.enc(self.enc_in(x)))
        q, idx, commit = self.vq(z)
        return q, idx, commit

    def decode(self, q):
        return self.dec_out(self.dec(self.from_z(q)))    # logits (B,1,T,H,W)

    def decode_ids(self, idx):
        return self.decode(self.vq.lookup(idx))

    def forward(self, x):
        q, idx, commit = self.encode(x)
        return self.decode(q), idx, commit

    @torch.no_grad()
    def tokens(self, x) -> torch.Tensor:
        """(B,1,T,H,W) -> flat token ids (B, 1024)."""
        _, idx, _ = self.encode(x)
        return idx.reshape(idx.shape[0], -1)
