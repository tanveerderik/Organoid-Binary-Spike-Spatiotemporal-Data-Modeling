"""A conditional 3-D U-Net over the raw voxel grid. No tokenizer, no codebook.

This is deliberately the plain thing. Two input channels -- the volume with the
hole blanked, and the hole mask itself -- three stride-2 stages down, three back
up with skip connections, one logit per voxel. Conditioning enters as FiLM
(Perez et al., AAAI 2018): gct and lct are embedded once and modulate every
block's features. That is the standard way to condition a convolutional
generator and it keeps the context path identical in spirit to the prefix tokens
MaskGIT-flat gets, so the comparison is between architectures rather than
between conditioning mechanisms.

Geometry: 48x120x224 -> 24x60x112 -> 12x30x56 -> 6x15x28. Every division is
exact, so the decoder returns to the input grid with no cropping and no padding
seam. Note what this buys over every token-based arm in the comparison: the
output is full-resolution, so the model can place a spike on an individual
electrode. The tokenizers cannot -- one token covers a (6,15,14) patch, 210
electrodes -- and that ceiling is the pipeline's measured spatial bottleneck.
If a plain U-Net is going to beat us anywhere, this is where and this is why,
and the baseline is built to give it every chance.

Full resolution is also where the memory goes: one width-32 activation over
1.29M voxels is 165 MB at batch 4 in fp16. The top level therefore runs a single
convolution on each side rather than the usual pair; the doubled stack lives at
half resolution and below, where it costs eight times less.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int) -> nn.GroupNorm:
    """GroupNorm with 8 groups where 8 divides the width, the largest divisor
    below it otherwise. Every width in this file is a multiple of 8, so this is
    the identity here -- it exists for the CVAE arm, whose injection layer runs
    at w3 + z_ch channels and is not."""
    g = next(k for k in range(min(8, c), 0, -1) if c % k == 0)
    return nn.GroupNorm(g, c)


class FiLM(nn.Module):
    """Per-channel scale and shift from the context vector.

    Zero-initialised, so the block starts as the identity and the network is a
    working unconditional inpainter at step 0. Conditioning then has to earn its
    contribution rather than having to recover from a random perturbation --
    which matters here because lct is only 9 numbers against 1.29M voxels.
    """

    def __init__(self, ctx_dim: int, c: int):
        super().__init__()
        self.to_ss = nn.Linear(ctx_dim, 2 * c)
        nn.init.zeros_(self.to_ss.weight)
        nn.init.zeros_(self.to_ss.bias)

    def forward(self, h: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        s, b = self.to_ss(ctx).chunk(2, dim=1)
        v = (None,) * 3
        return h * (1.0 + s[(...,) + v]) + b[(...,) + v]


class Block(nn.Module):
    """(norm -> FiLM -> SiLU -> conv) x n_conv, with a projected residual."""

    def __init__(self, cin: int, cout: int, ctx_dim: int, n_conv: int = 2):
        super().__init__()
        self.norms = nn.ModuleList()
        self.films = nn.ModuleList()
        self.convs = nn.ModuleList()
        c = cin
        for _ in range(n_conv):
            self.norms.append(_gn(c))
            self.films.append(FiLM(ctx_dim, c))
            self.convs.append(nn.Conv3d(c, cout, 3, padding=1))
            c = cout
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, h: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        r = self.skip(h)
        for n, f, cv in zip(self.norms, self.films, self.convs):
            h = cv(F.silu(f(n(h), ctx)))
        return h + r


class CondUNet3D(nn.Module):

    def __init__(self, *, ctx_in: int = 73, ctx_dim: int = 128,
                 widths: Tuple[int, ...] = (32, 64, 128, 256),
                 base_logit: float = 0.0,
                 spatial_size: Optional[Tuple[int, int]] = None):
        super().__init__()
        w0, w1, w2, w3 = widths
        self.embed = nn.Sequential(
            nn.Linear(ctx_in, ctx_dim), nn.SiLU(), nn.Linear(ctx_dim, ctx_dim))

        # Learned spatial embedding over the electrode grid, broadcast over
        # time. WITHOUT IT THIS BASELINE CANNOT DO THE TASK, and the first run
        # proved it: correlation between the generated site map and the real
        # one was 0.0389, against 0.1414 for MaskGIT-flat and 0.3553 for the
        # pipeline.
        #
        # The reason is structural, not a matter of capacity. FiLM is
        # per-channel and spatially uniform, so a global assay code can only
        # rescale feature maps; and under free generation the input (a zeroed
        # volume and an all-ones mask) is constant, so convolution has no
        # spatial signal to work from either. There is no path by which `gct`
        # can say "this preparation fires at THESE electrodes". Adding a
        # position basis P gives one: the output can compose as
        # sum_c gamma_c(gct) . P_c(h, w), which is the same mechanism the
        # pipeline's gct mapper uses and the same one MaskGIT-flat gets from
        # its per-token positional embeddings. Omitting it was an unfair
        # handicap on the baseline, not a finding about convolutional models.
        #
        # Time is deliberately excluded: clips are arbitrary temporal crops, so
        # translation invariance along t is correct and a time embedding would
        # be fitting the crop offset. Shared across every assay, so per-assay
        # storage stays zero and the arm stays in the learned class.
        self.pos = None
        if spatial_size is not None:
            H, W = map(int, spatial_size)
            self.pos = nn.Parameter(torch.randn(1, w0, 1, H, W) * 0.02)

        self.stem = nn.Conv3d(2, w0, 3, padding=1)
        self.e0 = Block(w0, w0, ctx_dim, n_conv=1)          # full res, one conv
        self.d1 = nn.Conv3d(w0, w1, 3, stride=2, padding=1)
        self.e1 = Block(w1, w1, ctx_dim)
        self.d2 = nn.Conv3d(w1, w2, 3, stride=2, padding=1)
        self.e2 = Block(w2, w2, ctx_dim)
        self.d3 = nn.Conv3d(w2, w3, 3, stride=2, padding=1)
        self.mid = Block(w3, w3, ctx_dim)

        self.u2 = nn.ConvTranspose3d(w3, w2, 2, stride=2)
        self.b2 = Block(2 * w2, w2, ctx_dim)
        self.u1 = nn.ConvTranspose3d(w2, w1, 2, stride=2)
        self.b1 = Block(2 * w1, w1, ctx_dim)
        self.u0 = nn.ConvTranspose3d(w1, w0, 2, stride=2)
        self.b0 = Block(2 * w0, w0, ctx_dim, n_conv=1)
        self.head = nn.Conv3d(w0, 1, 1)

        # With BCE at pos_weight = 1/rate the loss-optimal logit for a voxel
        # firing at the base rate is log(w*r/(1-r)) ~ 0, so the head starts
        # already calibrated for a featureless volume and the first epochs go
        # into structure instead of into finding the offset.
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, base_logit)

    def forward(self, x_vis: torch.Tensor, roi: torch.Tensor,
                gct: torch.Tensor, lct: torch.Tensor) -> torch.Tensor:
        ctx = self.embed(torch.cat([gct, lct], dim=1))
        h = self.stem(torch.cat([x_vis, roi], dim=1))
        if self.pos is not None:
            h = h + self.pos
        h0 = self.e0(h, ctx)
        h1 = self.e1(self.d1(h0), ctx)
        h2 = self.e2(self.d2(h1), ctx)
        h = self.mid(self.d3(h2), ctx)
        h = self.b2(torch.cat([self.u2(h), h2], 1), ctx)
        h = self.b1(torch.cat([self.u1(h), h1], 1), ctx)
        h = self.b0(torch.cat([self.u0(h), h0], 1), ctx)
        return self.head(h)
