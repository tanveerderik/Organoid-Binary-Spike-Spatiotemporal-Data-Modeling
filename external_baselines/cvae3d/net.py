"""A conditional VAE over the same 3-D U-Net backbone the deterministic arm uses.

Sohn, Lee & Yan, "Learning Structured Output Representation using Deep
Conditional Generative Models", NIPS 2015 -- the standard CVAE, with the
standard three networks:

    recognition  q(z | x, c)   sees the WHOLE clip; training only
    prior        p(z | c)      sees the visible remainder and the context
    generator    p(x | z, c)

`c` here is (visible volume, hole mask, gct, lct). The prior is *conditional*,
not N(0, I): at generation time the model must place a plausible clip inside a
specific hole in a specific preparation, and a fixed isotropic prior would force
the generator to absorb all of that through the skip connections, which is the
usual way a CVAE turns back into a deterministic net.

The point of this arm is narrow and worth stating. `unet3d` is trained to
predict the conditional MEAN, which is optimal for a ranking metric and a
disaster for a distributional one -- one context gives one field, so it has no
sample diversity at all. This arm has the same backbone, the same conditioning
and the same holes, and differs only by carrying a latent variable. Any gap
between the two on the avalanche/ISI/stat_error family is attributable to
stochasticity rather than to capacity or to conditioning, because everything
else is held fixed. It is also the continuous-latent answer to the question
MaskGIT-flat asks discretely: does the code need to be quantized?

The latent is a low-resolution GRID (z_ch channels at 6x15x28), not a global
vector. A single vector per clip cannot say *where* the extra spikes go, and
placement is the whole question here.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..unet3d.net import Block, CondUNet3D, _gn


class PosteriorEncoder(nn.Module):
    """q(z | x, c): three stride-2 stages over the full clip and the hole mask.

    The log-variance branch is initialised to `logvar_init`, not to zero, and
    that one number decides whether this arm works at all. At logvar 0 the first
    draw is z ~ N(0, 1) -- pure noise carrying nothing about the clip -- so the
    injection layer immediately learns to suppress z, the gradient back to this
    network dies, and the posterior never recovers. The first run collapsed
    exactly that way: val KL settled at 0.0089 nats/dim against a free-bits
    floor of 0.05, i.e. the latent declined capacity that was being offered for
    free. Starting at logvar = -4 (sigma ~ 0.135) makes z ~ mu_q, which is
    informative from step one, so the decoder can discover that reading it pays.

    Note what is NOT the cause here, because it is the usual suspect: the skip
    connections cannot be leaking the answer. They are computed from `x_vis` and
    the hole mask, so z is the only path into the decoder that carries anything
    about the hole contents.
    """

    def __init__(self, ctx_dim: int, z_ch: int,
                 widths: Tuple[int, int, int] = (32, 64, 128),
                 logvar_init: float = -4.0):
        super().__init__()
        a, b, c = widths
        self.z_ch = z_ch
        self.c1 = nn.Conv3d(2, a, 3, stride=2, padding=1)
        self.b1 = Block(a, a, ctx_dim, n_conv=1)
        self.c2 = nn.Conv3d(a, b, 3, stride=2, padding=1)
        self.b2 = Block(b, b, ctx_dim, n_conv=1)
        self.c3 = nn.Conv3d(b, c, 3, stride=2, padding=1)
        self.b3 = Block(c, c, ctx_dim, n_conv=1)
        self.out = nn.Conv3d(c, 2 * z_ch, 1)
        # Mean branch: small random, so different latent channels start
        # distinguishable. Log-variance branch: constant, see above.
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            self.out.weight[z_ch:].zero_()
            self.out.bias[z_ch:].fill_(logvar_init)

    def forward(self, x, roi, ctx):
        h = self.b1(self.c1(torch.cat([x, roi], 1)), ctx)
        h = self.b2(self.c2(h), ctx)
        h = self.b3(self.c3(h), ctx)
        return self.out(h).chunk(2, dim=1)


class CondVAE3D(nn.Module):
    """Backbone shared with `unet3d`, cut at the bottleneck to admit z."""

    def __init__(self, *, ctx_in: int = 73, ctx_dim: int = 128,
                 widths: Tuple[int, ...] = (32, 64, 128, 256),
                 z_ch: int = 4, base_logit: float = 0.0,
                 post_logvar_init: float = -4.0,
                 spatial_size: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.z_ch = z_ch
        self.unet = CondUNet3D(ctx_in=ctx_in, ctx_dim=ctx_dim, widths=widths,
                               base_logit=base_logit,
                               spatial_size=spatial_size)
        w3 = widths[-1]
        # p(z | c) reads the same bottleneck the generator will condition on,
        # so prior and generator cannot disagree about what the context is.
        self.prior_head = nn.Conv3d(w3, 2 * z_ch, 1)
        nn.init.zeros_(self.prior_head.weight)
        nn.init.zeros_(self.prior_head.bias)
        self.post = PosteriorEncoder(ctx_dim, z_ch,
                                     logvar_init=post_logvar_init)
        self.inject = nn.Sequential(_gn(w3 + z_ch),
                                    nn.Conv3d(w3 + z_ch, w3, 1))

    # -- the backbone, split so the bottleneck is addressable ------------
    def _encode(self, x_vis, roi, ctx):
        u = self.unet
        h = u.stem(torch.cat([x_vis, roi], dim=1))
        if u.pos is not None:
            h = h + u.pos
        h0 = u.e0(h, ctx)
        h1 = u.e1(u.d1(h0), ctx)
        h2 = u.e2(u.d2(h1), ctx)
        return u.mid(u.d3(h2), ctx), (h0, h1, h2)

    def _decode(self, h, skips, ctx):
        u = self.unet
        h0, h1, h2 = skips
        h = u.b2(torch.cat([u.u2(h), h2], 1), ctx)
        h = u.b1(torch.cat([u.u1(h), h1], 1), ctx)
        h = u.b0(torch.cat([u.u0(h), h0], 1), ctx)
        return u.head(h)

    def embed(self, gct, lct):
        return self.unet.embed(torch.cat([gct, lct], dim=1))

    @staticmethod
    def _kl(mq, lq, mp, lp) -> torch.Tensor:
        """Per-dimension KL(q || p) between two diagonal Gaussians."""
        return 0.5 * ((lp - lq) + (lq.exp() + (mq - mp) ** 2) / lp.exp() - 1.0)

    def forward(self, x_vis, roi, gct, lct, x_full=None, *, generator=None,
                temperature: float = 1.0):
        """Returns (logits, per-dim KL or None).

        With `x_full` this is the training path: z comes from the recognition
        network. Without it, z comes from the conditional prior -- which is the
        only path available at test time, and the only one that ever runs
        inside `sample`/`complete`.
        """
        ctx = self.embed(gct, lct)
        h, skips = self._encode(x_vis, roi, ctx)
        mp, lp = self.prior_head(h).chunk(2, dim=1)
        lp = lp.clamp(-8.0, 8.0)

        if x_full is not None:
            mq, lq = self.post(x_full, roi, ctx)
            lq = lq.clamp(-8.0, 8.0)
            eps = torch.randn(mq.shape, device=mq.device, dtype=mq.dtype)
            z = mq + (0.5 * lq).exp() * eps
            kl = self._kl(mq, lq, mp, lp)
        else:
            eps = torch.randn(mp.shape, generator=generator, device="cpu")
            z = mp + temperature * (0.5 * lp).exp() * eps.to(mp.device)
            kl = None

        h = self.inject(torch.cat([h, z], dim=1))
        return self._decode(h, skips, ctx), kl
