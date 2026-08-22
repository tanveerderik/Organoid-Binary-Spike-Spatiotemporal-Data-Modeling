"""Coupled point-process GLM -- Pillow et al., Nature 454:995-999, 2008.

    "Spatio-temporal correlations and visual signalling in a complete
     neuronal population"

The canonical generative model for multi-electrode spike data: each channel's
instantaneous firing probability is a nonlinear function of its own recent
spiking (the refractory / history filter) and its neighbours' recent spiking
(the coupling filters), on top of a static baseline.

    logit p(t,h,w) = b(h,w | assay) + c(gct, lct) + sum_{k>=1, dh, dw}
                     K[k,dh,dw] * y(t-k, h+dh, w+dw)

## What the sparsity forces

Pillow et al. fit one history filter and N-1 coupling filters *per neuron*, for
N ~ 27. Here N = 26,880 and the whole training set holds ~186k spikes, so a
per-electrode fit has no data and a full coupling matrix has 7.2e8 filters. The
model is therefore **translation-invariant**: a single kernel `K[k,dh,dw]`
shared across electrodes, which is exactly a causal 3-D convolution. Its
`dh=dw=0` column is the history/refractory filter and the rest are the coupling
filters, so nothing is dropped -- the filters are tied across space rather than
estimated separately. Per-electrode heterogeneity lives in `b(h,w|assay)`, fitted
from that assay's train marginals.

This tie is what makes the model estimable at 1.55e-4, and it is stated in the
paper as a consequence of the recording regime.

## Why this baseline is the real competition

Unlike the Dichotomized Gaussian, a GLM is a genuine *dynamical* model: it can
express refractoriness (a negative `K[1,0,0]`) and propagating activity, which
is where the DG structurally fails -- a latent-Gaussian correlation kernel with
a refractory dip is not positive-definite. Whatever margin the pipeline holds
over this one is a margin over the field's standard mechanistic account.

## Fitting and sampling

Fitted by maximum Bernoulli likelihood, teacher-forced on real history.
Generation is free-running and sequential over the 48 time bins, so the model
meets its own samples -- the exposure-bias gap the pipeline addresses in Stage
4C. Runaway self-excitation is the classic failure of a sampled GLM; the
per-step rate is clamped and the realised spike count is reported, so a
divergent fit is visible rather than silently producing a dense volume.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.protocol import BaselineMeta, ConditioningBatch, SpikeVolumeBaseline
from ..registry import register


class _GLMNet(nn.Module):
    """Causal conv kernel + context offset. The site baseline is supplied outside."""

    def __init__(self, n_lags: int, radius: int, ctx_dim: int = 73):
        super().__init__()
        self.n_lags = int(n_lags)
        self.radius = int(radius)
        k = 2 * radius + 1
        # (out=1, in=1, lag, dh, dw). Zero init: the model starts as the
        # inhomogeneous-Poisson null and has to earn every coupling weight.
        self.kernel = nn.Parameter(torch.zeros(1, 1, n_lags, k, k))
        self.ctx = nn.Sequential(
            nn.Linear(ctx_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.bias = nn.Parameter(torch.zeros(1))

    def drive(self, y_hist: torch.Tensor) -> torch.Tensor:
        """y_hist (B,1,n_lags+T,H,W) -> coupling drive (B,T,H,W).

        The temporal axis is *not* padded here: the caller supplies `n_lags`
        bins of history before the first predicted bin, so `conv3d` with no
        temporal padding is causal by construction -- output t sees inputs
        t-n_lags .. t-1 and never t itself.
        """
        r = self.radius
        return F.conv3d(y_hist, self.kernel, padding=(0, r, r)).squeeze(1)

    def ctx_offset(self, gct, lct) -> torch.Tensor:
        return self.ctx(torch.cat([gct, lct], dim=1)).squeeze(1)


class CoupledGLM(SpikeVolumeBaseline):

    def __init__(
        self,
        *,
        n_lags: int = 8,
        radius: int = 3,
        smooth_sites: float = 20.0,
        lr: float = 3e-2,
        epochs: int = 3,
        max_rate: float = 5e-3,
        device: str = "cuda",
    ):
        self.n_lags, self.radius = int(n_lags), int(radius)
        self.smooth_sites = float(smooth_sites)
        self.lr, self.epochs = float(lr), int(epochs)
        self.max_rate = float(max_rate)
        self.device = device

        self.net: Optional[_GLMNet] = None
        self.site_logit: Dict[int, torch.Tensor] = {}
        self.global_site: Optional[torch.Tensor] = None
        self.shape: Optional[tuple] = None
        self.fit_report: Dict[str, Any] = {}
        self._last_intensity: Optional[torch.Tensor] = None

        self.meta = BaselineMeta(
            name="coupled_glm",
            citation="Pillow, Shlens, Paninski, Sher, Litke, Chichilnisky & Simoncelli, Nature 454:995-999, 2008",
            family="voxel",
            conditioning=(
                "gct+lct through an MLP producing a per-clip log-rate offset; "
                "gct additionally via the per-assay site baseline"
            ),
            notes=(
                "Translation-invariant coupling: one shared kernel K[lag,dh,dw] "
                "instead of per-neuron filters, which a 26,880-channel array at "
                "1.55e-4 rate cannot support. K[:,0,0] is the history filter."
            ),
        )

    # ------------------------------------------------------------------
    def _site_logits(self, batch, device):
        rows = []
        for a in batch["assay_idx"].tolist():
            m = self.site_logit.get(int(a), self.global_site)
            rows.append(m)
        return torch.stack(rows).to(device)

    def fit(self, clips: Iterator[Dict[str, Any]], *, device: str = "cuda") -> None:
        self.device = device
        cached = list(clips)
        if not cached:
            raise RuntimeError("CoupledGLM.fit saw no clips")

        # -- per-electrode baseline from train marginals ----------------
        site_sum: Dict[int, torch.Tensor] = {}
        site_n: Dict[int, int] = {}
        for batch in cached:
            x = batch["x"]
            if x.dim() == 5:
                x = x.squeeze(1)
            v = (x > 0.5).float()
            self.shape = tuple(v.shape[1:])
            for i in range(v.shape[0]):
                a = int(batch["assay_idx"][i])
                site_sum[a] = site_sum.get(a, torch.zeros(v.shape[2:])) + v[i].sum(0)
                site_n[a] = site_n.get(a, 0) + v.shape[1]
        for a, cnt in site_sum.items():
            n = float(site_n[a])
            pm = float(cnt.sum()) / max(n * cnt.numel(), 1.0)
            p = ((cnt + self.smooth_sites * pm) / (n + self.smooth_sites)).clamp(1e-9, 1 - 1e-6)
            self.site_logit[a] = torch.log(p / (1 - p))
        self.global_site = torch.stack(list(self.site_logit.values())).mean(0)

        # -- maximum likelihood on the coupling kernel ------------------
        ctx_dim = cached[0]["global_ctx"].shape[1] + cached[0]["local_ctx"].shape[1]
        self.net = _GLMNet(self.n_lags, self.radius, ctx_dim).to(device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        hist = []
        for ep in range(self.epochs):
            tot, nb = 0.0, 0
            for batch in cached:
                x = batch["x"]
                if x.dim() == 5:
                    x = x.squeeze(1)
                v = (x > 0.5).float().to(device)
                B, T, H, W = v.shape
                base = self._site_logits(batch, device).unsqueeze(1)     # (B,1,H,W)
                off = self.net.ctx_offset(batch["global_ctx"].to(device),
                                          batch["local_ctx"].to(device)).view(B, 1, 1, 1)

                pad = torch.zeros(B, 1, self.n_lags, H, W, device=device)
                y_hist = torch.cat([pad, v.unsqueeze(1)], dim=2)[:, :, :-1]
                logits = self.net.drive(y_hist) + base + off
                loss = F.binary_cross_entropy_with_logits(logits, v)

                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                tot += float(loss); nb += 1
            hist.append(tot / max(nb, 1))
            print(f"  glm epoch {ep+1}/{self.epochs}  bce={hist[-1]:.6e}")

        k = self.net.kernel.detach().cpu()[0, 0]
        self.fit_report = {
            "bce_per_epoch": hist,
            "n_clips": sum(b["x"].shape[0] for b in cached),
            "n_lags": self.n_lags, "radius": self.radius,
            "history_filter": k[:, self.radius, self.radius].tolist(),
            "kernel_absmax": float(k.abs().max()),
            "coupling_mean_by_lag": k.mean(dim=(1, 2)).tolist(),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate(self, cond: ConditioningBatch, generator):
        device = cond.global_ctx.device
        T, H, W = cond.shape
        B = cond.batch_size
        self.net = self.net.to(device)

        rows = [self.site_logit.get(int(a), self.global_site) for a in cond.assay_idx.tolist()]
        base = torch.stack(rows).to(device).unsqueeze(1)
        off = self.net.ctx_offset(cond.global_ctx, cond.local_ctx).view(B, 1, 1, 1)
        max_logit = math.log(self.max_rate / (1 - self.max_rate))

        y = torch.zeros(B, 1, self.n_lags + T, H, W, device=device)
        inten = torch.zeros(B, T, H, W, device=device)
        for t in range(T):
            win = y[:, :, t:t + self.n_lags]
            logit = self.net.drive(win)[:, 0] + base[:, 0] + off[:, 0]
            logit = logit.clamp(max=max_logit)
            inten[:, t] = logit
            p = torch.sigmoid(logit)
            u = torch.rand(p.shape, generator=generator, device="cpu").to(device)
            y[:, 0, self.n_lags + t] = (u < p).float()
        return y[:, 0, self.n_lags:], inten

    @torch.no_grad()
    def sample(self, cond: ConditioningBatch, *, generator=None) -> torch.Tensor:
        v, inten = self._generate(cond, generator)
        self._last_intensity = inten
        return v

    @torch.no_grad()
    def sample_intensity(self, cond: ConditioningBatch, *, generator=None):
        """Log-odds field from the free-running rollout.

        Returned from the same trajectory `sample()` produced when available, so
        the calibrated row re-thresholds the *same* realisation rather than an
        independent one -- otherwise the two rows would differ by sampling noise
        as well as by threshold.
        """
        if self._last_intensity is not None:
            out, self._last_intensity = self._last_intensity, None
            return out
        return self._generate(cond, generator)[1]

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "site_logit": {int(k): v.cpu() for k, v in self.site_logit.items()},
            "global_site": self.global_site.cpu(),
            "shape": self.shape, "n_lags": self.n_lags, "radius": self.radius,
            "ctx_dim": self.net.ctx[0].in_features,
            "fit_report": self.fit_report,
        }, path)
        path.with_suffix(".fit.json").write_text(
            json.dumps(self.fit_report, indent=2, default=float))

    def load(self, path: Path, *, device: str = "cuda") -> None:
        d = torch.load(Path(path), map_location="cpu")
        self.n_lags, self.radius = d["n_lags"], d["radius"]
        self.net = _GLMNet(self.n_lags, self.radius, d["ctx_dim"])
        self.net.load_state_dict(d["state_dict"])
        self.net = self.net.to(device).eval()
        self.site_logit = {int(k): v for k, v in d["site_logit"].items()}
        self.global_site = d["global_site"]
        self.shape = d["shape"]
        self.fit_report = d.get("fit_report", {})
        self.device = device


@register("glm")
def _build(**kw) -> CoupledGLM:
    return CoupledGLM(**kw)
