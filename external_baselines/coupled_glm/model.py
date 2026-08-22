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
        # Global DC term. It MUST appear in both the training and the generation
        # logit, or the free-running rate calibration in fit() silently adjusts
        # a parameter that nothing reads -- which it did, running the bisection
        # to its +6.0 ceiling while the sampled rate never moved.
        self.bias = nn.Parameter(torch.zeros(1))

    def logit(self, y_hist, base, ctx_off):
        """The single place log-odds are assembled, so no path can omit a term."""
        return self.drive(y_hist) + base + ctx_off + self.bias

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
        epochs: int = 8,
        max_rate: float = 5e-3,
        calib_bins: int = 32,
        calib_rounds: int = 4,
        device: str = "cuda",
    ):
        self.n_lags, self.radius = int(n_lags), int(radius)
        self.smooth_sites = float(smooth_sites)
        self.lr, self.epochs = float(lr), int(epochs)
        self.max_rate = float(max_rate)
        self.calib_bins, self.calib_rounds = int(calib_bins), int(calib_rounds)
        self.calib_corr: Optional[torch.Tensor] = None
        self.calib_binid: Optional[torch.Tensor] = None
        self._calib_map: Optional[torch.Tensor] = None
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
    @staticmethod
    def _corr_map(corr: torch.Tensor, binid: torch.Tensor) -> torch.Tensor:
        """Per-electrode additive correction, looked up by baseline-quantile bin."""
        return corr[binid]

    def _site_logits(self, batch, device):
        rows = []
        for a in batch["assay_idx"].tolist():
            m = self.site_logit.get(int(a), self.global_site)
            rows.append(m)
        return torch.stack(rows).to(device)

    def fit(self, clips: Iterator[Dict[str, Any]], *, device: str = "cuda") -> None:
        self.device = device

        # Cache SPIKE COORDINATES, not volumes. Multiple ML epochs need multiple
        # passes, but a dense float32 cache of 120 batches is ~2.5 GB, while at
        # ~200 spikes per 1.29M-voxel clip the nonzero indices are a few hundred
        # int16 triples. Same data, exactly, three orders of magnitude smaller.
        cached = []
        site_sum: Dict[int, torch.Tensor] = {}
        site_n: Dict[int, int] = {}
        for batch in clips:
            x = batch["x"]
            if x.dim() == 5:
                x = x.squeeze(1)
            v = (x > 0.5)
            self.shape = tuple(v.shape[1:])
            cached.append({
                "nz": v.nonzero().to(torch.int16),        # (S,4) b,t,h,w
                "B": int(v.shape[0]),
                "global_ctx": batch["global_ctx"].clone(),
                "local_ctx": batch["local_ctx"].clone(),
                "assay_idx": batch["assay_idx"].clone(),
            })
            vf = v.float()
            for i in range(v.shape[0]):
                a = int(batch["assay_idx"][i])
                site_sum[a] = site_sum.get(a, torch.zeros(v.shape[2:])) + vf[i].sum(0)
                site_n[a] = site_n.get(a, 0) + v.shape[1]
        if not cached:
            raise RuntimeError("CoupledGLM.fit saw no clips")
        for a, cnt in site_sum.items():
            n = float(site_n[a])
            pm = float(cnt.sum()) / max(n * cnt.numel(), 1.0)
            p = ((cnt + self.smooth_sites * pm) / (n + self.smooth_sites)).clamp(1e-9, 1 - 1e-6)
            self.site_logit[a] = torch.log(p / (1 - p))
        self.global_site = torch.stack(list(self.site_logit.values())).mean(0)

        # -- maximum likelihood on the coupling kernel ------------------
        T, H, W = self.shape
        ctx_dim = cached[0]["global_ctx"].shape[1] + cached[0]["local_ctx"].shape[1]
        self.net = _GLMNet(self.n_lags, self.radius, ctx_dim).to(device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        hist = []
        for ep in range(self.epochs):
            tot, nb = 0.0, 0
            for batch in cached:
                B = batch["B"]
                v = torch.zeros(B, T, H, W, device=device)
                nz = batch["nz"].to(device).long()
                if nz.numel():
                    v[nz[:, 0], nz[:, 1], nz[:, 2], nz[:, 3]] = 1.0
                base = self._site_logits(batch, device).unsqueeze(1)     # (B,1,H,W)
                off = self.net.ctx_offset(batch["global_ctx"].to(device),
                                          batch["local_ctx"].to(device)).view(B, 1, 1, 1)

                pad = torch.zeros(B, 1, self.n_lags, H, W, device=device)
                y_hist = torch.cat([pad, v.unsqueeze(1)], dim=2)[:, :, :-1]
                logits = self.net.logit(y_hist, base, off)
                loss = F.binary_cross_entropy_with_logits(logits, v)

                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                tot += float(loss); nb += 1
            hist.append(tot / max(nb, 1))
            print(f"  glm epoch {ep+1}/{self.epochs}  bce={hist[-1]:.6e}")

        # -- free-running calibration ------------------------------------
        # Maximum likelihood fits the filters teacher-forced, on real history.
        # Sampled free-running the model meets its own output, and because the
        # coupling weights are net positive the loop is self-suppressing: fewer
        # spikes -> less drive -> fewer still, measured at 3.7e-5 against a
        # 1.55e-4 target. Refitting under rollout would be scheduled sampling,
        # i.e. importing the pipeline's own Stage 4C contribution into the
        # baseline, so the standard point-process remedy is used instead: keep
        # the ML filters and correct the baseline so the SIMULATED marginals
        # match the TRAIN marginals. Train data only.
        #
        # A single scalar DC offset is NOT adequate here, and the first version
        # of this code used one. At these rates sigmoid(l) ~ exp(l), so a shared
        # offset multiplies every electrode's rate by the same factor -- but the
        # array holds far more cold electrodes than hot ones, so almost all of
        # the added spikes land on cold sites and the activity spreads out
        # spatially. It fixed the rate and broke everything downstream:
        # avalanches 2.6x too large and ks_isi 0.54, the worst of any baseline,
        # from a model whose refractory filter is correct.
        #
        # The correction is therefore a function of the electrode's own
        # baseline, fitted in quantile bins of it. Per-electrode would be
        # preferable but is unestimable: ~1536 simulated bins per electrode at
        # 1e-4 gives ~0.15 spikes each. 32 bins pool ~840 electrodes and are
        # comfortably estimable while preserving the shape of the site map.
        self.net.eval()
        target_rate = float(np.mean([
            b["nz"].shape[0] / max(b["B"] * T * H * W, 1) for b in cached]))
        cal = cached[: min(8, len(cached))]
        g = torch.Generator().manual_seed(0)

        base_ref = self.global_site.to(device)                    # (H,W)
        edges = torch.quantile(
            base_ref.flatten().float(),
            torch.linspace(0, 1, self.calib_bins + 1, device=device)).clone()
        edges[0] -= 1.0
        edges[-1] += 1.0
        binid = torch.bucketize(base_ref, edges[1:-1])            # (H,W) in [0,bins-1]

        # target occupancy per bin, from the TRAIN site maps
        tgt_bin = torch.zeros(self.calib_bins, device=device)
        cnt_bin = torch.zeros(self.calib_bins, device=device)
        for a, cnt in site_sum.items():
            pr = (cnt.to(device) / max(float(site_n[a]), 1.0))
            tgt_bin.index_add_(0, binid.flatten(), pr.flatten())
            cnt_bin.index_add_(0, binid.flatten(), torch.ones_like(pr.flatten()))
        tgt_bin = tgt_bin / cnt_bin.clamp_min(1.0)

        corr = torch.zeros(self.calib_bins, device=device)
        for it in range(self.calib_rounds):
            with torch.no_grad():
                self._calib_map = self._corr_map(corr, binid)
                acc = torch.zeros(T if False else 1, device=device)
                got_bin = torch.zeros(self.calib_bins, device=device)
                gcnt = torch.zeros(self.calib_bins, device=device)
                rates = []
                for b in cal:
                    cb = ConditioningBatch(
                        global_ctx=b["global_ctx"].to(device),
                        local_ctx=b["local_ctx"].to(device),
                        assay_idx=b["assay_idx"], shape=(T, H, W))
                    v, _ = self._generate(cb, g)
                    rates.append(float(v.mean()))
                    site = v.mean(dim=(0, 1))                     # (H,W)
                    got_bin.index_add_(0, binid.flatten(), site.flatten())
                    gcnt.index_add_(0, binid.flatten(),
                                    torch.ones_like(site.flatten()))
                got_bin = got_bin / gcnt.clamp_min(1.0)
            step = torch.log((tgt_bin + 1e-9) / (got_bin + 1e-9)).clamp(-3.0, 3.0)
            corr = corr + step
            print(f"  calib round {it+1}/{self.calib_rounds}: "
                  f"rate {float(np.mean(rates)):.3e} -> target {target_rate:.3e}",
                  flush=True)

        with torch.no_grad():
            self._calib_map = self._corr_map(corr, binid)
            rates = []
            for b in cal:
                cb = ConditioningBatch(
                    global_ctx=b["global_ctx"].to(device),
                    local_ctx=b["local_ctx"].to(device),
                    assay_idx=b["assay_idx"], shape=(T, H, W))
                rates.append(float(self._generate(cb, g)[0].mean()))
        got = float(np.mean(rates))
        print(f"  free-running calibration: target {target_rate:.3e}  "
              f"achieved {got:.3e}", flush=True)
        if not (0.5 * target_rate < got < 2.0 * target_rate):
            raise RuntimeError(
                f"free-running calibration failed: achieved {got:.3e} vs "
                f"target {target_rate:.3e}. The sampled rate is not tracking "
                f"the correction.")
        self.calib_corr = corr.detach().cpu()
        self.calib_binid = binid.detach().cpu()
        self.net.train()

        # conv3d index j=0 is the OLDEST lag (t-n_lags) and j=n_lags-1 the most
        # recent (t-1). Reverse both reports into lag order -- index 0 = lag 1 --
        # so "the lag-1 coefficient" means what it says. A refractory filter is
        # a negative value at index 0, and reading the raw kernel order would
        # put it at the far end and invite exactly the wrong conclusion.
        k = self.net.kernel.detach().cpu()[0, 0].flip(0)
        self.fit_report = {
            "bce_per_epoch": hist,
            "n_clips": sum(b["B"] for b in cached),
            "n_lags": self.n_lags, "radius": self.radius,
            "lag_order": "index 0 = lag 1 (most recent)",
            "history_filter_by_lag": k[:, self.radius, self.radius].tolist(),
            "coupling_mean_by_lag": k.mean(dim=(1, 2)).tolist(),
            "kernel_absmax": float(k.abs().max()),
            "refractory": float(k[0, self.radius, self.radius]),
            "train_rate_target": float(target_rate),
            "free_running_rate_achieved": float(got),
            "calibration": "per-baseline-quantile correction, "
                           f"{self.calib_bins} bins x {self.calib_rounds} rounds",
            "calib_correction_by_bin": corr.detach().cpu().tolist(),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate(self, cond: ConditioningBatch, generator):
        device = cond.global_ctx.device
        T, H, W = cond.shape
        B = cond.batch_size
        self.net = self.net.to(device)

        rows = [self.site_logit.get(int(a), self.global_site) for a in cond.assay_idx.tolist()]
        base = torch.stack(rows).to(device)
        if self._calib_map is None and self.calib_corr is not None:
            self._calib_map = self._corr_map(self.calib_corr.to(device),
                                             self.calib_binid.to(device))
        if self._calib_map is not None:
            base = base + self._calib_map.to(device).unsqueeze(0)
        base = base.unsqueeze(1)
        off = self.net.ctx_offset(cond.global_ctx, cond.local_ctx).view(B, 1, 1, 1)
        max_logit = math.log(self.max_rate / (1 - self.max_rate))

        y = torch.zeros(B, 1, self.n_lags + T, H, W, device=device)
        inten = torch.zeros(B, T, H, W, device=device)
        for t in range(T):
            win = y[:, :, t:t + self.n_lags]
            logit = self.net.logit(win, base, off)[:, 0].clamp(max=max_logit)
            inten[:, t] = logit
            p = torch.sigmoid(logit)
            u = torch.rand(p.shape, generator=generator, device="cpu").to(device)
            y[:, 0, self.n_lags + t] = (u < p).float()
        return y[:, 0, self.n_lags:], inten

    @torch.no_grad()
    def sample(self, cond: ConditioningBatch, *, generator=None) -> torch.Tensor:
        return self._generate(cond, generator)[0]

    @torch.no_grad()
    def sample_intensity(self, cond: ConditioningBatch, *, generator=None):
        """Deliberately None: this model must not get a rank-thresholded row.

        Top-k over a point-process log-intensity is dominated by the static
        per-electrode baseline, so it selects the hottest electrodes at every
        time bin and returns time-columns rather than a spike train -- measured
        at stat_error 2.98 and ks_isi 0.93, against 0.87 and 0.37 for the
        model's own samples. That is an artefact of the calibration operation,
        not a property of the GLM, and publishing it would understate the
        baseline.

        Rate matching is instead done where it belongs for a point process: the
        free-running DC offset fitted in `fit()`, so the native samples already
        sit at the train rate. The invariant the comparison holds fixed is the
        rate itself, not the mechanism used to reach it; each baseline's
        mechanism is recorded in its run_config.
        """
        return None

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "site_logit": {int(k): v.cpu() for k, v in self.site_logit.items()},
            "global_site": self.global_site.cpu(),
            "shape": self.shape, "n_lags": self.n_lags, "radius": self.radius,
            "calib_corr": None if self.calib_corr is None else self.calib_corr,
            "calib_binid": None if self.calib_binid is None else self.calib_binid,
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
        self.calib_corr = d.get("calib_corr")
        self.calib_binid = d.get("calib_binid")
        self._calib_map = None
        self.fit_report = d.get("fit_report", {})
        self.device = device


@register("glm")
def _build(**kw) -> CoupledGLM:
    return CoupledGLM(**kw)
