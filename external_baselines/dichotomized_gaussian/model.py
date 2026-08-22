"""Dichotomized Gaussian baseline -- Macke et al., Neural Computation 21(2), 2009.

    "Generating spike trains with specified correlation coefficients"

The standard way to produce correlated binary spike trains with prescribed
first- and second-order structure: threshold a correlated latent Gaussian field.
It is the right statistical bar for this paper because it matches rate and
pairwise correlation *by construction* and nothing else -- so whatever the
pipeline wins by, it wins beyond second order.

## What the sparsity forces

At a voxel rate of 1.55e-4 the textbook recipe does not apply. A 26,880^2
empirical covariance is not just 2.9 GB, it is unestimable: train carries ~186k
spikes total, so the overwhelming majority of channel pairs never co-fire and
their sample covariance is noise. This implementation therefore uses the
*stationary* form -- a translation-invariant correlation kernel pooled over all
pairs at each displacement -- which is estimable, standard, and stated in the
paper as a property of the data rather than a convenience.

Three pieces, each fitted on TRAIN only:

  1. `theta(h,w | assay)` -- a per-electrode threshold from that assay's train
     marginals. Without it a stationary field would spread 200 spikes evenly
     over 26,880 electrodes, which is not a serious baseline; real activity sits
     on a minority of sites. This is the same information the pipeline's gct
     spatial-bias pretrain gets, from the same clips.
  2. `rho(dt, dh, dw)` -- a separable stationary correlation, `rho_t * rho_s`,
     fitted by inverting the DG relation at each displacement (below).
  3. `rate(lct)` -- log-rate regressed on the 9-d clip context, so the baseline
     is conditioned on exactly what the pipeline is conditioned on.

## The DG inversion

For two unit normals with correlation `lam`, thresholded at `theta`:

    P(both spike) = int_theta^inf phi(x) Phi((lam*x - theta)/sqrt(1-lam^2)) dx

which is monotone in `lam`, so the Gaussian correlation reproducing an observed
joint spike rate is a 1-D bisection. Fitting the *binary* correlation directly
into the Gaussian field would overstate the coupling badly at this sparsity.

## Sampling

The field is stationary, so it is drawn by circulant embedding: build the kernel
on the torus, take its spectrum, multiply the FFT of white noise by its square
root, transform back. 48x120x224 is ~5 MB per clip, so this is milliseconds and
needs no covariance matrix at any point. Negative spectral values (the fitted
kernel need not be PSD on the torus) are clamped -- the standard approximate
circulant-embedding fix -- and the amount clamped is recorded in the fit report,
because silently discarding a large negative part would mean the sampled field
is not the model that was fitted.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.stats import norm

from ..common.protocol import BaselineMeta, ConditioningBatch, SpikeVolumeBaseline
from ..registry import register


# ----------------------------------------------------------------------
# DG inversion
# ----------------------------------------------------------------------

def _joint_exceedance(theta: float, lam: float, theta2: float | None = None,
                      n_quad: int = 2001) -> float:
    """P(X>theta, Y>theta2) for unit normals with correlation `lam`."""
    t2 = theta if theta2 is None else theta2
    if lam <= 0.0:
        return float(norm.sf(theta) * norm.sf(t2))
    lam = min(lam, 0.999999)
    x = np.linspace(theta, theta + 12.0, n_quad)
    inner = norm.cdf((lam * x - t2) / math.sqrt(1.0 - lam * lam))
    return float(np.trapz(norm.pdf(x) * inner, x))


def _joint_exceedance_vec(th1: np.ndarray, th2: np.ndarray, w: np.ndarray,
                          lam: float, n_quad: int = 513) -> float:
    """Occurrence-weighted mean of P(X>th1_i, Y>th2_i) over a set of pairs.

    Vectorised over pairs so the heterogeneity-aware inversion stays cheap
    inside a bisection.
    """
    if lam <= 0.0:
        return float(np.sum(w * norm.sf(th1) * norm.sf(th2)) / max(w.sum(), 1e-12))
    lam = min(lam, 0.999999)
    r = math.sqrt(1.0 - lam * lam)
    u = np.linspace(0.0, 12.0, n_quad)                       # (Q,)
    x = th1[:, None] + u[None, :]                            # (P,Q)
    inner = norm.cdf((lam * x - th2[:, None]) / r)
    vals = np.trapz(norm.pdf(x) * inner, u, axis=1)          # (P,)
    return float(np.sum(w * vals) / max(w.sum(), 1e-12))


def gaussian_corr_heterogeneous(th1: np.ndarray, th2: np.ndarray,
                                w: np.ndarray, p_joint: float) -> float:
    """Latent correlation matching an observed POOLED joint rate.

    The homogeneous inversion assumes every voxel shares one threshold. Here
    they do not -- the site map spans orders of magnitude -- and that matters a
    great deal: heterogeneous marginals *by themselves* produce pooled
    co-activation above `p^2`, because the hot electrodes contribute most of
    both marginals and most of the joint. Inverting the pooled rate against a
    single mean threshold therefore attributes that heterogeneity to the latent
    coupling and double-counts it, since the sampler then applies the site map
    again.

    Concretely, the first version of this file did exactly that and produced
    persistence ~2.5x too high at every lag.
    """
    indep = _joint_exceedance_vec(th1, th2, w, 0.0)
    if not np.isfinite(p_joint) or p_joint <= indep:
        return 0.0
    if p_joint >= _joint_exceedance_vec(th1, th2, w, 0.999):
        return 0.999

    def f(lam):
        return _joint_exceedance_vec(th1, th2, w, lam) - p_joint

    try:
        return float(brentq(f, 0.0, 0.999, xtol=1e-5, maxiter=100))
    except ValueError:
        return 0.0


def gaussian_corr_for_joint(p: float, p_joint: float) -> float:
    """Latent correlation reproducing an observed joint spike rate.

    `p` is the marginal spike probability, `p_joint` the probability both
    voxels spike. Returns 0 when the pair is at or below independence, and is
    capped just short of 1 -- at p~1e-4 the achievable binary correlation is
    tiny, so the inversion routinely wants lam near its ceiling.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    theta = float(norm.isf(p))
    indep = p * p
    if not np.isfinite(p_joint) or p_joint <= indep:
        return 0.0
    ceiling = _joint_exceedance(theta, 0.999)
    if p_joint >= ceiling:
        return 0.999

    def f(lam):
        return _joint_exceedance(theta, lam) - p_joint

    try:
        return float(brentq(f, 0.0, 0.999, xtol=1e-6, maxiter=200))
    except ValueError:
        return 0.0


# ----------------------------------------------------------------------
# Baseline
# ----------------------------------------------------------------------

class DichotomizedGaussian(SpikeVolumeBaseline):

    def __init__(
        self,
        *,
        t_lags: int = 8,
        s_radius: int = 6,
        smooth_sites: float = 20.0,
        device: str = "cpu",
    ):
        self.t_lags = int(t_lags)
        self.s_radius = int(s_radius)
        self.smooth_sites = float(smooth_sites)
        self.device = device

        self.site_p: Dict[int, torch.Tensor] = {}        # assay -> (H,W) spike prob
        self.global_site: Optional[torch.Tensor] = None  # fallback (H,W)
        self.rho_t: Optional[np.ndarray] = None          # (t_lags+1,)
        self.rho_s: Optional[np.ndarray] = None          # (s_radius+1,)
        self.rate_coef: Optional[np.ndarray] = None      # (10,) lct -> log rate
        self.assay_lograte: Dict[int, float] = {}
        self.shape: Optional[tuple] = None
        self.fit_report: Dict[str, Any] = {}
        self._spectrum: Optional[torch.Tensor] = None
        self._last_intensity: Optional[torch.Tensor] = None

        self.meta = BaselineMeta(
            name="dichotomized_gaussian",
            citation="Macke, Berens, Ecker, Tolias & Bethge, Neural Computation 21(2):397-423, 2009",
            family="voxel",
            conditioning=(
                "gct via per-assay site propensity and per-assay base rate; "
                "lct via log-rate regression on all 9 clip features"
            ),
            notes=(
                "Stationary form: a full 26880^2 covariance is unestimable at "
                "1.55e-4 voxel rate. Latent correlations obtained by inverting "
                "the DG joint-exceedance relation, not by using binary "
                "correlations directly."
            ),
        )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, clips: Iterator[Dict[str, Any]], *, device: str = "cpu") -> None:
        site_sum: Dict[int, torch.Tensor] = {}
        site_n: Dict[int, int] = {}
        assay_rate: Dict[int, list] = {}

        n_vox = 0
        spikes = 0.0
        t_num = np.zeros(self.t_lags + 1)
        t_den = np.zeros(self.t_lags + 1)
        s_num = np.zeros(self.s_radius + 1)
        s_den = np.zeros(self.s_radius + 1)

        lct_rows, rate_rows = [], []
        n_clips = 0

        for batch in clips:
            x = batch["x"]
            if x.dim() == 5:
                x = x.squeeze(1)
            v = (x > 0.5).float().to(device)
            B, T, H, W = v.shape
            self.shape = (T, H, W)
            n_vox += v.numel()
            spikes += float(v.sum())

            for i in range(B):
                a = int(batch["assay_idx"][i])
                vi = v[i]
                site = vi.sum(dim=0).cpu()                 # (H,W)
                site_sum[a] = site_sum.get(a, torch.zeros_like(site)) + site
                site_n[a] = site_n.get(a, 0) + T
                r = float(vi.mean())
                assay_rate.setdefault(a, []).append(r)
                lct_rows.append(batch["local_ctx"][i].cpu().numpy())
                rate_rows.append(max(r, 1e-8))
                n_clips += 1

            # -- temporal joint-exceedance, pooled over all electrodes ----
            for k in range(self.t_lags + 1):
                if k == 0:
                    t_num[0] += float(v.sum()); t_den[0] += float(v.numel())
                elif T > k:
                    t_num[k] += float((v[:, k:] * v[:, :-k]).sum())
                    t_den[k] += float(v[:, k:].numel())

            # -- spatial joint-exceedance at each displacement radius -----
            for d in range(self.s_radius + 1):
                if d == 0:
                    s_num[0] += float(v.sum()); s_den[0] += float(v.numel())
                    continue
                acc = cnt = 0.0
                for dh, dw in ((d, 0), (-d, 0), (0, d), (0, -d)):
                    acc += float((torch.roll(v, shifts=(dh, dw), dims=(2, 3)) * v).sum())
                    cnt += float(v.numel())
                s_num[d] += acc; s_den[d] += cnt

        if not n_clips:
            raise RuntimeError("DG.fit saw no clips")

        p = spikes / max(n_vox, 1)
        self.fit_report["marginal_rate"] = p
        self.fit_report["n_clips"] = n_clips

        # -- per-electrode spike probability ---------------------------
        # Shrunk toward that assay's own mean rate with a Beta prior of
        # `smooth_sites` pseudo-observations. Most electrodes are dead, and an
        # unshrunk zero would give an infinite DG threshold there.
        for a, cnt in site_sum.items():
            n = float(site_n[a])
            p_mean = float(cnt.sum()) / max(n * cnt.numel(), 1.0)
            k = self.smooth_sites
            self.site_p[a] = ((cnt + k * p_mean) / (n + k)).clamp(1e-9, 1.0 - 1e-6)
        stacked = torch.stack(list(self.site_p.values()))
        self.global_site = stacked.mean(dim=0)
        self.assay_lograte = {a: float(np.log(max(np.mean(v), 1e-9)))
                              for a, v in assay_rate.items()}

        # -- correlation kernels via heterogeneity-aware DG inversion ---
        th_self, th_self2, w_self = self._threshold_pairs(site_sum, site_n, disp=None)
        self.rho_t = np.zeros(self.t_lags + 1)
        self.rho_t[0] = 1.0
        for k in range(1, self.t_lags + 1):
            pj = t_num[k] / max(t_den[k], 1.0)
            self.rho_t[k] = gaussian_corr_heterogeneous(th_self, th_self2, w_self, pj)

        self.rho_s = np.zeros(self.s_radius + 1)
        self.rho_s[0] = 1.0
        for d in range(1, self.s_radius + 1):
            t1, t2, wd = self._threshold_pairs(site_sum, site_n, disp=d)
            pj = s_num[d] / max(s_den[d], 1.0)
            self.rho_s[d] = gaussian_corr_heterogeneous(t1, t2, wd, pj)

        self.fit_report["independence_baseline_note"] = (
            "rho fitted against inhomogeneous marginals; a homogeneous "
            "inversion double-counts site heterogeneity as latent coupling")

        self.fit_report["rho_t"] = self.rho_t.tolist()
        self.fit_report["rho_s"] = self.rho_s.tolist()

        # -- lct -> log rate --------------------------------------------
        A = np.concatenate([np.asarray(lct_rows, dtype=np.float64),
                            np.ones((n_clips, 1))], axis=1)
        y = np.log(np.asarray(rate_rows, dtype=np.float64))
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.rate_coef = coef
        pred = A @ coef
        ss = float(((y - pred) ** 2).sum())
        st = float(((y - y.mean()) ** 2).sum())
        self.fit_report["lct_rate_r2"] = 1.0 - ss / max(st, 1e-12)

        # Build the spectrum now, not lazily at sample time, so the fit report
        # records how much negative spectral mass the circulant embedding had to
        # clamp. A large value means the sampled field is not the kernel that
        # was fitted, and that has to be visible rather than discovered later.
        self._spectrum = None
        if self.shape is not None:
            self._build_spectrum(self.shape, "cpu")
            self._spectrum = None

    def _threshold_pairs(self, site_sum, site_n, *, disp, n_bins: int = 24):
        """Binned (theta_1, theta_2, weight) pairs for the DG inversion.

        `disp=None` pairs each site with itself (the temporal case: same
        electrode at two times). `disp=d` pairs each site with its four
        neighbours at radius d (the spatial case). Both are binned -- 833k raw
        pairs inside a bisection would be far too slow, and the inversion is
        smooth in theta, so a 24x24 occupancy grid is ample.
        """
        p1, p2, wt = [], [], []
        for a, cnt in site_sum.items():
            n = float(site_n[a])
            pm = float(cnt.sum()) / max(n * cnt.numel(), 1.0)
            k = self.smooth_sites
            pr = ((cnt + k * pm) / (n + k)).clamp(1e-9, 0.5).numpy()
            if disp is None:
                p1.append(pr.ravel()); p2.append(pr.ravel())
                wt.append(np.full(pr.size, n))
            else:
                for ax, sh in ((0, disp), (0, -disp), (1, disp), (1, -disp)):
                    q = np.roll(pr, sh, axis=ax)
                    p1.append(pr.ravel()); p2.append(q.ravel())
                    wt.append(np.full(pr.size, n))
        p1 = np.concatenate(p1); p2 = np.concatenate(p2); wt = np.concatenate(wt)
        th1 = norm.isf(np.clip(p1, 1e-9, 0.5))
        th2 = norm.isf(np.clip(p2, 1e-9, 0.5))

        lo, hi = float(min(th1.min(), th2.min())), float(max(th1.max(), th2.max()))
        edges = np.linspace(lo, hi + 1e-6, n_bins + 1)
        i1 = np.clip(np.digitize(th1, edges) - 1, 0, n_bins - 1)
        i2 = np.clip(np.digitize(th2, edges) - 1, 0, n_bins - 1)
        flat = i1 * n_bins + i2
        w = np.bincount(flat, weights=wt, minlength=n_bins * n_bins)
        keep = w > 0
        ctr = 0.5 * (edges[:-1] + edges[1:])
        g1 = np.repeat(ctr, n_bins)[keep]
        g2 = np.tile(ctr, n_bins)[keep]
        return g1, g2, w[keep]

    # ------------------------------------------------------------------
    # Field sampling
    # ------------------------------------------------------------------

    def _build_spectrum(self, shape, device) -> torch.Tensor:
        """Spectrum of the separable stationary kernel on the torus."""
        T, H, W = shape
        dt = torch.arange(T, device=device)
        dt = torch.minimum(dt, T - dt).cpu().numpy()
        kt = np.interp(dt, np.arange(self.t_lags + 1), self.rho_t,
                       left=1.0, right=0.0)

        dh = torch.arange(H, device=device); dh = torch.minimum(dh, H - dh).cpu().numpy()
        dw = torch.arange(W, device=device); dw = torch.minimum(dw, W - dw).cpu().numpy()
        rad = np.sqrt(dh[:, None] ** 2 + dw[None, :] ** 2)
        ks = np.interp(rad, np.arange(self.s_radius + 1), self.rho_s,
                       left=1.0, right=0.0)

        kern = torch.from_numpy(
            (kt[:, None, None] * ks[None, :, :]).astype(np.float32)
        ).to(device)

        spec = torch.fft.rfftn(kern).real
        neg = float(spec[spec < 0].abs().sum())
        tot = float(spec.abs().sum())
        self.fit_report["spectrum_negative_fraction"] = neg / max(tot, 1e-12)
        return spec.clamp_min(0.0).sqrt()

    def _sample_field(self, B, shape, device, generator) -> torch.Tensor:
        T, H, W = shape
        if self._spectrum is None or self._spectrum.device != torch.device(device):
            self._spectrum = self._build_spectrum(shape, device)
        amp = self._spectrum
        noise = torch.randn(B, T, H, W, generator=generator,
                            device="cpu").to(device)
        f = torch.fft.rfftn(noise, dim=(1, 2, 3))
        field = torch.fft.irfftn(f * amp.unsqueeze(0), s=(T, H, W), dim=(1, 2, 3))
        std = field.flatten(1).std(dim=1).clamp_min(1e-8).view(B, 1, 1, 1)
        return field / std

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _target_rate(self, cond: ConditioningBatch) -> torch.Tensor:
        lct = cond.local_ctx.detach().cpu().numpy().astype(np.float64)
        A = np.concatenate([lct, np.ones((lct.shape[0], 1))], axis=1)
        r = np.exp(A @ self.rate_coef)
        return torch.tensor(np.clip(r, 1e-7, 1e-2), dtype=torch.float32)

    def _site_prob(self, cond: ConditioningBatch, target_rate) -> torch.Tensor:
        """Per-electrode spike probability for each clip, rescaled to its rate.

        The stored map is that assay's train marginal; multiplying it by
        (target_rate / map_mean) keeps the *shape* of the site distribution
        while matching the clip's own predicted rate.
        """
        rows = []
        for j, a in enumerate(cond.assay_idx.tolist()):
            m = self.site_p.get(int(a), self.global_site).clone()
            m = m * (float(target_rate[j]) / float(m.mean().clamp_min(1e-12)))
            rows.append(m.clamp(1e-9, 0.5))
        return torch.stack(rows)

    @torch.no_grad()
    def sample_intensity(self, cond: ConditioningBatch, *, generator=None):
        """Score from the realisation `sample()` just produced, if there is one.

        The calibrated row must re-threshold the SAME draw, not an independent
        one, or the two rows differ by sampling noise as well as by threshold --
        and the same rule is applied to every baseline.
        """
        if self._last_intensity is not None:
            out, self._last_intensity = self._last_intensity, None
            return out
        return self._draw_score(cond, generator)

    @torch.no_grad()
    def _draw_score(self, cond: ConditioningBatch, generator=None):
        """DG score: latent field minus the per-voxel threshold.

        This is the actual Dichotomized Gaussian construction for inhomogeneous
        marginals -- each voxel carries its own threshold `theta_v = Phi^-1(1-p_v)`
        and spikes when the correlated field exceeds it.

        The earlier version added a z-scored log-rate map to the field instead,
        which is NOT the DG and fails badly here: across a map where most
        electrodes are dead, log-rate has enormous variance, so the bias swamped
        the unit-variance field by tens of sigma and top-k simply selected every
        time bin of the few hottest electrodes. That produced persist1 = 0.83
        against a real 0.016, and spatial_coact of exactly zero. Thresholds are
        self-scaling: p in [1e-7, 1e-2] maps to theta in [5.2, 2.3], which is
        commensurate with a unit-variance field by construction.
        """
        device = cond.global_ctx.device
        T, H, W = cond.shape
        B = cond.batch_size
        field = self._sample_field(B, (T, H, W), device, generator)
        p_site = self._site_prob(cond, self._target_rate(cond)).to(device)  # (B,H,W)
        theta = torch.from_numpy(
            norm.isf(p_site.clamp(1e-9, 0.5).cpu().numpy())
        ).float().to(device)
        return field - theta.unsqueeze(1)

    @torch.no_grad()
    def sample(self, cond: ConditioningBatch, *, generator=None) -> torch.Tensor:
        from ..common.evaluate import binarise_at_rate
        score = self._draw_score(cond, generator)
        self._last_intensity = score
        return binarise_at_rate(score, self._target_rate(cond).to(score.device))

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "site_p": {int(k): v.cpu() for k, v in self.site_p.items()},
            "global_site": self.global_site.cpu(),
            "rho_t": self.rho_t, "rho_s": self.rho_s,
            "rate_coef": self.rate_coef,
            "assay_lograte": self.assay_lograte,
            "shape": self.shape,
            "t_lags": self.t_lags, "s_radius": self.s_radius,
            "fit_report": self.fit_report,
        }, path)
        path.with_suffix(".fit.json").write_text(
            json.dumps(self.fit_report, indent=2, default=float))

    def load(self, path: Path, *, device: str = "cpu") -> None:
        d = torch.load(Path(path), map_location="cpu")
        self.site_p = {int(k): v for k, v in d["site_p"].items()}
        self.global_site = d["global_site"]
        self.rho_t, self.rho_s = d["rho_t"], d["rho_s"]
        self.rate_coef = d["rate_coef"]
        self.assay_lograte = d["assay_lograte"]
        self.shape = d["shape"]
        self.t_lags, self.s_radius = d["t_lags"], d["s_radius"]
        self.fit_report = d.get("fit_report", {})
        self._spectrum = None


@register("dg")
def _build(**kw) -> DichotomizedGaussian:
    return DichotomizedGaussian(**kw)
