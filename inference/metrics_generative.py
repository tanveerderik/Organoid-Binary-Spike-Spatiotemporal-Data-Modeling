"""Generative evaluation for the Stage 3 priors.

Why not exact F1: it scores one guess against one truth and is maximised by
emitting the mode. A prior that always returns its single most likely map would
top that metric and be useless to sample from. Clip-to-clip reproducibility on
this data is ~0.49, so per-sample agreement is bounded well below 1 anyway.

Why not FID with a pretrained network: ImageNet features are meaningless for MEA
spike volumes. The VQVAE encoder is the domain feature extractor and its
reconstruction is validated, so Frechet distance is computed in ITS latent space.

Why not a learned discriminator (classifier two-sample test): the field is ~5%
sparse and a discriminator on that is a second model with its own failure modes.
Conditional retrieval below answers the same question by nearest-neighbour
ranking, needing no training and no density estimate.

The four measurements, and what each is for:

  frechet_token_latent  -- do generated samples occupy the region of latent space
      real data occupies? Reported against two references without which the
      absolute value means nothing: a real-vs-real split-half FLOOR (estimator
      noise at this sample size) and a real-vs-recon(real) ANCHOR (codec
      distortion alone, which no generation can beat).

  conditional_retrieval -- does generation respect its conditioning? Frechet is
      POOLED and therefore blind to this: a model that ignores global_ctx but
      reproduces the correct mixture over all assays scores a perfect Frechet
      distance. That is not hypothetical -- an earlier activity head emitted one
      pattern for every clip, which pooled Frechet would have called excellent.
      Retrieval asks whether a sample generated for assay A matches real clips
      from A better than clips from other assays. Chance is 1/n_assays.

  mea_statistics -- WHICH physical property is wrong. Frechet gives one number
      and no diagnosis; the sampler tuning only converged because temporal
      persistence and spatial clustering were separable and could be read off
      independently.

  masked_region_reconstruction -- inpainting quality, on the regimes where a
      ground truth for the hidden region actually exists.
"""
from typing import Dict, Optional, Sequence

import numpy as np
import torch


# ----------------------------------------------------------------------
# Frechet distance in the encoder's continuous latent space
# ----------------------------------------------------------------------

def latent_moments(feats: torch.Tensor):
    """Mean and covariance of (M, D) features, in float64."""
    f = feats.detach().to(torch.float64)
    mu = f.mean(dim=0)
    fc = f - mu
    cov = (fc.T @ fc) / max(f.shape[0] - 1, 1)
    return mu, cov


def frechet_distance(mu1, cov1, mu2, cov2, eps: float = 1e-6) -> float:
    """Frechet distance between two Gaussians.

    The matrix square root of cov1 @ cov2 is obtained via eigendecomposition of
    the symmetrised product, which avoids a scipy dependency and is stable here
    because both covariances are PSD by construction.
    """
    diff = mu1 - mu2
    d = cov1.shape[0]
    c1 = cov1 + eps * torch.eye(d, dtype=cov1.dtype, device=cov1.device)
    c2 = cov2 + eps * torch.eye(d, dtype=cov2.dtype, device=cov2.device)
    # sqrt(c1) via eigh, then eigenvalues of sqrt(c1) c2 sqrt(c1)
    ev1, V1 = torch.linalg.eigh(c1)
    s1 = V1 @ torch.diag(ev1.clamp_min(0).sqrt()) @ V1.T
    m = s1 @ c2 @ s1
    evm = torch.linalg.eigvalsh(m).clamp_min(0)
    return float(diff @ diff + torch.trace(c1) + torch.trace(c2) - 2.0 * evm.sqrt().sum())


def frechet_from_features(a: torch.Tensor, b: torch.Tensor) -> float:
    return frechet_distance(*latent_moments(a), *latent_moments(b))


def split_half_floor(feats: torch.Tensor, n_repeat: int = 5, seed: int = 0) -> float:
    """Real-vs-real Frechet across random halves.

    This is the noise floor. Any model distance must be read against it: at finite
    sample size two draws from the SAME distribution do not score zero, and how far
    from zero they score is the resolution limit of the measurement.
    """
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(n_repeat):
        perm = torch.randperm(feats.shape[0], generator=g)
        half = feats.shape[0] // 2
        vals.append(frechet_from_features(feats[perm[:half]], feats[perm[half:2 * half]]))
    return float(np.mean(vals))


# ----------------------------------------------------------------------
# Conditional retrieval -- the check Frechet cannot make
# ----------------------------------------------------------------------

def conditional_retrieval(
    gen_feats: torch.Tensor,
    gen_labels: torch.Tensor,
    real_feats: torch.Tensor,
    real_labels: torch.Tensor,
    topk: Sequence[int] = (1, 5),
) -> Dict[str, float]:
    """Can a generated sample be matched back to the condition it was generated for?

    Cosine nearest-neighbour against real clips, scored by whether the retrieved
    neighbours carry the same assay label. No training, no covariance, no density
    estimate -- so sample sparsity costs nothing here.

    Compare against ``chance``: a model that ignores conditioning lands there even
    if its pooled statistics are perfect.
    """
    a = torch.nn.functional.normalize(gen_feats.float(), dim=1)
    b = torch.nn.functional.normalize(real_feats.float(), dim=1)
    sim = a @ b.T                                        # (n_gen, n_real)
    out = {}
    maxk = min(max(topk), b.shape[0])
    idx = sim.topk(maxk, dim=1).indices
    hit = real_labels.to(idx.device)[idx] == gen_labels.to(idx.device).unsqueeze(1)
    for k in topk:
        kk = min(k, maxk)
        out[f"top{k}"] = float(hit[:, :kk].any(dim=1).float().mean())
    n_lab = int(real_labels.unique().numel())
    out["chance"] = 1.0 / max(n_lab, 1)
    out["n_conditions"] = float(n_lab)
    return out


# ----------------------------------------------------------------------
# Domain statistics -- what a reviewer in this field will look for
# ----------------------------------------------------------------------

def _avalanche_sizes(vol: torch.Tensor) -> np.ndarray:
    """Neuronal-avalanche sizes for one (T,H,W) binary volume.

    Standard MEA definition: bins with any population activity, grouped into runs
    separated by silent bins; size is the total spike count in a run. Organoid work
    cares about the shape of this distribution (criticality / power law), and it is
    a genuine test of population coordination -- an independent-Bernoulli sampler
    with correct marginals produces the wrong one.
    """
    per_t = vol.flatten(1).sum(dim=1)                    # (T,)
    active = per_t > 0
    sizes, run = [], 0.0
    for t in range(per_t.shape[0]):
        if active[t]:
            run += float(per_t[t])
        elif run > 0:
            sizes.append(run); run = 0.0
    if run > 0:
        sizes.append(run)
    return np.asarray(sizes, dtype=np.float64)


def _isi(vol: torch.Tensor) -> np.ndarray:
    """Inter-spike intervals pooled over electrodes of one (T,H,W) volume."""
    T = vol.shape[0]
    flat = vol.reshape(T, -1)                            # (T, H*W)
    out = []
    nz = flat.nonzero()
    if nz.numel() == 0:
        return np.zeros(0)
    order = nz[:, 1] * T + nz[:, 0]
    order, _ = order.sort()
    chan = torch.div(order, T, rounding_mode="floor")
    tt = order % T
    same = chan[1:] == chan[:-1]
    d = (tt[1:] - tt[:-1])[same]
    return d[d > 0].cpu().numpy().astype(np.float64)


def _burst_stats(vol: torch.Tensor, thresh_frac: float = 0.25) -> Dict[str, float]:
    """Population bursts: runs where population rate exceeds a fraction of its max."""
    per_t = vol.flatten(1).sum(dim=1).float()
    if float(per_t.max()) <= 0:
        return {"burst_rate": 0.0, "burst_mean_duration": 0.0}
    hot = per_t > (thresh_frac * float(per_t.max()))
    runs, cur = [], 0
    for v in hot.tolist():
        if v:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    return {"burst_rate": len(runs) / max(int(per_t.shape[0]), 1),
            "burst_mean_duration": float(np.mean(runs)) if runs else 0.0}


def mea_statistics(vols: torch.Tensor, lags: int = 7) -> Dict[str, float]:
    """Battery over a batch of (B,T,H,W) binary volumes."""
    v = vols.float()
    out: Dict[str, float] = {"rate": float(v.mean())}

    for k in range(1, lags + 1):
        if v.shape[1] <= k:
            continue
        a, b = v[:, k:], v[:, :-k]
        den = float(b.sum())
        out[f"persist{k}"] = float((a * b).sum() / den) if den > 0 else float("nan")

    num = den = 0.0
    for dh, dw in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        num += float((torch.roll(v, shifts=(dh, dw), dims=(2, 3)) * v).sum())
        den += float(v.sum())
    out["spatial_coact"] = num / den if den > 0 else float("nan")

    av, isi, br, bd = [], [], [], []
    for i in range(v.shape[0]):
        av.append(_avalanche_sizes(v[i]))
        isi.append(_isi(v[i]))
        b = _burst_stats(v[i]); br.append(b["burst_rate"]); bd.append(b["burst_mean_duration"])
    av = np.concatenate([a for a in av if a.size]) if any(a.size for a in av) else np.zeros(1)
    isi = np.concatenate([a for a in isi if a.size]) if any(a.size for a in isi) else np.zeros(1)
    out.update({
        "avalanche_mean": float(av.mean()), "avalanche_p90": float(np.percentile(av, 90)),
        "isi_mean": float(isi.mean()), "isi_p90": float(np.percentile(isi, 90)),
        "burst_rate": float(np.mean(br)), "burst_mean_duration": float(np.mean(bd)),
    })
    out["_avalanche_samples"] = av
    out["_isi_samples"] = isi
    return out


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic, no scipy."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.union1d(a, b)
    ca = np.searchsorted(np.sort(a), grid, side="right") / a.size
    cb = np.searchsorted(np.sort(b), grid, side="right") / b.size
    return float(np.abs(ca - cb).max())


def compare_statistics(real: Dict, gen: Dict) -> Dict[str, float]:
    """Scalar stats by RELATIVE difference; distributions by KS.

    Relative, not absolute: these quantities differ by orders of magnitude
    (rate ~0.08, avalanche_mean ~200), so a raw sum of absolute differences is
    just the avalanche term and every other property is invisible in the total.
    Each statistic is normalised by its real value so all contribute comparably,
    and the aggregate is a mean rather than a sum so it does not grow simply
    because more statistics were added.
    """
    out, rels = {}, []
    for k in sorted(real):
        if k.startswith("_"):
            continue
        rv, gv = real.get(k), gen.get(k)
        if rv is None or gv is None or np.isnan(rv) or np.isnan(gv):
            continue
        denom = max(abs(float(rv)), 1e-8)
        rel = abs(float(gv) - float(rv)) / denom
        out[f"rel_{k}"] = rel
        rels.append(rel)
    out["ks_avalanche"] = ks_distance(real["_avalanche_samples"], gen["_avalanche_samples"])
    out["ks_isi"] = ks_distance(real["_isi_samples"], gen["_isi_samples"])
    # Headline: mean relative error over the scalar battery, plus the two
    # distribution-shape terms, which are already on a 0-1 scale.
    ks = [v for v in (out["ks_avalanche"], out["ks_isi"]) if not np.isnan(v)]
    out["stat_error"] = float(np.mean(rels)) if rels else float("nan")
    out["stat_error_with_ks"] = float(np.mean(rels + ks)) if rels else float("nan")
    return out
