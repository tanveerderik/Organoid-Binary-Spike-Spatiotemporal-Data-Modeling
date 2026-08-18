#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Null-baseline statistics for interpreting the reported metrics.

Every headline number in this project is a raw score with no reference point.
A tolerance-F1 of 0.79 on a 1024-cell token grid with ~50 active tokens and a
+/-1 token neighbourhood is close to what a density- and support-matched random
predictor achieves, so the raw value alone cannot support a claim. This module
computes, in one pass over the TRAINING loader, the statistics that the null
models need:

  * active-token count mean / MAD / std, globally and per assay
    -> the constant-count predictor null
  * a monotone map from log mean firing density (local context feature 0) to
    active-token count
    -> the closed-form density null, which is the trivial solution a reviewer
       will construct, since total spike count is analytically recoverable from
       the conditioning vector
  * per-assay token activation frequency over the (Ttok, Htok, Wtok) grid
    -> the N0/N1/N2 localisation null ladder
  * per-assay per-electrode empirical firing-rate map
    -> the independent-Bernoulli generation surrogate

Nothing here is fit on validation or test data.
"""

import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..utils.constants import ACTIVITY_CTX_NAMES

try:
    from sklearn.isotonic import IsotonicRegression
except Exception:  # pragma: no cover - optional dependency
    IsotonicRegression = None


NULL_BASELINE_VERSION = 1


def _mad(values: np.ndarray) -> float:
    """Mean absolute deviation about the mean.

    This is exactly the test-set MAE of a constant predictor that always emits
    the training mean, which is the reference the count head must beat.
    """
    if values.size == 0:
        return 0.0
    return float(np.abs(values - values.mean()).mean())


@torch.no_grad()
def build_null_baselines(
    loader,
    model,
    *,
    device: str = "cuda",
    save_path: str = "ckpts/null_baselines.pkl",
    num_passes: int = 1,
    max_batches: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Collect null-model statistics from the training loader.

    ``model`` is used only for its patch geometry and blank-mask rule; no
    parameters are read, so this is valid before or after training.
    """
    model.eval()
    patch_size = tuple(int(v) for v in model.patch_size)

    counts: list[int] = []
    assays: list[int] = []
    log_density: list[float] = []

    token_hist: Dict[int, np.ndarray] = {}
    token_batches: Dict[int, int] = {}
    rate_sum: Dict[int, np.ndarray] = {}
    rate_frames: Dict[int, float] = {}

    grid = None
    seen = 0

    for _pass in range(int(num_passes)):
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break

            x = batch["x"].to(device, non_blocking=True).float()
            local_ctx = batch["local_ctx"].float().cpu().numpy()
            assay_idx = np.asarray(batch["assay_idx"]).reshape(-1).astype(np.int64)

            batch_size, _, frames, height, width = x.shape
            grid = (
                frames // patch_size[0],
                height // patch_size[1],
                width // patch_size[2],
            )

            blank = model.compute_blank_mask(x, grid)          # (B,N) bool
            active = (~blank).view(batch_size, *grid)          # (B,Tt,Ht,Wt)
            active_np = active.detach().cpu().numpy()

            per_sample_counts = active_np.reshape(batch_size, -1).sum(axis=1)

            # Per-electrode firing rate, in full (pre-pad) assay coordinates.
            rate = x.mean(dim=2)[:, 0].detach().cpu().numpy()   # (B,H,W)

            for b in range(batch_size):
                a = int(assay_idx[b])
                counts.append(int(per_sample_counts[b]))
                assays.append(a)
                log_density.append(float(local_ctx[b, 0]))

                if a not in token_hist:
                    token_hist[a] = np.zeros(grid, dtype=np.float64)
                    token_batches[a] = 0
                token_hist[a] += active_np[b].astype(np.float64)
                token_batches[a] += 1

                if a not in rate_sum:
                    rate_sum[a] = np.zeros((height, width), dtype=np.float64)
                    rate_frames[a] = 0.0
                rate_sum[a] += rate[b].astype(np.float64)
                rate_frames[a] += 1.0

            seen += batch_size

            if verbose and (batch_index % 25 == 0):
                print(f"[null_baselines] pass {_pass + 1} batch {batch_index} samples={seen}")

    if seen == 0:
        raise RuntimeError("build_null_baselines saw no samples.")

    counts_arr = np.asarray(counts, dtype=np.float64)
    assays_arr = np.asarray(assays, dtype=np.int64)
    log_density_arr = np.asarray(log_density, dtype=np.float64)

    # ---- constant-count null ------------------------------------------------
    count_stats = {
        "global": {
            "mean": float(counts_arr.mean()),
            "median": float(np.median(counts_arr)),
            "mad_about_mean": _mad(counts_arr),
            "std": float(counts_arr.std()),
            "n": int(counts_arr.size),
        },
        "per_assay": {},
    }
    for a in np.unique(assays_arr):
        sel = counts_arr[assays_arr == a]
        count_stats["per_assay"][int(a)] = {
            "mean": float(sel.mean()),
            "median": float(np.median(sel)),
            "mad_about_mean": _mad(sel),
            "std": float(sel.std()),
            "n": int(sel.size),
        }

    # ---- closed-form density null ------------------------------------------
    density_fit = _fit_density_to_count(log_density_arr, counts_arr)
    density_fit_per_assay = {}
    for a in np.unique(assays_arr):
        mask = assays_arr == a
        if int(mask.sum()) >= 8:
            density_fit_per_assay[int(a)] = _fit_density_to_count(
                log_density_arr[mask], counts_arr[mask]
            )

    # ---- localisation null ladder ------------------------------------------
    token_frequency = {
        int(a): (token_hist[a] / max(token_batches[a], 1)).astype(np.float32)
        for a in token_hist
    }

    # ---- generation surrogate ----------------------------------------------
    firing_rate_map = {
        int(a): (rate_sum[a] / max(rate_frames[a], 1.0)).astype(np.float32)
        for a in rate_sum
    }

    payload = {
        "version": NULL_BASELINE_VERSION,
        "feature_names": list(ACTIVITY_CTX_NAMES),
        "token_grid": tuple(int(v) for v in grid),
        "patch_size": patch_size,
        "samples_seen": int(seen),
        "count_stats": count_stats,
        "density_to_count": density_fit,
        "density_to_count_per_assay": density_fit_per_assay,
        "token_frequency": token_frequency,
        "firing_rate_map": firing_rate_map,
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as handle:
            pickle.dump(payload, handle)
        if verbose:
            print(f"[null_baselines] saved -> {save_path}")

    if verbose:
        g = count_stats["global"]
        print(
            "[null_baselines] active-token count: "
            f"mean={g['mean']:.2f}  MAD(constant-predictor MAE)={g['mad_about_mean']:.2f}  "
            f"std={g['std']:.2f}  n={g['n']}"
        )
        print(
            "[null_baselines] density->count fit: "
            f"kind={density_fit['kind']}  train_mae={density_fit['train_mae']:.2f}"
        )

    return payload


def _fit_density_to_count(log_density: np.ndarray, counts: np.ndarray) -> Dict[str, Any]:
    """Monotone map from log mean firing density to active-token count.

    Total spike count is ``exp(ctx[0]) * T * H * W`` by construction, so this is
    the closed-form solution available to anyone who reads the conditioning
    vector. Isotonic regression is used when available because the relationship
    is monotone but not linear; a least-squares line is the fallback.
    """
    order = np.argsort(log_density)
    x_sorted = log_density[order]
    y_sorted = counts[order]

    if IsotonicRegression is not None and x_sorted.size >= 8:
        model = IsotonicRegression(increasing=True, out_of_bounds="clip")
        model.fit(x_sorted, y_sorted)
        predicted = model.predict(log_density)
        return {
            "kind": "isotonic",
            "x": x_sorted.astype(np.float32),
            "y": model.predict(x_sorted).astype(np.float32),
            "train_mae": float(np.abs(predicted - counts).mean()),
        }

    slope, intercept = np.polyfit(log_density, counts, 1)
    predicted = slope * log_density + intercept
    return {
        "kind": "linear",
        "slope": float(slope),
        "intercept": float(intercept),
        "train_mae": float(np.abs(predicted - counts).mean()),
    }


def predict_count_from_density(fit: Dict[str, Any], log_density) -> np.ndarray:
    """Apply a stored density->count fit."""
    values = np.asarray(log_density, dtype=np.float64).reshape(-1)
    if fit.get("kind") == "isotonic":
        return np.interp(
            values,
            np.asarray(fit["x"], dtype=np.float64),
            np.asarray(fit["y"], dtype=np.float64),
        )
    return float(fit["slope"]) * values + float(fit["intercept"])


def load_null_baselines(path: str = "ckpts/null_baselines.pkl") -> Dict[str, Any]:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if int(payload.get("version", -1)) != NULL_BASELINE_VERSION:
        raise RuntimeError(
            f"null baseline file version {payload.get('version')} != "
            f"expected {NULL_BASELINE_VERSION}; rebuild it."
        )
    return payload


# ---------------------------------------------------------------------------
# Motif-prior nulls
# ---------------------------------------------------------------------------

MOTIF_NULL_VERSION = 1


@torch.no_grad()
def build_motif_null_baselines(
    loader,
    vqvae,
    *,
    K1: int,
    K2: int,
    device: str = "cuda",
    save_path: str = "ckpts/motif_null_baselines.pkl",
    max_batches: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Empirical motif statistics for null comparison, from TRAINING data only.

    Accumulates, over active tokens:
      * z1 code counts, globally / per assay / per (assay, token position)
      * alpha sums, at the same three levels

    The per-(assay, position) level is the strong null: "what motif usually
    occupies this latent location in this preparation". With ~930 training
    samples and ~85 active tokens each, a 31 x 1024 table averages only a few
    observations per cell, so predictions from it must be hierarchically
    smoothed toward the per-assay and global levels (see
    ``motif_null_predictions``). Unsmoothed per-cell counts would be noise.
    """
    from ..training.train_prior import _vq_codes_alpha_and_pmask

    vqvae.eval()
    z1_global = np.zeros(K1, dtype=np.float64)
    a_global = np.zeros(K2, dtype=np.float64)
    n_global = 0.0

    z1_assay: Dict[int, np.ndarray] = {}
    a_assay: Dict[int, np.ndarray] = {}
    n_assay: Dict[int, float] = {}

    z1_pos: Dict[int, np.ndarray] = {}
    a_pos: Dict[int, np.ndarray] = {}
    n_pos: Dict[int, np.ndarray] = {}

    n_tokens = None

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break

        x = batch["x"].to(device, non_blocking=True).float()
        global_ctx = batch["global_ctx"].to(device).float()
        local_ctx = batch["local_ctx"].to(device).float()
        assay_idx = np.asarray(batch["assay_idx"]).reshape(-1).astype(np.int64)

        codes, alpha, _pmask, _grid = _vq_codes_alpha_and_pmask(
            vqvae, x, global_ctx, local_ctx, None, device
        )
        active = codes[..., 0].ge(0)
        z1 = codes[..., 0].clamp_min(0)

        if n_tokens is None:
            n_tokens = int(codes.shape[1])

        z1_np = z1.cpu().numpy()
        alpha_np = alpha.float().cpu().numpy()
        active_np = active.cpu().numpy()

        for b in range(x.shape[0]):
            a_id = int(assay_idx[b])
            if a_id not in z1_assay:
                z1_assay[a_id] = np.zeros(K1, dtype=np.float64)
                a_assay[a_id] = np.zeros(K2, dtype=np.float64)
                n_assay[a_id] = 0.0
                z1_pos[a_id] = np.zeros((n_tokens, K1), dtype=np.float64)
                a_pos[a_id] = np.zeros((n_tokens, K2), dtype=np.float64)
                n_pos[a_id] = np.zeros(n_tokens, dtype=np.float64)

            positions = np.nonzero(active_np[b])[0]
            if positions.size == 0:
                continue
            codes_here = z1_np[b, positions]
            alphas_here = alpha_np[b, positions]

            np.add.at(z1_global, codes_here, 1.0)
            np.add.at(z1_assay[a_id], codes_here, 1.0)
            np.add.at(z1_pos[a_id], (positions, codes_here), 1.0)

            a_global += alphas_here.sum(axis=0)
            a_assay[a_id] += alphas_here.sum(axis=0)
            np.add.at(a_pos[a_id], positions, alphas_here)

            np.add.at(n_pos[a_id], positions, 1.0)
            n_assay[a_id] += float(positions.size)
            n_global += float(positions.size)

        if verbose and batch_index % 25 == 0:
            print(f"[motif_nulls] batch {batch_index}", flush=True)

    payload = {
        "version": MOTIF_NULL_VERSION,
        "K1": int(K1),
        "K2": int(K2),
        "Ntok": int(n_tokens),
        "active_token_observations": float(n_global),
        "z1_global": z1_global,
        "alpha_global": a_global,
        "n_global": n_global,
        "z1_assay": z1_assay,
        "alpha_assay": a_assay,
        "n_assay": n_assay,
        "z1_pos": z1_pos,
        "alpha_pos": a_pos,
        "n_pos": n_pos,
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as handle:
            pickle.dump(payload, handle)
        if verbose:
            print(f"[motif_nulls] saved -> {save_path}")

    if verbose:
        per_cell = n_global / max(len(z1_assay) * max(n_tokens or 1, 1), 1)
        print(
            f"[motif_nulls] {n_global:.0f} active-token observations across "
            f"{len(z1_assay)} assays; mean {per_cell:.2f} per (assay, position) cell"
        )
    return payload


def motif_null_predictions(
    payload: Dict[str, Any],
    assay_idx,
    positions: np.ndarray,
    *,
    level: str,
    smoothing: float = 4.0,
):
    """Return (z1 probability table, alpha estimate) for one null level.

    level: 'uniform' | 'global' | 'assay' | 'assay_position'

    'assay_position' is hierarchically smoothed toward 'assay' and then
    'global' with pseudo-count ``smoothing``, because the per-cell tables are
    sparse. Without smoothing the strong null would be dominated by cells seen
    once or not at all.
    """
    K1, K2 = int(payload["K1"]), int(payload["K2"])
    m = positions.shape[0]

    z1_global = payload["z1_global"] + 1e-9
    p_global = z1_global / z1_global.sum()
    alpha_global = payload["alpha_global"] / max(payload["n_global"], 1.0)

    if level == "uniform":
        return (
            np.tile(np.full(K1, 1.0 / K1), (m, 1)),
            np.tile(np.full(K2, 1.0 / K2), (m, 1)),
        )
    if level == "global":
        return np.tile(p_global, (m, 1)), np.tile(alpha_global, (m, 1))

    assay_idx = np.asarray(assay_idx).reshape(-1)
    z1_out = np.empty((m, K1), dtype=np.float64)
    alpha_out = np.empty((m, K2), dtype=np.float64)

    for i in range(m):
        a_id = int(assay_idx[i])
        if a_id in payload["z1_assay"]:
            counts_a = payload["z1_assay"][a_id]
            n_a = payload["n_assay"][a_id]
            p_assay = (counts_a + smoothing * p_global) / (n_a + smoothing)
            alpha_assay = (
                payload["alpha_assay"][a_id] + smoothing * alpha_global
            ) / (n_a + smoothing)
        else:
            p_assay, alpha_assay = p_global, alpha_global

        if level == "assay":
            z1_out[i], alpha_out[i] = p_assay, alpha_assay
            continue

        pos = int(positions[i])
        if a_id in payload["z1_pos"]:
            counts_p = payload["z1_pos"][a_id][pos]
            n_p = payload["n_pos"][a_id][pos]
            z1_out[i] = (counts_p + smoothing * p_assay) / (n_p + smoothing)
            alpha_out[i] = (
                payload["alpha_pos"][a_id][pos] + smoothing * alpha_assay
            ) / (n_p + smoothing)
        else:
            z1_out[i], alpha_out[i] = p_assay, alpha_assay

    z1_out /= z1_out.sum(axis=1, keepdims=True).clip(1e-12)
    alpha_out /= alpha_out.sum(axis=1, keepdims=True).clip(1e-12)
    return z1_out, alpha_out


def load_motif_null_baselines(path: str = "ckpts/motif_null_baselines.pkl") -> Dict[str, Any]:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if int(payload.get("version", -1)) != MOTIF_NULL_VERSION:
        raise RuntimeError(
            f"motif null file version {payload.get('version')} != "
            f"{MOTIF_NULL_VERSION}; rebuild it."
        )
    return payload
