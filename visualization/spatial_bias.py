#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:22:57 2026

@author: derik
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.ndimage import maximum_filter


ADJ_GAP_BINS = [(1, 1), (2, 2), (3, 3), (4, 6), (7, 12), (13, 24), (25, 48)]

def _gap_bin_label(gap_idx, gap_bins=None):
    gap_bins = gap_bins or ADJ_GAP_BINS
    lo, hi = gap_bins[gap_idx]
    return f"{lo}" if lo == hi else f"{lo}-{hi}"

def detect_spatial_peaks(img, min_rel_height=0.3, footprint=5):
    """
    Detect local maxima in spatial bias map.

    min_rel_height : peak must be above this fraction of max
    footprint : size of neighborhood for peak detection
    """

    data_max = maximum_filter(img, footprint)

    peaks = (img == data_max)

    thresh = img.max() * min_rel_height

    peaks &= (img > thresh)

    ys, xs = np.where(peaks)

    return xs, ys


@torch.no_grad()
def save_assaywise_spatial_maps(
    model,
    assay_indices,
    n_assays,
    out_dir="viz_spatial_bias",
    cmap="viridis",
    save_npy=True,
    save_png=True,
    assay_codebook=None,
):
    """
    Extracts assay-conditioned spatial prior maps from the VQVAE
    and saves both .npy arrays and heatmap figures with colorbars.

    assay_codebook:
        Tensor/array of shape (n_assays, global_ctx_dim). If provided,
        this is used directly to build global_ctx, which keeps visualization
        consistent with the dataset.
    """

    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device

    if model.spatial_map_prior is None:
        raise ValueError("Model has no spatial_map_prior module.")

    t_tok, h_tok, w_tok = model.token_grid
    pT, pH, pW = model.patch_size

    # fallback only if no codebook is provided
    if assay_codebook is None:
        g = torch.Generator()
        g.manual_seed(0)

        codebook = torch.randint(
            0, 2,
            (n_assays, model.global_ctx_in_dim),
            generator=g
        ).float()

        codebook = codebook * 2.0 - 1.0
        codebook = torch.nn.functional.normalize(codebook, dim=-1)
        assay_codebook = codebook
    else:
        assay_codebook = torch.as_tensor(assay_codebook, dtype=torch.float32)

    for assay_idx in assay_indices:

        # -------- build global context --------
        gct = assay_codebook[assay_idx].to(device=device, dtype=torch.float32).unsqueeze(0)

        # -------- global embedding --------
        g_emb = model._global_emb_only(gct)

        # -------- spatial bias --------
        
        full_H, full_W = model.spatial_map_prior.full_H, model.spatial_map_prior.full_W

        runtime_H = model.token_grid[1] * model.patch_size[1]
        runtime_W = model.token_grid[2] * model.patch_size[2]
        
        pad_top = 0
        pad_left = 0
        pad_bottom = runtime_H - full_H
        pad_right = runtime_W - full_W
        
        roi_hw = [(0, 0, full_H, full_W)]
        pad_hw = [(pad_top, pad_bottom, pad_left, pad_right)]
        
        spatial = model.spatial_map_prior(
            g_emb,
            grid=model.token_grid,
            roi_hw=roi_hw,
            pad_hw=pad_hw,
        )


        hw_bias = spatial["full_hw_support"][0].detach().cpu().numpy()

        if save_npy:
            np.save(
                os.path.join(out_dir, f"assay_{assay_idx:04d}_full_pix2d_support.npy"),
                hw_bias,
            )

        img_bias = hw_bias

        xs, ys = np.array([]), np.array([])

        if hasattr(model, "memory_pix") and model.memory_pix is not None:
            try:
                gt_hw = model.memory_pix.get(
                    gct,
                    device=device,
                    dtype=torch.float32,
                )[0].detach().cpu().numpy()
        
                gt_thresh = 0.0
                if getattr(model.memory_pix, "union_mode", "ema") == "ema":
                    gt_thresh = 0.01
        
                ys, xs = np.where(gt_hw > gt_thresh)
        
                max_points = 2000
                if len(xs) > max_points:
                    rng = np.random.default_rng(0)
                    idx = rng.choice(len(xs), max_points, replace=False)
                    xs, ys = xs[idx], ys[idx]
        
            except KeyError:
                xs, ys = detect_spatial_peaks(img_bias)
        else:
            xs, ys = detect_spatial_peaks(img_bias)

        plt.figure(figsize=(6, 5))

        im = plt.imshow(
            img_bias,
            cmap=cmap,
            origin="lower",
            aspect="auto"
        )

        plt.scatter(
            xs,
            ys,
            s=12,                    # slightly larger since no fill
            facecolors="none",      # makes interior transparent
            edgecolors="lime",      # edge color
            linewidths=0.6,
            label="GT memory support"
        )

        plt.title(f"Assay {assay_idx} spatial map")
        plt.xlabel("X")
        plt.ylabel("Y")

        cbar = plt.colorbar(im)
        cbar.set_label("Spatial support")

        plt.legend(loc="upper right")
        plt.tight_layout()

        plt.savefig(
            os.path.join(out_dir, f"assay_{assay_idx:04d}_spatial_support.png"),
            dpi=200
        )

        plt.close()
        
        
@torch.no_grad()
def save_assaywise_adjacency_diagnostics(
    model,
    assay_indices,
    n_assays,
    out_dir="viz_spatial_bias",
    assay_codebook=None,
):
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device

    if not hasattr(model, "memory_adj") or model.memory_adj is None:
        raise ValueError("model.memory_adj is missing. Load Stage 0 checkpoint with memory_adj.")

    if model.spatial_map_prior is None:
        raise ValueError("Model has no spatial_map_prior.")

    if assay_codebook is None:
        g = torch.Generator()
        g.manual_seed(0)
        assay_codebook = torch.randint(
            0, 2,
            (n_assays, model.global_ctx_in_dim),
            generator=g,
        ).float()
        assay_codebook = assay_codebook * 2.0 - 1.0
        assay_codebook = torch.nn.functional.normalize(assay_codebook, dim=-1)
    else:
        assay_codebook = torch.as_tensor(assay_codebook, dtype=torch.float32)

    pred_list, tgt_list, den_list, assay_list = [], [], [], []

    full_H, full_W = model.spatial_map_prior.full_H, model.spatial_map_prior.full_W
    runtime_H = model.token_grid[1] * model.patch_size[1]
    runtime_W = model.token_grid[2] * model.patch_size[2]

    roi_hw = [(0, 0, full_H, full_W)]
    pad_hw = [(0, runtime_H - full_H, 0, runtime_W - full_W)]

    for assay_idx in assay_indices:
        gct = assay_codebook[assay_idx].to(device=device, dtype=torch.float32).unsqueeze(0)

        try:
            tgt = model.memory_adj.get(gct, device=device, dtype=torch.float32)[0]
            den = model.memory_adj.get_den(gct, device=device, dtype=torch.float32)[0]
        except KeyError:
            continue

        g_emb = model._global_emb_only(gct)

        sp = model.spatial_map_prior(
            g_emb,
            grid=model.token_grid,
            roi_hw=roi_hw,
            pad_hw=pad_hw,
        )

        pred = sp["adjacency_probs"][0]

        pred_list.append(pred.detach().cpu().numpy())
        tgt_list.append(tgt.detach().cpu().numpy())
        den_list.append(den.detach().cpu().numpy())
        assay_list.append(int(assay_idx))

    if len(pred_list) == 0:
        print("[adjacency diagnostics] No assays matched memory_adj keys; skipping.")
        return

    pred_arr = np.stack(pred_list, axis=0)
    tgt_arr = np.stack(tgt_list, axis=0)
    den_arr = np.stack(den_list, axis=0)

    gap_bins = getattr(model.memory_adj, "gap_bins", None)
    num_bins = pred_arr.shape[1]
    if gap_bins is None:
        gap_bins = [(i + 1, i + 1) for i in range(num_bins)]
    gap_bins = [(int(a), int(b)) for a, b in gap_bins]
    gap_labels = [_gap_bin_label(i, gap_bins) for i in range(num_bins)]

    np.savez(
        os.path.join(out_dir, "assaywise_adjacency_pred_vs_gt.npz"),
        pred=pred_arr,
        target=tgt_arr,
        denominator=den_arr,
        assay_indices=np.array(assay_list),
        gap_bins=np.array(gap_bins, dtype=np.int64),
    )

    fig, axes = plt.subplots(1, num_bins, figsize=(4 * num_bins, 4), squeeze=False)
    axes = axes[0]

    lim_max = max(float(pred_arr.max()), float(tgt_arr.max())) * 1.1
    lim_max = max(lim_max, 1e-4)

    for gi in range(num_bins):
        ax = axes[gi]
        ax.scatter(tgt_arr[:, gi], pred_arr[:, gi], s=35, alpha=0.8)
        ax.plot([0, lim_max], [0, lim_max], linestyle="--", linewidth=1)

        if tgt_arr.shape[0] >= 2 and np.std(tgt_arr[:, gi]) > 0 and np.std(pred_arr[:, gi]) > 0:
            r = np.corrcoef(tgt_arr[:, gi], pred_arr[:, gi])[0, 1]
        else:
            r = np.nan

        mae = np.mean(np.abs(tgt_arr[:, gi] - pred_arr[:, gi]))

        ax.set_title(f"Gap {gap_labels[gi]}: r={r:.3f}, MAE={mae:.3e}")
        ax.set_xlabel("GT memory adjacency rate")
        ax.set_ylabel("Predicted adjacency rate")
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "adjacency_scatter_pred_vs_gt.png"), dpi=200)
    plt.close()

    data, labels = [], []

    for gi in range(num_bins):
        data.append(tgt_arr[:, gi])
        labels.append(f"GT\ngap {gap_labels[gi]}")
        data.append(pred_arr[:, gi])
        labels.append(f"Pred\ngap {gap_labels[gi]}")

    plt.figure(figsize=(2.2 * num_bins, 4))
    plt.boxplot(data, labels=labels, showfliers=True)

    for gi in range(num_bins):
        x_gt = 2 * gi + 1
        x_pr = 2 * gi + 2
        for a in range(tgt_arr.shape[0]):
            plt.plot(
                [x_gt, x_pr],
                [tgt_arr[a, gi], pred_arr[a, gi]],
                linewidth=0.5,
                alpha=0.35,
            )

    plt.ylabel("Adjacency / short-gap rate")
    plt.title("Memory-bank GT vs GCT-predicted adjacency rates")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "adjacency_boxplot_gt_vs_pred.png"), dpi=200)
    plt.close()