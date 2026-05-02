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
            s=30,
            c="white",
            edgecolors="black",
            linewidths=0.5,
            label="peaks"
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