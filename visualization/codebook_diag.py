#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:17:55 2026

@author: derik
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
from sklearn.decomposition import PCA


def load_latents(viz_root, max_blank=8000, max_active=8000):
    paths = sorted(glob.glob(os.path.join(viz_root, "**", "arrays.npz"), recursive=True))

    zb, za, l1_all = [], [], []

    for p in paths:
        d = np.load(p)

        if "z_blank" in d:
            zb.append(d["z_blank"])

        if "z_active" in d and "codes_l1_active" in d:
            za.append(d["z_active"])
            l1_all.append(d["codes_l1_active"])

    if not zb or not za:
        print("No valid latent data found.")
        return None

    zb = np.concatenate(zb, axis=0)
    za = np.concatenate(za, axis=0)
    l1 = np.concatenate(l1_all, axis=0).astype(int)

    rng = np.random.default_rng(0)

    if len(zb) > max_blank:
        idx = rng.choice(len(zb), max_blank, replace=False)
        zb = zb[idx]

    if len(za) > max_active:
        idx = rng.choice(len(za), max_active, replace=False)
        za = za[idx]
        l1 = l1[idx]

    return zb, za, l1


def plot_blank_active_tsne_l1(
    viz_root,
    out_png,
    max_active_per_l1=500,
    min_active_per_l1=1,
    max_blank=1000,
    random_state=0,
):
    data = load_latents(viz_root)
    if data is None:
        return

    zb, za, l1 = data

    zb = np.asarray(zb)
    za = np.asarray(za)
    l1 = np.asarray(l1).astype(int)

    if len(za) == 0:
        print("No active latents found for t-SNE.")
        return

    # ------------------------------------------------------------
    # Keep only L1 parents that actually appear in collected latents
    # ------------------------------------------------------------
    used_l1, counts = np.unique(l1, return_counts=True)
    keep_l1 = used_l1[counts >= min_active_per_l1]

    keep_mask = np.isin(l1, keep_l1)
    za = za[keep_mask]
    l1 = l1[keep_mask]

    if len(za) == 0:
        print("No active latents left after used-L1 filtering.")
        return

    # ------------------------------------------------------------
    # Balance active samples per used L1 parent
    # ------------------------------------------------------------
    rng = np.random.default_rng(random_state)
    selected = []

    for k in keep_l1:
        idx = np.where(l1 == k)[0]
        if len(idx) == 0:
            continue

        if max_active_per_l1 is not None and len(idx) > max_active_per_l1:
            idx = rng.choice(idx, size=max_active_per_l1, replace=False)

        selected.append(idx)

    if len(selected) == 0:
        print("No L1 groups selected for t-SNE.")
        return

    selected = np.concatenate(selected)
    za = za[selected]
    l1 = l1[selected]

    # ------------------------------------------------------------
    # Downsample blanks so they do not dominate
    # ------------------------------------------------------------
    if zb is not None and len(zb) > 0:
        if max_blank is not None and len(zb) > max_blank:
            blank_idx = rng.choice(len(zb), size=max_blank, replace=False)
            zb = zb[blank_idx]
    else:
        zb = np.zeros((0, za.shape[1]), dtype=za.dtype)

    # ------------------------------------------------------------
    # Build t-SNE input: blank latents + active latents
    # Dead codebook entries are NOT included here.
    # ------------------------------------------------------------
    X = np.concatenate([zb, za], axis=0)

    is_active = np.array(
        [False] * len(zb) + [True] * len(za),
        dtype=bool,
    )

    n_total = len(X)
    if n_total < 4:
        print(f"Too few points for t-SNE: n={n_total}")
        return

    perplexity = min(30, max(2, (n_total - 1) // 3))

    Z = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    ).fit_transform(X)

    Zb = Z[~is_active]
    Za = Z[is_active]

    # ------------------------------------------------------------
    # Compact color IDs, but colorbar labels show real L1 IDs
    # ------------------------------------------------------------
    used_l1_final = np.unique(l1)
    l1_to_color = {int(k): i for i, k in enumerate(used_l1_final)}
    l1_color = np.array([l1_to_color[int(k)] for k in l1])
    
    base = plt.get_cmap("turbo")
    colors = base(np.linspace(0.25, 0.95, len(used_l1_final)))
    
    cmap = ListedColormap(colors)

    bounds = np.arange(-0.5, len(used_l1_final) + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(6.8, 5.6))

    if len(Zb) > 0:
        plt.scatter(
            Zb[:, 0],
            Zb[:, 1],
            s=24,
            c="black",
            marker="x",
            linewidths=1.0,
            alpha=0.75,
            label="blank latents",
        )

    sc = plt.scatter(
        Za[:, 0],
        Za[:, 1],
        s=7,
        c=l1_color,
        cmap=cmap,
        norm=norm,
        alpha=0.85,
        label="active latents",
    )

    
    cbar = plt.colorbar(sc, ticks=np.arange(len(used_l1_final)))
    cbar.ax.set_yticklabels([str(k) for k in used_l1_final])
    cbar.set_label("Used L1 parent code index")
    
    plt.title("t-SNE of blank vs active latents")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()
    
    print(f"Saved used-L1 t-SNE: {out_png}")
    print("Used L1 parents:", used_l1_final.tolist())
    
def plot_blank_active_pca_l1(
    viz_root,
    out_png,
    max_active_per_l1=500,
    min_active_per_l1=1,
    max_blank=1000,
    random_state=0,
):
    data = load_latents(viz_root)
    if data is None:
        return

    zb, za, l1 = data

    zb = np.asarray(zb)
    za = np.asarray(za)
    l1 = np.asarray(l1).astype(int)

    if len(za) == 0:
        print("No active latents found for PCA.")
        return

    used_l1, counts = np.unique(l1, return_counts=True)
    keep_l1 = used_l1[counts >= min_active_per_l1]

    keep_mask = np.isin(l1, keep_l1)
    za = za[keep_mask]
    l1 = l1[keep_mask]

    if len(za) == 0:
        print("No active latents left after used-L1 filtering.")
        return

    rng = np.random.default_rng(random_state)
    selected = []

    for k in keep_l1:
        idx = np.where(l1 == k)[0]
        if len(idx) == 0:
            continue

        if max_active_per_l1 is not None and len(idx) > max_active_per_l1:
            idx = rng.choice(idx, size=max_active_per_l1, replace=False)

        selected.append(idx)

    if len(selected) == 0:
        print("No L1 groups selected for PCA.")
        return

    selected = np.concatenate(selected)
    za = za[selected]
    l1 = l1[selected]

    if zb is not None and len(zb) > 0:
        if max_blank is not None and len(zb) > max_blank:
            blank_idx = rng.choice(len(zb), size=max_blank, replace=False)
            zb = zb[blank_idx]
    else:
        zb = np.zeros((0, za.shape[1]), dtype=za.dtype)

    X = np.concatenate([zb, za], axis=0)

    is_active = np.array(
        [False] * len(zb) + [True] * len(za),
        dtype=bool,
    )

    if len(X) < 3:
        print(f"Too few points for PCA: n={len(X)}")
        return

    reducer = PCA(n_components=2, random_state=random_state)
    Z = reducer.fit_transform(X)

    Zb = Z[~is_active]
    Za = Z[is_active]

    used_l1_final = np.unique(l1)
    l1_to_color = {int(k): i for i, k in enumerate(used_l1_final)}
    l1_color = np.array([l1_to_color[int(k)] for k in l1])

    base = plt.get_cmap("turbo")
    colors = base(np.linspace(0.25, 0.95, len(used_l1_final)))
    cmap = ListedColormap(colors)

    bounds = np.arange(-0.5, len(used_l1_final) + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(6.8, 5.6))

    if len(Zb) > 0:
        plt.scatter(
            Zb[:, 0],
            Zb[:, 1],
            s=24,
            c="black",
            marker="x",
            linewidths=1.0,
            alpha=0.75,
            label="blank latents",
        )

    sc = plt.scatter(
        Za[:, 0],
        Za[:, 1],
        s=7,
        c=l1_color,
        cmap=cmap,
        norm=norm,
        alpha=0.85,
        label="active latents",
    )

    cbar = plt.colorbar(sc, ticks=np.arange(len(used_l1_final)))
    cbar.ax.set_yticklabels([str(k) for k in used_l1_final])
    cbar.set_label("Used L1 parent code index")

    evr = reducer.explained_variance_ratio_

    plt.title(
        f"PCA of blank vs active latents"
        f"(PC1={evr[0]:.2%}, PC2={evr[1]:.2%})"
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"Saved used-L1 PCA: {out_png}")
    print("Used L1 parents:", used_l1_final.tolist())
    print(f"PCA explained variance: PC1={evr[0]:.4f}, PC2={evr[1]:.4f}")