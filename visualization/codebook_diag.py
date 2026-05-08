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


def plot_blank_active_tsne_l1(viz_root, out_png, num_l1=32):
    data = load_latents(viz_root)
    if data is None:
        return

    zb, za, l1 = data

    X = np.concatenate([zb, za], axis=0)
    is_active = np.array([0] * len(zb) + [1] * len(za))

    Z = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=0,
    ).fit_transform(X)

    Zb = Z[is_active == 0]
    Za = Z[is_active == 1]

    # Bright-only discrete colormap.
    # Avoids the dark purple/black region of turbo.

    base = plt.cm.turbo
    bright_colors = base(np.linspace(0.18, 1.00, num_l1))
    cmap = ListedColormap(bright_colors, name="bright_turbo")

    bounds = np.arange(-0.5, num_l1 + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(6.2, 5.2))

    # blank = light gray with black edge, not black fill
    plt.scatter(
        Zb[:, 0],
        Zb[:, 1],
        s=10,
        facecolors="lightgray",
        edgecolors="black",
        linewidths=0.35,
        alpha=0.75,
        label="blank",
        zorder=1,
    )

    # active = bright colors
    sc = plt.scatter(
        Za[:, 0],
        Za[:, 1],
        s=18,
        c=l1,
        cmap=cmap,
        norm=norm,
        alpha=0.95,
        edgecolors="none",
        label="active",
        zorder=2,
    )

    cbar = plt.colorbar(sc, ticks=np.arange(num_l1))
    cbar.set_label("L1 code index")

    plt.title("Decoder-input latent t-SNE (blank vs active)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()