"""F2 -- what a motif is, and whether it is reused.

The paper's identity figure. Two bands, one question each.

Band A, the atlas: the twelve most-used alphabet entries, each drawn as the
MEAN REAL VOXEL PATCH the encoder assigned to it across the test split. This is
deliberately not a decoder rendering. A decoded codebook entry shows what the
model believes a code means; the empirical mean shows what the data does under
it, which is the thing a reviewer asks about when a paper claims its codes are
motifs rather than cluster labels.

Band B, reuse: the shared-core curve and the pairwise vocabulary overlap. The
overlap is RAREFIED to a common token budget. The raw statistic tracks sampling
effort at r = +0.96 -- recordings contribute 117 to 1756 tokens -- and read
naively it says organoid recordings share more motifs with each other than
slice recordings do, which is entirely the organoid recordings being longer.

Qualitative samples are deliberately absent. They would make a third band and a
weaker figure; F3 already carries the completion comparison quantitatively, and
an identity figure should make one point.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .style import PALETTE, SURFACE, INK, INK_2, INK_MUTED

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "reports" / "analysis_motif_reuse.json"
NPZ = ROOT / "reports" / "analysis_motif_reuse.npz"

N_TILES = 12
# Sequential = one hue, light to dark, from the chart surface to the series
# colour. Never a rainbow: this encodes magnitude, not identity.
RAMP = LinearSegmentedColormap.from_list(
    "motif", [SURFACE, "#c3d8f2", PALETTE["pipeline"], "#14365f"])


def _band_atlas(fig, gs):
    z = np.load(NPZ)
    ids, patches, ns = z["atlas_ids"], z["atlas_mean_patch"], z["atlas_n"]
    total = float(z["counts"].sum())
    sub = gs.subgridspec(2, N_TILES, hspace=0.08, wspace=0.16,
                         height_ratios=[2.4, 1.0])
    for i in range(N_TILES):
        p = patches[i]
        # Time-collapsed: which electrodes of the patch this motif lights up.
        ax = fig.add_subplot(sub[0, i])
        # equal aspect: a 15x14 electrode patch is nearly square and must
        # not be stretched into a tall rectangle by the subplot box.
        ax.imshow(p.sum(axis=0), cmap=RAMP, aspect="equal",
                  interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#d9d8d2"); sp.set_linewidth(0.5)
        ax.set_title(f"{ns[i] / total * 100:.1f}%", fontsize=5.2,
                     color=INK_2, pad=1.5)
        # Its temporal profile over the patch's six frames.
        axt = fig.add_subplot(sub[1, i])
        prof = p.sum(axis=(1, 2))
        axt.bar(range(len(prof)), prof, width=0.82,
                color=PALETTE["pipeline"], linewidth=0)
        axt.set_xticks([]); axt.set_yticks([])
        axt.set_ylim(0, float(patches.sum(axis=(2, 3)).max()))
        axt.grid(False)
        for sp in axt.spines.values():
            sp.set_visible(False)
    fig.text(0.008, 0.965, "A", fontsize=9, weight="bold", color=INK)
    fig.text(0.5, 0.995,
             "the twelve most-used motifs: mean real patch per entry "
             "(top, electrodes; bottom, its 6 frames)",
             ha="center", va="top", fontsize=6.5, color=INK_2)


def _band_reuse(fig, gs, d):
    d_V = d["V"]
    sub = gs.subgridspec(1, 2, wspace=0.34, width_ratios=[1.0, 1.15])

    # Left: how much of the alphabet is common property.
    ax = fig.add_subplot(sub[0, 0])
    core = d["shared_core"]["used_by_ge_k_rarefied"]
    ks = sorted(int(k) for k in core)
    ax.plot(ks, [core[str(k)] for k in ks], marker="o", ms=4, lw=1.6,
            color=PALETTE["pipeline"], mec=SURFACE, mew=0.8)
    ax.set_xlabel("used by at least $k$ recordings", fontsize=7)
    ax.set_ylabel("alphabet entries", fontsize=7)
    ax.annotate(f"{core['5']:.0f} of {d_V} entries\nused by $\\geq$5",
                (5, core["5"]), textcoords="offset points", xytext=(10, 10),
                fontsize=6.2, color=INK_2)
    ax.tick_params(labelsize=6)

    # Right: pairwise overlap, recordings blocked by preparation type.
    axm = fig.add_subplot(sub[0, 1])
    J = np.array(d["jaccard_rarefied"]["matrix"])
    prep = [r["prep"] for r in d["per_recording"]]
    order = ([i for i, p in enumerate(prep) if p == "organoid"]
             + [i for i, p in enumerate(prep) if p == "slice"])
    n_org = sum(p == "organoid" for p in prep)
    M = J[np.ix_(order, order)]
    np.fill_diagonal(M, np.nan)          # self-overlap is 1 by construction
    im = axm.imshow(M, cmap=RAMP, interpolation="nearest",
                    vmin=np.nanmin(M), vmax=np.nanmax(M))
    for v in (n_org - 0.5,):
        axm.axhline(v, color=INK, lw=0.8)
        axm.axvline(v, color=INK, lw=0.8)
    axm.set_xticks([n_org / 2 - 0.5, n_org + (len(prep) - n_org) / 2 - 0.5],
                   ["organoid", "slice"], fontsize=6.5)
    axm.set_yticks([n_org / 2 - 0.5, n_org + (len(prep) - n_org) / 2 - 0.5],
                   ["organoid", "slice"], fontsize=6.5)
    axm.tick_params(length=0)
    axm.grid(False)
    cb = fig.colorbar(im, ax=axm, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=5.5, length=2)
    cb.outline.set_visible(False)

    jr = d["jaccard_rarefied"]
    axm.set_xlabel(
        f"within {jr['mean_within_organoid']:.3f} / "
        f"{jr['mean_within_slice']:.3f}   across "
        f"{jr['mean_cross_prep']:.3f}\n"
        f"label-shuffled null {jr['null_mean']:.3f} $\\pm$ {jr['null_sd']:.3f}",
        fontsize=6.2, color=INK_2)
    fig.text(0.008, 0.50, "B", fontsize=9, weight="bold", color=INK)


def draw(fig) -> None:
    d = json.loads(SRC.read_text())
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.5], hspace=0.38,
                          left=0.09, right=0.97, top=0.91, bottom=0.14)
    _band_atlas(fig, gs[0])
    _band_reuse(fig, gs[1], d)
