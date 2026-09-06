"""F6 -- one ground-truth clip under all four generation settings.

The dump indexes the same underlying clip under every task, so the four rows
are one clip seen through the whole conditioning ladder rather than four
unrelated clips. That is why the strip shows a single hollow truth point with
four model points hanging off it.

Observed regions are washed and every panel states what fraction of it the
model actually generated. Without that, the spikes the model copied from the
visible remainder read as hits and every masked task looks better than it is.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._voxel import QUANTITIES, REFERENCE, legend, panel, sitemap, strip
from .style import INK

ROOT = Path(__file__).resolve().parents[2]
JSON = ROOT / "reports" / "qualitative_panels.json"
NPZ = ROOT / "reports" / "qualitative_panels.npz"
NICE = {"recon": ("R", "free generation"), "causal": ("C", "causal"),
        "noncausal": ("N", "non-causal"), "spatial": ("S", "spatial")}


def draw(fig):
    meta = json.loads(JSON.read_text())
    npz = np.load(NPZ)
    rows = meta["generation"]["all"]
    by = {(r["task"], r["clip"]): r for r in rows}
    picked = [by[tuple(k)] for k in meta["generation"]["picked"]]
    nw = meta["windows"]

    gs = fig.add_gridspec(len(picked) + 1, nw + 1,
                          height_ratios=[1] * len(picked) + [1.72],
                          hspace=0.20, wspace=0.06, left=0.135, right=0.865,
                          top=0.945, bottom=0.055)
    for i, r in enumerate(picked):
        tag, nice = NICE[r["task"]]
        vmax = float(npz[f"gen/{i}/field_vmax"])
        wins = npz[f"gen/{i}/windows"]
        for j in range(nw):
            ax = fig.add_subplot(gs[i, j])
            im = panel(ax, npz[f"gen/{i}/{j}/field"], npz[f"gen/{i}/{j}/gt"],
                       npz[f"gen/{i}/{j}/pred"], vmax=vmax,
                       observed=npz[f"gen/{i}/{j}/observed"])
            # Each ROW has its own windows -- they are chosen to expose that
            # row's mask -- so the frame range cannot be a shared column title.
            a, b = wins[j]
            frac = float(npz[f"gen/{i}/{j}/roi_frac"])
            ax.text(0.02, 0.955, f"frames {a}–{b-1}", transform=ax.transAxes,
                    fontsize=4.6, color="white", zorder=6, va="top")
            ax.text(0.02, 0.04, f"{frac*100:.0f}% generated",
                    transform=ax.transAxes, fontsize=4.6, color="white",
                    zorder=6, va="bottom")
            if j == 0:
                ax.set_ylabel(f"{tag}  {nice}", fontsize=5.5, labelpad=3,
                              color=REFERENCE if r["task"] == "recon" else INK)
        sitemap(fig.add_subplot(gs[i, nw]), npz[f"gen/{i}/sitemap"],
                f"hole $r$ {r['map_r_pred_hole']:.2f}/{r['map_r_gt_hole']:.2f}\n"
                f"clip $r$ {r['map_r_pred']:.2f}/{r['map_r_gt']:.2f}",
                "recording\nsite map" if i == 0 else None)
    # The map column is HOLE-ONLY here, unlike F5 which has no hole.
    #
    # The whole-clip correlation is symmetric -- the copied remainder is
    # identical in the model's clip and the truth's -- but it is not comparable
    # ACROSS these rows, and the rows are the comparison this figure exists to
    # make. For a temporal hole every site is observed in some frame, so the
    # time projection is filled in from copied frames and the number is mostly
    # about truth the model was handed; for the spatial hole 28.1% of sites are
    # never observed at all. That is why the whole-clip reading put the spatial
    # row last while free generation, which invents everything, sat above it.
    # Restricting to the hole makes every row the same measurement. The
    # whole-clip value is still printed under each row's site map.
    quantities = [("map_r_gt_hole", "map_r_pred_hole", None,
                   "spatial map $r$ in hole\nvs recording map")] + QUANTITIES[1:]
    strip(fig, gs[len(picked), :], rows,
          [(NICE[r["task"]][0], r) + ((REFERENCE,) if r["task"] == "recon" else ())
           for r in picked], quantities)
    cax = fig.add_axes([0.878, 0.40, 0.012, 0.52])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("activity field, max over window\n(scaled to the clip's 99.9th pct)",
                 fontsize=5)
    cb.ax.tick_params(labelsize=5, length=2)
    legend(fig, len(rows), observed=True,
           reference="free generation (observes nothing)")
