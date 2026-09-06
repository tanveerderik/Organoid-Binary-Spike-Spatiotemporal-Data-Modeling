"""F5 -- what the tokeniser's reconstruction looks like, and how typical it is.

Four recordings, two from each corpus, three time windows each. Panels are
maximum projections over the window rather than single frames: at this
occupancy a frame carries two to ten spikes over 26,880 sites, so a single
frame is visually empty and shows nothing about motif structure.

The strip underneath is what stops this being a gallery. The rows are SELECTED
-- best clip per recording by attained map correlation -- and the strip puts
each of them on the distribution over all 279 test clips, so the reader sees
both that they are good and how far above typical they are.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._voxel import QUANTITIES, legend, panel, sitemap, strip
from .style import INK, TEXT_W

ROOT = Path(__file__).resolve().parents[2]
JSON = ROOT / "reports" / "qualitative_panels.json"
NPZ = ROOT / "reports" / "qualitative_panels.npz"
TAGS = "ABCD"


def draw(fig):
    meta = json.loads(JSON.read_text())
    npz = np.load(NPZ)
    rows = meta["recon"]["all"]
    by = {r["sample"]: r for r in rows}
    picked = [by[s] for s in meta["recon"]["picked"]]
    nw = meta["windows"]

    gs = fig.add_gridspec(len(picked) + 1, nw + 1,
                          height_ratios=[1] * len(picked) + [1.62],
                          hspace=0.14, wspace=0.06, left=0.135, right=0.865,
                          top=0.945, bottom=0.055)
    for i, r in enumerate(picked):
        short = r["recording"].split("obj-")[-1].replace("_ecephys", "")
        short = short if len(short) < 12 else r["recording"].split("_")[0].replace("sub-", "")
        kind = "organoid" if r["recording"].startswith("sub-U") else "ex vivo"
        wins = npz[f"recon/{i}/windows"]
        for j in range(nw):
            ax = fig.add_subplot(gs[i, j])
            im = panel(ax, npz[f"recon/{i}/{j}/field"], npz[f"recon/{i}/{j}/gt"],
                       npz[f"recon/{i}/{j}/pred"], vmax=1.0)
            if i == 0:
                a, b = wins[j]
                ax.set_title(f"frames {a}–{b-1}", fontsize=5.5, pad=2)
            if j == 0:
                ax.set_ylabel(f"{TAGS[i]}  {short}\n{kind}", fontsize=5.5,
                              color=INK, labelpad=3)
        sitemap(fig.add_subplot(gs[i, nw]), npz[f"recon/{i}/sitemap"],
                f"$r$ {r['map_r_pred']:.2f} / {r['map_r_gt']:.2f}",
                "recording\nsite map" if i == 0 else None)
    strip(fig, gs[len(picked), :], rows,
          list(zip(TAGS, picked)), QUANTITIES)
    cax = fig.add_axes([0.878, 0.40, 0.012, 0.52])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("decoder $p$(spike), max over window", fontsize=5.5)
    cb.ax.tick_params(labelsize=5, length=2)
    legend(fig, len(rows))
