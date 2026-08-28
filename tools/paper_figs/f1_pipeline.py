"""F1 -- the pipeline, and the data it is shaped by.

The only figure not plotted from data, but every NUMBER in it is formatted from
the same JSON the tables read, so the schematic cannot drift from the results.
The routed-electrode panel uses a real recording's channel count rather than a
decorative pattern: the routed fraction is the single most clarifying fact about
this data, and a made-up version of it would undercut the point.

Four stages left to right, each a labelled box with the quantity that stage is
constrained by underneath.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from .style import PALETTE, SURFACE, INK, INK_2, INK_MUTED

ROOT = Path(__file__).resolve().parents[2]


def _load():
    pre = json.loads((ROOT / "reports" / "preproc_stats.json").read_text())
    prov = json.loads((ROOT / "reports" / "data_provenance.json").read_text())
    flat = json.loads((ROOT / "reports"
                       / "analysis_stage2b_flatten.json").read_text())
    sweep = json.loads((ROOT / "reports"
                        / "ablation_patch_size.json").read_text())
    ship = next(r for r in sweep["rows"] if r["is_shipped"])
    return pre, prov, flat, ship


def _box(ax, x, w, title, lines, accent=INK_2):
    ax.add_patch(Rectangle((x, 0.30), w, 0.46, transform=ax.transAxes,
                           facecolor="none", edgecolor="#d3d2cc", lw=0.7,
                           zorder=1))
    ax.text(x + w / 2, 0.795, title, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=7.2, color=accent, weight="bold")
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, 0.66 - i * 0.105, ln, transform=ax.transAxes,
                ha="center", va="center", fontsize=5.5, color=INK_2)


def _arrow(ax, x0, x1):
    ax.add_patch(FancyArrowPatch((x0, 0.53), (x1, 0.53),
                                 transform=ax.transAxes,
                                 arrowstyle="-|>", mutation_scale=7,
                                 lw=0.8, color=INK_MUTED, zorder=2))


def draw(fig) -> None:
    pre, prov, flat, ship = _load()
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    n_org = prov["by_preparation"].get("organoid slice", 0) or sum(
        1 for r in prov["used"] if r["dandiset"] == "000732")
    n_sli = prov["n_used"] - n_org
    ch = [r["n_routed_channels"] for r in prov["used"]]
    T, H, W = pre["clip_shape"]
    gt, gh, gw = pre["token_grid"]
    pt, ph, pw = pre["patch"]

    # Text outside $...$ is NOT LaTeX here -- matplotlib's default path
    # renders "\," and "\%" literally -- so escapes are avoided and anything
    # symbolic goes through mathtext.
    _box(ax, 0.000, 0.238, "recordings",
         [f"{prov['n_used']} recordings",
          f"{n_org} organoid, {n_sli} slice",
          f"{min(ch)}-{max(ch)} routed sites",
          f"20 kHz, {pre['frame_ms']:.0f} ms frames"],
         accent=INK)
    _box(ax, 0.254, 0.238, "clip",
         [f"${T}\\times{H}\\times{W}$",
          f"{pre['clip_ms']:.0f} ms, {pre['clip_voxels'] / 1e6:.2f}M voxels",
          f"rate {pre['clip_voxel_rate']:.1e}",
          f"$\\approx${pre['mean_spikes_per_clip']:.0f} spikes"])
    _box(ax, 0.508, 0.238, "tokenise",
         [f"patch $({pt},{ph},{pw})$",
          f"${gt}{{\\times}}{gh}{{\\times}}{gw}$ = {gt*gh*gw:,} tokens",
          f"{ship['blank_frac']*100:.0f}% blank: one token",
          "residual ladder 32/8/4"],
         accent=PALETTE["pipeline"])
    _box(ax, 0.762, 0.238, "motifs and priors",
         [f"{flat['F']:,} sums $\\rightarrow$ $V$ = {flat['distinct']}",
          f"{flat['merged']} duplicates merged",
          "activity prior: where",
          "motif prior: which motif"],
         accent=PALETTE["pipeline"])

    for x0, x1 in ((0.241, 0.251), (0.495, 0.505), (0.749, 0.759)):
        _arrow(ax, x0, x1)

    ax.text(0.5, 0.16,
            "conditioning: a per-recording code costing zero stored "
            "parameters per preparation, plus a local code",
            transform=ax.transAxes, ha="center", va="center", fontsize=6.3,
            color=INK_MUTED, style="italic")
