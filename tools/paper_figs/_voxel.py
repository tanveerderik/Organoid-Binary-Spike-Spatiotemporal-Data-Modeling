"""Shared drawing for the two qualitative appendix figures.

Both figures are the same object: rows of time-window panels over a continuous
field, a recording-site-map column, and a paired adherence strip underneath.
They differ only in what a row is (a recording, or a task).
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .style import (FIELD_RAMP, INK, INK_2, INK_MUTED, OBSERVED_WASH, VOXEL,
                    VOXEL_MARKER, VOXEL_REFERENCE)

# Free generation observes nothing, so it is not a completion setting at all --
# Appendix J calls it the no-visible-volume reference. It is drawn in the hue
# style.py reserves for the figure's reference register so that a reader does
# not compare it to the masked rows as if it were a fourth task.
REFERENCE = VOXEL_REFERENCE

FIELD_CMAP = LinearSegmentedColormap.from_list("field", FIELD_RAMP)
XR, XE = 0.0, 0.62          # "truth" and "model" columns of the strip
SIZES = {"hit": 3.0, "missed": 4.4, "hallucinated": 4.0}


def panel(ax, field, gt, pred, vmax=1.0, observed=None):
    im = ax.imshow(field, cmap=FIELD_CMAP, norm=Normalize(0, vmax),
                   interpolation="nearest", aspect="equal")
    for name, mask in (("hit", gt & pred), ("missed", gt & ~pred),
                       ("hallucinated", ~gt & pred)):
        y, x = np.nonzero(mask)
        ax.scatter(x, y, s=SIZES[name], c=VOXEL[name], marker=VOXEL_MARKER[name],
                   linewidths=0.30, edgecolors=INK, zorder=5)
    if observed is not None and observed.any():
        # Outside the ROI the model copies the observed truth rather than
        # generating it. A panel that does not say so presents copied spikes as
        # hits, which would flatter every masked task.
        import matplotlib.colors as mc
        r, g, b = mc.to_rgb(OBSERVED_WASH)
        h, w = observed.shape
        ax.imshow(np.dstack([np.full((h, w), r), np.full((h, w), g),
                             np.full((h, w), b), observed * 0.80]),
                  interpolation="nearest", aspect="equal", zorder=3)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(INK_MUTED); sp.set_linewidth(0.4)
    return im


def sitemap(ax, sm, label, title=None):
    ax.imshow(sm, cmap="cividis", interpolation="nearest", aspect="equal",
              norm=Normalize(0, float(np.percentile(sm, 99.9))))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(INK_MUTED); sp.set_linewidth(0.4)
    ax.set_xlabel(label, fontsize=5, color=INK_2, labelpad=1.5)
    if title:
        ax.set_title(title, fontsize=5.5, pad=2)


def strip(fig, cell, rows, picks, quantities):
    """picks: (tag, row) or (tag, row, colour). A colour marks a REFERENCE row."""
    """Paired truth->model segments for every clip, with the panels marked.

    Deliberately NOT two marginal distributions with two medians. That is the
    pooled reading, and on active-site ratio it reports a 33% shortfall where
    the per-clip median change is 0.0000 and half the clips move up.
    """
    sub = cell.subgridspec(1, len(quantities), wspace=0.62)
    for i, (rk, ek, dk, lab) in enumerate(quantities):
        ax = fig.add_subplot(sub[0, i])
        ref = np.array([r[rk] for r in rows], float)
        est = (np.array([r[ek] for r in rows], float) if ek
               else ref + np.array([r[dk] for r in rows], float))
        ok = ~(np.isnan(ref) | np.isnan(est))
        n = int(ok.sum())
        ax.plot(np.stack([np.full(n, XR), np.full(n, XE)]),
                np.stack([ref[ok], est[ok]]), color=INK_MUTED, lw=0.35,
                alpha=0.16, zorder=1, solid_capstyle="butt")
        m0 = float(np.median(ref[ok]))
        ax.plot([XR, XE], [m0, m0 + float(np.median(est[ok] - ref[ok]))],
                color=INK_2, lw=1.6, zorder=3, solid_capstyle="round")
        marks = []
        for pick in picks:
            tag, r = pick[0], pick[1]
            col = pick[2] if len(pick) > 2 else INK
            dashed = col != INK
            rv = r[rk]; ev = r[ek] if ek else r[rk] + r[dk]
            ax.plot([XR, XE], [rv, ev], color=col, lw=0.9 if dashed else 0.7,
                    ls=(0, (2.2, 1.4)) if dashed else "-", zorder=4)
            ax.scatter([XR], [rv], s=15, marker="o", facecolor="white",
                       edgecolor=col, lw=0.8, zorder=5)
            ax.scatter([XE], [ev], s=15, marker="o", facecolor=col,
                       edgecolor=col, lw=0.8, zorder=5)
            marks.append([ev, tag, col])
        # Row letters collide on three of the four quantities, which is exactly
        # where the figure is making its point, so they are pushed apart and
        # leadered back to the mark they belong to.
        lo, hi = ax.get_ylim(); gap = (hi - lo) * 0.055
        marks.sort(key=lambda m: m[0])
        for j in range(1, len(marks)):
            if marks[j][0] - marks[j - 1][0] < gap:
                marks[j][0] = marks[j - 1][0] + gap
        for ytxt, tag, col in marks:
            r = next(p[1] for p in picks if p[0] == tag)
            ev = r[ek] if ek else r[rk] + r[dk]
            ax.annotate(tag, (XE, ytxt), textcoords="offset points",
                        xytext=(6.0, -2.2), fontsize=5.5, color=col, zorder=6)
            if abs(ytxt - ev) > 1e-12:
                ax.plot([XE, XE + 0.085], [ev, ytxt], color=INK_MUTED, lw=0.4,
                        zorder=4)
        ax.set_xlim(-0.30, 1.06); ax.set_xticks([XR, XE])
        ax.set_xticklabels(["truth", "model"], fontsize=5.5)
        ax.set_title(lab, fontsize=5.5, pad=3)
        ax.margins(y=0.10); ax.grid(False)
        ax.tick_params(axis="y", labelsize=5, length=2, pad=1)
        ax.tick_params(axis="x", length=0, pad=1)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_visible(True)
            ax.spines[s_].set_color(INK_MUTED); ax.spines[s_].set_linewidth(0.4)


def legend(fig, n, observed=False, reference=None):
    h = [Line2D([], [], marker="o", ls="", ms=3, mfc=VOXEL["hit"], mec=INK,
                mew=0.4, label="hit"),
         Line2D([], [], marker="X", ls="", ms=3.4, mfc=VOXEL["missed"], mec=INK,
                mew=0.4, label="missed"),
         Line2D([], [], marker="P", ls="", ms=3.4, mfc=VOXEL["hallucinated"],
                mec=INK, mew=0.4, label="hallucinated"),
         Line2D([], [], marker="o", ls="", ms=3.2, mfc="white", mec=INK,
                label="truth"),
         Line2D([], [], marker="o", ls="", ms=3.2, mfc=INK, mec=INK, label="model"),
         Line2D([], [], ls="-", lw=1.6, c=INK_2, label=f"median clip (n={n})")]
    if observed:
        h.append(Patch(fc=OBSERVED_WASH, alpha=0.80, ec="none", label="observed"))
    if reference:
        h.append(Line2D([], [], ls=(0, (2.2, 1.4)), lw=1.0, c=REFERENCE,
                        marker="o", ms=3.2, mfc=REFERENCE, mec=REFERENCE,
                        label=reference))
    fig.legend(handles=h, loc="upper center", ncol=len(h), fontsize=5.5,
               frameon=False, bbox_to_anchor=(0.5, 1.003), handletextpad=0.2,
               columnspacing=0.8)


QUANTITIES = [
    ("map_r_gt", "map_r_pred", None, "spatial map $r$\nvs assay map"),
    ("ref_logmean", None, "d_logmean", "log mean rate"),
    ("ref_active", None, "d_active", "active-site ratio"),
    ("ref_trend", None, "d_trend", "temporal trend"),
]
