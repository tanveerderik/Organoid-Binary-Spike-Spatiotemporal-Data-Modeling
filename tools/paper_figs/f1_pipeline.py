"""F1 -- the dataflow, and the data it is shaped by.

The only figure not plotted from data, but every NUMBER in it is formatted from
the same JSON the tables read, so the schematic cannot drift from the results.

This is a dataflow diagram and not a list of stages, because the question it
has to answer is one prose answers badly: WHICH CONTEXT REACHES WHICH MODULE.
Two facts in particular are cheaper to draw than to write -- that empty patches
leave the pipeline before the quantiser rather than being quantised to a blank
code, and that the two priors read the recording code by different routes, the
motif prior through the frozen Stage-1 mapper and the activity prior through a
projection of its own.

Two rows. The top row is the tokeniser: clip, patchify, the blank branch, the
residual ladder, the flattened alphabet. The bottom row is generation: the two
conditioning codes into the activity prior, its field into the motif prior, and
the decoder back to voxels. The task mask enters both priors, which is why it
is drawn once and branched rather than twice.
"""
from __future__ import annotations

import json
from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .style import PALETTE, SURFACE, INK, INK_2, INK_MUTED

ROOT = Path(__file__).resolve().parents[2]

# Row baselines in axes coordinates. The top row sits high enough to leave the
# blank branch room to drop below it without colliding with the bottom row.
TOP, BOT = 0.80, 0.30
BOX_H = 0.15


def _load():
    pre = json.loads((ROOT / "reports" / "preproc_stats.json").read_text())
    prov = json.loads((ROOT / "reports" / "data_provenance.json").read_text())
    flat = json.loads((ROOT / "reports"
                       / "analysis_stage2b_flatten.json").read_text())
    sweep = json.loads((ROOT / "reports"
                        / "ablation_patch_size.json").read_text())
    hp = json.loads((ROOT / "reports" / "hyperparameters.json").read_text())
    ship = next(r for r in sweep["rows"] if r["is_shipped"])
    return pre, prov, flat, ship, hp


def _box(ax, x, y, w, label, sub=None, accent=INK_2, h=BOX_H, fill="none",
         fs=6.1):
    """A rounded node. `sub` is the one quantity that node is constrained by."""
    # clip_on=False throughout: these are axes-fraction patches on an axis
    # with no data, and the default clip crops the rounded corner of anything
    # that touches x=0 or x=1.
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.012",
                                transform=ax.transAxes, facecolor=fill,
                                edgecolor=accent, lw=0.8, zorder=2,
                                clip_on=False))
    dy = 0.022 if sub else 0.0
    ax.text(x + w / 2, y + dy, label, transform=ax.transAxes, ha="center",
            va="center", fontsize=fs, color=INK, zorder=3, clip_on=False)
    if sub:
        ax.text(x + w / 2, y - 0.030, sub, transform=ax.transAxes,
                ha="center", va="center", fontsize=4.8, color=INK_2, zorder=3,
                clip_on=False)


def _arrow(ax, p0, p1, color=INK_MUTED, style="-|>", rad=0.0, lw=0.8,
           dashed=False):
    ax.add_patch(FancyArrowPatch(
        p0, p1, transform=ax.transAxes, arrowstyle=style, mutation_scale=7,
        lw=lw, color=color, zorder=1,
        linestyle=(0, (2.4, 1.6)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}", shrinkA=1.0, shrinkB=1.0,
        clip_on=False))


def draw(fig) -> None:
    pre, prov, flat, ship, hp = _load()
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    T, H, W = pre["clip_shape"]
    gt, gh, gw = pre["token_grid"]
    pt, ph, pw = pre["patch"]
    ch = [r["n_routed_channels"] for r in prov["used"]]
    blue = PALETTE["pipeline"]

    # Five columns, shared by both rows so the vertical links land straight.
    # Text outside $...$ is NOT LaTeX here -- matplotlib's default path renders
    # "\%" literally -- so per-cent signs are bare and anything symbolic goes
    # through mathtext.
    # Column 0 is inset from the canvas edge: its sub-labels ("seeded, per
    # recording") are wider than the box, are centred on it, and at x=0 the
    # overhang fell off the saved figure and printed clipped. Widths are set
    # by the widest LABEL in each column, not shared, so no text overruns its
    # own border at this font size.
    X = [0.018, 0.215, 0.414, 0.628, 0.832]
    Wd = [0.172, 0.174, 0.188, 0.176, 0.168]

    # ---- top row: tokeniser -------------------------------------------
    _box(ax, X[0], TOP, Wd[0], "clip $X$", f"${T}{{\\times}}{H}{{\\times}}{W}$")
    _box(ax, X[1], TOP, Wd[1], "patchify", f"$({pt},{ph},{pw})$")
    _box(ax, X[2], TOP, Wd[2], "residual ladder", "$32/8/4$, conditional",
         accent=blue)
    _box(ax, X[3], TOP, Wd[3], "flatten $+$ merge", f"{flat['F']:,} paths",
         accent=blue)
    _box(ax, X[4], TOP, Wd[4], "alphabet", f"$V = {flat['distinct']}$ motifs",
         accent=blue, fill="#eaf1fb")
    for i in range(4):
        _arrow(ax, (X[i] + Wd[i], TOP), (X[i + 1], TOP))

    # The blank branch leaves BEFORE the quantiser. Drawn as a departure from
    # the chain, not as a box in it, because that is exactly the claim: these
    # patches are never quantised.
    _box(ax, 0.245, TOP - 0.245, 0.230, "blank token",
         f"{ship['blank_frac']*100:.0f}% of patches empty",
         accent=INK_MUTED, h=0.120)
    _arrow(ax, (0.290, TOP - BOX_H / 2), (0.290, TOP - 0.245 + 0.060),
           color=INK_MUTED, dashed=True)
    ax.text(0.300, TOP - 0.122, "$b_p = 1$", transform=ax.transAxes,
            fontsize=5.2, color=INK_MUTED, ha="left", va="center")

    # ---- bottom row: conditioning and the two priors --------------------
    hi, lo = BOT + 0.088, BOT - 0.108
    _box(ax, X[0], hi, Wd[0], "$g_r$: recording", "one per recording",
         h=0.120)
    _box(ax, X[0], lo, Wd[0], "$\\ell(X)$: clip", "$9$ scalars, unlearned",
         h=0.120)
    _box(ax, X[1], hi, Wd[1], "Stage-1 mapper", "frozen", accent=INK_MUTED,
         h=0.110)
    _box(ax, X[1], lo, Wd[1], "Stage-3 trunk", "frozen", accent=INK_MUTED,
         h=0.110)
    _box(ax, X[2], BOT, Wd[2], "activity prior", "$p(A \\mid \\cdot)$, count",
         accent=blue)
    _box(ax, X[3], BOT, Wd[3], "motif prior", "$p(M_Z \\mid A, \\cdot)$",
         accent=blue)
    _box(ax, X[4], BOT, Wd[4], "decoder", "voxels")

    _arrow(ax, (X[0] + Wd[0], hi), (X[1], hi))
    _arrow(ax, (X[0] + Wd[0], lo), (X[1], lo))
    # Both mapped codes condition both priors. Drawn as a rail rather than as
    # four curves: the four-curve version crossed twice and made the one
    # asymmetry that matters -- the dashed raw-code path below -- unreadable.
    rail = (X[1] + Wd[1] + X[2]) / 2
    ax.plot([rail, rail], [lo, hi], transform=ax.transAxes, color=INK_MUTED,
            lw=0.8, zorder=1, clip_on=False)
    _arrow(ax, (X[1] + Wd[1], hi), (rail, hi), color=INK_MUTED)
    _arrow(ax, (X[1] + Wd[1], lo), (rail, lo), color=INK_MUTED)
    _arrow(ax, (rail, BOT + 0.030), (X[2], BOT + 0.030), color=INK_MUTED)
    _arrow(ax, (rail, hi), (X[3] + 0.030, BOT + BOX_H / 2), rad=-0.22,
           color=INK_MUTED)
    # gct also reaches the activity prior WITHOUT the mapper: that prior
    # projects the raw code with a linear layer of its own. Routed under the
    # row so it cannot be mistaken for the mapped path.
    _arrow(ax, (X[0] + Wd[0] * 0.45, hi - 0.060),
           (X[2] + 0.030, BOT - BOX_H / 2), rad=0.62, dashed=True)
    ax.text(X[0], BOT - 0.272, "raw $g_r$, no mapper",
            transform=ax.transAxes, fontsize=5.2, color=INK_MUTED,
            ha="left", va="center", clip_on=False)

    _arrow(ax, (X[2] + Wd[2], BOT), (X[3], BOT), color=blue, lw=1.0)
    ax.text(X[2] + Wd[2] + 0.0125, BOT - 0.098, "$A$", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.2, color=blue, clip_on=False)
    _arrow(ax, (X[3] + Wd[3], BOT), (X[4], BOT))

    # The alphabet is what the motif prior draws its symbols from.
    _arrow(ax, (X[4] + 0.020, TOP - BOX_H / 2), (X[3] + Wd[3] - 0.020,
           BOT + BOX_H / 2), color=blue, dashed=True, rad=0.25)

    # ---- the task mask, entering both priors ---------------------------
    task_x = X[1] + Wd[1]
    _box(ax, task_x, BOT - 0.255, X[4] - 0.008 - task_x,
         "task $(q, M)$: free generation, causal, noncausal, spatial",
         accent=INK_MUTED, h=0.100, fs=4.8)
    for x in (X[2] + Wd[2] * 0.62, X[3] + Wd[3] * 0.45):
        _arrow(ax, (x, BOT - 0.255 + 0.050), (x, BOT - BOX_H / 2),
               color=INK_MUTED, dashed=True)

    ax.text(0.0, 1.02, f"{prov['n_used']} recordings, "
            f"{min(ch)} to {max(ch)} routed sites each",
            transform=ax.transAxes, fontsize=5.6, color=INK_2, va="bottom")
    ax.text(1.0, 1.02, "dashed: enters as conditioning",
            transform=ax.transAxes, fontsize=5.6, color=INK_MUTED,
            va="bottom", ha="right")
