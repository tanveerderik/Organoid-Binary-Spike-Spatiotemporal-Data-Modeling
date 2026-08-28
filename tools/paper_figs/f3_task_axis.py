"""F3 -- completion accuracy on the four tasks, at site and voxel level.

Form: a horizontal dot plot per level, not grouped bars. Bars would need 48
rectangles and, because two of the four hues sit below 3:1 on the light
surface, a printed number on every one of them to satisfy the relief rule --
which is the label-every-point anti-pattern. Dots carry the same information at
a fifth of the ink and let the null sit on the same row as the arms it beats.

Two panels with independent x axes because site AP spans 0.02-0.74 and voxel AP
0.003-0.10. They are two charts, not one chart with two scales.

The panels disagree, and that is the finding: our prior is at or near its own
alphabet's ceiling for WHERE a spike lands and nowhere near it for WHEN. Both
are shown at equal size; shrinking the one we lose would be picking the figure.
"""
from __future__ import annotations

import json
from pathlib import Path

from .style import (ARMS, CROSS_ARM, MARKER, PALETTE, TASKS, INK, INK_2,
                    SURFACE)

ROOT = Path(__file__).resolve().parents[2]
EB = ROOT / "reports" / "external_baselines"


def _load():
    rows = json.loads((EB / "cross_model_tests.json").read_text())["rows"]
    nulls = json.loads((EB / "task_eval_nulls.json").read_text())["tasks"]
    ours, arm, verdict = {}, {}, {}
    for r in rows:
        if r["arm"] not in CROSS_ARM.values():
            continue
        ours[(r["family"], r["task"])] = r["pipeline"]
        arm[(r["family"], r["task"], r["arm"])] = r["arm_value"]
        verdict[(r["family"], r["task"], r["arm"])] = r["verdict"]
    return ours, arm, verdict, nulls


def _panel(ax, fam, fld, ours, arm, verdict, nulls, title, show_legend):
    ys = list(range(len(TASKS)))[::-1]          # first task at the top
    for y, (tkey, tlab) in zip(ys, TASKS):
        # The null the whole figure exists to report: a static per-recording
        # site map, no model, no clip-specific information -- and every learned
        # arm is to its left.
        ax.plot(nulls[tkey]["arms"]["marginal"][fld], y, marker="|",
                ms=11, mew=1.6, color=PALETTE["null"], zorder=5,
                label="_nolegend_")
        # The comparison the paper makes, drawn as a pair: a connector between
        # our value and the matched flat tokenizer's, and a mark when the
        # paired test goes our way. Without it the figure shows four dots and
        # leaves the reader to guess which differences survived a test.
        a = ours[(fam, tkey)]
        b = arm.get((fam, tkey, CROSS_ARM["maskgit_flat"]))
        if b is not None:
            ax.plot([a, b], [y, y], lw=1.0, color=INK_2, alpha=0.35, zorder=2,
                    solid_capstyle="round")
            if verdict.get((fam, tkey, CROSS_ARM["maskgit_flat"])) == "pipeline":
                ax.annotate("$\\ast$", (max(a, b), y), textcoords="offset points",
                            xytext=(0, 5.5), ha="center", va="bottom",
                            fontsize=8, color=INK)
        for key, lab in ARMS:
            v = (ours[(fam, tkey)] if key == "pipeline"
                 else arm.get((fam, tkey, CROSS_ARM[key])))
            if v is None:
                continue
            categorical = key in ("pipeline", "maskgit_flat")
            # A surface-coloured ring so coincident marks still read as two.
            # Ours and the flat tokenizer land within 0.002 on three of the
            # four voxel rows -- that near-identity is a result, and it has to
            # be visible as two marks rather than one.
            ax.plot(v, y, marker=MARKER[key], ms=5.5 if categorical else 4.5,
                    color=PALETTE[key],
                    mfc=PALETTE[key] if categorical else "none",
                    mec=SURFACE if categorical else PALETTE[key],
                    mew=1.0 if categorical else 1.2,
                    ls="none", zorder=4, label="_nolegend_")
    ax.set_yticks(ys, [t for _, t in TASKS])
    ax.set_xlabel(title)
    ax.set_ylim(-0.6, len(TASKS) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(left=0)
    ax.margins(x=0.06)


def draw(fig) -> None:
    ours, arm, verdict, nulls = _load()
    axes = fig.subplots(1, 2, sharey=True)
    _panel(axes[0], "site AP", "site_ap", ours, arm, verdict, nulls,
           "site AP $\\uparrow$  (which electrode)", True)
    _panel(axes[1], "voxel AP", "ap", ours, arm, verdict, nulls,
           "voxel AP $\\uparrow$  (electrode and frame)", False)

    # Legend once, above both panels. Identity is never colour-alone: the two
    # categorical arms are filled, the two references are open, and every entry
    # carries both a distinct marker and its name.
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="none", marker=MARKER[k], ms=5.5 if f else 4.5,
                      color=PALETTE[k], mfc=PALETTE[k] if f else "none",
                      mew=1.2, label=lab)
               for (k, lab), f in zip(ARMS, (True, True, False, False))]
    handles.append(Line2D([], [], ls="none", marker="|", ms=11, mew=1.6,
                          color=PALETTE["null"],
                          label="null: recording site map"))
    # Anchored in inches above the axes, not as a figure fraction: the figure
    # height is a page-budget knob and a fractional anchor slides down onto the
    # panels as the figure shrinks. The asterisk is explained in the caption,
    # so there is no second annotation line competing for this strip.
    H = float(fig.get_size_inches()[1])
    fig.legend(handles=handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1 + 0.16 / H), handletextpad=0.4,
               columnspacing=1.1, labelcolor=INK_2)
    fig.subplots_adjust(wspace=0.12)
