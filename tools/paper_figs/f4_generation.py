"""F4 -- the four generation families, each arm against its own null.

Form: four small-multiple dumbbells. Each row pairs an arm's full-context value
(filled) with the SAME arm's random-context value (hollow); the connector's
length is that arm's conditioning gain, and its direction says whether context
helped. A plain bar chart of full-context values would hide the two facts this
figure exists to show -- that the flat tokenizer's connector is longer than
ours, and that the U-Net's points the wrong way on two of the four families.

Every panel is oriented so RIGHTWARD IS BETTER. Two of the four families are
lower-is-better, and a reader scanning four panels should not have to hold
which two in their head; the axis is inverted instead, and the direction arrow
in each panel title says so.

Panels do not share an x axis: family C spans 0.000-0.027 and family A spans
0.6-1.4. They are four charts, not one chart with four scales.
"""
from __future__ import annotations

import json
from pathlib import Path

from .style import (FAM_ARM, MARKER, PALETTE, INK, INK_2, SURFACE)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "reports" / "external_baselines" / "generation_families.json"

FULL, NULL = "global_full_local", "random"
# Display order: subject, peer, then the reference classes. The lookups are
# last and muted -- they are memorisation ceilings, not competitors.
ORDER = [("pipeline", "Ours"), ("maskgit_flat", "MaskGIT-flat"),
         ("unet3d", "3D U-Net$^\\dagger$"), ("cvae3d", "3D CVAE"),
         ("dg", "$\\mathit{ref}$ DG"), ("glm", "$\\mathit{ref}$ GLM")]
# Titles are shortened to fit four panels across one column width; the full
# family names and their definitions live in the caption and in Section 5.
SHORT = {"A. Conditional accuracy": "A. Cond. accuracy",
         "B. Adherence": "B. Adherence",
         "C. Spatial placement, lookup-proof": "C. Spatial place.",
         "D. Marginal realism": "D. Marg. realism"}


def draw(fig) -> None:
    d = json.loads(SRC.read_text())
    fams = list(d["families"].items())
    axes = fig.subplots(1, len(fams), sharey=True)

    for ax, (title, blk) in zip(axes, fams):
        lower_better = blk["better"] == "low"
        ys = list(range(len(ORDER)))[::-1]
        for y, (key, lab) in zip(ys, ORDER):
            v = blk["arms"].get(FAM_ARM[key])
            if v is None:
                continue
            full, null = v[FULL], v[NULL]
            categorical = key in ("pipeline", "maskgit_flat")
            col = PALETTE[key]
            ax.plot([null, full], [y, y], lw=1.2, color=col, alpha=0.45,
                    zorder=2, solid_capstyle="round")
            ax.plot(null, y, marker="o", ms=3.6, mfc=SURFACE, mec=col,
                    mew=1.0, ls="none", zorder=3)
            ax.plot(full, y, marker=MARKER[key],
                    ms=5.5 if categorical else 4.5, color=col,
                    mfc=col if categorical else "none",
                    mec=SURFACE if categorical else col,
                    mew=1.0 if categorical else 1.2, ls="none", zorder=4)
        arrow = "$\\downarrow$" if lower_better else "$\\uparrow$"
        ax.set_title(f"{SHORT[title]} {arrow}", fontsize=6.8, color=INK,
                     pad=4)
        if lower_better:
            ax.invert_xaxis()          # rightward is better in every panel
        ax.set_ylim(-0.6, len(ORDER) - 0.4)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=6)
        ax.margins(x=0.10)

    axes[0].set_yticks(list(range(len(ORDER)))[::-1], [l for _, l in ORDER])

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], ls="-", lw=1.2, color=INK_2, alpha=0.45,
               marker="o", ms=3.6, mfc=SURFACE, mec=INK_2, mew=1.0,
               label="random context (own null)"),
        Line2D([], [], ls="none", marker="o", ms=5.5, color=INK_2,
               label="full context"),
    ]
    # Two bands above the panels, and they must not share one. The legend hangs
    # DOWN from its anchor while the annotation grows UP from its baseline, so
    # anchoring the legend at 1.24 and the text at 1.13 overlapped them in the
    # middle. Annotation on top, legend beneath it, with a gap between.
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.16), handletextpad=0.4,
               columnspacing=1.4, labelcolor=INK_2)
    fig.text(0.5, 1.21, "rightward is better in every panel; the connector is "
                        "that arm's conditioning gain", ha="center",
             va="bottom", fontsize=6.5, color=INK_2)
    fig.subplots_adjust(wspace=0.42)
