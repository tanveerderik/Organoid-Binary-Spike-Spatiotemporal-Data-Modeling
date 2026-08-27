#!/usr/bin/env python3
"""Render the paper's figures from the committed report artifacts.

Figures are output, so they live with the other generated artifacts under
`reports/paper_figures/` and are tracked. The LaTeX source is not tracked --
`paper/` is gitignored drafting material -- so this script mirrors each rendered
figure into `paper/figures/` when that directory exists, which keeps the
Overleaf bundle in step without putting the manuscript in the code history.

Same rule as `tools/make_paper_tables.py` and
`external_baselines/diagnose_table.py`: nothing is drawn from a number typed by
hand. Each panel reads the JSON or NPZ that produced it, so a reviewer's
question about a figure resolves to a file path.

    python tools/make_paper_figures.py

Every figure below is currently a PLACEHOLDER. Replace the body of each
`draw_*` function with the real render; the plumbing, sizing and mirroring are
already correct, so a figure can be finished one at a time.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

OUT = Path("reports/paper_figures")
MIRROR = Path("paper/figures")

# Column width of the ICLR single-column body, in inches. Figures are drawn at
# the size they will occupy so that font sizes in the PDF match the surrounding
# text -- scaling a figure in LaTeX is what makes axis labels unreadable.
TEXT_W = 5.5


def _placeholder(ax, title: str, source: str) -> None:
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.01, 0.02), 0.98, 0.96, fill=False, ls="--",
                               lw=1.0, color="0.6", transform=ax.transAxes))
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=11,
            weight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.34, f"PLACEHOLDER — render from {source}", ha="center",
            va="center", fontsize=8, style="italic", color="0.35",
            transform=ax.transAxes)


def draw_f1_pipeline(ax) -> None:
    """Pipeline and data schematic. Drawn by hand, not from data."""
    _placeholder(ax, "F1 — pipeline and data",
                 "hand-drawn; constants from main.py")


def draw_f2_qualitative(ax) -> None:
    """Qualitative samples, two projections per arm.

    Do NOT scatter voxels. At a voxel rate of 1.5e-4 -- about 200 spikes in
    1.3M voxels -- a raw scatter is indistinguishable speckle at any panel size
    a 9-page paper affords, for every arm including the good ones. Two
    projections instead, per arm: a time-collapsed spatial map on a log colour
    scale (where), and an electrode-by-time raster ordered by array position
    (when). Those are the two axes the results actually separate on.
    """
    _placeholder(ax, "F2 — qualitative, two projections per arm",
                 "reports/external_baselines/dumps/*.npz")


def draw_f3_task_axis(ax) -> None:
    """Voxel and site average precision on the four completion tasks."""
    _placeholder(ax, "F3 — task axis, paired AP",
                 "task_eval_*.json, cross_model_tests.json")


def draw_f4_generation(ax) -> None:
    """Four generation families, each bar against that arm's own null."""
    _placeholder(ax, "F4 — four generation families",
                 "external_baselines/comparison.json")


FIGURES = [
    ("f1_pipeline", draw_f1_pipeline, (TEXT_W, 2.2)),
    ("f2_qualitative", draw_f2_qualitative, (TEXT_W, 3.2)),
    ("f3_task_axis", draw_f3_task_axis, (TEXT_W, 2.2)),
    ("f4_generation", draw_f4_generation, (TEXT_W, 2.2)),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, draw, figsize in FIGURES:
        fig, ax = plt.subplots(figsize=figsize)
        draw(ax)
        path = OUT / f"{name}.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")
        if MIRROR.is_dir():
            shutil.copy2(path, MIRROR / f"{name}.pdf")
            print(f"   mirrored -> {MIRROR / f'{name}.pdf'}")

    if not MIRROR.is_dir():
        print(f"\n{MIRROR} does not exist, so nothing was mirrored. That is "
              "expected on a fresh clone: the manuscript is not tracked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
