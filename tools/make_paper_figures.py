#!/usr/bin/env python3
"""Render the paper's figures from the committed report artifacts.

Figures are output, so they live with the other generated artifacts under
`reports/paper_figures/` and are tracked. The LaTeX source is not tracked --
`paper/` is gitignored drafting material -- so this script mirrors each rendered
figure into `paper/figures/` when that directory exists, which keeps the
Overleaf bundle in step without putting the manuscript in the code history.

Same rule as `tools/make_paper_tables.py`: nothing is drawn from a number typed
by hand. Each panel reads the JSON or NPZ that produced it.

    python tools/make_paper_figures.py [f3 f4 ...]

Naming a figure renders only that one, which is what the inspect-and-iterate
loop wants; with no argument it renders all of them.
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

# Before pyplot: matplotlib stamps a CreationDate into every PDF, so without a
# fixed epoch two identical renders differ byte for byte and the reproducibility
# check in tests/ can never pass.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from MAGVIT_project.tools.paper_figs.style import TEXT_W, apply_style  # noqa: E402

OUT = ROOT / "reports" / "paper_figures"
MIRROR = ROOT / "paper" / "figures"

# Heights are a page-budget decision, not a drawing decision. The main text is
# capped at 9 pages and four full-width floats plus their captions were costing
# ~2.3 of them, so each figure is drawn at the smallest height its own content
# stays legible at. Changing a number here changes the page count; re-compile
# and re-check before keeping an edit.
FIGURES = [
    ("f1_pipeline",   "f1_pipeline",   (TEXT_W, 2.10)),
    ("f2_motifs",     "f2_motifs",     (TEXT_W, 2.45)),
    ("f3_task_axis",  "f3_task_axis",  (TEXT_W, 1.75)),
    ("f4_generation", "f4_generation", (TEXT_W, 1.9)),
    # Appendix figures. Not page-budgeted against the 9-page main text, but
    # still column width: a wider float would be scaled down on the page and
    # the voxel panels are already at the limit of what prints legibly.
    ("f5_reconstruction", "f5_reconstruction", (TEXT_W, 5.35)),
    ("f6_generation",     "f6_generation",     (TEXT_W, 5.35)),
]


def main(argv=None) -> int:
    want = set(argv or [])
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, module, figsize in FIGURES:
        if want and name not in want and name.split("_")[0] not in want:
            continue
        mod = importlib.import_module(f"MAGVIT_project.tools.paper_figs.{module}")
        fig = plt.figure(figsize=figsize)
        mod.draw(fig)
        path = OUT / f"{name}.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")
        if MIRROR.is_dir():
            shutil.copy2(path, MIRROR / f"{name}.pdf")
    if not MIRROR.is_dir():
        print(f"\n{MIRROR} does not exist, so nothing was mirrored. That is "
              "expected on a fresh clone: the manuscript is not tracked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
