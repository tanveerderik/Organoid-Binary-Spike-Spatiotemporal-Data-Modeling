"""No figure label may overrun the box it sits in.

Eyeballing a 320pt-wide figure does not catch a label that overruns its border
by two points, and at \\textwidth the reader sees it at full page width. This
renders every figure and measures each text against the patch it is centred in,
so the check is a number and not a judgement.

The bug that motivated it: Figure 1's task strip overran its box by 31 px, and
three node labels overran theirs, in a figure that had been visually reviewed.
"""
import importlib
import os

import pytest

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle   # noqa: E402

from MAGVIT_project.tools import make_paper_figures as MPF  # noqa: E402
from MAGVIT_project.tools.paper_figs.style import apply_style  # noqa: E402

# A label may sit this many points outside its border before it reads as
# broken. Rounded corners and the box pad make an exact zero too strict.
TOLERANCE_PX = 0.5


def _overflows(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        boxes = [p for p in ax.patches
                 if isinstance(p, (FancyBboxPatch, Rectangle))]
        for t in ax.texts:
            if not t.get_text().strip():
                continue
            tb = t.get_window_extent(renderer=r)
            cx, cy = (tb.x0 + tb.x1) / 2, (tb.y0 + tb.y1) / 2
            for b in boxes:
                pb = b.get_window_extent(renderer=r)
                if not (pb.x0 <= cx <= pb.x1 and pb.y0 <= cy <= pb.y1):
                    continue          # this text is not inside this box
                over = max(pb.x0 - tb.x0, tb.x1 - pb.x1,
                           pb.y0 - tb.y0, tb.y1 - pb.y1)
                if over > TOLERANCE_PX:
                    out.append(f"{t.get_text()[:40]!r} by {over:.1f}px")
                break
    return out


@pytest.mark.parametrize("name,mod,size", MPF.FIGURES,
                         ids=[f[0] for f in MPF.FIGURES])
def test_no_label_overruns_its_box(name, mod, size):
    apply_style()
    fig = plt.figure(figsize=size)
    try:
        importlib.import_module(
            f"MAGVIT_project.tools.paper_figs.{mod}").draw(fig)
        bad = _overflows(fig)
    finally:
        plt.close(fig)
    assert not bad, f"{name}: labels outside their box: " + "; ".join(bad)
