#!/usr/bin/env python3
"""The paper's figures must render, carry no placeholder, and be reproducible.

Byte stability is the load-bearing one. matplotlib stamps a creation date into
every PDF unless SOURCE_DATE_EPOCH is pinned before pyplot is imported, so
without it two identical renders differ and "regenerate before every compile"
silently churns the manuscript.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from MAGVIT_project.tools.tests._release_guard import needs_paper, needs_reports

# The release omits the manuscript and the report artifacts;
# absent material skips, present-but-wrong still fails.
pytestmark = [needs_paper, needs_reports]


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "paper_figures"
GEN = ROOT / "tools" / "make_paper_figures.py"
NAMES = ("f1_pipeline", "f2_motifs", "f3_task_axis", "f4_generation")


def _render():
    subprocess.run([sys.executable, str(GEN)], cwd=ROOT, check=True,
                   env={**__import__("os").environ,
                        "PYTHONPATH": str(ROOT.parent)})


def _hashes():
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(OUT.glob("*.pdf"))}


def test_all_four_figures_render():
    _render()
    for name in NAMES:
        p = OUT / f"{name}.pdf"
        assert p.exists() and p.stat().st_size > 2000, name


def test_no_placeholder_survives():
    for name in NAMES:
        assert b"PLACEHOLDER" not in (OUT / f"{name}.pdf").read_bytes(), name


def test_figures_are_byte_stable_across_runs():
    _render()
    first = _hashes()
    _render()
    assert first == _hashes(), "figures are not reproducible run to run"


def test_f1_shows_which_context_reaches_which_module():
    """F1 exists to answer a question prose answers badly.

    A reviewer asked for the dataflow: the blank branch leaving before the
    quantiser, the ladder, the flattened alphabet, both conditioning codes with
    their routes into the two priors, and the task mask. If any of those labels
    disappears the figure has stopped doing its job, so they are asserted
    against the rendered PDF's text rather than against the source.
    """
    _render()
    text = subprocess.run(
        ["pdftotext", str(OUT / "f1_pipeline.pdf"), "-"],
        check=True, capture_output=True, text=True).stdout
    flat = " ".join(text.split())
    for label in ("clip", "patchify", "blank token", "residual ladder",
                  "flatten", "alphabet", "assay", "Stage-1 mapper",
                  "Stage-3 trunk", "activity prior", "motif prior", "decoder",
                  "task", "free generation", "causal", "noncausal", "spatial"):
        assert label in flat, f"F1 no longer labels {label!r}"
    # The asymmetry corrected in this revision: the activity prior does NOT
    # read the assay code through the frozen mapper.
    assert "no mapper" in flat, "F1 no longer shows the raw-code path"
    # LaTeX escapes do not survive matplotlib's default text path; a literal
    # backslash here means a percent sign was written as "\\%".
    assert "\\%" not in flat, "an escaped per-cent leaked into the figure"


def test_f1_is_tall_enough_to_be_a_diagram():
    """The height is a page-budget decision and a legibility decision at once.

    It was 1.55in when the figure was four text boxes. A dataflow with two rows
    does not fit in that, and silently shrinking it back would make the figure
    unreadable rather than making the paper shorter -- the prose it replaces
    would have to come back.
    """
    import re
    src = GEN.read_text()
    line = next(l for l in src.splitlines() if '"f1_pipeline"' in l)
    height = float(re.search(r"TEXT_W,\s*([0-9.]+)", line).group(1))
    assert height >= 2.0, f"F1 rendered at {height}in; it needs two rows"
