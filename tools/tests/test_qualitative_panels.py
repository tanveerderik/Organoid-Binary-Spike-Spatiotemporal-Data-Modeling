#!/usr/bin/env python3
"""The two qualitative appendix figures are SELECTED examples.

That makes them the easiest place in the paper to mislead by accident, so the
guards here are about provenance rather than appearance: the selection has to
come from a recorded rule over a recorded ranking, the panels have to be drawn
from the pinned evaluation protocol, and the caption's framing has to stay
paired rather than pooled.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from MAGVIT_project.tools.tests._release_guard import needs_paper, needs_reports

# The release omits the manuscript and the report artifacts;
# absent material skips, present-but-wrong still fails.
pytestmark = [needs_paper, needs_reports]


ROOT = Path(__file__).resolve().parents[2]
PANELS = ROOT / "reports" / "qualitative_panels.json"
NPZ = ROOT / "reports" / "qualitative_panels.npz"
APPENDIX = ROOT / "paper" / "sections" / "99_appendix.tex"
BUDGET = ROOT / "reports" / "eval_budget.json"


@pytest.fixture(scope="module")
def panels():
    return json.loads(PANELS.read_text())


def _flat(text: str) -> str:
    """LaTeX wraps at column 79, so any phrase worth matching straddles a
    newline. Every earlier version of this check passed vacuously."""
    return " ".join(text.split())


def test_selection_ranking_is_recorded_not_just_the_winners(panels):
    """A pick with no ranking behind it cannot be audited.

    The figure claims its rows are far above typical; that claim is only
    checkable if the population it was drawn from ships with it.
    """
    assert len(panels["recon"]["all"]) == 279, "not the full test split"
    assert len(panels["recon"]["picked"]) == 4
    assert set(panels["recon"]["picked"]) <= {r["sample"] for r in panels["recon"]["all"]}
    assert panels["selection_rule"]


def test_recon_rows_span_both_corpora(panels):
    """One contribution rests on the corpus spanning two preparation types.

    A panel figure drawn entirely from the organoid recordings would quietly
    illustrate only half of it.
    """
    by = {r["sample"]: r for r in panels["recon"]["all"]}
    recs = [by[s]["recording"] for s in panels["recon"]["picked"]]
    organoid = [r for r in recs if r.startswith("sub-U")]
    assert len(organoid) == 2 and len(recs) - len(organoid) == 2, recs
    assert len(set(recs)) == 4, "a recording is used twice"


def test_generation_rows_are_one_clip_under_all_four_tasks(panels):
    picked = panels["generation"]["picked"]
    assert [t for t, _ in picked] == ["recon", "causal", "noncausal", "spatial"]
    assert len({c for _, c in picked}) == 1, "rows are not the same clip"


def test_panels_come_from_the_pinned_evaluation_protocol(panels):
    """The generation panels must be drawn at the protocol the paper reports.

    Every number in Section 5 comes from the 70-batch budget over all 279 test
    clips and 31 recordings. An earlier dump was taken at 12 batches, which --
    because the test loader is an unshuffled DeterministicSubset -- is an
    ordered PREFIX: all 24 of its clips came from a single recording. Panels
    drawn from it would show one recording under a protocol the paper does not
    report.
    """
    pinned = json.loads(BUDGET.read_text())["budgets"]["task_axis"]["batches"]
    got = panels["gen_dump_meta"]["batches"]
    assert got >= pinned, (
        f"generation panels come from a {got}-batch dump but the pinned "
        f"protocol is {pinned}. Re-run:\n"
        f"  python -m external_baselines.task_eval --model pipeline "
        f"--batches {pinned} --dump 8 --dump-field 8\n"
        f"then python tools/make_qualitative_panels.py")


def test_every_drawn_array_exists(panels):
    npz = np.load(NPZ)
    nw = panels["windows"]
    for kind, n in (("recon", len(panels["recon"]["picked"])),
                    ("gen", len(panels["generation"]["picked"]))):
        for i in range(n):
            assert f"{kind}/{i}/sitemap" in npz.files
            for j in range(nw):
                for arr in ("field", "gt", "pred"):
                    assert f"{kind}/{i}/{j}/{arr}" in npz.files, f"{kind}/{i}/{j}/{arr}"


def test_every_panel_states_how_much_of_it_was_generated(panels):
    """The wash alone cannot carry the temporal tasks.

    A pixel is washed only where it is outside the ROI in EVERY frame of the
    window. For causal and non-causal the hole is an interval of time, so a
    window straddling the cut has no fully-observed pixel and receives no wash
    at all even when three quarters of it was visible. The per-panel generated
    fraction is therefore load-bearing, not decoration, and it is asserted for
    every panel of every row.
    """
    npz = np.load(NPZ)
    nw = panels["windows"]
    tasks = [t for t, _ in panels["generation"]["picked"]]
    for i, task in enumerate(tasks):
        fracs = [float(npz[f"gen/{i}/{j}/roi_frac"]) for j in range(nw)]
        assert all(0.0 <= f <= 1.0 for f in fracs), (task, fracs)
        if task == "recon":
            assert fracs == [1.0] * nw, "free generation observes nothing"
            assert not any(npz[f"gen/{i}/{j}/observed"].any() for j in range(nw))
        else:
            assert min(fracs) < 1.0, f"{task}: no panel is partly observed"
        # Wherever a window is entirely inside the hole in no frame at all, the
        # wash must be present -- that is the case the mark exists for.
        for j, f in enumerate(fracs):
            if f == 0.0:
                assert npz[f"gen/{i}/{j}/observed"].all(), (task, j)


def test_caption_numbers_are_the_CURRENT_ones(panels):
    """Existence is not currency.

    check_paper_numbers.py asks whether a number appears somewhere under
    reports/, which a stale value can satisfy by coincidence -- that is how a
    superseded U-Net figure survived in the appendix. These quantities are
    regenerated whenever the dump is re-run, so the caption is checked against
    the values in the file that produced the figure, not against a constant.
    """
    tex = _flat(APPENDIX.read_text())
    att = panels["generation"]["picked_attained"]
    for task, v in att.items():
        assert f"${v}$" in tex, (
            f"caption does not quote the current attained fraction for "
            f"{task} (${v}$); it was regenerated and the caption was not")
    whole = panels["generation"]["picked_attained_whole_clip"]
    lo, hi = min(whole.values()), max(whole.values())
    assert f"${lo}$--${hi}$" in tex, (
        f"caption does not quote the current whole-clip range "
        f"(${lo}$--${hi}$)")
    rows = {r["task"]: r for r in panels["generation"]["all"]}
    spatial = [r for r in panels["generation"]["all"]
               if r["task"] == "spatial"
               and [r["task"], r["clip"]] in panels["generation"]["picked"]]
    if spatial:
        f = round(spatial[0]["frac_sites_never_observed"], 3)
        assert f"${f}$" in tex, (
            f"caption does not quote the current never-observed site fraction "
            f"(${f}$)")


def test_reconstruction_caption_numbers_are_the_CURRENT_ones(panels):
    """Same currency guard for F5.

    These do not move when the generation dump is re-run, which is exactly why
    they are worth pinning: a quantity that changes rarely is one nobody
    re-reads, and the summary block is regenerated on every invocation of the
    panel script.
    """
    tex = _flat(APPENDIX.read_text())
    for sample, v in panels["recon"]["picked_attained"].items():
        assert f"${v}$" in tex, (
            f"F5 caption does not quote the current attained fraction for "
            f"{sample} (${v}$)")
    m = panels["recon"]["summary"]["map_r"]
    for field in ("median_truth", "median_model", "frac_model_above_truth"):
        assert f"${m[field]}$" in tex, f"F5 caption stale on map_r.{field}"
    lg = panels["recon"]["summary"]["logmean"]
    assert f"${lg['median_paired_change']}$" in tex, "F5 caption stale on log rate"
    assert f"${lg['frac_model_above_truth']}$" in tex, "F5 caption stale on over-production"
    ac = panels["recon"]["summary"]["active"]
    for field in ("median_truth", "median_model", "frac_model_above_truth"):
        assert f"${ac[field]}$" in tex, f"F5 caption stale on active.{field}"
    assert f"${len(panels['recon']['all'])}$ test clips" in tex


def test_captions_quote_paired_statistics_not_pooled_ones(panels):
    """Active-site ratio is where the two readings disagree.

    Marginal medians are 0.0713 and 0.0479 -- a third apart, which reads as a
    systematic shortfall. The median PER-CLIP change is 0.0 with half the clips
    moving up. The caption has to carry the paired number, and the metric rule
    in Appendix H says why.
    """
    tex = _flat(APPENDIX.read_text())
    s = panels["recon"]["summary"]["active"]
    assert s["median_paired_change"] == 0.0
    assert "median \\emph{per-clip} change is $0.0$" in tex
    assert f"${s['frac_model_above_truth']}$ of clips moving up" in tex


def test_free_generation_is_marked_as_the_reference_not_a_fourth_task(panels):
    """Task 0 observes nothing, so it is the zero-context reference and not a
    completion setting -- Appendix J says so in prose. If the figure drew it
    identically to the three masked rows, a reader would compare it to them as
    a peer, which is the reading the prose exists to prevent.
    """
    src = (ROOT / "tools" / "paper_figs" / "f6_generation.py").read_text()
    assert "REFERENCE" in src, "free generation is not styled as a reference"
    assert 'r["task"] == "recon"' in src, "the reference row is not selected"
    tex = _flat(APPENDIX.read_text())
    assert "dashed in crimson" in tex
    assert "not a completion setting" in tex


def test_captions_do_not_claim_the_spatial_map_is_followed(panels):
    """The static site map beats our placement and the paper says so.

    A caption asserting the generation "follows the assay map" would contradict
    Appendix N. The figures state an attained fraction against an explicit
    ceiling instead.
    """
    tex = _flat(APPENDIX.read_text())
    for banned in ("follows the assay map", "follows the recording map",
                   "reproduces the site map"):
        assert banned not in tex, banned
    assert "achievable" in tex and "attain" in tex


def test_both_figures_are_referenced_from_a_related_section(panels):
    """A float nobody points at is a float a reviewer never reaches."""
    tex = _flat(APPENDIX.read_text())
    assert "\\ref{fig:qual_recon}" in tex, "recon figure never referenced"
    assert "\\ref{fig:qual_gen}" in tex, "generation figure never referenced"
    # The pointers must sit in the sections the figures belong to, not only
    # inside the figure section itself.
    body = tex.split("\\section{Qualitative panels}")[0]
    assert "\\ref{fig:qual_recon}" in body and "\\ref{fig:qual_gen}" in body
