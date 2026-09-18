"""The revision requires the adjacent spike-generation literature to be cited in
Related Work, and requires the section not to drift back into a survey.

The cap was a line count until 2026-09-11, which measured the wrapping rather
than the content. It is now a word count, and it has moved twice. The
collaborator review asked for the recent autoregressive spike-forecasting work,
which cost about 120 words after two existing sentences were compressed to pay
for part of it. The provenance audit of 2026-09-12 then added Wu et al. 2026,
about 60 words, and paid for two thirds of it by deleting a peer-method sentence
that Section 5 already carried and by not restating the sorted-unit contrast
twice. The binding constraint on any further growth is the page gate,
`tools/check_page_budget.py`, not this number.
"""
from pathlib import Path

from MAGVIT_project.tools.tests._release_guard import needs_paper, needs_reports

# The release omits the manuscript and the report artifacts;
# absent material skips, present-but-wrong still fails.
pytestmark = needs_paper


ROOT = Path(__file__).resolve().parents[2]
RELATED = ROOT / "paper" / "sections" / "02_related.tex"
BIB = ROOT / "paper" / "refs.bib"

REQUIRED_KEYS = ["kapoor2024ldns", "molano2018spikegan",
                 "minnick2026spikeprophecy", "minnick2026implicit",
                 "wu2026generative"]
MAX_WORDS = 550  # measured 2026-09-12, after Wu et al. 2026 was cited


def test_bib_has_the_adjacent_generative_work():
    bib = BIB.read_text()
    missing = [k for k in REQUIRED_KEYS if f"{{{k}," not in bib]
    assert not missing, f"missing bib entries: {missing}"


def test_related_work_cites_them():
    text = RELATED.read_text()
    missing = [k for k in REQUIRED_KEYS if k not in text]
    assert not missing, f"not cited in Related Work: {missing}"


def test_they_are_marked_adjacent_not_equivalent():
    # The reviewer asked for these to be positioned as adjacent work rather than
    # as methods addressed to the same object, and separately warned against
    # claiming priority over neural spike generation in general.
    text = RELATED.read_text()
    assert "adjacent" in text


def test_related_work_did_not_grow():
    n = len(RELATED.read_text().split())
    assert n <= MAX_WORDS, f"Related Work grew to {n} words (cap {MAX_WORDS})"
