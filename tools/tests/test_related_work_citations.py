"""The revision requires the adjacent spike-generation literature to be cited in
Related Work, and requires the section not to grow while doing it -- the new
sentences are paid for out of explanatory prose, not appended.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELATED = ROOT / "paper" / "sections" / "02_related.tex"
BIB = ROOT / "paper" / "refs.bib"

REQUIRED_KEYS = ["kapoor2024ldns", "molano2018spikegan"]
MAX_LINES = 56  # the length of the section before the citations were added


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
    n = len(RELATED.read_text().splitlines())
    assert n <= MAX_LINES, f"Related Work grew to {n} lines (cap {MAX_LINES})"
