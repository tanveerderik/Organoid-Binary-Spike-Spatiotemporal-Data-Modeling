"""The reviewer flagged a rebuttal-like cadence running through the manuscript.

Two kinds of finding, and they need different tests.

The first is a list of specific sentences the reviewer quoted. Those are pinned
by string: the point is not that any one is wrong in isolation, it is that they
came back once already during editing and a reviewer should not have to flag
them twice.

The second is structural and cannot be pinned by string, because the
constructions are individually fine and only the density is the problem --
"X rather than Y" and the pseudo-cleft "the alphabet IS WHAT is shared". Those
get a per-file ceiling, set a little above the current count so ordinary
editing does not trip them and a drift back toward the old voice does.
"""
import re

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECTIONS = sorted((ROOT / "paper" / "sections").glob("*.tex"))
BODY = [p for p in SECTIONS if p.name[0] == "0" and p.name[:2] <= "07"]
APPENDIX = [p for p in SECTIONS if p.name.startswith("99")]

# The manuscript is not part of the anonymous code release: paper/ is drafting
# material and ships separately as the submission PDF. A reviewer running this
# suite from the release should see a skip and not a wall of failures, so an
# ABSENT manuscript skips while a PRESENT but empty one still fails loudly.
_HAS_PAPER = (ROOT / "paper" / "sections").is_dir()
pytestmark = pytest.mark.skipif(
    not _HAS_PAPER, reason="manuscript not present (paper/ ships separately)")


# Quoted by the reviewer, or removed for the same reason. "Beating uniform is
# worth nothing" was in an earlier PDF and is already gone; pinned anyway.
BANNED = [
    "The interpretation is one sentence",
    "What ranks above us",
    "What is ours is",
    "This is the largest gap in the paper",
    "should read the site-map row as the honest state of the art",
    "Rarefaction is not optional",
    "Rarefaction is necessary",
    "deliberately not hiding",
    "Measure at the decoder",
    "Beating uniform is worth nothing",
    "The argument is aimed at",
    "The oracle-code decomposition says why",
    "The arms above us are above us",
    "A parameter count becomes a capability claim",
    "The U-Net's one win is on the family where",
    "The third statistic is the one the corpus was assembled for",
    "The two settings worth reading twice",
    "This is the sense in which",
    "is not merely wasted capacity but",
]

# Ceilings, not zeroes. Current counts at the time of writing: body 6 and 2,
# appendix 5 and 0.
RATHER_THAN = {"body": 8, "appendix": 8}
PSEUDO_CLEFT = {"body": 3, "appendix": 2}
_CLEFT = re.compile(r"\bis what\b|\bis the one\b|\bis precisely\b")


def _flat(p: Path) -> str:
    # Sources wrap at column 79, so any phrase can straddle a newline.
    return " ".join(p.read_text().split())


def test_sections_exist():
    # paper/ is gitignored, so a checkout without the manuscript would make
    # every test here vacuously pass. Fail loudly instead.
    assert BODY, f"no body sections under {ROOT / 'paper' / 'sections'}"
    assert APPENDIX, "no appendix found"


def test_no_banned_phrases():
    hits = []
    for f in SECTIONS:
        text = _flat(f)
        for phrase in BANNED:
            if phrase in text:
                hits.append(f"{f.name}: {phrase!r}")
    assert not hits, "rhetorical phrasing still present:\n  " + "\n  ".join(hits)


def test_no_interrogative_appendix_headings():
    """Appendix headings are noun phrases, not questions to the reader.

    "Why deduplication uses relative distance and not usage frequency" reads as
    a conversational aside; "Deduplication criterion" says the same and reads
    as a paper.
    """
    bad = [h for f in APPENDIX
           for h in re.findall(r"\\paragraph\{(Why|What|How|Which)\b[^}]*\}",
                               f.read_text())]
    assert not bad, f"interrogative headings: {bad}"


def _count(paths, pattern) -> int:
    if isinstance(pattern, str):
        return sum(_flat(p).count(pattern) for p in paths)
    return sum(len(pattern.findall(_flat(p))) for p in paths)


def test_rather_than_density():
    for name, paths in (("body", BODY), ("appendix", APPENDIX)):
        n = _count(paths, "rather than")
        assert n <= RATHER_THAN[name], (
            f"{name}: {n} 'rather than' constructions, ceiling "
            f"{RATHER_THAN[name]}")


def test_pseudo_cleft_density():
    for name, paths in (("body", BODY), ("appendix", APPENDIX)):
        n = _count(paths, _CLEFT)
        assert n <= PSEUDO_CLEFT[name], (
            f"{name}: {n} pseudo-cleft constructions, ceiling "
            f"{PSEUDO_CLEFT[name]}")


# Simplified-English guards. The manuscript targets ASD-STE100 direction: plain
# punctuation, short sentences, no constructions that read as machine prose.
_COMMENT = re.compile(r"^%.*$", re.M)
_EMDASH = re.compile(r"(?<!-)---(?!-)")


def _prose(p: Path) -> str:
    """File text with comment lines dropped: the section separators are rules
    of forty hyphens and would otherwise register as em dashes."""
    return _COMMENT.sub("", p.read_text())


def test_no_em_dashes():
    """Em dashes read as machine-written and STE has no use for them.

    Every one was replaced with a comma, colon, parenthesis or a sentence
    break, chosen per site. En dashes stay: `1.4--2.6` and `841--1020` are
    numeric ranges and correct as they are.
    """
    hits = {f.name: len(_EMDASH.findall(_prose(f))) for f in SECTIONS}
    hits = {k: v for k, v in hits.items() if v}
    assert not hits, f"em dashes are back: {hits}"


def test_no_unicode_dashes_or_ellipsis():
    """A literal em dash, en dash or ellipsis character means text arrived by
    paste rather than through LaTeX, and prints wrong under pdflatex."""
    bad = {f.name: [c for c in "—–…" if c in _prose(f)]
           for f in SECTIONS}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, f"unicode punctuation in source: {bad}"


def test_no_filler_vocabulary():
    """Words that signal generated prose. Technical uses are not on the list:
    "harness" stays because the evaluation harness is a real object here."""
    filler = ["delve", "leverage", "crucial", "pivotal", "testament",
              "seamless", "showcase", "underscore", "intricate", "myriad",
              "plethora", "nuanced", "in today's", "deep dive",
              "it is worth noting"]
    hits = []
    for f in SECTIONS:
        low = " ".join(_prose(f).split()).lower()
        hits += [f"{f.name}: {w!r}" for w in filler if w in low]
    assert not hits, "filler vocabulary:\n  " + "\n  ".join(hits)
