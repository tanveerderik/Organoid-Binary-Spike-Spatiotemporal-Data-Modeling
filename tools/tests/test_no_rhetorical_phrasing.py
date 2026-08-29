"""The reviewer flagged a rebuttal-like cadence running through the manuscript.

These are the specific strings called out. The point of pinning them in a test
is not that any one phrase is wrong in isolation -- it is that they came back
once already during editing, and a reviewer reading the next draft should not
have to flag them a second time.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECTIONS = sorted((ROOT / "paper" / "sections").glob("*.tex"))

# "Beating uniform is worth nothing" was in the reviewed PDF but is not in the
# current sources; an earlier edit already removed it. Not pinned.
BANNED = [
    "The interpretation is one sentence",
    "What ranks above us",
    "What is ours is",
    "This is the largest gap in the paper",
    "should read the site-map row as the honest state of the art",
    "Rarefaction is not optional",
    "deliberately not hiding",
    "Measure at the decoder",
]


def test_sections_exist():
    # paper/ is gitignored, so a checkout without the manuscript would make the
    # phrase test vacuously pass. Fail loudly instead.
    assert SECTIONS, f"no section files found under {ROOT / 'paper' / 'sections'}"


def test_no_banned_phrases():
    hits = []
    for f in SECTIONS:
        text = f.read_text()
        for phrase in BANNED:
            if phrase in text:
                hits.append(f"{f.name}: {phrase!r}")
    assert not hits, "rhetorical phrasing still present:\n  " + "\n  ".join(hits)
