"""Standing manuscript conventions that a reader notices and a compile does not.

LaTeX is happy to typeset a figure nothing points at, a section that is a
heading followed straight by a table, and an appendix whose floats share a
counter with the main text. All three have been introduced by editing at least
once, so they are checked here rather than remembered.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
SECTIONS = sorted((PAPER / "sections").glob("*.tex"))


def _all_prose() -> str:
    return re.sub(r"^%.*$", "", "".join(p.read_text() for p in SECTIONS),
                  flags=re.M)


def test_sections_exist():
    # paper/ is gitignored; without it every test here passes vacuously.
    assert SECTIONS, f"no sections under {PAPER / 'sections'}"


def test_every_float_is_referenced():
    """A float nothing points at is a float the reader never reaches."""
    text = _all_prose()
    labels = re.findall(r"\\label\{((?:fig|tab):[^}]+)\}", text)
    refs = set(re.findall(r"\\(?:ref|autoref|Cref|cref)\{([^}]+)\}", text))
    orphans = sorted({l for l in labels if l not in refs})
    assert not orphans, f"floats never \\ref'd from prose: {orphans}"


def test_no_section_opens_straight_into_a_float():
    """A heading followed immediately by a table is a hanging section.

    Every section says in prose what its float shows before showing it.
    """
    bad = []
    for f in SECTIONS:
        body = re.sub(r"^%.*$", "", f.read_text(), flags=re.M)
        for m in re.finditer(r"\\section\{([^}]*)\}\s*(?:\\label\{[^}]*\}\s*)?"
                             r"(\\begin\{(?:table|figure)\})", body):
            bad.append(f"{f.name}: {m.group(1)!r}")
    assert not bad, "sections opening straight into a float: " + ", ".join(bad)


def test_supplement_floats_are_numbered_separately():
    """Table 2 must mean the main text and Table S2 the supplement."""
    main = (PAPER / "main.tex").read_text()
    for needed in (r"\setcounter{figure}{0}", r"\setcounter{table}{0}",
                   r"\renewcommand{\thefigure}{S\arabic{figure}}",
                   r"\renewcommand{\thetable}{S\arabic{table}}"):
        assert needed in main, f"missing after \\appendix: {needed}"
    at = main.index(r"\appendix")
    assert main.index(r"\renewcommand{\thefigure}") > at, \
        "the S-series renumbering must come after \\appendix"


def test_no_code_identifiers_in_prose():
    """No class, function, variable or path names, and no command recipes.

    Archive identifiers are not code and are allowed: dandiset numbers, object
    and subject ids, the DANDI licence tag, the conversion tool, and the two
    NWB array names a reader needs to load the right data.
    """
    allowed = {"000732", "001132", "001603", "0.241130.1903",
               "dandi:OpenAccess", "neuroconv", "sub-U",
               "obj-1wcxx1y", "obj-m6lpfz", "obj-paunmv",
               r"binary\_unit\_burst", r"binary\_ch\_burst"}
    text = _all_prose()
    found = {m for m in re.findall(r"\\texttt\{([^{}]*)\}", text)}
    assert not (found - allowed), \
        f"code identifiers in prose: {sorted(found - allowed)}"
    assert r"\begin{verbatim}" not in text, \
        "command recipe in the manuscript; point at the release README instead"
