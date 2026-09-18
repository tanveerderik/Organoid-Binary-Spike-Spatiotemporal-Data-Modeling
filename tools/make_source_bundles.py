#!/usr/bin/env python3
"""Build the two shippable LaTeX source packages.

Neither is a copy of `paper/`. Both are derived, because `paper/` is drafting
material and carries things that must not travel:

**ICLR bundle.** The rendered PDF is anonymous, but the *source* is not: the
`\\ificlrfinal` branches hold the author block's public GitHub URL as plaintext,
and identity in supplementary material is an ICLR desk reject. Relying on a
conditional to hide it only works if nobody opens the .tex. So the conditionals
are RESOLVED to their anonymous branch and the final-only text is deleted
outright, and the preprint-only files never enter the tree.

**arXiv bundle.** Source uploaded to arXiv is publicly downloadable. The
drafting comments -- who accepted authorship and when, whose funding is still
to be added, which disclosure still needs confirming -- are internal notes, not
archival record. Comments are stripped from both bundles.

Comment stripping keeps a trailing `%` where one ends a line, because there it
is suppressing a space rather than introducing a note.

Unreferenced figures are dropped: `f2_qualitative.pdf` is in `paper/figures/`
and in no `\\includegraphics`.

    python tools/make_source_bundles.py --dest paper/deliverables_<date>
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

SUPPORT = ["math_commands.tex", "refs.bib", "iclr2027_conference.sty",
           "iclr2027_conference.bst", "fancyhdr.sty", "natbib.sty"]

# Files that exist only to build the non-anonymous preprint.
PREPRINT_ONLY = {"authors_2.tex", "authors_3.tex", "funding.tex",
                 "acknowledgments.tex", "main_preprint.tex"}

_COND = re.compile(r"\\ificlrfinal(?P<final>.*?)\\else(?P<anon>.*?)\\fi", re.S)
_FULL_LINE_COMMENT = re.compile(r"^[ \t]*%.*\n", re.M)
_TRAILING_COMMENT = re.compile(r"(?<!\\)%.*$", re.M)


def resolve_conditionals(text: str, final: bool) -> str:
    """Replace every \\ificlrfinal...\\else...\\fi with the branch that is kept.

    The other branch is deleted, not commented out. That is the point: a
    reviewer who opens the ICLR source must not find the URL sitting in a
    dead branch.
    """
    return _COND.sub(lambda m: m.group("final" if final else "anon"), text)


def strip_comments(text: str) -> str:
    text = _FULL_LINE_COMMENT.sub("", text)
    # Keep the % itself: at end of a line it suppresses the following space.
    return _TRAILING_COMMENT.sub("%", text)


def used_figures(texts: list[str]) -> set[str]:
    names: set[str] = set()
    for t in texts:
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", t):
            names.add(Path(m.group(1)).name)
    return names


def build(dest: Path, final: bool, main_src: Path, extra: list[str]) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "sections").mkdir(parents=True)
    (dest / "tables").mkdir()
    (dest / "figures").mkdir()

    def emit(src: Path, out: Path) -> str:
        text = src.read_text(encoding="utf-8")
        if src.suffix in {".tex", ".bib"}:
            text = strip_comments(resolve_conditionals(text, final))
        out.write_text(text, encoding="utf-8")
        return text

    bodies = [emit(main_src, dest / "main.tex")]
    for sub in ("sections", "tables"):
        for f in sorted((PAPER / sub).glob("*.tex")):
            if f.name in PREPRINT_ONLY:
                continue
            bodies.append(emit(f, dest / sub / f.name))
    for name in SUPPORT + extra:
        emit(PAPER / name, dest / name)

    wanted = used_figures(bodies)
    for f in sorted((PAPER / "figures").glob("*")):
        if f.name in wanted:
            shutil.copy2(f, dest / "figures" / f.name)

    dropped = [f.name for f in sorted((PAPER / "figures").glob("*"))
               if f.name not in wanted]
    print(f"  {dest.name}: {len(wanted)} figures kept"
          + (f", dropped {', '.join(dropped)}" if dropped else ""))
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help="directory for both bundles")
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    preprint = PAPER / "main_preprint.tex"
    if not preprint.is_file():
        sys.exit("run tools/make_preprint.py --authors 3 first")

    build(dest / "iclr_source", final=False, main_src=PAPER / "main.tex",
          extra=[])
    build(dest / "arxiv_source", final=True, main_src=preprint,
          extra=["authors_3.tex", "funding.tex"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
