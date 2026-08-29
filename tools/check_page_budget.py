#!/usr/bin/env python3
"""Assert the scientific body fits the ICLR page limit.

The body is Sections 1-7. The Ethics, Reproducibility and Use-of-AI statements
sit between the Conclusion and the References; `paper/main.tex` records that they
do NOT count toward the limit.

    !! UNVERIFIED against the official ICLR 2027 call for papers as of
    !! 2026-08-29. If the CFP counts all non-reference prose, lower LIMIT by the
    !! number of pages the statements occupy and re-run. The difference is large:
    !! under the current rule the body is over by four typeset lines; under the
    !! other one it is over by roughly a page.

Counting by hand from a rendered PDF is what produced the earlier, wrong claim
that the paper already fitted in nine pages. This is the only measurement that
should be quoted.

    python tools/check_page_budget.py            # compile, then check
    python tools/check_page_budget.py --pdf X    # check an existing PDF

Exit status is 0 when the body fits and 1 when it does not, so this can gate a
commit or a loop.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
LIMIT = 9

# pdftotext letterspaces the small-caps headings this template uses, so
# "REFERENCES" can arrive as "R EFERENCES" or "R E F E R E N C E S". Match the
# letters with optional gaps rather than the literal word.
#
# Case-SENSITIVE, and the capital R is the whole discriminator. This template
# renders the bibliography heading as plain "References" (measured, not assumed
# -- the statement headings ARE letterspaced small caps, but this one is not, so
# both spellings are accepted). Ordinary prose says "memorisation references"
# and "oracle-code references" in lower case on pages 2 and 6; matching
# case-insensitively finds those and reports the bibliography starting on
# page 2.
_REFS = re.compile(r"R\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S"
                   r"|R\s*e\s*f\s*e\s*r\s*e\s*n\s*c\s*e\s*s")
_STMT = re.compile(
    r"(E\s*THICS\s+S\s*TATEMENT|R\s*EPRODUCIBILITY\s+S\s*TATEMENT|"
    r"U\s*SE\s+OF\s+AI\s+S\s*TATEMENT)", re.I)
# The running header and the template's margin line numbers precede the body on
# every page and must be stripped before asking what a page opens with.
_HEADER = re.compile(r"Under review as a conference paper at ICLR\s*\d{0,4}", re.I)
_LEADING_LINENOS = re.compile(r"^(?:\d{1,4}\s+)+")


def _page_body(page_text: str) -> str:
    """Page text with the running header and margin line numbers removed."""
    t = " ".join(page_text.split())
    t = _HEADER.sub("", t, count=1).strip()
    return _LEADING_LINENOS.sub("", t).strip()


def _opens_with(pattern: re.Pattern, page_text: str) -> bool:
    """True when `pattern` is the first thing on the page after the header.

    Position, not mere presence. Section 5 uses the words "memorisation
    references" and "oracle-code references" in running prose, so searching the
    whole page for /references/i finds the bibliography on page 2. A heading is
    a heading only where it opens a page.
    """
    m = pattern.search(_page_body(page_text))
    return m is not None and m.start() == 0


def body_end_page(pdf_text: str) -> int:
    """1-based page on which Sections 1-7 end.

    `pdf_text` is `pdftotext` output, pages separated by form feeds. Pages that
    carry nothing but the required statements are walked back over; a page that
    carries body prose *and* a statement heading still counts as body, which is
    the case that matters when the Conclusion spills.
    """
    pages = pdf_text.split("\f")
    refs_page = None
    for i, page in enumerate(pages, start=1):
        # Searched anywhere on the page, not just at the top: the statements run
        # on, so the bibliography commonly begins partway down the page that
        # starts with the tail of the Reproducibility statement.
        if _REFS.search(_page_body(page)):
            refs_page = i
            break
    if refs_page is None:
        raise ValueError("no References heading found in the PDF text")

    page = refs_page - 1
    while page >= 1 and _opens_with(_STMT, pages[page - 1]):
        page -= 1
    return page


def _compile(outdir: Path) -> Path:
    tectonic = shutil.which("tectonic") or "/home/derik/.local/bin/tectonic"
    subprocess.run([tectonic, "-X", "compile", "main.tex", "--outdir", str(outdir)],
                   cwd=PAPER, check=True, capture_output=True)
    return outdir / "main.pdf"


def _text(pdf: Path) -> str:
    out = pdf.with_suffix(".txt")
    subprocess.run(["pdftotext", str(pdf), str(out)], check=True, capture_output=True)
    return out.read_text(encoding="utf-8", errors="replace")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=None, help="check this PDF instead of compiling")
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(args.pdf) if args.pdf else _compile(Path(tmp))
        end = body_end_page(_text(pdf))

    over = end - LIMIT
    print(f"scientific body (Sections 1-7) ends on page {end}; limit is {LIMIT}")
    if over > 0:
        print(f"OVER by {over} page(s)")
        return 1
    print("within the limit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
