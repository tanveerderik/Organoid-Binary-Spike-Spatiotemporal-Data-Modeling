#!/usr/bin/env python3
"""Check that every number in the manuscript's prose exists in an artifact.

The generated tables cannot drift -- a script writes them. Prose can, and this
is where drift actually happens: a value quoted in a sentence stays put when
the run behind it is repeated at a different protocol. The paper is not tracked
in git, so a diff will not catch it either.

The check is deliberately crude and deliberately loud. It pulls every numeric
literal out of `paper/sections/*.tex` and `paper/main.tex`, then looks for it
anywhere in the committed JSON under `reports/` -- at the precision it is
quoted with, and also one digit either side, since prose rounds. Anything it
cannot find is printed for a human to resolve. A hit is not proof the number is
used correctly; a miss is strong evidence it is wrong.

    python tools/check_paper_numbers.py [--quiet]

Exit status is 1 if any number is unaccounted for.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = sorted((ROOT / "paper" / "sections").glob("*.tex")) + [ROOT / "paper" / "main.tex"]
REPORTS = ROOT / "reports"

# Structural constants: shapes, counts and settings that are properties of the
# design rather than results of a run. They are stated in the Method and are not
# expected to appear in a report artifact.
STRUCTURAL = {
    "6", "15", "14", "48", "120", "224", "220", "961", "1024", "32", "8", "4",
    "64", "128", "9", "3", "2", "1", "0", "12", "16", "31", "26400", "26880",
    "17.5", "20", "300", "600", "200", "100", "5", "7", "10", "36", "288",
    "1260", "210", "312", "0.05", "0.5", "0.1", "2.0", "6000", "4000",
    "201104178", "201306108", "000732", "001132", "001603", "0.241130.1903",
    "2022", "2024", "2025", "1.0", "0.0", "50", "1756", "117", "40", "24", "80",
    "56", "63", "0.02", "0.2", "0.25", "1.5", "2.4", "2133", "1069", "426",
    "638", "279", "70", "20260821", "20260822", "833", "26.9",
    # The hole distributions in Section 3. These are the parameters of the
    # task definition -- the ranges dataset.py draws from -- not measurements,
    # so no run reports them.
    "75", "30", "60", "0.30", "0.60", "0.75",
}

NUM = re.compile(r"(?<![A-Za-z0-9_.])(\d+(?:[.,]\d+)*)(?![A-Za-z0-9_])")
SKIP_ENV = re.compile(r"\\(?:label|ref|cite\w*|input|includegraphics|"
                      r"usepackage|newcommand|documentclass|bibliography\w*)"
                      r"\s*(\{[^{}]*\}|\[[^\]]*\])*")
COMMENT = re.compile(r"(?<!\\)%.*$", re.M)


def _corpus(only: str | None = None) -> str:
    """Every committed report artifact, concatenated as text.

    Numbers are matched against the raw JSON text rather than against parsed
    values because prose rounds and a parsed comparison would need a tolerance
    per field. Substring matching over the serialised form finds `0.2635`
    inside `0.26351...` for free.
    """
    parts = []
    for p in sorted(REPORTS.rglob("*.json")):
        if only and not p.match(only):
            continue
        try:
            parts.append(p.read_text())
        except (OSError, UnicodeDecodeError):
            continue
    if not only:
        for p in sorted((ROOT / "paper" / "tables").glob("*.tex")):
            parts.append(p.read_text())
    return "\n".join(parts)


def _variants(tok: str) -> list[str]:
    """The token as written, plus the neighbouring roundings.

    `0.72` in prose can be `0.7205` in the artifact, and `2.04` can be
    `2.0351`. Trailing-digit neighbours catch round-half-up in either
    direction without accepting an arbitrary value.
    """
    plain = tok.replace(",", "")
    out = {plain, tok}
    if "." in plain:
        whole, frac = plain.split(".", 1)
        try:
            scale = 10 ** len(frac)
            v = round(float(plain) * scale)
            for d in (-1, 1):
                out.add(f"{(v + d) / scale:.{len(frac)}f}")
        except ValueError:
            pass
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--only", default=None,
                    help="restrict the corpus to report paths matching this "
                         "glob, e.g. '*task_eval_*.json'. Use after a protocol "
                         "change: the default corpus includes every historical "
                         "report, so a stale value still finds a match.")
    ap.add_argument("--files", default=None,
                    help="restrict the check to these manuscript files "
                         "(comma-separated basenames)")
    a = ap.parse_args()

    corpus = _corpus(a.only)
    wanted = set(a.files.split(",")) if a.files else None
    missing: list[tuple[str, int, str, str]] = []
    checked = 0
    for tex in TEX:
        if wanted and tex.name not in wanted:
            continue
        for lineno, line in enumerate(tex.read_text().splitlines(), 1):
            line = COMMENT.sub("", line)
            line = SKIP_ENV.sub(" ", line)
            for m in NUM.finditer(line):
                tok = m.group(1)
                if tok.replace(",", "").replace(".", "") == "":
                    continue
                if tok.replace(",", "") in STRUCTURAL:
                    continue
                checked += 1
                if not any(v in corpus for v in _variants(tok)):
                    missing.append((tex.name, lineno, tok, line.strip()[:96]))

    if not a.quiet:
        for name, lineno, tok, ctx in missing:
            print(f"{name}:{lineno}  {tok}\n    {ctx}")
    print(f"\n{checked} numbers checked, {len(missing)} unaccounted for "
          f"({len(STRUCTURAL)} structural constants exempt)")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
