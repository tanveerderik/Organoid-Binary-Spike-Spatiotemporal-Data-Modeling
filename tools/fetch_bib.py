#!/usr/bin/env python3
"""Fetch a bibliography entry from its publisher and print it ready to paste.

Entries in `paper/refs.bib` are not to be typed. Typing them is what put a
wrong given name on a cited author, dropped three names from the middle of a
fifteen-author list, and invented an author for a data deposit -- all of which
a fetch would have made impossible. `check_refs.py` catches those afterwards;
this catches them by never letting them happen.

    python tools/fetch_bib.py 10.1016/j.celrep.2026.116984
    python tools/fetch_bib.py arXiv:2509.23896 --key patel2025neuroai
    python tools/fetch_bib.py https://doi.org/10.1038/s41593-025-02111-0
    python tools/fetch_bib.py 10.1109/JSSC.2014.2359219 --append

Sources, chosen automatically:

  DOI       content negotiation against doi.org, which routes to whichever
            agency minted it -- Crossref for journals, DataCite for deposits
  arXiv id  arxiv.org/bibtex/<id>

Output is rewritten into this file's house style: fields in a fixed order,
values aligned, author lists wrapped, non-ASCII turned into LaTeX escapes so
the entry compiles under pdflatex, and the publisher's noise fields (ISSN,
publisher, url, month, abstract) dropped. The citation key is derived as
<family><year><word> unless --key says otherwise.

What this does NOT do is decide anything. It never edits an existing entry,
and with --append it refuses to touch a key that is already present. Check the
printed entry against what you meant to cite before pasting it: the fetch
guarantees the metadata matches the identifier, not that the identifier is the
work you meant.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "paper" / "refs.bib"

UA = "fetch_bib.py/1.0 (bibliography import)"
if os.environ.get("CHECK_REFS_CONTACT"):
    UA += f" ({os.environ['CHECK_REFS_CONTACT']})"

# House style: this order, these names, everything else dropped.
FIELD_ORDER = ["title", "author", "journal", "booktitle", "volume", "number",
               "pages", "year", "doi", "note", "url"]
DROP = {"issn", "isbn", "publisher", "month", "abstract", "copyright",
        "language", "eprint", "archiveprefix", "primaryclass", "address",
        "editor", "series", "keywords", "urldate", "file", "issue_date"}

WIDTH = 78

# Enough of the Latin-1 supplement and Latin Extended-A to cover author names;
# anything outside it is reported rather than silently mangled.
ACCENTS = {
    "̀": "`", "́": "'", "̂": "^", "̃": "~", "̈": '"',
    "̊": "r", "̧": "c", "̄": "=", "̆": "u", "̌": "v",
    "̇": ".", "̨": "k", "̋": "H",
}
LITERAL = {"ø": r"{\o}", "Ø": r"{\O}", "ß": r"{\ss}", "æ": r"{\ae}",
           "Æ": r"{\AE}", "œ": r"{\oe}", "Œ": r"{\OE}", "ł": r"{\l}",
           "Ł": r"{\L}", "đ": r"{\dj}", "–": "--", "—": "---",
           "’": "'", "‘": "`", "“": "``", "”": "''", " ": " "}


def to_latex(s: str) -> str:
    """Turn accented characters into LaTeX escapes: Gänswein -> G{\\"a}nswein."""
    out = []
    for ch in s:
        if ch in LITERAL:
            out.append(LITERAL[ch])
            continue
        if ord(ch) < 128:
            out.append(ch)
            continue
        decomposed = unicodedata.normalize("NFD", ch)
        if len(decomposed) == 2 and decomposed[1] in ACCENTS:
            base, mark = decomposed[0], ACCENTS[decomposed[1]]
            out.append(f"{{\\{mark}{base}}}" if mark.isalpha()
                       else f"{{\\{mark}{base}}}")
        else:
            out.append(ch)
            print(f"warning: no LaTeX escape for {ch!r} (U+{ord(ch):04X}); "
                  "check it compiles", file=sys.stderr)
    return "".join(out)


# --------------------------------------------------------------------------

def fetch(url: str, accept: str | None = None) -> str:
    headers = {"User-Agent": UA}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


_ARXIV = re.compile(r"(?:arxiv[:/]|abs/)((?:\d{4}\.\d{4,5})(?:v\d+)?"
                    r"|(?:[a-z\-]+(?:\.[A-Z]{2})?/\d{7}))", re.I)
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.I)


def fetch_raw(identifier: str) -> tuple[str, str, str]:
    """Return (bibtex, kind, normalized id) for a DOI, arXiv id, or URL."""
    text = identifier.strip()

    m = _ARXIV.search(text)
    if m and "10." not in text.split(m.group(1))[0][-6:]:
        arxiv_id = m.group(1)
        return fetch(f"https://arxiv.org/bibtex/{arxiv_id}"), "arxiv", arxiv_id
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", text):
        return fetch(f"https://arxiv.org/bibtex/{text}"), "arxiv", text

    m = _DOI.search(text)
    if m:
        doi = m.group(1).rstrip(".,;)")
        url = "https://doi.org/" + urllib.parse.quote(doi, safe="/:")
        return fetch(url, "application/x-bibtex"), "doi", doi

    sys.exit(f"could not find a DOI or arXiv id in {identifier!r}")


# --------------------------------------------------------------------------

def parse_one(bibtex: str) -> tuple[str, dict[str, str]]:
    """Parse the single entry the publisher returned."""
    m = re.search(r"@(\w+)\s*\{", bibtex)
    if not m:
        sys.exit("the publisher did not return a BibTeX entry:\n" + bibtex[:400])
    kind = m.group(1).lower()
    i, depth = m.end(), 1
    while i < len(bibtex) and depth:
        depth += (bibtex[i] == "{") - (bibtex[i] == "}")
        i += 1
    body = bibtex[m.end():i - 1]
    _, _, rest = body.partition(",")

    fields: dict[str, str] = {}
    for fm in re.finditer(r"(\w+)\s*=\s*", rest):
        name = fm.group(1).lower()
        j = fm.end()
        if j >= len(rest):
            break
        if rest[j] == "{":
            d, k = 1, j + 1
            while k < len(rest) and d:
                d += (rest[k] == "{") - (rest[k] == "}")
                k += 1
            value = rest[j + 1:k - 1]
        elif rest[j] == '"':
            k = rest.index('"', j + 1) + 1
            value = rest[j + 1:k - 1]
        else:
            k = j
            while k < len(rest) and rest[k] not in ",\n":
                k += 1
            value = rest[j:k]
        fields[name] = " ".join(value.split())
    return kind, fields


def normalize(kind: str, fields: dict[str, str], source: str,
              ident: str) -> tuple[str, dict[str, str]]:
    out: dict[str, str] = {}

    if source == "arxiv":
        # arXiv's export is @misc with eprint/archivePrefix. The house style
        # for a preprint is @article with the id in the journal field.
        kind = "article"
        bare = ident.split("v")[0]
        fields["journal"] = f"arXiv preprint arXiv:{bare}"
        fields.setdefault("doi", f"10.48550/arXiv.{bare}")
    elif kind in {"inproceedings", "incollection", "conference"}:
        kind = "inproceedings"

    for name in FIELD_ORDER:
        value = fields.get(name, "").strip()
        if not value or name in DROP:
            continue
        if name == "author":
            value = " and ".join(a.strip() for a in value.split(" and "))
        if name == "pages":
            value = value.replace("–", "--")
            if "--" not in value and "-" in value:
                value = re.sub(r"\s*-+\s*", "--", value)
        out[name] = to_latex(value)

    if source == "doi":
        # Crossref lowercases the DOI it echoes back; keep the one we asked
        # with, which is the form the publisher prints.
        out["doi"] = ident
    if "doi" in out and "url" in out:
        # A DOI already resolves; a url alongside it is noise in this bib.
        del out["url"]
    return kind, out


_STOPWORDS = {"a", "an", "the", "of", "on", "in", "for", "and", "with", "to",
              "using", "from", "at", "by", "is", "are"}


def make_key(fields: dict[str, str]) -> str:
    author = fields.get("author", "")
    first = author.split(" and ")[0] if author else ""
    if "," in first:
        family = first.split(",")[0]
    else:
        family = first.split()[-1] if first.split() else "anon"
    family = re.sub(r"[^A-Za-z]", "", unicodedata.normalize(
        "NFKD", re.sub(r"[{}\\]", "", family))).lower()
    year = re.sub(r"[^0-9]", "", fields.get("year", ""))[:4]
    words = re.sub(r"[^A-Za-z ]", " ", re.sub(r"[{}\\]", "",
                                              fields.get("title", ""))).split()
    tail = next((w.lower() for w in words if w.lower() not in _STOPWORDS
                 and len(w) > 3), "")
    return f"{family}{year}{tail}"


def render(kind: str, key: str, fields: dict[str, str]) -> str:
    pad = max((len(f) for f in fields), default=5)
    lines = [f"@{kind}{{{key},"]
    for name, value in fields.items():
        prefix = f"  {name.ljust(pad)} = {{"
        indent = " " * len(prefix)
        wrapped, line = [], prefix
        for word in value.split():
            if len(line) + len(word) + 1 > WIDTH and line.strip() != prefix.strip():
                wrapped.append(line.rstrip())
                line = indent + word + " "
            else:
                line += word + " "
        wrapped.append(line.rstrip())
        lines.append("\n".join(wrapped) + "},")
    lines[-1] = lines[-1][:-1]  # no comma after the last field
    lines.append("}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("identifier", help="DOI, arXiv id, or a URL holding one")
    ap.add_argument("--key", help="citation key (default: <family><year><word>)")
    ap.add_argument("--append", action="store_true",
                    help="append to paper/refs.bib instead of printing")
    ap.add_argument("--bib", default=str(BIB))
    args = ap.parse_args()

    try:
        raw, source, ident = fetch_raw(args.identifier)
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        sys.exit(f"fetch failed for {args.identifier}: {exc}")

    kind, fields = parse_one(raw)
    kind, fields = normalize(kind, fields, source, ident)
    if not fields.get("author"):
        print("warning: the record carries no author list; check it by hand",
              file=sys.stderr)
    key = args.key or make_key(fields)
    entry = render(kind, key, fields)

    if not args.append:
        print(entry)
        return 0

    bib_path = Path(args.bib)
    existing = bib_path.read_text(encoding="utf-8")
    if re.search(r"@\w+\s*\{\s*" + re.escape(key) + r"\s*,", existing):
        sys.exit(f"{key} is already in {bib_path}; not touching it")
    with bib_path.open("a", encoding="utf-8") as fh:
        fh.write("\n" + entry + "\n")
    print(f"appended {key} to {bib_path}")
    print("now run: python tools/check_refs.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
