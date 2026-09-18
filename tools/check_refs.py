#!/usr/bin/env python3
r"""Verify every entry in paper/refs.bib against an authoritative record.

The failure this exists to catch is silent. A wrong given name on a real
person's paper, a dropped author in the middle of a sixteen-author list, or a
lab head promoted to an author of a data deposit all survive every other gate
in this repository: check_paper_numbers.py checks numbers, make_release.py
checks anonymity, and nothing checks that a cited human's name is their name.
An audit on 2026-09-18 found four such defects among 36 entries, every one of
them in a list that had been typed out rather than fetched.

The upstream fix is `tools/fetch_bib.py`, which imports an entry from its DOI
or arXiv id so there is nothing to mistype. This script is the net under it,
for entries already in the file and for the cases where a publisher's own
record is wrong.

Sources, in the order they are tried:

  arXiv id   -> export.arxiv.org/api/query, falling back to scraping
                arxiv.org/abs/<id> (the API answers 406 from some networks).
                The id comes from the doi 10.48550/arXiv.NNNN or from a
                journal field reading "arXiv preprint arXiv:NNNN"
  DOI        -> api.crossref.org/works/<doi>, falling back to
                api.datacite.org/dois/<doi> for deposits (DANDI mints
                DataCite DOIs, which Crossref answers 404 for)
  no id      -> Crossref bibliographic search on the title, then arXiv title
                search; a hit is accepted ONLY if the normalized title matches
                exactly, and a match found this way verifies the author list
                but not the venue, so those entries are reported with
                scope "authors" rather than "full"

An entry that reaches none of those is reported as UNVERIFIED rather than
passed: quietly counting it as clean would misrepresent the coverage. Run with
--list-unverified to see them.

Comparison is deliberately tolerant where bibliographic records legitimately
disagree, and strict where a defect would hide:

  strict    the set, count and ORDER of family names
  strict    each given name where both records spell it out ("Daniel" against
            "Dietmar" fails; comparing only the initial is exactly what lets a
            wrong first name through)
  tolerant  abbreviation ("J." against "Jonathan") and middle initials
            ("Kevin" against "Kevin C.")
  tolerant  where a multi-word name divides, when all the words agree
            ("Jones, Ian Lloyd" against given "Ian", family "Lloyd Jones")
  tolerant  accents, LaTeX escapes, case, punctuation, particle spacing
  tolerant  title case and the year when a record carries both a print and an
            online date (either may be cited)

Output is written to reports/refs_verification.json so that the pytest guard
can assert the verdict offline. The report records a hash of each entry as it
was checked, so editing an entry invalidates its verdict rather than leaving a
stale pass behind.

    python tools/check_refs.py                 # check every entry
    python tools/check_refs.py --key patel2025neuroai
    python tools/check_refs.py --list-unverified

Exit status is 1 if any entry MISMATCHES, 0 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "paper" / "refs.bib"
REPORT = ROOT / "reports" / "refs_verification.json"

# Crossref routes requests carrying a contact address to its polite pool, and
# answers faster for them. That address identifies the sender, so it comes from
# the environment and never from this file: the anonymous code release ships
# tools/, and a mailto here would be a deanonymizing string sitting in the
# supplementary material.
#
#     export CHECK_REFS_CONTACT="mailto:you@example.edu"
#
UA = "check_refs.py/1.0 (bibliography verification)"
if os.environ.get("CHECK_REFS_CONTACT"):
    UA += f" ({os.environ['CHECK_REFS_CONTACT']})"
PAUSE = 0.4  # seconds between requests


# --------------------------------------------------------------------------
# bib parsing
# --------------------------------------------------------------------------

def parse_bib(text: str) -> list[dict]:
    """Parse @type{key, field = {value}, ...} entries. Brace-aware."""
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{", text):
        kind = m.group(1).lower()
        if kind in {"comment", "preamble", "string"}:
            continue
        i = m.end()
        depth = 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[m.end():i - 1]
        key, _, rest = body.partition(",")
        entries.append({
            "key": key.strip(),
            "type": kind,
            "fields": parse_fields(rest),
            "raw": body,
        })
    return entries


def parse_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for m in re.finditer(r"(\w+)\s*=\s*", text):
        name = m.group(1).lower()
        i = m.end()
        if i >= len(text):
            break
        if text[i] == "{":
            depth, j = 1, i + 1
            while j < len(text) and depth:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            value = text[i + 1:j - 1]
        elif text[i] == '"':
            j = text.index('"', i + 1) + 1
            value = text[i + 1:j - 1]
        else:
            j = i
            while j < len(text) and text[j] not in ",\n":
                j += 1
            value = text[i:j]
        fields[name] = " ".join(value.split())
    return fields


# --------------------------------------------------------------------------
# name normalization
# --------------------------------------------------------------------------

_ACCENT_CMD = re.compile(r"\\[`'\"^~=.uvHtcdbr]\s*")
_OTHER_CMD = re.compile(r"\\[a-zA-Z]+\s*")


def deaccent(s: str) -> str:
    """Strip LaTeX escapes and Unicode accents: {G{\"a}nswein} -> ganswein."""
    s = _ACCENT_CMD.sub("", s)
    s = _OTHER_CMD.sub("", s)          # \'{e} handled above; \ss etc. dropped
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s


def norm_name_part(s: str) -> str:
    s = deaccent(s).lower()
    s = re.sub(r"[^a-z ]+", " ", s)    # hyphens, apostrophes, periods out
    return " ".join(s.split())


def split_bib_authors(field: str) -> list[str]:
    """Split on ' and ' at brace depth 0."""
    out, buf, depth, i = [], [], 0, 0
    tokens = field.split(" and ")
    # Re-join tokens that were split inside braces (e.g. "{Smith and Sons}").
    for tok in tokens:
        buf.append(tok)
        depth += tok.count("{") - tok.count("}")
        if depth <= 0:
            out.append(" and ".join(buf))
            buf, depth = [], 0
    if buf:
        out.append(" and ".join(buf))
    return [a.strip() for a in out if a.strip()]


def bib_author_parts(author: str) -> tuple[str, str]:
    """Return (family, given) from either 'Family, Given' or 'Given Family'."""
    if "," in author:
        family, given = author.split(",", 1)
        return norm_name_part(family), norm_name_part(given)
    words = deaccent(author).split()
    if not words:
        return "", ""
    # Particles stay with the family name: "van den Oord", "van der Molen".
    particles = {"van", "von", "del", "della", "de", "der", "den", "di", "da",
                 "dos", "la", "le", "bin", "ibn"}
    cut = len(words) - 1
    while cut > 0 and words[cut - 1].lower() in particles:
        cut -= 1
    return (norm_name_part(" ".join(words[cut:])),
            norm_name_part(" ".join(words[:cut])))


def first_initial(given: str) -> str:
    return given[0] if given else ""


def given_conflict(bib_given: str, ref_given: str) -> bool:
    """True when two given names cannot be the same person's.

    Tolerant of abbreviation, strict about spelling. 'Kevin' against
    'Kevin C.' agrees; 'J.' against 'Jonathan' agrees; 'Daniel' against
    'Dietmar' does not, even though both start with D -- comparing only the
    initial is what lets a wrong first name through, which is the defect this
    whole script exists to catch.
    """
    b, r = bib_given.split(), ref_given.split()
    if not b or not r:
        return bool(b) != bool(r)
    if first_initial(b[0]) != first_initial(r[0]):
        return True
    # Both spelled out in full: they must match exactly.
    if len(b[0]) > 1 and len(r[0]) > 1 and b[0] != r[0]:
        return True
    return False


# --------------------------------------------------------------------------
# record fetching
# --------------------------------------------------------------------------

def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def record_author(family: str, given: str) -> tuple[str, str]:
    """Normalize one (family, given) pair from a metadata record.

    IEEE records reach Crossref with the whole name in `family` and `given`
    empty ("Yihui Chen", ""). Splitting those keeps a malformed record from
    being reported as a defect in the bibliography.
    """
    family, given = family.strip(), given.strip()
    if not given and " " in family:
        return bib_author_parts(family)
    return norm_name_part(family), norm_name_part(given)


def crossref_message(msg: dict) -> dict:
    authors = []
    for a in msg.get("author", []):
        authors.append(record_author(a.get("family", ""), a.get("given", "")))
    years = []
    for field in ("published-print", "published-online", "issued"):
        parts = msg.get(field, {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            years.append(int(parts[0][0]))
    return {
        "source": "crossref",
        "scope": "full",
        "title": (msg.get("title") or [""])[0],
        "authors": authors,
        "venue": (msg.get("container-title") or [""])[0],
        "years": sorted(set(years)),
        "volume": msg.get("volume"),
        "issue": msg.get("issue"),
        "page": msg.get("page"),
    }


def crossref_record(doi: str) -> dict:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    try:
        return crossref_message(json.loads(fetch(url))["message"])
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    return datacite_record(doi)


def datacite_record(doi: str) -> dict:
    """DANDI and other repositories mint DataCite DOIs, not Crossref ones."""
    url = "https://api.datacite.org/dois/" + urllib.parse.quote(doi, safe="/")
    attrs = json.loads(fetch(url))["data"]["attributes"]
    authors = []
    for a in attrs.get("creators", []):
        family, given = a.get("familyName", ""), a.get("givenName", "")
        if not family and not given:
            family, given = bib_author_parts(a.get("name", ""))
            authors.append((family, given))
        else:
            authors.append(record_author(family, given))
    titles = attrs.get("titles") or [{}]
    year = attrs.get("publicationYear")
    return {
        "source": "datacite",
        "scope": "full",
        "title": titles[0].get("title", ""),
        "authors": authors,
        "venue": (attrs.get("publisher") or {}).get("name")
                 if isinstance(attrs.get("publisher"), dict)
                 else attrs.get("publisher", ""),
        "years": [int(year)] if year else [],
        "volume": None, "issue": None, "page": None,
    }


ATOM = "{http://www.w3.org/2005/Atom}"
_ABS_TITLE = re.compile(r'<h1 class="title[^"]*">(?:<span[^>]*>.*?</span>)?'
                        r'(.*?)</h1>', re.S)
_ABS_AUTHORS = re.compile(r'<div class="authors">(.*?)</div>', re.S)
_ABS_LINK = re.compile(r"<a [^>]*>(.*?)</a>", re.S)
_ABS_DATE = re.compile(r"Submitted on \d{1,2} \w+ (\d{4})")
_TAG = re.compile(r"<[^>]+>")


def _detag(s: str) -> str:
    import html
    return " ".join(html.unescape(_TAG.sub(" ", s)).split())


def arxiv_api_record(arxiv_id: str) -> dict:
    url = ("http://export.arxiv.org/api/query?id_list="
           + urllib.parse.quote(arxiv_id))
    root = ET.fromstring(fetch(url))
    entry = root.find(ATOM + "entry")
    if entry is None or entry.findtext(ATOM + "title") is None:
        raise LookupError(f"arXiv has no entry for {arxiv_id}")
    authors = [bib_author_parts((a.findtext(ATOM + "name") or "").strip())
               for a in entry.findall(ATOM + "author")]
    dates = [entry.findtext(ATOM + "published") or "",
             entry.findtext(ATOM + "updated") or ""]
    years = sorted({int(d[:4]) for d in dates if d[:4].isdigit()})
    return {
        "source": "arxiv", "scope": "full",
        "title": " ".join((entry.findtext(ATOM + "title") or "").split()),
        "authors": authors, "venue": "arXiv", "years": years,
        "volume": None, "issue": None, "page": None,
    }


def arxiv_abs_record(arxiv_id: str) -> dict:
    """Scrape arxiv.org/abs. The API answers 406 from some networks."""
    html_text = fetch(f"https://arxiv.org/abs/{arxiv_id}").decode(
        "utf-8", "replace")
    m_title = _ABS_TITLE.search(html_text)
    m_auth = _ABS_AUTHORS.search(html_text)
    if not m_title or not m_auth:
        raise LookupError(f"could not parse arxiv.org/abs/{arxiv_id}")
    authors = [bib_author_parts(_detag(a))
               for a in _ABS_LINK.findall(m_auth.group(1))]
    years = [int(y) for y in _ABS_DATE.findall(html_text)]
    return {
        "source": "arxiv-abs", "scope": "full",
        "title": _detag(m_title.group(1)),
        "authors": authors, "venue": "arXiv", "years": sorted(set(years)),
        "volume": None, "issue": None, "page": None,
    }


def arxiv_record(arxiv_id: str) -> dict:
    try:
        return arxiv_api_record(arxiv_id)
    except (urllib.error.HTTPError, urllib.error.URLError, ET.ParseError):
        return arxiv_abs_record(arxiv_id)


def search_record(title: str) -> dict:
    """Find a record by title alone, for entries carrying no identifier.

    Accepted only on an exact normalized-title match, because a near match is
    how you end up verifying an entry against a different paper. A hit
    establishes the author list; it does not establish that the entry cites
    the right venue, so the caller narrows the comparison accordingly.
    """
    want = norm_title(title)
    if not want:
        raise LookupError("entry has no title to search on")

    url = ("https://api.crossref.org/works?rows=5&select=title,author,"
           "container-title,volume,issue,page,published-print,"
           "published-online,issued&query.bibliographic="
           + urllib.parse.quote(title))
    try:
        for item in json.loads(fetch(url))["message"]["items"]:
            if norm_title((item.get("title") or [""])[0]) == want:
                rec = crossref_message(item)
                rec["source"] = "crossref-search"
                rec["scope"] = "authors"
                return rec
    except (urllib.error.HTTPError, urllib.error.URLError,
            json.JSONDecodeError, KeyError):
        pass

    time.sleep(PAUSE)
    query = urllib.parse.quote(f'ti:"{title}"')
    try:
        root = ET.fromstring(fetch(
            f"http://export.arxiv.org/api/query?search_query={query}&max_results=5"))
        entries = root.findall(ATOM + "entry")
    except (urllib.error.HTTPError, urllib.error.URLError, ET.ParseError):
        entries = []
    for entry in entries:
        if norm_title(" ".join((entry.findtext(ATOM + "title") or "").split())) == want:
            authors = [bib_author_parts((a.findtext(ATOM + "name") or "").strip())
                       for a in entry.findall(ATOM + "author")]
            return {
                "source": "arxiv-search", "scope": "authors",
                "title": " ".join((entry.findtext(ATOM + "title") or "").split()),
                "authors": authors, "venue": "arXiv", "years": [],
                "volume": None, "issue": None, "page": None,
            }

    time.sleep(PAUSE)
    arxiv_id = arxiv_search_html(title, want)
    if arxiv_id:
        rec = arxiv_abs_record(arxiv_id)
        if norm_title(rec["title"]) == want:
            rec["source"], rec["scope"] = "arxiv-html-search", "authors"
            return rec
    raise LookupError("no record found by title search")


_SEARCH_HIT = re.compile(
    r'<p class="list-title[^"]*">\s*<a href="[^"]*?/abs/([^"?]+)"', re.S)


def arxiv_search_html(title: str, want: str) -> str | None:
    """Search arxiv.org's web UI. Used when the API is unreachable (406)."""
    # searchtype=title, not "all": the simple-search form ignores the
    # advanced form's terms-0-field parameter and would rank on full text.
    url = ("https://arxiv.org/search/?searchtype=title&size=50&query="
           + urllib.parse.quote(title))
    try:
        page = fetch(url).decode("utf-8", "replace")
    except (urllib.error.HTTPError, urllib.error.URLError):
        return None
    for block in page.split('<li class="arxiv-result">')[1:]:
        m_id = re.search(r'/abs/([^"?<\s]+)', block)
        m_title = re.search(r'<p class="title[^"]*">(.*?)</p>', block, re.S)
        if m_id and m_title and norm_title(_detag(m_title.group(1))) == want:
            return m_id.group(1)
    return None


_ARXIV_IN_DOI = re.compile(r"10\.48550/arxiv\.([\w.\-/]+)", re.I)
_ARXIV_IN_TEXT = re.compile(r"arxiv[:\s]+((?:\d{4}\.\d{4,5})|(?:[a-z\-]+/\d{7}))",
                            re.I)


def identify(fields: dict[str, str]) -> tuple[str, str] | None:
    """Return ('doi'|'arxiv', value), preferring arXiv for arXiv DOIs."""
    doi = fields.get("doi", "").strip()
    if doi:
        m = _ARXIV_IN_DOI.search(doi)
        if m:
            return "arxiv", m.group(1)
    for field in ("journal", "booktitle", "note", "eprint"):
        m = _ARXIV_IN_TEXT.search(fields.get(field, ""))
        if m:
            return "arxiv", m.group(1)
    if doi:
        return "doi", doi
    return None


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def norm_title(s: str) -> str:
    s = deaccent(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(s.split())


def compare(entry: dict, record: dict) -> list[str]:
    """Return a list of problems. Empty means the entry matches."""
    problems: list[str] = []
    fields = entry["fields"]

    bib_authors = [bib_author_parts(a)
                   for a in split_bib_authors(fields.get("author", ""))]
    ref_authors = record["authors"]

    if not ref_authors:
        problems.append("record carries no author list; cannot verify authors")
    elif len(bib_authors) != len(ref_authors):
        bib_fams = [f for f, _ in bib_authors]
        ref_fams = [f for f, _ in ref_authors]
        missing = [f for f in ref_fams if f not in bib_fams]
        extra = [f for f in bib_fams if f not in ref_fams]
        problems.append(
            f"author count {len(bib_authors)} != {len(ref_authors)}"
            + (f"; missing {', '.join(missing)}" if missing else "")
            + (f"; not in record: {', '.join(extra)}" if extra else ""))
    else:
        for i, ((bf, bg), (rf, rg)) in enumerate(zip(bib_authors, ref_authors),
                                                 start=1):
            if bf != rf:
                # Records disagree about where a multi-word name divides:
                # Crossref carries ("Ian", "Lloyd Jones") where the entry has
                # ("Jones", "Ian Lloyd"). Same person, same words, so this is
                # segmentation, not a defect -- but only when every word
                # agrees.
                if sorted(f"{bf} {bg}".split()) == sorted(f"{rf} {rg}".split()):
                    continue
                problems.append(f"author {i}: family '{bf}' != '{rf}'")
            elif given_conflict(bg, rg):
                problems.append(
                    f"author {i} ({rf}): given '{bg or '-'}' != '{rg or '-'}'")

    if record.get("scope") == "authors":
        # Reached by title search: the title matched exactly by construction,
        # and the record describes some version of the work, not necessarily
        # the venue the entry cites. Authors are the whole verdict here.
        return problems

    bt, rt = norm_title(fields.get("title", "")), norm_title(record["title"])
    if bt and rt and bt != rt:
        # A record may hold the preprint title while the entry cites the
        # published one; report it, but say which is which.
        problems.append(f"title '{bt}' != record '{rt}'")

    year = fields.get("year", "").strip()
    if year.isdigit() and record["years"] and int(year) not in record["years"]:
        problems.append(
            f"year {year} not in record {record['years']}")

    for name in ("volume", "number", "pages"):
        want = {"volume": "volume", "number": "issue", "pages": "page"}[name]
        have, ref = fields.get(name, "").strip(), record.get(want)
        if have and ref:
            h = re.sub(r"[^0-9]", "", have)
            r = re.sub(r"[^0-9]", "", str(ref))
            if h and r and h != r:
                problems.append(f"{name} '{have}' != record '{ref}'")
    return problems


def entry_hash(entry: dict) -> str:
    return hashlib.sha256(
        " ".join(entry["raw"].split()).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--key", action="append",
                    help="check only these entry keys (repeatable)")
    ap.add_argument("--list-unverified", action="store_true",
                    help="print the entries with no DOI or arXiv id and stop")
    ap.add_argument("--bib", default=str(BIB))
    ap.add_argument("--report", default=str(REPORT))
    args = ap.parse_args()

    bib_path = Path(args.bib)
    entries = parse_bib(bib_path.read_text(encoding="utf-8"))
    if args.key:
        wanted = set(args.key)
        entries = [e for e in entries if e["key"] in wanted]
        missing = wanted - {e["key"] for e in entries}
        if missing:
            sys.exit(f"no such entry: {', '.join(sorted(missing))}")

    if args.list_unverified:
        for e in entries:
            if identify(e["fields"]) is None:
                print(f"  {e['key']:<28} {e['fields'].get('title','')[:60]}")
        return 0

    NET_ERRORS = (urllib.error.URLError, urllib.error.HTTPError, LookupError,
                  ET.ParseError, json.JSONDecodeError, KeyError)

    results, failed, unverified = [], [], []
    for e in entries:
        ident = identify(e["fields"])
        source = f"{ident[0]}:{ident[1]}" if ident else "title-search"
        try:
            if ident is None:
                record = search_record(e["fields"].get("title", ""))
                source = record["source"]
            elif ident[0] == "arxiv":
                record = arxiv_record(ident[1])
            else:
                record = crossref_record(ident[1])
        except NET_ERRORS as exc:
            print(f"UNVERIFIED  {e['key']:<28} ({source}) {exc}")
            results.append({"key": e["key"], "hash": entry_hash(e),
                            "status": "unverified", "source": source,
                            "reason": str(exc)})
            unverified.append(e["key"])
            time.sleep(PAUSE)
            continue

        problems = compare(e, record)
        status = "mismatch" if problems else "ok"
        results.append({"key": e["key"], "hash": entry_hash(e),
                        "status": status,
                        "source": record["source"],
                        "scope": record.get("scope", "full"),
                        "problems": problems})
        if problems:
            failed.append(e["key"])
            print(f"MISMATCH  {e['key']}  ({record['source']})")
            for p in problems:
                print(f"    - {p}")
        time.sleep(PAUSE)

    checked = [r for r in results if r["status"] in {"ok", "mismatch"}]
    report = {
        "bib": str(bib_path.relative_to(ROOT)) if bib_path.is_relative_to(ROOT)
               else str(bib_path),
        "n_entries": len(entries),
        "n_checked": len(checked),
        "n_ok": sum(1 for r in checked if r["status"] == "ok"),
        "n_full_scope": sum(1 for r in checked
                            if r.get("scope") == "full"),
        "n_authors_only": sum(1 for r in checked
                              if r.get("scope") == "authors"),
        "n_mismatch": len(failed),
        "n_unverified": len(unverified),
        "entries": results,
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n{report['n_ok']}/{report['n_checked']} checked entries match "
          f"their record ({report['n_full_scope']} in full, "
          f"{report['n_authors_only']} authors only); "
          f"{report['n_mismatch']} mismatch, "
          f"{report['n_unverified']} could not be verified.")
    if unverified:
        print("unverified: " + ", ".join(sorted(unverified)))
    print(f"report -> {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
