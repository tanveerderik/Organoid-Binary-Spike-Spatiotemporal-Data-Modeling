"""Assert that every bibliography entry has been checked against its record.

`tools/check_refs.py` does the fetching; this module does the gating, offline,
so the suite does not depend on Crossref or arXiv being reachable. It asserts
three things about `reports/refs_verification.json`:

  1. no entry MISMATCHES its record;
  2. every entry in `refs.bib` appears in the report with the hash it has now,
     so editing an entry invalidates its verdict instead of inheriting a stale
     pass;
  3. the entries that could not be verified are exactly the four we know
     cannot be, each for a reason recorded below. A new unverifiable entry
     fails the test and forces a decision rather than quietly widening the
     hole.

Refresh the report with `python tools/check_refs.py` after touching refs.bib.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from MAGVIT_project.tools.tests._release_guard import needs_paper, needs_reports

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_refs import entry_hash, parse_bib  # noqa: E402

pytestmark = [needs_paper, needs_reports]

ROOT = Path(__file__).resolve().parents[2]
BIB = ROOT / "paper" / "refs.bib"
REPORT = ROOT / "reports" / "refs_verification.json"

# Entries no online record can settle, with what stands in for one. Each was
# checked by hand on 2026-09-18; do not add to this list without doing the
# same and saying so here.
EXPECTED_UNVERIFIED = {
    # Dandiset 000732 was withdrawn from the DANDI archive and has no DOI, so
    # neither Crossref nor DataCite holds it. The entry's author and title
    # come from the downloaded NWB files themselves: experimenter
    # 'Sharf, Tal', lab 'Kosik Lab, UCSB', experiment_description matching the
    # title, session_start_time in 2023. See reports/data_provenance.json.
    "dandi000732",
    # IJCAI 2001 predates IJCAI's DOI coverage and Elkan never posted it to
    # arXiv. Title, author, venue and pages 973-978 confirmed against the
    # IJCAI proceedings listing and the author's own copy at
    # cseweb.ucsd.edu/~elkan/rescale.pdf.
    "elkan2001costsensitive",
    # NeurIPS 2015 has no DOI and no arXiv posting. Title and the three
    # authors confirmed against the NeurIPS 2015 proceedings index at
    # papers.nips.cc/paper_files/paper/2015.
    "sohn2015cvae",
    # The arXiv posting is titled "Improving Variational Inference with
    # Inverse Autoregressive Flow"; the NeurIPS 2016 proceedings version this
    # entry cites is titled "Improved ...". The checker refuses a title that
    # does not match exactly, which is the behaviour we want. Confirmed
    # against proceedings.neurips.cc/paper/2016.
    "kingma2016iaf",
}


@pytest.fixture(scope="module")
def report() -> dict:
    if not REPORT.is_file():
        pytest.fail(
            f"{REPORT.relative_to(ROOT)} is missing. "
            "Run: python tools/check_refs.py")
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_no_entry_mismatches_its_record(report):
    bad = [e for e in report["entries"] if e["status"] == "mismatch"]
    detail = "\n".join(
        f"  {e['key']} ({e.get('source', '?')}): " + "; ".join(e["problems"])
        for e in bad)
    assert not bad, f"bibliography entries disagree with their records:\n{detail}"


def test_report_covers_every_entry_as_it_stands_now(report):
    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    checked = {e["key"]: e["hash"] for e in report["entries"]}

    missing = [e["key"] for e in entries if e["key"] not in checked]
    assert not missing, (
        "entries absent from the verification report: "
        + ", ".join(sorted(missing))
        + ". Run: python tools/check_refs.py")

    stale = [e["key"] for e in entries
             if checked[e["key"]] != entry_hash(e)]
    assert not stale, (
        "entries edited since they were verified: " + ", ".join(sorted(stale))
        + ". Run: python tools/check_refs.py")

    dropped = sorted(set(checked) - {e["key"] for e in entries})
    assert not dropped, (
        "report holds entries no longer in refs.bib: " + ", ".join(dropped)
        + ". Run: python tools/check_refs.py")


def test_unverifiable_set_has_not_grown(report):
    unverified = {e["key"] for e in report["entries"]
                  if e["status"] == "unverified"}
    new = unverified - EXPECTED_UNVERIFIED
    assert not new, (
        "entries that no longer verify: " + ", ".join(sorted(new))
        + ". Either the record moved and the entry needs fixing, or the fetch "
          "failed transiently -- re-run tools/check_refs.py before adding "
          "anything to EXPECTED_UNVERIFIED.")

    gone = EXPECTED_UNVERIFIED - unverified
    assert not gone, (
        "these now verify and should be dropped from EXPECTED_UNVERIFIED: "
        + ", ".join(sorted(gone)))
