"""Shared skip predicates for tests needing material the release omits.

The anonymous code release ships the pipeline, not the manuscript and not the
JSON report artifacts. `RELEASE.md` promises that tests depending on those skip
themselves -- but six modules raised FileNotFoundError, CalledProcessError or a
bare assertion instead, so the exact command RELEASE.md advertises failed on a
clean unpack with 14 failures and 29 errors. These predicates are the guard.

ABSENT material skips; PRESENT but wrong still fails loudly, which is what the
working repository needs. Same rule as `test_paper_cross_references.py`, which
had it right all along.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

HAS_PAPER = (ROOT / "paper" / "sections").is_dir()
HAS_REPORTS = any((ROOT / "reports").glob("*.json"))

needs_paper = pytest.mark.skipif(
    not HAS_PAPER, reason="manuscript not present (paper/ ships separately)")
needs_reports = pytest.mark.skipif(
    not HAS_REPORTS,
    reason="reports/*.json not in the code release (see RELEASE.md)")
