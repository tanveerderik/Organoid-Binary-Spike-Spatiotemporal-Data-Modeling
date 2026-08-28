#!/usr/bin/env python3
"""The paper's figures must render, carry no placeholder, and be reproducible.

Byte stability is the load-bearing one. matplotlib stamps a creation date into
every PDF unless SOURCE_DATE_EPOCH is pinned before pyplot is imported, so
without it two identical renders differ and "regenerate before every compile"
silently churns the manuscript.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "paper_figures"
GEN = ROOT / "tools" / "make_paper_figures.py"
NAMES = ("f1_pipeline", "f2_motifs", "f3_task_axis", "f4_generation")


def _render():
    subprocess.run([sys.executable, str(GEN)], cwd=ROOT, check=True,
                   env={**__import__("os").environ,
                        "PYTHONPATH": str(ROOT.parent)})


def _hashes():
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(OUT.glob("*.pdf"))}


def test_all_four_figures_render():
    _render()
    for name in NAMES:
        p = OUT / f"{name}.pdf"
        assert p.exists() and p.stat().st_size > 2000, name


def test_no_placeholder_survives():
    for name in NAMES:
        assert b"PLACEHOLDER" not in (OUT / f"{name}.pdf").read_bytes(), name


def test_figures_are_byte_stable_across_runs():
    _render()
    first = _hashes()
    _render()
    assert first == _hashes(), "figures are not reproducible run to run"
