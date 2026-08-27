#!/usr/bin/env python3
"""Shape and internal consistency of the motif-reuse measurement.

Deliberately asserts NOTHING about the values. The paper's thesis is that the
motif alphabet is shared across recordings rather than partitioned among them,
and this measurement is allowed to come back saying the opposite -- that is the
point of running it before the prose is written. A test that forbade a low
reuse ratio would not be a test, it would be an assumption with a green tick.

What is asserted instead: the decomposition is well formed (conditioning cannot
raise entropy, the ratio is a fraction), the blank token is excluded, and the
null actually ran with enough shuffles to give the observed value a reference.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "analysis_motif_reuse.json"

pytestmark = pytest.mark.skipif(
    not OUT.exists(),
    reason="run: python analysis/motif_reuse.py --batches 24")


def _d():
    return json.loads(OUT.read_text())


def test_shape_of_the_measurement():
    d = _d()
    assert d["V"] == 961
    assert d["n_recordings"] == 31
    assert len(d["per_recording"]) == 31
    # The blank token is excluded everywhere: it is 91.7% of all tokens and
    # every recording emits it, so including it would drive every overlap
    # statistic to ~1.0 while measuring nothing about motifs.
    assert d["excludes_blank"] is True
    assert 0 < d["vocab_global"] <= d["V"]


def test_entropy_decomposition_is_well_formed():
    d = _d()
    e = d["entropy"]
    # Conditioning cannot increase entropy.
    assert e["H_code_given_recording"] <= e["H_code"] + 1e-9
    # reuse_ratio is H(code|recording)/H(code): 1.0 means the recording label
    # tells you nothing about the next motif -- a fully shared alphabet --
    # and near 0 means each recording has a private vocabulary.
    assert 0.0 <= e["reuse_ratio"] <= 1.0


def test_shared_core_is_monotone_decreasing_in_k():
    d = _d()
    core = d["shared_core"]["used_by_ge_k"]
    ks = sorted(int(k) for k in core)
    vals = [core[str(k)] for k in ks]
    assert vals == sorted(vals, reverse=True), core
    # k=1 is the global vocabulary by definition.
    assert core["1"] == d["vocab_global"]


def test_null_is_present_and_comparable():
    d = _d()
    n = d["label_shuffle_null"]
    assert n["n_shuffles"] >= 200
    assert n["sd"] > 0.0
    # A z near zero means the observed overlap is indistinguishable from what
    # you get when recording labels are meaningless -- i.e. fully shared. A
    # large negative z means partition.
    assert "z" in n


# ---------------------------------------------------------------------------
# The n-gram motif mining is a SEPARATE artifact, produced before the Stage-2B
# flatten. It is the paper's causal evidence that a motif is a unit rather than
# a cluster label -- 3-gram lift over a matched null, a transplant test, and an
# order-sensitivity test. Whether it survived the move to V=961 depends
# entirely on which alphabet it mined over, and the flat null ladder had to be
# refit for exactly this reason, so it is checked rather than assumed.
# ---------------------------------------------------------------------------

MOTIFS = ROOT / "reports" / "evaluation_report_code_motifs.json"


@pytest.mark.skipif(not MOTIFS.exists(), reason="motif mining artifact absent")
def test_ngram_motifs_are_mined_over_level_one_codes():
    d = json.loads(MOTIFS.read_text())
    syms = {v for m in d["motifs"] for v in m}
    # -1 is the blank code. Level 1 is the 32-entry parent codebook, which
    # Stage 2B's deduplication does not touch, so a mining run confined to
    # [0, 31] is unaffected by the flatten to V=961 and stays citable. A symbol
    # above 31 would mean the artifact mined a flat alphabet and predates the
    # current one, in which case it must not be cited.
    assert max(syms) <= 31, (
        f"motif symbols reach {max(syms)}, so this artifact was mined over a "
        f"flat alphabet and predates V=961. Do not cite it; re-mine first.")
    assert min(syms) >= -1
