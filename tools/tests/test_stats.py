"""Cluster-aware statistics (utils/stats.py) behave as documented.

The recording-level re-analysis of every paired comparison rests on these
functions, so they are checked on cases with a known answer: clustered data
where a clip-level test is confidently wrong, duplicate folding, and the
standard multiplicity corrections.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))
from MAGVIT_project.utils.stats import (  # noqa: E402
    DUPLICATE_OF, bh, hierarchical_bootstrap, holm, merge_duplicates,
    paired_clip_test, paired_recording_test, recording_means)


def test_duplicates_fold_onto_their_canonical_recording():
    assert DUPLICATE_OF == {10: 4, 11: 3, 13: 6}
    assert merge_duplicates([4, 10, 3, 11, 6, 13, 0]).tolist() == [4, 4, 3, 3, 6, 6, 0]


def test_recording_means_weight_recordings_not_clips():
    ids, m = recording_means([1, 1, 1, 1, 5], [0, 0, 0, 0, 1])
    assert ids.tolist() == [0, 1] and m.tolist() == [1.0, 5.0]


def test_clustered_effect_is_significant_per_clip_but_not_per_recording():
    """Six recordings, 50 clips each, recording-level offsets but no true effect:
    the clip-level test treats 300 clips as independent and is fooled."""
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(6), 50)
    offset = np.array([0.30, 0.25, 0.20, -0.05, 0.22, 0.18])[labels]
    a = offset + 0.01 * rng.standard_normal(labels.size)
    b = np.zeros(labels.size)
    assert paired_clip_test(a, b)["p"] < 1e-10
    rec = paired_recording_test(a, b, labels)
    assert rec["n_recordings"] == 6
    assert rec["p"] > 0.05          # n = 6 cannot reach significance with a sign flip


def test_hierarchical_bootstrap_ci_covers_the_equal_weight_mean():
    rng = np.random.default_rng(1)
    labels = np.repeat(np.arange(20), [1] * 10 + [40] * 10)   # very unequal clip counts
    rec_effect = rng.normal(0.1, 0.05, 20)
    d = rec_effect[labels] + 0.02 * rng.standard_normal(labels.size)
    strata = {i: ("a" if i < 10 else "b") for i in range(20)}
    out = hierarchical_bootstrap(d, labels, strata, n_boot=2000, seed=0)
    eq = np.mean([d[labels == i].mean() for i in range(20)])
    assert abs(out["estimate"] - eq) < 1e-12
    assert out["ci_low"] < eq < out["ci_high"]
    assert out["excludes_zero"]


def test_multiplicity_corrections_match_reference_values():
    p = [0.01, 0.04, 0.03, 0.005]
    np.testing.assert_allclose(bh(p), [0.02, 0.04, 0.04, 0.02])
    np.testing.assert_allclose(holm(p), [0.03, 0.06, 0.06, 0.02])
