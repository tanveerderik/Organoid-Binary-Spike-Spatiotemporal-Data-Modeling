#!/usr/bin/env python3
"""The dump must sample recordings, not the head of the loader.

The test split is a DeterministicSubset and is not shuffled, so taking clips in
loop order takes an ordered PREFIX. An earlier dump of 24 clips drew all 24
from a single recording, and raising --batches does not help: the quota fills
in the first couple of batches whatever the budget is. Panels built from that
would show one recording while citing a protocol that covers 31.

The second property is just as load-bearing. The supplement figure shows one
clip under all four settings, which is only possible because every mode walks
the same batch order and therefore keeps the same underlying clips. A change
that made selection mode-dependent would silently turn those rows into four
unrelated clips.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from external_baselines.task_eval import _dump_clips

MODES = ("recon", "causal", "noncausal", "spatial")
# Assay ids repeat within a batch and across batches, as the real prefix does.
BATCHES = [[0, 0, 0, 0], [0, 0, 1, 1], [1, 2, 2, 3],
           [3, 3, 4, 4], [5, 5, 6, 6], [7, 8, 9, 10]]
N_WANT, N_FIELD = 8, 2


def _run():
    store = {"_count": {}}
    T, H, W = 4, 3, 3
    for mode in MODES:
        for bi, assays in enumerate(BATCHES):
            b = len(assays)
            x = torch.zeros(b, 1, T, H, W)
            x[:, 0, 0, 0, 0] = 1
            z = torch.zeros(b, 1, T, H, W)
            _dump_clips(store, mode, bi, assays, x, torch.ones(b, 1, T, H, W),
                        z, z, z, None, N_WANT, N_FIELD)
            store["_count"][mode] = sum(
                1 for k in store
                if k.startswith(f"{mode}/") and k.endswith("/assay"))
    return store


def _assays(store, mode):
    return [int(store[k][0]) for k in sorted(store)
            if k.startswith(f"{mode}/") and k.endswith("/assay")]


def test_each_dumped_clip_comes_from_a_different_recording():
    store = _run()
    for mode in MODES:
        got = _assays(store, mode)
        assert len(got) == N_WANT, f"{mode}: dumped {len(got)}, wanted {N_WANT}"
        assert len(set(got)) == len(got), f"{mode}: repeated recording in {got}"


def test_all_modes_keep_the_same_clips():
    store = _run()
    picked = {mode: tuple(_assays(store, mode)) for mode in MODES}
    assert len(set(picked.values())) == 1, (
        f"modes disagree, so a figure cannot show one clip under all four: "
        f"{picked}")


def test_field_is_capped_independently_of_the_clip_count():
    """`--dump-field` bounds the dense arrays; the binary volumes are cheap.

    The field is 2.6 MB per clip in float16 and the dump is tracked in git, so
    raising --dump must not drag the dense arrays up with it.
    """
    store = _run()
    for mode in MODES:
        n = sum(1 for k in store if k.startswith(f"{mode}/") and k.endswith("/field"))
        assert n == N_FIELD, f"{mode}: {n} fields, expected {N_FIELD}"


def test_bookkeeping_keys_are_not_written_to_the_archive():
    """`_assays` is a dict of lists; np.savez would choke or store an object
    array. It is popped beside `_count` at write time."""
    src = (Path(__file__).resolve().parents[1] / "task_eval.py").read_text()
    assert 'dump_store.pop("_assays", None)' in src
