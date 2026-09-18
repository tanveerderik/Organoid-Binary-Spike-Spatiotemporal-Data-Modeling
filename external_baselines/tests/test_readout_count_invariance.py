"""The shared readout pins the spike COUNT; only the ordering is the model's.

This is the invariant behind a result that looks like a bug in the report:
`log_mean_firing_density` is bit-identical for every model on every task. It
is not a four-way tie and it is not a copy-paste error -- that feature is
`log(x.mean() + 1e-6)` over a fixed volume, i.e. a pure function of the spike
count, and `merged_hard` writes `round(rate * |ROI|)` spikes whatever the model
ranked. Measured on real volumes the models share as little as 2% of their
spikes (IoU 0.0196, pipeline vs maskgit_flat) while agreeing on the count
exactly.

If someone later makes the count model-dependent -- a per-model threshold, a
calibration scale -- these tests fail, and the "not a model comparison" note in
diagnose_table.py has to come out with them.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/media/derik/Seagate Desktop Drive/organoid_data")
from MAGVIT_project.external_baselines.task_eval import merged_hard   # noqa: E402
from MAGVIT_project.utils.recon import compute_activity_ctx           # noqa: E402
from MAGVIT_project.utils.constants import ACTIVITY_CTX_NAMES         # noqa: E402

T, H, W = 12, 10, 16
RATE = 0.02


def _fixture(seed: int, roi_frac: float = 1.0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(2, 1, T, H, W, generator=g) < 0.03).float()
    roi = torch.zeros(2, 1, T, H, W)
    roi[:, :, : max(1, int(round(T * roi_frac)))] = 1.0
    rate = torch.full((2,), RATE)
    return x, roi, rate


def _fields(n: int):
    """n DIFFERENT score fields -- stand-ins for n different models."""
    return [torch.rand(2, 1, T, H, W, generator=torch.Generator().manual_seed(s))
            for s in range(100, 100 + n)]


def test_count_is_model_independent():
    x, roi, rate = _fixture(0)
    counts = {tuple(int(merged_hard(f, x, roi, rate)[b, 0].sum())
                    for b in range(2)) for f in _fields(4)}
    assert len(counts) == 1, f"readout let the count vary by model: {counts}"


def test_volumes_really_do_differ():
    """Guards the test above from passing for the wrong reason."""
    x, roi, rate = _fixture(0)
    hs = [merged_hard(f, x, roi, rate) > 0.5 for f in _fields(4)]
    diffs = [int((hs[0] ^ h).sum()) for h in hs[1:]]
    assert all(d > 0 for d in diffs), f"fields produced identical volumes: {diffs}"


def test_log_mean_firing_density_is_invariant():
    i0 = ACTIVITY_CTX_NAMES.index("log_mean_firing_density")
    x, roi, rate = _fixture(0)
    vals = [compute_activity_ctx(merged_hard(f, x, roi, rate)[0, 0].numpy())[i0]
            for f in _fields(4)]
    assert max(vals) - min(vals) == 0.0, f"expected bit-identical, got {vals}"


def test_placement_features_are_not_invariant():
    """The other eight features must still separate the models."""
    x, roi, rate = _fixture(0)
    ctx = np.stack([compute_activity_ctx(
        merged_hard(f, x, roi, rate)[0, 0].numpy()) for f in _fields(4)])
    i0 = ACTIVITY_CTX_NAMES.index("log_mean_firing_density")
    moved = [n for i, n in enumerate(ACTIVITY_CTX_NAMES)
             # np.ptp(), not ndarray.ptp(): NumPy 2 removed the method.
             if i != i0 and np.ptp(ctx[:, i]) > 0]
    assert len(moved) >= 6, f"only {len(moved)} placement features moved: {moved}"


def test_count_follows_the_rate_not_the_scores():
    x, roi, rate = _fixture(0)
    n = int((roi[0, 0] > 0.5).sum())
    h = merged_hard(_fields(1)[0], x, roi, rate)
    outside = int(((x[0, 0] > 0.5) & (roi[0, 0] <= 0.5)).sum())
    assert int(h[0, 0].sum()) == round(RATE * n) + outside


def test_partial_roi_keeps_truth_outside():
    x, roi, rate = _fixture(1, roi_frac=0.5)
    h = merged_hard(_fields(1)[0], x, roi, rate)
    out = roi[0, 0] <= 0.5
    assert torch.equal(h[0, 0][out], (x[0, 0] > 0.5).float()[out])


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for f in fns:
        try:
            f(); print(f"  [PASS] {f.__name__}")
        except Exception:
            bad += 1; print(f"  [FAIL] {f.__name__}"); traceback.print_exc()
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    raise SystemExit(1 if bad else 0)
