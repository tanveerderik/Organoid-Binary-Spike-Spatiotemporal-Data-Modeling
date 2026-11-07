"""The visible-only descriptor must carry no information about the hidden region.

The 9-d local descriptor used to be computed
from the whole clip, so the hidden spike count follows from it by arithmetic.
`CTX_SOURCE=visible` is the fix. These tests pin the properties that make it a
fix rather than a relabelling:

* changing ANY hidden voxel leaves the visible-only descriptor bit-identical;
* the whole-clip path is unchanged when no mask is given;
* the whole-clip descriptor really does determine the hidden count exactly
  (the arithmetic the exact count control in canonical_nulls.py relies on);
* the voxel hole matches the token geometry predict_mask_from_spec uses.

Synthetic volumes only: no data, checkpoint or GPU.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))
from MAGVIT_project.utils.recon import (  # noqa: E402
    compute_activity_ctx, compute_activity_ctx_batch, hidden_voxel_mask_from_spec,
    token_roi_to_voxels)

SHAPE = (48, 120, 224)
PATCH = (6, 15, 14)
SPECS = {
    "recon": {"type": "recon"},
    "causal": {"type": "causal", "prefix_frames": 23},
    "noncausal": {"type": "noncausal", "time_spans": [(13, 27)]},
    "spatial": {"type": "spatial", "spatial_box": (31, 88, 50, 161)},
}


def _volume(seed, rate=2e-4):
    rng = np.random.default_rng(seed)
    return (rng.random(SHAPE) < rate).astype(np.float32)


@pytest.mark.parametrize("name", ["causal", "noncausal", "spatial"])
def test_visible_descriptor_ignores_the_hidden_region(name):
    hidden = hidden_voxel_mask_from_spec(SPECS[name], SHAPE, PATCH)
    assert hidden.any() and (~hidden).any()
    x = _volume(1)
    ref = compute_activity_ctx(x, visible_mask=~hidden)
    for seed in range(3):
        y = x.copy()
        noise = _volume(100 + seed, rate=5e-3)
        y[hidden] = noise[hidden]                  # rewrite the hole entirely
        assert np.array_equal(compute_activity_ctx(y, visible_mask=~hidden), ref)
    # ...while the whole-clip descriptor does see the change
    y = x.copy()
    y[hidden] = 1.0
    assert not np.array_equal(compute_activity_ctx(y), compute_activity_ctx(x))


def test_recon_hides_everything_so_the_visible_descriptor_is_constant():
    hidden = hidden_voxel_mask_from_spec(SPECS["recon"], SHAPE, PATCH)
    assert hidden.all()
    a = compute_activity_ctx(_volume(2), visible_mask=~hidden)
    b = compute_activity_ctx(_volume(3), visible_mask=~hidden)
    assert np.array_equal(a, b)


def test_full_path_unchanged_and_all_visible_matches_it():
    x = _volume(4)
    full = compute_activity_ctx(x)
    assert np.array_equal(full, compute_activity_ctx(x, visible_mask=None))
    np.testing.assert_allclose(
        compute_activity_ctx(x, visible_mask=np.ones(SHAPE, bool)), full,
        rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("name", ["causal", "noncausal", "spatial"])
def test_whole_clip_descriptor_determines_the_hidden_count(name):
    """The arithmetic behind the reviewers' point and the exact count control."""
    hidden = hidden_voxel_mask_from_spec(SPECS[name], SHAPE, PATCH)
    x = _volume(5)
    lct0 = compute_activity_ctx(x)[0]
    total = (np.exp(np.float64(lct0)) - 1e-6) * np.prod(SHAPE)
    recovered = total - x[~hidden].sum()
    assert abs(recovered - x[hidden].sum()) < 0.5   # exact up to float32 rounding


def test_hole_is_whole_tokens_with_the_token_rounding_rules():
    T, H, W = SHAPE
    pt, ph, pw = PATCH
    c = hidden_voxel_mask_from_spec(SPECS["causal"], SHAPE, PATCH)
    first_hidden = (23 // pt) * pt                     # prefix FLOORED to tokens
    assert not c[:first_hidden].any() and c[first_hidden:].all()
    n = hidden_voxel_mask_from_spec(SPECS["noncausal"], SHAPE, PATCH)
    lo, hi = (13 // pt) * pt, ((27 - 1) // pt + 1) * pt  # span EXPANDED to tokens
    assert n[lo:hi].all() and not n[:lo].any() and not n[hi:].any()
    s = hidden_voxel_mask_from_spec(SPECS["spatial"], SHAPE, PATCH)
    y0, y1 = (31 // ph) * ph, ((88 - 1) // ph + 1) * ph
    x0, x1 = (50 // pw) * pw, ((161 - 1) // pw + 1) * pw
    assert s[:, y0:y1, x0:x1].all()
    assert s.sum() == T * (y1 - y0) * (x1 - x0)


def test_token_roi_expansion_agrees_with_the_spec_mask():
    pt, ph, pw = PATCH
    grid = (SHAPE[0] // pt, SHAPE[1] // ph, SHAPE[2] // pw)
    for name, spec in SPECS.items():
        hv = hidden_voxel_mask_from_spec(spec, SHAPE, PATCH)
        tok = hv[::pt, ::ph, ::pw]                     # one voxel per token
        assert tok.shape == grid
        roi = torch.from_numpy(tok.reshape(1, -1).astype(np.float32))
        back = token_roi_to_voxels(roi, grid, PATCH, SHAPE)[0, 0].numpy() > 0.5
        assert np.array_equal(back, hv), name


def test_batch_helper_matches_reference_and_none_is_zero():
    xs = np.stack([_volume(6), _volume(7)])[:, None]
    hid = np.stack([hidden_voxel_mask_from_spec(SPECS["causal"], SHAPE, PATCH)] * 2)[:, None]
    xt, ht = torch.from_numpy(xs), torch.from_numpy(hid.astype(np.float32))
    full = compute_activity_ctx_batch(xt, None, "full").numpy()
    vis = compute_activity_ctx_batch(xt, ht, "visible").numpy()
    for b in range(2):
        assert np.array_equal(full[b], compute_activity_ctx(xs[b, 0]))
        assert np.array_equal(vis[b], compute_activity_ctx(xs[b, 0], visible_mask=~hid[b, 0]))
    assert not compute_activity_ctx_batch(xt, None, "none").numpy().any()
    with pytest.raises(ValueError):
        compute_activity_ctx_batch(xt, None, "visible")
