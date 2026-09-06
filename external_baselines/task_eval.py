#!/usr/bin/env python3
"""Table 2 -- the task axis.

Every number in reports/external_baselines/diagnostics.md is measured at
task_id=0: common/evaluate.py:205 hardcodes ``task_ids=[0]*B`` and
analysis/generate_regimes.py:92 defaults ``--task-id 0``. The model trains on
four tasks at 25% each (dataset.py:452), so three quarters of training has
never been scored -- and it is the three quarters that is EASIEST to score,
because a completion task leaves ground truth inside the hole. Free generation
admits only distribution comparisons; completion admits a per-voxel accuracy,
paired per clip.

Every clip runs under EVERY task, so the four columns are paired and a clip
that is simply busier cannot move one column relative to another.

Arms
----
model     Activity is TRUE outside the ROI and predicted by the activity prior
          inside it; the motif prior then fills ROI tokens conditioned on the
          visible ones. The merge is required, not cosmetic:
          ``sample_hard_activity_gumbel_topk`` zeroes activity outside the ROI
          (model/prior.py:1497) and ``iterative_unmask`` nulls inactive tokens
          (``f[~active] = f_null_id``), so an unmerged partial-ROI decode blanks
          the visible region entirely. iterative_unmask's own soft-field line,
          ``a_prob = torch.where(roi, a_prob, a)``, assumes the merged field.

oracle    The TRUE codes everywhere, decoded. The tokenizer ceiling: what this
          arm would score if the prior named every ROI token correctly. It
          separates "the prior cannot predict the hole" from "the tokenizer
          cannot represent it", which one number alone cannot do.

separable The rank-1 null: the clip's own VISIBLE temporal profile times the
          assay's TRAIN spatial marginal. Uses no held-out voxel, applies
          unchanged to all four tasks, and is the honest "marginals plus what
          you can see" competitor.

marginal  The assay's TRAIN per-site marginal, constant in time. Exactly what
          DG and the GLM memorise, with no model at all.

copy      Persistence: the mean of the visible frames adjacent to the hole.
          Defined for the two temporal tasks; not defined for `spatial`, whose
          hole spans every frame, and reported as n/a rather than faked.

Metric
------
Per-clip EXACT average precision over ROI voxels only, computed grid-free by
sorting (no threshold grid, so no resolution limit and no rank-normalisation
needed -- the fixed linspace(0,1) grid in PRCurveAccumulator cannot resolve the
marginal arm, whose field lives near 1e-3, and would score it near zero for a
reason that has nothing to do with its ranking).

Scoring is restricted to ROI voxels. Scoring the whole volume would hand every
arm the visible region for free and compress every contrast to nothing.

Exact, not tolerant: tolerant credit favours blur, and a completion task is
precisely where a blurred arm should not be rewarded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/media/derik/Seagate Desktop Drive/organoid_data")
from MAGVIT_project.external_baselines.common import data as bdata      # noqa: E402
from MAGVIT_project.external_baselines.common.pipeline import build_pipeline  # noqa: E402

bdata.ensure_repo_cwd()

from MAGVIT_project.inference.decode import decode_flat_ids_to_xgen      # noqa: E402
# Canonical schemas, imported rather than restated. Every label and every bin
# edge in the per-task battery comes from utils/constants.py, which exists so
# "training, inference, metrics, reports and visualization cannot silently
# disagree". The adjacency rate and the spatial-support fraction are the same
# functions training reports, not re-implementations.
from MAGVIT_project.utils.constants import (                             # noqa: E402
    ACTIVITY_CTX_NAMES, DEFAULT_GAP_BINS, gap_bin_labels)
from MAGVIT_project.utils.recon import compute_activity_ctx              # noqa: E402
from MAGVIT_project.training.stage4_activity import _hard_gap_rates      # noqa: E402
from MAGVIT_project.inference.metrics_gen import (                       # noqa: E402
    evaluate_generation_global_metrics)
from MAGVIT_project.inference.sample_prior import (                      # noqa: E402
    iterative_unmask_motif_given_activity)
from MAGVIT_project.training.train_prior import (                        # noqa: E402
    _make_activity_in_from_codes, _vq_codes_and_pmask_for_prior)

MODES = ("recon", "causal", "noncausal", "spatial")
TASK_ID = {"recon": 0, "causal": 1, "noncausal": 2, "spatial": 3}
ARMS = ("model", "model_mc", "oracle", "separable", "marginal",
        "marginal_xa", "copy")


# ---------------------------------------------------------------- masks ----
def _clip_seed(batch, i: int) -> int:
    """The key `deterministic_validation_task_and_masks` uses, so a clip gets
    the same hole here as it did in validation."""
    parts = []
    for key in ("path", "assay_name", "assay_idx"):
        v = batch.get(key, None)
        if torch.is_tensor(v) and v.ndim > 0:
            v = v[i].detach().cpu().item()
        elif isinstance(v, np.ndarray) and v.ndim > 0:
            v = v[i].item()
        elif isinstance(v, (list, tuple)) and len(v) > i:
            v = v[i]
        parts.append(str(v))
    d = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(d[:8], "little", signed=False)


def masks_for_task(batch, x, mode: str):
    """Same mask families and the same per-clip draws as
    `deterministic_validation_task_and_masks`, with the task FORCED so that one
    clip contributes to every column."""
    B, _, T, H, W = x.shape
    specs = []
    for i in range(B):
        seed = _clip_seed(batch, i)
        if mode == "recon" or T < 2:
            specs.append({"type": "recon"})
        elif mode == "causal":
            unit = ((seed >> 8) % 10000) / 9999.0
            pf = int(np.clip(round(T * (0.25 + 0.50 * unit)), 1, T - 1))
            specs.append({"type": "causal", "prefix_frames": pf})
        elif mode == "noncausal":
            L = max(1, int(round(0.30 * T)))
            s = int((seed >> 20) % (max(0, T - L) + 1))
            specs.append({"type": "noncausal", "time_spans": [(s, s + L)]})
        else:
            bh, bw = max(1, int(round(0.50 * H))), max(1, int(round(0.50 * W)))
            y0 = int((seed >> 32) % (max(0, H - bh) + 1))
            x0 = int((seed >> 44) % (max(0, W - bw) + 1))
            specs.append({"type": "spatial",
                          "spatial_box": (y0, y0 + bh, x0, x0 + bw)})
    return specs


def roi_voxels(vq, pmask_tok, grid, thw):
    """Token ROI -> voxel ROI, the same expansion training/eval_vqvae.py:213 uses."""
    Tp, Hp, Wp = thw
    pm = pmask_tok.unsqueeze(-1) if pmask_tok.dim() == 2 else pmask_tok
    pd = (vq.patch_size[0] * vq.patch_size[1] * vq.patch_size[2] * vq.out_chans)
    v = vq.unpatchify(pm.float().expand(-1, -1, pd), grid)
    return (v[:, :1, :Tp, :Hp, :Wp] > 0.5).float()


# --------------------------------------------------------------- metric ----
def clip_ap(score: torch.Tensor, target: torch.Tensor, seed: int = 0) -> float:
    """Exact step-wise AP for one clip, by sorting. No threshold grid, so no
    interpolation and no resolution limit -- see utils.metrics for why the
    trapezoid form is not used.

    Ties are broken UNIFORMLY AT RANDOM (permute, then stable sort), not by
    index. This is not a detail here: the null arms are massively tied -- the
    copy field takes values in {0, 1/6, ..., 1} and 99.98% of every field is
    zero -- and index order is raster order, which is spatially structured, so
    index tie-breaking would hand the nulls a systematic and meaningless
    advantage or penalty. The model arm has continuous logits and is unaffected.
    """
    s = score.reshape(-1).float().cpu().numpy()
    t = (target.reshape(-1) > 0.5).cpu().numpy().astype(np.float64)
    n_pos = float(t.sum())
    if n_pos < 1.0:
        return float("nan")
    # numpy, not torch: torch 1.12.1 has no `stable=` on argsort, and stability
    # is the whole mechanism -- permute at random, then a STABLE sort makes the
    # permutation the tie order.
    perm = np.random.default_rng(seed).permutation(s.size)
    s, t = s[perm], t[perm]
    order = np.argsort(-s, kind="stable")
    t = t[order]
    prec = np.cumsum(t) / np.arange(1, t.size + 1)
    return float((prec * t).sum() / n_pos)


def site_ap(field, real, roi_vox, b: int, seed: int) -> float:
    """AP over SITES, collapsing time inside the hole.

    The spatiotemporal AP above asks "which voxel". This asks only "which
    electrode fires somewhere in the hole", which is the part a static rate map
    can answer. The two together decompose an arm's score into WHERE and WHEN:
    an arm that nails the spatial support but places spikes in the wrong frames
    scores well here and badly above, and that is exactly the signature of
    scoring a SAMPLE against a metric that wants a posterior.
    """
    m = roi_vox[b, 0] > 0.5
    site = m.any(dim=0)                                   # (H,W) touched by the hole
    if not bool(site.any()):
        return float("nan")
    tgt = ((real[b, 0] * m).amax(dim=0) > 0.5)[site].float()
    den = m.float().sum(dim=0).clamp(min=1.0)
    scr = ((field[b, 0] * m).sum(dim=0) / den)[site]
    return clip_ap(scr, tgt, seed=seed)


# ----------------------------------------------------------------- nulls ----
def train_site_maps(batches: int, seed: int = 0):
    """Per-assay per-site TRAIN marginal, in expected spikes per site per clip.

    The null arms built from this map contain no model, so they must not vary
    by model -- but they did, by up to 5% (recon `marginal` ranged 0.0848 to
    0.0892 over the four runs). Two causes, both upstream: the shared train
    loader shuffles with no explicit generator (dataset.py:1160) so its order
    follows the GLOBAL torch RNG, which model construction has already
    perturbed; and it uses ``persistent_workers=True`` (dataset.py:1143) whose
    per-worker numpy RNGs (dataset.py:986) draw the random temporal crop, so
    the clip CONTENTS drift too once the pool has been touched. The test loader
    is immune -- it is a ``DeterministicSubset`` (dataset.py:1134) -- which is
    why the scored clips were identical throughout and only the nulls moved.

    This function therefore builds its own loader and is a pure function of
    `seed`. Verified across processes with different pre-burn states.
    """
    shared = bdata.loader_for("train")
    ds = shared.dataset
    base = getattr(ds, "dataset", ds)
    # A PRIVATE, single-process loader. The shared one cannot be made
    # reproducible from here: it is built with `persistent_workers=True`
    # (dataset.py:1143), so its workers are spawned on the FIRST train
    # iteration in the process and their numpy RNGs advance from there.
    # `volume_shape()` (common/data.py:96) takes one train batch, so a baseline
    # that calls it during construction burns the worker pool before this
    # function ever runs -- and no amount of generator-pinning here can undo
    # that. num_workers=0 keeps the draw in this process, where resetting
    # `base.rng` below is authoritative.
    if hasattr(base, "rng"):
        base.rng = np.random.default_rng(np.random.SeedSequence([int(seed)]))
    g = torch.Generator().manual_seed(int(seed))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=shared.batch_size, shuffle=True, generator=g,
        num_workers=0, collate_fn=shared.collate_fn, drop_last=True)

    acc, cnt = {}, {}
    for bi, batch in enumerate(loader):
        if bi >= batches:
            break
        x = batch["x"]
        x = x.squeeze(1) if x.dim() == 5 else x
        for i, a in enumerate(batch["assay_idx"].tolist()):
            m = x[i].sum(0).numpy()
            acc[a] = acc.get(a, 0) + m
            cnt[a] = cnt.get(a, 0) + 1
    return {a: acc[a] / cnt[a] for a in acc}, sum(cnt.values())


def cross_assay_maps(maps):
    """For each assay, the mean TRAIN map over every OTHER assay.

    `marginal` is a SEEN-assay lookup: it reads the test clip's own assay off a
    table built from that assay's train clips. That is precisely the 833k
    memorised numbers DG and the GLM carry, and on this split -- temporal
    within assay -- no model ever faces an unseen assay, so the lookup is never
    charged for it. `marginal_xa` is the same null with the clip's own assay
    withheld: what a per-assay table is worth when the assay is new. Reporting
    only `marginal` would say we lose to a lookup table and stop there.
    """
    keys = [a for a in maps if maps[a].shape == next(iter(maps.values())).shape]
    tot = sum(maps[a] for a in keys)
    n = max(1, len(keys) - 1)
    return {a: (tot - maps[a]) / n for a in keys}


def spatial_prior_field(maps, assays, thw, device):
    """(B,1,T,H,W) train marginal broadcast over time. Falls back to the pooled
    map for an assay with no train clips."""
    T, H, W = thw
    pool = np.mean([m for m in maps.values()], axis=0) if maps else np.zeros((H, W))
    out = torch.zeros((len(assays), 1, T, H, W), dtype=torch.float32)
    for i, a in enumerate(assays):
        m = maps.get(int(a), pool)
        if m.shape != (H, W):
            m = pool if pool.shape == (H, W) else np.zeros((H, W))
        out[i, 0] = torch.from_numpy(np.asarray(m, dtype=np.float32))
    return out.to(device)


def visible_time_profile(real, roi_vox):
    """(B,T) mean rate per frame over VISIBLE voxels. Frames with no visible
    voxel (the whole frame is inside the hole) fall back to the clip mean, so
    the profile never reads a held-out voxel."""
    vis = (1.0 - roi_vox)
    num = (real * vis).sum(dim=(1, 3, 4))
    den = vis.sum(dim=(1, 3, 4)).clamp(min=1.0)
    prof = num / den
    any_vis = vis.sum(dim=(1, 3, 4)) > 0
    clip_mean = (prof * any_vis).sum(1) / any_vis.sum(1).clamp(min=1)
    return torch.where(any_vis, prof, clip_mean.unsqueeze(1))


def copy_field(real, roi_vox, mode, span: int = 6):
    """Persistence: mean of up to `span` visible frames adjacent to the hole,
    held constant across it. Undefined for `spatial` (the hole spans all frames)."""
    if mode == "spatial":
        return None
    # recon: every frame is inside the hole, so there is no adjacent visible
    # frame to persist. An all-zero field is not a weak null, it is no null.
    if bool((roi_vox.amax(dim=(1, 3, 4)) > 0.5).all()):
        return None
    B, _, T, H, W = real.shape
    out = torch.zeros_like(real)
    for b in range(B):
        hole = roi_vox[b, 0].amax(dim=(1, 2)) > 0.5      # (T,) frames in the hole
        vis_t = torch.nonzero(~hole, as_tuple=False).squeeze(-1)
        if vis_t.numel() == 0:
            continue
        hole_t = torch.nonzero(hole, as_tuple=False).squeeze(-1)
        if hole_t.numel() == 0:
            out[b] = real[b]
            continue
        lo, hi = int(hole_t.min()), int(hole_t.max())
        before = vis_t[vis_t < lo][-span:]
        after = vis_t[vis_t > hi][:span]
        take = torch.cat([before, after]) if after.numel() else before
        if take.numel() == 0:
            take = vis_t[:span]
        out[b, 0] = real[b, 0, take].mean(0, keepdim=True).expand(T, H, W)
    return out


# --------------------------------------------------------------- battery ----
def assay_rates(maps, thw):
    """Per-assay voxel rate from the deterministic TRAIN site map.

    Reuses `train_site_maps` rather than taking a second train pass, so the
    readout target inherits the same seed-pinned draw and cannot drift.
    """
    n = float(thw[0] * thw[1] * thw[2])
    return {a: float(np.asarray(m).sum()) / n for a, m in maps.items()}


# ------------------------------------------------------------------ dump ----
def _dump_clips(store, mode, bi, assays, x, roi_vox, src, gen_hard, own_vol,
                p_oracle, n_want, want_field):
    """Persist the very volumes this run scored, for supplement figures.

    Written from INSIDE the scoring loop, from the same locals the metrics are
    computed from, so a frame cannot drift from the number printed beside it.
    Re-deriving these in a separate script would reproduce the code path but not
    the RNG stream, and the whole point of the panel is that it shows what the
    table measured.

    Binary volumes are stored as int16 nonzero coordinates -- ~200 spikes in
    1.29M voxels, so dense storage would be four orders of magnitude of zeros.
    The continuous field is dense and only saved for the first `--dump-field`
    clips per task: it is what shows that an arm can RANK electrodes well while
    GENERATING them poorly, which is the U-Net story, but at 2.6 MB/clip in
    float16 it is not something to keep for every clip.
    """
    have = store["_count"].get(mode, 0)
    store["_shape"] = np.asarray(list(x.shape[-3:]), np.int32)
    if have >= n_want:
        return
    # One clip per RECORDING, not the first `n_want` clips seen.
    #
    # The test loader is a DeterministicSubset and is not shuffled, so taking
    # clips in loop order takes an ordered PREFIX: an earlier dump of 24 clips
    # drew all 24 from a single recording, and raising --batches does not help
    # because the quota fills in the first couple of batches either way. A
    # supplement panel built from that would show one recording while claiming
    # the protocol that covers 31.
    #
    # Scoring is untouched -- this only decides which already-scored volumes
    # are persisted. Every mode walks the same batch order, so the modes still
    # agree on which underlying clips they keep, which is what lets a figure
    # show one clip under all four settings.
    seen = store.setdefault("_assays", {}).setdefault(mode, [])
    for b in range(x.shape[0]):
        if have >= n_want:
            break
        if int(assays[b]) in seen:
            continue
        seen.append(int(assays[b]))
        tag = f"{mode}/clip{have:02d}"
        store[f"{tag}/assay"] = np.asarray([assays[b]], np.int32)
        for nm, vol in (("real", (x[b, 0] > 0.5)),
                        ("gen_shared_readout", (gen_hard[b, 0] > 0.5)),
                        ("gen_own_count", (own_vol[b, 0] > 0.5))):
            idx = torch.nonzero(vol, as_tuple=False).to(torch.int16).cpu().numpy()
            store[f"{tag}/{nm}"] = idx
        # The ROI is DENSE, not sparse -- on `recon` it is the whole volume, so
        # coordinates would cost 7.7 MB for a mask that packbits stores in 161
        # KB. Unpack with np.unpackbits(...)[:T*H*W].reshape(shape).
        store[f"{tag}/roi_packed"] = np.packbits(
            (roi_vox[b, 0] > 0.5).cpu().numpy().ravel())
        if have < want_field:
            store[f"{tag}/field"] = (src[b, 0].detach().float().cpu()
                                     .numpy().astype(np.float16))
            if p_oracle is not None:
                store[f"{tag}/field_oracle"] = (
                    p_oracle[b, 0].detach().float().cpu()
                    .numpy().astype(np.float16))
        have += 1
    store["_count"][mode] = have


def merged_hard(prob, x, roi_vox, rate_b):
    """Completed BINARY volume: truth outside the ROI, model inside it.

    ONE readout for every model -- rank-based top-N within the ROI at the
    assay's TRAIN rate, the rule in common/evaluate.py:75. Score scales differ
    across models, so a shared numeric cut would compare calibrations rather
    than orderings; and the count comes from train, never from the held-out
    clip, so nothing leaks.
    """
    out = (x > 0.5).float().clone()
    for b in range(prob.shape[0]):
        m = roi_vox[b, 0] > 0.5
        n = int(m.sum())
        if n == 0:
            continue
        k = max(0, min(n, int(round(float(rate_b[b]) * n))))
        sel = torch.zeros(n, device=prob.device, dtype=out.dtype)
        if k:
            sel[torch.topk(prob[b, 0][m], k).indices] = 1.0
        out[b, 0][m] = sel
    return out


def own_hard(prob, x, roi_vox):
    """Completed volume using the MODEL'S OWN predicted count.

    `merged_hard` takes the count from the assay's train rate, which removes
    score-scale differences between models but also removes the count itself
    as something a model can get right or wrong. That is not neutral for a
    model with a count head: within an assay the imposed count is a CONSTANT
    (on `recon`, where the ROI is the whole volume, its within-assay variance
    is exactly zero) while the true ROI count has sd ~103 spikes, and the
    activity count head tracks that variation at within-assay r = +0.83..+0.92.
    Grading only under the shared readout hides all of it.

    The arms do NOT share a scale, and assuming they did was a bug: the
    pipeline returns `sigmoid(decode logits)`, but MaskGIT-flat's `decode()`
    returns logits (its own comment says so) and the coupled GLM's `complete()`
    returns `inten`, which holds a logit. Summing those gives a large negative
    number, which clamped to zero and made every baseline predict 0 spikes.

    So the field is converted to a probability first, by an explicit rule that
    is checked rather than assumed: a field already inside [0,1] is taken as a
    probability, anything outside it is a score and gets a sigmoid. This is the
    same score-vs-probability distinction the readout rule already draws
    elsewhere; the difference is that here it is detected from the data instead
    of being assumed uniform.

    The ROI sum of that probability is the arm's own expected spike count. No
    threshold is involved -- `best_thr_tol` is untouched. Reported ALONGSIDE
    the shared readout, never instead of it: the shared one is what makes the
    arms comparable, this one is what makes the count a capability rather than
    a constant.
    """
    out = (x > 0.5).float().clone()
    lo, hi = float(prob.min()), float(prob.max())
    pr = prob if (lo >= -1e-6 and hi <= 1.0 + 1e-6) else torch.sigmoid(prob)
    ks = []
    for b in range(prob.shape[0]):
        m = roi_vox[b, 0] > 0.5
        n = int(m.sum())
        if n == 0:
            ks.append(0)
            continue
        k = int(round(float(pr[b, 0][m].sum().clamp(min=0.0))))
        k = max(0, min(n, k))
        ks.append(k)
        sel = torch.zeros(n, device=prob.device, dtype=out.dtype)
        if k:
            sel[torch.topk(prob[b, 0][m], k).indices] = 1.0
        out[b, 0][m] = sel
    return out, ks


@torch.no_grad()
def battery(vq, vol, gct, roi_hw=None, pad_hw=None):
    """Canonical per-clip diagnostics on a completed BINARY volume.

    Returns (gap_rates (B,nbins), lct (B,9), spatial_violation (B,)).
    """
    gaps = _hard_gap_rates(vol, DEFAULT_GAP_BINS).float().cpu().numpy()
    lct = np.stack([compute_activity_ctx(vol[b, 0].cpu().numpy())
                    for b in range(vol.shape[0])])
    # `logits` drives only the SOFT rows; the fraction read below is the HARD
    # one, computed from x_hard. A finite stand-in keeps the call well-formed.
    lg = torch.where(vol > 0.5,
                     torch.full_like(vol, 4.0), torch.full_like(vol, -4.0))
    viol = np.full((vol.shape[0],), np.nan, dtype=float)
    try:
        rows = evaluate_generation_global_metrics(
            model=vq, logits=lg, x_hard=vol, global_ctx=gct,
            roi_hw=roi_hw, pad_hw=pad_hw, gap_bins=DEFAULT_GAP_BINS)
        for b, row in enumerate(rows):
            v = row.get("global_spatial_metrics", {}).get(
                "spatial_hard_violation_fraction", None)
            if v is not None:
                viol[b] = float(v)
    except Exception as exc:                       # reported, never silent
        print(f"  [battery] spatial metrics unavailable: {exc}", flush=True)
    return gaps, lct, viol


# ------------------------------------------------------------------ arms ----
@torch.no_grad()
def model_field(P, cfg, x, gct, lct, tid, specs, dev, oracle: bool = False):
    """Returns (prob_volume, roi_vox, roi_token_fraction)."""
    vq, ap, mp = P["vqvae"], P["activity_prior"], P["motif_prior"]
    blank = P["blank_code"]

    # The encoder is a transformer with `enc_attn_mask_kind="temporal_band_bi"`
    # over a +-5 time-token window (model/vqvae.py:65), so a VISIBLE token's
    # code can depend on voxels inside the hole. Encoding the full clip and
    # then calling the non-ROI codes "visible" hands the model the answer it is
    # being asked to predict. Zero the hole first.
    #
    # The ORACLE arm deliberately encodes the FULL clip: its whole purpose is
    # to be the ceiling reached by naming the true tokens.
    pm0 = vq.predict_mask_from_spec(specs, vq.token_grid, device=dev)
    roi_pre = roi_voxels(vq, pm0, vq.token_grid, x.shape[-3:])
    x_enc = x if oracle else x * (1.0 - roi_pre)

    codes, pmask, grid = _vq_codes_and_pmask_for_prior(
        vq, x_enc, gct, lct, specs, dev)
    token_grid = tuple(map(int, grid)) if grid is not None else vq.token_grid
    roi = (pmask.squeeze(-1) if pmask.dim() == 3 else pmask).bool()
    gt_active = codes[..., 0].ne(blank)

    if oracle:
        vis_f, vis_active = mp.flat_ids_from_codes(codes)
        flat = torch.full_like(vis_f, -1)
        flat[vis_active] = vis_f[vis_active]
    else:
        a_in = _make_activity_in_from_codes(
            codes, pmask, blank_code=blank, a_mask_id=ap.a_mask_id)
        a_out = ap(global_ctx=gct, local_ctx=lct, task_id=tid, a_in=a_in,
                   roi_mask=pmask, count_target=None, count_teacher_prob=0.0)
        hard_roi = ap.sample_hard_activity_for_generation(
            a_out, roi_mask=pmask, readout=cfg["activity_readout"],
            tau=cfg["activity_readout_tau"], count_mode="expected",
            count_scale=cfg["activity_count_scale"]).long()
        # THE MERGE. Truth outside the ROI, prediction inside. Without it the
        # visible region decodes blank -- see the module docstring.
        activity = torch.where(roi, hard_roi.bool(), gt_active).long()

        a_prob = None
        if cfg["activity_field"] == "soft":
            a_prob = torch.sigmoid(a_out["cell_logits"].float())
            if a_prob.dim() == 3 and a_prob.size(-1) == 1:
                a_prob = a_prob.squeeze(-1)

        motif = iterative_unmask_motif_given_activity(
            mp, activity=activity, global_ctx=gct, local_ctx=lct, task_id=tid,
            roi_mask=roi, visible_codes=codes, steps=cfg["motif_steps"],
            temperature=cfg["motif_temperature"], top_k=cfg["motif_top_k"],
            activity_prob=a_prob)
        flat = motif["flat_ids"]

    dec = decode_flat_ids_to_xgen(
        vq, flat, flat_codebook=mp.flat_codebook, grid=token_grid,
        global_ctx=gct, local_ctx=lct, roi_hw=None, pad_hw=None)
    prob = torch.sigmoid(dec["logits"].float())
    if prob.dim() == 4:
        prob = prob.unsqueeze(1)
    roi_vox = roi_voxels(vq, pmask, token_grid, x.shape[-3:])
    return prob, roi_vox, float(roi.float().mean())


def load_baseline(name: str, device: str, ckpt: str = None):
    """`ckpt` overrides the canonical path -- needed to score a seed replicate
    without displacing the shipped checkpoint, which must stay exactly where the
    reported numbers were produced from."""
    from MAGVIT_project.external_baselines import registry
    b = registry.build(name)
    b.load(Path(ckpt) if ckpt else
           Path("ckpts/external_baselines") / f"{name}.pt", device=device)
    return b


@torch.no_grad()
def baseline_field(b, cond, x, roi_vox, gen, oracle: bool = False):
    """(field, conditioned) for an external baseline.

    `conditioned` is False when the model has no completion mechanism and the
    harness fell back to free generation. That distinction is the point of the
    column: a model that cannot read the visible remainder is not merely
    scoring badly, it is answering a different question.
    """
    if oracle:
        fn = getattr(b, "oracle_field", None)
        if fn is None:
            return None, False
        v, used = fn(x), True
    else:
        v = b.complete(cond, x, roi_vox, generator=gen)
        used = v is not None
        if v is None:
            v = b.sample_intensity(cond, generator=gen)
            if v is None:
                v = b.sample(cond, generator=gen)
    v = v.float()
    if v.dim() == 4:
        v = v.unsqueeze(1)
    return v.to(roi_vox.device), used


# ------------------------------------------------------------------ main ----
def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--batches", type=int, default=8)
    ap_.add_argument("--dump", type=int, default=0,
                     help="save this many scored clips per task to "
                          "reports/external_baselines/dumps/<model>.npz for "
                          "supplement figures (0 = off, no behaviour change)")
    ap_.add_argument("--dump-field", type=int, default=2,
                     help="of those, how many also keep the dense continuous "
                          "score field (float16, ~2.6 MB/clip)")
    ap_.add_argument("--mc", type=int, default=6,
                     help="token samplings averaged for the model_mc arm")
    ap_.add_argument("--train-batches", type=int, default=60)
    ap_.add_argument("--device", default="cuda")
    ap_.add_argument("--seed", type=int, default=20260822)
    ap_.add_argument("--phase", default="4b")
    ap_.add_argument("--model", default="pipeline",
                     help="pipeline | maskgit_flat | unet3d | cvae3d | glm | dg")
    ap_.add_argument("--ckpt", default=None,
                     help="score this checkpoint instead of the canonical one")
    ap_.add_argument("--out", default=None)
    args = ap_.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    out_path = Path(args.out or
                    f"reports/external_baselines/task_eval_{args.model}.json")
    is_pipe = args.model == "pipeline"
    cfg = dict(activity_field="soft", motif_steps=12, motif_temperature=1.0,
               motif_top_k=5, activity_readout="gumbel",
               activity_readout_tau=1.0, activity_count_scale=1.0)
    if is_pipe:
        P = build_pipeline(args.device, phase=args.phase, quiet=False)
        B_EXT = None
    else:
        P = build_pipeline(args.device, phase=args.phase, quiet=True)
        B_EXT = load_baseline(args.model, args.device, args.ckpt)
        print(f"loaded baseline: {args.model}")
    gen = torch.Generator().manual_seed(args.seed)
    dump_store = {"_count": {}}
    conditioned_flag = {}

    print(f"\ntrain site maps from {args.train_batches} batches ...", flush=True)
    maps, n_train = train_site_maps(args.train_batches, seed=args.seed)
    xa_maps = cross_assay_maps(maps)
    print(f"  {len(maps)} assays from {n_train} clips "
          f"({len(xa_maps)} cross-assay maps)")

    res = {m: {a: [] for a in ARMS} for m in MODES}
    res_site = {m: {a: [] for a in ARMS} for m in MODES}
    bat = {m: {"model_gap": [], "real_gap": [], "model_lct": [],
               "real_lct": [], "model_viol": [], "real_viol": [],
               # same three tables again under the model's OWN predicted
               # count, plus the per-clip counts the calibration table needs
               "own_gap": [], "own_lct": [], "own_viol": [],
               "k_true": [], "k_assay": [], "k_own": [], "k_assay_idx": []}
           for m in MODES}
    rates = None
    roi_frac = {m: [] for m in MODES}
    keys = {m: [] for m in MODES}

    for mode in MODES:
        tid_val = TASK_ID[mode]
        print(f"\n=== task {tid_val}  {mode} ===", flush=True)
        for bi, batch in enumerate(bdata.iter_raw("test", batches=args.batches)):
            x = batch["x"].to(dev).float()
            if x.dim() == 4:
                x = x.unsqueeze(1)
            gct = batch["global_ctx"].to(dev).float()
            lct = batch["local_ctx"].to(dev).float()
            assays = batch["assay_idx"].tolist()
            B = x.shape[0]
            tid = torch.full((B,), tid_val, device=dev, dtype=torch.long)
            specs = masks_for_task(batch, x, mode)

            # `model` is ONE sample. `model_mc` averages `--mc` independent
            # token samplings -- a Monte-Carlo posterior mean. AP is a ranking
            # metric, and a sample is not the optimal ranking even when the
            # model is perfect: it commits to specific placements, while every
            # null here reports a rate. Scoring only the sample would compare a
            # draw against a posterior and call the difference model quality.
            if is_pipe:
                draws = [model_field(P, cfg, x, gct, lct, tid, specs, dev)
                         for _ in range(max(1, args.mc))]
                p_model, roi_vox, rf = draws[0]
                p_mc = (torch.stack([d[0] for d in draws]).mean(0)
                        if len(draws) > 1 else None)
                p_oracle, _, _ = model_field(P, cfg, x, gct, lct, tid, specs,
                                             dev, oracle=True)
                conditioned_flag[mode] = True
            else:
                # Geometry comes from the SHIPPED vqvae for every model, so a
                # baseline is scored on exactly the ROI the pipeline was.
                pm0 = P["vqvae"].predict_mask_from_spec(
                    specs, P["vqvae"].token_grid, device=dev)
                roi_vox = roi_voxels(P["vqvae"], pm0, P["vqvae"].token_grid,
                                     x.shape[-3:])
                rf = float((pm0 > 0.5).float().mean())
                cond, _ = bdata.split_batch(batch)
                cond = cond.to(dev)
                ds = [baseline_field(B_EXT, cond, x, roi_vox, gen)
                      for _ in range(max(1, args.mc))]
                p_model, used = ds[0]
                p_mc = (torch.stack([d[0] for d in ds]).mean(0)
                        if len(ds) > 1 else None)
                p_oracle, _ = baseline_field(B_EXT, cond, x, roi_vox, gen,
                                             oracle=True)
                conditioned_flag[mode] = used
            roi_frac[mode].append(rf)

            if rates is None:
                rates = assay_rates(maps, x.shape[-3:])
            rate_b = torch.tensor(
                [rates.get(int(a), float(np.mean(list(rates.values()))))
                 for a in assays], device=dev)
            src = p_mc if p_mc is not None else p_model
            if src is not None:
                gen_hard = merged_hard(src, x, roi_vox, rate_b)
                own_vol, k_own = own_hard(src, x, roi_vox)
                if args.dump:
                    _dump_clips(dump_store, mode, bi, assays, x, roi_vox, src,
                                gen_hard, own_vol, p_oracle, args.dump,
                                args.dump_field)
                # The volume is padded 220->224 so the width divides the patch
                # grid; the global memory map is unpadded. Without roi_hw/pad_hw
                # the spatial metric refuses the size mismatch outright.
                rhw, phw = batch.get("roi_hw", None), batch.get("pad_hw", None)
                g_gap, g_lct, g_vio = battery(
                    P["vqvae"], gen_hard, gct, rhw, phw)
                r_gap, r_lct, r_vio = battery(
                    P["vqvae"], (x > 0.5).float(), gct, rhw, phw)
                bat[mode]["model_gap"].append(g_gap)
                bat[mode]["real_gap"].append(r_gap)
                bat[mode]["model_lct"].append(g_lct)
                bat[mode]["real_lct"].append(r_lct)
                bat[mode]["model_viol"].append(g_vio)
                bat[mode]["real_viol"].append(r_vio)
                o_gap, o_lct, o_vio = battery(
                    P["vqvae"], own_vol, gct, rhw, phw)
                bat[mode]["own_gap"].append(o_gap)
                bat[mode]["own_lct"].append(o_lct)
                bat[mode]["own_viol"].append(o_vio)
                for b in range(x.shape[0]):
                    mm = roi_vox[b, 0] > 0.5
                    nn = int(mm.sum())
                    if nn == 0:
                        continue
                    bat[mode]["k_true"].append(float(x[b, 0][mm].sum()))
                    bat[mode]["k_assay"].append(
                        float(round(float(rate_b[b]) * nn)))
                    bat[mode]["k_own"].append(float(k_own[b]))
                    bat[mode]["k_assay_idx"].append(int(assays[b]))

            sp = spatial_prior_field(maps, assays, x.shape[-3:], dev)
            sp_xa = spatial_prior_field(xa_maps, assays, x.shape[-3:], dev)
            # With a full ROI (recon) nothing is visible, so the clip-derived
            # temporal profile is identically zero and the separable field
            # degenerates to an all-ties volume. That is not a weak null, it is
            # an undefined one -- report n/a rather than a meaningless number.
            has_vis = bool(((1.0 - roi_vox).sum() > 0).item())
            p_sep = (sp * visible_time_profile(x, roi_vox).view(B, 1, -1, 1, 1)
                     if has_vis else None)
            p_copy = copy_field(x, roi_vox, mode)

            fields = {"model": p_model, "model_mc": p_mc, "oracle": p_oracle,
                      "separable": p_sep, "marginal": sp,
                      "marginal_xa": sp_xa, "copy": p_copy}
            for b in range(B):
                m = roi_vox[b, 0] > 0.5
                if not bool(m.any()):
                    continue
                tgt = x[b, 0][m]
                if float(tgt.sum()) < 1.0:
                    continue          # no spike in the hole: AP undefined
                keys[mode].append(_clip_seed(batch, b))
                sd = args.seed + int(keys[mode][-1] % 100003)
                for arm, f in fields.items():
                    res[mode][arm].append(
                        float("nan") if f is None
                        else clip_ap(f[b, 0][m], tgt, seed=sd))
                    res_site[mode][arm].append(
                        float("nan") if f is None
                        else site_ap(f, x, roi_vox, b, sd))
            print(f"  batch {bi}  clips={len(keys[mode])}  roi_tok={rf:.3f}",
                  flush=True)

    # ---- report ----
    out = {"seed": args.seed, "batches": args.batches, "mc": args.mc,
           "model": args.model, "phase": args.phase,
           "n_train_clips": n_train, "tasks": {}}
    print("\n" + "=" * 78)
    print("Per-clip EXACT average precision on ROI voxels (higher is better)")
    print("=" * 78)
    print(f"{'task':<12}" + "".join(f"{a:>12}" for a in ARMS) + f"{'clips':>8}{'ROI':>7}")
    for mode in MODES:
        row = {}
        line = f"{mode:<12}"
        for a in ARMS:
            v = np.array(res[mode][a], dtype=float)
            mu = float(np.nanmean(v)) if v.size and not np.all(np.isnan(v)) else float("nan")
            row[a] = {"mean": mu,
                      "sd": float(np.nanstd(v)) if v.size else float("nan"),
                      "n": int(np.sum(~np.isnan(v)))}
            line += "         n/a" if np.isnan(mu) else f"{mu:>12.4f}"
        n = len(res[mode]["model"])
        rf = float(np.mean(roi_frac[mode])) if roi_frac[mode] else float("nan")
        print(line + f"{n:>8}{rf:>7.2f}")
        out["tasks"][mode] = {"arms": row, "n_clips": n, "roi_token_frac": rf,
                              "conditioned": bool(conditioned_flag.get(mode, False)),
                              "per_clip": {a: res[mode][a] for a in ARMS}}

        # canonical battery, same schema for every task
        B_ = bat[mode]
        if B_["model_gap"]:
            gg = np.concatenate(B_["model_gap"], 0)
            rg = np.concatenate(B_["real_gap"], 0)
            gl = np.concatenate(B_["model_lct"], 0)
            rl = np.concatenate(B_["real_lct"], 0)
            gv = np.concatenate(B_["model_viol"], 0)
            rv = np.concatenate(B_["real_viol"], 0)
            out["tasks"][mode]["battery"] = {
                "gap_bins": [list(b) for b in DEFAULT_GAP_BINS],
                "gap_labels": list(gap_bin_labels()),
                "adjacency_gap_rate": {
                    "model": np.nanmean(gg, 0).tolist(),
                    "real": np.nanmean(rg, 0).tolist(),
                },
                "activity_ctx_names": list(ACTIVITY_CTX_NAMES),
                "local_ctx": {
                    "model": np.nanmean(gl, 0).tolist(),
                    "real": np.nanmean(rl, 0).tolist(),
                    "mae": np.nanmean(np.abs(gl - rl), 0).tolist(),
                },
                "spatial_support_violation": {
                    "model": float(np.nanmean(gv)),
                    "real": float(np.nanmean(rv)),
                },
                "spatial_consistency": {
                    "model": float(max(0.0, 1.0 - np.nanmean(gv))),
                    "real": float(max(0.0, 1.0 - np.nanmean(rv))),
                },
                "readout": "roi_topN_at_assay_train_rate",
            }

            # -- the same battery under the model's OWN predicted count, plus
            # the count-calibration numbers. Separate key: the shared-readout
            # block above is what makes the four arms comparable and must not
            # move. This block is what lets a count head earn credit.
            if B_["own_lct"]:
                og = np.concatenate(B_["own_gap"], 0)
                ol = np.concatenate(B_["own_lct"], 0)
                ov = np.concatenate(B_["own_viol"], 0)
                kt = np.asarray(B_["k_true"], float)
                ka = np.asarray(B_["k_assay"], float)
                ko = np.asarray(B_["k_own"], float)
                ai = np.asarray(B_["k_assay_idx"], int)

                def _wa(u, v):
                    """Within-assay correlation.

                    Pooled correlation is not the question: both counts track
                    assay identity, so pooled r is high for a constant. What
                    matters is whether a count moves with the CLIP inside an
                    assay -- and the assay-rate count cannot, by construction.
                    """
                    u2, v2 = u.copy(), v.copy()
                    for a in np.unique(ai):
                        i = ai == a
                        if i.sum() > 1:
                            u2[i] -= u2[i].mean(); v2[i] -= v2[i].mean()
                        else:
                            u2[i] = 0.0; v2[i] = 0.0
                    if u2.std() <= 0 or v2.std() <= 0:
                        return None      # a constant has no correlation
                    return float(np.corrcoef(u2, v2)[0, 1])

                out["tasks"][mode]["battery_own"] = {
                    "gap_bins": [list(b) for b in DEFAULT_GAP_BINS],
                    "gap_labels": list(gap_bin_labels()),
                    "adjacency_gap_rate": {
                        "model": np.nanmean(og, 0).tolist(),
                        "real": np.nanmean(rg, 0).tolist(),
                    },
                    "activity_ctx_names": list(ACTIVITY_CTX_NAMES),
                    "local_ctx": {
                        "model": np.nanmean(ol, 0).tolist(),
                        "real": np.nanmean(rl, 0).tolist(),
                        "mae": np.nanmean(np.abs(ol - rl), 0).tolist(),
                    },
                    "spatial_support_violation": {
                        "model": float(np.nanmean(ov)),
                        "real": float(np.nanmean(rv)),
                    },
                    "spatial_consistency": {
                        "model": float(max(0.0, 1.0 - np.nanmean(ov))),
                        "real": float(max(0.0, 1.0 - np.nanmean(rv))),
                    },
                    "readout": "roi_topN_at_model_own_expected_count",
                }
                out["tasks"][mode]["count_calibration"] = {
                    "n": int(kt.size),
                    "true_mean": float(kt.mean()), "true_sd": float(kt.std()),
                    "assay_mean": float(ka.mean()), "assay_sd": float(ka.std()),
                    "own_mean": float(ko.mean()), "own_sd": float(ko.std()),
                    "assay_mae": float(np.abs(ka - kt).mean()),
                    "own_mae": float(np.abs(ko - kt).mean()),
                    "assay_bias": float((ka - kt).mean() / max(kt.mean(), 1e-9)),
                    "own_bias": float((ko - kt).mean() / max(kt.mean(), 1e-9)),
                    "assay_r_within_assay": _wa(ka, kt),
                    "own_r_within_assay": _wa(ko, kt),
                    "assay_r_pooled": float(np.corrcoef(ka, kt)[0, 1]),
                    "own_r_pooled": float(np.corrcoef(ko, kt)[0, 1]),
                }

    print("\n" + "=" * 78)
    print("SITE-level AP: time collapsed inside the hole -- WHERE only")
    print("=" * 78)
    print(f"{'task':<12}" + "".join(f"{a:>12}" for a in ARMS))
    for mode in MODES:
        line = f"{mode:<12}"
        for a in ARMS:
            v = np.array(res_site[mode][a], dtype=float)
            mu = float(np.nanmean(v)) if v.size and not np.all(np.isnan(v)) else float("nan")
            line += "         n/a" if np.isnan(mu) else f"{mu:>12.4f}"
            out["tasks"][mode]["arms"][a]["site_mean"] = mu
        print(line)
        out["tasks"][mode]["per_clip_site"] = {a: res_site[mode][a] for a in ARMS}

    # ---- paired tests, model vs each null ----
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None
    if wilcoxon is not None:
        print("\npaired Wilcoxon, model vs arm (positive delta = model wins)")
        for mode in MODES:
            mv = np.array(res[mode]["model"], dtype=float)
            for a in ARMS:
                if a == "model":
                    continue
                av = np.array(res[mode][a], dtype=float)
                ok = ~(np.isnan(mv) | np.isnan(av))
                if ok.sum() < 6 or np.allclose(mv[ok], av[ok]):
                    continue
                d = float(np.mean(mv[ok] - av[ok]))
                p = float(wilcoxon(mv[ok], av[ok]).pvalue)
                print(f"  {mode:<11} vs {a:<10} delta={d:+.4f}  p={p:.3g}  n={int(ok.sum())}")
                out["tasks"][mode]["arms"][a]["delta_vs_model"] = d
                out["tasks"][mode]["arms"][a]["wilcoxon_p"] = p

    dst = out_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")

    if args.dump:
        n = dump_store.pop("_count")
        dump_store.pop("_assays", None)
        # Provenance travels WITH the volumes. A figure built from these has to
        # be able to name the protocol that produced them, and a loose npz in a
        # folder cannot be traced back once the shell history is gone.
        dump_store["_meta"] = np.asarray([json.dumps({
            "model": args.model, "ckpt": args.ckpt, "seed": args.seed,
            "batches": args.batches, "mc": args.mc,
            "clips_per_task": n,
            "shape": [int(v) for v in dump_store["_shape"]],
            "report": str(dst),
            "readouts": {
                "gen_shared_readout": "top-N in ROI at the assay TRAIN rate; "
                                      "the one readout every arm shares",
                "gen_own_count": "top-N at the arm's OWN expected count "
                                 "(sum of sigmoid over the ROI)",
                "field": "continuous MC-averaged score, float16; this is what "
                         "AP ranks, and it is NOT what the binary panels show",
                "real": "ground truth",
                "roi_packed": "the hole, np.packbits over the flattened "
                              "(T,H,W) mask; unpackbits[:T*H*W].reshape(shape)"},
            "note": "int16 nonzero coordinates (n,3) = (t,y,x) for binary "
                    "volumes; dense (T,H,W) float16 for fields.",
        })], dtype=object)
        dd = Path("reports/external_baselines/dumps")
        dd.mkdir(parents=True, exist_ok=True)
        tag = args.model + (f"_{Path(args.ckpt).stem}" if args.ckpt else "")
        f = dd / f"{tag}.npz"
        np.savez_compressed(f, **dump_store)
        print(f"wrote {f}  ({f.stat().st_size/1e6:.1f} MB)  clips/task={n}")


if __name__ == "__main__":
    main()
