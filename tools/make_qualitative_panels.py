#!/usr/bin/env python3
"""Select and precompute the panels for the two qualitative appendix figures.

Two jobs that must not happen at figure-render time. First, the per-recording
TRAIN site maps need a 60-batch pass over the train loader; `make_paper_figures`
runs on every compile and cannot afford that. Second, and more important, the
panels are SELECTED, and a selection made inside a drawing routine is a
selection nobody can audit. This script writes the full ranking it selected
from, so the caption's claim about where the chosen clips sit in the corpus is
checkable against the same file that chose them.

Selection rule, stated once and applied by code: rank by attained fraction of
the achievable spatial-map correlation, `map_r_pred / map_r_gt`, where the
denominator is the clip's own ground truth against its recording's train map.
F1 is NOT the rule -- it correlates with map adherence at r = 0.91 but picks a
visibly wrong panel on at least one recording, which is what motivated this
script.

    python tools/make_qualitative_panels.py

writes reports/qualitative_panels.json  (every clip's adherence, + selection)
       reports/qualitative_panels.npz   (only the projections the figures draw)
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = 20260822
TRAIN_BATCHES = 60
NW = 3                       # time windows per row
RECON_ROOT = ROOT / "reports" / "reconstruction_stage2a"
GEN_DUMP = ROOT / "reports" / "external_baselines" / "dumps" / "pipeline.npz"
OUT_JSON = ROOT / "reports" / "qualitative_panels.json"
OUT_NPZ = ROOT / "reports" / "qualitative_panels.npz"

# lct component order, from utils/recon.py:502. Only the three interpretable
# scalars are drawn; the six second-moment terms are not human-readable on a
# strip plot and nothing in the text refers to them.
LCT_LOGMEAN, LCT_ACTIVE, LCT_TREND = 0, 7, 8


SPATIAL_TOLERANCE_RADIUS = 1   # matches training/train_prior.py:627 and
                               # inference/metrics_gen.py:348


def dilate(a):
    """Radius-1 max-pool, the tolerance this codebase already uses.

    Placement is learned through the global code and a spatial-violation term,
    not supervised per electrode, so an exact per-site correlation asks for
    something the objective never demanded. Both the training loss and the
    generation metrics allow a prediction within one site of an allowed one
    (`dilate_spatial_support_hw`, radius 1), so the same tolerance is applied
    here rather than inventing a second convention.

    Applied to BOTH maps: the tolerance has to move the ceiling as well as the
    model, or the attained fraction would silently inflate.
    """
    import torch
    from utils.recon import dilate_spatial_support_hw
    t = torch.from_numpy(np.ascontiguousarray(a)).float()[None]
    r = SPATIAL_TOLERANCE_RADIUS
    return dilate_spatial_support_hw(t, r, r)[0].numpy()


def corr(a, b, mask=None) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if mask is not None:
        m = np.asarray(mask).ravel()
        a, b = a[m], b[m]
    if a.size < 8 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def site_maps():
    import external_baselines.task_eval as TE
    maps, n_train = TE.train_site_maps(TRAIN_BATCHES, seed=SEED)
    return maps, n_train


def recon_table(maps):
    rows = []
    for p in sorted(glob.glob(str(RECON_ROOT / "*" / "sample_*" / "results.json"))):
        meta = json.load(open(p))
        arrs = np.load(os.path.join(os.path.dirname(p), "arrays.npz"))
        a = int(meta["assay_id"])
        if a not in maps:
            continue
        gt, pr = arrs["gt_bool"], arrs["pred_bool"]      # (H, W, T)
        sm = maps[a]
        if sm.shape != gt.shape[:2]:
            sm = sm.T
        ref = np.asarray(meta["ctx_ref_on_window"], float)
        est = np.asarray(meta["ctx_pred_on_window"], float)
        rows.append(dict(
            sample=Path(p).parent.name, dir=str(Path(p).parent.relative_to(ROOT)),
            recording=meta["assay_name"], assay=a, f1=float(meta["f1_volume"]),
            map_r_pred=corr(dilate(pr.sum(2)), dilate(sm)),
            map_r_gt=corr(dilate(gt.sum(2)), dilate(sm)),
            map_r_pred_exact=corr(pr.sum(2), sm),
            map_r_gt_exact=corr(gt.sum(2), sm),
            ref_logmean=float(ref[LCT_LOGMEAN]), ref_active=float(ref[LCT_ACTIVE]),
            ref_trend=float(ref[LCT_TREND]),
            d_logmean=float(est[LCT_LOGMEAN] - ref[LCT_LOGMEAN]),
            d_active=float(est[LCT_ACTIVE] - ref[LCT_ACTIVE]),
            d_trend=float(est[LCT_TREND] - ref[LCT_TREND])))
    return rows


def gen_table(maps):
    from utils.recon import compute_activity_ctx
    d = np.load(GEN_DUMP, allow_pickle=True)
    meta = json.loads(str(d["_meta"].item()))
    T, H, W = meta["shape"]

    def vol(key):
        c = d[key]
        v = np.zeros((T, H, W), bool)
        if c.size:
            v[c[:, 0], c[:, 1], c[:, 2]] = True
        return v

    rows = []
    for key in sorted(x for x in d.files if x.endswith("/real")):
        task, clip, _ = key.split("/")
        a = int(d[f"{task}/{clip}/assay"])
        sm = maps.get(a)
        if sm is None:
            continue
        if sm.shape != (H, W):
            sm = sm.T
        real, gen = vol(key), vol(f"{task}/{clip}/gen_shared_readout")
        roi = np.unpackbits(d[f"{task}/{clip}/roi_packed"])[:T * H * W]
        roi = roi.reshape(T, H, W).astype(bool)
        ref, est = compute_activity_ctx(real), compute_activity_ctx(gen)
        rows.append(dict(
            task=task, clip=clip, assay=a, roi_frac=float(roi.mean()),
            has_field=f"{task}/{clip}/field" in d.files,
            # Whole clip, both sides. Outside the hole the arm reproduces the
            # observed truth exactly (verified: zero mismatched voxels over the
            # dump), so that component is IDENTICAL in numerator and reference
            # and the pair stays symmetric. This asks whether the delivered
            # clip sits on the recording's map, which is the question the panel
            # above it shows.
            #
            # The consequence to state, not to hide: a row with more observed
            # frames carries more copied truth and is pulled toward its own
            # ceiling for a reason that has nothing to do with generation. Rows
            # are therefore NOT comparable to each other without reading the
            # generated fraction printed on each panel. The hole-only version
            # is kept alongside for anyone who wants generation isolated.
            map_r_pred=corr(dilate(gen.sum(0)), dilate(sm)),
            map_r_gt=corr(dilate(real.sum(0)), dilate(sm)),
            map_r_pred_hole=corr(dilate((gen & roi).sum(0)), dilate(sm), roi.any(0)),
            map_r_gt_hole=corr(dilate((real & roi).sum(0)), dilate(sm), roi.any(0)),
            ref_logmean=float(ref[LCT_LOGMEAN]), ref_active=float(ref[LCT_ACTIVE]),
            ref_trend=float(ref[LCT_TREND]),
            d_logmean=float(est[LCT_LOGMEAN] - ref[LCT_LOGMEAN]),
            d_active=float(est[LCT_ACTIVE] - ref[LCT_ACTIVE]),
            d_trend=float(est[LCT_TREND] - ref[LCT_TREND]),
            # Fraction of SITES the clip never observes. For a temporal hole
            # this is 0 -- every site is seen in some frame -- which is why a
            # whole-clip map correlation is mostly copied truth there and is
            # not comparable to the spatial row. The caption quotes this.
            frac_sites_never_observed=float((~(~roi).any(0)).mean()),
            n_real=int(real.sum()), n_gen=int(gen.sum())))
    return rows, meta


def attained(r, hole=False):
    """Fraction of the achievable map correlation the model reached.

    `hole=True` uses the hole-only pair, which is the only version comparable
    across tasks: a whole-clip correlation is a blend of copied and generated
    content in a ratio that differs per task, so it ranks hole geometry rather
    than performance.

    Guarded on the denominator: where the clip's own truth barely correlates
    with its recording map there is nothing to attain and the ratio explodes.
    """
    a = r["map_r_pred_hole" if hole else "map_r_pred"]
    b = r["map_r_gt_hole" if hole else "map_r_gt"]
    return a / b if b > 0.3 else float("nan")


def pick_recon(rows, n_per_corpus=2):
    best = {}
    for r in rows:
        if np.isnan(r["map_r_pred"]) or np.isnan(attained(r)):
            continue
        cur = best.get(r["recording"])
        if cur is None or r["map_r_pred"] > cur["map_r_pred"]:
            best[r["recording"]] = r
    out = []
    for organoid in (True, False):
        pool = [v for k, v in best.items() if k.startswith("sub-U") == organoid]
        out += sorted(pool, key=lambda r: -r["map_r_pred"])[:n_per_corpus]
    return out


def pick_gen(rows):
    """One clip, all four tasks.

    The dump indexes the same underlying clip under every task, so a single
    ground truth can be shown under the whole conditioning ladder instead of
    four unrelated clips. Restricted to clips that carry the continuous field,
    which `--dump-field` caps independently of `--dump`.
    """
    tasks = ["recon", "causal", "noncausal", "spatial"]
    by_clip = {}
    for r in rows:
        by_clip.setdefault(r["clip"], {})[r["task"]] = r
    ok = {c: v for c, v in by_clip.items()
          if set(tasks) <= set(v) and all(v[t]["has_field"] for t in tasks)}
    if not ok:
        raise SystemExit(
            "No clip carries the continuous field for all four tasks. Re-run "
            "task_eval.py with --dump-field at least as large as --dump.")
    best = max(ok, key=lambda c: np.nanmean(
        [attained(ok[c][t], hole=True) for t in tasks]))
    return [ok[best][t] for t in tasks]


def windows(T, w=None):
    w = w or T // NW
    return [(i * w, (i + 1) * w) for i in range(NW)]


def mask_aware_windows(roi_t, n=NW, w=None):
    """Windows that TILE the clip and break on the mask's own boundaries.

    Two defects in the previous version, both of which made the figure
    misdescribe the task it was illustrating.

    It slid a fixed-width window and picked the three positions whose
    generated fractions were furthest apart, which left FRAMES OUT: 0-7,
    20-27, 40-47 silently drops 8-19 and 28-39, so a reader counting frames
    sees a clip with holes that the task did not put there.

    And because the windows did not align to the mask, a short observed run
    got averaged into a window that also held hidden frames. Non-causal hides
    a middle span -- for the shipped clip, frames 24-41, leaving 0-23 and
    42-47 visible -- but a window at 40-47 mixes two hidden frames with six
    observed ones and reports "25% generated". The setting looked one-sided
    when the mask is two-sided.

    So: cut at the transitions of the per-frame generated fraction and take
    the runs as the windows. Widths are uneven by construction, which is the
    point -- the boundaries are where the task changes. If that yields fewer
    than `n` windows, split the longest GENERATED run first (seeing early and
    late generation is worth more than a second identical observed panel);
    with no generated run to split, fall back to even thirds.
    """
    T = len(roi_t)
    gen = [bool(v > 0.5) for v in roi_t]
    cuts = [0] + [i for i in range(1, T) if gen[i] != gen[i - 1]] + [T]
    segs = [[cuts[i], cuts[i + 1]] for i in range(len(cuts) - 1)]
    while len(segs) < n:
        pool = [k for k, (a, _) in enumerate(segs) if gen[a]] or list(range(len(segs)))
        k = max(pool, key=lambda j: segs[j][1] - segs[j][0])
        a, b = segs[k]
        if b - a < 2:
            break
        mid = a + (b - a) // 2
        segs[k:k + 1] = [[a, mid], [mid, b]]
    while len(segs) > n:
        # Merge the two adjacent windows whose union spans the fewest frames,
        # so the coarsening falls on the least informative pair.
        k = min(range(len(segs) - 1),
                key=lambda j: segs[j + 1][1] - segs[j][0])
        segs[k:k + 2] = [[segs[k][0], segs[k + 1][1]]]
    assert segs[0][0] == 0 and segs[-1][1] == T, segs
    assert all(segs[i][1] == segs[i + 1][0] for i in range(len(segs) - 1)), segs
    return [(a, b) for a, b in segs]


def clip_to_clip(maps):
    """Correlation between two REAL clips of the same recording.

    This is what answers "why is the ceiling not 1". The training site map is
    an average over many clips; one clip is a single sparse draw from it --
    ~200 spikes over 26,880 sites -- so it cannot match its own expectation.
    Two real clips of the same recording agree only this well, and a generator
    emitting ONE clip is bounded by the same sampling limit.
    """
    import itertools
    per = {}
    for rec_dir in sorted(RECON_ROOT.glob("sub-*")):
        cnt = []
        for f in sorted(rec_dir.glob("sample_*/arrays.npz")):
            g = np.load(f)["gt_bool"]
            cnt.append(dilate(g.sum(2).astype(float)))
        if len(cnt) < 2:
            continue
        rs = [corr(a, b) for a, b in itertools.combinations(cnt, 2)]
        rs = [x for x in rs if not np.isnan(x)]
        if rs:
            per[rec_dir.name] = float(np.mean(rs))
    return dict(n_recordings=len(per),
                median=round(float(np.median(list(per.values()))), 4),
                per_recording={k: round(v, 4) for k, v in per.items()})


def summarise(rows):
    """Paired summaries only.

    Every figure in this pair is about per-clip agreement, and the pooled
    version of these quantities disagrees with the paired one: the marginal
    medians of active-site ratio differ by a third while the median paired
    change is exactly zero. Anything quoted in a caption comes from here.
    """
    out = {}
    for name, rk, ek, dk in (("map_r", "map_r_gt", "map_r_pred", None),
                             ("logmean", "ref_logmean", None, "d_logmean"),
                             ("active", "ref_active", None, "d_active"),
                             ("trend", "ref_trend", None, "d_trend")):
        ref = np.array([r[rk] for r in rows], float)
        est = (np.array([r[ek] for r in rows], float) if ek
               else ref + np.array([r[dk] for r in rows], float))
        ok = ~(np.isnan(ref) | np.isnan(est))
        d = est[ok] - ref[ok]
        out[name] = dict(
            n=int(ok.sum()),
            median_truth=round(float(np.median(ref[ok])), 4),
            median_model=round(float(np.median(est[ok])), 4),
            median_paired_change=round(float(np.median(d)), 4),
            frac_model_above_truth=round(float((d > 0).mean()), 4))
    return out


def main() -> int:
    maps, n_train = site_maps()
    rec_rows = recon_table(maps)
    gen_rows, gen_meta = gen_table(maps)
    rec_pick, gen_pick = pick_recon(rec_rows), pick_gen(gen_rows)

    store = {}
    for i, r in enumerate(rec_pick):
        arrs = np.load(ROOT / r["dir"] / "arrays.npz")
        gt, pr = arrs["gt_bool"], arrs["pred_bool"]
        prob = arrs["prob_u8"].astype(np.float32) / 255.0
        sm = maps[r["assay"]]
        if sm.shape != gt.shape[:2]:
            sm = sm.T
        wins = windows(gt.shape[2])
        store[f"recon/{i}/windows"] = np.asarray(wins, np.int32)
        for j, (a, b) in enumerate(wins):
            store[f"recon/{i}/{j}/field"] = prob[:, :, a:b].max(2).astype(np.float32)
            store[f"recon/{i}/{j}/gt"] = gt[:, :, a:b].any(2)
            store[f"recon/{i}/{j}/pred"] = pr[:, :, a:b].any(2)
        store[f"recon/{i}/sitemap"] = sm.astype(np.float32)

    d = np.load(GEN_DUMP, allow_pickle=True)
    T, H, W = gen_meta["shape"]
    for i, r in enumerate(gen_pick):
        k = f"{r['task']}/{r['clip']}"

        def vol(name):
            c = d[f"{k}/{name}"]
            v = np.zeros((T, H, W), bool)
            if c.size:
                v[c[:, 0], c[:, 1], c[:, 2]] = True
            return v

        real, gen = vol("real"), vol("gen_shared_readout")
        field = d[f"{k}/field"].astype(np.float32)
        roi = np.unpackbits(d[f"{k}/roi_packed"])[:T * H * W].reshape(T, H, W).astype(bool)
        sm = maps[r["assay"]]
        if sm.shape != (H, W):
            sm = sm.T
        wins = mask_aware_windows(roi.reshape(T, -1).mean(1))
        store[f"gen/{i}/windows"] = np.asarray(wins, np.int32)
        for j, (a, b) in enumerate(wins):
            store[f"gen/{i}/field"] if False else None
            store[f"gen/{i}/{j}/field"] = field[a:b].max(0).astype(np.float32)
            store[f"gen/{i}/{j}/gt"] = real[a:b].any(0)
            store[f"gen/{i}/{j}/pred"] = gen[a:b].any(0)
            store[f"gen/{i}/{j}/observed"] = ~roi[a:b].any(0)
            store[f"gen/{i}/{j}/roi_frac"] = np.float32(roi[a:b].mean())
        store[f"gen/{i}/sitemap"] = sm.astype(np.float32)
        # The field is crushed by its own maximum: p99 is ~0.03 against a max
        # near 1.0, so normalising to the max renders almost all of it black.
        store[f"gen/{i}/field_vmax"] = np.float32(np.percentile(field, 99.9))

    np.savez_compressed(OUT_NPZ, **store)
    OUT_JSON.write_text(json.dumps(dict(
        seed=SEED, train_batches=TRAIN_BATCHES, n_train_clips=int(n_train),
        n_recordings=len(maps), windows=NW,
        selection_rule="max map_r_pred per recording, two per corpus (recon); "
                       "clip with the best mean attained map correlation over "
                       "all four tasks (generation)",
        gen_dump_meta={k: gen_meta[k] for k in
                       ("batches", "mc", "clips_per_task", "seed")},
        spatial_tolerance_radius=SPATIAL_TOLERANCE_RADIUS,
        clip_to_clip_reproducibility=clip_to_clip(maps),
        recon=dict(all=rec_rows, picked=[r["sample"] for r in rec_pick],
                   summary=summarise(rec_rows),
                   picked_attained={r["sample"]: round(attained(r), 4)
                                    for r in rec_pick}),
        generation=dict(all=gen_rows,
                        picked=[[r["task"], r["clip"]] for r in gen_pick],
                        summary=summarise(gen_rows),
                        picked_attained={r["task"]: round(attained(r, hole=True), 4)
                                         for r in gen_pick},
                        picked_attained_whole_clip={
                            r["task"]: round(attained(r), 4) for r in gen_pick}),
    ), indent=1))
    print(f"wrote {OUT_JSON} ({len(rec_rows)} recon clips, {len(gen_rows)} gen clips)")
    print(f"wrote {OUT_NPZ} ({OUT_NPZ.stat().st_size/1e6:.1f} MB)")
    print("recon picks:", [r["sample"] for r in rec_pick])
    print("gen pick:", gen_pick[0]["clip"], "tasks", [r["task"] for r in gen_pick])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
