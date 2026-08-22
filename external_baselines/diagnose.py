#!/usr/bin/env python3
"""Interpretable diagnostics for a generative baseline.

The comparison table says which model scores better. It does not say whether a
model works, and at this sparsity a model can score respectably while being
degenerate -- which is the specific failure a dense 3-D MaskGIT is prone to
here. If the decoder emits a nearly flat field, the rate-matched Bernoulli
readout still produces the right NUMBER of spikes and simply scatters them,
reproducing "all blank predictions" while hiding it behind a correct rate.

So every number below is reported against something that makes it readable:
chance, the `random`-context control, or the real-vs-real value.

    1. RECONSTRUCTION   can the tokenizer represent a clip at all?
                        AUPRC via the pipeline's own PRCurveAccumulator, exact
                        and at the pipeline's (1,1,1) tolerance, so it is
                        directly comparable to AUPRC_cond.

    2. DEGENERACY       is the generated field informative, or flat?
                        AUPRC of the generated per-voxel field against the clip
                        it was conditioned on, versus the SAME field scored
                        against a different clip. Equal values mean the field
                        carries nothing clip-specific.

    3. LOCAL CONTEXT    are the 9 lct features actually followed?
                        Recomputed from the generated volume with
                        utils.recon.compute_activity_ctx and compared to the
                        vector the model was given, against the `random` control.

    4. SPATIAL MAP      does activity land on the right electrodes?
                        Per-electrode map correlation, cosine and active-site
                        IoU against the true clip, against the same controls.

    5. ADJACENCY        P(spike at neighbour | spike) by spatial displacement
                        and temporal lag, generated versus real.

    python external_baselines/diagnose.py --baseline maskgit_flat
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/media/derik/Seagate Desktop Drive/organoid_data")

from MAGVIT_project.external_baselines import registry
from MAGVIT_project.external_baselines.common import data as bdata
from MAGVIT_project.external_baselines.common.ladder import ContextLadder
from MAGVIT_project.utils.metrics import PRCurveAccumulator
from MAGVIT_project.utils.recon import compute_activity_ctx
from MAGVIT_project.utils.constants import ACTIVITY_CTX_NAMES

BANK = "ckpts/context_prior.pkl"


# ----------------------------------------------------------------------

def _as5d(t):
    """PRCurveAccumulator reduces over a channel axis, so it needs (B,C,T,H,W)."""
    return t if t.dim() == 5 else t.unsqueeze(1)


def _step_ap(tp, fp, fn):
    """Step-wise average precision -- sum_i (R_i - R_{i-1}) * P_i.

    `PRCurveAccumulator._summarize` integrates the PR curve with
    `torch.trapezoid`, i.e. it draws a straight line between adjacent operating
    points. Linear interpolation is valid in ROC space and invalid in PR space
    (Davis & Goadrich, ICML 2006): the reachable curve between two points is
    convex-downward, so the chord always sits above it.

    That is not a pedantic distinction here. A tokenizer trained with
    pos_weight ~= 7300 saturates its sigmoid, so on some clips a SINGLE voxel
    sits at exactly 1.0. If that voxel happens to be a real spike, the curve
    gains a point at (recall ~ 0.001, precision 1.0), and the chord from there
    down to the next point sweeps 0.4 of recall at a fictitious precision --
    worth +0.21 AUPRC on one measured clip, from one voxel.

    The step-wise sum takes the precision actually achieved at each operating
    point and never interpolates, so it cannot be moved by an isolated point.
    On this coarse 200-point grid it is a mild UNDER-estimate, which is the
    right direction for a claim.
    """
    prec = (tp / (tp + fp).clamp(min=1.0)).flip(0)     # thresholds high -> low
    rec = (tp / (tp + fn).clamp(min=1.0)).flip(0)      # recall now ascending
    d = torch.diff(rec, prepend=rec.new_zeros(1))
    return float((d.clamp(min=0) * prec).sum())


def _recall_gap(tp, fn):
    """Largest single-step recall jump: how badly the grid resolves this model."""
    rec = tp / (tp + fn).clamp(min=1.0)
    return float(torch.diff(rec).abs().max())


def _auprc(prob, target, device, tol=None, *, full=False):
    prob, target = _as5d(prob), _as5d(target)
    acc = PRCurveAccumulator(
        num_thr=200, device=device,
        radius_t=None if tol is None else tol[0],
        radius_h=0 if tol is None else tol[1],
        radius_w=0 if tol is None else tol[2],
    )
    acc.update(prob, target, None)
    out = acc._summarize(acc.tp, acc.fp, acc.fn, acc.thr_grid)
    if not full:
        return out
    return out + (_step_ap(acc.tp, acc.fp, acc.fn), _recall_gap(acc.tp, acc.fn))


def reconstruction(baseline, batches, device):
    """Section 1 -- can the tokenizer represent a clip at all?"""
    tok = getattr(baseline, "tokenizer", None)
    if tok is None:
        return None
    tok.eval()
    # The pipeline's encoder is context-conditioned; the flat tokenizers are not.
    # Withholding gct/lct from the pipeline would measure a model that does not
    # exist, so the tokenizer declares what it needs and gets exactly that.
    wants_cond = bool(getattr(baseline, "tokenizer_wants_cond", False))
    ex_p, ex_t, ids_all = [], [], []
    n_sp_true = n_sp_rec = 0.0
    for cond, real in bdata.iter_split("test", batches=batches):
        x = real.unsqueeze(1).to(device)
        with torch.no_grad():
            logits, idx, _ = tok(x, cond.to(device)) if wants_cond else tok(x)
            p = torch.sigmoid(logits)
        ex_p.append(p.cpu()); ex_t.append(x.cpu())
        ids_all.append(idx.reshape(idx.shape[0], -1).cpu())
        n_sp_true += float(x.sum())
        # count at a rate-matched threshold, the same readout generation uses
        k = int(round(float(x.sum())))
        flat = p.reshape(-1)
        thr = torch.topk(flat, max(k, 1)).values.min() if k > 0 else 1.0
        n_sp_rec += float((p >= thr).sum())
    P = torch.cat(ex_p).to(device); T = torch.cat(ex_t).to(device)
    exact = _auprc(P, T, device, None, full=True)
    toler = _auprc(P, T, device, (1, 1, 1), full=True)
    ids = torch.cat(ids_all)
    V = baseline.cfg["n_codes"]
    # The pipeline shim parks inactive tokens in bin V (the VQ blank slot). It is
    # not a codebook entry, so it is counted separately and dropped before the
    # usage and perplexity figures -- otherwise ~89% blanks would dominate both.
    cnt_all = torch.bincount(ids.reshape(-1), minlength=V + 1).float()
    blank_frac = float(cnt_all[V] / cnt_all.sum()) if cnt_all.numel() > V else 0.0
    cnt = cnt_all[:V]
    pr = cnt / cnt.sum()
    perp = float(torch.exp(-(pr[pr > 0] * pr[pr > 0].log()).sum()))
    return {
        "auprc_exact": exact[0], "best_f1_exact": exact[1],
        "auprc_tol111": toler[0], "best_f1_tol111": toler[1],
        "ap_step_exact": exact[3], "ap_step_tol111": toler[3],
        "recall_gap_exact": exact[4], "recall_gap_tol111": toler[4],
        "codes_used": int((cnt > 0).sum()), "n_codes": baseline.cfg["n_codes"],
        "codebook_perplexity": perp,
        "spikes_true": n_sp_true, "spikes_recovered_at_matched_thr": n_sp_rec,
        "base_rate": float(T.mean()),
        "blank_token_fraction": blank_frac,
    }


def _field(baseline, cond, gen):
    """Continuous per-voxel field for a conditioning batch, if the model has one."""
    f = baseline.sample_intensity(cond, generator=gen)
    return None if f is None else f.float().cpu()


def degeneracy(baseline, batches, device, ladder, gen):
    """Section 2 -- informative field, or flat field plus a rate knob?"""
    aligned, shifted, stds, blanks, uniq = [], [], [], [], []
    for cond, real in bdata.iter_split("test", batches=batches):
        cond_d = ladder.apply(cond.to(device), "global_full_local")
        v = baseline.sample(cond_d, generator=gen).float().cpu()
        f = _field(baseline, cond_d, gen)
        blanks.append(float((v.flatten(1).sum(1) == 0).float().mean()))
        if f is None:
            continue
        stds.append(float(f.flatten(1).std(dim=1).mean()))
        B = f.shape[0]
        pr = torch.sigmoid(f).to(device)
        tg = real.to(device)
        aligned.append(_auprc(pr, tg, device, (1, 1, 1))[0])
        # same field, scored against a DIFFERENT clip in the batch
        roll = torch.roll(tg, 1, dims=0)
        shifted.append(_auprc(pr, roll, device, (1, 1, 1))[0])
        if hasattr(baseline, "prior"):
            ids = baseline.prior.generate(
                cond_d.global_ctx, cond_d.local_ctx, steps=baseline.cfg["steps"],
                generator=gen, device=device)
            uniq += [int(len(torch.unique(ids[i]))) for i in range(ids.shape[0])]
    out = {"all_blank_fraction": float(np.mean(blanks))}
    if aligned:
        out.update({
            "auprc_vs_own_clip": float(np.mean(aligned)),
            "auprc_vs_other_clip": float(np.mean(shifted)),
            "clip_specific_margin": float(np.mean(aligned) - np.mean(shifted)),
            "field_std": float(np.mean(stds)),
        })
    if uniq:
        out["unique_tokens_per_sample"] = float(np.mean(uniq))
        out["n_codes"] = baseline.cfg["n_codes"]
    return out



def _map_controls(gms, tms, assays, *, max_pairs=4000, seed=0):
    """Is the spatial map clip-specific, or a per-assay lookup?

    `map_pearson_r` alone cannot tell those apart. A model that emits one fixed
    average map per assay scores well on it -- and DG and the GLM do exactly
    that, because `assay_idx` reaches them unchanged in EVERY regime (the ladder
    randomises gct/lct, not assay identity), so their train-fitted site maps are
    live even under `random`.

    The discriminator is to score the same generated map against a DIFFERENT
    clip from the SAME assay. A per-assay lookup scores identically; a model
    that reads this clip's context loses ground. `within_assay_gap` is that
    loss, and it is the only number here a static site map cannot fake.

    Uses only volumes already generated -- no extra draws, so every other
    figure in this report is unchanged.
    """
    if not gms:
        return {}
    rng = np.random.default_rng(seed)
    by_assay = {}
    for k, a in enumerate(assays):
        by_assay.setdefault(a, []).append(k)

    def _r(i, j):
        g, t = gms[i], tms[j]
        if g.std() == 0 or t.std() == 0:
            return np.nan
        return float(np.corrcoef(g, t)[0, 1])

    own = [_r(k, k) for k in range(len(gms))]
    same, diff = [], []
    for k, a in enumerate(assays):
        pool = [j for j in by_assay[a] if j != k]
        if pool:
            same.append(_r(k, int(rng.choice(pool))))
        other = [j for j in range(len(gms)) if assays[j] != a]
        if other:
            diff.append(_r(k, int(rng.choice(other))))
    o, s_, d = (float(np.nanmean(x)) if x else np.nan for x in (own, same, diff))
    return {
        "map_r_own_clip": o,
        "map_r_other_clip_same_assay": s_,
        "map_r_other_assay": d,
        "within_assay_gap": o - s_,
        "assay_identity_component": s_ - d,
        "n_same_assay_pairs": len(same),
    }


def context_and_space(baseline, batches, device, ladder, gen, regimes):
    """Sections 3-5 -- lct adherence, spatial map, adjacency."""
    # Two different questions, and under partial context they diverge:
    #   lct_pairs   got vs the vector the model was GIVEN  -> ADHERENCE
    #   lct_true    got vs the held-out clip's own lct     -> ACCURACY
    # Under `global_full_local` these are the same vector, so the distinction
    # only becomes visible on the partial rungs -- which is exactly where the
    # practical question lives ("I know the prep and roughly how active it is").
    res = {r: {"lct_err": [], "lct_pairs": [], "lct_true": [], "map_r": [],
               "map_cos": [], "site_iou": [], "gm": [], "tm": [], "assay": []}
           for r in regimes}
    adj = {r: [] for r in regimes}
    adj_real = []
    disp = (1, 2, 3)
    lags = (1, 2, 3, 4, 5, 6, 7)

    def adjacency(v):                       # v (T,H,W) float
        out = []
        s = float(v.sum())
        if s == 0:
            return [np.nan] * (len(disp) + len(lags))
        for d in disp:
            acc = 0.0
            for dh, dw in ((d, 0), (-d, 0), (0, d), (0, -d)):
                acc += float((torch.roll(v, shifts=(dh, dw), dims=(1, 2)) * v).sum())
            out.append(acc / (4 * s))
        for k in lags:
            out.append(float((v[k:] * v[:-k]).sum()) / s)
        return out

    for cond, real in bdata.iter_split("test", batches=batches):
        cond_d = cond.to(device)
        # The clip's OWN lct, recomputed the same way the generated one is, so
        # the two are commensurable. Not a leak: it is a metric target, never
        # reaches a model.
        true_lct = np.stack([compute_activity_ctx(real[i].numpy())
                             for i in range(real.shape[0])])
        for i in range(real.shape[0]):
            adj_real.append(adjacency(real[i]))
        for r in regimes:
            cr = ladder.apply(cond_d, r)
            v = baseline.sample(cr, generator=gen).float().cpu()
            lct_used = cr.local_ctx.cpu().numpy()
            for i in range(v.shape[0]):
                got = compute_activity_ctx(v[i].numpy())
                res[r]["lct_err"].append(np.abs(got - lct_used[i]))
                res[r]["lct_pairs"].append((got, lct_used[i]))
                res[r]["lct_true"].append((got, true_lct[i]))
                gm = v[i].sum(0).numpy().ravel()
                tm = real[i].sum(0).numpy().ravel()
                if gm.std() > 0 and tm.std() > 0:
                    res[r]["map_r"].append(float(np.corrcoef(gm, tm)[0, 1]))
                    res[r]["map_cos"].append(
                        float(gm @ tm / (np.linalg.norm(gm) * np.linalg.norm(tm))))
                res[r]["gm"].append(gm); res[r]["tm"].append(tm)
                res[r]["assay"].append(str(cond.assay_name[i])
                                       if i < len(cond.assay_name)
                                       else int(cond.assay_idx[i]))
                ga, ta = gm > 0, tm > 0
                u = float((ga | ta).sum())
                res[r]["site_iou"].append(float((ga & ta).sum() / u) if u else np.nan)
                adj[r].append(adjacency(v[i]))

    out = {"adjacency_labels": [f"space_d{d}" for d in disp] + [f"time_lag{k}" for k in lags],
           "adjacency_real": np.nanmean(np.array(adj_real, float), axis=0).tolist(),
           "regimes": {}}
    for r in regimes:
        pairs = res[r]["lct_pairs"]
        got = np.array([p[0] for p in pairs]); want = np.array([p[1] for p in pairs])
        tp = res[r]["lct_true"]
        got_t = np.array([q[0] for q in tp]); want_t = np.array([q[1] for q in tp])

        def _rr(g, w):
            ok = np.isfinite(g) & np.isfinite(w)
            if ok.sum() <= 3 or g[ok].std() == 0 or w[ok].std() == 0:
                return np.nan, np.nan
            return (float(np.corrcoef(g[ok], w[ok])[0, 1]),
                    float(np.mean(np.abs(g[ok] - w[ok]))))

        per_feat = {}
        for j, nm in enumerate(ACTIVITY_CTX_NAMES):
            rr, mae = _rr(got[:, j], want[:, j])
            rt, maet = _rr(got_t[:, j], want_t[:, j])
            per_feat[nm] = {"r": rr, "mae": mae, "r_vs_true": rt, "mae_vs_true": maet}
        # Per-clip scalar, so the comparison admits a PAIRED test. The mean-r
        # summaries above are pooled over clips and cannot be tested that way.
        # Each feature is divided by its spread across the real test clips, so
        # the nine features contribute comparably instead of the metric being
        # whichever one has the largest units.
        sd = np.nanstd(want_t, axis=0)
        sd[~np.isfinite(sd) | (sd <= 0)] = 1.0
        per_clip = np.nanmean(np.abs(got_t - want_t) / sd, axis=1)

        out["regimes"][r] = {
            "lct_z_mae_per_clip": [float(v) for v in per_clip],
            "lct_z_mae": float(np.nanmean(per_clip)),
            # Single-number summaries: the mean over the 9 features of the
            # correlation between requested and realised (adherence) and
            # between realised and the true clip (accuracy).
            "lct_r_mean_vs_used": float(np.nanmean(
                [per_feat[n]["r"] for n in ACTIVITY_CTX_NAMES])),
            "lct_r_mean_vs_true": float(np.nanmean(
                [per_feat[n]["r_vs_true"] for n in ACTIVITY_CTX_NAMES])),
            **_map_controls(res[r]["gm"], res[r]["tm"], res[r]["assay"]),
            "lct_per_feature": per_feat,
            "lct_mae_mean": float(np.nanmean(np.array(res[r]["lct_err"]))),
            "map_pearson_r": float(np.nanmean(res[r]["map_r"])) if res[r]["map_r"] else np.nan,
            "map_cosine": float(np.nanmean(res[r]["map_cos"])) if res[r]["map_cos"] else np.nan,
            "active_site_iou": float(np.nanmean(res[r]["site_iou"])),
            "adjacency": np.nanmean(np.array(adj[r], float), axis=0).tolist(),
        }
    return out


# ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--regimes",
                    default="random,global_only,global_partial_local,global_full_local")
    ap.add_argument("--ablate", default=None, choices=("assay_map",),
                    help="assay_map: delete the fitted per-assay site maps so "
                         "every clip falls back to the pooled `global_site`. "
                         "This is exactly what DG and the coupled GLM already "
                         "do for an assay they never saw (`site_p.get(a, "
                         "global_site)`), so it simulates a held-out assay with "
                         "no refit. Favourable to them: global_site is the mean "
                         "over all 31 fitted assays INCLUDING the test one, a "
                         "1/31 leak we are not removing.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bdata.ensure_repo_cwd()
    b = registry.build(args.baseline)
    # The pipeline adapter reads main.CKPTS, so there is no per-baseline file.
    ckpt = Path(args.ckpt) if args.ckpt else (
        None if args.baseline == "pipeline"
        else Path(f"ckpts/external_baselines/{args.baseline}.pt"))
    b.load(ckpt, device=args.device)
    if args.ablate == "assay_map":
        n = 0
        for attr in ("site_p", "site_logit"):
            d = getattr(b, attr, None)
            if isinstance(d, dict):
                n += len(d); d.clear()
        if n == 0:
            raise SystemExit(f"{args.baseline} has no per-assay site map to ablate")
        # The per-assay BASE RATE is deliberately kept: lct carries
        # log_mean_firing_density, so a rate is available for an unseen assay.
        # Only the memorised spatial map is withheld.
        print(f"[ablate] dropped {n} per-assay site maps -> pooled global_site\n")

    gen = torch.Generator().manual_seed(args.seed)
    ladder = ContextLadder(BANK, seed=args.seed)
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]

    print(f"\nbaseline : {b.meta.name}\ncheckpoint: {ckpt}\n"
          f"clips    : {args.batches} test batches\n")

    rec = reconstruction(b, args.batches, args.device)
    if rec:
        print("=" * 74)
        print("1. TOKENIZER RECONSTRUCTION   (encode -> VQ -> decode, real clips)")
        print(f"   base rate (chance AUPRC)      {rec['base_rate']:.3e}")
        print(f"   AP step-wise exact            {rec['ap_step_exact']:.4f}   "
              f"= {rec['ap_step_exact']/max(rec['base_rate'],1e-12):,.0f}x chance"
              f"   <- READ THIS ONE")
        print(f"   AP step-wise tolerant (1,1,1) {rec['ap_step_tol111']:.4f}")
        print(f"   best F1 exact / tolerant      {rec['best_f1_exact']:.4f} / "
              f"{rec['best_f1_tol111']:.4f}")
        print(f"   -- trapezoid AUPRC, as utils/metrics.py reports it --")
        print(f"   AUPRC exact / tolerant        {rec['auprc_exact']:.4f} / "
              f"{rec['auprc_tol111']:.4f}")
        print(f"   interpolation inflation       {rec['auprc_exact']-rec['ap_step_exact']:+.4f} / "
              f"{rec['auprc_tol111']-rec['ap_step_tol111']:+.4f}")
        print(f"   largest recall step on grid   {rec['recall_gap_exact']:.4f} / "
              f"{rec['recall_gap_tol111']:.4f}   (small = the grid resolves this model)")
        print(f"   codebook used                 {rec['codes_used']}/{rec['n_codes']}  "
              f"perplexity {rec['codebook_perplexity']:.1f}")
        if rec.get("blank_token_fraction"):
            print(f"   blank (inactive) tokens       {rec['blank_token_fraction']:.3f}  "
                  f"(excluded from the two figures above)")

    deg = degeneracy(b, args.batches, args.device, ladder, gen)
    print("\n" + "=" * 74)
    print("2. IS THE GENERATED FIELD INFORMATIVE?   (the 'all blank' check)")
    print(f"   all-blank samples             {deg['all_blank_fraction']:.3f}")
    if "auprc_vs_own_clip" in deg:
        print(f"   AUPRC vs its OWN clip         {deg['auprc_vs_own_clip']:.4f}")
        print(f"   AUPRC vs a DIFFERENT clip     {deg['auprc_vs_other_clip']:.4f}   <- control")
        print(f"   clip-specific margin          {deg['clip_specific_margin']:+.4f}   "
              f"{'INFORMATIVE' if deg['clip_specific_margin'] > 0 else 'NOT clip-specific'}")
        print(f"   decoder field std             {deg['field_std']:.3f}  "
              f"(a flat field would be ~0)")
    if "unique_tokens_per_sample" in deg:
        print(f"   unique tokens per sample      {deg['unique_tokens_per_sample']:.1f}"
              f" of {deg['n_codes']} codes / 1024 slots")

    cs = context_and_space(b, args.batches, args.device, ladder, gen, regimes)
    print("\n" + "=" * 74)
    print("3. LOCAL CONTEXT ADHERENCE   (9 lct features recomputed from the sample)")
    hdr = "   " + f"{'feature':26s}" + "".join(f"{r[:16]:>18s}" for r in regimes)
    print(hdr)
    for nm in ACTIVITY_CTX_NAMES:
        row = "   " + f"{nm:26s}"
        for r in regimes:
            e = cs["regimes"][r]["lct_per_feature"][nm]
            row += f"{('%+.2f / %+.2f' % (e['r'], e['r_vs_true'])):>18s}"
        print(row)
    print("   " + f"{'MEAN over 9 features':26s}" +
          "".join(f"{('%+.2f / %+.2f' % (cs['regimes'][r]['lct_r_mean_vs_used'], cs['regimes'][r]['lct_r_mean_vs_true'])):>18s}"
                  for r in regimes))
    print("   each cell is  r(vs the lct GIVEN) / r(vs the TRUE clip's lct).")
    print("   They coincide under global_full_local and separate on the partial rungs:")
    print("   the first is obedience, the second is whether obedience was enough.")
    print("   " + f"{'per-clip z-MAE vs true':26s}" +
          "".join(f"{cs['regimes'][r]['lct_z_mae']:18.4f}" for r in regimes)
          + "   <- lower is better, paired-testable")

    print("\n" + "=" * 74)
    print("4. SPATIAL MAP ADHERENCE   (per-electrode counts vs the true clip)")
    print("   " + f"{'metric':26s}" + "".join(f"{r[:16]:>18s}" for r in regimes))
    for k, lab in (("map_pearson_r", "pearson r"), ("map_cosine", "cosine"),
                   ("active_site_iou", "active-site IoU"),
                   ("map_r_other_clip_same_assay", "  vs other clip, same assay"),
                   ("map_r_other_assay", "  vs a different assay"),
                   ("within_assay_gap", "  WITHIN-ASSAY GAP"),
                   ("assay_identity_component", "  assay-identity component")):
        if k not in cs["regimes"][regimes[0]]:
            continue
        print("   " + f"{lab:26s}" +
              "".join(f"{cs['regimes'][r][k]:18.4f}" for r in regimes))
    print("   " + "-" * 26 + "\n   WITHIN-ASSAY GAP is the only row a fixed per-assay"
          " site map cannot fake.")

    print("\n" + "=" * 74)
    print("5. ADJACENCY   P(spike at neighbour | spike)")
    print("   " + f"{'':16s}{'REAL':>12s}" + "".join(f"{r[:14]:>16s}" for r in regimes))
    for i, lab in enumerate(cs["adjacency_labels"]):
        print("   " + f"{lab:16s}{cs['adjacency_real'][i]:12.5f}" +
              "".join(f"{cs['regimes'][r]['adjacency'][i]:16.5f}" for r in regimes))

    out = {"baseline": b.meta.name, "checkpoint": str(ckpt),
           "batches": args.batches, "reconstruction": rec,
           "degeneracy": deg, "context_and_space": cs}
    path = Path(args.out or f"reports/external_baselines/diagnose_{args.baseline}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
