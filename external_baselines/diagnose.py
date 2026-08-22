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


def _auprc(prob, target, device, tol=None):
    prob, target = _as5d(prob), _as5d(target)
    acc = PRCurveAccumulator(
        num_thr=200, device=device,
        radius_t=None if tol is None else tol[0],
        radius_h=0 if tol is None else tol[1],
        radius_w=0 if tol is None else tol[2],
    )
    acc.update(prob, target, None)
    return acc._summarize(acc.tp, acc.fp, acc.fn, acc.thr_grid)


def reconstruction(baseline, batches, device):
    """Section 1 -- can the tokenizer represent a clip at all?"""
    tok = getattr(baseline, "tokenizer", None)
    if tok is None:
        return None
    tok.eval()
    ex_p, ex_t, ids_all = [], [], []
    n_sp_true = n_sp_rec = 0.0
    for cond, real in bdata.iter_split("test", batches=batches):
        x = real.unsqueeze(1).to(device)
        with torch.no_grad():
            logits, idx, _ = tok(x)
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
    exact = _auprc(P, T, device, None)
    toler = _auprc(P, T, device, (1, 1, 1))
    ids = torch.cat(ids_all)
    cnt = torch.bincount(ids.reshape(-1), minlength=baseline.cfg["n_codes"]).float()
    pr = cnt / cnt.sum()
    perp = float(torch.exp(-(pr[pr > 0] * pr[pr > 0].log()).sum()))
    return {
        "auprc_exact": exact[0], "best_f1_exact": exact[1],
        "auprc_tol111": toler[0], "best_f1_tol111": toler[1],
        "codes_used": int((cnt > 0).sum()), "n_codes": baseline.cfg["n_codes"],
        "codebook_perplexity": perp,
        "spikes_true": n_sp_true, "spikes_recovered_at_matched_thr": n_sp_rec,
        "base_rate": float(T.mean()),
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


def context_and_space(baseline, batches, device, ladder, gen, regimes):
    """Sections 3-5 -- lct adherence, spatial map, adjacency."""
    res = {r: {"lct_err": [], "lct_pairs": [], "map_r": [], "map_cos": [],
               "site_iou": []} for r in regimes}
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
                gm = v[i].sum(0).numpy().ravel()
                tm = real[i].sum(0).numpy().ravel()
                if gm.std() > 0 and tm.std() > 0:
                    res[r]["map_r"].append(float(np.corrcoef(gm, tm)[0, 1]))
                    res[r]["map_cos"].append(
                        float(gm @ tm / (np.linalg.norm(gm) * np.linalg.norm(tm))))
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
        per_feat = {}
        for j, nm in enumerate(ACTIVITY_CTX_NAMES):
            g, w = got[:, j], want[:, j]
            ok = np.isfinite(g) & np.isfinite(w)
            rr = float(np.corrcoef(g[ok], w[ok])[0, 1]) if ok.sum() > 3 and g[ok].std() > 0 and w[ok].std() > 0 else np.nan
            per_feat[nm] = {"r": rr, "mae": float(np.mean(np.abs(g[ok] - w[ok])))}
        out["regimes"][r] = {
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
    ap.add_argument("--regimes", default="random,global_full_local")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bdata.ensure_repo_cwd()
    b = registry.build(args.baseline)
    ckpt = Path(args.ckpt or f"ckpts/external_baselines/{args.baseline}.pt")
    b.load(ckpt, device=args.device)
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
        print(f"   AUPRC exact                   {rec['auprc_exact']:.4f}   "
              f"= {rec['auprc_exact']/max(rec['base_rate'],1e-12):,.0f}x chance")
        print(f"   AUPRC tolerant (1,1,1)        {rec['auprc_tol111']:.4f}   "
              f"<- comparable to the pipeline's AUPRC_cond")
        print(f"   best F1 exact / tolerant      {rec['best_f1_exact']:.4f} / "
              f"{rec['best_f1_tol111']:.4f}")
        print(f"   codebook used                 {rec['codes_used']}/{rec['n_codes']}  "
              f"perplexity {rec['codebook_perplexity']:.1f}")

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
            row += f"{('r=%+.2f mae=%.3f' % (e['r'], e['mae'])):>18s}"
        print(row)

    print("\n" + "=" * 74)
    print("4. SPATIAL MAP ADHERENCE   (per-electrode counts vs the true clip)")
    print("   " + f"{'metric':26s}" + "".join(f"{r[:16]:>18s}" for r in regimes))
    for k, lab in (("map_pearson_r", "pearson r"), ("map_cosine", "cosine"),
                   ("active_site_iou", "active-site IoU")):
        print("   " + f"{lab:26s}" +
              "".join(f"{cs['regimes'][r][k]:18.4f}" for r in regimes))

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
