#!/usr/bin/env python3
"""What the dense arm's codes CONTAIN, not just how many it uses.

The first pass compared `active_codes_nonblank` / `perplexity_nonblank` between
the two arms and reported the dense arm as using MORE codes less uniformly.
That comparison was not sound. "nonblank" in those metrics means *declared
active by the model*, and the dense arm declares everything active -- so its
numbers were computed over all 1024 tokens, 92% of which are genuinely empty,
while the sparse arm's were computed over the ~8% that carry a spike. Different
denominators, different populations.

This script fixes that and then asks the question the counts cannot answer.
The TRUE blank mask is recomputed from the raw input for BOTH arms
(`compute_blank_mask`, which reads the volume, not the flag), and every
statistic is reported on matched populations:

  CONTENT TOKENS   patches with at least one spike -- what the alphabet is for
  BLANK TOKENS     patches that are exactly zero

Reported per level:
  * codes used, usage entropy in nats, perplexity  -- on CONTENT tokens only
  * for the dense arm, the same on BLANK tokens, plus how much of its alphabet
    blanks consume and how much it SHARES with content tokens. A code used by
    both cannot be read off to mean either.
  * eta^2 of code identity on patch statistics (spike count, spatial spread,
    temporal spread) -- the fraction of content variance the code explains.
    This is the content measure; entropy is only a usage measure. Same
    statistic already used for code->activity in Stage 1.
  * usage-weighted codebook norms and the effective rank of the used vectors,
    because unused entries are LARGE, not near init, so an unweighted norm is
    misleading.

    python ablations/sparse_encoder_content.py --batches 24
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent))
import MAGVIT_project.main as M                                # noqa: E402
from MAGVIT_project.ablations.sparse_encoder import build      # noqa: E402

CK = ROOT / "ckpts" / "ablations"
OUT = ROOT / "reports" / "ablation_sparse_encoder_content.json"


def _entropy_nats(counts):
    c = np.asarray(counts, np.float64); c = c[c > 0]
    if c.sum() <= 0: return 0.0, 1.0, 0
    p = c / c.sum()
    h = float(-(p * np.log(p)).sum())
    return h, float(np.exp(h)), int(len(c))


def _eta2(labels, values):
    """Fraction of variance in `values` explained by group identity `labels`.

    eta^2 = 1 - SS_within / SS_total. 0 means the code says nothing about the
    patch; 1 means the code determines it. Groups with a single member
    contribute no within-variance and would inflate this, so they are dropped.
    """
    labels = np.asarray(labels); values = np.asarray(values, np.float64)
    if values.size < 2: return float("nan")
    keep = np.zeros(labels.shape, bool)
    uniq, cnt = np.unique(labels, return_counts=True)
    big = set(uniq[cnt >= 2].tolist())
    for i, l in enumerate(labels): keep[i] = l in big
    if keep.sum() < 2: return float("nan")
    labels, values = labels[keep], values[keep]
    ss_tot = float(((values - values.mean()) ** 2).sum())
    if ss_tot <= 0: return float("nan")
    ss_w = 0.0
    for u in np.unique(labels):
        v = values[labels == u]
        ss_w += float(((v - v.mean()) ** 2).sum())
    return 1.0 - ss_w / ss_tot


def _eta2_null(labels, values, n_perm=8, rng=None):
    """eta^2 together with its permutation null.

    Raw eta^2 rises with the NUMBER of groups regardless of signal: 1024 ladder
    paths over ~15k tokens will explain variance by counting alone. Shuffling
    the labels keeps the group-size profile and destroys only the association,
    so `excess = eta2 - eta2_null` is the part that is not an artifact of
    granularity. Without this the nested ladder below is uninterpretable --
    every deeper level would look better by construction.
    """
    rng = rng or np.random.default_rng(0)
    obs = _eta2(labels, values)
    nulls = []
    for _ in range(n_perm):
        nulls.append(_eta2(rng.permutation(labels), values))
    nulls = [v for v in nulls if v == v]
    nul = float(np.mean(nulls)) if nulls else float("nan")
    return obs, nul, (obs - nul)


def _ladder_eta2(codes, content, stats, names=("spike_count", "temporal_spread",
                                               "spatial_spread")):
    """eta^2 of the LADDER PATH on patch content, nested z1 -> z1z2 -> z1z2z3.

    The decoder consumes the SUM of the levels, so an individual child index is
    not a meaningful unit: child k under parent A and child k under parent B are
    unrelated vectors, and pooling them across parents is what drove the
    per-level child eta^2 to ~0. The unit that matches how the alphabet is used
    is the path -- which is precisely the V=961 flat alphabet (32x8x4 = 1024
    ladder sums, deduped).

    Nested, so the INCREMENT from one row to the next is what that level buys.
    """
    out = []
    key = None
    for l in range(codes.shape[1]):
        cl = codes[:, l]
        key = cl.astype(np.int64) if key is None else key * 4096 + cl.astype(np.int64)
        valid = content & (codes[:, :l + 1] >= 0).all(axis=1)
        k = key[valid]
        row = {"levels_used": l + 1, "n_paths": int(np.unique(k).size),
               "n_tokens": int(valid.sum())}
        for i, nm in enumerate(names):
            obs, nul, exc = _eta2_null(k, stats[valid][:, i])
            row[nm] = {"eta2": obs, "eta2_perm_null": nul, "excess": exc}
        out.append(row)
    return out


@torch.no_grad()
def collect(model, loader, device, n_batches, patch):
    """Codes + per-patch content statistics, with the TRUE blank mask."""
    pT, pH, pW = patch
    codes_all, blank_all, stats_all = [], [], []
    for i, batch in enumerate(loader):
        if i >= n_batches: break
        x = batch["x"].to(device).float()
        if x.dim() == 4: x = x.unsqueeze(1)
        out = model(
            x,
            local_ctx=batch["local_ctx"].to(device).float(),
            global_ctx=batch["global_ctx"].to(device).float(),
        )
        codes = out["codes"]                                  # (B,N,L)
        grid = out["grid"]
        # TRUE blank mask, read off the volume -- NOT out["blank_mask"], which
        # the dense arm zeroes by construction.
        true_blank = model.compute_blank_mask(x, grid)        # (B,N)

        B, C, T, H, W = x.shape
        t, h, w = grid
        pat = (x.view(B, C, t, pT, h, pH, w, pW)
                 .permute(0, 2, 4, 6, 3, 5, 7, 1)
                 .reshape(B, t * h * w, pT * pH * pW))
        cnt = pat.sum(-1)                                     # spikes in patch
        # spatial / temporal spread of the spikes inside the patch
        pv = pat.view(B, -1, pT, pH, pW)
        ti = torch.arange(pT, device=device).view(1, 1, pT, 1, 1).float()
        hi = torch.arange(pH, device=device).view(1, 1, 1, pH, 1).float()
        den = cnt.clamp_min(1e-6)
        mt = (pv * ti).sum((2, 3, 4)) / den
        vt = ((pv * (ti - mt.view(B, -1, 1, 1, 1)) ** 2).sum((2, 3, 4)) / den)
        mh = (pv * hi).sum((2, 3, 4)) / den
        vh = ((pv * (hi - mh.view(B, -1, 1, 1, 1)) ** 2).sum((2, 3, 4)) / den)

        codes_all.append(codes.cpu().numpy())
        blank_all.append(true_blank.cpu().numpy())
        stats_all.append(torch.stack([cnt, vt, vh], -1).cpu().numpy())
    return (np.concatenate(codes_all).reshape(-1, codes_all[0].shape[-1]),
            np.concatenate(blank_all).reshape(-1),
            np.concatenate(stats_all).reshape(-1, 3))


def analyse(name, model, codes, blank, stats):
    L = codes.shape[1]
    content, blanks = ~blank, blank
    r = {"arm": name,
         "n_tokens": int(codes.shape[0]),
         "true_blank_frac": float(blank.mean()),
         "levels": []}
    for l in range(L):
        cl = codes[:, l]
        valid = cl >= 0                       # blank_code = -1 in the sparse arm
        cc = cl[content & valid]
        hb, pb, nb = _entropy_nats(np.bincount(cc)) if cc.size else (0, 1, 0)
        lev = {"level": l + 1,
               "content_codes_used": nb,
               "content_entropy_nats": hb,
               "content_perplexity": pb,
               "content_uniformity": (pb / nb) if nb else float("nan")}
        cbl = cl[blanks & valid]
        if cbl.size:
            hh, pp, nn_ = _entropy_nats(np.bincount(cbl))
            shared = len(set(np.unique(cc).tolist()) & set(np.unique(cbl).tolist()))
            lev.update(blank_codes_used=nn_, blank_entropy_nats=hh,
                       codes_shared_content_and_blank=shared,
                       frac_content_codes_also_used_by_blanks=(
                           shared / nb if nb else float("nan")))
        else:
            lev.update(blank_codes_used=0, blank_entropy_nats=0.0,
                       codes_shared_content_and_blank=0,
                       frac_content_codes_also_used_by_blanks=0.0)
        if cc.size:
            s = stats[content & valid]
            lev["eta2_code_to_content"] = {
                "spike_count": _eta2(cc, s[:, 0]),
                "temporal_spread": _eta2(cc, s[:, 1]),
                "spatial_spread": _eta2(cc, s[:, 2])}
        r["levels"].append(lev)

    r["ladder"] = _ladder_eta2(codes, content, stats)

    emb = model.vq.tree_embeds[0].detach().float().cpu().numpy().reshape(-1, 64)
    used = np.unique(codes[:, 0][(codes[:, 0] >= 0) & content])
    if used.size:
        U = emb[used]
        sv = np.linalg.svd(U - U.mean(0, keepdims=True), compute_uv=False)
        p = (sv ** 2) / max((sv ** 2).sum(), 1e-12)
        r["l1_codebook"] = {
            "n_used_by_content": int(used.size),
            "mean_norm_used": float(np.linalg.norm(U, axis=1).mean()),
            "mean_norm_unused": float(np.linalg.norm(
                emb[np.setdiff1d(np.arange(emb.shape[0]), used)], axis=1).mean())
                if used.size < emb.shape[0] else float("nan"),
            "effective_rank": float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())),
            "blank_token_norm": float(np.linalg.norm(
                model.vq.blank_token.detach().float().cpu().numpy()))}
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--ckpt", default=None,
                    help="analyse ONE checkpoint (e.g. the shipped tokenizer) "
                         "instead of the two ablation arms")
    ap.add_argument("--tag", default="shipped", help="name for --ckpt output")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ad = M.find_assays()
    _tr, val, _te, _m = M.make_loaders(assay_dict=ad,
                                       assay_indices=list(ad.keys()),
                                       per_assay_quota=M.per_assay_quota_stage12)
    b0 = next(iter(val)); img = tuple(b0["x"].shape[-3:])
    full = tuple(map(int, b0["full_hw"][0]))

    arms = ([(a.tag, False, Path(a.ckpt))] if a.ckpt else
            [("sparse", False, CK / "sparse_abl_sparse_best.pt"),
             ("dense", True, CK / "sparse_abl_dense_best.pt")])
    res = []
    for name, dense, ck in arms:
        model = build(img, full, device, dense=dense, seed=0)
        sd = torch.load(ck, map_location="cpu")
        model.load_state_dict(sd.get("model", sd.get("state_dict", sd)), strict=False)
        model.eval()
        codes, blank, stats = collect(model, val, device, a.batches, M.patch_size)
        res.append(analyse(name, model, codes, blank, stats))
        print(f"[{name}] {codes.shape[0]} tokens, true blank frac "
              f"{blank.mean():.4f}", flush=True)

    out = (OUT if not a.ckpt else
           OUT.with_name(f"ablation_sparse_encoder_content_{a.tag}.json"))
    out.write_text(json.dumps({"arms": res, "batches": a.batches,
                               "ckpt": a.ckpt}, indent=1))

    print("\nON CONTENT TOKENS ONLY (patches with >=1 spike) -- matched population")
    hdr = f"{'arm':<8}{'lvl':>4}{'codes':>7}{'entropy':>9}{'ppl':>8}{'unif':>7}"
    hdr += f"{'blankCodes':>12}{'shared':>8}{'eta2 cnt':>10}{'eta2 tspr':>10}"
    print(hdr); print("-" * len(hdr))
    for r in res:
        for lev in r["levels"]:
            e = lev.get("eta2_code_to_content", {})
            print(f"{r['arm']:<8}{lev['level']:>4}{lev['content_codes_used']:>7}"
                  f"{lev['content_entropy_nats']:>9.3f}{lev['content_perplexity']:>8.1f}"
                  f"{lev['content_uniformity']:>7.2f}"
                  f"{lev['blank_codes_used']:>12}"
                  f"{lev['codes_shared_content_and_blank']:>8}"
                  f"{e.get('spike_count', float('nan')):>10.3f}"
                  f"{e.get('temporal_spread', float('nan')):>10.3f}")
    print("\nlevel-1 codebook geometry")
    for r in res:
        g = r.get("l1_codebook", {})
        print(f"  {r['arm']:<8} used_by_content={g.get('n_used_by_content')}  "
              f"eff_rank={g.get('effective_rank', float('nan')):.2f}  "
              f"|used|={g.get('mean_norm_used', float('nan')):.2f}  "
              f"|unused|={g.get('mean_norm_unused', float('nan')):.2f}  "
              f"|blank_tok|={g.get('blank_token_norm', float('nan')):.3f}")
    print("\nLADDER PATH eta^2 on content tokens, nested (decoder sees the SUM)")
    print("excess = eta2 - permutation null; the null grows with n_paths, so")
    print("raw eta2 across rows is NOT comparable and excess is.")
    hh = (f"{'arm':<9}{'levels':>7}{'paths':>7}"
          f"{'cnt eta2':>10}{'null':>7}{'excess':>8}"
          f"{'tspr eta2':>11}{'null':>7}{'excess':>8}")
    print(hh); print("-" * len(hh))
    for r in res:
        for row in r.get("ladder", []):
            c, t = row["spike_count"], row["temporal_spread"]
            print(f"{r['arm']:<9}{row['levels_used']:>7}{row['n_paths']:>7}"
                  f"{c['eta2']:>10.3f}{c['eta2_perm_null']:>7.3f}{c['excess']:>8.3f}"
                  f"{t['eta2']:>11.3f}{t['eta2_perm_null']:>7.3f}{t['excess']:>8.3f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
