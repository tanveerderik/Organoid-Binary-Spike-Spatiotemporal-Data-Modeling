#!/usr/bin/env python3
"""Is the motif alphabet shared across recordings, or private to each?

The paper claims generation works because a compact alphabet of spatiotemporal
motifs is REUSED. Nothing in reports/ has ever checked the reuse half. This
does, on the frozen shipped tokenizer, with no retraining: encode the test
clips, tally which flat codes each recording emits, and ask three questions.

  1. How much of the alphabet does one recording use, and how much of it is
     shared? `shared_core` counts entries used by at least k recordings.

  2. How much does knowing the recording tell you about the code? The entropy
     decomposition H(code) vs H(code | recording). The ratio is 1.0 when the
     recording label is uninformative -- a fully shared alphabet -- and falls
     toward 0 as each recording acquires a private vocabulary.

  3. Is the observed cross-recording overlap distinguishable from chance? The
     label-shuffle null reassigns clips to recordings at random, preserving
     each recording's clip count, and recomputes the mean pairwise Jaccard.
     Without it, "recordings share 60% of their codes" is uninterpretable: two
     random samples from one distribution already share most of it.

The blank token is EXCLUDED throughout. It is 91.7% of all tokens, every
recording emits it, and including it would drive every overlap statistic to
~1.0 while measuring nothing about motifs.

Jaccard is reported on code SETS (unweighted) and the entropy decomposition on
code COUNTS (weighted). They answer different questions -- which motifs are
available to a recording, and how often it reaches for them -- and a shared
alphabet used with very different frequencies is a real and reportable outcome.

RAW JACCARD IS NOT USABLE HERE and is emitted only so the confound is visible.
Recordings contribute wildly unequal token counts (117 to 1756 on the test
split, because the archive's recordings differ in length and firing rate), and a
set's size grows with sampling effort, so raw pairwise Jaccard tracks token
count almost perfectly -- r = 0.96 against log tokens on the first run. Read
naively it says organoid recordings share more motifs with each other than slice
recordings do; that is entirely the organoid recordings being longer. The
rarefied statistic is the one to quote: every recording is subsampled to the
same token budget, many times, and Jaccard is averaged over draws. Same fix
ecology uses for species richness, same reason.

Plug-in entropy is downward-biased at small sample sizes, which pushes
H(code | recording) down and the reuse ratio down, so the plug-in ratio is a
CONSERVATIVE estimate of sharing. The Miller-Madow correction is reported
alongside it.

    python analysis/motif_reuse.py --batches 24
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
import MAGVIT_project.main as M                                    # noqa: E402
from MAGVIT_project.ablations.sparse_encoder import build          # noqa: E402

OUT = ROOT / "reports" / "analysis_motif_reuse.json"
CKPT = "ckpts/vqvae_stage2a_best.pt"
FLAT = "ckpts/stage2b_flat_codebook.pt"
BLANK_CODE = -1
# Ladder shape (32, 8, 4). K2/K3 are the strides the flatten uses; they are
# read from the checkpoint rather than hard-coded so a re-trained ladder of a
# different shape fails loudly instead of silently mislabelling every token.
_LADDER_EXPECTED = (32, 8, 4)


def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy in nats of a count vector. Empty -> 0.0."""
    tot = counts.sum()
    if tot <= 0:
        return 0.0
    p = counts[counts > 0] / tot
    return float(-(p * np.log(p)).sum())


def _entropy_mm(counts: np.ndarray) -> float:
    """Miller-Madow bias-corrected entropy, nats.

    Plug-in entropy underestimates when the number of observations is small
    relative to the alphabet, which is exactly this regime: 117 tokens over 961
    possible codes for the smallest recording. The correction adds
    (K_observed - 1) / (2N).
    """
    n = counts.sum()
    if n <= 0:
        return 0.0
    k = int((counts > 0).sum())
    return _entropy(counts) + (k - 1) / (2.0 * n)


def _rarefied_sets(counts: np.ndarray, n_sub: int, rng) -> list:
    """One draw of `n_sub` tokens from each recording's own multiset.

    Sampling WITHOUT replacement, because the question is what a recording's
    vocabulary looks like at a fixed observation budget -- the same budget for
    every recording, which is the whole point.
    """
    out = []
    for row in counts:
        tot = int(row.sum())
        if tot <= n_sub:
            out.append(set(np.nonzero(row)[0].tolist()))
            continue
        # Expand to a token list only via repeat, then choose without
        # replacement. `row` is small (<= 961 entries) so this is cheap.
        toks = np.repeat(np.arange(row.size), row)
        pick = rng.choice(toks, size=n_sub, replace=False)
        out.append(set(np.unique(pick).tolist()))
    return out


def _jaccard_matrix(sets) -> np.ndarray:
    n = len(sets)
    J = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            u = len(sets[i] | sets[j])
            J[i, j] = J[j, i] = (len(sets[i] & sets[j]) / u) if u else 0.0
    return J


def _block_mean(J: np.ndarray, rows, cols, same: bool) -> float:
    """Mean of an off-diagonal block. `same` excludes the i==j diagonal."""
    vals = [J[i, j] for i in rows for j in cols if (i < j if same else True)]
    return float(np.mean(vals)) if vals else float("nan")


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--flat", default=FLAT)
    ap.add_argument("--batches", type=int, default=0,
                    help="0 = the whole test split. Anything less risks "
                         "leaving recordings with no sampled clip, whose "
                         "empty code set would then read as zero overlap.")
    ap.add_argument("--shuffles", type=int, default=500)
    ap.add_argument("--rarefy-draws", type=int, default=200,
                    help="subsample draws per recording for the rarefied "
                         "Jaccard, which is the only comparable one")
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve the checkpoint paths against the project root, not the cwd.
    # main.py's data and cache paths are relative to the project directory, so
    # this script has to be run from there anyway -- but a checkpoint path that
    # silently depends on cwd is the kind of thing that reads as a missing file
    # rather than as a wrong working directory.
    flat_path = Path(a.flat)
    ckpt_path = Path(a.ckpt)
    if not flat_path.is_absolute():
        flat_path = ROOT / flat_path
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    for q in (flat_path, ckpt_path):
        if not q.exists():
            raise SystemExit(f"missing checkpoint: {q}")

    flat = torch.load(str(flat_path), map_location="cpu")
    merge_map = flat["merge_map"].long()
    V = int(flat["embed"].shape[0])
    ladder = tuple(flat["ladder"])
    if ladder != _LADDER_EXPECTED:
        raise SystemExit(
            f"ladder is {ladder}, not {_LADDER_EXPECTED}. The flatten strides "
            f"below are wrong for it -- update them before trusting any number "
            f"this script prints.")
    _, K2, K3 = ladder

    ad = M.find_assays()
    _tr, _val, test, _meta = M.make_loaders(
        assay_dict=ad, assay_indices=list(ad.keys()),
        per_assay_quota=M.per_assay_quota_stage12)
    b0 = next(iter(test))
    model = build(tuple(b0["x"].shape[-3:]), tuple(map(int, b0["full_hw"][0])),
                  dev, dense=False, seed=0)
    sd = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(sd.get("model", sd.get("state_dict", sd)),
                          strict=False)
    model.eval()

    # The loader's assay ids are NOT 0..n-1. `find_assays` indexes the full
    # archive listing (0..33 here) and the unused recordings leave gaps, so the
    # ids are sparse while `counts` needs dense rows. Conflating the two silently
    # writes recording 31's tokens into row 31 of a 31-row array -- or into the
    # wrong recording entirely. Build the mapping explicitly.
    assay_ids = sorted(ad.keys())
    row_of = {aid: i for i, aid in enumerate(assay_ids)}
    n_assays = len(assay_ids)
    counts = np.zeros((n_assays, V), dtype=np.int64)
    clip_rows = []            # (dense row, per-clip count vector)
    mm = merge_map.to(dev)

    for i, batch in enumerate(test):
        if a.batches and i >= a.batches:
            break
        x = batch["x"].to(dev).float()
        if x.dim() == 4:
            x = x.unsqueeze(1)
        out = model(x, local_ctx=batch["local_ctx"].to(dev).float(),
                    global_ctx=batch["global_ctx"].to(dev).float())
        codes = out["codes"]                       # (B, N, 3) ladder codes
        aidx = batch["assay_idx"].long().cpu().numpy()

        # Flatten exactly as MaskGITMotifPrior.flat_ids_from_codes does
        # (model/prior.py:393). Restated rather than imported because building
        # the prior would pull in two context mappers and a checkpoint this
        # analysis does not need; the arithmetic is four lines and its
        # equivalence is asserted in the plan's Task 1B Step 4 check
        # (all 1024 nominal triples map onto exactly 961 distinct ids).
        c0, c1, c2 = (codes[..., 0].long(), codes[..., 1].long(),
                      codes[..., 2].long())
        active = c0.ne(BLANK_CODE)                 # blank excluded here
        nominal = (c0.clamp_min(0) * K2 + c1.clamp_min(0)) * K3 + c2.clamp_min(0)
        fid = mm[nominal.clamp(0, merge_map.numel() - 1)]

        for b in range(fid.shape[0]):
            r = row_of.get(int(aidx[b]))
            if r is None:
                raise SystemExit(
                    f"clip carries assay_idx {int(aidx[b])}, which is not in "
                    f"find_assays() keys {assay_ids}. The index spaces have "
                    f"diverged; every tally below would be attributed wrongly.")
            ids = fid[b][active[b]].cpu().numpy()
            row = np.bincount(ids, minlength=V)
            counts[r] += row
            clip_rows.append((r, row))
        if i % 20 == 0:
            print(f"  batch {i}  clips={len(clip_rows)}", flush=True)

    if counts.sum() <= 0:
        raise SystemExit("no active tokens encoded -- check the checkpoint")

    # Coverage guard. A recording with no sampled clip has an EMPTY code set,
    # which scores Jaccard 0.0 against everything and reads as "shares nothing"
    # when it means "was never looked at". That is not a finding and must not be
    # allowed to reach the JSON.
    covered = counts.sum(axis=1) > 0
    if not covered.all():
        missing = [assay_ids[i] for i in np.nonzero(~covered)[0]]
        raise SystemExit(
            f"{len(missing)} of {n_assays} recordings got no clip: {missing}.\n"
            f"Raise --batches (0 = whole test split). An uncovered recording's "
            f"empty code set would read as zero overlap, which is a sampling "
            f"artifact, not a measurement.")

    # data_provenance.json keys by the SAME sparse assay id as find_assays,
    # so it is joined on that id and then mapped to the dense row.
    prov = json.loads((ROOT / "reports" / "data_provenance.json").read_text())
    prep = {row_of[r["assay_idx"]]:
            ("organoid" if r["dandiset"] == "000732" else "slice")
            for r in prov["used"] if r["assay_idx"] in row_of}
    name = {row_of[r["assay_idx"]]: r["assay_name"]
            for r in prov["used"] if r["assay_idx"] in row_of}
    if len(prep) != n_assays:
        raise SystemExit(
            f"provenance covers {len(prep)} of {n_assays} loaded recordings. "
            f"Regenerate with tools/extract_provenance.py before continuing; "
            f"an unlabelled recording cannot be placed in a preparation block.")

    sets = [set(np.nonzero(counts[i])[0].tolist()) for i in range(n_assays)]
    per_rec = []
    for i in range(n_assays):
        c = counts[i]
        tot_i = int(c.sum())
        top = int(np.sort(c)[::-1][:10].sum())
        per_rec.append({
            "row": i,
            "assay_idx": assay_ids[i],
            "assay_name": name.get(i, f"assay_{i}"),
            "prep": prep.get(i, "unknown"),
            "n_tokens": tot_i,
            "vocab": int((c > 0).sum()),
            "top10_frac": (top / tot_i) if tot_i else 0.0,
        })

    used_by = (counts > 0).sum(axis=0)             # (V,) recordings per entry
    shared_core = {str(k): int((used_by >= k).sum())
                   for k in (1, 2, 5, 10, 16, 24, 31)}

    # Weighted decomposition. H(code | recording) is the recording-weighted
    # mean of each recording's own code entropy.
    tot = counts.sum()
    H_code = _entropy(counts.sum(axis=0))
    w = counts.sum(axis=1) / tot
    H_cond = float(sum(w[i] * _entropy(counts[i]) for i in range(n_assays)))

    J = _jaccard_matrix(sets)
    org = [i for i in range(n_assays) if prep.get(i) == "organoid"]
    sli = [i for i in range(n_assays) if prep.get(i) == "slice"]
    allrec = list(range(n_assays))
    obs_mean = _block_mean(J, allrec, allrec, same=True)

    # ---- rarefaction: the only size-comparable overlap statistic ----------
    # The shared-core count has the same confound as Jaccard. "Used by >= k
    # recordings" rewards entries that the LONG recordings happen to emit, and
    # the long recordings here are all organoid, so the raw count silently
    # weights one preparation type. It is rarefied on the same budget.
    n_sub = int(counts.sum(axis=1).min())
    rng_r = np.random.default_rng(a.seed + 1)
    Jr = np.zeros_like(J)
    voc_r = np.zeros(n_assays)
    core_r = {k: 0.0 for k in (1, 2, 5, 10, 16, 24, 31)}
    for _ in range(a.rarefy_draws):
        ss = _rarefied_sets(counts, n_sub, rng_r)
        Jr += _jaccard_matrix(ss)
        voc_r += np.array([len(x) for x in ss], float)
        ub = np.zeros(V, dtype=np.int32)
        for x in ss:
            if x:
                ub[np.fromiter(x, dtype=np.int64, count=len(x))] += 1
        for k in core_r:
            core_r[k] += float((ub >= k).sum())
    Jr /= a.rarefy_draws
    voc_r /= a.rarefy_draws
    core_r = {str(k): v / a.rarefy_draws for k, v in core_r.items()}

    # And the same null, rarefied the same way, so the two are comparable.
    rng_rn = np.random.default_rng(a.seed + 2)
    labels_r = np.array([r[0] for r in clip_rows])
    rows_r = np.stack([r[1] for r in clip_rows])
    null_r = np.empty(min(a.shuffles, 200), dtype=float)
    for sidx in range(null_r.size):
        perm = rng_rn.permutation(labels_r)
        cc = np.zeros_like(counts)
        np.add.at(cc, perm, rows_r)
        ss = _rarefied_sets(cc, min(n_sub, int(cc.sum(axis=1).min())), rng_rn)
        null_r[sidx] = _block_mean(_jaccard_matrix(ss), allrec, allrec, same=True)

    # Diagnostic that motivates all of the above: how much of the RAW statistic
    # is explained by sampling effort alone.
    mean_raw = (J.sum(axis=1) - 1.0) / (n_assays - 1)
    tok_per = counts.sum(axis=1).astype(float)
    r_size = float(np.corrcoef(np.log(tok_per), mean_raw)[0, 1])

    # Label-shuffle null: reassign CLIPS to recordings at random, preserving
    # each recording's clip count, and recompute the mean pairwise Jaccard.
    # This is the reference the observed value has to be read against.
    rng = np.random.default_rng(a.seed)
    labels = np.array([r[0] for r in clip_rows])
    rows = np.stack([r[1] for r in clip_rows])
    null = np.empty(a.shuffles, dtype=float)
    for s in range(a.shuffles):
        perm = rng.permutation(labels)
        cc = np.zeros_like(counts)
        np.add.at(cc, perm, rows)
        ss = [set(np.nonzero(cc[i])[0].tolist()) for i in range(n_assays)]
        null[s] = _block_mean(_jaccard_matrix(ss), allrec, allrec, same=True)

    res = {
        "codebook_source": str(flat_path.relative_to(ROOT)),
        "ckpt": str(ckpt_path.relative_to(ROOT)),
        "excludes_blank": True,
        "V": V,
        "n_recordings": n_assays,
        "n_clips": len(clip_rows),
        "n_tokens": int(tot),
        "vocab_global": int((counts.sum(axis=0) > 0).sum()),
        "per_recording": per_rec,
        "shared_core": {
            "_warning": "Raw counts use unequal sampling effort (tokens range "
                        "117-1756), so 'used by >= k' favours entries the long "
                        "recordings emit -- and every long recording here is "
                        "organoid. Quote used_by_ge_k_rarefied.",
            "used_by_ge_k": shared_core,
            "used_by_ge_k_rarefied": {k: round(v, 1) for k, v in core_r.items()},
            "rarefied_budget_tokens": n_sub,
        },
        "entropy": {
            "H_code": H_code,
            "H_code_given_recording": H_cond,
            "reuse_ratio": (H_cond / H_code) if H_code > 0 else 0.0,
            "H_code_mm": _entropy_mm(counts.sum(axis=0)),
            "H_code_given_recording_mm": float(
                sum(w[i] * _entropy_mm(counts[i]) for i in range(n_assays))),
            "reuse_ratio_mm": float(
                sum(w[i] * _entropy_mm(counts[i]) for i in range(n_assays))
                / _entropy_mm(counts.sum(axis=0))),
        },
        "jaccard_raw_SIZE_CONFOUNDED": {
            "_warning": "Do not quote. Tracks sampling effort, not vocabulary "
                        "overlap: see corr_log_tokens_vs_mean_jaccard. Use "
                        "jaccard_rarefied instead.",
            "corr_log_tokens_vs_mean_jaccard": r_size,
            "tokens_min": int(tok_per.min()),
            "tokens_max": int(tok_per.max()),
            "observed_mean": obs_mean,
            "mean_within_organoid": _block_mean(J, org, org, same=True),
            "mean_within_slice": _block_mean(J, sli, sli, same=True),
            "mean_cross_prep": _block_mean(J, org, sli, same=False),
            "matrix": J.round(4).tolist(),
        },
        "jaccard_rarefied": {
            "n_tokens_per_recording": n_sub,
            "draws": int(a.rarefy_draws),
            "mean_vocab_at_budget": float(voc_r.mean()),
            "observed_mean": _block_mean(Jr, allrec, allrec, same=True),
            "mean_within_organoid": _block_mean(Jr, org, org, same=True),
            "mean_within_slice": _block_mean(Jr, sli, sli, same=True),
            "mean_cross_prep": _block_mean(Jr, org, sli, same=False),
            "matrix": Jr.round(4).tolist(),
            "null_mean": float(null_r.mean()),
            "null_sd": float(null_r.std()),
            "z": float((_block_mean(Jr, allrec, allrec, same=True)
                        - null_r.mean()) / null_r.std())
                 if null_r.std() > 0 else 0.0,
        },
        "label_shuffle_null": {
            "mean_jaccard": float(null.mean()),
            "sd": float(null.std()),
            "n_shuffles": int(a.shuffles),
            "z": float((obs_mean - null.mean()) / null.std())
                 if null.std() > 0 else 0.0,
        },
    }
    # Persist the raw tally. Every statistic above is derived from it, so a
    # follow-up question does not need another encode pass over the test split.
    np.savez_compressed(Path(a.out).with_suffix(".npz"),
                        counts=counts, assay_ids=np.array(assay_ids))
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(f"\nwrote {a.out}")
    print(f"  clips / tokens           {res['n_clips']} / {res['n_tokens']:,}")
    print(f"  global vocabulary        {res['vocab_global']} of {V}")
    print(f"  shared core, raw      >=16 {shared_core['16']}   "
          f"=31 {shared_core['31']}  (effort-confounded)")
    print(f"  shared core, rarefied >=16 {core_r['16']:.1f}   "
          f">=5 {core_r['5']:.1f}   >=2 {core_r['2']:.1f}")
    print(f"  H(code)                  {H_code:.4f} nats")
    print(f"  H(code | recording)      {H_cond:.4f} nats")
    print(f"  reuse ratio              {res['entropy']['reuse_ratio']:.4f}")
    print(f"  reuse ratio (Miller-Madow) {res['entropy']['reuse_ratio_mm']:.4f}")
    print(f"\n  RAW Jaccard is size-confounded and is not quotable:")
    print(f"    corr(log tokens, mean J) {r_size:+.3f}   "
          f"tokens {int(tok_per.min())}-{int(tok_per.max())}")
    print(f"    observed {obs_mean:.4f}  shuffled {null.mean():.4f}"
          f" +/- {null.std():.4f}")
    jr = res["jaccard_rarefied"]
    print(f"\n  RAREFIED to {n_sub} tokens per recording, "
          f"{a.rarefy_draws} draws  (quote this one)")
    print(f"    mean vocabulary at budget  {jr['mean_vocab_at_budget']:.1f}")
    print(f"    observed {jr['observed_mean']:.4f}   "
          f"shuffled {jr['null_mean']:.4f} +/- {jr['null_sd']:.4f}"
          f"   z = {jr['z']:+.2f}")
    print(f"    within-org / within-slice / cross  "
          f"{jr['mean_within_organoid']:.4f} / "
          f"{jr['mean_within_slice']:.4f} / {jr['mean_cross_prep']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
