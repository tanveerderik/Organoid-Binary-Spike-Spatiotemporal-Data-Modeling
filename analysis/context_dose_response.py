#!/usr/bin/env python3
"""Does the local descriptor actually STEER generation? A dose-response test.

Reviewers asked how much of the reported performance comes from the descriptor
`lct`, which is computed from the whole clip. The complementary question is the
one that makes full-clip conditioning a legitimate *controlled generation*
setting: if you ask the model for a different density, temporal spread or trend,
does the generated volume follow?

For each held-out clip the true gct and lct are kept, ONE descriptor feature is
moved along a grid, and the shipped pipeline generates freely (task 0, every
token masked -- the same path as analysis/generate_regimes.py). The realised
descriptor of each generated volume is measured with the same function the
dataset uses (utils.recon.compute_activity_ctx). Reported per feature:

  slope        realised change / requested change, paired within clip
               (1 = the model follows exactly, 0 = it ignores the request),
               with a 95% CI from a bootstrap over recordings
  r            correlation of requested vs realised across the grid
  off_target   how much the OTHER features move when only this one is changed

Grid: log density is shifted by log(k) for k in {0.5, 0.71, 1, 1.41, 2}; the
other features by {-1, -0.5, 0, 0.5, 1} x their across-clip SD on the batch.

    CTX_SOURCE=full python analysis/context_dose_response.py --batches 24

Writes reports/context_dose_response.json. Needs the GPU machine (checkpoints
and data).
"""
import sys, argparse, json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/media/derik/Seagate Desktop Drive/organoid_data")
from MAGVIT_project import main as M                                           # noqa: E402
from MAGVIT_project.inference.sample_prior import iterative_unmask_motif_given_activity  # noqa: E402
from MAGVIT_project.inference.decode import decode_flat_ids_to_xgen           # noqa: E402
from MAGVIT_project.training.train_prior import _vq_codes_and_pmask_for_prior  # noqa: E402
from MAGVIT_project.training.stage4_activity import (                          # noqa: E402
    _batch_to_device, _make_activity_in_from_codes)
from MAGVIT_project.utils.constants import ACTIVITY_CTX_NAMES, ACTIVITY_CTX_INDEX  # noqa: E402
from MAGVIT_project.utils.recon import compute_activity_ctx_batch              # noqa: E402

FEATURES = ("log_mean_firing_density", "var_t", "temporal_trend_score",
            "active_site_ratio", "var_x")
DENSITY_FACTORS = (0.5, 0.71, 1.0, 1.41, 2.0)
SD_STEPS = (-1.0, -0.5, 0.0, 0.5, 1.0)

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--phase", default="4b")
ap.add_argument("--split", default="test", choices=("val", "test"))
ap.add_argument("--batches", type=int, default=24)
ap.add_argument("--features", default=",".join(FEATURES))
ap.add_argument("--motif-steps", type=int, default=12)
ap.add_argument("--motif-top-k", type=int, default=5)
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=20261107)
ap.add_argument("--out", default="reports/context_dose_response.json")
args = ap.parse_args()
if M.CTX_SOURCE != "full":
    raise SystemExit("dose-response is defined for the full-clip descriptor "
                     "(CTX_SOURCE=full): it asks whether a REQUESTED statistic "
                     "is realised")

feats = [f.strip() for f in args.features.split(",") if f.strip()]
for f in feats:
    if f not in ACTIVITY_CTX_INDEX:
        raise SystemExit(f"unknown feature {f!r}; choose from {ACTIVITY_CTX_NAMES}")

dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)
torch.manual_seed(args.seed)

assay_dict = M.find_assays()
train_loader, val_loader, test_loader, meta = M.make_loaders(
    assay_dict=assay_dict, assay_indices=list(assay_dict.keys()),
    per_assay_quota=M.per_assay_quota_stage12)
loader = {"val": val_loader, "test": test_loader}[args.split]
b0 = next(iter(train_loader)); _, _, T0, H0, W0 = b0["x"].shape
vqvae = M.make_vqvae((T0, H0, W0), dev, full_spatial_size=tuple(map(int, b0["full_hw"][0])))
M.load_stage1_gct_for_eval(vqvae)
vqvae.eval()
prior = M._load_stage4_eval_prior(vqvae, dev, phase=args.phase, load_motif=True)
activity_prior, motif_prior = prior.activity_prior, prior.motif_prior
blank_code = getattr(vqvae.vq, "blank_code", -1)


@torch.no_grad()
def generate(gct, lct, task_id, x_ref, batch):
    """Free generation; identical to analysis/generate_regimes.generate with the
    shipped defaults (soft activity field, gumbel readout, 12 motif steps)."""
    B = x_ref.shape[0]
    codes, pmask, grid = _vq_codes_and_pmask_for_prior(
        vqvae, x_ref, gct, lct, [{"type": "recon"}] * B, device)
    token_grid = tuple(map(int, grid)) if grid is not None else (
        activity_prior.Ttok, activity_prior.Htok, activity_prior.Wtok)
    roi = torch.ones(codes.shape[0], codes.shape[1], device=device, dtype=torch.bool)
    predict_mask = roi.unsqueeze(-1) if pmask.dim() == 3 else roi
    a_in = _make_activity_in_from_codes(
        codes, predict_mask, blank_code=blank_code, a_mask_id=activity_prior.a_mask_id)
    a_out = activity_prior(global_ctx=gct, local_ctx=lct, task_id=task_id,
                           a_in=a_in, roi_mask=predict_mask,
                           count_target=None, count_teacher_prob=0.0)
    hard = activity_prior.sample_hard_activity_for_generation(
        a_out, roi_mask=predict_mask, readout="gumbel", tau=1.0,
        count_mode="expected", count_scale=1.0).long()
    a_prob = torch.sigmoid(a_out["cell_logits"].float())
    if a_prob.dim() == 3 and a_prob.size(-1) == 1:
        a_prob = a_prob.squeeze(-1)
    motif = iterative_unmask_motif_given_activity(
        motif_prior, activity=hard, global_ctx=gct, local_ctx=lct, task_id=task_id,
        roi_mask=roi, visible_codes=codes, steps=args.motif_steps,
        temperature=1.0, top_k=args.motif_top_k, activity_prob=a_prob)
    decoded = decode_flat_ids_to_xgen(
        vqvae, motif["flat_ids"], flat_codebook=motif_prior.flat_codebook,
        grid=token_grid, global_ctx=gct, local_ctx=lct,
        roi_hw=batch.get("roi_hw", None), pad_hw=batch.get("pad_hw", None))
    return decoded["x_gen"]


rows = []                      # one row per (clip, feature, level)
for bi, batch in enumerate(loader):
    if bi >= args.batches:
        break
    x, gct, lct_true, task_true, _ = _batch_to_device(batch, device)
    lct_true = compute_activity_ctx_batch(x, None, "full").to(device)
    task_id = torch.zeros_like(task_true)
    assays = [int(a) for a in batch["assay_idx"]]
    sd = lct_true.std(0).clamp_min(1e-6)
    for f in feats:
        j = ACTIVITY_CTX_INDEX[f]
        levels = ([float(np.log(k)) for k in DENSITY_FACTORS]
                  if f == "log_mean_firing_density"
                  else [s * float(sd[j]) for s in SD_STEPS])
        for lev in levels:
            lct = lct_true.clone()
            lct[:, j] = lct[:, j] + lev
            x_gen = generate(gct, lct, task_id, x, batch)
            got = compute_activity_ctx_batch((x_gen > 0.5).float(), None, "full").cpu()
            for i in range(x.shape[0]):
                rows.append({"batch": bi, "clip": i, "assay": assays[i],
                             "feature": f, "requested_delta": lev,
                             "requested": [float(v) for v in lct[i].cpu()],
                             "true": [float(v) for v in lct_true[i].cpu()],
                             "realised": [float(v) for v in got[i]]})
    print(f"batch {bi}: {len(rows)} generations", flush=True)


def summarise(feature):
    j = ACTIVITY_CTX_INDEX[feature]
    R = [r for r in rows if r["feature"] == feature]
    by_clip = {}
    for r in R:
        by_clip.setdefault((r["batch"], r["clip"]), []).append(r)
    dx, dy, lab, off = [], [], [], []
    for (_, _), rs in by_clip.items():
        base = [r for r in rs if abs(r["requested_delta"]) < 1e-12]
        if not base:
            continue
        b = np.asarray(base[0]["realised"])
        for r in rs:
            dx.append(r["requested_delta"])
            dy.append(r["realised"][j] - b[j])
            lab.append(r["assay"])
            others = np.delete(np.asarray(r["realised"]) - b, j)
            off.append(float(np.abs(others).mean()))
    dx, dy, lab = np.asarray(dx), np.asarray(dy), np.asarray(lab)
    slope = float((dx * dy).sum() / max((dx * dx).sum(), 1e-12))
    rng = np.random.default_rng(args.seed)
    ids = np.unique(lab)
    boots = []
    for _ in range(args.n_boot):
        pick = rng.choice(ids, size=ids.size, replace=True)
        m = np.concatenate([np.where(lab == a)[0] for a in pick])
        boots.append((dx[m] * dy[m]).sum() / max((dx[m] ** 2).sum(), 1e-12))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"n_generations": len(R), "n_clips": len(by_clip),
            "slope": slope, "slope_ci95": [float(lo), float(hi)],
            "r": float(np.corrcoef(dx, dy)[0, 1]) if dx.std() > 0 else None,
            "mean_abs_off_target_change": float(np.mean(off)) if off else None}


res = {"ctx_source": M.CTX_SOURCE, "phase": args.phase, "split": args.split,
       "batches": args.batches, "seed": args.seed,
       "grid": {"log_mean_firing_density": [float(np.log(k)) for k in DENSITY_FACTORS],
                "other_features_sd_steps": list(SD_STEPS)},
       "summary": {f: summarise(f) for f in feats},
       "rows": rows}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
print(f"\n{'feature':<26}{'slope':>8}{'95% CI':>20}{'r':>8}{'off-target':>12}")
for f, s in res["summary"].items():
    print(f"{f:<26}{s['slope']:>8.3f}{str([round(v, 3) for v in s['slope_ci95']]):>20}"
          f"{(s['r'] or float('nan')):>8.3f}{(s['mean_abs_off_target_change'] or 0):>12.4f}")
print(f"wrote {args.out}")
