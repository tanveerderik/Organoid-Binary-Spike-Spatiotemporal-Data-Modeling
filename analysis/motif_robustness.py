"""How much does Stage 4A degrade when it is fed 4B's activity map instead of the truth?

This is the question Stage 4C existed to answer, and dropping 4C only makes
sense if the degradation is small. Three arms, paired on the same clips and the
same deterministic masks, graded on the SAME token set (roi & gt_active) so the
arms are comparable:

  oracle    activity = ground truth. The ceiling: 4A as it was trained and
            evaluated in isolation.
  model     activity = 4B's sampled hard map inside the ROI, ground truth
            outside. Exactly what inference does, and what 4C saw at its ep1.
  marginal  activity = a random map with the SAME per-clip cell count as the
            model arm, placed uniformly in the ROI. The null: it isolates how
            much of 4A's performance needs the activity map to be in the RIGHT
            PLACE, as opposed to merely having the right density.

Note the activity map does two things at once, and both are reproduced here:
it conditions the motif prior (activity_prob) AND it selects which tokens are
masked for prediction (_build_inference_aligned_motif_inputs uses hard_roi).
A wrong map therefore asks 4A to fill the wrong cells, not just with worse
conditioning.
"""
import sys, json, argparse
sys.path.insert(0, "/media/derik/Seagate Desktop Drive/organoid_data")
import torch, numpy as np
from MAGVIT_project import main as M
from MAGVIT_project.training.stage4_activity import (
    _batch_to_device, _vq_codes_and_pmask, _make_activity_in_from_codes,
    _build_inference_aligned_motif_inputs, deterministic_validation_task_and_masks)

ap = argparse.ArgumentParser()
ap.add_argument("--batches", type=int, default=24)
ap.add_argument("--split", default="val", choices=("val", "test"))
ap.add_argument("--seed", type=int, default=20260821)
ap.add_argument("--protocol", default="inpaint", choices=("inpaint", "free"),
                help="inpaint = deterministic val masks, 4A still sees visible "
                     "codes. free = recon spec, FULL ROI, no visible codes at "
                     "all -- the actual four-regime generation setting and the "
                     "genuinely out-of-distribution one for a motif prior that "
                     "trained on teacher-forced activity.")
ap.add_argument("--motif-ckpt", default=None,
                help="optional state_dict to load into the motif prior (e.g. the "
                     "Stage 4D adapted checkpoint). Default = whatever "
                     "_load_stage4_eval_prior loads, i.e. ckpts/motif_prior_best.pt.")
ap.add_argument("--out", default="reports/motif_robustness.json")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
assay_dict = M.find_assays()
train_loader, val_loader, test_loader, meta = M.make_loaders(
    assay_dict=assay_dict, assay_indices=list(assay_dict.keys()),
    per_assay_quota=M.per_assay_quota_stage12)
loader = {"val": val_loader, "test": test_loader}[args.split]
b0 = next(iter(train_loader)); _, _, T0, H0, W0 = b0["x"].shape
vqvae = M.make_vqvae((T0, H0, W0), dev, full_spatial_size=tuple(map(int, b0["full_hw"][0])))
M.load_stage1_gct_for_eval(vqvae)
blank_code = getattr(vqvae.vq, "blank_code", -1)
# Pinned to the Stage 4A control unless --motif-ckpt says otherwise, so the
# recorded baselines (model 0.15698 free / 0.16366 inpaint, oracle 0.23257 /
# 0.23582) stay reproducible even though the eval loader now prefers 4D.
prior = M._load_stage4_eval_prior(vqvae, dev, phase="4b", load_motif=True,
                                  prefer_adapted_motif=False)
ap_, mp_ = prior.activity_prior, prior.motif_prior
if args.motif_ckpt:
    _sd = torch.load(args.motif_ckpt, map_location=dev)
    for _k in ("model", "state_dict", "motif_prior"):
        if isinstance(_sd, dict) and _k in _sd and isinstance(_sd[_k], dict):
            _sd = _sd[_k]; break
    _missing, _unexpected = mp_.load_state_dict(_sd, strict=False)
    if _missing or _unexpected:
        raise SystemExit(f"motif ckpt mismatch: missing={_missing} unexpected={_unexpected}")
    print(f"loaded motif weights from {args.motif_ckpt} (strict)")
ap_.eval(); mp_.eval()

g = torch.Generator(device="cpu").manual_seed(args.seed)
acc = {k: {"rr": [], "rank": []} for k in ("oracle", "model", "marginal", "model_soft")}
agree = []

@torch.no_grad()
def score(arm, activity_full, motif_targets, hard_roi, gct, lct, tid, pmask):
    mi = _build_inference_aligned_motif_inputs(mp_, motif_targets, hard_roi)
    logits = mp_.forward_with_activity_prob(
        f_in=mi["f_in"], activity_prob=activity_full,
        global_ctx=gct, local_ctx=lct, task_id=tid, roi_mask=pmask)
    fl = logits["flat"].float()
    tgt = motif_targets["f"].long()
    msk = mi["targets"]["f_loss_mask"].bool()
    if not bool(msk.any()):
        return
    tl = fl.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    rank = (fl > tl.unsqueeze(-1)).sum(-1) + 1
    acc[arm]["rr"].append((1.0 / rank.float())[msk].cpu())
    acc[arm]["rank"].append(rank[msk].cpu())

with torch.no_grad():
    for bi, batch in enumerate(loader):
        if bi >= args.batches:
            break
        x, gct, lct, tid, ms = _batch_to_device(batch, dev)
        if args.protocol == "free":
            # recon maps to all-ones in predict_mask_from_spec (model/vqvae.py:670)
            ms = [{"type": "recon"}] * x.shape[0]
            tid = torch.zeros_like(tid)
        else:
            tid, ms = deterministic_validation_task_and_masks(batch, x, dev)
        codes, pmask, _ = _vq_codes_and_pmask(vqvae, x, gct, lct, ms, dev)
        if args.protocol == "free":
            frac = float((pmask > 0.5).float().mean())
            if frac < 0.999:
                raise SystemExit(f"free protocol expected a full ROI, got {frac:.3f}")
        mt = mp_.make_targets_from_codes(codes=codes, predict_mask=pmask,
                                         blank_code=blank_code)
        a_in = _make_activity_in_from_codes(codes, pmask, blank_code=blank_code,
                                            a_mask_id=ap_.a_mask_id)
        out = ap_(global_ctx=gct, local_ctx=lct, task_id=tid, a_in=a_in,
                  roi_mask=pmask, count_target=None, count_teacher_prob=0.0)
        roi = (pmask.squeeze(-1) if pmask.dim() == 3 else pmask).bool()
        gt = mt["active"].to(dev).float()

        hard_model = ap_.sample_hard_activity_gridtopk(
            out, count_mode="expected", count_temperature=1.0,
            count_stochastic_round=False, roi_mask=pmask).float() * roi.float()

        # Marginal arm: same count per clip, uniformly placed inside the ROI.
        hard_marg = torch.zeros_like(hard_model)
        for i in range(hard_model.shape[0]):
            k = int(hard_model[i].sum().item())
            idx = torch.nonzero(roi[i], as_tuple=False).squeeze(-1)
            if k > 0 and idx.numel() > 0:
                pick = idx[torch.randperm(idx.numel(), generator=g)[:min(k, idx.numel())]]
                hard_marg[i, pick] = 1.0

        hard_oracle = (gt * roi.float())
        agree.append(float(((hard_model.bool() & gt.bool()) & roi).sum().item()) /
                     max(1.0, float((gt.bool() & roi).sum().item())))

        for arm, hard in (("oracle", hard_oracle), ("model", hard_model),
                          ("marginal", hard_marg)):
            full = torch.where(roi, hard, gt)
            score(arm, full, mt, hard, gct, lct, tid, pmask)

        # model_soft: same STRUCTURAL map as "model" (so the same cells are
        # masked and scored -- paired, identical token set), but the activity
        # stream carries 4B's calibrated probability instead of a hard 0/1.
        # This is what Stage 4D trained on; the hard "model" arm is what
        # inference/sample_prior.py currently feeds (it clamps to long).
        soft = torch.sigmoid(out["cell_logits"].float())
        if soft.dim() == 3:
            soft = soft.squeeze(-1)
        score("model_soft", torch.where(roi, soft, gt), mt, hard_model,
              gct, lct, tid, pmask)

print(f"\nbatches={args.batches} split={args.split} protocol={args.protocol}   "
      f"4B activity recall inside ROI = {np.mean(agree):.4f}")
print(f"\n{'arm':<12}{'MRR':>10}{'medRank':>10}{'meanRank':>11}{'top1':>9}{'top10':>9}{'n_tok':>10}")
res = {}
for arm in ("oracle", "model", "model_soft", "marginal"):
    if not acc[arm]["rr"]:
        continue
    rr = torch.cat(acc[arm]["rr"]); rk = torch.cat(acc[arm]["rank"]).float()
    res[arm] = {"mrr": float(rr.mean()), "median_rank": float(rk.median()),
                "mean_rank": float(rk.mean()), "top1": float((rk <= 1).float().mean()),
                "top10": float((rk <= 10).float().mean()), "n": int(rk.numel())}
    r = res[arm]
    print(f"{arm:<12}{r['mrr']:>10.5f}{r['median_rank']:>10.1f}{r['mean_rank']:>11.1f}"
          f"{r['top1']:>9.4f}{r['top10']:>9.4f}{r['n']:>10d}")

if {"oracle", "model", "marginal"} <= set(res):
    o, m, g_ = res["oracle"], res["model"], res["marginal"]
    span = o["mrr"] - g_["mrr"]
    print(f"\noracle - model  MRR {o['mrr']-m['mrr']:+.5f}   "
          f"model - marginal {m['mrr']-g_['mrr']:+.5f}")
    if span > 1e-9:
        print(f"model recovers {(m['mrr']-g_['mrr'])/span*100:.1f}% of the "
              f"marginal->oracle span in MRR")
    print(f"median rank  oracle {o['median_rank']:.0f}  model {m['median_rank']:.0f}"
          f"  marginal {g_['median_rank']:.0f}")

json.dump({"args": vars(args), "activity_recall": float(np.mean(agree)),
           "arms": res}, open(args.out, "w"), indent=2)
print(f"\nwrote {args.out}")
