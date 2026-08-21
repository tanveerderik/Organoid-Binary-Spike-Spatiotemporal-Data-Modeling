#!/usr/bin/env python3
"""Stage 4 sample generation across the four conditioning regimes.

FREE generation: every token is masked, so nothing about the target clip leaks
through visible tokens. The regimes differ ONLY in which context vectors the
model is given, which is the axis the paper compares along:

    random                 gct and lct both drawn unconditionally from the
                           training context bank -- a plausible pair, but not
                           this clip's. Nothing about the held-out clip is used.
    global_only            gct pinned to the held-out clip; lct RETRIEVED from
                           the bank conditioned on that gct. "I know which
                           preparation, not what it did."
    global_partial_local   gct pinned; a named SUBSET of lct pinned to the clip;
                           the remaining lct features retrieved. "I know the
                           preparation and roughly how active, not the
                           spatiotemporal shape."
    global_full_local      gct and lct both pinned to the held-out clip. Full
                           context, still no visible tokens.

This is a monotone ladder of conditioning, so the statistics should approach the
real reference from `random` to `global_full_local`. That ordering is the claim;
the tree written here is what tests it.

Everything runs through the SAME code path the scored pipeline uses --
activity prior -> iterative_unmask_motif_given_activity -> decode_flat_ids_to_xgen
with the gumbel readout -- so these samples are the pipeline being reported on,
not a re-derivation of it.

The context bank must be the one built from TRAIN clips under the current split
(see $JOB/tmp/rebuild_context_bank.py): retrieving conditioning from a bank that
contains val/test clips would leak the very clips the samples are compared to.

Output tree (inference.generation_output.GenerationWriter)::

    <out>/manifest.json          one row per sample, all regimes -- load this
    <out>/summary.json           per-regime aggregates + vs-real comparison
    <out>/real_reference.json    statistics of the real held-out clips
    <out>/<regime>/videos/<assay>__s0000.mp4
    <out>/<regime>/stats/<assay>__s0000.json
"""
import sys, argparse, json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/media/derik/Seagate Desktop Drive/organoid_data")
from MAGVIT_project import main as M
from MAGVIT_project.inference.generation_output import GenerationWriter, summary_table
from MAGVIT_project.inference.sample_context import ContextBankSampler
from MAGVIT_project.inference.sample_prior import iterative_unmask_motif_given_activity
from MAGVIT_project.inference.decode import decode_flat_ids_to_xgen
from MAGVIT_project.training.train_prior import _vq_codes_and_pmask_for_prior
from MAGVIT_project.training.stage4_activity import (
    _batch_to_device, _make_activity_in_from_codes)
from MAGVIT_project.utils.constants import ACTIVITY_CTX_NAMES, ACTIVITY_CTX_INDEX

# Which lct features "partial local" pins. These two are the ones an
# experimenter could actually state up front -- how active the culture is and on
# what fraction of sites -- while the variances, covariances and trend describe
# the spatiotemporal shape we want the model to invent.
PARTIAL_LOCAL_DEFAULT = ("log_mean_firing_density", "active_site_ratio")

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--phase", default="4b", choices=("4b", "4b_refine"),
                help="which activity prior to sample from. Default 4b: Stage 4B-refine "
                     "was rejected (all epoch variation inside 0.92 seed sd, "
                     "motif-MRR declining t=-10.45) and 4c would silently load "
                     "activity_prior_refined_best.pt instead of the shipped 4B.")
ap.add_argument("--split", default="test", choices=("val", "test"))
ap.add_argument("--batches", type=int, default=16,
                help="held-out batches; each contributes one sample per regime per clip")
ap.add_argument("--samples-per-clip", type=int, default=1,
                help="independent draws per clip per regime")
ap.add_argument("--motif-steps", type=int, default=12)
ap.add_argument("--motif-temperature", type=float, default=1.0)
ap.add_argument("--motif-top-k", type=int, default=5)
ap.add_argument("--activity-readout", default="gumbel", choices=("gumbel", "topk"))
ap.add_argument("--activity-readout-tau", type=float, default=1.0)
ap.add_argument("--activity-count-scale", type=float, default=1.0)
ap.add_argument("--activity-field", default="soft", choices=("hard", "soft"),
                help="What the motif prior is told about each activity cell. "
                     "'soft' (default) = sigmoid(cell_logits) inside the ROI. "
                     "'hard' = 0/1 from the readout, the pre-2026-08-21 "
                     "behaviour, kept so the earlier sample sets reproduce. "
                     "The hard "
                     "readout still decides WHICH cells are active either way; "
                     "this only changes how confident the motif prior is told "
                     "the activity prior was.")
ap.add_argument("--task-id", type=int, default=0,
                help="dataset task_id_map: recon 0, causal 1, noncausal 2, spatial 3. "
                     "Free generation is RECON: predict_mask_from_spec maps a recon "
                     "spec to all-ones (model/vqvae.py:670), so recon is the task the "
                     "model was actually trained on with a full ROI.")
ap.add_argument("--partial-local", default=",".join(PARTIAL_LOCAL_DEFAULT),
                help="comma-separated lct feature names pinned in the partial regime")
ap.add_argument("--bank", default="ckpts/context_prior.pkl")
ap.add_argument("--activity-ckpt", default=None)
ap.add_argument("--motif-ckpt", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--seed", type=int, default=20260821)
ap.add_argument("--no-video", action="store_true", help="statistics only, skip mp4s")
ap.add_argument("--regimes", default="random,global_only,global_partial_local,global_full_local")
args = ap.parse_args()

REGIMES = [r.strip() for r in args.regimes.split(",") if r.strip()]
partial_names = [n.strip() for n in args.partial_local.split(",") if n.strip()]
for n in partial_names:
    if n not in ACTIVITY_CTX_INDEX:
        raise SystemExit(f"unknown lct feature {n!r}; choose from {ACTIVITY_CTX_NAMES}")
partial_idx = [ACTIVITY_CTX_INDEX[n] for n in partial_names]
out_root = Path(args.out or f"reports/generation_regimes_{args.phase}")

if args.activity_ckpt:
    for _k in ("activity_prior_best", "activity_prior_best_loss",
               "activity_prior_best_hard_metric", "activity_prior_refined_best"):
        if _k in M.CKPTS:
            M.CKPTS[_k] = M.Path(args.activity_ckpt)
if args.motif_ckpt:
    # Set both keys: "motif_prior_ship" is what the eval loader prefers, and
    # "motif_prior_best" is the fallback, so an explicit --motif-ckpt wins
    # either way.
    M.CKPTS["motif_prior_best"] = M.Path(args.motif_ckpt)
    M.CKPTS["motif_prior_ship"] = M.Path(args.motif_ckpt)

dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)
torch.manual_seed(args.seed)
if dev == "cuda":
    torch.cuda.manual_seed_all(args.seed)

assay_dict = M.find_assays()
train_loader, val_loader, test_loader, meta = M.make_loaders(
    assay_dict=assay_dict, assay_indices=list(assay_dict.keys()),
    per_assay_quota=M.per_assay_quota_stage12)
loader = {"val": val_loader, "test": test_loader}[args.split]

b0 = next(iter(train_loader)); _, _, T0, H0, W0 = b0["x"].shape
vqvae = M.make_vqvae((T0, H0, W0), dev, full_spatial_size=tuple(map(int, b0["full_hw"][0])))
# Without the Stage-1 memory bank the decoded spatial-support term silently
# takes a fallback branch; the same trap as in generation_seed_spread.py.
M.load_stage1_gct_for_eval(vqvae)
vqvae.eval()
prior = M._load_stage4_eval_prior(vqvae, dev, phase=args.phase, load_motif=True)
activity_prior, motif_prior = prior.activity_prior, prior.motif_prior
_ship = M.CKPTS.get("motif_prior_ship")
_MOTIF_CKPT_USED = _ship if (_ship is not None and _ship.exists()) \
    else M.CKPTS.get("motif_prior_best")
blank_code = getattr(vqvae.vq, "blank_code", -1)

bank = ContextBankSampler.from_file(args.bank, model=vqvae, device=device, seed=args.seed)
print(f"context bank: {args.bank}  {bank.N} entries, "
      f"{len(set(bank.assay_ids.tolist()))} assays", flush=True)
print(f"phase {args.phase}  split {args.split}  regimes {REGIMES}")
print(f"partial-local pins: {partial_names}")
print(f"writing -> {out_root}", flush=True)

writer = GenerationWriter(out_root, fps=30)


def contexts_for(regime, gct_true, lct_true, assay_ids):
    """(gct, lct) for one batch under one regime, as float tensors on device."""
    B = gct_true.shape[0]
    g_np = gct_true.detach().cpu().numpy().astype(np.float32)
    l_np = lct_true.detach().cpu().numpy().astype(np.float32)

    if regime == "global_full_local":
        return gct_true, lct_true, {"global": True, "local": True}

    g_out = np.empty_like(g_np)
    l_out = np.empty_like(l_np)
    for i in range(B):
        if regime == "random":
            # Unconditional: both halves come from the bank, so the sample is
            # not tied to this clip at all.
            s = bank.sample_unconditional(batch_size=1)
            g_out[i], l_out[i] = s["global_ctx"][0], s["local_ctx"][0]
        elif regime == "global_only":
            s = bank.sample_given_global(g_np[i], batch_size=1)
            g_out[i], l_out[i] = g_np[i], s["local_ctx"][0]
        elif regime == "global_partial_local":
            pin = {int(j): float(l_np[i, j]) for j in partial_idx}
            s = bank.sample_given_partial_local(
                global_ctx=g_np[i], partial_local=pin, batch_size=1)
            l_ret = np.array(s["local_ctx"][0], dtype=np.float32)
            # The retrieved neighbour is a real clip, so its pinned features are
            # only close to the requested ones. Overwrite them with the exact
            # values, otherwise "pinned" is a suggestion rather than a condition.
            for j in partial_idx:
                l_ret[j] = l_np[i, j]
            g_out[i], l_out[i] = g_np[i], l_ret
        else:
            raise SystemExit(f"unknown regime {regime!r}")

    ctx_flags = {
        "random": {"global": False, "local": False},
        "global_only": {"global": True, "local": False},
        "global_partial_local": {"global": True, "local": "partial"},
    }[regime]
    return (torch.from_numpy(g_out).to(device),
            torch.from_numpy(l_out).to(device),
            ctx_flags)


@torch.no_grad()
def generate(gct, lct, task_id, x_ref, batch):
    """Free generation: every token masked, so visible codes are never read.

    The ROI comes from a RECON mask spec rather than the batch's own spec.
    predict_mask_from_spec maps "recon" to all-ones (model/vqvae.py:670), which
    is exactly the full ROI free generation needs -- and it is a task the model
    was trained on, so the task embedding and the ROI agree. Using the batch's
    spatial/noncausal spec with task_id=recon (or the reverse) would hand the
    model a combination it never saw.

    The recon spec also triggers build_latent_hidden_mask's Bernoulli
    substitution (recon_drop_p=0.5, model/vqvae.py:616), but that only feeds
    the VQ-VAE decoder's own reconstruction. out["predict_mask"] is taken
    straight from predict_mask_from_spec, and the codes come from the
    encoder+quantizer, so neither is affected -- and the codes are discarded
    anyway once the ROI is all-True.
    """
    B = x_ref.shape[0]
    recon_spec = [{"type": "recon"}] * B
    codes, pmask, grid = _vq_codes_and_pmask_for_prior(
        vqvae, x_ref, gct, lct, recon_spec, device)
    token_grid = tuple(map(int, grid)) if grid is not None else (
        activity_prior.Ttok, activity_prior.Htok, activity_prior.Wtok)
    if not bool((pmask > 0.5).all()):
        raise RuntimeError(
            "recon spec did not yield a full ROI: "
            f"{float((pmask > 0.5).float().mean()):.3f} of tokens supervised. "
            "Free generation requires every token masked.")
    # iterative_unmask_motif_given_activity computes visible_active as
    # vis_active & ~roi, so an all-True roi discards the encoded codes entirely.
    roi = torch.ones(codes.shape[0], codes.shape[1], device=device, dtype=torch.bool)
    predict_mask = roi.unsqueeze(-1) if pmask.dim() == 3 else roi

    a_in = _make_activity_in_from_codes(
        codes, predict_mask, blank_code=blank_code, a_mask_id=activity_prior.a_mask_id)
    a_out = activity_prior(global_ctx=gct, local_ctx=lct, task_id=task_id,
                           a_in=a_in, roi_mask=predict_mask,
                           count_target=None, count_teacher_prob=0.0)
    hard = activity_prior.sample_hard_activity_for_generation(
        a_out, roi_mask=predict_mask, readout=args.activity_readout,
        tau=args.activity_readout_tau, count_mode="expected",
        count_scale=args.activity_count_scale).long()

    if args.activity_field == "soft":
        a_prob = torch.sigmoid(a_out["cell_logits"].float())
        if a_prob.dim() == 3 and a_prob.size(-1) == 1:
            a_prob = a_prob.squeeze(-1)
    else:
        a_prob = None

    motif = iterative_unmask_motif_given_activity(
        motif_prior, activity=hard, global_ctx=gct, local_ctx=lct, task_id=task_id,
        roi_mask=roi, visible_codes=codes, steps=args.motif_steps,
        temperature=args.motif_temperature, top_k=args.motif_top_k,
        activity_prob=a_prob)
    decoded = decode_flat_ids_to_xgen(
        vqvae, motif["flat_ids"], flat_codebook=motif_prior.flat_codebook,
        grid=token_grid, global_ctx=gct, local_ctx=lct,
        roi_hw=batch.get("roi_hw", None), pad_hw=batch.get("pad_hw", None))
    return decoded["x_gen"], hard


n_done = 0
for bi, batch in enumerate(loader):
    if bi >= args.batches:
        break
    x, gct_true, lct_true, task_true, ms = _batch_to_device(batch, device)
    B = x.shape[0]
    task_id = torch.full_like(task_true, int(args.task_id))
    assay_names = list(batch.get("assay_name", [f"assay{int(a)}" for a in batch["assay_idx"]]))
    assay_ids = [int(a) for a in batch["assay_idx"]]

    # Real reference, once per clip -- not per regime, or it would be counted
    # four times and the reference statistics would silently reweight.
    writer.add_real(x[:, 0].detach().float().cpu())

    for regime in REGIMES:
        for rep in range(args.samples_per_clip):
            gct, lct, flags = contexts_for(regime, gct_true, lct_true, assay_ids)
            x_gen, hard = generate(gct, lct, task_id, x, batch)
            extra = [{
                "rep": rep,
                "activity_cells": int(hard[i].sum().item()),
                "generated_spikes": float(x_gen[i, 0].sum().item()),
                "real_spikes": float(x[i, 0].sum().item()),
                # The conditioning actually handed to the model, so a row can be
                # reproduced from the manifest alone.
                "gct_l2_vs_true": float(torch.linalg.norm(gct[i] - gct_true[i]).item()),
                "lct_used": [float(v) for v in lct[i].detach().cpu()],
                "lct_true": [float(v) for v in lct_true[i].detach().cpu()],
            } for i in range(B)]
            writer.add_batch(
                regime, x_gen[:, 0].detach().float().cpu(),
                assay_names=assay_names,
                task_ids=[int(args.task_id)] * B,
                context={"global": bool(flags["global"]),
                         "local": bool(flags["local"] is True),
                         "visible": False},
                # The clip this sample is ALIGNED WITH, for every regime.
                #
                # Previously None, with the comment "free generation: no
                # per-sample target". That was wrong for the conditioned rungs:
                # in global_full_local the gct/lct are pinned to exactly this
                # clip, so this clip IS the target, and in random the same clip
                # is the control -- the sample is generated from unrelated
                # context, so its distance to x is the null.
                #
                # This is the only paired, per-clip measurement in the output.
                # summary.json's stat_error compares POOLED generated stats to
                # POOLED real stats, which cannot see conditioning at all: a
                # model that ignores its context and emits the dataset average
                # scores well on it, while conditioning sharpens each sample and
                # can push pooled dispersion further away. Same failure mode as
                # distribution_match in Stage 4B. Read vs_truth, not stat_error,
                # for whether conditioning works.
                truth_vols=x[:, 0].detach().float().cpu(),
                extra=extra,
                save_video=not args.no_video,
            )
            n_done += B
    print(f"  batch {bi + 1}/{args.batches}: {n_done} samples written", flush=True)

summary = writer.close()
# The regime flags the writer stores are booleans; record the exact recipe
# alongside them so "partial" is recoverable and the run is reproducible.
(out_root / "run_config.json").write_text(json.dumps({
    "args": vars(args),
    "partial_local_features": partial_names,
    "regime_definitions": {
        "random": "gct and lct both sampled unconditionally from the train bank",
        "global_only": "gct pinned to the held-out clip; lct retrieved given gct",
        "global_partial_local":
            f"gct pinned; lct features {partial_names} pinned exactly, rest retrieved",
        "global_full_local": "gct and lct both pinned to the held-out clip",
    },
    "free_generation": True,
    "activity_checkpoint": str(M.CKPTS.get("activity_prior_refined_best"
                                           if args.phase == "4b_refine"
                                           else "activity_prior_best_hard_metric")),
    # Record what was ACTUALLY loaded, not the fallback key: the eval loader
    # prefers "motif_prior_ship" (the Stage 4C adapted prior) when it exists.
    "motif_checkpoint": str(_MOTIF_CKPT_USED),
    "context_bank": args.bank,
}, indent=2, default=str))

print()
print(summary_table(summary))
print(f"\nwrote {out_root}/  (manifest.json, summary.json, run_config.json)")
