#!/usr/bin/env python3
"""Is the sparse encoder load-bearing, or just a speed optimisation?

91.7% of tokens are blank (`avg_vq_blank_frac` in the shipped Stage 2A report).
The shipped design routes those to a single learned `blank_token` and quantizes
only the ~8.3% that carry a spike. The claim to be tested is stronger than
"it is faster": without the routing, the EMA codebook is handed a training set
that is 92% one identical point (a blank patch is EXACTLY the zero vector, so a
linear projection maps every one of them to the bias), and the cluster
statistics collapse onto it.

Two arms, identical in every other respect -- same seed, same data order, same
optimizer, same schedule, same epoch count, same losses:

    sparse   the shipped configuration
    dense    `dense_ablation=True`: every token declared active

DECISION METRIC is codebook occupancy, not reconstruction. `active_codes_
nonblank_l{1,2,3}` and `perplexity_nonblank_l{1,2,3}` are already logged every
epoch by `fit_vqvae`; collapse shows up there within a few epochs and needs no
new instrumentation. Reconstruction AUPRC is recorded too, but it is the
secondary axis -- a collapsed codebook can still reconstruct passably once the
decoder learns a bias, which is precisely the failure mode being demonstrated.

HONEST SCOPE, state it with the result: the dense arm ablates the sparse design
as a PACKAGE, not the quantizer routing alone. Two losses key off `blank_mask`
(`blank_patch_logit_hinge_loss`, `blank_active_decoder_separation_loss`,
train_vqvae.py:764,780) and both go inactive when nothing is blank. That is the
right scope for the question a reviewer asks -- "why this encoder design" -- but
it is not an isolation of the EMA update, and must not be reported as one.

Writes reports/ablation_sparse_encoder.json. Touches no shipped checkpoint:
everything lands in ckpts/ablations/.

    python ablations/sparse_encoder.py --epochs 80
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

# main.py uses package-relative imports, so it can only be imported as
# `MAGVIT_project.main` with the PARENT of the repo on sys.path -- importing it
# as a bare `main` raises "attempted relative import with no known parent
# package". Same gotcha the Stage 4B/4C smoke runs hit.
import MAGVIT_project.main as M                            # noqa: E402
from MAGVIT_project.model.vqvae import TransformerVQVAE    # noqa: E402
from MAGVIT_project.training.train_vqvae import fit_vqvae  # noqa: E402

CK = ROOT / "ckpts" / "ablations"
OUT = ROOT / "reports" / "ablation_sparse_encoder.json"


def build(img_size, full_hw, device, *, dense: bool, seed: int):
    """Same constructor call as `main.make_vqvae`, plus the ablation flag.

    Reproduced rather than imported because make_vqvae takes no such flag and
    editing it would put an ablation switch on the shipped build path. The
    argument list is kept in the same order as main.py:747-779 so a diff
    between the two is readable.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TransformerVQVAE(
        img_size=img_size,
        full_spatial_size=full_hw,
        patch_size=M.patch_size,
        encoder_embed_dim=64, encoder_depth=2, encoder_num_heads=4,
        code_dim=64,
        num_codes=M.num_codes,
        num_quantizers=M.num_quantizers,
        decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=4,
        in_chans=1, out_chans=1,
        local_ctx_in_dim=9, local_emb_dim=32,
        global_ctx_in_dim=M.dim_assay_for_emb, global_emb_dim=32,
        use_spatial_map_prior=M.USE_GCT_PRETRAIN_MODULE,
        gap_bins=M.gap_bins,
        enc_attn_mask_kind="none",
        dec_attn_mask_kind="temporal_causal",
        dense_ablation=dense,
    ).to(device)
    for m in model.modules():
        if isinstance(m, (torch.nn.LayerNorm, torch.nn.GroupNorm)):
            m.float()
    # Stage 0 GCT pretrain, exactly as main() does before Stage 2A. It supplies
    # `memory_adj`, without which fit_vqvae refuses to run the ISI term
    # (train_vqvae.py:611). Both arms load the identical frozen checkpoint, so
    # it is common ground, not a difference between them.
    M.load_spatial_pretrain_if_available(model)
    return model


def run_arm(name, *, dense, epochs, seed, loaders, img_size, full_hw, device,
            blank_thr, tok_entropy=0.0):
    train_loader, val_loader = loaders
    print("\n" + "=" * 78)
    print(f"ARM: {name}   dense_ablation={dense}   "
          f"lambda_tok_entropy={tok_entropy}   epochs={epochs} seed={seed}")
    print("=" * 78, flush=True)

    model = build(img_size, full_hw, device, dense=dense, seed=seed)
    M.freeze_for_stage(model, 2)
    model._set_training_prob_threshold(0.5)

    opt = M.make_optimizer(model, lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                     eta_min=1e-5)
    t0 = time.time()
    # Hyperparameters copied from run_stage2a (main.py:1193-1250). The only
    # deviations are the epoch budget and the schedules that are defined in
    # units of it -- holding those at their 300-epoch values inside a 40-epoch
    # run would mean neither arm ever leaves warm-up.
    rep = fit_vqvae(
        model, train_loader, val_loader, opt, sch,
        epochs=epochs,
        ckpt_best_path=str(CK / f"sparse_abl_{name}_best.pt"),
        ckpt_last_path=str(CK / f"sparse_abl_{name}_last.pt"),
        early_stop_patience=10**9,          # no early stop: arms must be paired
        val_metric_name="AUPRC", val_metric_goal="max",
        use_ROI_mask=False,
        lambda_ctx=float(M.STAGE2_LAMBDA_CTX),
        lambda_ctx_field=5e-2,
        ctx_start_epoch=5, ctx_warmup_epochs=15,
        cfg_ctx_drop_start=0.0, cfg_ctx_drop_end=0.0,
        cfg_ctx_start_epoch=10**9, cfg_ctx_warmup_epochs=1,
        blank_logit_margin=blank_thr,
        # Token-level profile entropy. Ramped in at 3/4 of the budget: entropy
        # is stationary at the uniform distribution, so it has nothing to
        # sharpen until reconstruction has put structure in the patch.
        lambda_tok_entropy=float(tok_entropy),
        tok_entropy_start_epoch=int(epochs * 0.75),
        tok_entropy_warmup_epochs=max(1, int(epochs * 0.10)),
        lambda_code_norm=0.10,
        code_norm_rms_ceiling=10.0, code_norm_token_ceiling=14.0,
        pos_weight_start=100.0, pos_weight_end=1.0,
        pos_decay_epochs=max(1, int(epochs * 2 / 3)),
        # The staged level-activation schedule (20/35/50/65) arrives via
        # common_fit_kwargs and is NOT overridden here. Those epoch numbers are
        # absolute, not proportional: each level needs ~15 epochs to populate by
        # EMA between activation and carrying full loss weight, and compressing
        # that window is itself worth 3.4x (main.py:1031-1049). Rescaling it to
        # a short budget would ablate the warm-up schedule instead of the
        # encoder. This is what fixes the epoch budget at >= 80.
        # Best-checkpoint tracking ON so val AUPRC is recorded, but both
        # ckpt paths point into ckpts/ablations/ -- no shipped artifact is
        # reachable from this run.
        save_start_epoch=0,
        **M.common_fit_kwargs(model),
    )
    rep["_arm"] = {"name": name, "dense_ablation": dense, "seed": seed,
                   "lambda_tok_entropy": float(tok_entropy),
                   "epochs": epochs, "wall_seconds": time.time() - t0}
    return rep


def summarise(reps: dict) -> dict:
    """Codebook occupancy per epoch, per level, per arm -- the decision axis."""
    s = {}
    for name, rep in reps.items():
        log = rep.get("history", {}).get("train_log", [])
        row = {"epochs_logged": len(log)}
        for lvl in (1, 2, 3):
            ac = [e.get(f"active_codes_nonblank_l{lvl}") for e in log]
            pp = [e.get(f"perplexity_nonblank_l{lvl}") for e in log]
            ac = [v for v in ac if v is not None]
            pp = [v for v in pp if v is not None]
            row[f"l{lvl}"] = {
                "active_codes_final": (ac[-1] if ac else None),
                "active_codes_max": (max(ac) if ac else None),
                "perplexity_final": (pp[-1] if pp else None),
                "perplexity_max": (max(pp) if pp else None),
                "active_codes_curve": ac,
            }
        row["blank_frac_final"] = (log[-1].get("avg_vq_blank_frac")
                                   if log else None)
        if log and log[-1].get("tok_h_model"):
            row["tok_entropy"] = {
                "h_model_final": log[-1].get("tok_h_model"),
                "h_true_final": log[-1].get("tok_h_true"),
                "frac_violating_final": log[-1].get("tok_frac_violating"),
                "loss_final": log[-1].get("loss_tok_entropy"),
                "h_model_curve": [e.get("tok_h_model") for e in log],
            }
        row["best_val_auprc"] = rep.get("best_val")
        row["wall_seconds"] = rep["_arm"]["wall_seconds"]
        s[name] = row
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--arms", default="sparse,dense",
                    help="sparse | dense | tokent (sparse + token entropy)")
    ap.add_argument("--tok-entropy", type=float, default=0.05,
                    help="lambda for the `tokent` arm")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assay_dict = M.find_assays()
    train_loader, val_loader, _test, meta = M.make_loaders(
        assay_dict=assay_dict,
        assay_indices=list(assay_dict.keys()),
        per_assay_quota=M.per_assay_quota_stage12,
    )
    b0 = next(iter(train_loader))
    img_size = tuple(b0["x"].shape[-3:])
    full_hw = tuple(map(int, b0["full_hw"][0]))
    p0 = M.compute_p0_from_loader(train_loader, max_batches=100)
    blank_thr = float(np.log(p0 / (1 - p0)))
    print(f"img_size={img_size} full_hw={full_hw} p0={p0:.3e} "
          f"blank_logit_margin={blank_thr:.3f}", flush=True)

    CK.mkdir(parents=True, exist_ok=True)
    reps = {}
    for name in [s.strip() for s in a.arms.split(",") if s.strip()]:
        reps[name] = run_arm(
            name, dense=(name == "dense"), epochs=a.epochs, seed=a.seed,
            loaders=(train_loader, val_loader), img_size=img_size,
            full_hw=full_hw, device=device, blank_thr=blank_thr,
            tok_entropy=(a.tok_entropy if name == "tokent" else 0.0))
        # MERGE, never overwrite. This path is fixed, so a short smoke run used
        # to clobber a completed one: a 4-epoch --arms tokent run destroyed the
        # 80-epoch sparse and dense results, and the checkpoints carry weights
        # only, so the per-epoch curves were unrecoverable. Only the arms this
        # invocation actually ran are replaced.
        prev = json.loads(OUT.read_text()) if OUT.is_file() else {}
        arms_all = {**prev.get("arms", {}), **reps}
        summ_all = {**prev.get("summary", {}), **summarise(reps)}
        OUT.write_text(json.dumps(
            {"arms": arms_all, "summary": summ_all,
             "scope_caveat": __doc__.split("HONEST SCOPE, ")[1].split("Writes")[0].strip()},
            indent=1))
        print(f"[ablation] wrote {OUT} after arm '{name}'", flush=True)

    s = summarise(reps)
    print("\n" + "=" * 78)
    hdr = f"{'arm':<9}{'blank_frac':>11}" + "".join(
        f"{f'L{l} codes':>10}{f'L{l} ppl':>9}" for l in (1, 2, 3)) + f"{'val AUPRC':>11}"
    print(hdr); print("-" * len(hdr))
    for k, v in s.items():
        bf = v["blank_frac_final"]
        line = f"{k:<9}{(f'{bf:.4f}' if bf is not None else '--'):>11}"
        for l in (1, 2, 3):
            c = v[f"l{l}"]["active_codes_final"]
            p = v[f"l{l}"]["perplexity_final"]
            line += f"{(f'{c:.1f}' if c is not None else '--'):>10}"
            line += f"{(f'{p:.1f}' if p is not None else '--'):>9}"
        va = v["best_val_auprc"]
        line += f"{(f'{va:.4f}' if va is not None else '--'):>11}"
        print(line)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
