"""Sampling noise in the Stage 4 generation composite.

The composite is scored by SAMPLING the prior over a capped number of
validation batches, so it carries stochastic noise of its own. Two checkpoints
separated by less than that noise are not distinguishable, and any claim that
one beat another needs this number attached.

Scores one fixed checkpoint repeatedly, varying only the sampling seed. The
validation masks stay deterministic throughout, so the spread measured here is
MaskGIT decoding noise alone -- not data variation, not mask variation.

Unlike the other two scripts in this directory this one needs the GPU: it runs
the real activity -> MaskGIT -> VQ-VAE path. Do not run it against a busy GPU
with large --batches; the defaults are deliberately small.

  python analysis/generation_seed_spread.py --phase 4b --seeds 8

Read alongside analysis/generation_metric_reference.py, which gives the scale
this spread should be judged against: if the seed spread is comparable to the
real-vs-real ceiling spread (0.0239 on the distribution subscore), the metric is
at its resolution limit and finer comparisons are not supportable.
"""
import sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, "/media/derik/Seagate Desktop Drive/organoid_data")
from MAGVIT_project import main as M
from MAGVIT_project.training.stage4_activity import evaluate_true_stage4_generation

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--phase", default="4b", choices=("4b", "4b_refine"))
ap.add_argument("--split", default="val", choices=("val", "test"))
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--batches", type=int, default=4)
ap.add_argument("--motif-steps", type=int, default=12)
ap.add_argument("--base-seed", type=int, default=20260820)
ap.add_argument("--activity-ckpt", default=None,
                help="Override the Stage 4B/4B-refine activity checkpoint. Useful for "
                     "scoring a specific candidate, or a smoke checkpoint.")
ap.add_argument("--motif-ckpt", default=None,
                help="Override the Stage 4A motif checkpoint.")
ap.add_argument("--out", default=None)
args = ap.parse_args()

if args.activity_ckpt:
    for _k in ("activity_prior_best", "activity_prior_best_loss",
               "activity_prior_best_hard_metric", "activity_prior_refined_best"):
        if _k in M.CKPTS:
            M.CKPTS[_k] = M.Path(args.activity_ckpt)
    print(f"activity checkpoint override: {args.activity_ckpt}")
if args.motif_ckpt:
    M.CKPTS["motif_prior_best"] = M.Path(args.motif_ckpt)
    print(f"motif checkpoint override: {args.motif_ckpt}")

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {dev}")

assay_dict = M.find_assays()
train_loader, val_loader, test_loader, meta = M.make_loaders(
    assay_dict=assay_dict,
    assay_indices=list(assay_dict.keys()),
    per_assay_quota=M.per_assay_quota_stage12,
)
loader = {"val": val_loader, "test": test_loader}[args.split]

b0 = next(iter(train_loader))
_, _, T0, H0, W0 = b0["x"].shape
model = M.make_vqvae((T0, H0, W0), dev,
                     full_spatial_size=tuple(map(int, b0["full_hw"][0])))
# The Stage 1 global-memory bank must be attached BEFORE scoring. Without
# memory_tok/memory_pix/memory_adj, _safe_generation_global_rows returns []
# and the composite silently swaps its spatial term from the decoded
# violation (~0.487) to the hard-activity fallback (~0.004). At weight 0.05
# that inflates the composite by ~0.024 -- six times the sampling noise this
# script exists to measure, and it produces numbers that look like a real
# improvement over the training-time figures. Same failure mode as the stale
# gap_bins default: a silent fallback, not an error.
M.load_stage1_gct_for_eval(model)
prior = M._load_stage4_eval_prior(model, dev, phase=args.phase, load_motif=True)
blank_code = getattr(model.vq, "blank_code", -1)

# Components worth tracking individually: the three that compare generated
# statistics to real ones, plus the composite.
KEYS = ("generation_metric", "generation_stat_error",
        "generation_ks_avalanche", "generation_ks_isi",
        "decoded_spike_count_relative_error", "hard_count_mae",
        "decoded_spatial_support_violation")

runs = []
print(f"\nscoring {args.seeds} seeds x {args.batches} batches "
      f"({args.motif_steps} motif steps) on {args.split}\n", flush=True)
for i in range(args.seeds):
    seed = int(args.base_seed) + 1009 * i
    g = evaluate_true_stage4_generation(
        prior.activity_prior,
        prior.motif_prior,
        model,
        loader,
        blank_code=blank_code,
        max_batches=args.batches,
        motif_steps=args.motif_steps,
        gap_bins=M.gap_bins,
        deterministic_masks=True,     # masks fixed; only sampling varies
        seed=seed,
        null_seed=seed,
    )
    src = g.get("spatial_support_metric_source")
    if src != "decoded_global_memory":
        raise SystemExit(
            f"spatial_support_metric_source is {src!r}, expected "
            "'decoded_global_memory'. The composite took its fallback branch, "
            "so these numbers are not comparable to the training-time ones. "
            "Check that the Stage 1 memory bank loaded."
        )
    runs.append({k: float(g[k]) for k in KEYS if k in g})
    print(f"  seed {seed}: composite={runs[-1]['generation_metric']:.6f} "
          f"stat_error={runs[-1].get('generation_stat_error', float('nan')):.4f}",
          flush=True)

summary = {}
for k in KEYS:
    vals = [r[k] for r in runs if k in r]
    if not vals:
        continue
    summary[k] = {
        "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
        "min": float(np.min(vals)), "max": float(np.max(vals)),
        "range": float(np.max(vals) - np.min(vals)),
    }

print("\n" + "=" * 78)
print(f"{'component':<40}{'mean':>10}{'sd':>10}{'range':>10}")
print("=" * 78)
for k, v in summary.items():
    print(f"{k:<40}{v['mean']:>10.5f}{v['sd']:>10.5f}{v['range']:>10.5f}")
print("=" * 78)

gm = summary.get("generation_metric")
if gm:
    print(f"\nSampling noise on the composite: sd {gm['sd']:.5f}, "
          f"range {gm['range']:.5f} over {args.seeds} seeds.")
    print("Do not claim a checkpoint difference smaller than this without "
          "averaging over seeds.")

out = args.out or f"reports/generation_seed_spread_{args.phase}.json"
p = M.Path(out); p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({
    "config": vars(args), "runs": runs, "summary": summary,
}, indent=2))
print(f"\nwrote {out}")
