"""Weight-sensitivity analysis for the Stage 4 generation composite.

The composite combines nine terms with hand-set weights. There is no principled
derivation for 0.35 vs 0.30, so the defence is not an argument -- it is showing
that the checkpoint the composite selects does not depend on the exact weights.

Three analyses, all post-hoc over a training report (no retraining, no GPU):

  1. RECOMPUTE CHECK. Rebuild the composite from its logged components and
     require it to reproduce the logged value. If this fails, everything below
     is meaningless.
  2. DIRICHLET PERTURBATION. Resample the weight vector around the nominal one
     and record which epoch each perturbed composite selects. Report the
     fraction that still select the nominal epoch, and how the alternatives
     score under the NOMINAL weights -- the question is not whether the
     argmax moves, it is whether moving costs anything.
  3. LEAVE-ONE-OUT. Drop each term, renormalise the rest, re-select.

Usage:
  python gen_weight_sensitivity.py --report reports/training_report_prior_4B.json
"""
import sys, json, argparse
import numpy as np

sys.path.insert(0, "/media/derik/Seagate Desktop Drive/organoid_data")

# (name, nominal weight, how to turn logged components into a [0,1] goodness)
TERMS = [
    ("distribution_match",          0.35, lambda r: 1.0 / (1.0 + r["generation_stat_error"])),
    ("avalanche_match",             0.15, lambda r: 1.0 - r["generation_ks_avalanche"]),
    ("isi_match",                   0.10, lambda r: 1.0 - r["generation_ks_isi"]),
    ("decoded_count_consistency",   0.15, lambda r: max(0.0, 1.0 - r["decoded_spike_count_relative_error"])),
    ("activity_count_consistency",  0.05, lambda r: max(0.0, 1.0 - r["hard_count_mae"] / max(r["hard_target_count_mean"], 1.0))),
    ("local_context_consistency",   0.10, lambda r: 1.0 / (1.0 + r["generated_local_context_mae"])),
    ("short_gap_consistency",       0.05, lambda r: 1.0 / (1.0 + r["decoded_short_gap_mae"])),
    ("spatial_consistency",         0.05, lambda r: max(0.0, 1.0 - r["decoded_spatial_support_violation"])),
    ("duplicate_penalty",          -0.05, lambda r: r["hard_duplicate_token_rate"]),
]

ap = argparse.ArgumentParser()
ap.add_argument("--report", required=True)
ap.add_argument("--draws", type=int, default=2000)
ap.add_argument("--concentration", type=float, default=50.0,
                help="Dirichlet concentration; lower = wilder perturbation")
ap.add_argument("--seed", type=int, default=20260820)
ap.add_argument("--out", default=None)
args = ap.parse_args()

raw = json.load(open(args.report))
rows = raw["history"] if isinstance(raw, dict) and "history" in raw else (
    raw["val"] if isinstance(raw, dict) and "val" in raw else raw
)

# keep only epochs that were actually scored
scored = []
for row in rows:
    r = {k[4:]: v for k, v in row.items() if k.startswith("gen_")}
    if "generation_metric" in r:
        scored.append((int(row.get("epoch", len(scored) + 1)), r))
if not scored:
    raise SystemExit(f"No epochs with gen_* components in {args.report}")
print(f"{len(scored)} scored epochs: {[e for e, _ in scored]}")


def goodness_vector(r):
    return np.array([f(r) for _, _, f in TERMS], dtype=float)


G = np.stack([goodness_vector(r) for _, r in scored])       # (E, 9)
W = np.array([w for _, w, _ in TERMS], dtype=float)          # (9,)
epochs = np.array([e for e, _ in scored])
logged = np.array([r["generation_metric"] for _, r in scored])

# ---- 1. recompute check --------------------------------------------------
recomputed = G @ W
err = np.abs(recomputed - logged)
print("\n[1] recompute check")
for e, lo, re_, d in zip(epochs, logged, recomputed, err):
    print(f"  ep{e:4d}  logged={lo:.8f}  recomputed={re_:.8f}  |d|={d:.2e}")
ok = bool(err.max() < 1e-8)
print(f"  max |error| = {err.max():.2e}  ->  {'OK' if ok else 'MISMATCH'}")
if not ok:
    raise SystemExit("Recomputation does not reproduce the logged composite; "
                     "the TERMS table is out of sync with stage4_activity.py.")

nominal_best = int(epochs[int(np.argmax(recomputed))])
print(f"\nnominal selection: epoch {nominal_best} "
      f"(composite {recomputed.max():.6f})")

results = {"report": args.report, "scored_epochs": epochs.tolist(),
           "nominal_best_epoch": nominal_best,
           "nominal_composite": float(recomputed.max())}

if len(scored) < 2:
    print("\nOnly one scored epoch -- perturbation analysis needs >=2. "
          "Rerun after a full training run.")
    results["note"] = "insufficient epochs for perturbation analysis"
else:
    rng = np.random.default_rng(args.seed)
    pos = W.copy(); pos[-1] = abs(pos[-1])
    p = pos / pos.sum()

    # ---- 2. Dirichlet perturbation ---------------------------------------
    picks, regrets = [], []
    for _ in range(args.draws):
        w = rng.dirichlet(p * args.concentration)
        w = w / w.sum() * pos.sum()
        w[-1] = -w[-1]
        sel = int(np.argmax(G @ w))
        picks.append(int(epochs[sel]))
        # what did choosing that epoch cost, measured in NOMINAL composite?
        regrets.append(float(recomputed.max() - recomputed[sel]))
    picks = np.array(picks); regrets = np.array(regrets)
    stable = float((picks == nominal_best).mean())
    uniq, cnt = np.unique(picks, return_counts=True)
    order = np.argsort(-cnt)
    print(f"\n[2] Dirichlet perturbation ({args.draws} draws, "
          f"concentration {args.concentration})")
    print(f"  selects nominal epoch: {stable*100:.1f}%")
    print("  epoch distribution:")
    for i in order[:6]:
        print(f"    ep{uniq[i]:4d}  {cnt[i]/len(picks)*100:5.1f}%")
    print(f"  regret under nominal weights: mean {regrets.mean():.6f}  "
          f"p95 {np.percentile(regrets,95):.6f}  max {regrets.max():.6f}")
    results["dirichlet"] = {
        "draws": args.draws, "concentration": args.concentration,
        "fraction_selecting_nominal": stable,
        "epoch_distribution": {int(u): int(c) for u, c in zip(uniq, cnt)},
        "regret_mean": float(regrets.mean()),
        "regret_p95": float(np.percentile(regrets, 95)),
        "regret_max": float(regrets.max()),
    }

    # ---- 3. leave-one-out ------------------------------------------------
    print("\n[3] leave-one-out")
    loo = {}
    for i, (name, _, _) in enumerate(TERMS):
        w = W.copy(); w[i] = 0.0
        s = np.sign(W); s[s == 0] = 1
        scale = np.abs(W).sum() / max(np.abs(w).sum(), 1e-12)
        sel = int(np.argmax(G @ (w * scale)))
        ep = int(epochs[sel])
        reg = float(recomputed.max() - recomputed[sel])
        loo[name] = {"selected_epoch": ep, "regret": reg}
        flag = "" if ep == nominal_best else "   <- moves"
        print(f"  drop {name:<28} -> ep{ep:4d}  regret {reg:.6f}{flag}")
    results["leave_one_out"] = loo
    n_moved = sum(1 for v in loo.values() if v["selected_epoch"] != nominal_best)
    print(f"  {n_moved}/{len(TERMS)} terms change the selection")

out = args.out or args.report.replace(".json", "_weight_sensitivity.json")
json.dump(results, open(out, "w"), indent=2)
print(f"\nwrote {out}")
