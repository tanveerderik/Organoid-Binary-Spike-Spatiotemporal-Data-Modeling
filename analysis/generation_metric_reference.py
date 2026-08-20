"""Reference ladder for the generation composite: floor, rungs, and ceiling.

The composite is a number in [0,1] with no intrinsic scale, which is the first
thing a reviewer will object to. This establishes what the scale means, using
the same mea_statistics / compare_statistics path the composite itself uses.

Rungs, all scored against the same real reference half:

  real-vs-real   split the held-out real data in two and score one half against
                 the other. This is the CEILING: with finite samples even real
                 data does not reproduce its own statistics exactly, so no
                 generator can score better. Repeated over several disjoint
                 splits to get a spread.
  temporal shuffle   independently permute each electrode's time series. Keeps
                 the exact firing rate and per-electrode marginals, destroys
                 temporal persistence, avalanches and ISI structure.
  spatial shuffle    permute electrode identities within each frame. Keeps rate
                 and temporal structure, destroys spatial co-activation.
  mean-field     independent Bernoulli at the matched global rate. Keeps only
                 the rate. This is the FLOOR.

Everything runs on CPU: no model is involved, so this does not contend with a
training run on the GPU.
"""
import sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, "/media/derik/Seagate Desktop Drive/organoid_data")
from MAGVIT_project import main as M
from MAGVIT_project.inference.metrics_generative import (
    mea_statistics, compare_statistics,
)

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test", choices=("val", "test"))
ap.add_argument("--batches", type=int, default=24)
ap.add_argument("--splits", type=int, default=4, help="disjoint real-vs-real splits")
ap.add_argument("--seed", type=int, default=20260820)
ap.add_argument("--out", default="reports/generation_metric_reference.json")
args = ap.parse_args()

torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)

print("loading assays...", flush=True)
assay_dict = M.find_assays()
assay_indices = sorted(assay_dict.keys())
train_loader, val_loader, test_loader, meta = M.make_loaders(
    assay_dict=assay_dict,
    assay_indices=assay_indices,
    per_assay_quota=M.per_assay_quota_stage12,
)
loader = {"val": val_loader, "test": test_loader}[args.split]

vols = []
for i, batch in enumerate(loader):
    if i >= args.batches:
        break
    x = batch["x"]
    if x.dim() == 5:
        x = x.squeeze(1)
    vols.append((x > 0.5).float().cpu())
V = torch.cat(vols)
print(f"collected {tuple(V.shape)} from {args.split}", flush=True)
N = V.shape[0]
if N < 4:
    raise SystemExit(f"need >=4 clips, got {N}")


def score(real_vol, other_vol):
    """The three distribution terms of the composite, and their weighted sum."""
    r = mea_statistics(real_vol)
    g = mea_statistics(other_vol)
    c = compare_statistics(r, g)
    stat_error = float(c["stat_error"])
    ks_av = float(c["ks_avalanche"])
    ks_isi = float(c["ks_isi"])
    dist_match = 1.0 / (1.0 + stat_error)
    # 0.35 / 0.15 / 0.10 of the composite -- the 0.60 that is measured against
    # real data. The remaining 0.40 are model-vs-target consistency terms with
    # no real-vs-real analogue, so they are excluded rather than assumed
    # perfect.
    sub = 0.35 * dist_match + 0.15 * (1 - ks_av) + 0.10 * (1 - ks_isi)
    per_stat = {k: v for k, v in c.items() if k.startswith("rel_")}
    return {
        "stat_error": stat_error,
        "ks_avalanche": ks_av,
        "ks_isi": ks_isi,
        "distribution_match": dist_match,
        "distribution_subscore_0.60": sub,
        "distribution_subscore_normalized": sub / 0.60,
        "per_statistic_relative_error": per_stat,
    }


results = {}

# ---- ceiling: disjoint real-vs-real splits -------------------------------
print("\n[ceiling] real vs real", flush=True)
ceil_runs = []
for s in range(args.splits):
    perm = rng.permutation(N)
    a, b = perm[: N // 2], perm[N // 2 : 2 * (N // 2)]
    r = score(V[a], V[b])
    ceil_runs.append(r)
    print(f"  split {s}: stat_error={r['stat_error']:.4f} "
          f"ks_av={r['ks_avalanche']:.4f} ks_isi={r['ks_isi']:.4f} "
          f"sub={r['distribution_subscore_normalized']:.4f}", flush=True)


def agg(runs, key):
    vals = [r[key] for r in runs]
    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals)),
            "min": float(np.min(vals)), "max": float(np.max(vals))}


results["ceiling_real_vs_real"] = {
    "n_splits": args.splits,
    **{k: agg(ceil_runs, k) for k in
       ("stat_error", "ks_avalanche", "ks_isi",
        "distribution_subscore_normalized")},
    "runs": ceil_runs,
}

# reference half for every degraded rung
perm = rng.permutation(N)
REF, HALF = V[perm[: N // 2]], V[perm[N // 2 : 2 * (N // 2)]]

# ---- temporal shuffle ----------------------------------------------------
print("\n[rung] temporal shuffle (rate kept, time structure destroyed)", flush=True)
tsh = HALF.clone()
for i in range(tsh.shape[0]):
    idx = torch.randperm(tsh.shape[1])
    tsh[i] = tsh[i][idx]
results["rung_temporal_shuffle"] = score(REF, tsh)
print(f"  stat_error={results['rung_temporal_shuffle']['stat_error']:.4f} "
      f"sub={results['rung_temporal_shuffle']['distribution_subscore_normalized']:.4f}")

# ---- spatial shuffle -----------------------------------------------------
print("\n[rung] spatial shuffle (electrode identity permuted per frame)", flush=True)
ssh = HALF.clone()
B, T, H, W = ssh.shape
flat = ssh.reshape(B, T, H * W)
for i in range(B):
    for t in range(T):
        flat[i, t] = flat[i, t][torch.randperm(H * W)]
ssh = flat.reshape(B, T, H, W)
results["rung_spatial_shuffle"] = score(REF, ssh)
print(f"  stat_error={results['rung_spatial_shuffle']['stat_error']:.4f} "
      f"sub={results['rung_spatial_shuffle']['distribution_subscore_normalized']:.4f}")

# ---- floor: mean-field ---------------------------------------------------
print("\n[floor] mean-field Bernoulli at matched rate", flush=True)
rate = float(HALF.mean())
mf = (torch.rand_like(HALF) < rate).float()
results["floor_mean_field"] = {"rate": rate, **score(REF, mf)}
print(f"  rate={rate:.6f} "
      f"stat_error={results['floor_mean_field']['stat_error']:.4f} "
      f"sub={results['floor_mean_field']['distribution_subscore_normalized']:.4f}")

results["config"] = {
    "split": args.split, "batches": args.batches, "clips": int(N),
    "volume_shape": list(V.shape[1:]), "seed": args.seed,
    "note": ("Only the 0.60 of the composite that compares generated to real "
             "statistics is scored here. The remaining 0.40 (decoded/activity "
             "count consistency, local context, short gap, spatial violation, "
             "duplicate penalty) are model-vs-target terms with no real-vs-real "
             "analogue."),
}

out = M.Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2))

print("\n" + "=" * 74)
print(f"{'rung':<26}{'stat_error':>12}{'ks_av':>9}{'ks_isi':>9}{'sub/0.60':>11}")
print("=" * 74)
c = results["ceiling_real_vs_real"]
print(f"{'CEILING real-vs-real':<26}{c['stat_error']['mean']:>12.4f}"
      f"{c['ks_avalanche']['mean']:>9.4f}{c['ks_isi']['mean']:>9.4f}"
      f"{c['distribution_subscore_normalized']['mean']:>11.4f}"
      f"  (sd {c['distribution_subscore_normalized']['sd']:.4f})")
for key, label in (("rung_temporal_shuffle", "temporal shuffle"),
                   ("rung_spatial_shuffle", "spatial shuffle"),
                   ("floor_mean_field", "FLOOR mean-field")):
    r = results[key]
    print(f"{label:<26}{r['stat_error']:>12.4f}{r['ks_avalanche']:>9.4f}"
          f"{r['ks_isi']:>9.4f}{r['distribution_subscore_normalized']:>11.4f}")
print("=" * 74)
print(f"\nwrote {out}")
