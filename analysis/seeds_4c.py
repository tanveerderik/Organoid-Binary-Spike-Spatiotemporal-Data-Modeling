"""Seed variance for Stage 4C.

4C's marginal contribution over the soft-field-only arm is +0.015 MRR from a
single run. The old 4C (now 4b_refine) was rejected on exactly this argument --
its whole range sat inside 0.92 seed sd -- so the same standard has to be
applied here before the +0.015 can be claimed.

Runs the REAL dispatch path (STAGE4_PHASES=("4c",) -> run_stage4_prior), which
also re-loads the 4A control per seed, so each run starts from the same init and
only the stochastic parts differ: MaskGIT corruption, the per-sample Bernoulli
substitution draw, 4B's Gumbel activity readout, dataloader order, dropout.

Writes ckpts/motif_prior_adapt_SEED{n}.pt -- the shipped
ckpts/motif_prior_adapt_best.pt is never touched.
"""
import sys, json, hashlib, pathlib, argparse
sys.path.insert(0, "/media/derik/Seagate Desktop Drive/organoid_data")
import torch, numpy as np, random
from MAGVIT_project import main as M

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="101,202,303")
ap.add_argument("--out", default="reports/stage4c_seed_spread.json",
                help="output path; override to keep a re-run from replacing "
                     "the recorded seed spread")
args = ap.parse_args()

M.RUN_EVAL = M.RUN_VIZ = M.RUN_VIDEO_GEN = M.RUN_PLOTTER = M.RUN_CODEBOOK_DEBUG = False
M.STAGE4_PHASES = ("4c",)

GUARD = "ckpts/motif_prior_adapt_best.pt"
guard_md5 = hashlib.md5(open(GUARD, "rb").read()).hexdigest()
ctrl_md5 = hashlib.md5(open("ckpts/motif_prior_best.pt", "rb").read()).hexdigest()

dev = "cuda" if torch.cuda.is_available() else "cpu"
ad = M.find_assays()
tr, va, te, meta = M.make_loaders(assay_dict=ad, assay_indices=list(ad.keys()),
                                  per_assay_quota=M.per_assay_quota_stage12)
b0 = next(iter(tr)); _, _, T0, H0, W0 = b0["x"].shape

out = {}
for seed in [int(s) for s in args.seeds.split(",")]:
    print("\n" + "#" * 78)
    print(f"# STAGE 4C  seed {seed}")
    print("#" * 78, flush=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if dev == "cuda":
        torch.cuda.manual_seed_all(seed)

    M.CKPTS["motif_prior_ship"] = M.Path(f"ckpts/motif_prior_adapt_SEED{seed}.pt")
    M.REPORTS["prior_adapt"] = M.Path(f"reports/training_report_prior_4C_adapt_SEED{seed}.json")

    model = M.make_vqvae((T0, H0, W0), dev,
                         full_spatial_size=tuple(map(int, b0["full_hw"][0])))
    M.load_spatial_pretrain_if_available(model)
    reports = M.run_stage4_prior(model, tr, va, dev)

    val = reports["4c"]["val"]
    best = max(val, key=lambda e: e["mrr_z1"])
    out[seed] = {
        "best_epoch": val.index(best) + 1,
        "best_mrr": best["mrr_z1"],
        "oracle_at_best": best.get("oracle_mrr_z1"),
        "epochs_run": len(val),
        "ckpt": str(M.CKPTS["motif_prior_ship"]),
    }
    print(f"[seed {seed}] best val mrr={best['mrr_z1']:.5f} at epoch "
          f"{out[seed]['best_epoch']} of {len(val)}", flush=True)

    assert hashlib.md5(open(GUARD, "rb").read()).hexdigest() == guard_md5, \
        "shipped checkpoint was modified"
    assert hashlib.md5(open("ckpts/motif_prior_best.pt", "rb").read()).hexdigest() == ctrl_md5, \
        "4A control was modified"
    del model
    torch.cuda.empty_cache()

# The shipped run, for reference.
shipped = json.load(open("reports/training_report_prior_4C_adapt.json"))["val"]
sb = max(shipped, key=lambda e: e["mrr_z1"])
out["shipped"] = {"best_epoch": shipped.index(sb) + 1, "best_mrr": sb["mrr_z1"],
                  "oracle_at_best": sb.get("oracle_mrr_z1"),
                  "epochs_run": len(shipped),
                  "ckpt": "ckpts/motif_prior_adapt_best.pt"}

vals = [v["best_mrr"] for v in out.values()]
mean = sum(vals) / len(vals)
sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
out["_summary"] = {"n": len(vals), "mean_best_val_mrr": mean, "sd_best_val_mrr": sd,
                   "min": min(vals), "max": max(vals)}
print("\n" + "=" * 78)
for k, v in out.items():
    if not str(k).startswith("_"):
        print(f"  {str(k):>8}  best val mrr {v['best_mrr']:.5f}  ep {v['best_epoch']:>3}  "
              f"oracle {v['oracle_at_best']:.5f}  ckpt {v['ckpt']}")
print(f"\n  val-MRR across {len(vals)} runs: mean {mean:.5f}  sd {sd:.5f}  "
      f"range {min(vals):.5f}-{max(vals):.5f}")
print("=" * 78)
json.dump(out, open(args.out, "w"), indent=2)
print(f"wrote {args.out}")
