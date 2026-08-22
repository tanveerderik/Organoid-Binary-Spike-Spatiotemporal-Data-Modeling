#!/usr/bin/env python3
"""Fit, sample and score an external baseline on the frozen held-out split.

    python external_baselines/run_baseline.py --list
    python external_baselines/run_baseline.py --baseline dg --fit --score

Scoring writes a tree identical in shape to the pipeline's own
`reports/generation_regimes_*`, so `analysis/compare_regimes.py` can pair a
baseline against the shipped model directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/media/derik/Seagate Desktop Drive/organoid_data")

from MAGVIT_project.external_baselines import registry
from MAGVIT_project.external_baselines.common import data as bdata
from MAGVIT_project.external_baselines.common.evaluate import (
    REFERENCE_PROTOCOL, evaluate_baseline)

CKPT_DIR = Path("ckpts/external_baselines")
OUT_DIR = Path("reports/external_baselines")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show registered baselines")
    ap.add_argument("--baseline", help="registered name, e.g. dg")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--fit-batches", type=int, default=120,
                    help="TRAIN batches used for fitting")
    # Scoring knobs default to REFERENCE_PROTOCOL -- the protocol the existing
    # generation_regimes_* sets were produced under. Overriding any of them
    # produces a run that fails clip-alignment verification, which is intended:
    # the comparative table is only meaningful if every row used one protocol.
    ap.add_argument("--batches", type=int, default=None,
                    help=f"held-out batches (default {REFERENCE_PROTOCOL['batches']})")
    ap.add_argument("--samples-per-clip", type=int, default=None,
                    help=f"draws per clip per regime "
                         f"(default {REFERENCE_PROTOCOL['samples_per_clip']})")
    ap.add_argument("--split", default=None, choices=("val", "test"))
    ap.add_argument("--no-verify", action="store_true",
                    help="skip clip-alignment check against the reference set")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default="", help="suffix for the output directory")
    ap.add_argument("--no-rate-cal", action="store_true")
    ap.add_argument("--videos", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("registered baselines:")
        for n in registry.available():
            b = registry.build(n)
            print(f"  {n:24s} {b.meta.family:12s} {b.meta.citation}")
        return 0

    if not args.baseline:
        ap.error("--baseline is required (or use --list)")

    bdata.ensure_repo_cwd()
    name = args.baseline
    tag = f"_{args.tag}" if args.tag else ""
    ckpt = CKPT_DIR / f"{name}{tag}.pt"
    out = OUT_DIR / f"{name}{tag}"

    baseline = registry.build(name)
    print(f"baseline : {baseline.meta.name}")
    print(f"citation : {baseline.meta.citation}")
    print(f"family   : {baseline.meta.family}")
    print(f"cond     : {baseline.meta.conditioning}")

    if args.fit:
        print(f"\nfitting on {args.fit_batches} TRAIN batches ...")
        baseline.fit(bdata.iter_raw("train", batches=args.fit_batches),
                     device=args.device)
        baseline.save(ckpt)
        print(f"saved {ckpt}")
        print(json.dumps(baseline.fit_report, indent=2, default=float)[:1400])
    elif args.score:
        if not ckpt.exists():
            raise SystemExit(f"no fitted checkpoint at {ckpt}; run with --fit first")
        baseline.load(ckpt, device=args.device)
        print(f"loaded {ckpt}")

    if args.score:
        print(f"\nscoring under the reference protocol ...")
        summary = evaluate_baseline(
            baseline, split=args.split, batches=args.batches,
            samples_per_clip=args.samples_per_clip,
            out_root=out, device=args.device, seed=args.seed,
            save_video=args.videos, rate_calibrate=not args.no_rate_cal,
            verify=not args.no_verify,
        )
        print(f"\nwrote {out}")
        for regime, e in summary.items():
            vr = e.get("vs_real", {})
            print(f"\n  {regime}   n={e['n_samples']}")
            print(f"    rate        {e['stats']['rate']:.3e}")
            print(f"    stat_error  {vr.get('stat_error', float('nan')):.4f}")
            print(f"    ks_avalanche{vr.get('ks_avalanche', float('nan')):8.4f}")
            print(f"    ks_isi      {vr.get('ks_isi', float('nan')):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
