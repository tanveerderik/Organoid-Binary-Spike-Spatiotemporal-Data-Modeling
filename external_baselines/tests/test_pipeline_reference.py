#!/usr/bin/env python3
"""Does the pipeline adapter reproduce `analysis/generate_regimes.py` exactly?

Two claims are load-bearing and both are cheap to check, so neither is left as
an argument in a docstring:

  1. **No leakage.** The adapter feeds a ZERO volume as `x_ref` where
     generate_regimes feeds the real held-out clip. If free generation truly
     discards the encoded codes, the two must agree BITWISE under the same
     seed. If they ever diverge, the adapter is reading the clip -- and so is
     the shipped generation path, which would be the far bigger finding.

  2. **Same model.** The adapter must load the same three checkpoints the
     shipped generation run loads, and binarise at the same `best_thr_tol`.

    python external_baselines/tests/test_pipeline_reference.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, "/media/derik/Seagate Desktop Drive/organoid_data")

from MAGVIT_project.external_baselines import registry
from MAGVIT_project.external_baselines.common import data as bdata
from MAGVIT_project.external_baselines.common.pipeline import build_pipeline

SEED = 20260822
n_pass = n_fail = 0


def check(name, ok, detail=""):
    global n_pass, n_fail
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    n_pass, n_fail = n_pass + bool(ok), n_fail + (not ok)


def main() -> int:
    bdata.ensure_repo_cwd()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    b = registry.build("pipeline")
    b.load(None, device=dev)
    p = build_pipeline(dev, phase="4b")

    cond, real = next(bdata.iter_split("test", batches=1))
    cond = cond.to(dev)

    # -- 1. zero x_ref vs the real clip, bitwise -------------------------
    def draw(x_ref):
        torch.manual_seed(SEED)
        if dev == "cuda":
            torch.cuda.manual_seed_all(SEED)
        d = b._generate(cond, x_ref=x_ref)
        x = d["x_gen"]
        return (x.squeeze(1) if x.dim() == 5 else x).clone()

    a = draw(None)                                   # what generation uses
    c = draw(real.unsqueeze(1).to(dev).float())      # what generate_regimes uses
    check("free generation ignores x_ref (bitwise)", bool(torch.equal(a, c)),
          f"differing voxels = {int((a != c).sum())}")
    check("sample is binary", bool(((a == 0) | (a == 1)).all()))
    check("sample shape", tuple(a.shape) == (real.shape[0],) + cond.shape,
          f"{tuple(a.shape)}")
    check("sample is not empty", float(a.sum()) > 0, f"spikes = {float(a.sum()):.0f}")

    # -- 2. same checkpoints, same threshold -----------------------------
    ex = b.meta.extra
    for k in ("vq_ckpt", "motif_ckpt", "activity_ckpt"):
        check(f"{k} resolved", Path(str(ex[k])).exists(), str(ex[k]))
    check("threshold is the shipped best_thr_tol",
          abs(ex["best_thr_tol"] - float(p["vqvae"].best_thr_tol.item())) < 1e-12,
          f"{ex['best_thr_tol']:.6f}")

    # -- 3. intensity is the field the sample came from ------------------
    torch.manual_seed(SEED)
    v = b.sample(cond)
    f = b.sample_intensity(cond)
    check("intensity matches its own sample at the threshold",
          bool(torch.equal((torch.sigmoid(f) >= ex["best_thr_tol"]).float(), v)))
    check("intensity is not flat", float(f.flatten(1).std(1).mean()) > 1e-3,
          f"std = {float(f.flatten(1).std(1).mean()):.3f}")

    # -- 4. the adapter cannot write ------------------------------------
    try:
        b.save(Path("/dev/null"))
        check("save() refuses", False)
    except NotImplementedError:
        check("save() refuses", True)
    try:
        b.fit(iter([]), device=dev)
        check("fit() refuses", False)
    except NotImplementedError:
        check("fit() refuses", True)

    print(f"\n{n_pass}/{n_pass + n_fail} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
