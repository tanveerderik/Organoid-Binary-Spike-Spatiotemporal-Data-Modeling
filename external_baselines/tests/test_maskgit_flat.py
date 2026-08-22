#!/usr/bin/env python3
"""End-to-end checks for the MaskGIT-flat baseline on a tiny fit.

Cheap enough to run before every real training job. It catches the failures
that would otherwise only surface after hours of GPU: an unresolved mask token
left in the output, a codebook that collapsed to one entry, a sample that is not
binary or not at the requested rate, and a save/load round trip that does not
reproduce.

Also pins the CPU-generator convention. Every baseline is handed the same CPU
`torch.Generator` so one seed means the same thing across methods; sampling
with a CUDA generator here would silently desynchronise this model from the
others and make the comparison irreproducible rather than merely different.

    python external_baselines/tests/test_maskgit_flat.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, "/media/derik/Seagate Desktop Drive/organoid_data")
from MAGVIT_project.external_baselines.common import data as bdata
from MAGVIT_project.external_baselines.maskgit_flat.model import MaskGITFlat
from MAGVIT_project.external_baselines.maskgit_flat.prior import cosine_schedule
from MAGVIT_project.external_baselines.maskgit_flat.tokenizer import GRID

FAIL = []


def chk(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAIL.append(name)


def main() -> int:
    # -- mask schedule, before touching the GPU ------------------------
    print("\nmask schedule")
    chk("gamma(0) = 1 (all masked at the start)",
        abs(float(cosine_schedule(torch.tensor(0.0))) - 1.0) < 1e-6)
    chk("gamma(1) = 0 (nothing masked at the end)",
        abs(float(cosine_schedule(torch.tensor(1.0)))) < 1e-6)
    r = torch.linspace(0, 1, 32)
    chk("gamma is monotone decreasing",
        bool((cosine_schedule(r)[1:] <= cosine_schedule(r)[:-1] + 1e-7).all()))

    bdata.ensure_repo_cwd()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = MaskGITFlat(tok_epochs=2, prior_epochs=3, layers=2, d_model=128, width=64)
    m.fit(bdata.iter_raw("train", batches=3), device=dev)

    print("\nfit")
    r = m.fit_report
    chk("tokenizer ran", len(r["tokenizer_bce_per_epoch"]) == 2)
    chk("prior loss decreased",
        r["prior_ce_per_epoch"][-1] < r["prior_ce_per_epoch"][0],
        f"{r['prior_ce_per_epoch'][0]:.3f} -> {r['prior_ce_per_epoch'][-1]:.3f}")
    chk("codebook did not collapse", r["codes_used"] > 8,
        f"{r['codes_used']}/{r['n_codes']} used")

    cond, real = next(bdata.iter_split("test", batches=1))
    cond = cond.to(dev)
    g = torch.Generator().manual_seed(1)

    print("\ngeneration")
    v = m.sample(cond, generator=g)
    chk("shape matches the data", tuple(v.shape) == tuple(real.shape))
    chk("output is binary", bool(torch.all((v == 0) | (v == 1))))
    tr = float(m._target_rate(cond).mean())
    chk("rate hits the lct-predicted target",
        abs(float(v.mean()) - tr) / tr < 0.15,
        f"{float(v.mean()):.3e} vs {tr:.3e}")

    ids = m.prior.generate(cond.global_ctx, cond.local_ctx, steps=6,
                           generator=g, device=dev)
    chk("prior fills the whole grid",
        tuple(ids.shape) == (cond.batch_size, int(np.prod(GRID))))
    chk("no mask token survives decoding",
        int((ids == m.prior.mask_id).sum()) == 0,
        f"{int((ids == m.prior.mask_id).sum())} unresolved")
    chk("token ids within codebook", int(ids.max()) < m.cfg["n_codes"])
    chk("tokenize() returns the token family",
        tuple(m.tokenize(real.to(dev)).shape)
        == (real.shape[0], int(np.prod(GRID))))

    print("\npersistence and seeding")
    p = pathlib.Path(tempfile.mkdtemp()) / "mg.pt"
    m.save(p)
    m2 = MaskGITFlat()
    m2.load(p, device=dev)
    v2 = m2.sample(cond, generator=torch.Generator().manual_seed(1))
    chk("save/load reproduces bit-exactly under the same seed",
        torch.equal(v, v2), f"{int((v != v2).sum())} voxels differ")
    v3 = m.sample(cond, generator=torch.Generator().manual_seed(2))
    chk("a different seed gives a different sample", not torch.equal(v, v3))

    print(f"\n{'ALL PASS' if not FAIL else str(len(FAIL)) + ' FAILED: ' + ', '.join(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
