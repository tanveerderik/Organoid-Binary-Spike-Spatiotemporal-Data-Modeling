#!/usr/bin/env python3
"""The `marginal` null must be a pure function of its seed.

It contains no model, so it must be identical in every row of Table 2. It was
not: over the four model runs the recon value ranged 0.0848 to 0.0892, ~5%.
Two upstream causes, both in the SHARED train loader:

  * ``shuffle=True`` with no explicit generator (dataset.py:1160), so the clip
    order follows the GLOBAL torch RNG -- which model construction has already
    advanced by a model-dependent amount;
  * ``persistent_workers=True`` (dataset.py:1143) over workers whose numpy RNG
    (dataset.py:986) draws the random temporal crop, so once anything has taken
    a train batch the clip CONTENTS drift too. ``volume_shape()``
    (common/data.py:96) takes exactly one, during baseline construction.

`train_site_maps` therefore builds its own single-process seeded loader. This
test asserts that, and asserts the control still fails -- a determinism test
that would pass on the broken code tests nothing.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from MAGVIT_project.external_baselines import task_eval as TE      # noqa: E402
from MAGVIT_project.external_baselines.common import data as bdata  # noqa: E402

SEED = 20260822
BATCHES = 8


def digest(maps) -> str:
    h = hashlib.sha256()
    for a in sorted(maps):
        h.update(str(a).encode())
        h.update(np.ascontiguousarray(maps[a], dtype=np.float64).tobytes())
    return h.hexdigest()


def unpinned_digest(batches: int) -> str:
    """The old behaviour: iterate the SHARED loader with no generator pinned."""
    loader = bdata.loader_for("train")
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "generator"):
        sampler.generator = None
    loader.generator = None
    acc = {}
    for bi, b in enumerate(bdata.iter_raw("train", batches=batches)):
        x = b["x"]
        x = x.squeeze(1) if x.dim() == 5 else x
        for i, a in enumerate(b["assay_idx"].tolist()):
            acc[a] = acc.get(a, 0) + x[i].sum(0).numpy()
    return digest(acc)


def main() -> int:
    checks = []

    def check(name, ok, detail=""):
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    torch.manual_seed(SEED)
    m1, n1 = TE.train_site_maps(BATCHES, seed=SEED)
    d1 = digest(m1)

    # perturb the global RNG the way a different model's init would, and burn
    # the persistent train-worker pool the way volume_shape() does
    torch.manual_seed(4242)
    _ = torch.randn(987_654)
    _ = bdata.volume_shape()
    _ = torch.randn(31_337)

    m2, n2 = TE.train_site_maps(BATCHES, seed=SEED)
    d2 = digest(m2)

    check("same clip count", n1 == n2, f"{n1} vs {n2}")
    check("same assays", set(m1) == set(m2), f"{len(m1)} vs {len(m2)}")
    check("map is byte-identical under a perturbed global RNG "
          "and a burned worker pool", d1 == d2, f"{d1[:16]} vs {d2[:16]}")

    m3, _ = TE.train_site_maps(BATCHES, seed=SEED + 1)
    check("a DIFFERENT seed gives a different map (the seed is live)",
          digest(m3) != d1)

    u1 = unpinned_digest(BATCHES)
    torch.manual_seed(999)
    _ = torch.randn(555_555)
    u2 = unpinned_digest(BATCHES)
    check("control: the unpinned path is still non-reproducible "
          "(so this test can actually fail)", u1 != u2)

    print(f"\n{sum(checks)}/{len(checks)} passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
