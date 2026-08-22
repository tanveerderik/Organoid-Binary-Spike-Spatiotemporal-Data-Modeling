"""Second sanctioned bridge to `main`: construction of the SHIPPED model.

`common/data.py` bridges the *data*. This file bridges the *model*, and it
exists for one reason: the diagnostics in `external_baselines/diagnose.py` are
only worth reading if the shipped pipeline is measured by the same code on the
same clips. Re-implementing the generation path here to avoid the import would
give parity today and silent drift the first time Stage 4 moves, which is
exactly the failure `common/data.py`'s docstring warns about.

Everything below is construction only -- no generation logic lives here. The
generation path itself is in `pipeline_reference/model.py` and mirrors
`analysis/generate_regimes.py` call for call.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from typing import Any, Dict

import torch

_PKG_PARENT = "/media/derik/Seagate Desktop Drive/organoid_data"
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from MAGVIT_project import main as M  # noqa: E402

from . import data as bdata  # noqa: E402

_CACHE: Dict[str, Any] = {}


def build_pipeline(device: str, *, phase: str = "4b", quiet: bool = False):
    """VQ-VAE + Stage-4 priors, exactly as `analysis/generate_regimes.py` builds them.

    Returns a dict with `vqvae`, `activity_prior`, `motif_prior`, `blank_code`,
    `full_hw` and the checkpoint paths actually loaded, so the provenance lands
    in the results JSON rather than in a comment.
    """
    key = f"{device}|{phase}"
    if key in _CACHE:
        return _CACHE[key]

    bdata.ensure_repo_cwd()
    train, _, _, _ = bdata.build_loaders(verbose=not quiet)
    b0 = next(iter(train))
    _, _, T0, H0, W0 = b0["x"].shape
    full_hw = tuple(map(int, b0["full_hw"][0]))

    def _build():
        vqvae = M.make_vqvae((T0, H0, W0), device, full_spatial_size=full_hw)
        # Without the Stage-1 memory bank the decoded spatial-support term
        # silently takes a fallback branch -- same trap as generation_seed_spread.
        M.load_stage1_gct_for_eval(vqvae)
        vqvae.eval()
        prior = M._load_stage4_eval_prior(vqvae, device, phase=phase, load_motif=True)
        return vqvae, prior

    vqvae, prior = _quiet(_build) if quiet else _build()

    ship = M.CKPTS.get("motif_prior_ship")
    motif_used = ship if (ship is not None and Path(ship).exists()) \
        else M.CKPTS.get("motif_prior_best")

    out = {
        "vqvae": vqvae,
        "activity_prior": prior.activity_prior,
        "motif_prior": prior.motif_prior,
        "blank_code": int(getattr(vqvae.vq, "blank_code", -1)),
        "shape": (int(T0), int(H0), int(W0)),
        "full_hw": full_hw,
        "vq_ckpt": str(M.select_ckpt(2, prefer_best=True)),
        "motif_ckpt": str(motif_used),
        "activity_ckpt": str(M.CKPTS.get("activity_prior_best_hard_metric")),
        "best_thr_tol": float(vqvae.best_thr_tol.item()),
    }
    _CACHE[key] = out
    return out


def _quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()
