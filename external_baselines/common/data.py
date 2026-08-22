"""The ONLY file under external_baselines/ permitted to import the main pipeline.

Rationale for the restriction: `main.py` is a 4.7k-line monolith holding the
run configuration, and the external baselines must sit on *exactly* the same
temporal split and the same volume geometry as the shipped model or the
comparison is meaningless. Re-declaring those constants here would give parity
today and silent drift in a month. So we import them -- but through one file,
so a change to `main.py` breaks one import site instead of five.

`external_baselines/tests/test_import_boundary.py` enforces both halves of the rule.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import torch

_PKG_PARENT = "/media/derik/Seagate Desktop Drive/organoid_data"
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from MAGVIT_project import main as M  # noqa: E402  (path juggling must precede)

from .protocol import ConditioningBatch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_repo_cwd() -> None:
    """Pin the process CWD to the repo root, permanently.

    Not a context manager, and not optional. `main.find_assays` globs
    `../output_data/...` relatively AND `NpzBurstDataset` stores the relative
    paths it discovered, so *loading* a clip needs the same CWD that discovery
    used -- including inside DataLoader worker processes, which inherit the CWD
    at fork time. Restoring the old directory after discovery therefore breaks
    the workers with a `FileNotFoundError` that reads like missing data.

    Making the stored paths absolute would be tidier, but that means editing
    `dataset.py` for the baselines' benefit, and the whole point of this package
    is that it cannot perturb the shipped pipeline. So: one chdir, announced.
    """
    if Path(os.getcwd()).resolve() != REPO_ROOT:
        os.chdir(REPO_ROOT)


# ----------------------------------------------------------------------
# Split-faithful loaders
# ----------------------------------------------------------------------

_CACHE: Dict[str, Any] = {}


def build_loaders(*, verbose: bool = True):
    """train/val/test loaders identical to the ones `main.main()` builds.

    Same assay discovery, same `per_assay_quota_stage12`, same seed, so
    `split_files_temporally` reproduces the 1069/426/638 file split the shipped
    numbers were measured on. Cached: the loaders are expensive to construct and
    every baseline wants the same three.
    """
    if "loaders" in _CACHE:
        return _CACHE["loaders"]

    ensure_repo_cwd()
    assay_dict = M.find_assays() if verbose else _quiet(M.find_assays)
    train, val, test, meta = M.make_loaders(
        assay_dict=assay_dict,
        assay_indices=list(assay_dict.keys()),
        per_assay_quota=M.per_assay_quota_stage12,
    )
    if not assay_dict:
        raise RuntimeError(f"no assays discovered from {REPO_ROOT}")
    _CACHE["loaders"] = (train, val, test, meta)
    return _CACHE["loaders"]


def _quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def loader_for(split: str):
    train, val, test, _ = build_loaders()
    return {"train": train, "val": val, "test": test}[split]


def volume_shape() -> tuple[int, int, int]:
    """(T,H,W) of a collated clip, read from the data rather than assumed."""
    if "shape" not in _CACHE:
        batch = next(iter(loader_for("train")))
        _, _, T, H, W = batch["x"].shape
        _CACHE["shape"] = (int(T), int(H), int(W))
    return _CACHE["shape"]


# ----------------------------------------------------------------------
# Batch -> (conditioning, target)
# ----------------------------------------------------------------------

def split_batch(batch: Dict[str, Any]) -> tuple[ConditioningBatch, torch.Tensor]:
    """Separate what a baseline may see from what it is scored against.

    Returns `(cond, real)` where `real` is (B,T,H,W) binary. The two are handed
    to different places on purpose: `cond` goes into `sample()`, `real` goes
    only into the metric, and nothing holds both at once inside a baseline.
    """
    x = batch["x"]
    if x.dim() == 5:                      # (B,1,T,H,W) -> (B,T,H,W)
        x = x.squeeze(1)
    real = (x > 0.5).float()
    B, T, H, W = real.shape

    cond = ConditioningBatch(
        global_ctx=batch["global_ctx"].float(),
        local_ctx=batch["local_ctx"].float(),
        assay_idx=batch["assay_idx"].long(),
        shape=(T, H, W),
        assay_name=tuple(batch.get("assay_name", ["?"] * B)),
    )
    return cond, real


def iter_split(
    split: str,
    *,
    batches: Optional[int] = None,
) -> Iterator[tuple[ConditioningBatch, torch.Tensor]]:
    """Yield `(cond, real)` pairs from a split, capped at `batches`."""
    loader = loader_for(split)
    for i, batch in enumerate(loader):
        if batches is not None and i >= batches:
            break
        yield split_batch(batch)


def iter_raw(split: str, *, batches: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Raw collated batches. For `fit()` on TRAIN only -- carries `x`."""
    loader = loader_for(split)
    for i, batch in enumerate(loader):
        if batches is not None and i >= batches:
            break
        yield batch
