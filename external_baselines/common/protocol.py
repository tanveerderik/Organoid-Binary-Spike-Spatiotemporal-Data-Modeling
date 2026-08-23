"""The contract every external baseline implements.

The point of this file is *enforced comparability*. A baseline receives a
`ConditioningBatch` and nothing else, so it is structurally incapable of seeing
the held-out clip it is being scored against. Leakage is the first thing a
reviewer suspects in a generative comparison; here it is impossible by
construction rather than by discipline.

Two consequences worth stating up front:

  * Every baseline gets exactly the same context the main pipeline gets --
    gct (64-d, assay level) and lct (9-d, clip level). A baseline that ignores
    one of them is making a modelling choice, and must say so in `notes`.
  * Every baseline emits the same thing: a binary (B,T,H,W) volume on the same
    grid as the real data. That is precisely the input `mea_statistics` takes,
    so scoring is model-agnostic and no baseline gets a bespoke metric path.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence

import torch


def _move(v, device):
    """Geometry may arrive as a tensor, a tuple of tensors, or plain ints."""
    if torch.is_tensor(v):
        return v.to(device)
    if isinstance(v, (list, tuple)):
        return type(v)(_move(e, device) for e in v)
    return v


@dataclass(frozen=True)
class ConditioningBatch:
    """Everything a baseline is allowed to know when generating.

    Deliberately does NOT carry `x`. If you find yourself wanting the target
    volume here, you are about to write a leak.
    """

    global_ctx: torch.Tensor          # (B, 64) assay-level context
    local_ctx: torch.Tensor           # (B, 9)  clip-level context
    assay_idx: torch.Tensor           # (B,)    long, assay identity
    shape: tuple[int, int, int]       # (T, H, W) of the volume to produce
    assay_name: Sequence[str] = ()    # provenance only, never a model input

    # Array GEOMETRY, not clip content: where this recording's electrodes sit in
    # the full pooled frame (`roi_hw`) and the symmetric padding applied to reach
    # a patch multiple (`pad_hw`). Both are fixed per assay and known before any
    # spikes are recorded, which is the same standing DG/GLM already have through
    # their train-fitted per-assay site maps. The pipeline's decoder needs them to
    # place its spatial-support bias; baselines that don't, ignore them.
    roi_hw: Any = None
    pad_hw: Any = None

    @property
    def batch_size(self) -> int:
        return int(self.global_ctx.shape[0])

    def to(self, device) -> "ConditioningBatch":
        return ConditioningBatch(
            global_ctx=self.global_ctx.to(device),
            local_ctx=self.local_ctx.to(device),
            assay_idx=self.assay_idx.to(device),
            shape=self.shape,
            assay_name=self.assay_name,
            roi_hw=_move(self.roi_hw, device),
            pad_hw=_move(self.pad_hw, device),
        )


@dataclass
class BaselineMeta:
    """Paper-facing provenance. Goes straight into the results JSON."""

    name: str
    citation: str
    family: str                       # "voxel" | "voxel+token"
    conditioning: str                 # how gct/lct enter the model, in one line
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class SpikeVolumeBaseline(abc.ABC):
    """An external generative baseline over binary spike volumes.

    Subclasses live in their own directory under `external_baselines/` and are registered
    in `external_baselines/registry.py`. Nothing in the main pipeline may import them.
    """

    meta: BaselineMeta

    # -- fitting ---------------------------------------------------------
    @abc.abstractmethod
    def fit(self, clips: Iterator[Dict[str, Any]], *, device: str) -> None:
        """Fit on TRAIN clips only.

        `clips` yields collated batches straight from the shared loader, so a
        baseline may look at `x` here -- this is training. It must never retain
        anything keyed by a val/test clip.
        """

    # -- generation ------------------------------------------------------
    @abc.abstractmethod
    @torch.no_grad()
    def sample(
        self,
        cond: ConditioningBatch,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Return a binary (B,T,H,W) float tensor. No thresholding by the caller."""

    # -- persistence -----------------------------------------------------
    @abc.abstractmethod
    def save(self, path: Path) -> None: ...

    @abc.abstractmethod
    def load(self, path: Path, *, device: str) -> None: ...

    # -- optional --------------------------------------------------------
    @torch.no_grad()
    def sample_intensity(
        self,
        cond: ConditioningBatch,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Optional[torch.Tensor]:
        """Continuous per-voxel score in (B,T,H,W), or None if unavailable.

        This exists to make the rate comparison fair. The shipped pipeline
        binarises its decoder with an F1-selected threshold and consequently
        under-produces spikes by ~25pp, while a point-process baseline emits
        spikes directly and pays no such tax. Comparing the two on `rate` then
        measures a threshold, not a model.

        Given an intensity, `common.evaluate` can re-binarise every model --
        baselines and the pipeline alike -- at a threshold calibrated to the
        TRAIN assay mean rate, which uses no held-out information and is
        available to all. Baselines that cannot produce a score return None and
        appear only in the native-binarisation row.
        """
        return None

    @torch.no_grad()
    def complete(
        self,
        cond: ConditioningBatch,
        real: torch.Tensor,
        roi: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Optional[torch.Tensor]:
        """Continuous per-voxel score INSIDE `roi`, given everything outside it.

        `real` is (B,1,T,H,W) and `roi` is a float 0/1 mask of the same shape;
        a model may read `real` only where `roi` is 0. That is not an honour
        system -- `external_baselines/tests/test_task_completion.py` re-runs
        each `complete` with the held-out region replaced by noise and requires
        a bitwise-identical result.

        Returning None means the model has NO completion mechanism, which is a
        reportable capability fact and not a failure: the harness falls back to
        free generation and labels the column, rather than inventing a
        conditioning path the method does not have. The Dichotomized Gaussian
        is the honest case -- its clip-level output is a static per-site
        probability, so it cannot use the visible remainder at all.
        """
        return None

    def tokenize(self, vols: torch.Tensor) -> Optional[torch.Tensor]:
        """Token ids for the token-family metrics (MRR/CE), or None.

        Only baselines that live in the pipeline's token space can implement
        this. Statistical baselines return None and are scored on the voxel
        family alone -- which is stated in the results table, not hidden.
        """
        return None

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.meta.name!r}>"
