"""Score any `SpikeVolumeBaseline` on the frozen held-out split.

Deliberately reuses `inference.generation_output.GenerationWriter`, so a
baseline writes the *same tree* as the pipeline's own sample sets. That is what
lets `analysis/compare_regimes.py` run its paired cross-set test between a
baseline and the shipped model without a line of new statistics code -- same
clips, same regime names, same per-clip `vs_truth` rows, same BH-FDR family.

Two rows are produced per baseline where possible:

  native   -- the baseline binarises however it naturally does.
  rate_cal -- every model re-binarised at a threshold hitting the TRAIN assay
              mean rate. Removes the threshold artefact described in
              `protocol.SpikeVolumeBaseline.sample_intensity`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..common import data as bdata
from ..common.protocol import SpikeVolumeBaseline
from MAGVIT_project.inference.generation_output import GenerationWriter


# ----------------------------------------------------------------------
# Train-set rate calibration
# ----------------------------------------------------------------------

_RATE_CACHE: Dict[str, Any] = {}


def train_assay_rates(*, batches: int = 60) -> Dict[int, float]:
    """Mean spike rate per assay over TRAIN clips.

    Uses no held-out data, so calibrating a generation threshold against it is
    legitimate for every model in the table.
    """
    key = f"rates_{batches}"
    if key in _RATE_CACHE:
        return _RATE_CACHE[key]
    tot: Dict[int, list] = {}
    for batch in bdata.iter_raw("train", batches=batches):
        x = batch["x"]
        if x.dim() == 5:
            x = x.squeeze(1)
        x = (x > 0.5).float()
        for i, a in enumerate(batch["assay_idx"].tolist()):
            tot.setdefault(int(a), []).append(float(x[i].mean()))
    rates = {a: float(sum(v) / len(v)) for a, v in tot.items()}
    _RATE_CACHE[key] = rates
    return rates


def binarise_at_rate(
    intensity: torch.Tensor,          # (B,T,H,W) continuous score
    target_rate: torch.Tensor,        # (B,) desired fraction of active voxels
) -> torch.Tensor:
    """Per-clip top-N thresholding so each volume hits its target rate exactly.

    Rank-based rather than a global cut: the score scales differ across models,
    and a shared numeric threshold would compare calibrations, not orderings.
    """
    B = intensity.shape[0]
    flat = intensity.reshape(B, -1)
    n_vox = flat.shape[1]
    out = torch.zeros_like(flat)
    for i in range(B):
        k = int(round(float(target_rate[i]) * n_vox))
        k = max(0, min(n_vox, k))
        if k == 0:
            continue
        idx = torch.topk(flat[i], k).indices
        out[i, idx] = 1.0
    return out.reshape(intensity.shape)


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------

@torch.no_grad()
def evaluate_baseline(
    baseline: SpikeVolumeBaseline,
    *,
    split: str = "test",
    batches: int = 16,
    out_root: str | Path,
    device: str = "cuda",
    seed: int = 20260821,
    regime: str = "global_full_local",
    save_video: bool = False,
    rate_calibrate: bool = True,
) -> Dict[str, Any]:
    """Generate and score. Returns the summary dict; writes the full tree."""
    out_root = Path(out_root)
    writer = GenerationWriter(out_root)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    rates = train_assay_rates() if rate_calibrate else {}
    n_intensity = 0

    for cond, real in bdata.iter_split(split, batches=batches):
        cond_d = cond.to(device)
        writer.add_real(real)

        vols = baseline.sample(cond_d, generator=gen).float().cpu()
        _check_volume(vols, real, baseline.meta.name)
        writer.add_batch(
            regime, vols,
            assay_names=cond.assay_name,
            task_ids=[-1] * cond.batch_size,
            context={"global": True, "local": True, "visible": False},
            truth_vols=real,
            save_video=save_video,
        )

        if rate_calibrate:
            inten = baseline.sample_intensity(cond_d, generator=gen)
            if inten is not None:
                n_intensity += 1
                tgt = torch.tensor(
                    [rates.get(int(a), float(real[i].mean()))
                     for i, a in enumerate(cond.assay_idx.tolist())],
                    dtype=torch.float32,
                )
                cal = binarise_at_rate(inten.float().cpu(), tgt)
                writer.add_batch(
                    f"{regime}__rate_cal", cal,
                    assay_names=cond.assay_name,
                    task_ids=[-1] * cond.batch_size,
                    context={"global": True, "local": True, "visible": False},
                    truth_vols=real,
                    save_video=False,
                )

    summary = writer.close()

    cfg = {
        "baseline": baseline.meta.name,
        "citation": baseline.meta.citation,
        "family": baseline.meta.family,
        "conditioning": baseline.meta.conditioning,
        "notes": baseline.meta.notes,
        "extra": baseline.meta.extra,
        "split": split,
        "batches": batches,
        "seed": seed,
        "regime": regime,
        "rate_calibrated_batches": n_intensity,
        "volume_shape": list(bdata.volume_shape()),
    }
    (out_root / "run_config.json").write_text(json.dumps(cfg, indent=2, default=float))
    return summary


def _check_volume(vols: torch.Tensor, real: torch.Tensor, name: str) -> None:
    if vols.shape != real.shape:
        raise ValueError(
            f"{name}.sample returned {tuple(vols.shape)}, expected {tuple(real.shape)}"
        )
    u = torch.unique(vols)
    if not torch.all((u == 0) | (u == 1)):
        raise ValueError(
            f"{name}.sample must return a BINARY volume; got values {u[:8].tolist()}"
        )
