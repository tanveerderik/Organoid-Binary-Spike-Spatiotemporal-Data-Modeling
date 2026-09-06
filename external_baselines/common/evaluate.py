"""Score any `SpikeVolumeBaseline` under the pipeline's own generation protocol.

The deliverable is a comparative table, so every row of it has to be produced
the same way. That is enforced here rather than left to the operator:

  * `REFERENCE_PROTOCOL` mirrors the defaults `analysis/generate_regimes.py` was
    run with to produce `reports/generation_regimes_*` -- 70 test batches, 4
    samples per clip, seed 20260821, the same context bank and the same pinned
    partial-local features. Baselines do not get their own protocol.
  * The sampling loop replicates that script's ordering exactly (batch, then
    regime, then rep), because sample indices are what the cross-set paired test
    joins on.
  * `verify_against_reference()` then *checks* the result against a real model
    manifest, clip by clip, and raises if the two sets do not describe the same
    held-out clips in the same order. A table built from mismatched sets is
    worse than no table, and this is the only way to know.

Reused deliberately: `inference.generation_output.GenerationWriter`, so a
baseline writes the same tree as the pipeline and
`analysis/compare_regimes.py` can pair against it with no new statistics code.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ..common import data as bdata
from ..common.ladder import REGIMES, ContextLadder
from ..common.protocol import SpikeVolumeBaseline
from MAGVIT_project.inference.generation_output import GenerationWriter

# Must match analysis/generate_regimes.py's defaults. If that script's defaults
# change, the existing generation_regimes_* sets become incomparable to new
# baseline runs and both must be regenerated -- hence a single constant.
#
# 70, not 8. The test loader is unshuffled, so a batch count is a PREFIX of the
# split rather than a sample of it: at 8 batches these sets covered 32 clips
# from 4 of the 31 recordings. 70 batches is the whole test split. Every set
# under reports/generation_regimes_* and every baseline scored set was
# regenerated together when this changed.
REFERENCE_PROTOCOL: Dict[str, Any] = {
    "split": "test",
    "batches": 70,
    "samples_per_clip": 4,
    "seed": 20260821,
    "bank": "ckpts/context_prior.pkl",
    "partial_local": ("log_mean_firing_density", "active_site_ratio"),
    "regimes": REGIMES,
}

REFERENCE_SET = Path("reports/generation_regimes_4c_soft")


# ----------------------------------------------------------------------
# Train-set rate calibration
# ----------------------------------------------------------------------

_RATE_CACHE: Dict[str, Any] = {}


def train_assay_rates(*, batches: int = 60) -> Dict[int, float]:
    """Mean spike rate per assay over TRAIN clips. No held-out information."""
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
    _RATE_CACHE[key] = {a: float(sum(v) / len(v)) for a, v in tot.items()}
    return _RATE_CACHE[key]


def binarise_at_rate(intensity: torch.Tensor, target_rate: torch.Tensor) -> torch.Tensor:
    """Per-clip top-N thresholding so each volume hits its target rate exactly.

    Rank-based, not a shared numeric cut: score scales differ across models, so
    a common threshold would compare calibrations rather than orderings.
    """
    B = intensity.shape[0]
    flat = intensity.reshape(B, -1)
    n_vox = flat.shape[1]
    out = torch.zeros_like(flat)
    for i in range(B):
        k = max(0, min(n_vox, int(round(float(target_rate[i]) * n_vox))))
        if k:
            out[i, torch.topk(flat[i], k).indices] = 1.0
    return out.reshape(intensity.shape)


# ----------------------------------------------------------------------
# Protocol verification
# ----------------------------------------------------------------------

def _clip_keys(manifest: List[Dict], regime: str) -> List[tuple]:
    return [(r["assay"], round(float(r["real_spikes"]), 3))
            for r in manifest if r["regime"] == regime]


def verify_against_reference(
    out_root: Path,
    reference: Path = REFERENCE_SET,
    *,
    strict: bool = True,
) -> Dict[str, Any]:
    """Assert a baseline run covers the same clips, in the same order, as a
    pipeline sample set. Returns a per-regime report."""
    ref_p = Path(reference) / "manifest.json"
    if not ref_p.exists():
        return {"checked": False, "reason": f"no reference manifest at {ref_p}"}
    ref = json.loads(ref_p.read_text())
    got = json.loads((Path(out_root) / "manifest.json").read_text())

    report: Dict[str, Any] = {"checked": True, "reference": str(reference),
                              "regimes": {}, "not_in_reference": []}
    problems = []
    for regime in REGIMES:
        a, b = _clip_keys(ref, regime), _clip_keys(got, regime)

        # `local_only` is off the ladder -- a control, not a rung between the
        # others -- so `analysis/generate_regimes.py` does not produce it and no
        # pipeline set has it to align against. Requiring it here would fail
        # every baseline run made after it was added (be56772), and
        # `compare_table.py` joins on its own four-rung REGIMES, so this rung is
        # never compared anyway. Record the skip rather than pass silently: an
        # empty reference is the ONLY licence to skip, and a regime the
        # reference does have is still checked to the sample.
        if not a:
            report["not_in_reference"].append(regime)
            report["regimes"][regime] = {
                "reference_n": 0, "baseline_n": len(b), "matched": None,
                "skipped": "regime absent from the reference set",
            }
            continue

        n = min(len(a), len(b))
        same = sum(1 for i in range(n) if a[i] == b[i])
        report["regimes"][regime] = {
            "reference_n": len(a), "baseline_n": len(b), "matched": same,
        }
        if len(a) != len(b):
            problems.append(f"{regime}: n differs ({len(a)} vs {len(b)})")
        elif same != n:
            first = next(i for i in range(n) if a[i] != b[i])
            problems.append(f"{regime}: clip mismatch at index {first} "
                            f"({a[first]} vs {b[first]})")
    report["ok"] = not problems
    report["problems"] = problems
    if problems and strict:
        raise RuntimeError(
            "baseline run is not clip-aligned with " + str(reference) + ":\n  "
            + "\n  ".join(problems)
            + "\nThe comparative table joins on (regime, sample), so this must "
              "be fixed rather than noted."
        )
    return report


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------

@torch.no_grad()
def evaluate_baseline(
    baseline: SpikeVolumeBaseline,
    *,
    out_root: str | Path,
    device: str = "cuda",
    split: Optional[str] = None,
    batches: Optional[int] = None,
    samples_per_clip: Optional[int] = None,
    seed: Optional[int] = None,
    regimes: Optional[tuple] = None,
    save_video: bool = False,
    rate_calibrate: bool = True,
    verify: bool = True,
) -> Dict[str, Any]:
    """Generate across the conditioning ladder and score. Writes the full tree.

    Every keyword defaults to `REFERENCE_PROTOCOL`; overriding one produces a
    run that `verify_against_reference` will reject, which is the intent.
    """
    P = REFERENCE_PROTOCOL
    split = split or P["split"]
    batches = batches if batches is not None else P["batches"]
    reps = samples_per_clip if samples_per_clip is not None else P["samples_per_clip"]
    seed = seed if seed is not None else P["seed"]
    regimes = tuple(regimes or P["regimes"])

    out_root = Path(out_root)
    writer = GenerationWriter(out_root)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    ladder = ContextLadder(P["bank"], partial=P["partial_local"], seed=seed)

    rates = train_assay_rates() if rate_calibrate else {}
    n_intensity = 0

    for bi, (cond, real) in enumerate(bdata.iter_split(split, batches=batches)):
        cond_d = cond.to(device)
        # Once per batch, not once per regime -- four calls would silently
        # reweight the real reference statistics by a factor of four.
        writer.add_real(real)

        for regime in regimes:
            cond_r = ladder.apply(cond_d, regime)
            flags = {"global": regime != "random",
                     "local": regime == "global_full_local",
                     "visible": False}
            for rep in range(reps):
                vols = baseline.sample(cond_r, generator=gen).float().cpu()
                _check_volume(vols, real, baseline.meta.name)
                extra = [{
                    "rep": rep,
                    "real_spikes": float(real[i].sum()),
                    "generated_spikes": float(vols[i].sum()),
                    "lct_used": [float(v) for v in cond_r.local_ctx[i].cpu()],
                    "lct_true": [float(v) for v in cond.local_ctx[i].cpu()],
                } for i in range(vols.shape[0])]
                writer.add_batch(
                    regime, vols,
                    assay_names=cond.assay_name,
                    task_ids=[0] * cond.batch_size,
                    context=flags, truth_vols=real, extra=extra,
                    save_video=save_video,
                )

                if rate_calibrate:
                    inten = baseline.sample_intensity(cond_r, generator=gen)
                    if inten is not None:
                        n_intensity += 1
                        tgt = torch.tensor(
                            [rates.get(int(a), float(real[i].mean()))
                             for i, a in enumerate(cond.assay_idx.tolist())],
                            dtype=torch.float32)
                        cal = binarise_at_rate(inten.float().cpu(), tgt)
                        writer.add_batch(
                            f"{regime}__rate_cal", cal,
                            assay_names=cond.assay_name,
                            task_ids=[0] * cond.batch_size,
                            context=flags, truth_vols=real,
                            extra=[{**e, "generated_spikes": float(cal[i].sum())}
                                   for i, e in enumerate(extra)],
                            save_video=False,
                        )
        print(f"  batch {bi + 1}/{batches} done", flush=True)

    summary = writer.close()

    cfg = {
        "baseline": baseline.meta.name,
        "citation": baseline.meta.citation,
        "family": baseline.meta.family,
        "conditioning": baseline.meta.conditioning,
        "notes": baseline.meta.notes,
        "extra": baseline.meta.extra,
        "protocol": {"split": split, "batches": batches,
                     "samples_per_clip": reps, "seed": seed,
                     "regimes": list(regimes)},
        "regime_definitions": ladder.definitions(),
        "rate_calibrated_batches": n_intensity,
        "volume_shape": list(bdata.volume_shape()),
    }
    if verify:
        cfg["protocol_verification"] = verify_against_reference(out_root, strict=True)
    (out_root / "run_config.json").write_text(json.dumps(cfg, indent=2, default=float))
    return summary


def _check_volume(vols: torch.Tensor, real: torch.Tensor, name: str) -> None:
    if vols.shape != real.shape:
        raise ValueError(
            f"{name}.sample returned {tuple(vols.shape)}, expected {tuple(real.shape)}")
    u = torch.unique(vols)
    if not torch.all((u == 0) | (u == 1)):
        raise ValueError(
            f"{name}.sample must return a BINARY volume; got values {u[:8].tolist()}")
