#!/usr/bin/env python3
"""Where does each arm keep its capacity: shared weights, or a per-assay table?

This is the scalability argument, and it is a property of the checkpoints
rather than of any run, so it is measured here once and written to JSON for
`diagnose_table.py` to render. Nothing is typed into prose.

The distinction that matters:

    SHARED      parameters reused for every assay. Amortised: adding an assay
                costs nothing.
    PER-ASSAY   a table indexed by assay. Adding an assay costs another row,
                so total size grows linearly with the number of preparations.

DG and the coupled GLM keep a (H, W) site map per assay. The GLM's shared part
is ~5k parameters against ~833k memorised ones, so its capacity IS the table.
Our pipeline and MaskGIT-flat keep nothing per assay: `gct` is a fixed random
+/-1 code regenerated from seed 0 (`dataset.py:403-415`), so it is a handle,
not storage.

    python external_baselines/param_census.py

Caveat recorded in the output and rendered with it: "no stored table" is not
"no memorisation". Assay-specific information certainly lives in the shared
weights of a trained model. The claim this file supports is narrower and
checkable -- parameter count does not grow with the number of assays.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CK = Path("ckpts")
EB = CK / "external_baselines"

# Ours is three checkpoints; MaskGIT-flat is one file holding two modules.
OURS = {"VQ-VAE (stage 2A)": CK / "vqvae_stage2a_best.pt",
        "motif prior": CK / "motif_prior_best.pt",
        "activity prior": CK / "activity_prior_best_hard_metric.pt"}

# Every key classified by hand. A heuristic got this wrong twice -- first by
# skipping DG's numpy/scalar parameters (reporting zero fitted values, which
# reads as rhetoric), then by counting config and fit metadata as capacity.
# Four classes:
#   per_assay  indexed by assay; grows with the number of preparations
#   fitted     genuinely learned, shared across assays
#   map        a shared lookup surface (global fallback, calibration bins) --
#              not per-assay, but not learned structure either
#   config     hyperparameters and reports; not capacity, excluded
CLASSES = {
    "dg": {"site_p": "per_assay", "assay_lograte": "per_assay",
           "rho_t": "fitted", "rho_s": "fitted", "rate_coef": "fitted",
           "global_site": "map",
           "shape": "config", "t_lags": "config", "s_radius": "config",
           "fit_report": "config"},
    "glm": {"site_logit": "per_assay",
            "state_dict": "fitted",
            "global_site": "map", "calib_binid": "map", "calib_corr": "map",
            "shape": "config", "n_lags": "config", "radius": "config",
            "ctx_dim": "config", "fit_report": "config"},
}


def _numel(o) -> int:
    """Count fitted values, including bare scalars.

    DG's fitted parameters (`rho_t`, `rho_s`, `rate_coef`) are stored as plain
    Python floats, so a tensor-only count reported DG as having zero learned
    parameters. That reads as a rhetorical exaggeration rather than a
    measurement, and it is the kind of number a reviewer checks.
    """
    if torch.is_tensor(o):
        return int(o.numel())
    if isinstance(o, np.ndarray):        # DG keeps rho_t/rho_s/rate_coef here
        return int(o.size)
    if isinstance(o, bool):
        return 0
    if isinstance(o, (int, float)):
        return 1
    if isinstance(o, dict):
        return sum(_numel(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return sum(_numel(v) for v in o)
    return 0


def _n_assays(o) -> int:
    return len(o) if isinstance(o, dict) else 0


def _budget(fr: dict) -> dict:
    """Epochs, selection point and the val curve's tail, from the fit report."""
    v = fr.get("val_roi_bce_per_epoch") or []
    ep = fr.get("epochs_run")
    be = fr.get("best_epoch")
    return {
        "epochs_run": ep,
        "best_epoch": be,
        "early_stopped": (None if (ep is None or be is None) else be < ep),
        "n_train_clips": fr.get("n_clips"),
        "best_val": fr.get("best_val_roi_bce"),
        # Mean val improvement per epoch over the last five. Near zero means
        # converged; visibly positive means the budget, not the model, set the
        # number.
        "val_slope_last5": (None if len(v) < 6 else (v[-6] - v[-1]) / 5.0),
        "fit_seconds": fr.get("fit_seconds"),
    }


def census() -> dict:
    out = {}

    shared = 0
    parts = {}
    for name, p in OURS.items():
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu")
        sd = d.get("model", d.get("state_dict", d))
        n = _numel(sd) if isinstance(sd, dict) else 0
        parts[name] = n
        shared += n
    out["pipeline"] = {"shared": shared, "per_assay": 0, "n_assays": 0,
                       "parts": parts,
                       "per_assay_note": "gct is a seeded random code "
                                         "(dataset.py:403-415), not storage"}

    # The other learned arms: one checkpoint each, every weight shared. The
    # module keys differ, so the parts are named per arm rather than guessed --
    # a missing key would otherwise report the arm as smaller than it is, which
    # is the direction that flatters us.
    NEURAL = {"maskgit_flat": ("tokenizer", "prior"),
              "unet3d": ("net",),
              "cvae3d": ("net",)}
    for name, keys in NEURAL.items():
        p = EB / f"{name}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu")
        missing = [k for k in keys if k not in d]
        if missing:
            raise RuntimeError(
                f"{name}.pt is missing weight group(s) {missing}; update NEURAL "
                "rather than under-reporting the arm")
        parts = {k: _numel(d[k]) for k in keys}
        # Split out the shared spatial embedding. It is one (H, W) surface --
        # the same KIND of object DG and the GLM are criticised for keeping --
        # and the defence is that theirs is one per assay while this is one for
        # all 31. That defence only lands if the number is visible, so it gets
        # its own line instead of hiding inside "net".
        emb = 0
        sd = d[keys[0]] if isinstance(d.get(keys[0]), dict) else {}
        for pk in ("pos", "unet.pos"):
            if pk in sd:
                n = _numel(sd[pk])
                parts[keys[0]] -= n
                parts["spatial embedding (shared, all assays)"] = n
                emb = n
                break
        out[name] = {
            "shared": sum(parts.values()), "per_assay": 0, "n_assays": 0,
            # Reported in the SAME two columns the lookup arms use, so the
            # comparison is like for like: `fitted` is learned structure,
            # `shared_maps` is a lookup surface. The whole claim is that ours is
            # one surface for all 31 preparations and theirs is one each, and
            # that claim is only checkable if both are on the same axis.
            "fitted": sum(parts.values()) - emb,
            "shared_maps": emb,
            "parts": parts,
            "per_assay_note": "same seeded random gct code as ours",
            # Training budget, so "the baseline was undertrained" is answerable
            # from the artifacts instead of from memory. `best_epoch` equal to
            # `epochs_run` means the schedule ran out before validation did --
            # the arm may still have had headroom, and the report says so.
            "budget": _budget(d.get("fit_report", {}))}

    for name in ("dg", "glm"):
        p = EB / f"{name}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu")
        cls = CLASSES[name]
        unknown = sorted(set(d) - set(cls))
        if unknown:                       # never silently misclassify capacity
            raise RuntimeError(
                f"{name}.pt has unclassified keys {unknown}; add them to "
                "CLASSES so they are not silently counted as zero")
        buckets = {"per_assay": 0, "fitted": 0, "map": 0, "config": 0}
        for k, v in d.items():
            buckets[cls[k]] += _numel(v)
        na = max((_n_assays(d[k]) for k, c in cls.items()
                  if c == "per_assay" and isinstance(d.get(k), dict)),
                 default=0)
        out[name] = {
            "shared": buckets["fitted"] + buckets["map"],
            "fitted": buckets["fitted"],
            "shared_maps": buckets["map"],
            "per_assay": (buckets["per_assay"] // na) if na else 0,
            "per_assay_total": buckets["per_assay"],
            "n_assays": na,
            "parts": {k: _numel(v) for k, v in d.items()
                      if cls[k] != "config" and _numel(v)},
            "per_assay_note": "one (H,W) site map per assay"}
    return out


def main() -> int:
    c = census()
    hdr = (f"{'arm':<16}{'fitted':>14}{'shared maps':>13}{'per assay':>12}"
           f"{'assays':>8}{'total now':>14}")
    print(hdr)
    print("-" * len(hdr))
    for k, v in c.items():
        tot = v["shared"] + v["per_assay"] * max(v["n_assays"], 0)
        print(f"{k:<16}{v.get('fitted', v['shared']):>14,}"
              f"{v.get('shared_maps', 0):>13,}{v['per_assay']:>12,}"
              f"{v['n_assays']:>8}{tot:>14,}")
    print("\nprojected per-assay storage")
    print(f"{'arm':<16}" + "".join(f"{n:>14}" for n in (31, 100, 1000)))
    for k, v in c.items():
        print(f"{k:<16}" + "".join(f"{v['per_assay'] * n:>14,}"
                                   for n in (31, 100, 1000)))

    f = Path("reports/external_baselines/param_census.json")
    f.write_text(json.dumps(
        {"arms": c,
         "projection_assays": [31, 100, 1000],
         "caveat": "Zero per-assay storage is not zero memorisation: "
                   "assay-specific information can live in shared weights. "
                   "The checkable claim is that parameter count does not grow "
                   "with the number of assays. gct is a SEEDED RANDOM code, so "
                   "a new assay still needs training exposure -- this is not a "
                   "zero-shot transfer claim."},
        indent=1))
    print(f"\nwrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
