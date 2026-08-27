#!/usr/bin/env python3
"""Rebuild the ablation summary from the saved checkpoints.

`ablations/sparse_encoder.py` writes to a FIXED path, so a low-`--epochs` smoke
run silently overwrites a completed one -- which is what happened to the
80-epoch sparse/dense arms. The checkpoints survive but carry weights only, so
the per-epoch curves are gone for good.

What the report actually needs from that file is `best_val_auprc` and the epoch
budget per arm; every codebook number now comes from
`ablation_sparse_encoder_content.json`, which was untouched. Those two are
recoverable by evaluating the saved `*_best.pt` on val with the SAME routine
`fit_vqvae` used for its own val metric (`evaluate_vqvae`, pos_weight=1), so
this reconstructs the file rather than retraining 7 h.

The recovered number is the best checkpoint scored once, not the max over a
training curve. Those coincide when the best epoch is the saved one, which is
what `*_best.pt` means, but the provenance is recorded either way.

    python ablations/recover_ablation_val.py --arms sparse,dense
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent))
import MAGVIT_project.main as M                                    # noqa: E402
from MAGVIT_project.ablations.sparse_encoder import build          # noqa: E402
from MAGVIT_project.training.eval_vqvae import evaluate_vqvae      # noqa: E402

CK = ROOT / "ckpts" / "ablations"
OUT = ROOT / "reports" / "ablation_sparse_encoder.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="sparse,dense")
    ap.add_argument("--out", default=None,
                    help="explicit output path. This script is the SECOND "
                         "writer of ablation_sparse_encoder.json; it merges "
                         "rather than truncating, but pass --out to keep a "
                         "recovery run away from the shipped file entirely.")
    ap.add_argument("--epochs", type=int, default=80,
                    help="budget the recovered arms were trained for")
    a = ap.parse_args()
    out_path = Path(a.out) if a.out else OUT
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ad = M.find_assays()
    _tr, val, _te, _m = M.make_loaders(assay_dict=ad,
                                       assay_indices=list(ad.keys()),
                                       per_assay_quota=M.per_assay_quota_stage12)
    b0 = next(iter(val))
    img = tuple(b0["x"].shape[-3:]); full = tuple(map(int, b0["full_hw"][0]))

    existing = (json.loads(out_path.read_text()) if out_path.is_file()
                else {"arms": {}, "summary": {}})
    summary = dict(existing.get("summary", {}))

    for name in [s.strip() for s in a.arms.split(",") if s.strip()]:
        ck = CK / f"sparse_abl_{name}_best.pt"
        if not ck.is_file():
            print(f"  {name}: no checkpoint, skipped"); continue
        model = build(img, full, dev, dense=(name == "dense"), seed=0)
        sd = torch.load(ck, map_location="cpu")
        model.load_state_dict(sd.get("model", sd.get("state_dict", sd)), strict=False)
        model.eval()
        rep = evaluate_vqvae(model, val, use_amp=True, pos_weight=1,
                             use_ROI_mask=False,
                             recon_tolerance=(2, 2, 2),
                             metric_tolerance=(2, 2, 2))
        auprc = float(rep["AUPRC"])
        summary[name] = {
            **summary.get(name, {}),
            "best_val_auprc": auprc,
            "epochs_logged": int(a.epochs),
            "recovered": True,
            "recovery_note": (
                "per-epoch curves lost when a 4-epoch smoke overwrote the "
                "fixed output path; AUPRC recomputed from *_best.pt with "
                "evaluate_vqvae, the same routine fit_vqvae used for val"),
        }
        print(f"  {name:<8} val AUPRC (exact) = {auprc:.4f}   [recovered]", flush=True)

    out = {**existing, "summary": summary}
    out_path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
