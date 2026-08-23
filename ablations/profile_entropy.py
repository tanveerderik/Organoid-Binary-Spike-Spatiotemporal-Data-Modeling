#!/usr/bin/env python3
"""Within-token profile entropy: does the decoder concentrate mass inside a token?

The peak term exists to stop the decoder spreading probability flat inside a
token. It was measured at temporal entropy 1.699 against a log(6)=1.792 flat
ceiling, with REAL data at 0.312 -- i.e. 95% of maximum entropy against real
data's 17%. That measurement was taken with peak_radius_t=0, which never
compared across frames; the default is now 3. Whether the fix landed has not
been re-measured, and it is the mechanism behind the 6.75x over-count.
"""
import sys
from pathlib import Path
import json
import numpy as np, torch
R = Path("/media/derik/Seagate Desktop Drive/organoid_data/MAGVIT_project")
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent))
import MAGVIT_project.main as M
from MAGVIT_project.ablations.sparse_encoder import build

pT,pH,pW = M.patch_size
dev = "cuda" if torch.cuda.is_available() else "cpu"
ad = M.find_assays()
_tr, val, _te, _m = M.make_loaders(assay_dict=ad, assay_indices=list(ad.keys()),
                                   per_assay_quota=M.per_assay_quota_stage12)
b0 = next(iter(val))
model = build(tuple(b0["x"].shape[-3:]), tuple(map(int,b0["full_hw"][0])), dev,
              dense=False, seed=0)
sd = torch.load("ckpts/vqvae_stage2a_best.pt", map_location="cpu")
model.load_state_dict(sd.get("model", sd.get("state_dict", sd)), strict=False)
model.eval()

def profiles(v, grid):
    """(B,1,T,H,W) -> per-token temporal (6) and spatial (210) profiles."""
    B = v.shape[0]; t,h,w = grid
    p = (v.view(B,1,t,pT,h,pH,w,pW).permute(0,2,4,6,3,5,7,1)
          .reshape(B, t*h*w, pT, pH*pW))
    return p.sum(-1), p.sum(-2)          # (B,N,pT), (B,N,pH*pW)

def ent(x):
    x = x.clamp_min(0); s = x.sum(-1, keepdim=True)
    ok = s.squeeze(-1) > 1e-8
    q = x / s.clamp_min(1e-8)
    return (-(q * q.clamp_min(1e-12).log()).sum(-1))[ok]

MT, MS, RT, RS = [], [], [], []
with torch.no_grad():
    for i,b in enumerate(val):
        if i >= 8: break
        x = b["x"].to(dev).float()
        if x.dim()==4: x = x.unsqueeze(1)
        out = model(x, local_ctx=b["local_ctx"].to(dev).float(),
                       global_ctx=b["global_ctx"].to(dev).float())
        pr = torch.sigmoid(out["logits_vol"])
        if pr.dim()==4: pr = pr.unsqueeze(1)
        grid = out["grid"]
        act = ~model.compute_blank_mask(x, grid)        # score on CONTENT tokens
        mt, ms = profiles(pr, grid); rt, rs = profiles(x, grid)
        MT.append(ent(mt[act]).cpu()); MS.append(ent(ms[act]).cpu())
        RT.append(ent(rt[act]).cpu()); RS.append(ent(rs[act]).cpu())

cat = lambda L: torch.cat(L).numpy()
ct, cs = np.log(pT), np.log(pH*pW)
print(f"\nwithin-token profile entropy, content tokens only, n={len(cat(MT)):,}")
print(f"{'':<12}{'model':>9}{'real':>9}{'flat ceiling':>14}{'model % of flat':>17}")
print('-'*61)
for nm, m, r_, c in (("temporal", cat(MT), cat(RT), ct),
                     ("spatial",  cat(MS), cat(RS), cs)):
    print(f"{nm:<12}{m.mean():>9.3f}{r_.mean():>9.3f}{c:>14.3f}{100*m.mean()/c:>16.1f}%")
out = {"n_content_tokens": int(len(cat(MT))), "batches": 8,
       "ckpt": "ckpts/vqvae_stage2a_best.pt",
       "patch": [int(pT), int(pH), int(pW)],
       "temporal": {"model": float(cat(MT).mean()), "real": float(cat(RT).mean()),
                    "flat_ceiling": float(ct)},
       "spatial": {"model": float(cat(MS).mean()), "real": float(cat(RS).mean()),
                   "flat_ceiling": float(cs)},
       "prior_measurement_peak_radius_t0": {
           "temporal_model": 1.699, "temporal_real": 0.312,
           "temporal_ceiling": 1.792, "spatial_model": 3.19,
           "spatial_ceiling": 5.35,
           "source": "code comment, utils/losses.py:118-127"}}
f = R / "reports" / "analysis_profile_entropy.json"
f.write_text(json.dumps(out, indent=1))
print(f"\nprior measurement (peak_radius_t=0): temporal model 1.699, real 0.312, ceiling 1.792")
print(f"wrote {f}")
