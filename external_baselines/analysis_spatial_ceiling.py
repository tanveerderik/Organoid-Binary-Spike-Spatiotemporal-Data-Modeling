"""Can gct produce the electrode map, and is that ability being used?

Three numbers on the same 32 test clips:

  1. LOOKUP CEILING   the assay's TRAIN marginal map, correlated with each test
                      clip's own map. This is precisely the information DG and
                      the GLM memorise, with no model at all -- the best any
                      per-assay lookup can do.
  2. gct HEAD         what `model.spatial_map_prior` predicts from gct alone.
                      A learned map, so it would transfer to an unseen assay.
  3. GENERATED        what our samples actually achieve (0.35 from diagnostics).
"""
import sys, numpy as np, torch
sys.path.insert(0, "/media/derik/Seagate Desktop Drive/organoid_data")
from MAGVIT_project.external_baselines.common import data as bdata
from MAGVIT_project.external_baselines.common.pipeline import build_pipeline
bdata.ensure_repo_cwd()
dev = "cuda"
P = build_pipeline(dev, phase="4b", quiet=True)
vq = P["vqvae"]
print("spatial_map_prior present :", vq.spatial_map_prior is not None)
print("diagnostics flag enabled  :",
      getattr(vq, "use_learned_spatial_prior_diagnostics", False))

# ---- 1. train marginal per assay -------------------------------------
acc, cnt = {}, {}
for batch in bdata.iter_raw("train", batches=60):
    x = batch["x"]; x = x.squeeze(1) if x.dim() == 5 else x
    for i, a in enumerate(batch["assay_idx"].tolist()):
        m = x[i].sum(0).numpy()
        acc[a] = acc.get(a, 0) + m; cnt[a] = cnt.get(a, 0) + 1
train_map = {a: acc[a] / cnt[a] for a in acc}
print(f"train marginals for {len(train_map)} assays "
      f"from {sum(cnt.values())} clips")

r_lookup, r_head, r_head_true = [], [], []
for cond, real in bdata.iter_split("test", batches=8):
    condd = cond.to(dev)
    with torch.no_grad():
        g = vq._global_emb_only(condd.global_ctx.float())
        sp = vq.spatial_map_prior(g, grid=vq.token_grid,
                                  roi_hw=condd.roi_hw, pad_hw=condd.pad_hw)
    # hw_support is on the CLIP grid; full_hw_support is the whole pooled frame
    # before the spatial crop, so it does not align with `real`.
    key = "hw_support" if sp["hw_support"].shape[-2:] == real.shape[-2:] else "full_hw_support"
    pred = sp[key].float().cpu().numpy()
    if pred.shape[-2:] != real.shape[-2:]:
        raise SystemExit(f"no aligned map: {sp['hw_support'].shape} / "
                         f"{sp['full_hw_support'].shape} vs {real.shape}")
    for i, a in enumerate(cond.assay_idx.tolist()):
        tm = real[i].sum(0).numpy().ravel()
        if tm.std() == 0:
            continue
        if a in train_map and train_map[a].shape == real[i].shape[-2:]:
            lm = train_map[a].ravel()
            if lm.std() > 0:
                r_lookup.append(np.corrcoef(lm, tm)[0, 1])
        pm = pred[i].ravel()
        if pm.std() > 0:
            r_head.append(np.corrcoef(pm, tm)[0, 1])
            if a in train_map and train_map[a].shape == real[i].shape[-2:] and train_map[a].ravel().std() > 0:
                r_head_true.append(np.corrcoef(pm, train_map[a].ravel())[0, 1])

print(f"\n  1. LOOKUP CEILING  train assay marginal vs test clip : {np.mean(r_lookup):.4f}"
      f"   (n={len(r_lookup)})")
print(f"  2. gct HEAD        spatial_map_prior vs test clip    : {np.mean(r_head):.4f}")
print(f"     gct HEAD        spatial_map_prior vs the lookup   : {np.mean(r_head_true):.4f}"
      f"   <- how much of the map gct already carries")
print(f"  3. GENERATED       our samples, full context         : 0.3501")
print(f"     DG generated (memorised map)                      : 0.6256")
