"""Stage 2B: flatten the three-level ladder into ONE discrete codebook, with dedupe.

The levels exist to PRODUCE centroids under the tolerance ladder. At inference
there is no hierarchy: each token carries one entry

    e_flat[f] = s0*z1[a] + s1*z2[a,b] + s2*z3[a,b,c],   f = (a*K2 + b)*K3 + c

Deterministic -- no fitting, no data beyond counting occupancy. Provenance
(a,b,c) is retained for every surviving entry so hierarchy-coloured plots and
motif graphs stay possible after flattening.

Outputs (CKPTS["stage2b_flat"]):
  embed        (M, D)   surviving flat entries, in occupancy order
  provenance   (M, 3)   the (z1,z2,z3) triple each entry came from
  flat_index   (M,)     original f, so old code sequences remap
  counts       (M,)     train-set token counts
  merge_map    (F,)     original f -> surviving row, -1 if never occupied

Promoted verbatim from the scratch script that produced the shipped
stage2b_flat_codebook.pt; only the I/O paths and the entry point changed.
"""
import json

import numpy as np
import torch

DEDUP_EPS = 0.01


@torch.no_grad()
def run_stage2b_flatten(
    model,
    train_loader,
    device,
    *,
    batch_to_device,
    out_path,
    report_path,
    source_ckpt,
    max_batches: int = 200,
):
    print("\n" + "=" * 80)
    print("STAGE 2B: flatten + dedupe the three-level ladder")
    print("=" * 80, flush=True)

    model.eval()
    K1, K2, K3 = [int(k) for k in model.vq.num_codes_per_level]
    scales = [float(x) for x in model.vq.level_scales]
    E1, E2, E3 = (model.vq.tree_embeds[i].detach().float() for i in range(3))
    F = K1 * K2 * K3

    # e_flat[(a,b,c)] = s0*E1[a] + s1*E2[a,b] + s2*E3[a,b,c]
    E = (
        scales[0] * E1[:, None, None, :]
        + scales[1] * E2[:, :, None, :]
        + scales[2] * E3
    ).reshape(F, -1)
    prov = torch.stack(
        torch.meshgrid(
            torch.arange(K1), torch.arange(K2), torch.arange(K3), indexing="ij"
        ),
        -1,
    ).reshape(F, 3)
    print(f"ladder ({K1},{K2},{K3}) scales {scales} -> flat {F} x {E.shape[1]}", flush=True)

    blank = int(getattr(model.vq, "blank_code", -1))
    counts = np.zeros(F, dtype=np.int64)
    for i, batch in enumerate(train_loader):
        x, g, l, _, _ = batch_to_device(batch, device)
        codes = model(
            x,
            global_ctx=g,
            local_ctx=l,
            predict_mask_spec=[{"type": "recon"}] * x.shape[0],
        )["codes"].long()
        z1, z2, z3 = codes[..., 0], codes[..., 1], codes[..., 2]
        m = z1 != blank
        f = ((z1[m] * K2 + z2[m]) * K3 + z3[m]).cpu().numpy()
        np.add.at(counts, f, 1)
        if i + 1 >= max_batches:
            break

    occ = counts > 0
    print(
        f"\nOCCUPANCY {int(occ.sum())}/{F} entries occur | {int(counts.sum())} tokens"
        f" | {counts.sum() / max(occ.sum(), 1):.0f} per occupied entry",
        flush=True,
    )
    nz = counts[occ]
    print(
        f"  usage min {nz.min()} median {int(np.median(nz))} max {nz.max()}"
        f" | entries <20 tokens: {int((nz < 20).sum())}"
    )

    # ---- dedupe among OCCUPIED entries: report the curve, merge at DEDUP_EPS
    Eo = E[torch.tensor(occ, device=E.device)]
    scale = float(Eo.norm(dim=1).mean())
    D = torch.cdist(Eo, Eo)
    D.fill_diagonal_(float("inf"))
    nnd = D.min(1).values
    print(
        f"\nnearest-neighbour distance: mean {nnd.mean():.4f} median {nnd.median():.4f}"
        f" (mean entry norm {scale:.4f})"
    )
    for eps in (0.005, 0.01, 0.02, 0.05, 0.10):
        print(f"  within {eps:.1%} of mean norm: {int((nnd < eps * scale).sum())}/{int(occ.sum())}")

    thr = DEDUP_EPS * scale
    order = np.argsort(-counts[occ])  # most-used first wins a collision
    idx_occ = np.where(occ)[0]
    keep, assign = [], {}
    Eo_np = Eo.cpu().numpy()
    for oi in order:
        v = Eo_np[oi]
        hit = None
        for kj in keep:
            if np.linalg.norm(v - Eo_np[kj]) < thr:
                hit = kj
                break
        if hit is None:
            keep.append(oi)
            assign[oi] = oi
        else:
            assign[oi] = hit
    keep = np.array(keep, dtype=np.int64)
    print(f"\nDEDUPE at {DEDUP_EPS:.1%}: {len(idx_occ)} occupied -> {len(keep)} distinct")

    row_of = {int(k): r for r, k in enumerate(keep)}
    merge_map = np.full(F, -1, dtype=np.int64)
    for oi, tgt in assign.items():
        merge_map[idx_occ[oi]] = row_of[int(tgt)]

    out = {
        "embed": Eo[torch.tensor(keep, device=Eo.device)].cpu(),
        "provenance": prov[torch.tensor(idx_occ[keep])].cpu(),
        "flat_index": torch.tensor(idx_occ[keep]),
        "counts": torch.tensor(counts[idx_occ[keep]]),
        "merge_map": torch.tensor(merge_map),
        "ladder": (K1, K2, K3),
        "level_scales": scales,
        "dedup_eps": DEDUP_EPS,
        "source_ckpt": str(source_ckpt),
    }
    torch.save(out, str(out_path))
    json.dump(
        {
            "F": F,
            "occupied": int(occ.sum()),
            "distinct": int(len(keep)),
            "tokens": int(counts.sum()),
            "dedup_eps": DEDUP_EPS,
            "mean_entry_norm": scale,
            "nn_dist_mean": float(nnd.mean()),
            "usage_min": int(nz.min()),
            "usage_median": int(np.median(nz)),
            "usage_max": int(nz.max()),
            "entries_under_20": int((nz < 20).sum()),
        },
        open(str(report_path), "w"),
        indent=2,
    )
    print(f"saved {out_path} and {report_path}")
    return out
