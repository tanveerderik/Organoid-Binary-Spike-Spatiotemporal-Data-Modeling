"""Stage 2B: flatten the three-level ladder into ONE discrete codebook, with dedupe.

The levels exist to PRODUCE centroids under the tolerance ladder. At inference
there is no hierarchy: each token carries one entry

    e_flat[f] = s0*z1[a] + s1*z2[a,b] + s2*z3[a,b,c],   f = (a*K2 + b)*K3 + c

The ONLY reduction this stage performs is merging entries whose flat sums
coincide. Every nominal triple keeps a decodable row otherwise, so any code the
encoder can emit can be decoded -- there is no out-of-vocabulary outcome.

Duplicate criterion, matching the one Stage 2A uses for duplicate restarts
(``TreeVQ.duplicate_rel_dist_thresh``, model/base.py:592):

    rel_dist(i,j) = ||e_i - e_j|| / (0.5 * (||e_i|| + ||e_j||))

Pairwise-relative, normalised by the two entries' OWN norms. A global scale
(mean norm over the codebook) is wrong here because entry norms span more than
an order of magnitude -- rarely-used leaves are EMA quotients with tiny
denominators and carry norms ~7, while heavily-used ones sit near ~1.6. A global
threshold is simultaneously far too tight for the former and too loose for the
latter.

Counting is retained for diagnostics and to decide which entry survives a
collision (most-used wins). It is NOT a filter: a low count is not evidence that
an entry is invalid, and the occupancy of a finite pass is not a property of the
codebook.

Outputs (CKPTS["stage2b_flat"]):
  embed        (M, D)   surviving flat entries, in nominal (a,b,c) order
  provenance   (M, 3)   the (z1,z2,z3) triple each entry came from
  flat_index   (M,)     original f, so old code sequences remap
  counts       (M,)     train-set token counts (diagnostic)
  merge_map    (F,)     original f -> surviving row; always >= 0
"""
import json

import numpy as np
import torch

# Matches TreeVQ.duplicate_rel_dist_thresh used by Stage 2A duplicate restarts.
DUP_REL_DIST_THRESH = 0.05

# Thresholds reported so the choice above is auditable rather than asserted.
_REPORT_THRESHOLDS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)


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

    # ---- counting: diagnostics + collision tie-break only, never a filter
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

    seen = int((counts > 0).sum())
    print(
        f"\nUSAGE (diagnostic) {seen}/{F} entries seen in {int(counts.sum())} tokens"
        f" | min {counts.min()} median {int(np.median(counts))} max {counts.max()}",
        flush=True,
    )

    # ---- dedupe over ALL entries, pairwise-relative distance (Stage 2A criterion)
    x = E.to(device).float()
    x_norm = x.norm(dim=1).clamp_min(1e-8)
    rel = torch.cdist(x, x) / (0.5 * (x_norm[:, None] + x_norm[None, :])).clamp_min(1e-8)
    rel.fill_diagonal_(float("inf"))

    nn_rel = rel.min(1).values
    print(
        f"\nnearest-neighbour RELATIVE distance: min {nn_rel.min():.4f} "
        f"median {nn_rel.median():.4f} max {nn_rel.max():.4f}"
    )
    print("  collision curve (entries with a neighbour closer than t):")
    for t in _REPORT_THRESHOLDS:
        mark = "  <-- selected" if abs(t - DUP_REL_DIST_THRESH) < 1e-12 else ""
        print(f"    rel < {t:<6.3f} : {int((nn_rel < t).sum()):4d}/{F}{mark}")

    # Greedy merge, most-used entry wins a collision. Ties broken by nominal
    # index so the result is deterministic.
    order = np.lexsort((np.arange(F), -counts))
    rel_cpu = rel.cpu()
    keep, assign = [], {}
    for i in order:
        i = int(i)
        if keep:
            k_idx = torch.tensor(keep)
            d = rel_cpu[i, k_idx]
            j = int(d.argmin())
            if float(d[j]) < DUP_REL_DIST_THRESH:
                assign[i] = keep[j]
                continue
        keep.append(i)
        assign[i] = i

    keep = np.sort(np.array(keep, dtype=np.int64))   # nominal order
    merged = F - len(keep)
    print(
        f"\nDEDUPE at rel<{DUP_REL_DIST_THRESH}: {F} nominal -> {len(keep)} distinct "
        f"({merged} merged)"
    )

    row_of = {int(k): r for r, k in enumerate(keep)}
    merge_map = np.empty(F, dtype=np.int64)
    for i, tgt in assign.items():
        merge_map[i] = row_of[int(tgt)]
    assert (merge_map >= 0).all(), "every nominal triple must map to a row"

    kept = torch.tensor(keep)
    out = {
        "embed": E[kept].cpu(),
        "provenance": prov[kept].cpu(),
        "flat_index": kept,
        "counts": torch.tensor(counts[keep]),
        "merge_map": torch.tensor(merge_map),
        "ladder": (K1, K2, K3),
        "level_scales": scales,
        "dup_rel_dist_thresh": DUP_REL_DIST_THRESH,
        "source_ckpt": str(source_ckpt),
    }
    torch.save(out, str(out_path))
    json.dump(
        {
            "F": F,
            "distinct": int(len(keep)),
            "merged": int(merged),
            "entries_seen_in_count_pass": seen,
            "tokens": int(counts.sum()),
            "dup_rel_dist_thresh": DUP_REL_DIST_THRESH,
            "dup_criterion": "||ei-ej|| / (0.5*(||ei||+||ej||))  [Stage 2A duplicate_rel_dist_thresh]",
            "collision_curve": {
                f"{t}": int((nn_rel < t).sum()) for t in _REPORT_THRESHOLDS
            },
            "nn_rel_min": float(nn_rel.min()),
            "nn_rel_median": float(nn_rel.median()),
            "usage_min": int(counts.min()),
            "usage_median": int(np.median(counts)),
            "usage_max": int(counts.max()),
        },
        open(str(report_path), "w"),
        indent=2,
    )
    print(f"saved {out_path} and {report_path}")
    return out
