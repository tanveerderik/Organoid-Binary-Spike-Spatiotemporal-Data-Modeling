from typing import Optional, Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SparseTokenTransformerEncoder

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None







def infer_activity_coordinate_mode_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> str:
    """Infer coordinate parameterization for legacy checkpoint handling."""
    has_grid = any(key.startswith("grid_head.") for key in state_dict)
    has_axis = any(
        key.startswith(prefix)
        for key in state_dict
        for prefix in ("t_head.", "h_head.", "w_head.")
    )
    # SparseRegionActivityPrior factorizes the joint grid softmax as
    # region x within-region, so it carries no monolithic grid_head.
    has_region = any(
        key.startswith(prefix)
        for key in state_dict
        for prefix in ("region_head.", "within_head.")
    )
    if has_region and has_axis:
        raise RuntimeError(
            "Activity state dict contains both region and factorized-axis heads."
        )
    # MaskGITActivityPrior has no coordinate head at all -- it emits one Bernoulli
    # per cell, so there is nothing to factorize. It still reports "joint_dense"
    # because that is the only parameterization it is compatible with, and the
    # loader compares this against the constructed module's coordinate_mode.
    if any(key.startswith("cell_head.") or key.startswith("cell_queries")
           for key in state_dict):
        return "joint_dense"
    if has_region:
        return "joint_dense"
    if has_grid and has_axis:
        raise RuntimeError(
            "Activity state dict contains both joint and factorized coordinate heads."
        )
    if has_grid:
        return "joint_dense"
    if has_axis:
        return "factorized"
    raise RuntimeError(
        "Activity state dict contains neither grid_head nor t/h/w coordinate heads."
    )


@torch.no_grad()
def build_activity_targets_from_codes(
    codes: torch.Tensor,
    token_grid: Tuple[int, int, int],
    Kmax: int,
    blank_code: int = -1,
    predict_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Builds activity targets for the Stage 3B activity prior.

    Does NOT assign target j to query j.
    """

    Ttok, Htok, Wtok = map(int, token_grid)
    Ntok = Ttok * Htok * Wtok

    if codes.dim() != 3:
        raise ValueError(f"Expected codes (B,N,2), got {tuple(codes.shape)}")

    B, N, _ = codes.shape
    if N != Ntok:
        raise ValueError(f"codes N={N} does not match token_grid product {Ntok}")

    device = codes.device

    active = codes[..., 0].ne(blank_code)          # (B,N)
    
    if predict_mask is not None:
        if predict_mask.dim() == 3:
            predict_mask = predict_mask.squeeze(-1)
        predict_mask = predict_mask.to(device=codes.device).bool()
        if predict_mask.shape != active.shape:
            raise ValueError(
                f"predict_mask shape {tuple(predict_mask.shape)} does not match active {tuple(active.shape)}"
            )
        active_full = active.clone()
        visible_active = active_full & ~predict_mask
        active = active & predict_mask
    
    raw_count = active.sum(dim=1)                  # (B,)
    
    overflow = raw_count > int(Kmax)

    if overflow.any():
        overflow_counts = (
            raw_count[overflow]
            .detach()
            .cpu()
            .tolist()
        )
    
        largest_count = int(
            raw_count.max().item()
        )
    
        Ntok = int(
            active.shape[1]
        )
    
        raise RuntimeError(
            f"Active-token target exceeds Kmax={Kmax}. "
            f"Overflow counts={overflow_counts}; "
            f"largest count={largest_count}; "
            f"Ntok={Ntok}; "
            f"required ratio={largest_count / Ntok:.6f}. "
            "Increase Kmax and restart Stage 3B. "
            "Targets must not be silently clipped."
        )
    
    count_target = raw_count.long()

    activity_flat = active.long()                  # (B,N)  ROI-restricted targets

    # Full-clip activity and the visible-active set. Structural statistics such
    # as adjacency must be measured on the COMPOSED clip -- visible tokens plus
    # whatever the model predicts inside the ROI -- never inside the ROI alone.
    # An ROI is an arbitrary window: measuring persistence within it truncates
    # every run of activity at the window edge, so the "target rate" would track
    # mask geometry rather than the data (measured: 0.43 unmasked vs 0.25 for a
    # short noncausal span). The contexts are all full-clip, so the structural
    # target must be too.
    if predict_mask is not None:
        activity_flat_full = active_full.long()
        visible_active_flat = visible_active.long()
    else:
        activity_flat_full = activity_flat
        visible_active_flat = torch.zeros_like(activity_flat)

    active_flat_padded = torch.zeros(
        B, Kmax, device=device, dtype=torch.long
    )
    active_valid = torch.zeros(
        B, Kmax, device=device, dtype=torch.bool
    )

    t_target = torch.zeros(B, Kmax, device=device, dtype=torch.long)
    h_target = torch.zeros(B, Kmax, device=device, dtype=torch.long)
    w_target = torch.zeros(B, Kmax, device=device, dtype=torch.long)

    for b in range(B):
        flat_ids = torch.where(active[b])[0]        # deterministic THW order
        flat_ids = flat_ids[:Kmax]

        k = flat_ids.numel()
        if k == 0:
            continue

        active_flat_padded[b, :k] = flat_ids
        active_valid[b, :k] = True
        
        t = torch.div(flat_ids, Htok * Wtok, rounding_mode="floor")
        rem = flat_ids % (Htok * Wtok)
        h = torch.div(rem, Wtok, rounding_mode="floor")
        w = rem % Wtok

        t_target[b, :k] = t
        h_target[b, :k] = h
        w_target[b, :k] = w

    return {
        "count_target": count_target,              # (B,)
        "raw_count": raw_count,                    # (B,)
        "activity_flat": activity_flat,
        "activity_flat_full": activity_flat_full,
        "visible_active_flat": visible_active_flat,            # (B,N)
        "predict_mask": predict_mask if predict_mask is not None else None,
        # Padded GT active-token set. Matching happens in loss.
        "active_flat_padded": active_flat_padded,  # (B,Kmax)
        "active_valid": active_valid,              # (B,Kmax)
        "t_target": t_target,                      # (B,Kmax)
        "h_target": h_target,                      # (B,Kmax)
        "w_target": w_target,                      # (B,Kmax)
    }


def _soft_coactivation_rate(
    grid: torch.Tensor,
    shifts: Tuple[Tuple[int, int, int], ...],
) -> torch.Tensor:
    """P(a shifted neighbour is active | this token is active), per sample.

    ``grid`` is (B,T,H,W) with values in [0,1]; hard 0/1 targets are a special
    case, so predictions and targets go through identical code. Each shift
    contributes the co-activation mass over the overlapping region, normalised
    by the activity mass in that same region -- so the result is a conditional
    rate rather than a raw count, and is comparable between a sparse prediction
    and a sparse target.
    """
    num = grid.new_zeros(grid.shape[0])
    den = grid.new_zeros(grid.shape[0])
    for dt, dh, dw in shifts:
        a = grid[:,
                 max(dt, 0):grid.shape[1] + min(dt, 0),
                 max(dh, 0):grid.shape[2] + min(dh, 0),
                 max(dw, 0):grid.shape[3] + min(dw, 0)]
        b = grid[:,
                 max(-dt, 0):grid.shape[1] - max(dt, 0),
                 max(-dh, 0):grid.shape[2] - max(dh, 0),
                 max(-dw, 0):grid.shape[3] - max(dw, 0)]
        # ``b`` is the conditioning set (the token that is active); ``a`` is its
        # shifted neighbour. Normalising by b gives P(neighbour | active), which
        # is the quantity measured on the data; normalising by a would give the
        # reverse conditional.
        num = num + (a * b).sum(dim=(1, 2, 3))
        den = den + b.sum(dim=(1, 2, 3))
    return num / den.clamp_min(1e-6)




class MaskGITMotifPrior(nn.Module):
    """MaskGIT-style codebook motif prior over VQ-VAE latent tokens.

    The three-level ladder (z1, z2, z3) exists only to PRODUCE centroids during
    Stage 2A. Stage 2B sums them into one flat entry per token, so at prior time
    there is no hierarchy: each token carries a single discrete id.

    That collapses the old two-stage cascade (predict z1, then a Dirichlet over
    the K2 convex-hull coefficients of the chosen parent) into one categorical
    prediction. The alpha branch is gone: its conditional mean sat in the hull
    interior where no real residual lives, and its own mode was the only
    priorable part, so a discrete flat code carries the same information without
    the sampling pathology.

    Token structure:
        a_i : 0 blank, 1 active
        f_i : flat code id, 0..V-1, meaningful only if active

    Input special IDs:
        a_mask_id   = 2

        f_mask_id   = V
        f_null_id   = V + 1

    Output:
        logits["flat"] : (B, N, V)

    Stage 2B gives every nominal (a,b,c) triple a decodable row, so merge_map is
    total and every code the encoder can emit is in the alphabet. There is no
    out-of-vocabulary outcome to guard against.
    """

    def __init__(
        self,
        flat_codebook: Optional[torch.Tensor] = None,
        merge_map: Optional[torch.Tensor] = None,
        token_grid: Tuple[int, int, int] = (8, 8, 16),
        num_tasks: int = 4,
        gct_mapper: Optional[nn.Module] = None,
        lct_mapper: Optional[nn.Module] = None,
        gct_latent_dim: int = 32,
        lct_latent_dim: int = 32,
        d_model: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        dropout: float = 0.25,
        pad_mask: bool = False,
    ):
        super().__init__()

        if flat_codebook is None:
            raise ValueError(
                "MaskGITMotifPrior requires the frozen Stage-2B flat codebook."
            )
        flat_cb = flat_codebook.detach().float().clone()
        if flat_cb.dim() != 2:
            raise ValueError(
                f"flat_codebook must be (V, D), got {tuple(flat_cb.shape)}"
            )

        self.V = int(flat_cb.shape[0])
        self.num_tasks = int(num_tasks)
        self.d_model = int(d_model)
        self.token_grid = tuple(int(g) for g in token_grid)
        Tt, Hh, Ww = self.token_grid
        self.max_len = Tt * Hh * Ww

        self.register_buffer("flat_codebook", flat_cb, persistent=True)

        if merge_map is None:
            raise ValueError(
                "MaskGITMotifPrior requires the Stage-2B merge_map to convert "
                "(z1,z2,z3) code triples into flat ids."
            )
        self.register_buffer(
            "merge_map", merge_map.detach().long().clone(), persistent=True
        )

        # Normalized pairwise squared distances between flat entries, used by
        # the distance-weighted generation metrics. Same normalization as the
        # old z1 matrix so reported numbers stay on a comparable scale.
        flat_d2 = torch.cdist(flat_cb, flat_cb, p=2).pow(2)
        nonzero = flat_d2[flat_d2 > 0]
        norm = nonzero.mean() if nonzero.numel() else flat_d2.new_tensor(1.0)
        self.register_buffer(
            "flat_distance_matrix", flat_d2 / norm.clamp_min(1e-8), persistent=True
        )

        # activity ids
        self.a_blank_id = 0
        self.a_active_id = 1
        self.a_mask_id = 2

        # flat code ids
        self.f_mask_id = self.V
        self.f_null_id = self.V + 1

        self.ctx_len = 3
        self.use_sparse_motif_encoder = True
        self.pad_mask = bool(pad_mask)

        self.gct_mapper = gct_mapper
        self.lct_mapper = lct_mapper

        for mapper in (self.gct_mapper, self.lct_mapper):
            if mapper is not None:
                for p in mapper.parameters():
                    p.requires_grad_(False)
                mapper.eval()

        # Input embeddings for transformer input
        self.a_emb = nn.Embedding(3, d_model)               # blank, active, mask
        self.flat_emb = nn.Embedding(self.V + 2, d_model)   # codes + mask + null

        self.roi_emb = nn.Embedding(2, d_model)  # 0 visible/context, 1 ROI/predict

        self.a_to_code = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.flat_head = nn.Linear(d_model, self.V)

        # Prefix context tokens
        self.task_emb = nn.Embedding(num_tasks, d_model)
        self.gct_proj = nn.Linear(gct_latent_dim, d_model)
        self.lct_proj = nn.Linear(lct_latent_dim, d_model)

        # Factorised positions. A flat nn.Embedding(max_len) has to learn each
        # of the 1024 sites independently; t/h/w factors share statistics across
        # every site with the same coordinate, which is the structure the grid
        # actually has.
        self.pos_t = nn.Embedding(Tt, d_model)
        self.pos_h = nn.Embedding(Hh, d_model)
        self.pos_w = nn.Embedding(Ww, d_model)

        self.drop = nn.Dropout(dropout)

        self.ctx_fuse = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = SparseTokenTransformerEncoder(
            dim=d_model,
            depth=n_layer,
            num_heads=n_head,
            mlp_ratio=4.0,
            drop=dropout,
            attn_drop=dropout,
        )
        self.ln_f = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def flat_ids_from_codes(
        self, codes: torch.Tensor, blank_code: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B,N,3) ladder codes -> (flat ids, active mask)."""
        if codes.dim() != 3 or codes.size(-1) < 3:
            raise ValueError(
                f"Expected codes shape (B,N,3), got {tuple(codes.shape)}"
            )
        a, b, c = codes[..., 0].long(), codes[..., 1].long(), codes[..., 2].long()
        active = a.ne(int(blank_code))
        K2, K3 = 8, 4
        nominal = (a.clamp_min(0) * K2 + b.clamp_min(0)) * K3 + c.clamp_min(0)
        f = self.merge_map[nominal.clamp(0, self.merge_map.numel() - 1)]
        return f, active

    def _pos_embed(self, N: int, device) -> torch.Tensor:
        Tt, Hh, Ww = self.token_grid
        if N != Tt * Hh * Ww:
            raise ValueError(
                f"N={N} does not match token_grid {self.token_grid} "
                f"(= {Tt * Hh * Ww})."
            )
        idx = torch.arange(N, device=device)
        t = torch.div(idx, Hh * Ww, rounding_mode="floor")
        h = torch.div(idx, Ww, rounding_mode="floor") % Hh
        w = idx % Ww
        return (self.pos_t(t) + self.pos_h(h) + self.pos_w(w)).unsqueeze(0)

    def _map_ctx(self, mapper: Optional[nn.Module], x: torch.Tensor, proj: nn.Linear):
        if mapper is None:
            return x.to(device=proj.weight.device, dtype=proj.weight.dtype)

        with torch.no_grad():
            p = next(mapper.parameters())
            x = x.to(device=p.device, dtype=p.dtype)
            y = mapper(x)

        return y.to(device=proj.weight.device, dtype=proj.weight.dtype)

    def _build_prefix(
        self,
        global_ctx: torch.Tensor,
        local_ctx: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        if global_ctx is None:
            raise ValueError("global_ctx is required.")
        if local_ctx is None:
            raise ValueError("local_ctx is required.")
        if task_id is None:
            raise ValueError("task_id is required.")

        g = self._map_ctx(self.gct_mapper, global_ctx, self.gct_proj)
        l = self._map_ctx(self.lct_mapper, local_ctx, self.lct_proj)

        g_tok = self.gct_proj(g).unsqueeze(1)
        l_tok = self.lct_proj(l).unsqueeze(1)
        t_tok = self.task_emb(task_id.long()).unsqueeze(1)

        return torch.cat([g_tok, l_tok, t_tok], dim=1)

    def _embed_layer_streams(
        self,
        a_in,
        f_in,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-position token fusion. Keeps transformer length N, not 2N."""
        x_tok = self.a_emb(a_in) + self.flat_emb(f_in)

        if roi_mask is not None:
            if roi_mask.dim() == 3:
                roi_mask = roi_mask.squeeze(-1)
            roi_mask = roi_mask.to(device=a_in.device).bool().long()
            x_tok = x_tok + self.roi_emb(roi_mask)

        return x_tok

    def _compute_motif_logits(
        self,
        h: torch.Tensor,
        *,
        activity_ids: Optional[torch.Tensor] = None,
        activity_prob: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if activity_prob is not None:
            blank_e = self.a_emb.weight[self.a_blank_id]
            active_e = self.a_emb.weight[self.a_active_id]
            a_info = (
                (1.0 - activity_prob).unsqueeze(-1) * blank_e
                + activity_prob.unsqueeze(-1) * active_e
            )
        elif activity_ids is not None:
            a_info = self.a_emb(activity_ids.long().clamp(0, 1))
        else:
            raise ValueError("Need either activity_ids or activity_prob.")

        h_code = h + self.a_to_code(a_info)
        return {"flat": self.flat_head(h_code)}

    def _build_motif_sparse_mask(
        self,
        a_in: torch.Tensor,
        f_in: torch.Tensor,
        *,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """True = keep token in the sparse transformer."""
        if targets is not None and "active" in targets:
            active = targets["active"].to(device=a_in.device).bool()
        else:
            active = a_in.eq(self.a_active_id)

        return active | f_in.eq(self.f_mask_id)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward_with_activity_prob(
        self,
        f_in: torch.LongTensor,
        *,
        activity_prob: torch.Tensor,
        global_ctx: torch.Tensor,
        local_ctx: torch.Tensor,
        task_id: torch.Tensor,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Differentiable motif forward used when refining the activity prior."""
        B, N = f_in.shape
        if activity_prob.shape != (B, N):
            raise ValueError(
                f"activity_prob must have shape {(B, N)}, "
                f"got {tuple(activity_prob.shape)}"
            )
        if N > self.max_len:
            raise ValueError(f"N={N} exceeds max_len={self.max_len}.")

        blank_e = self.a_emb.weight[self.a_blank_id]
        active_e = self.a_emb.weight[self.a_active_id]
        a_soft_e = (
            (1.0 - activity_prob).unsqueeze(-1) * blank_e
            + activity_prob.unsqueeze(-1) * active_e
        )

        prefix = self._build_prefix(global_ctx, local_ctx, task_id)
        ctx_tok = self.ctx_fuse(prefix.mean(dim=1)).unsqueeze(1)

        x_tok = a_soft_e + self.flat_emb(f_in)

        if roi_mask is not None:
            if roi_mask.dim() == 3:
                roi_mask = roi_mask.squeeze(-1)
            roi_mask_bool = roi_mask.to(device=f_in.device).bool()
            x_tok = x_tok + self.roi_emb(roi_mask_bool.long())
        else:
            roi_mask_bool = None

        x_tok = self.drop(x_tok + self._pos_embed(N, f_in.device) + ctx_tok)

        visible_or_masked = f_in.ne(self.f_null_id)
        pred_active = activity_prob.gt(0.05)
        if roi_mask_bool is not None:
            sparse_keep = visible_or_masked | (roi_mask_bool & pred_active)
        else:
            sparse_keep = visible_or_masked | pred_active

        h, _, _, _ = self.blocks(
            tokens=x_tok, active_mask=sparse_keep, pos_embed=None, fill_value=0.0
        )
        h = self.ln_f(h)

        return self._compute_motif_logits(h, activity_prob=activity_prob)

    def forward(
        self,
        a_in: torch.LongTensor,
        f_in: torch.LongTensor,
        *,
        global_ctx: torch.Tensor,
        local_ctx: torch.Tensor,
        task_id: torch.Tensor,
        roi_mask: Optional[torch.Tensor] = None,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        loss_weights: Tuple[float, ...] = (1.0,),
        activity_prob: Optional[torch.Tensor] = None,
    ):
        """
        a_in, f_in: (B,N)

        activity_prob: optional (B,N) float in [0,1]. When given, the activity
        stream is embedded as the convex blend
        ``(1-p)*a_emb[blank] + p*a_emb[active]`` instead of the hard lookup
        ``a_emb(a_in)``, and the same blend is used for the logit head. At
        p in {0,1} the blend equals the hard lookup exactly, so passing a
        binary tensor reproduces the a_in path and passing None leaves this
        forward bit-identical to before.

        This exists so Stage 4D can hand the motif prior 4B's CALIBRATED
        probability field rather than only its thresholded map. The generation
        interface currently does ``activity.long().clamp(0,1)``
        (inference/sample_prior.py), which discards exactly the uncertainty a
        consumer needs to know which cells to trust.

        targets:
            {
                "a":            (B,N), 0/1
                "f":            (B,N), 0..V-1
                "a_loss_mask":  (B,N) bool
                "f_loss_mask":  (B,N) bool, usually predict_mask & active
            }
        """
        B, N = a_in.shape
        if f_in.shape != (B, N):
            raise ValueError("a_in and f_in must both have shape (B,N).")
        if N > self.max_len:
            raise ValueError(f"N={N} exceeds max_len={self.max_len}.")

        prefix = self._build_prefix(global_ctx, local_ctx, task_id)
        ctx_tok = self.ctx_fuse(prefix.mean(dim=1)).unsqueeze(1)

        if roi_mask is None and targets is not None:
            roi_mask = targets.get("predict_mask", None)

        if activity_prob is None:
            x_tok = self._embed_layer_streams(a_in, f_in, roi_mask=roi_mask)
        else:
            if activity_prob.shape != (B, N):
                raise ValueError(
                    f"activity_prob must have shape {(B, N)}, "
                    f"got {tuple(activity_prob.shape)}"
                )
            ap = activity_prob.to(dtype=self.a_emb.weight.dtype)
            blank_e = self.a_emb.weight[self.a_blank_id]
            active_e = self.a_emb.weight[self.a_active_id]
            x_tok = (
                (1.0 - ap).unsqueeze(-1) * blank_e + ap.unsqueeze(-1) * active_e
            ) + self.flat_emb(f_in)
            if roi_mask is not None:
                _rm = roi_mask.squeeze(-1) if roi_mask.dim() == 3 else roi_mask
                x_tok = x_tok + self.roi_emb(_rm.to(device=f_in.device).bool().long())
        x_tok = self.drop(x_tok + self._pos_embed(N, a_in.device) + ctx_tok)

        sparse_keep = self._build_motif_sparse_mask(
            a_in, f_in, targets=targets, roi_mask=roi_mask
        )

        h, _, _, _ = self.blocks(
            tokens=x_tok, active_mask=sparse_keep, pos_embed=None, fill_value=0.0
        )
        h = self.ln_f(h)

        if activity_prob is None:
            activity_ids = (
                targets["a"].long() if targets is not None else a_in.long().clamp(0, 1)
            )
            logits = self._compute_motif_logits(h, activity_ids=activity_ids)
        else:
            logits = self._compute_motif_logits(
                h, activity_prob=activity_prob.to(dtype=self.a_emb.weight.dtype)
            )

        if targets is None:
            return logits, None, {}

        f_t = targets["f"].long()
        f_loss_mask = targets.get("f_loss_mask", targets["z_loss_mask"]).bool()

        f_target = torch.where(f_loss_mask, f_t, torch.full_like(f_t, -100))
        if f_loss_mask.any():
            loss_flat = F.cross_entropy(
                logits["flat"].reshape(-1, self.V),
                f_target.reshape(-1),
                ignore_index=-100,
            )
        else:
            loss_flat = logits["flat"].sum() * 0.0

        loss = float(loss_weights[0]) * loss_flat

        with torch.no_grad():
            n_tok = f_loss_mask.sum()
            if n_tok > 0:
                pred = logits["flat"].argmax(-1)
                acc = (pred[f_loss_mask] == f_t[f_loss_mask]).float().mean()
            else:
                acc = loss_flat * 0.0

        aux = {
            "loss": loss.detach(),
            "loss_flat": loss_flat.detach(),
            "flat_acc": acc.detach(),
            "flat_loss_tokens": n_tok.detach(),
        }

        return logits, loss, aux

    # ------------------------------------------------------------------
    # targets / corruption
    # ------------------------------------------------------------------

    def make_targets_from_codes(
        self,
        codes: torch.Tensor,
        predict_mask: torch.Tensor,
        blank_code: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """
        codes: (B,N,3) ladder codes from the VQ-VAE.
        predict_mask: (B,N) or (B,N,1); True means the prior should predict here.
        """
        if predict_mask.dim() == 3:
            predict_mask = predict_mask.squeeze(-1)

        f, active = self.flat_ids_from_codes(codes, blank_code=blank_code)
        pmask = predict_mask.bool()

        return {
            "a": active.long(),
            "f": f,
            "active": active,
            "predict_mask": pmask,
            "a_loss_mask": pmask,
            "f_loss_mask": pmask & active,
            "z_loss_mask": pmask & active,
        }

    def corrupt_inputs_from_targets(
        self,
        targets: Dict[str, torch.Tensor],
        ensure_at_least_one_mask: bool = True,
        full_mask_prob: float = 0.15,
    ):
        """
        Motif masking over the single flat stream.

        Activity is supplied separately and is not corrupted here.
        """
        a = targets["a"].long()
        f = targets["f"].long().clamp(0, self.V - 1)
        active = targets["active"].bool()
        pmask = targets["predict_mask"].bool()

        B, N = a.shape
        device = a.device

        gamma = torch.rand((B, 1), device=device)

        # Sometimes train exactly on the inference starting state: everything
        # inside predict_mask masked.
        if full_mask_prob > 0:
            full_mask = torch.rand((B, 1), device=device) < float(full_mask_prob)
            gamma = torch.where(full_mask, torch.ones_like(gamma), gamma)

        valid = pmask & active
        m = (torch.rand((B, N), device=device) < gamma) & valid

        if ensure_at_least_one_mask:
            for b in range(B):
                idx_valid = torch.where(valid[b])[0]
                if idx_valid.numel() > 0 and not m[b].any():
                    idx = idx_valid[torch.randint(idx_valid.numel(), (1,), device=device)]
                    m[b, idx] = True

        f_in = f.clone()
        # Inactive positions must not expose a fake clamped code 0.
        f_in[~active] = self.f_null_id
        f_in[m] = self.f_mask_id

        targets = dict(targets)
        targets["a_loss_mask"] = m
        targets["f_loss_mask"] = m & active
        targets["z_loss_mask"] = targets["f_loss_mask"]
        targets["gamma"] = gamma.squeeze(1)

        return a.clone(), f_in, targets


class HierarchicalCodebookPrior(nn.Module):
    """
    Wrapper around:
        1. MaskGITActivityPrior
        2. MaskGITMotifPrior

    This class only routes calls.
    Training losses should remain in train_prior.py or small loss helpers.
    """

    def __init__(
        self,
        activity_prior: "MaskGITActivityPrior",
        motif_prior: MaskGITMotifPrior,
    ):
        super().__init__()
        self.activity_prior = activity_prior
        self.motif_prior = motif_prior

    def forward_activity(
        self,
        global_ctx,
        local_ctx,
        task_id,
        a_in=None,
        roi_mask=None,
        count_target=None,
        count_teacher_prob: float = 0.0,
    ):
        return self.activity_prior(
            global_ctx=global_ctx,
            local_ctx=local_ctx,
            task_id=task_id,
            a_in=a_in,
            roi_mask=roi_mask,
            count_target=count_target,
            count_teacher_prob=count_teacher_prob,
        )

    def forward_motif_teacher_forced(
        self,
        a_in: torch.LongTensor,
        f_in: torch.LongTensor,
        *,
        global_ctx: torch.Tensor,
        local_ctx: torch.Tensor,
        task_id: torch.Tensor,
        roi_mask: Optional[torch.Tensor] = None,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        loss_weights: Tuple[float, ...] = (1.0,),
    ):
        return self.motif_prior(
            a_in,
            f_in,
            global_ctx=global_ctx,
            local_ctx=local_ctx,
            task_id=task_id,
            roi_mask=roi_mask,
            targets=targets,
            loss_weights=loss_weights,
        )

    def forward_joint_soft(
        self,
        f_in: torch.LongTensor,
        *,
        global_ctx: torch.Tensor,
        local_ctx: torch.Tensor,
        task_id: torch.Tensor,
        roi_mask=None,
    ):
        """
        Differentiable activity -> motif path.

        Used for later activity-prior refinement with frozen motif prior
        and voxel/statistical biological losses.
        """

        activity_out = self.forward_activity(
            global_ctx=global_ctx,
            local_ctx=local_ctx,
            task_id=task_id,
            roi_mask=roi_mask,
        )
        
        activity_prob = self.activity_prior.soft_activity_flat(
            activity_out,
            roi_mask=roi_mask,
        )

        motif_logits = self.motif_prior.forward_with_activity_prob(
            f_in=f_in,
            activity_prob=activity_prob,
            global_ctx=global_ctx,
            local_ctx=local_ctx,
            task_id=task_id,
            roi_mask=roi_mask,
        )

        return {
            "activity_out": activity_out,
            "activity_prob": activity_prob,
            "motif_logits": motif_logits,
        }

    @torch.no_grad()
    def sample_activity(
        self,
        global_ctx,
        local_ctx,
        task_id,
        *,
        roi_mask=None,
        count_temperature=1.0,
        count_mode="expected",
        count_stochastic_round=False,
        coord_temperature=1.0,
    ):
        out = self.forward_activity(
            global_ctx,
            local_ctx,
            task_id,
            roi_mask=roi_mask,
        )
        return self.activity_prior.sample_hard_activity_gridtopk(
            out,
            count_temperature=count_temperature,
            count_mode=count_mode,
            count_stochastic_round=count_stochastic_round,
            roi_mask=roi_mask,
        )

# ======================================================================
# Dense (mask-modelling) activity prior
# ======================================================================
class MaskGITActivityPrior(nn.Module):
    """Masked-token prior over the binary activity field, decoded iteratively.

    One Bernoulli per token of the (Ttok,Htok,Wtok) grid, conditioned on
    ``global_ctx`` / ``local_ctx`` / task id and on whatever activity is already
    known. Sampling is MaskGIT-style: predict every unknown cell, commit the most
    confident fraction, re-predict the rest conditioned on what was committed.
    A single round of independent draws would be mean-field -- correct marginals,
    no clustering -- so the conditioning between rounds is what carries joint
    structure.

    This replaced a DETR-style set-prediction prior. Hungarian matching assigns
    each query one target cell per clip and the assignment changes clip to clip,
    so queries converge to blurs and the scored quantity -- their objectness-
    weighted sum -- is never measured by the objective. That model underfit its
    own training data (0.4032 assay-mean F1 against 0.6390 for the empirical
    train marginal, with a 0.003 train/test gap) and, lacking a tractable
    permutation-invariant density, could be neither scored by NLL nor sampled
    coherently.

    The trunk is retained from that design and is the part that worked: a
    SparseTokenTransformerEncoder over visible-active tokens plus always-present
    context / ROI-occupancy / CLS tokens. Only the readout differs.
    """

    def __init__(
        self,
        global_dim: int,
        local_dim: int,
        num_tasks: int,
        token_grid: Tuple[int, int, int],
        Kmax: int = 256,
        d_model: int = 256,
        n_layer: int = 4,
        n_head: int = 8,
        dropout: float = 0.1,
        region_grid: Optional[Tuple[int, int, int]] = None,
        n_decoder_layer: int = 4,
        region_extent: int = 4,
        n_maskgit_layer: int = 3,
    ):
        super().__init__()

        self.Ttok, self.Htok, self.Wtok = map(int, token_grid)
        self.Ntok = self.Ttok * self.Htok * self.Wtok
        self.Kmax = int(Kmax)
        self.d_model = int(d_model)
        # Retained because the checkpoint loader compares it against the value
        # inferred from the state dict; a per-cell field admits no other value.
        self.coordinate_mode = "joint_dense"

        self.global_proj = nn.Linear(global_dim, d_model)
        self.local_proj = nn.Linear(local_dim, d_model)
        self.task_emb = nn.Embedding(num_tasks, d_model)

        # Activity input IDs: 0 = visible blank, 1 = visible active, 2 = masked.
        self.a_mask_id = 2
        self.a_emb = nn.Embedding(3, d_model)
        self.a_pos_emb = nn.Parameter(torch.randn(self.Ntok, d_model) * 0.02)

        self.count_head = nn.Linear(d_model, Kmax + 1)

        # Region partition, used only to summarise ROI occupancy into the memory.
        # Derived from the token grid so a patch-size change does not require
        # re-tuning a hard-coded partition.
        if region_grid is None:
            region_grid = tuple(
                self._auto_region_count(span, int(region_extent))
                for span in (self.Ttok, self.Htok, self.Wtok)
            )
        self.Rt, self.Rh, self.Rw = map(int, region_grid)
        for name, span, div in (("Ttok", self.Ttok, self.Rt),
                                ("Htok", self.Htok, self.Rh),
                                ("Wtok", self.Wtok, self.Rw)):
            if div < 1 or span % div != 0:
                valid = [d for d in range(1, span + 1) if span % d == 0]
                raise ValueError(
                    f"region_grid gives {div} regions along {name}, but {div} "
                    f"does not divide {name}={span}. The partition must tile the "
                    f"token grid exactly. Valid region counts for this axis: "
                    f"{valid}. Pass region_grid=None to derive one automatically."
                )
        self.n_regions = self.Rt * self.Rh * self.Rw
        self.region_size = self.Ntok // self.n_regions

        flat = torch.arange(self.Ntok)
        fdiv = lambda a, b: torch.div(a, b, rounding_mode="floor")
        t_idx = fdiv(flat, self.Htok * self.Wtok)
        h_idx = fdiv(flat, self.Wtok) % self.Htok
        w_idx = flat % self.Wtok
        st, sh, sw = self.Ttok // self.Rt, self.Htok // self.Rh, self.Wtok // self.Rw
        self.register_buffer(
            "region_of_token",
            (fdiv(t_idx, st) * (self.Rh * self.Rw)
             + fdiv(h_idx, sh) * self.Rw + fdiv(w_idx, sw)).long(),
            persistent=False,
        )
        self.register_buffer(
            "within_of_token",
            ((t_idx % st) * (sh * sw) + (h_idx % sh) * sw + (w_idx % sw)).long(),
            persistent=False,
        )

        self.memory_encoder = SparseTokenTransformerEncoder(
            dim=d_model, depth=n_layer, num_heads=n_head,
            mlp_ratio=4.0, drop=dropout, attn_drop=dropout,
        )
        self.count_cls = nn.Parameter(torch.randn(d_model) * 0.02)
        self.roi_region_embed = nn.Parameter(torch.randn(self.n_regions, d_model) * 0.02)
        self.roi_occupancy_proj = nn.Linear(1, d_model)
        self.extra_type_embed = nn.Parameter(torch.randn(5, d_model) * 0.02)

        # One query per token: query i always reads out cell i, so there is
        # nothing to match and no permutation to resolve.
        self.cell_queries = nn.Parameter(torch.randn(self.Ntok, d_model) * 0.02)

        # Per-cell conditioning ON THE QUERY, matching what MaskGITMotifPrior
        # gives its tokens (a_emb + roi_emb).
        #
        # Without these the query for cell i is the same tensor for every clip,
        # and every clip-specific fact has to arrive through cross-attention to
        # `memory`. But memory holds only visible-ACTIVE tokens: a visible-blank
        # cell and a masked cell are both absent from it, so the model could not
        # tell which cells it was being asked to predict -- the only mask signal
        # was 16 region-occupancy tokens at 64-cell granularity. Measured
        # consequence: within-assay residual correlation with the truth was
        # +0.0043 even with 36% of the clip visible.
        #
        # Safe against leakage: maskgit_activity_loss restricts the BCE to
        # predict_mask, so visible cells are never scored and telling the query
        # "cell i is visible-active" cannot buy a free correct answer.
        #
        # Zero-initialised, so a freshly built model reproduces the old forward
        # exactly and an existing checkpoint loads unchanged.
        self.q_a_emb = nn.Embedding(3, d_model)
        self.q_roi_emb = nn.Embedding(2, d_model)
        nn.init.zeros_(self.q_a_emb.weight)
        nn.init.zeros_(self.q_roi_emb.weight)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=int(n_head), dim_feedforward=int(4 * d_model),
            dropout=float(dropout), batch_first=True, norm_first=True,
        )
        self.maskgit_decoder = nn.TransformerDecoder(layer, num_layers=int(n_maskgit_layer))
        self.maskgit_norm = nn.LayerNorm(d_model)
        self.cell_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.cell_head.bias)
        self.n_maskgit_layer = int(n_maskgit_layer)

        # Post-hoc calibration temperature, fitted on validation. Ranking metrics
        # are invariant to it; NLL is not, and NLL is what matters for a prior.
        self.register_buffer("logit_temperature", torch.ones(()))

    @staticmethod
    def _auto_region_count(span: int, target_extent: int) -> int:
        """Region count for one axis: divisor of ``span`` whose region extent is
        closest to ``target_extent``, preferring fewer regions on a tie."""
        divisors = [d for d in range(1, int(span) + 1) if int(span) % d == 0]
        return min(divisors, key=lambda r: (abs(span / r - target_extent), r))

    def _region_occupancy(
        self,
        roi_mask: Optional[torch.Tensor],
        B: int,
        device,
        dtype,
    ) -> torch.Tensor:
        """Fraction of each region that is inside the ROI. (B, n_regions)"""
        if roi_mask is None:
            # No ROI supplied means every token is a candidate.
            return torch.ones((B, self.n_regions), device=device, dtype=dtype)
        if roi_mask.dim() == 3:
            roi_mask = roi_mask.squeeze(-1)
        roi = roi_mask.to(device=device, dtype=dtype)
        occupancy = torch.zeros((B, self.n_regions), device=device, dtype=dtype)
        occupancy.index_add_(1, self.region_of_token.to(device), roi)
        return occupancy / float(self.region_size)

    def _build_memory(
        self,
        g: torch.Tensor,        # (B,D)
        l: torch.Tensor,        # (B,D)
        task: torch.Tensor,     # (B,D)
        a_in: Optional[torch.Tensor],
        roi_mask: Optional[torch.Tensor],
    ):
        B = g.shape[0]
        device = g.device
        dtype = g.dtype
        D = self.d_model

        if a_in is not None:
            if a_in.dim() == 3:
                a_in = a_in.squeeze(-1)
            a_in = a_in.to(device=device).long()
            if a_in.shape[1] != self.Ntok:
                raise ValueError(
                    f"a_in shape {tuple(a_in.shape)} does not match Ntok={self.Ntok}"
                )
            grid_tokens = self.a_emb(a_in) + self.a_pos_emb.unsqueeze(0)
            grid_tokens = grid_tokens.to(dtype=dtype)
            # Only visible-active tokens carry information beyond the ROI.
            grid_active = a_in.eq(1)
        else:
            grid_tokens = torch.zeros((B, self.Ntok, D), device=device, dtype=dtype)
            grid_active = torch.zeros((B, self.Ntok), device=device, dtype=torch.bool)

        occupancy = self._region_occupancy(roi_mask, B, device, dtype)
        roi_tokens = (
            self.roi_occupancy_proj(occupancy.unsqueeze(-1))
            + self.roi_region_embed.unsqueeze(0)
            + self.extra_type_embed[4].view(1, 1, D)
        )

        cls_token = (
            self.count_cls.view(1, 1, D).expand(B, 1, D)
            + self.extra_type_embed[3].view(1, 1, D)
        )
        ctx_tokens = torch.stack(
            [
                g + self.extra_type_embed[0],
                l + self.extra_type_embed[1],
                task + self.extra_type_embed[2],
            ],
            dim=1,
        )

        extras = torch.cat([ctx_tokens, cls_token, roi_tokens], dim=1)
        n_extra = extras.shape[1]

        tokens = torch.cat([grid_tokens, extras], dim=1)
        active = torch.cat(
            [
                grid_active,
                torch.ones((B, n_extra), device=device, dtype=torch.bool),
            ],
            dim=1,
        )

        full_out, padded, key_pad_mask, lengths = self.memory_encoder(
            tokens, active_mask=active, pos_embed=None
        )
        # Extras sit at fixed indices and are always active, so they can be read
        # back from the scattered output without tracking per-sample gather order.
        cls_out = full_out[:, self.Ntok + 3]
        return padded, key_pad_mask, cls_out, lengths

    @staticmethod
    def _expected_count_from_logits(count_logits: torch.Tensor) -> torch.Tensor:
        count_prob = F.softmax(count_logits.float(), dim=-1)
        count_values = torch.arange(
            count_prob.shape[-1],
            device=count_prob.device,
            dtype=count_prob.dtype,
        )
        return (count_prob * count_values.unsqueeze(0)).sum(dim=-1)

    @staticmethod
    def _shared_count_bias(
        event_logits_raw: torch.Tensor,
        target_count: torch.Tensor,
        *,
        iterations: int = 32,
        initial_half_width: float = 30.0,
        max_expansions: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find one shared bias per sample that matches expected event mass.

        Returns
        -------
        bias : (B,)
            Shared additive logit offset.
        residual : (B,)
            ``achieved_mass - target``, i.e. how far the solve missed. This is
            the diagnostic that was previously absent: a fixed bracket silently
            returns its own boundary whenever the required offset falls outside
            it, so the soft event mass disagrees with the count head with no
            indication that anything went wrong.

        The bracket is expanded geometrically until it contains the root. Mass
        is monotonically increasing in the offset, so bracketing is sufficient
        for bisection to converge. Expansion is capped; if the cap is reached
        without bracketing, ``residual`` reports the shortfall rather than the
        result being silently wrong.
        """
        target = target_count.to(
            device=event_logits_raw.device,
            dtype=event_logits_raw.dtype,
        ).clamp(0.0, float(event_logits_raw.shape[1]))

        def _mass(bias: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(
                event_logits_raw + bias.unsqueeze(1)
            ).sum(dim=1)

        half_width = float(initial_half_width)
        lo = event_logits_raw.new_full((event_logits_raw.shape[0],), -half_width)
        hi = event_logits_raw.new_full((event_logits_raw.shape[0],), half_width)

        # Expand until mass(lo) <= target <= mass(hi) for every sample. Each
        # pass evaluates both endpoints once and performs a single host sync,
        # and the number of passes is bounded so this cannot stall the loop.
        for _ in range(int(max_expansions)):
            lo_high = _mass(lo) > target
            hi_low = _mass(hi) < target
            if not bool((lo_high | hi_low).any()):
                break
            half_width *= 2.0
            lo = torch.where(lo_high, torch.full_like(lo, -half_width), lo)
            hi = torch.where(hi_low, torch.full_like(hi, half_width), hi)

        for _ in range(int(iterations)):
            mid = 0.5 * (lo + hi)
            too_small = _mass(mid) < target
            lo = torch.where(too_small, mid, lo)
            hi = torch.where(too_small, hi, mid)

        bias = 0.5 * (lo + hi)
        residual = _mass(bias) - target
        return bias, residual

    def select_counts(
        self,
        count_logits: torch.Tensor,
        *,
        mode: str = "expected",
        temperature: float = 1.0,
        stochastic_round: bool = False,
    ) -> torch.Tensor:
        """Convert count logits into one integer count per sample.

        ``expected`` is the stable default. It uses the probability-weighted
        count and then rounds it. ``categorical`` and ``argmax`` are retained
        for explicit stochastic or modal sampling experiments.
        """
        mode = str(mode).lower()
        if mode not in ("expected", "categorical", "argmax"):
            raise ValueError(
                f"Unsupported count selection mode={mode!r}. "
                "Use 'expected', 'categorical', or 'argmax'."
            )

        if mode == "argmax" or float(temperature) <= 0.0:
            return count_logits.argmax(dim=-1)

        scaled_logits = count_logits / max(float(temperature), 1e-6)

        if mode == "categorical":
            return torch.distributions.Categorical(logits=scaled_logits).sample()

        count_prob = F.softmax(scaled_logits, dim=-1)
        count_values = torch.arange(
            count_prob.shape[-1],
            device=count_prob.device,
            dtype=count_prob.dtype,
        )
        count_expected = (count_prob * count_values.unsqueeze(0)).sum(dim=-1)

        if stochastic_round:
            count_floor = count_expected.floor()
            count_selected = count_floor + torch.bernoulli(
                count_expected - count_floor
            )
        else:
            count_selected = count_expected.round()

        return count_selected.long()

    def coordinate_metadata(self) -> Dict[str, int | str]:
        # Standalone: this class no longer inherits from a prior that supplies a
        # base implementation, so the grid fields the checkpoint loader compares
        # against are produced here directly.
        return {
            "coordinate_mode": self.coordinate_mode,
            "Ttok": int(self.Ttok),
            "Htok": int(self.Htok),
            "Wtok": int(self.Wtok),
            "Ntok": int(self.Ntok),
            "architecture": "maskgit",
            "readout": "per_cell_bernoulli",
            "n_maskgit_layer": int(self.n_maskgit_layer),
            "n_regions": int(self.n_regions),
            "region_size": int(self.region_size),
            "region_grid": f"{self.Rt}x{self.Rh}x{self.Rw}",
        }

    def forward(
        self,
        global_ctx: torch.Tensor,
        local_ctx: torch.Tensor,
        task_id: torch.Tensor,
        a_in: Optional[torch.Tensor] = None,
        roi_mask: Optional[torch.Tensor] = None,
        count_target: Optional[torch.Tensor] = None,
        count_teacher_prob: float = 0.0,
        apply_temperature: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B = global_ctx.shape[0]
        g = self.global_proj(global_ctx)
        l = self.local_proj(local_ctx)
        task = self.task_emb(task_id.long())

        memory, memory_pad, cls_out, _ = self._build_memory(g, l, task, a_in, roi_mask)

        queries = self.cell_queries.unsqueeze(0).expand(B, -1, -1)
        if a_in is not None:
            a_q = a_in.squeeze(-1) if a_in.dim() == 3 else a_in
            queries = queries + self.q_a_emb(a_q.to(queries.device).long())
        if roi_mask is not None:
            r_q = roi_mask.squeeze(-1) if roi_mask.dim() == 3 else roi_mask
            queries = queries + self.q_roi_emb(r_q.to(queries.device).bool().long())

        h = self.maskgit_norm(
            self.maskgit_decoder(
                queries,
                memory,
                memory_key_padding_mask=memory_pad,
            )
        )
        cell_logits = self.cell_head(h).squeeze(-1)            # (B,Ntok)
        if apply_temperature:
            cell_logits = cell_logits / self.logit_temperature.clamp_min(1e-3)

        out = {
            "cell_logits": cell_logits,
            "count_logits": self.count_head(cls_out),
        }
        if roi_mask is not None:
            roi = roi_mask.squeeze(-1) if roi_mask.dim() == 3 else roi_mask
            out["roi_flat"] = roi.to(torch.bool)
        return out

    def activity_prob_flat(
        self,
        out: Dict[str, torch.Tensor],
        roi_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """(B,Ntok) probabilities, zeroed outside ROI."""
        p = torch.sigmoid(out["cell_logits"])
        if roi_mask is not None:
            roi = roi_mask.squeeze(-1) if roi_mask.dim() == 3 else roi_mask
            p = p * roi.to(device=p.device, dtype=p.dtype)
        return p

    def activity_prob_grid(
        self,
        out: Dict[str, torch.Tensor],
        roi_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        p = self.activity_prob_flat(out, roi_mask=roi_mask)
        return p.view(-1, self.Ttok, self.Htok, self.Wtok)

    @torch.no_grad()
    def sample_hard_activity_topk(
        self,
        out: Dict[str, torch.Tensor],
        roi_mask: Optional[torch.Tensor] = None,
        count_mode: str = "expected",
        count_temperature: float = 1.0,
        count_stochastic_round: bool = False,
        count_scale: float = 1.0,
    ) -> torch.Tensor:
        """Deterministic MAP-style readout: the top-K cells. Best point estimate,
        and the correct thing to score with F1 -- but it returns the SAME map for
        the same conditioning every time, so it is not a sample. Use
        ``maskgit_sample`` when you need draws from the prior."""
        counts = self.select_counts(
            out["count_logits"], mode=count_mode,
            temperature=count_temperature, stochastic_round=count_stochastic_round,
        )
        counts = self._apply_count_scale(counts, count_scale)
        score = self.activity_prob_flat(out, roi_mask=roi_mask)
        device = score.device
        B = score.shape[0]
        if roi_mask is not None:
            roi = (roi_mask.squeeze(-1) if roi_mask.dim() == 3 else roi_mask).bool()
            score = score.masked_fill(~roi, -1.0)
        else:
            roi = torch.ones_like(score, dtype=torch.bool)

        activity = torch.zeros(B, self.Ntok, device=device, dtype=torch.long)
        for b in range(B):
            nv = int(roi[b].sum().item())
            n = int(counts[b].clamp(0, min(self.Kmax, max(nv, 0))).item())
            if n <= 0:
                continue
            activity[b, torch.topk(score[b], k=n).indices] = 1
        return activity

    @staticmethod
    def _apply_count_scale(counts, count_scale: float):
        """Multiplicative recalibration of the predicted ROI count.

        The count head is unbiased under the RANDOM masks it trains on
        (count_expected_mean 39.79 vs count_target_mean 40.01 in the 4B run-2
        log) but over-predicts by +13.2% on train clips and +15.2% on val clips
        under ``deterministic_validation_task_and_masks``, whose ROI-size
        distribution it never saw and which is what validation AND generation
        use. Fitting a single scale on TRAIN clips under that same protocol --
        disjoint clips, no leakage from val -- takes held-out count MAE from
        10.500 to 8.913 and the bias from +15.2% to +2.4% at a = 0.8892. An
        affine fit (8.908) and an ROI-size covariate (8.978) add nothing over
        the one scalar.

        Default 1.0 is a no-op, so nothing changes unless a fitted value is
        passed deliberately. The scale is protocol-specific and checkpoint-
        specific: refit it whenever either changes, and note that the root fix
        is to match the training mask distribution to the generation protocol,
        which would also move placement but invalidates every existing
        comparison.
        """
        if count_scale == 1.0:
            return counts
        if count_scale <= 0.0:
            raise ValueError(f"count_scale must be positive, got {count_scale}")
        return (counts.float() * float(count_scale)).round().to(counts.dtype)

    def soft_activity_flat(self, out, roi_mask=None):
        return self.activity_prob_flat(out, roi_mask=roi_mask)

    def soft_activity_grid(self, out, clamp: bool = True, roi_mask=None):
        g = self.activity_prob_grid(out, roi_mask=roi_mask)
        return g.clamp(0.0, 1.0) if clamp else g

    def sample_hard_activity_gridtopk(
        self, out, count_temperature: float = 1.0, count_mode: str = "expected",
        count_stochastic_round: bool = False, roi_mask=None,
    ):
        return self.sample_hard_activity_topk(
            out, roi_mask=roi_mask, count_mode=count_mode,
            count_temperature=count_temperature,
            count_stochastic_round=count_stochastic_round,
        )

    @torch.no_grad()
    def sample_hard_activity_gumbel_topk(
        self,
        out: Dict[str, torch.Tensor],
        roi_mask: Optional[torch.Tensor] = None,
        count_mode: str = "expected",
        count_temperature: float = 1.0,
        count_stochastic_round: bool = False,
        tau: float = 1.0,
        count_scale: float = 1.0,
    ) -> torch.Tensor:
        """A DRAW of K distinct active cells, probability proportional to
        p^(1/tau). This is the sampling counterpart of
        ``sample_hard_activity_topk``, which is a MAP point estimate.

        Use this whenever the map feeds a generative pipeline whose output is
        scored distributionally; use the topk version for F1 and other
        agreement-with-one-target metrics. Getting this backwards makes the
        metric penalise a sharper model: a deterministic top-K over a confident
        field returns the same concentrated cells every time, so the generated
        ensemble is under-dispersed and its avalanche/ISI statistics degrade as
        the model improves. Measured on three Stage 4B checkpoints, switching
        this readout moved the generation composite +0.051 to +0.063 (t 13-14
        over 4 paired seeds), more than the entire model-to-oracle gap -- and a
        perfect oracle map read out deterministically still scored BELOW a
        trained model read out this way.

        Gumbel top-K: adding Gumbel(0,1) noise to log-probabilities and taking
        the top K is exactly sampling K distinct items without replacement with
        probability proportional to p (Plackett-Luce). tau=1.0 is the exact
        draw; tau -> 0 collapses back to ``sample_hard_activity_topk``.

        The count is taken from the same ``select_counts`` call the
        deterministic path uses, so the two readouts place an IDENTICAL number
        of cells and differ only in which. ``count_mode="categorical"`` is
        available but measured worse than ``"expected"`` here (composite 0.7287
        vs 0.7417): the count posterior's spread degrades the decoded count
        error without buying diversity that helps.
        """
        counts = self.select_counts(
            out["count_logits"], mode=count_mode,
            temperature=count_temperature, stochastic_round=count_stochastic_round,
        )
        counts = self._apply_count_scale(counts, count_scale)
        score = self.activity_prob_flat(out, roi_mask=roi_mask)
        device = score.device
        B = score.shape[0]
        if roi_mask is not None:
            roi = (roi_mask.squeeze(-1) if roi_mask.dim() == 3 else roi_mask).bool()
        else:
            roi = torch.ones_like(score, dtype=torch.bool)

        logits = score.float().clamp_min(1e-9).log() / max(float(tau), 1e-6)
        u = torch.rand_like(logits).clamp_(1e-9, 1.0 - 1e-9)
        perturbed = (logits - torch.log(-torch.log(u))).masked_fill(~roi, -1e9)

        activity = torch.zeros(B, self.Ntok, device=device, dtype=torch.long)
        for b in range(B):
            nv = int(roi[b].sum().item())
            n = int(counts[b].clamp(0, min(self.Kmax, max(nv, 0))).item())
            if n <= 0:
                continue
            activity[b, torch.topk(perturbed[b], k=n).indices] = 1
        return activity

    def sample_hard_activity_for_generation(
        self, out, roi_mask=None, readout: str = "gumbel",
        tau: float = 1.0, count_mode: str = "expected",
        count_scale: float = 1.0,
    ) -> torch.Tensor:
        """Single entry point for the generation path, so the readout is one
        named choice rather than a call site that silently means MAP.

        ``count_scale`` recalibrates the predicted ROI count; see
        ``_apply_count_scale``. It is 1.0 (no-op) unless a value fitted on train
        clips under this protocol is passed in."""
        if readout == "topk":
            return self.sample_hard_activity_topk(
                out, roi_mask=roi_mask, count_mode=count_mode,
                count_temperature=1.0, count_stochastic_round=False,
                count_scale=count_scale,
            )
        if readout == "gumbel":
            return self.sample_hard_activity_gumbel_topk(
                out, roi_mask=roi_mask, count_mode=count_mode, tau=tau,
                count_scale=count_scale,
            )
        raise ValueError(
            f"Unsupported activity readout={readout!r}. Use 'gumbel' (a draw, "
            "correct for distributional scoring) or 'topk' (MAP, correct for F1)."
        )

    @torch.no_grad()
    def maskgit_sample(
        self,
        global_ctx: torch.Tensor,
        local_ctx: torch.Tensor,
        task_id: torch.Tensor,
        a_in: Optional[torch.Tensor] = None,
        roi_mask: Optional[torch.Tensor] = None,
        n_steps: int = 10,
        temperature: float = 1.0,
        gumbel_scale: float = 1.0,
        enforce_count: bool = False,
        counts: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_trace: bool = False,
    ):
        """Iterative confidence-based decoding. Returns (B,Ntok) long 0/1.

        Each round predicts every still-unknown cell, samples it, commits only the
        most confident fraction, and re-predicts the rest conditioned on what was
        just committed. That conditioning is the entire point: a single round of
        independent Bernoulli draws is mean-field and yields speckle with correct
        marginals and no clustering or bursts. Cells decided late see the cells
        decided early, so the joint structure is carried by the model rather than
        assumed away.

        The unknown set is passed as ``roi_mask`` each round and shrinks as cells
        commit, which keeps the invariant the model was trained under:
        ``a_in == 2`` exactly where ``roi_mask`` is true.

        ``gumbel_scale`` anneals to zero across rounds. Early rounds are noisy so
        draws differ; late rounds are near-greedy so the sample stays coherent.
        With ``gumbel_scale=0`` and ``temperature -> 0`` this degenerates to
        deterministic decoding, which is a useful ablation but is not a sample.
        """
        device = global_ctx.device
        B = global_ctx.shape[0]

        if roi_mask is None:
            unknown = torch.ones(B, self.Ntok, dtype=torch.bool, device=device)
        else:
            r = roi_mask.squeeze(-1) if roi_mask.dim() == 3 else roi_mask
            unknown = r.to(device=device).bool().clone()

        if a_in is None:
            state = torch.full((B, self.Ntok), int(self.a_mask_id),
                               dtype=torch.long, device=device)
        else:
            state = a_in.to(device=device).long().clone()
        state[unknown] = int(self.a_mask_id)

        total = unknown.sum(dim=1)                    # cells to decide, per sample
        decided = torch.zeros(B, self.Ntok, dtype=torch.long, device=device)
        trace = []

        steps = max(1, int(n_steps))
        for step in range(1, steps + 1):
            if not bool(unknown.any()):
                break

            out = self.forward(
                global_ctx=global_ctx, local_ctx=local_ctx, task_id=task_id,
                a_in=state, roi_mask=unknown.to(global_ctx.dtype),
            )
            logits = out["cell_logits"].float()
            p = torch.sigmoid(logits / max(float(temperature), 1e-6))

            u = torch.rand(p.shape, device=device, generator=generator)
            draw = (u < p).long()
            conf = torch.where(draw.bool(), p, 1.0 - p)

            # Anneal the noise: 1 -> 0 across rounds.
            scale = float(gumbel_scale) * (1.0 - (step - 1) / steps)
            if scale > 0:
                gu = torch.rand(p.shape, device=device, generator=generator).clamp_(1e-9, 1 - 1e-9)
                conf = conf + scale * (-torch.log(-torch.log(gu)))
            conf = conf.masked_fill(~unknown, -float("inf"))

            # Cosine schedule on how many cells may REMAIN unknown after this round.
            keep_frac = float(math.cos(0.5 * math.pi * step / steps))
            remain = torch.ceil(total.float() * keep_frac).long()
            if step == steps:
                remain = torch.zeros_like(remain)

            for b in range(B):
                n_unk = int(unknown[b].sum().item())
                n_commit = max(0, n_unk - int(remain[b].item()))
                if n_commit <= 0:
                    continue
                idx = torch.topk(conf[b], k=n_commit).indices
                decided[b, idx] = draw[b, idx]
                state[b, idx] = draw[b, idx]        # 1 = active, 0 = visible blank
                unknown[b, idx] = False

            if return_trace:
                trace.append(int(unknown.sum().item()))

        if enforce_count:
            # Re-balance to the count head's K without discarding the sample: keep
            # the K cells the final pass ranked highest, filling from the sampled
            # actives first so the draw is respected where it can be.
            # An explicit `counts` overrides the head. Needed for evaluation:
            # scoring joint structure against real data requires the same number
            # of active cells on both sides, otherwise every rate-derived
            # statistic is confounded by count error.
            if counts is None:
                counts = self.select_counts(out["count_logits"], mode="expected")
            counts = counts.to(device=device)
            final_p = torch.sigmoid(out["cell_logits"].float())
            roi_all = (roi_mask.squeeze(-1) if roi_mask is not None and roi_mask.dim() == 3
                       else roi_mask)
            for b in range(B):
                valid = (roi_all[b].bool() if roi_all is not None
                         else torch.ones(self.Ntok, dtype=torch.bool, device=device))
                k = int(counts[b].clamp(0, int(valid.sum().item())).item())
                score = final_p[b].masked_fill(~valid, -1.0)
                score = score + decided[b].float()    # sampled actives rank first
                decided[b] = torch.zeros_like(decided[b])
                if k > 0:
                    decided[b, torch.topk(score, k=k).indices] = 1

        return (decided, trace) if return_trace else decided


def maskgit_activity_loss(
    activity_prior: "MaskGITActivityPrior",
    out: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    *,
    lambda_bce: float = 1.0,
    pos_weight: float = 1.0,
    lambda_count: float = 1.0,
    lambda_count_neighbor: float = 0.25,
    lambda_count_distance: float = 0.05,
    count_neighbor_k: int = 11,
    count_neighbor_tau: float = 2.0,
    count_distance_scale: float = 5.0,
    lambda_adj_t: float = 0.0,
    lambda_adj_s: float = 0.0,
    lambda_spatial: float = 0.0,
    allowed_support: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Loss for the per-cell Bernoulli readout.

    Two things are worth stating explicitly:

    1. ``pos_weight`` defaults to 1.0, not to the class-balancing ratio. Weighting
       up the ~5% positives speeds early training and improves F1/AUPRC, both of
       which are rank-based and blind to it -- but it inflates every probability
       and wrecks NLL. Measured: a run at pos_weight up to 30 reached the best
       AUPRC in the ladder (0.7523) while its NLL, 0.2246, was WORSE than the
       per-assay marginal's 0.1315. For a prior that gets sampled, calibration is
       the property that matters, so the default is the honest one.

    2. The auxiliary terms are meaningful here in a way they are not for the set
       model. In set prediction they act on ``soft_activity``, which is already
       marginalized over queries, so they are invariant to how the queries split
       the mass and can be satisfied by any decomposition -- which is why the
       adjacency arm moved gapMAE to a best-of-any 0.1634 and left exact F1 flat
       at 0.3943. On a per-cell field there is no decomposition: the loss acts
       directly on the predicted map.

    The count term is exact CE plus ordinal neighbour and distance terms.
    Oracle counts are worth only +0.025 exact F1 on this data, so the extra
    neighbour and distance terms are not where the remaining error lives.
    """
    cell_logits = out["cell_logits"]
    device, dtype = cell_logits.device, cell_logits.dtype
    B = cell_logits.shape[0]

    pm = targets.get("predict_mask", None)
    if pm is not None:
        roi = (pm.squeeze(-1) if pm.dim() == 3 else pm).to(device).bool()
    else:
        roi = torch.ones_like(cell_logits, dtype=torch.bool)

    # ---------------- 1. per-cell BCE, ROI only ----------------
    y = targets["activity_flat"].to(device=device, dtype=dtype)
    pw = torch.as_tensor(float(pos_weight), device=device, dtype=dtype)
    with torch.cuda.amp.autocast(enabled=False):
        el = F.binary_cross_entropy_with_logits(
            cell_logits.float(), y.float(), pos_weight=pw.float(), reduction="none"
        )
    denom = roi.float().sum().clamp_min(1.0)
    loss_bce = (el * roi.float()).sum() / denom

    # ---------------- 2. count: exact + ordered-neighbour + expected distance ----
    # Originally this was exact CE alone, justified by oracle counts being worth
    # only +0.025 exact F1. That reasoning was about PLACEMENT and does not carry
    # to generation: the sampler lights exactly K cells, so count error moves the
    # sampled rate directly and corrupts every rate-derived statistic. Measured
    # with exact CE alone, loss_count plateaued near 3.1 and sampling produced a
    # rate of 0.0197 against 0.0835 real. The ordinal terms are what make a
    # 257-way head converge -- plain CE treats "off by one" and "off by two
    # hundred" as equally wrong.
    count_logits_loss = out["count_logits"].float()
    n_bins = count_logits_loss.shape[-1]
    count_target = targets["count_target"].to(device).long().clamp(0, n_bins - 1)

    loss_count_exact = F.cross_entropy(count_logits_loss, count_target)

    count_values = torch.arange(n_bins, device=device, dtype=torch.float32)
    count_distance = (count_values.unsqueeze(0) - count_target.float().unsqueeze(1)).abs()

    nk = max(1, min(int(count_neighbor_k), n_bins))
    near_distance, near_index = torch.topk(count_distance, k=nk, largest=False, dim=-1)
    neighbor_target = torch.zeros_like(count_logits_loss).scatter(
        1, near_index,
        F.softmax(-near_distance / max(float(count_neighbor_tau), 1e-6), dim=-1),
    )
    count_log_prob = F.log_softmax(count_logits_loss, dim=-1)
    loss_count_neighbor = -(neighbor_target * count_log_prob).sum(dim=-1).mean()
    loss_count_distance = (
        count_log_prob.exp() * count_distance
    ).sum(dim=-1).mean() / max(float(count_distance_scale), 1e-6)

    loss_count = (
        loss_count_exact
        + float(lambda_count_neighbor) * loss_count_neighbor
        + float(lambda_count_distance) * loss_count_distance
    )

    # ---------------- 3. composed clip for the aux terms ----------------
    # The predicted field is zero outside the ROI and visible_active is zero
    # inside it, so the two are disjoint and their sum is a complete clip. Both
    # sides of every aux comparison are then measured on the full grid, with no
    # ROI truncation on either -- scoring a rate on a truncated support against a
    # bank accumulated over whole clips compares different quantities.
    loss_adj_t = cell_logits.sum() * 0.0
    loss_adj_s = cell_logits.sum() * 0.0
    loss_spatial = cell_logits.sum() * 0.0

    need_grid = (lambda_adj_t > 0 or lambda_adj_s > 0 or lambda_spatial > 0)
    if need_grid:
        p = torch.sigmoid(cell_logits) * roi.to(dtype)
        soft_g = p.view(B, activity_prior.Ttok, activity_prior.Htok, activity_prior.Wtok)
        vis = targets.get("visible_active_flat", None)
        if vis is not None:
            soft_g = (soft_g + vis.to(device=device, dtype=dtype).view_as(soft_g)).clamp(0.0, 1.0)
        tgt_g = targets["activity_flat_full"].to(device=device, dtype=dtype).view_as(soft_g)

    if lambda_adj_t > 0 or lambda_adj_s > 0:
        from .spatial_map import (
            TOKEN_ADJ_SHIFTS, TOKEN_ADJ_TEMPORAL_IDX, TOKEN_ADJ_SPATIAL_IDX,
        )
        s_shifts = tuple(TOKEN_ADJ_SHIFTS[i] for i in TOKEN_ADJ_SPATIAL_IDX)
        bank_t = targets.get("adj_target_t", None)
        bank_s = targets.get("adj_target_s", None)

        if lambda_adj_t > 0:
            if bank_t is not None and bank_t.dim() == 2:
                # Per band, matched lag for lag. Persistence decays across lags
                # (0.595 -> 0.429 on this data); a single pooled scalar discards
                # that curve, and because the per-lag denominators differ a mean
                # of rates is not even the pooled rate the loss would compute.
                ref = bank_t.to(device=device, dtype=dtype)
                per_band = []
                for gi in range(ref.shape[1]):
                    gap = gi + 1
                    if soft_g.shape[1] <= gap:
                        continue
                    per_band.append(
                        (_soft_coactivation_rate(soft_g, ((gap, 0, 0),)) - ref[:, gi]).abs()
                    )
                if per_band:
                    loss_adj_t = torch.stack(per_band, dim=0).mean()
            else:
                t_shifts = tuple(
                    TOKEN_ADJ_SHIFTS[i] for i in TOKEN_ADJ_TEMPORAL_IDX
                    if soft_g.shape[1] > TOKEN_ADJ_SHIFTS[i][0]
                )
                if t_shifts:
                    loss_adj_t = (
                        _soft_coactivation_rate(soft_g, t_shifts)
                        - _soft_coactivation_rate(tgt_g, t_shifts)
                    ).abs().mean()

        if lambda_adj_s > 0 and s_shifts:
            ref_s = (
                bank_s.to(device=device, dtype=dtype)
                if bank_s is not None
                else _soft_coactivation_rate(tgt_g, s_shifts)
            )
            if ref_s.dim() == 2:
                ref_s = ref_s.mean(dim=1)
            loss_adj_s = (_soft_coactivation_rate(soft_g, s_shifts) - ref_s).abs().mean()

    # ---------------- 4. spatial support consistency ----------------
    if lambda_spatial > 0:
        from ..utils.losses import spatial_support_violation_loss
        # Soft "any active in this electrode column": 1 - prod_t (1 - p).
        pred_support = 1.0 - (1.0 - soft_g.clamp(0.0, 1.0 - 1e-6)).prod(dim=1)
        allowed = allowed_support
        if allowed is None:
            allowed = F.max_pool2d(
                tgt_g.any(dim=1).float()[:, None], kernel_size=3, stride=1, padding=1
            )[:, 0]
        loss_spatial = spatial_support_violation_loss(
            pred_support, allowed.to(device=device, dtype=pred_support.dtype)
        )

    total = (
        float(lambda_bce) * loss_bce
        + float(lambda_count) * loss_count
        + float(lambda_adj_t) * loss_adj_t
        + float(lambda_adj_s) * loss_adj_s
        + float(lambda_spatial) * loss_spatial
    )
    # count_expected_mean is consumed by the Stage 3C logger, and is the
    # quantity the sampler uses for K, so it is worth watching directly: a
    # drift here moves the generated rate and with it every rate-derived
    # statistic in the generation composite.
    with torch.no_grad():
        count_expected_mean = activity_prior._expected_count_from_logits(
            out["count_logits"]
        ).float().mean()
        target_count_mean = count_target.float().mean()

    return {
        "loss": total,
        "loss_bce": loss_bce.detach(),
        "loss_count": loss_count.detach(),
        "loss_adj_t": loss_adj_t.detach(),
        "loss_adj_s": loss_adj_s.detach(),
        "loss_spatial": loss_spatial.detach(),
        "count_expected_mean": count_expected_mean,
        "count_target_mean": target_count_mean,
    }
