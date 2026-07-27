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



class DETRActivityPrior(nn.Module):
    """
    DETR-style sparse activity prior with factorized THW coordinate heads.

    Repository convention:
        token grid = (Ttok,Htok,Wtok)
        flat = t * Htok * Wtok + h * Wtok + w

    Outputs:
        count_logits : (B,Kmax+1)
        event_logits : (B,Kmax)
        t_logits     : (B,Kmax,Ttok)
        h_logits     : (B,Kmax,Htok)
        w_logits     : (B,Kmax,Wtok)
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
    ):
        super().__init__()

        self.Ttok, self.Htok, self.Wtok = map(int, token_grid)
        self.Ntok = self.Ttok * self.Htok * self.Wtok
        self.Kmax = int(Kmax)
        self.d_model = int(d_model)

        self.global_proj = nn.Linear(global_dim, d_model)
        self.local_proj = nn.Linear(local_dim, d_model)
        self.task_emb = nn.Embedding(num_tasks, d_model)
        
        # Activity input IDs:
        #   0 = visible blank
        #   1 = visible active
        #   2 = masked / predict this token
        self.a_mask_id = 2
        self.a_emb = nn.Embedding(3, d_model)
        
        self.a_pos_emb = nn.Parameter(torch.randn(self.Ntok, d_model) * 0.02)
        
        self.a_pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.event_queries = nn.Parameter(torch.randn(Kmax, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)

        self.count_head = nn.Linear(d_model, Kmax + 1)
        self.event_head = nn.Linear(d_model, 1)

        self.t_head = nn.Linear(d_model, self.Ttok)
        self.h_head = nn.Linear(d_model, self.Htok)
        self.w_head = nn.Linear(d_model, self.Wtok)

    
    
    def _encode_activity_input(
        self,
        a_in: Optional[torch.Tensor],
        device,
        dtype,
    ) -> torch.Tensor:
        """
        Encodes masked activity input.
    
        a_in:
            (B,N), with IDs:
                0 = visible blank
                1 = visible active
                2 = masked / predict token
    
        Returns:
            a_vec: (B,D)
        """
        if a_in is None:
            return None
    
        if a_in.dim() == 3:
            a_in = a_in.squeeze(-1)
    
        a_in = a_in.to(device=device).long()
    
        if a_in.shape[1] != self.Ntok:
            raise ValueError(
                f"a_in shape {tuple(a_in.shape)} does not match Ntok={self.Ntok}"
            )
    
        a_tok = self.a_emb(a_in)  # (B,N,D)
    
        pos = self.a_pos_emb.to(device=device, dtype=a_tok.dtype)
        a_tok = a_tok + pos.unsqueeze(0)
    
        # Pool visible + masked state together.
        # This lets the prior know both:
        #   - which activity is visible
        #   - which tokens need prediction
        a_vec = a_tok.mean(dim=1)  # (B,D)
    
        a_vec = self.a_pool(a_vec).to(dtype=dtype)
        return a_vec
        


    def apply_roi_axis_mask(
        self,
        out: Dict[str, torch.Tensor],
        roi_mask: Optional[torch.Tensor],
        fill_value: float = -1e4,
    ) -> Dict[str, torch.Tensor]:
        if roi_mask is None:
            return out
    
        if roi_mask.dim() == 3:
            roi_mask = roi_mask.squeeze(-1)
    
        B = roi_mask.shape[0]
        roi = roi_mask.to(device=out["event_logits"].device).bool()
        roi_grid = roi.view(B, self.Ttok, self.Htok, self.Wtok)
    
        valid_t = roi_grid.any(dim=3).any(dim=2)  # (B,T)
        valid_h = roi_grid.any(dim=3).any(dim=1)  # (B,H)
        valid_w = roi_grid.any(dim=2).any(dim=1)  # (B,W)
    
        out = dict(out)
        out["t_logits"] = out["t_logits"].masked_fill(~valid_t[:, None, :], fill_value)
        out["h_logits"] = out["h_logits"].masked_fill(~valid_h[:, None, :], fill_value)
        out["w_logits"] = out["w_logits"].masked_fill(~valid_w[:, None, :], fill_value)
    
        return out
    
    

    def forward(
        self,
        global_ctx: torch.Tensor,  # (B,G)
        local_ctx: torch.Tensor,   # (B,L)
        task_id: torch.Tensor,     # (B,)
        a_in: Optional[torch.Tensor] = None,
        roi_mask: Optional[torch.Tensor] = None,  # kept only for backward compatibility / axis masking
    ) -> Dict[str, torch.Tensor]:
    
        B = global_ctx.shape[0]
    
        g = self.global_proj(global_ctx).unsqueeze(1)      # (B,1,D)
        l = self.local_proj(local_ctx).unsqueeze(1)        # (B,1,D)
        task = self.task_emb(task_id.long()).unsqueeze(1)  # (B,1,D)
    
        a_vec = self._encode_activity_input(
            a_in=a_in,
            device=global_ctx.device,
            dtype=global_ctx.dtype,
        )
    
        if a_vec is None:
            a_tok = torch.zeros_like(g)
        else:
            a_tok = a_vec.unsqueeze(1)                     # (B,1,D)
    
        q = self.event_queries.unsqueeze(0).expand(B, -1, -1)  # (B,K,D)
    
        # Event queries are conditioned on visible/masked activity state.
        q = q + a_tok
    
        x = torch.cat([g, l, task, a_tok, q], dim=1)       # (B,4+K,D)
        h = self.encoder(x)
    
        ctx_h = h[:, :4].mean(dim=1)                      # (B,D)
        ev_h = h[:, 4:]                                   # (B,K,D)
    
        out = {
            "count_logits": self.count_head(ctx_h),
            "event_logits": self.event_head(ev_h).squeeze(-1),
            "t_logits": self.t_head(ev_h),
            "h_logits": self.h_head(ev_h),
            "w_logits": self.w_head(ev_h),
        }
    
        # Optional: still use predict mask for axis restriction.
        # You can remove this later if you want pure a_in-only model input.
        return self.apply_roi_axis_mask(out, roi_mask)

    def soft_activity_grid(
        self,
        out: Dict[str, torch.Tensor],
        clamp: bool = True,
    ) -> torch.Tensor:
        """
        Converts factorized event coordinate distributions to soft activity grid.

        Returns:
            activity_grid: (B,Ttok,Htok,Wtok)
        """

        event_p = torch.sigmoid(out["event_logits"])         # (B,K)

        pt = F.softmax(out["t_logits"], dim=-1)              # (B,K,T)
        ph = F.softmax(out["h_logits"], dim=-1)              # (B,K,H)
        pw = F.softmax(out["w_logits"], dim=-1)              # (B,K,W)

        # Equivalent to summing event_p[k] * pt[k,t] * ph[k,h] * pw[k,w]
        activity = (
            event_p[:, :, None, None, None]
            * pt[:, :, :, None, None]
            * ph[:, :, None, :, None]
            * pw[:, :, None, None, :]
        ).sum(dim=1)                                        # (B,T,H,W)

        return activity.clamp(0.0, 1.0) if clamp else activity

    def soft_activity_flat(
        self,
        out: Dict[str, torch.Tensor],
        roi_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            activity_flat: (B,Ntok), THW-flattened
        """
        flat = self.soft_activity_grid(out, clamp=False).reshape(
            out["event_logits"].shape[0],
            self.Ntok,
        )
        
        if roi_mask is not None:
            if roi_mask.dim() == 3:
                roi_mask = roi_mask.squeeze(-1)
            roi_mask = roi_mask.to(device=flat.device, dtype=flat.dtype)
            flat = flat * roi_mask
        
        return flat


    @torch.no_grad()
    def sample_hard_activity_gridtopk(
        self,
        out: Dict[str, torch.Tensor],
        count_temperature: float = 1.0,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        count_logits = out["count_logits"]
        B = count_logits.shape[0]
        device = count_logits.device
    
        if count_temperature <= 0:
            counts = count_logits.argmax(dim=-1)
        else:
            counts = torch.distributions.Categorical(
                logits=count_logits / float(count_temperature)
            ).sample()
    
        score = self.soft_activity_flat(out, roi_mask=roi_mask)  # (B,N)
    
        if roi_mask is not None:
            if roi_mask.dim() == 3:
                roi_mask = roi_mask.squeeze(-1)
            roi_mask = roi_mask.to(device=device).bool()
            score = score.masked_fill(~roi_mask, -1.0)
    
        activity = torch.zeros(B, self.Ntok, device=device, dtype=torch.long)
    
        for b in range(B):
            if roi_mask is not None:
                max_valid = int(roi_mask[b].sum().item())
            else:
                max_valid = self.Ntok
    
            n = int(counts[b].clamp(0, min(self.Kmax, max_valid)).item())
            if n <= 0:
                continue
    
            idx = torch.topk(score[b], k=n).indices
            activity[b, idx] = 1
    
        return activity


    @torch.no_grad()
    def sample_hard_activity(
        self,
        out: Dict[str, torch.Tensor],
        count_temperature: float = 1.0,
        coord_temperature: float = 1.0,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inference-only hard activity sampling.

        Returns:
            activity_flat: (B,Ntok), 0/1, THW-flattened
        """

        count_logits = out["count_logits"]                   # (B,K+1)
        B = count_logits.shape[0]
        device = count_logits.device

        if count_temperature <= 0:
            counts = count_logits.argmax(dim=-1)             # (B,)
        else:
            counts = torch.distributions.Categorical(
                logits=count_logits / count_temperature
            ).sample()                                       # (B,)

        event_score = torch.sigmoid(out["event_logits"])     # (B,K)

        if coord_temperature <= 0:
            t = out["t_logits"].argmax(dim=-1)               # (B,K)
            h = out["h_logits"].argmax(dim=-1)               # (B,K)
            w = out["w_logits"].argmax(dim=-1)               # (B,K)
        else:
            t = torch.distributions.Categorical(
                logits=out["t_logits"] / coord_temperature
            ).sample()                                       # (B,K)
            h = torch.distributions.Categorical(
                logits=out["h_logits"] / coord_temperature
            ).sample()                                       # (B,K)
            w = torch.distributions.Categorical(
                logits=out["w_logits"] / coord_temperature
            ).sample()                                       # (B,K)

        flat = t * (self.Htok * self.Wtok) + h * self.Wtok + w # (B,K)
        
        if roi_mask is not None:
            if roi_mask.dim() == 3:
                roi_mask = roi_mask.squeeze(-1)
            roi_mask = roi_mask.to(device=device).bool()
        
            valid = roi_mask.gather(1, flat.clamp(0, self.Ntok - 1))
            event_score = event_score.masked_fill(~valid, -1.0)

        activity = torch.zeros(
            B,
            self.Ntok,
            device=device,
            dtype=torch.long,
        )

        for b in range(B):
            n = int(counts[b].clamp(0, self.Kmax).item())
            if n <= 0:
                continue

            chosen_queries = torch.topk(event_score[b], k=n).indices
            chosen_flat = flat[b, chosen_queries]
                
            if roi_mask is not None:
                valid_chosen = roi_mask[b].gather(0, chosen_flat.clamp(0, self.Ntok - 1))
                chosen_flat = chosen_flat[valid_chosen]
            
            activity[b, chosen_flat] = 1

        return activity


@torch.no_grad()
def build_activity_targets_from_codes(
    codes: torch.Tensor,
    token_grid: Tuple[int, int, int],
    Kmax: int,
    blank_code: int = -1,
    predict_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Builds sparse activity targets for DETR-style matching.

    Does NOT assign target j to query j.
    Hungarian matching is handled inside detr_activity_loss().
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

    activity_flat = active.long()                  # (B,N)

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
        "activity_flat": activity_flat,            # (B,N)
        "predict_mask": predict_mask if predict_mask is not None else None,
        # Padded GT active-token set. Matching happens in loss.
        "active_flat_padded": active_flat_padded,  # (B,Kmax)
        "active_valid": active_valid,              # (B,Kmax)
        "t_target": t_target,                      # (B,Kmax)
        "h_target": h_target,                      # (B,Kmax)
        "w_target": w_target,                      # (B,Kmax)
    }


def detr_activity_loss(
    out: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    activity_prior: DETRActivityPrior,
    *,
    lambda_count: float = 1.0,
    lambda_obj: float = 1.0,
    lambda_coord: float = 1.0,
    lambda_soft_count: float = 0.1,
    lambda_soft_grid: float = 0.0,
    lambda_dup: float = 0.01,
    no_object_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    DETR-style activity loss.

    Hungarian matching assigns predicted event queries to GT active tokens.
    Unmatched queries are trained as no-object.
    """

    if linear_sum_assignment is None:
        raise ImportError(
            "scipy is required for Hungarian matching. Install scipy or replace "
            "linear_sum_assignment with another assignment solver."
        )

    count_logits = out["count_logits"]              # (B,K+1)
    event_logits = out["event_logits"]              # (B,K)
    t_logits = out["t_logits"]                      # (B,K,T)
    h_logits = out["h_logits"]                      # (B,K,H)
    w_logits = out["w_logits"]                      # (B,K,W)

    B, Kmax = event_logits.shape
    device = event_logits.device

    count_target = targets["count_target"].long()   # (B,)
    active_valid = targets["active_valid"].bool()   # (B,K)
    t_target_all = targets["t_target"].long()       # (B,K)
    h_target_all = targets["h_target"].long()       # (B,K)
    w_target_all = targets["w_target"].long()       # (B,K)

    # ------------------------------------------------------------
    # 1. Count classification
    # ------------------------------------------------------------
    loss_count = F.cross_entropy(count_logits, count_target)

    # ------------------------------------------------------------
    # 2. Hungarian matching per sample
    # ------------------------------------------------------------
    event_target = torch.zeros_like(event_logits)    # (B,K)
    matched_query = []
    matched_t = []
    matched_h = []
    matched_w = []

    logpt = F.log_softmax(t_logits, dim=-1)          # (B,K,T)
    logph = F.log_softmax(h_logits, dim=-1)          # (B,K,H)
    logpw = F.log_softmax(w_logits, dim=-1)          # (B,K,W)

    obj_prob = torch.sigmoid(event_logits)           # (B,K)

    for b in range(B):
        n_gt = int(active_valid[b].sum().item())
        if n_gt <= 0:
            continue

        gt_t = t_target_all[b, :n_gt]                # (n_gt,)
        gt_h = h_target_all[b, :n_gt]                # (n_gt,)
        gt_w = w_target_all[b, :n_gt]                # (n_gt,)

        # Cost shape: (K queries, n_gt targets)
        # Lower is better.
        cost_t = -logpt[b][:, gt_t]                  # (K,n_gt)
        cost_h = -logph[b][:, gt_h]                  # (K,n_gt)
        cost_w = -logpw[b][:, gt_w]                  # (K,n_gt)

        # Prefer high-objectness queries for real active tokens.
        cost_obj = -obj_prob[b].clamp(1e-6, 1 - 1e-6).log().unsqueeze(1)

        cost = (
            lambda_obj * cost_obj
            + lambda_coord * (cost_t + cost_h + cost_w)
        )

        row_ind, col_ind = linear_sum_assignment(
            cost.detach().cpu().float().numpy()
        )

        row_ind = torch.as_tensor(row_ind, device=device, dtype=torch.long)
        col_ind = torch.as_tensor(col_ind, device=device, dtype=torch.long)

        event_target[b, row_ind] = 1.0

        matched_query.append(
            torch.stack([
                torch.full_like(row_ind, b),
                row_ind,
            ], dim=1)
        )
        matched_t.append(gt_t[col_ind])
        matched_h.append(gt_h[col_ind])
        matched_w.append(gt_w[col_ind])

    # ------------------------------------------------------------
    # 3. Object/no-object loss
    # ------------------------------------------------------------
    obj_weight = torch.ones_like(event_target)
    obj_weight[event_target < 0.5] = float(no_object_weight)

    loss_obj = F.binary_cross_entropy_with_logits(
        event_logits,
        event_target,
        weight=obj_weight,
    )

    # ------------------------------------------------------------
    # 4. Coordinate CE only on matched queries
    # ------------------------------------------------------------
    if len(matched_query) > 0:
        matched_query = torch.cat(matched_query, dim=0)      # (M,2)
        matched_t = torch.cat(matched_t, dim=0)              # (M,)
        matched_h = torch.cat(matched_h, dim=0)              # (M,)
        matched_w = torch.cat(matched_w, dim=0)              # (M,)

        b_idx = matched_query[:, 0]
        q_idx = matched_query[:, 1]

        loss_t = F.cross_entropy(t_logits[b_idx, q_idx], matched_t)
        loss_h = F.cross_entropy(h_logits[b_idx, q_idx], matched_h)
        loss_w = F.cross_entropy(w_logits[b_idx, q_idx], matched_w)

        loss_coord = loss_t + loss_h + loss_w
    else:
        loss_coord = event_logits.sum() * 0.0

    # ------------------------------------------------------------
    # 5. Soft count regularizer
    # ------------------------------------------------------------
    event_p = torch.sigmoid(event_logits)                    # (B,K)
    soft_count = event_p.sum(dim=1)                          # (B,)

    loss_soft_count = F.mse_loss(
        soft_count,
        count_target.float(),
    )

    # ------------------------------------------------------------
    # 6. Optional soft grid BCE
    # ------------------------------------------------------------
    soft_activity = activity_prior.soft_activity_flat(
        out,
        roi_mask=targets.get("predict_mask", None),
    )

    if lambda_soft_grid > 0:
        with torch.cuda.amp.autocast(enabled=False):
            loss_soft_grid = F.binary_cross_entropy(
                soft_activity.float().clamp(1e-6, 1.0 - 1e-6),
                targets["activity_flat"].float(),
            )
    else:
        loss_soft_grid = soft_activity.sum() * 0.0

    # ------------------------------------------------------------
    # 7. Duplicate occupancy penalty
    # ------------------------------------------------------------
    soft_grid_raw = activity_prior.soft_activity_grid(out, clamp=False)  # (B,T,H,W)
    loss_dup = F.relu(soft_grid_raw - 1.0).pow(2).mean()

    loss = (
        lambda_count * loss_count
        + lambda_obj * loss_obj
        + lambda_coord * loss_coord
        + lambda_soft_count * loss_soft_count
        + lambda_soft_grid * loss_soft_grid
        + lambda_dup * loss_dup
    )

    with torch.no_grad():
        hard_count = count_logits.argmax(dim=-1)
        count_acc = (hard_count == count_target).float().mean()

    aux = {
        "loss": loss.detach(),
        "loss_count": loss_count.detach(),
        "loss_obj": loss_obj.detach(),
        "loss_coord": loss_coord.detach(),
        "loss_soft_count": loss_soft_count.detach(),
        "loss_soft_grid": loss_soft_grid.detach(),
        "loss_dup": loss_dup.detach(),
        "pred_count_mean": soft_count.detach().mean(),
        "hard_count_mean": hard_count.float().detach().mean(),
        "target_count_mean": count_target.float().detach().mean(),
        "target_raw_count_mean": targets["raw_count"].float().detach().mean(),
        "count_acc": count_acc.detach(),
        "matched_tokens": event_target.sum().detach(),
    }

    return loss, aux


class MaskGITMotifPrior(nn.Module):
    """MaskGIT-style codebook motif prior over VQ-VAE latent tokens.
    
    Token structure:
        a_i  : 0 blank, 1 active
        z1_i : level-1 code, 0..K1-1, meaningful only if active
        z2_i : level-2 child code, 0..K2-1, meaningful only if active
    
    Input special IDs:
        a_mask_id  = 2
    
        z1_mask_id = K1
        z1_null_id = K1 + 1
    
        z2_mask_id = K2
        z2_null_id = K2 + 1
    
    Output:
        logits["z1"] : (B,N,K1)
        logits["z2"] : (B,N,K2)
    """
    
    def __init__(
        self,
        K1: int = 32,
        K2: int = 8,
        num_tasks: int = 4,
        gct_mapper: Optional[nn.Module] = None,
        lct_mapper: Optional[nn.Module] = None,
        gct_latent_dim: int = 16,
        lct_latent_dim: int = 16,
        d_model: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        max_len: int = 5040,
        dropout: float = 0.25,
        pad_mask: bool = False,
    ):
        super().__init__()
    
        self.K1 = int(K1)
        self.K2 = int(K2)
        self.num_tasks = int(num_tasks)
        self.d_model = int(d_model)
        self.max_len = int(max_len)
    
        # activity ids
        self.a_blank_id = 0
        self.a_active_id = 1
        self.a_mask_id = 2
    
        # hierarchical code ids
        self.z1_mask_id = self.K1
        self.z1_null_id = self.K1 + 1
    
        self.z2_mask_id = self.K2
        self.z2_null_id = self.K2 + 1
    
        self.ctx_len = 3
        self.use_sparse_motif_encoder = True
        self.pad_mask = bool(pad_mask)
    
        self.gct_mapper = gct_mapper
        self.lct_mapper = lct_mapper
    
        if self.gct_mapper is not None:
            for p in self.gct_mapper.parameters():
                p.requires_grad_(False)
            self.gct_mapper.eval()
    
        if self.lct_mapper is not None:
            for p in self.lct_mapper.parameters():
                p.requires_grad_(False)
            self.lct_mapper.eval()
            
        # Input embeddings for transformer input
        self.a_emb = nn.Embedding(3, d_model)          # blank, active, mask
        self.z1_emb = nn.Embedding(K1 + 2, d_model)    # K1 codes + mask + null
        self.z2_emb = nn.Embedding(K2 + 2, d_model)    # K2 codes + mask + null
        
        self.roi_emb = nn.Embedding(2, d_model)  # 0 visible/context, 1 ROI/predict

        self.a_to_z1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.z1_head = nn.Linear(d_model, K1)
        
        self.z1_to_z2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.z2_head = nn.Linear(d_model, K2)
    
        # Prefix context tokens
        self.task_emb = nn.Embedding(num_tasks, d_model)        
        self.gct_proj = nn.Linear(gct_latent_dim, d_model)
        self.lct_proj = nn.Linear(lct_latent_dim, d_model)
    
        self.pos_emb = nn.Embedding(max_len, d_model)
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
        z1_in,
        z2_in,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compact per-position hierarchical token fusion.
    
        Inputs:
            a_in, z1_in, z2_in: (B, N) and also the ROI (if not given, consider all tokens)
    
        Output:
            x_tok: (B, N, D)
    
        This keeps transformer length N, not 3N.
        """
        a_e = self.a_emb(a_in)
        z1_e = self.z1_emb(z1_in)
        z2_e = self.z2_emb(z2_in)
    
        x_tok = a_e + z1_e + z2_e
        
        if roi_mask is not None:
            if roi_mask.dim() == 3:
                roi_mask = roi_mask.squeeze(-1)
            roi_mask = roi_mask.to(device=a_in.device).bool().long()
            x_tok = x_tok + self.roi_emb(roi_mask)
        
        return x_tok
    
    
    def _expected_emb_from_logits(
        self,
        logits: torch.Tensor,
        emb: nn.Embedding,
        n_classes: int,
    ) -> torch.Tensor:
        """
        Soft expected embedding from predicted class probabilities.
        Excludes mask/null IDs by only using emb.weight[:n_classes].
        """
        p = F.softmax(logits, dim=-1)
        return p @ emb.weight[:n_classes]
    
    
    def _compute_motif_logits(
        self,
        h: torch.Tensor,
        *,
        activity_ids: Optional[torch.Tensor] = None,   # (B,N), 0 blank / 1 active
        activity_prob: Optional[torch.Tensor] = None,  # (B,N), soft active probability
        z1_teacher: Optional[torch.Tensor] = None,
        z1_teacher_prob: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
    
        if activity_prob is not None:
            blank_e = self.a_emb.weight[self.a_blank_id]    # (D,)
            active_e = self.a_emb.weight[self.a_active_id]  # (D,)
            a_info = (
                (1.0 - activity_prob).unsqueeze(-1) * blank_e
                + activity_prob.unsqueeze(-1) * active_e
            )  # (B,N,D)
    
        elif activity_ids is not None:
            activity_ids = activity_ids.long().clamp(0, 1)
            a_info = self.a_emb(activity_ids)  # (B,N,D)
    
        else:
            raise ValueError("Need either activity_ids or activity_prob.")
    
        h_z1 = h + self.a_to_z1(a_info)  # (B,N,D)
        z1_logits = self.z1_head(h_z1)   # (B,N,K1)
    
        z1_pred_info = self._expected_emb_from_logits(
            z1_logits,
            self.z1_emb,
            n_classes=self.K1,
        )  # (B,N,D)
        
        if z1_teacher is not None and z1_teacher_prob > 0.0:
            z1_teacher = z1_teacher.long().clamp(0, self.K1 - 1)
            z1_gt_info = self.z1_emb(z1_teacher)
        
            if z1_teacher_prob >= 1.0:
                z1_info = z1_gt_info
            else:
                z1_info = (
                    float(z1_teacher_prob) * z1_gt_info
                    + (1.0 - float(z1_teacher_prob)) * z1_pred_info
                )
        else:
            z1_info = z1_pred_info  # (B,N,D)
    
        h_z2 = h_z1 + self.z1_to_z2(z1_info)  # (B,N,D)
        z2_logits = self.z2_head(h_z2)        # (B,N,K2)
    
        return {
            "z1": z1_logits,
            "z2": z2_logits,
        }
    
    def _build_motif_sparse_mask(
        self,
        a_in: torch.Tensor,
        z1_in: torch.Tensor,
        z2_in: torch.Tensor,
        *,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns sparse attention mask for motif prior.
    
        True = keep token in sparse transformer.
    
        Stage 3A:
            use all GT-active tokens.
    
        Stage 3C / inference-like:
            keep visible active tokens plus ROI/prediction tokens.
        """
        if targets is not None and "active" in targets:
            active = targets["active"].to(device=a_in.device).bool()
        else:
            active = a_in.eq(self.a_active_id)
    
        keep = active.clone()

        # Keep explicitly masked motif positions.
        # These should already be active in Stage 3A, but this is safer.
        keep = keep | z1_in.eq(self.z1_mask_id) | z2_in.eq(self.z2_mask_id)
        
        return keep
    
    
    def forward_with_activity_prob(
        self,
        z1_in: torch.LongTensor,
        z2_in: torch.LongTensor,
        *,
        activity_prob: torch.Tensor,       # (B,N), differentiable soft activity
        global_ctx: torch.Tensor,
        local_ctx: torch.Tensor,
        task_id: torch.Tensor,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Differentiable motif forward used when refining the activity prior.
    
        activity_prob comes from DETRActivityPrior.soft_activity_flat(out).
        This allows voxel/statistical losses after soft decoding to backprop
        into the activity prior.
        """
    
        B, N = z1_in.shape
    
        if z2_in.shape != (B, N):
            raise ValueError("z1_in and z2_in must have shape (B,N).")
    
        if activity_prob.shape != (B, N):
            raise ValueError(
                f"activity_prob must have shape {(B, N)}, got {tuple(activity_prob.shape)}"
            )
    
        if N > self.max_len:
            raise ValueError(f"N={N} exceeds max_len={self.max_len}.")
    
        # Build a soft activity embedding for the input stream.
        blank_e = self.a_emb.weight[self.a_blank_id]
        active_e = self.a_emb.weight[self.a_active_id]
        a_soft_e = (
            (1.0 - activity_prob).unsqueeze(-1) * blank_e
            + activity_prob.unsqueeze(-1) * active_e
        )  # (B,N,D)
    
        z1_e = self.z1_emb(z1_in)
        z2_e = self.z2_emb(z2_in)
    
        prefix = self._build_prefix(global_ctx, local_ctx, task_id)
        ctx_tok = self.ctx_fuse(prefix.mean(dim=1)).unsqueeze(1)
        
        x_tok = a_soft_e + z1_e + z2_e
        
        if roi_mask is not None:
            if roi_mask.dim() == 3:
                roi_mask = roi_mask.squeeze(-1)
            roi_mask_bool = roi_mask.to(device=z1_in.device).bool()
            x_tok = x_tok + self.roi_emb(roi_mask_bool.long())
        else:
            roi_mask_bool = None
        
        tok_pos = self.pos_emb(
            torch.arange(N, device=z1_in.device)
        ).unsqueeze(0)
        
        x_tok = x_tok + tok_pos + ctx_tok
        x_tok = self.drop(x_tok)
        
        visible_active_or_masked = (
            z1_in.ne(self.z1_null_id)
            | z2_in.ne(self.z2_null_id)
        )
        
        pred_active = activity_prob.gt(0.05)
        
        if roi_mask_bool is not None:
            sparse_keep = visible_active_or_masked | (roi_mask_bool & pred_active)
        else:
            sparse_keep = visible_active_or_masked | pred_active
        
        h, _, _, sparse_lengths = self.blocks(
            tokens=x_tok,
            active_mask=sparse_keep,
            pos_embed=None,
            fill_value=0.0,
        )
        
        h = self.ln_f(h)
    
        logits = self._compute_motif_logits(
            h,
            activity_prob=activity_prob,
        )
    
        return logits
    
    def forward(
        self,
        a_in: torch.LongTensor,
        z1_in: torch.LongTensor,
        z2_in: torch.LongTensor,
        *,
        global_ctx: torch.Tensor,
        local_ctx: torch.Tensor,
        task_id: torch.Tensor,
        roi_mask: Optional[torch.Tensor] = None,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        loss_weights: Tuple[float, float] = (1.0, 1.0),
    ):
        """
        a_in, z1_in, z2_in:
            (B,N)
    
        targets:
            {
                "a":           (B,N), 0/1
                "z1":          (B,N), 0..K1-1
                "z2":          (B,N), 0..K2-1
                "a_loss_mask": (B,N) bool
                "z_loss_mask": (B,N) bool, usually predict_mask & active
            }
        """
        B, N = a_in.shape
    
        if z1_in.shape != (B, N) or z2_in.shape != (B, N):
            raise ValueError("a_in, z1_in, and z2_in must all have shape (B,N).")
    
        if N > self.max_len:
            raise ValueError(f"N={N} exceeds max_len={self.max_len}.")
    
        prefix = self._build_prefix(global_ctx, local_ctx, task_id)  # (B,3,D)
        ctx_tok = self.ctx_fuse(prefix.mean(dim=1)).unsqueeze(1)     # (B,1,D)
        
        if roi_mask is None and targets is not None:
            roi_mask = targets.get("predict_mask", None)
        
        x_tok = self._embed_layer_streams(a_in, z1_in, z2_in, roi_mask=roi_mask)
        
        tok_pos = self.pos_emb(
            torch.arange(N, device=a_in.device)
        ).unsqueeze(0)
        
        x_tok = x_tok + tok_pos + ctx_tok
        x_tok = self.drop(x_tok)
        
        sparse_keep = self._build_motif_sparse_mask(
            a_in,
            z1_in,
            z2_in,
            targets=targets,
            roi_mask=roi_mask,
        )
        
        h, _, _, sparse_lengths = self.blocks(
            tokens=x_tok,
            active_mask=sparse_keep,
            pos_embed=None,
            fill_value=0.0,
        )
        
        h = self.ln_f(h)
                

        
        if targets is not None:
            activity_ids = targets["a"].long()
        else:
            activity_ids = a_in.long().clamp(0, 1)
        
        z1_teacher = targets["z1"] if targets is not None else None
        z1_teacher_prob = targets.get("z1_teacher_prob", 0.0) if targets is not None else 0.0
        
        logits = self._compute_motif_logits(
            h,
            activity_ids=activity_ids,
            z1_teacher=z1_teacher,
            z1_teacher_prob=float(z1_teacher_prob),
        )
            
        if targets is None:
            return logits, None, {}
    
        z1_t = targets["z1"].long()
        z2_t = targets["z2"].long()
    
        z1_loss_mask = targets.get("z1_loss_mask", targets["z_loss_mask"]).bool()
        z2_loss_mask = targets.get("z2_loss_mask", targets["z_loss_mask"]).bool()
    
        ignore_z1 = torch.full_like(z1_t, -100)
        ignore_z2 = torch.full_like(z2_t, -100)
    
        z1_target = torch.where(z1_loss_mask, z1_t, ignore_z1)
        z2_target = torch.where(z2_loss_mask, z2_t, ignore_z2)
        
    
    
        if z1_loss_mask.any():
            loss_z1 = F.cross_entropy(
                logits["z1"].reshape(-1, self.K1),
                z1_target.reshape(-1),
                ignore_index=-100,
            )
        else:
            loss_z1 = logits["z1"].sum() * 0.0
        
        if z2_loss_mask.any():
            loss_z2 = F.cross_entropy(
                logits["z2"].reshape(-1, self.K2),
                z2_target.reshape(-1),
                ignore_index=-100,
            )
        else:
            loss_z2 = logits["z2"].sum() * 0.0
    
        w1, w2 = loss_weights
        loss = w1 * loss_z1 + w2 * loss_z2
    
        aux = {
            "loss": loss.detach(),
            "loss_z1": loss_z1.detach(),
            "loss_z2": loss_z2.detach(),
            "z1_loss_tokens": z1_loss_mask.sum().detach(),
            "z2_loss_tokens": z2_loss_mask.sum().detach(),
        }
    
        return logits, loss, aux

    @staticmethod
    def make_targets_from_codes(
        codes: torch.Tensor,
        predict_mask: torch.Tensor,
        blank_code: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """
        codes:
            (B,N,2), from VQVAE.
    
        predict_mask:
            (B,N) or (B,N,1), where True/1 means prior should predict this token.
        """
        if codes.dim() != 3 or codes.size(-1) < 2:
            raise ValueError(f"Expected codes shape (B,N,2), got {tuple(codes.shape)}")
    
        if predict_mask.dim() == 3:
            predict_mask = predict_mask.squeeze(-1)
    
        z1_raw = codes[..., 0].long()
        z2_raw = codes[..., 1].long()
    
        active = z1_raw.ne(int(blank_code))
    
        a = active.long()
        z1 = z1_raw.clamp_min(0)
        z2 = z2_raw.clamp_min(0)
    
        pmask = predict_mask.bool()
    
        return {
            "a": a,
            "z1": z1,
            "z2": z2,
            "active": active,
            "predict_mask": pmask,
            "a_loss_mask": pmask,
            "z1_loss_mask": pmask & active,
            "z2_loss_mask": pmask & active,
            "z_loss_mask": pmask & active,
        }

    def corrupt_inputs_from_targets(
        self,
        targets: Dict[str, torch.Tensor],
        ensure_at_least_one_mask: bool = True,
        full_mask_prob: float = 0.15,
    ):
        """
        Pairwise motif masking.
        
        z1 and z2 jointly identify one hierarchical motif. They therefore use
        the exact same patch mask and are predicted together. Activity is
        supplied separately and is not corrupted here.
        """
        a = targets["a"].long()
        z1 = targets["z1"].long().clamp(0, self.K1 - 1)
        z2 = targets["z2"].long().clamp(0, self.K2 - 1)
    
        active = targets["active"].bool()
        pmask = targets["predict_mask"].bool()
    
        B, N = a.shape
        device = a.device
    
        gamma = torch.rand((B, 1), device=device)
        
        # Sometimes train exactly on the inference starting state:
        # a/z1/z2 all masked inside predict_mask.
        if full_mask_prob > 0:
            full_mask = torch.rand((B, 1), device=device) < float(full_mask_prob)
            gamma = torch.where(full_mask, torch.ones_like(gamma), gamma)
        else:
            full_mask = torch.zeros((B, 1), device=device, dtype=torch.bool)
    
        valid_z = pmask & active
        m = (torch.rand((B, N), device=device) < gamma) & valid_z
        # z2 should also be masked anywhere the coarse state is masked.
        # This makes training match inference start:
        # a=MASK, z1=MASK, z2=MASK.
    
        if ensure_at_least_one_mask:
            for b in range(B):
                valid_z = torch.where(pmask[b] & active[b])[0]
        
                if valid_z.numel() > 0 and not (m[b] & active[b]).any():
                    idx = valid_z[torch.randint(valid_z.numel(), (1,), device=device)]
                    m[b, idx] = True
    
        a_in = a.clone()
        z1_in = z1.clone()
        z2_in = z2.clone()
    
        # Inactive/blank positions should not expose fake clamped code 0.
        z1_in[~active] = self.z1_null_id
        z2_in[~active] = self.z2_null_id
    
        # Coarse-stage masking.
        # Activity is externally supplied / teacher-forced.
        # Do not mask a.
        z1_in[m] = self.z1_mask_id
    
        # Fine-stage masking.
        z2_in[m] = self.z2_mask_id
    
        targets = dict(targets)
        targets["a_loss_mask"] = m
        targets["z1_loss_mask"] = m & active
        targets["z2_loss_mask"] = m & active
        targets["z_loss_mask"] = targets["z1_loss_mask"] | targets["z2_loss_mask"]
        targets["gamma"] = gamma.squeeze(1)
    
        return a_in, z1_in, z2_in, targets
    
    
class HierarchicalCodebookPrior(nn.Module):
    """
    Wrapper around:
        1. DETRActivityPrior
        2. MaskGITMotifPrior

    This class only routes calls.
    Training losses should remain in train_prior.py or small loss helpers.
    """

    def __init__(
        self,
        activity_prior: DETRActivityPrior,
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
    ):
        return self.activity_prior(
            global_ctx=global_ctx,
            local_ctx=local_ctx,
            task_id=task_id,
            a_in=a_in,
            roi_mask=roi_mask,
        )

    def forward_motif_teacher_forced(
        self,
        a_in: torch.LongTensor,
        z1_in: torch.LongTensor,
        z2_in: torch.LongTensor,
        *,
        global_ctx: torch.Tensor,
        local_ctx: torch.Tensor,
        task_id: torch.Tensor,
        roi_mask: Optional[torch.Tensor] = None,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        loss_weights: Tuple[float, float] = (1.0, 1.0),
    ):
        return self.motif_prior(
            a_in,
            z1_in,
            z2_in,
            global_ctx=global_ctx,
            local_ctx=local_ctx,
            task_id=task_id,
            roi_mask=roi_mask,
            targets=targets,
            loss_weights=loss_weights,
        )

    def forward_joint_soft(
        self,
        z1_in: torch.LongTensor,
        z2_in: torch.LongTensor,
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
            z1_in=z1_in,
            z2_in=z2_in,
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
            roi_mask=roi_mask,
        )