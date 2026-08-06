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
    DETR-style sparse activity prior with configurable coordinate heads.

    Repository convention:
        token grid = (Ttok,Htok,Wtok)
        flat = t * Htok * Wtok + h * Wtok + w

    Outputs:
        count_logits : (B,Kmax+1)
        event_logits : (B,Kmax)
        factorized:
            t_logits : (B,Kmax,Ttok)
            h_logits : (B,Kmax,Htok)
            w_logits : (B,Kmax,Wtok)
        joint_dense:
            grid_logits : (B,Kmax,Ntok)
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
        coordinate_mode: str = "factorized",
    ):
        super().__init__()

        self.Ttok, self.Htok, self.Wtok = map(int, token_grid)
        self.Ntok = self.Ttok * self.Htok * self.Wtok
        self.Kmax = int(Kmax)
        self.d_model = int(d_model)
        self.coordinate_mode = str(coordinate_mode).lower()
        if self.coordinate_mode not in ("factorized", "joint_dense"):
            raise ValueError(
                f"Unsupported coordinate_mode={coordinate_mode!r}. "
                "Use 'factorized' or 'joint_dense'."
            )

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

        # Detached count-to-event FiLM. The count head controls event quantity,
        # while contextualized event queries retain responsibility for ranking
        # and coordinates. Zero initialization makes this an identity mapping
        # at construction time.
        self.count_event_film = nn.Linear(1, 2 * d_model)
        nn.init.zeros_(self.count_event_film.weight)
        nn.init.zeros_(self.count_event_film.bias)

        self.event_head = nn.Linear(d_model, 1)
        if self.coordinate_mode == "factorized":
            self.t_head = nn.Linear(d_model, self.Ttok)
            self.h_head = nn.Linear(d_model, self.Htok)
            self.w_head = nn.Linear(d_model, self.Wtok)
            self.grid_head = None
        else:
            self.t_head = None
            self.h_head = None
            self.w_head = None
            self.grid_head = nn.Linear(d_model, self.Ntok)

    def flatten_coordinates(
        self,
        t: torch.Tensor,
        h: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """Convert THW token coordinates to the repository's flat order."""
        return t.long() * (self.Htok * self.Wtok) + h.long() * self.Wtok + w.long()

    def unflatten_coordinates(
        self,
        flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert repository-order flat token IDs to THW coordinates."""
        flat = flat.long()
        t = torch.div(flat, self.Htok * self.Wtok, rounding_mode="floor")
        remainder = flat % (self.Htok * self.Wtok)
        h = torch.div(remainder, self.Wtok, rounding_mode="floor")
        w = remainder % self.Wtok
        return t, h, w

    def coordinate_metadata(self) -> Dict[str, int | str]:
        return {
            "coordinate_mode": self.coordinate_mode,
            "Ttok": int(self.Ttok),
            "Htok": int(self.Htok),
            "Wtok": int(self.Wtok),
            "Ntok": int(self.Ntok),
        }

    
    
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
    
        # Preserve token-state/position interactions by applying nonlinear
        # processing before the permutation-invariant mean pool.
        a_tok = self.a_pool(a_tok)  # (B,N,D)
        a_vec = a_tok.mean(dim=1)   # (B,D)

        return a_vec.to(dtype=dtype)
        


    def apply_roi_mask(
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
        if roi.shape != (B, self.Ntok):
            raise ValueError(
                f"roi_mask must have shape {(B, self.Ntok)}, got {tuple(roi.shape)}"
            )

        out = dict(out)
        if self.coordinate_mode == "joint_dense":
            # Avoid an all-masked softmax. Empty-ROI samples retain one harmless
            # fallback logit, and downstream ROI multiplication makes their
            # activity exactly zero.
            safe_roi = roi.clone()
            empty_roi = ~safe_roi.any(dim=1)
            if bool(empty_roi.any()):
                safe_roi[empty_roi, 0] = True
            out["grid_logits"] = out["grid_logits"].masked_fill(
                ~safe_roi[:, None, :],
                fill_value,
            )
            out["roi_empty"] = empty_roi
            return out

        roi_grid = roi.view(B, self.Ttok, self.Htok, self.Wtok)
    
        valid_t = roi_grid.any(dim=3).any(dim=2)  # (B,T)
        valid_h = roi_grid.any(dim=3).any(dim=1)  # (B,H)
        valid_w = roi_grid.any(dim=2).any(dim=1)  # (B,W)
    
        out["t_logits"] = out["t_logits"].masked_fill(~valid_t[:, None, :], fill_value)
        out["h_logits"] = out["h_logits"].masked_fill(~valid_h[:, None, :], fill_value)
        out["w_logits"] = out["w_logits"].masked_fill(~valid_w[:, None, :], fill_value)
    
        return out

    def apply_roi_axis_mask(
        self,
        out: Dict[str, torch.Tensor],
        roi_mask: Optional[torch.Tensor],
        fill_value: float = -1e4,
    ) -> Dict[str, torch.Tensor]:
        """Backward-compatible alias for the generalized ROI mask."""
        return self.apply_roi_mask(out, roi_mask, fill_value=fill_value)
    
    

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
        iterations: int = 24,
    ) -> torch.Tensor:
        """Find one shared bias/sample that matches expected event mass."""
        target = target_count.to(
            device=event_logits_raw.device,
            dtype=event_logits_raw.dtype,
        ).clamp(0.0, float(event_logits_raw.shape[1]))
        lo = event_logits_raw.new_full((event_logits_raw.shape[0],), -30.0)
        hi = event_logits_raw.new_full((event_logits_raw.shape[0],), 30.0)
        for _ in range(int(iterations)):
            mid = 0.5 * (lo + hi)
            mass = torch.sigmoid(event_logits_raw + mid.unsqueeze(1)).sum(dim=1)
            too_small = mass < target
            lo = torch.where(too_small, mid, lo)
            hi = torch.where(too_small, hi, mid)
        return 0.5 * (lo + hi)

    def forward(
        self,
        global_ctx: torch.Tensor,  # (B,G)
        local_ctx: torch.Tensor,   # (B,L)
        task_id: torch.Tensor,     # (B,)
        a_in: Optional[torch.Tensor] = None,
        roi_mask: Optional[torch.Tensor] = None,
        count_target: Optional[torch.Tensor] = None,
        count_teacher_prob: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Run count and event branches with detached count conditioning."""
        B = global_ctx.shape[0]
        g = self.global_proj(global_ctx).unsqueeze(1)
        l = self.local_proj(local_ctx).unsqueeze(1)
        task = self.task_emb(task_id.long()).unsqueeze(1)

        a_vec = self._encode_activity_input(
            a_in=a_in, device=global_ctx.device, dtype=global_ctx.dtype
        )
        a_tok = torch.zeros_like(g) if a_vec is None else a_vec.unsqueeze(1)
        q = self.event_queries.unsqueeze(0).expand(B, -1, -1) + a_tok
        x = torch.cat([g, l, task, a_tok, q], dim=1)
        h = self.encoder(x)
        ctx_h = h[:, :4].mean(dim=1)
        ev_h = h[:, 4:]

        count_logits = self.count_head(ctx_h)
        predicted_count = self._expected_count_from_logits(count_logits).to(
            dtype=ev_h.dtype
        ).detach()
        teacher_prob = float(max(0.0, min(1.0, count_teacher_prob)))
        if count_target is not None and teacher_prob > 0.0:
            teacher_count = count_target.to(
                device=ev_h.device, dtype=ev_h.dtype
            ).detach()
            conditioned_count = (
                teacher_prob * teacher_count
                + (1.0 - teacher_prob) * predicted_count
            )
        else:
            conditioned_count = predicted_count

        count_scalar = (conditioned_count / max(float(self.Kmax), 1.0)).unsqueeze(-1)
        gamma_beta = self.count_event_film(count_scalar)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        ev_h_counted = ev_h * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        event_logits_raw = self.event_head(ev_h_counted).squeeze(-1)
        event_bias = self._shared_count_bias(
            event_logits_raw.detach().float(),
            conditioned_count.detach().float(),
        ).to(dtype=event_logits_raw.dtype)
        event_logits = event_logits_raw + event_bias.unsqueeze(1)
        out = {
            "count_logits": count_logits,
            "count_expected": predicted_count,
            "count_condition": conditioned_count,
            "event_logits_raw": event_logits_raw,
            "event_calibration_bias": event_bias,
            "event_logits": event_logits,
        }
        if self.coordinate_mode == "joint_dense":
            out["grid_logits"] = self.grid_head(ev_h_counted)
        else:
            out["t_logits"] = self.t_head(ev_h_counted)
            out["h_logits"] = self.h_head(ev_h_counted)
            out["w_logits"] = self.w_head(ev_h_counted)
        return self.apply_roi_mask(out, roi_mask)

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

    def soft_activity_grid(
        self,
        out: Dict[str, torch.Tensor],
        clamp: bool = True,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Converts event coordinate distributions to a soft activity grid.

        Returns:
            activity_grid: (B,Ttok,Htok,Wtok)
        """

        activity = self.soft_activity_flat(out, roi_mask=roi_mask).view(
            out["event_logits"].shape[0],
            self.Ttok,
            self.Htok,
            self.Wtok,
        )
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
        event_p = torch.sigmoid(out["event_logits"])  # (B,K)
        if self.coordinate_mode == "joint_dense":
            grid_p = F.softmax(out["grid_logits"], dim=-1)  # (B,K,N)
            flat = (event_p.unsqueeze(-1) * grid_p).sum(dim=1)
        else:
            pt = F.softmax(out["t_logits"], dim=-1)
            ph = F.softmax(out["h_logits"], dim=-1)
            pw = F.softmax(out["w_logits"], dim=-1)
            activity = (
                event_p[:, :, None, None, None]
                * pt[:, :, :, None, None]
                * ph[:, :, None, :, None]
                * pw[:, :, None, None, :]
            ).sum(dim=1)
            flat = activity.reshape(out["event_logits"].shape[0], self.Ntok)
        
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
        count_mode: str = "expected",
        count_stochastic_round: bool = False,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        count_logits = out["count_logits"]
        B = count_logits.shape[0]
        device = count_logits.device

        counts = self.select_counts(
            count_logits,
            mode=count_mode,
            temperature=count_temperature,
            stochastic_round=count_stochastic_round,
        )
    
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
        count_mode: str = "expected",
        count_stochastic_round: bool = False,
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

        counts = self.select_counts(
            count_logits,
            mode=count_mode,
            temperature=count_temperature,
            stochastic_round=count_stochastic_round,
        )                                                     # (B,)

        event_score = torch.sigmoid(out["event_logits"])     # (B,K)

        if self.coordinate_mode == "joint_dense":
            if coord_temperature <= 0:
                flat = out["grid_logits"].argmax(dim=-1)
            else:
                flat = torch.distributions.Categorical(
                    logits=out["grid_logits"] / coord_temperature
                ).sample()
        else:
            if coord_temperature <= 0:
                t = out["t_logits"].argmax(dim=-1)
                h = out["h_logits"].argmax(dim=-1)
                w = out["w_logits"].argmax(dim=-1)
            else:
                t = torch.distributions.Categorical(
                    logits=out["t_logits"] / coord_temperature
                ).sample()
                h = torch.distributions.Categorical(
                    logits=out["h_logits"] / coord_temperature
                ).sample()
                w = torch.distributions.Categorical(
                    logits=out["w_logits"] / coord_temperature
                ).sample()
            flat = self.flatten_coordinates(t, h, w)
        
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
    lambda_count_neighbor: float = 0.25,
    lambda_count_distance: float = 0.05,
    lambda_obj: float = 1.0,
    lambda_coord: float = 1.0,
    lambda_soft_count: float = 0.1,
    lambda_soft_grid: float = 0.0,
    lambda_dup: float = 0.10,
    no_object_weight: float = 0.1,
    count_neighbor_k: int = 11,
    count_neighbor_tau: float = 2.0,
    count_distance_scale: float = 5.0,
    soft_count_beta: float = 5.0,
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
    coordinate_mode = activity_prior.coordinate_mode
    if coordinate_mode == "joint_dense":
        grid_logits = out["grid_logits"]            # (B,K,N)
        t_logits = h_logits = w_logits = None
    else:
        t_logits = out["t_logits"]                  # (B,K,T)
        h_logits = out["h_logits"]                  # (B,K,H)
        w_logits = out["w_logits"]                  # (B,K,W)
        grid_logits = None

    B, Kmax = event_logits.shape
    device = event_logits.device

    count_target = targets["count_target"].long()   # (B,)
    active_valid = targets["active_valid"].bool()   # (B,K)
    t_target_all = targets["t_target"].long()       # (B,K)
    h_target_all = targets["h_target"].long()       # (B,K)
    w_target_all = targets["w_target"].long()       # (B,K)
    flat_target_all = targets["active_flat_padded"].long()  # (B,K)

    # ------------------------------------------------------------
    # 1. Count classification: exact + ordered-neighbor + expected distance
    # ------------------------------------------------------------
    # Keep the ordinal count objective in FP32. Under AMP, count_logits may
    # be float16/bfloat16 while softmax is promoted to float32; scatter_
    # requires the destination and source dtypes to match exactly. FP32 also
    # preserves the small neighbor-target probabilities more accurately.
    count_logits_loss = count_logits.float()

    loss_count_exact = F.cross_entropy(
        count_logits_loss,
        count_target,
    )

    count_values = torch.arange(
        count_logits.shape[-1],
        device=device,
        dtype=torch.float32,
    )
    count_distance = (
        count_values.unsqueeze(0)
        - count_target.float().unsqueeze(1)
    ).abs()

    neighbor_k = max(1, min(int(count_neighbor_k), count_logits.shape[-1]))
    near_distance, near_index = torch.topk(
        count_distance,
        k=neighbor_k,
        largest=False,
        dim=-1,
    )
    neighbor_target_local = F.softmax(
        -near_distance / max(float(count_neighbor_tau), 1e-6),
        dim=-1,
    )
    neighbor_target = torch.zeros_like(count_logits_loss).scatter(
        1,
        near_index,
        neighbor_target_local,
    )
    count_log_prob = F.log_softmax(count_logits_loss, dim=-1)
    count_prob = count_log_prob.exp()

    loss_count_neighbor = -(
        neighbor_target * count_log_prob
    ).sum(dim=-1).mean()

    loss_count_distance = (
        count_prob * count_distance
    ).sum(dim=-1).mean() / max(float(count_distance_scale), 1e-6)

    loss_count = (
        float(lambda_count) * loss_count_exact
        + float(lambda_count_neighbor) * loss_count_neighbor
        + float(lambda_count_distance) * loss_count_distance
    )

    # ------------------------------------------------------------
    # 2. Hungarian matching per sample
    # ------------------------------------------------------------
    event_target = torch.zeros_like(event_logits)    # (B,K)
    matched_query = []
    matched_t = []
    matched_h = []
    matched_w = []
    matched_flat = []

    # Matching is discrete, so do not retain a backward graph for the dense
    # log-probability tables. These tensors are reused for diagnostics below.
    with torch.no_grad():
        if coordinate_mode == "joint_dense":
            log_grid = F.log_softmax(grid_logits.float(), dim=-1)
            logpt = logph = logpw = None
        else:
            logpt = F.log_softmax(t_logits.float(), dim=-1)
            logph = F.log_softmax(h_logits.float(), dim=-1)
            logpw = F.log_softmax(w_logits.float(), dim=-1)
            log_grid = None

        obj_prob = torch.sigmoid(event_logits.float())  # (B,K)

    for b in range(B):
        n_gt = int(active_valid[b].sum().item())
        if n_gt <= 0:
            continue

        gt_t = t_target_all[b, :n_gt]                # (n_gt,)
        gt_h = h_target_all[b, :n_gt]                # (n_gt,)
        gt_w = w_target_all[b, :n_gt]                # (n_gt,)
        gt_flat = flat_target_all[b, :n_gt]          # (n_gt,)

        # Cost shape: (K queries, n_gt targets)
        # Lower is better.
        if coordinate_mode == "joint_dense":
            cost_coord = -log_grid[b][:, gt_flat]    # (K,n_gt)
        else:
            cost_t = -logpt[b][:, gt_t]              # (K,n_gt)
            cost_h = -logph[b][:, gt_h]              # (K,n_gt)
            cost_w = -logpw[b][:, gt_w]              # (K,n_gt)
            cost_coord = cost_t + cost_h + cost_w

        # Prefer high-objectness queries for real active tokens.
        cost_obj = -obj_prob[b].clamp(1e-6, 1 - 1e-6).log().unsqueeze(1)

        cost = (
            lambda_obj * cost_obj
            + lambda_coord * cost_coord
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
        matched_flat.append(gt_flat[col_ind])

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
        matched_flat = torch.cat(matched_flat, dim=0)        # (M,)

        b_idx = matched_query[:, 0]
        q_idx = matched_query[:, 1]

        if coordinate_mode == "joint_dense":
            loss_coord = F.cross_entropy(
                grid_logits[b_idx, q_idx].float(),
                matched_flat,
            )
            loss_t = loss_h = loss_w = loss_coord.detach() * 0.0
        else:
            loss_t = F.cross_entropy(t_logits[b_idx, q_idx], matched_t)
            loss_h = F.cross_entropy(h_logits[b_idx, q_idx], matched_h)
            loss_w = F.cross_entropy(w_logits[b_idx, q_idx], matched_w)
            loss_coord = loss_t + loss_h + loss_w
    else:
        loss_coord = event_logits.sum() * 0.0
        loss_t = loss_h = loss_w = loss_coord

    # ------------------------------------------------------------
    # 5. Soft count regularizer
    # ------------------------------------------------------------
    event_p = torch.sigmoid(event_logits)                    # (B,K)
    soft_count = event_p.sum(dim=1)                          # (B,)

    loss_soft_count = F.smooth_l1_loss(
        soft_count,
        count_target.float(),
        beta=max(float(soft_count_beta), 1e-6),
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
    soft_grid_raw = activity_prior.soft_activity_grid(
        out,
        clamp=False,
        roi_mask=targets.get("predict_mask", None),
    )  # (B,T,H,W)
    loss_dup = F.relu(soft_grid_raw - 1.0).pow(2).mean()

    loss = (
        loss_count
        + lambda_obj * loss_obj
        + lambda_coord * loss_coord
        + lambda_soft_count * loss_soft_count
        + lambda_soft_grid * loss_soft_grid
        + lambda_dup * loss_dup
    )

    with torch.no_grad():
        count_mode = count_logits.argmax(dim=-1)
        count_expected = (
            count_prob * count_values.unsqueeze(0)
        ).sum(dim=-1)
        count_expected_round = count_expected.round().long()
        count_acc = (count_mode == count_target).float().mean()
        count_topk_k = min(5, count_logits.shape[-1])
        count_topk = count_logits.topk(count_topk_k, dim=-1).indices
        count_topk_acc = (
            count_topk == count_target.unsqueeze(-1)
        ).any(dim=-1).float().mean()
        count_expected_error = count_expected - count_target.float()
        count_mode_error = count_mode.float() - count_target.float()
        count_entropy = -(
            count_prob * count_log_prob
        ).sum(dim=-1).mean()

        if coordinate_mode == "joint_dense":
            grid_log_prob_diag = log_grid
            grid_prob_diag = grid_log_prob_diag.exp()
            joint_grid_entropy = -(
                grid_prob_diag * grid_log_prob_diag
            ).sum(dim=-1).mean()
            joint_grid_max_probability = grid_prob_diag.max(dim=-1).values.mean()
        else:
            pt_diag = logpt.exp()
            ph_diag = logph.exp()
            pw_diag = logpw.exp()
            joint_grid_entropy = (
                -(pt_diag * pt_diag.clamp_min(1e-12).log()).sum(dim=-1)
                -(ph_diag * ph_diag.clamp_min(1e-12).log()).sum(dim=-1)
                -(pw_diag * pw_diag.clamp_min(1e-12).log()).sum(dim=-1)
            ).mean()
            joint_grid_max_probability = (
                pt_diag.max(dim=-1).values
                * ph_diag.max(dim=-1).values
                * pw_diag.max(dim=-1).values
            ).mean()

        if len(matched_query) > 0:
            if coordinate_mode == "joint_dense":
                matched_coord_logits = grid_logits[b_idx, q_idx].float()
                matched_flat_top1 = (
                    matched_coord_logits.argmax(dim=-1) == matched_flat
                ).float().mean()
                top5_k = min(5, matched_coord_logits.shape[-1])
                matched_flat_top5 = (
                    matched_coord_logits.topk(top5_k, dim=-1).indices
                    == matched_flat.unsqueeze(-1)
                ).any(dim=-1).float().mean()
                matched_t_accuracy = matched_h_accuracy = matched_w_accuracy = (
                    matched_flat_top1.new_zeros(())
                )
                factorized_axis_metrics_available = matched_flat_top1.new_zeros(())
            else:
                matched_t_logits = t_logits[b_idx, q_idx].float()
                matched_h_logits = h_logits[b_idx, q_idx].float()
                matched_w_logits = w_logits[b_idx, q_idx].float()
                matched_t_accuracy = (
                    matched_t_logits.argmax(dim=-1) == matched_t
                ).float().mean()
                matched_h_accuracy = (
                    matched_h_logits.argmax(dim=-1) == matched_h
                ).float().mean()
                matched_w_accuracy = (
                    matched_w_logits.argmax(dim=-1) == matched_w
                ).float().mean()
                matched_joint_logits = (
                    F.log_softmax(matched_t_logits, dim=-1)[:, :, None, None]
                    + F.log_softmax(matched_h_logits, dim=-1)[:, None, :, None]
                    + F.log_softmax(matched_w_logits, dim=-1)[:, None, None, :]
                ).reshape(matched_flat.shape[0], activity_prior.Ntok)
                matched_flat_top1 = (
                    matched_joint_logits.argmax(dim=-1) == matched_flat
                ).float().mean()
                top5_k = min(5, matched_joint_logits.shape[-1])
                matched_flat_top5 = (
                    matched_joint_logits.topk(top5_k, dim=-1).indices
                    == matched_flat.unsqueeze(-1)
                ).any(dim=-1).float().mean()
                factorized_axis_metrics_available = matched_flat_top1.new_ones(())
        else:
            zero_metric = event_logits.new_zeros((), dtype=torch.float32)
            matched_flat_top1 = zero_metric
            matched_flat_top5 = zero_metric
            matched_t_accuracy = zero_metric
            matched_h_accuracy = zero_metric
            matched_w_accuracy = zero_metric
            factorized_axis_metrics_available = zero_metric

    aux = {
        "loss": loss.detach(),
        "loss_count": loss_count.detach(),
        "loss_count_exact": loss_count_exact.detach(),
        "loss_count_neighbor": loss_count_neighbor.detach(),
        "loss_count_distance": loss_count_distance.detach(),
        "loss_obj": loss_obj.detach(),
        "loss_coord": loss_coord.detach(),
        "loss_t": loss_t.detach(),
        "loss_h": loss_h.detach(),
        "loss_w": loss_w.detach(),
        "loss_soft_count": loss_soft_count.detach(),
        "loss_soft_grid": loss_soft_grid.detach(),
        "loss_dup": loss_dup.detach(),
        "pred_count_mean": soft_count.detach().mean(),
        "event_soft_count_mean": soft_count.detach().mean(),
        "count_expected_mean": count_expected.detach().mean(),
        "count_mode_mean": count_mode.float().detach().mean(),
        "hard_count_mean": count_mode.float().detach().mean(),
        "target_count_mean": count_target.float().detach().mean(),
        "target_raw_count_mean": targets["raw_count"].float().detach().mean(),
        "count_acc": count_acc.detach(),
        "count_top5_acc": count_topk_acc.detach(),
        "count_expected_mae": count_expected_error.abs().mean().detach(),
        "count_expected_rmse": count_expected_error.square().mean().sqrt().detach(),
        "count_mode_mae": count_mode_error.abs().mean().detach(),
        "count_within_1": (
            (count_expected_round - count_target).abs() <= 1
        ).float().mean().detach(),
        "count_within_3": (
            (count_expected_round - count_target).abs() <= 3
        ).float().mean().detach(),
        "count_within_5": (
            (count_expected_round - count_target).abs() <= 5
        ).float().mean().detach(),
        "count_entropy": count_entropy.detach(),
        "matched_flat_top1_acc": matched_flat_top1.detach(),
        "matched_flat_top5_acc": matched_flat_top5.detach(),
        "matched_t_acc": matched_t_accuracy.detach(),
        "matched_h_acc": matched_h_accuracy.detach(),
        "matched_w_acc": matched_w_accuracy.detach(),
        "factorized_axis_metrics_available": factorized_axis_metrics_available.detach(),
        "joint_grid_entropy": joint_grid_entropy.detach(),
        "joint_grid_max_probability": joint_grid_max_probability.detach(),
        "count_head_event_gap": (
            count_expected - soft_count
        ).abs().mean().detach(),
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
        z1_codebook: Optional[torch.Tensor] = None,
        z2_codebook: Optional[torch.Tensor] = None,
        z1_scale: float = 1.0,
        z2_scale: float = 1.0,
        hull_margin_fraction: float = 0.0,
    ):
        super().__init__()
    
        self.K1 = int(K1)
        self.K2 = int(K2)
        self.num_tasks = int(num_tasks)
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.z1_scale = float(z1_scale)
        self.z2_scale = float(z2_scale)
        self.hull_margin_fraction = float(hull_margin_fraction)

        if z1_codebook is None or z2_codebook is None:
            raise ValueError("MaskGITMotifPrior requires frozen z1/z2 codebooks.")
        z1_cb = z1_codebook.detach().float().clone()
        z2_cb = z2_codebook.detach().float().clone()
        if z1_cb.shape[0] != self.K1 or z2_cb.shape[:2] != (self.K1, self.K2):
            raise ValueError(
                f"Codebook shape mismatch: z1={tuple(z1_cb.shape)}, "
                f"z2={tuple(z2_cb.shape)}, expected K1={self.K1}, K2={self.K2}."
            )
        self.register_buffer("z1_codebook", z1_cb, persistent=True)
        self.register_buffer("z2_codebook", z2_cb, persistent=True)

        z1_scaled = self.z1_scale * z1_cb
        z1_d2 = torch.cdist(z1_scaled, z1_scaled, p=2).pow(2)
        nonzero = z1_d2[z1_d2 > 0]
        norm = nonzero.mean() if nonzero.numel() else z1_d2.new_tensor(1.0)
        self.register_buffer(
            "z1_distance_matrix",
            z1_d2 / norm.clamp_min(1e-8),
            persistent=True,
        )
    
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
        # Deterministic convex-hull coefficient predictor.  The historical
        # name ``alpha_mu_head`` is retained so checkpoints produced by the
        # current Stage 3A implementation load strictly, but its output is
        # simply the alpha logits; there is no learned variance head.
        self.alpha_mu_head = nn.Linear(d_model, K2)
    
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
        alpha_in: Optional[torch.Tensor] = None,
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

        if alpha_in is not None:
            if alpha_in.shape != (*z2_in.shape, self.K2):
                raise ValueError(
                    f"alpha_in must have shape {(*z2_in.shape, self.K2)}, "
                    f"got {tuple(alpha_in.shape)}"
                )
            alpha_in = alpha_in.to(device=z2_e.device, dtype=z2_e.dtype)
            alpha_e = alpha_in @ self.z2_emb.weight[:self.K2]
            visible_alpha = z2_in.ge(0) & z2_in.lt(self.K2)
            z2_e = torch.where(visible_alpha.unsqueeze(-1), alpha_e, z2_e)
    
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
        alpha_logits = self.alpha_mu_head(h_z2)  # (B,N,K2)
        alpha_mean = F.softmax(alpha_logits, dim=-1)

        return {
            "z1": z1_logits,
            # Compatibility aliases: existing decoder/training code expects
            # ``z2`` and ``alpha_mu`` to contain K2 alpha logits.
            "z2": alpha_logits,
            "alpha_logits": alpha_logits,
            "alpha_mu": alpha_logits,
            "alpha_mean": alpha_mean,
            # Temporary output compatibility for the current sampler.  This
            # is a constant, non-parameter tensor and therefore does not add
            # alpha_log_std_head keys to the state dict.  exp(-20) makes the
            # old logistic-normal sampling path effectively deterministic.
            "alpha_log_std": torch.full_like(alpha_logits, -20.0),
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
        activity_prob: torch.Tensor,
        global_ctx: torch.Tensor,
        local_ctx: torch.Tensor,
        task_id: torch.Tensor,
        roi_mask: Optional[torch.Tensor] = None,
        alpha_in: Optional[torch.Tensor] = None,
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

        if alpha_in is not None:
            expected_shape = (*z2_in.shape, self.K2)
            if alpha_in.shape != expected_shape:
                raise ValueError(
                    f"alpha_in must have shape {expected_shape}, "
                    f"got {tuple(alpha_in.shape)}"
                )

            alpha_in = alpha_in.to(
                device=z2_e.device,
                dtype=z2_e.dtype,
            )

            alpha_e = (
                alpha_in @ self.z2_emb.weight[:self.K2]
            )  # (B,N,D)

            has_alpha = alpha_in.sum(dim=-1, keepdim=True).gt(0)
            z2_e = torch.where(has_alpha, alpha_e, z2_e)
    
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
        alpha_in: Optional[torch.Tensor] = None,
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
        
        x_tok = self._embed_layer_streams(
            a_in, z1_in, z2_in, roi_mask=roi_mask, alpha_in=alpha_in
        )
        
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
        z1_loss_mask = targets.get("z1_loss_mask", targets["z_loss_mask"]).bool()
        alpha_loss_mask = targets.get(
            "alpha_loss_mask",
            targets.get("z2_loss_mask", targets["z_loss_mask"]),
        ).bool()

        ignore_z1 = torch.full_like(z1_t, -100)
        z1_target = torch.where(z1_loss_mask, z1_t, ignore_z1)

        if z1_loss_mask.any():
            loss_z1 = F.cross_entropy(
                logits["z1"].reshape(-1, self.K1),
                z1_target.reshape(-1),
                ignore_index=-100,
            )
        else:
            loss_z1 = logits["z1"].sum() * 0.0

        alpha_target = targets.get("alpha", None)
        if alpha_target is None:
            raise KeyError("Stage 3 motif targets must include exact convex alpha.")
        alpha_target = alpha_target.to(
            device=logits["alpha_mu"].device,
            dtype=logits["alpha_mu"].dtype,
        )

        if alpha_loss_mask.any():
            eps = 1e-8
            a_t = alpha_target[alpha_loss_mask].float().clamp_min(eps)
            a_t = a_t / a_t.sum(dim=-1, keepdim=True).clamp_min(eps)

            alpha_logits = logits["alpha_mu"][alpha_loss_mask].float()
            alpha_log_prob = F.log_softmax(alpha_logits, dim=-1)
            alpha_mean = alpha_log_prob.exp()

            # Deterministic distribution matching used by the completed 3A:
            # KL(alpha_target || alpha_pred).
            loss_alpha_kl = (
                a_t * (a_t.log() - alpha_log_prob)
            ).sum(dim=-1).mean()

            parent = z1_t[alpha_loss_mask].clamp(0, self.K1 - 1)
            children = (
                self.z2_scale
                * (1.0 + max(0.0, self.hull_margin_fraction))
                * self.z2_codebook[parent]
            ).float()
            r_pred = torch.einsum("mk,mkd->md", alpha_mean, children)
            r_tgt = torch.einsum("mk,mkd->md", a_t, children)
            loss_alpha_residual = F.mse_loss(r_pred, r_tgt)
            residual_den = r_tgt.pow(2).mean().clamp_min(eps)
            loss_alpha_residual_nmse = (
                loss_alpha_residual / residual_den
            )

            alpha_mae = (alpha_mean - a_t).abs().mean()
            alpha_target_entropy = -(a_t * a_t.log()).sum(dim=-1).mean()
            alpha_pred_entropy = -(
                alpha_mean * alpha_log_prob
            ).sum(dim=-1).mean()

            loss_alpha = loss_alpha_kl + loss_alpha_residual
        else:
            zero = logits["alpha_mu"].sum() * 0.0
            loss_alpha_kl = zero
            loss_alpha_residual = zero
            loss_alpha_residual_nmse = zero
            alpha_mae = zero
            alpha_target_entropy = zero
            alpha_pred_entropy = zero
            loss_alpha = zero

        w1, w2 = loss_weights
        loss = w1 * loss_z1 + w2 * loss_alpha
    
        aux = {
            "loss": loss.detach(),
            "loss_z1": loss_z1.detach(),
            "loss_z2": loss_alpha.detach(),
            "loss_alpha": loss_alpha.detach(),
            "loss_alpha_kl": loss_alpha_kl.detach(),
            "loss_alpha_residual": loss_alpha_residual.detach(),
            "loss_alpha_residual_nmse": loss_alpha_residual_nmse.detach(),
            "alpha_mae": alpha_mae.detach(),
            "alpha_target_entropy": alpha_target_entropy.detach(),
            "alpha_pred_entropy": alpha_pred_entropy.detach(),
            "z1_loss_tokens": z1_loss_mask.sum().detach(),
            "z2_loss_tokens": alpha_loss_mask.sum().detach(),
            "alpha_loss_tokens": alpha_loss_mask.sum().detach(),
        }
    
        return logits, loss, aux

    def make_targets_from_codes(
        self,
        codes: torch.Tensor,
        predict_mask: torch.Tensor,
        blank_code: int = -1,
        alpha: Optional[torch.Tensor] = None,
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
    
        if alpha is None:
            alpha = F.one_hot(
                z2.clamp(0, self.K2 - 1), num_classes=self.K2
            ).to(dtype=torch.float32)
        else:
            alpha = alpha.to(device=codes.device, dtype=torch.float32)
            if alpha.shape != (*z1.shape, self.K2):
                raise ValueError(
                    f"alpha must have shape {(*z1.shape, self.K2)}, "
                    f"got {tuple(alpha.shape)}"
                )

        return {
            "a": a,
            "z1": z1,
            "z2": z2,
            "alpha": alpha,
            "active": active,
            "predict_mask": pmask,
            "a_loss_mask": pmask,
            "z1_loss_mask": pmask & active,
            "z2_loss_mask": pmask & active,
            "alpha_loss_mask": pmask & active,
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
        alpha = targets["alpha"].float()
    
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
        alpha_in = alpha.clone()
    
        # Inactive/blank positions should not expose fake clamped code 0.
        z1_in[~active] = self.z1_null_id
        z2_in[~active] = self.z2_null_id
        alpha_in[~active] = 0.0
    
        # Coarse-stage masking.
        # Activity is externally supplied / teacher-forced.
        # Do not mask a.
        z1_in[m] = self.z1_mask_id
    
        # Fine-stage masking.
        z2_in[m] = self.z2_mask_id
        alpha_in[m] = 0.0
    
        targets = dict(targets)
        targets["a_loss_mask"] = m
        targets["z1_loss_mask"] = m & active
        targets["z2_loss_mask"] = m & active
        targets["alpha_loss_mask"] = m & active
        targets["z_loss_mask"] = targets["z1_loss_mask"] | targets["alpha_loss_mask"]
        targets["gamma"] = gamma.squeeze(1)
    
        return a_in, z1_in, z2_in, alpha_in, targets
    
    
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