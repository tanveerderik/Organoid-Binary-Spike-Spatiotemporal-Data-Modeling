from typing import Optional, Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalTokenMGITTransformer(nn.Module):
    """MaskGIT-style hierarchical prior over VQ-VAE latent tokens.
    
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
        logits["a"]  : (B,N,2)
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
        d_model: int = 512,
        n_layer: int = 8,
        n_head: int = 8,
        max_len: int = 4096,
        dropout: float = 0.1,
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
    
        # Conditional embeddings for staged hierarchy
        self.a_cond_emb = nn.Embedding(2, d_model)        # blank / active
        self.z1_cond_null_id = self.K1
        self.z1_cond_emb = nn.Embedding(K1 + 1, d_model) # z1 codes + null
        
        # Staged prediction heads
        self.a_head = nn.Linear(d_model, 2)
        self.z1_head = nn.Linear(2 * d_model, K1)  # [h, a]
        self.z2_head = nn.Linear(3 * d_model, K2)  # [h, a, z1]
    
        # Prefix context tokens
        self.task_emb = nn.Embedding(num_tasks, d_model)
        self.gct_proj = nn.Linear(gct_latent_dim, d_model)
        self.lct_proj = nn.Linear(lct_latent_dim, d_model)
    
        self.pos_emb = nn.Embedding(max_len + self.ctx_len, d_model)
        self.drop = nn.Dropout(dropout)
    
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
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
    
    
    def _teacher_condition_ids(self, targets: Dict[str, torch.Tensor]):
        a_used = targets["a"].long().clamp(0, 1)
    
        active = targets["active"].bool()
        z1_raw = targets["z1"].long().clamp(0, self.K1 - 1)
    
        z1_null = torch.full_like(z1_raw, self.z1_cond_null_id)
        z1_used = torch.where(active, z1_raw, z1_null)
    
        return a_used, z1_used
    
    
    def _pred_condition_ids(self, a_logits: torch.Tensor, z1_logits: Optional[torch.Tensor] = None):
        a_used = a_logits.argmax(dim=-1).long().clamp(0, 1)
    
        if z1_logits is None:
            return a_used, None
    
        z1_pred = z1_logits.argmax(dim=-1).long().clamp(0, self.K1 - 1)
        z1_null = torch.full_like(z1_pred, self.z1_cond_null_id)
    
        z1_used = torch.where(
            a_used.eq(self.a_active_id),
            z1_pred,
            z1_null,
        )
    
        return a_used, z1_used
    
    def forward(
        self,
        a_in: torch.LongTensor,
        z1_in: torch.LongTensor,
        z2_in: torch.LongTensor,
        *,
        global_ctx: torch.Tensor,
        local_ctx: torch.Tensor,
        task_id: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        loss_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        activity_pos_weight: float = 20,
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
    
        prefix = self._build_prefix(global_ctx, local_ctx, task_id)
    
        x_tok = self.a_emb(a_in) + self.z1_emb(z1_in) + self.z2_emb(z2_in)
        x = torch.cat([prefix, x_tok], dim=1)
    
        pos = torch.arange(self.ctx_len + N, device=x.device)
        x = x + self.pos_emb(pos).unsqueeze(0)
        x = self.drop(x)
    
        key_padding_mask = None
            
        h = self.blocks(x, src_key_padding_mask=key_padding_mask)
        h = self.ln_f(h[:, self.ctx_len:, :])
        
        # Stage 1: p(a | h)
        a_logits = self.a_head(h)
        
        # Stage 2: p(z1 | h, a)
        if targets is not None:
            a_used, z1_used_teacher = self._teacher_condition_ids(targets)
        else:
            a_used, _ = self._pred_condition_ids(a_logits)
        
        a_cond = self.a_cond_emb(a_used)
        z1_logits = self.z1_head(torch.cat([h, a_cond], dim=-1))
        
        # Stage 3: p(z2 | h, a, z1)
        if targets is not None:
            z1_used = z1_used_teacher
        else:
            _, z1_used = self._pred_condition_ids(a_logits, z1_logits)
        
        z1_cond = self.z1_cond_emb(z1_used)
        z2_logits = self.z2_head(torch.cat([h, a_cond, z1_cond], dim=-1))
        
        logits = {
            "a": a_logits,
            "z1": z1_logits,
            "z2": z2_logits,
        }
            
        if targets is None:
            return logits, None, {}
    
        a_t = targets["a"].long()
        z1_t = targets["z1"].long()
        z2_t = targets["z2"].long()
    
        a_loss_mask = targets["a_loss_mask"].bool()
        z1_loss_mask = targets.get("z1_loss_mask", targets["z_loss_mask"]).bool()
        z2_loss_mask = targets.get("z2_loss_mask", targets["z_loss_mask"]).bool()
    
        ignore_a = torch.full_like(a_t, -100)
        ignore_z1 = torch.full_like(z1_t, -100)
        ignore_z2 = torch.full_like(z2_t, -100)
    
        a_target = torch.where(a_loss_mask, a_t, ignore_a)
        z1_target = torch.where(z1_loss_mask, z1_t, ignore_z1)
        z2_target = torch.where(z2_loss_mask, z2_t, ignore_z2)
        
        activity_class_weight = torch.tensor(
            [1.0, activity_pos_weight],
            device=logits["a"].device,
            dtype=logits["a"].dtype,
        )
        
        loss_a = F.cross_entropy(
            logits["a"].reshape(-1, 2),
            a_target.reshape(-1),
            ignore_index=-100,
            weight=activity_class_weight,
        )
    
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
    
        wa, w1, w2 = loss_weights
        loss = wa * loss_a + w1 * loss_z1 + w2 * loss_z2
    
        aux = {
            "loss": loss.detach(),
            "loss_a": loss_a.detach(),
            "loss_z1": loss_z1.detach(),
            "loss_z2": loss_z2.detach(),
            "a_loss_tokens": a_loss_mask.sum().detach(),
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
        mode_probs=(0.50, 0.25, 0.25),
    ):
        """
        Hierarchy-aware corruption.

        mode 0: mask a,z1,z2        -> learn p(a)
        mode 1: reveal a, mask z1,z2 -> learn p(z1 | a)
        mode 2: reveal a,z1, mask z2 -> learn p(z2 | a,z1)
        """
        a = targets["a"].long()
        z1 = targets["z1"].long()
        z2 = targets["z2"].long()
        active = targets["active"].bool()
        pmask = targets["predict_mask"].bool()

        a_in = a.clone()
        z1_in = z1.clone()
        z2_in = z2.clone()

        # visible blanks should not expose fake clamped code 0
        z1_in[~active] = self.z1_null_id
        z2_in[~active] = self.z2_null_id

        B, N = a.shape
        device = a.device

        mode = torch.multinomial(
            torch.tensor(mode_probs, device=device, dtype=torch.float),
            num_samples=B * N,
            replacement=True,
        ).view(B, N)

        # only apply hierarchy modes on prediction positions
        m0 = pmask & (mode == 0)
        m1 = pmask & (mode == 1)
        m2 = pmask & (mode == 2)

        # mode 0: predict activity first
        a_in[m0] = self.a_mask_id
        z1_in[m0] = self.z1_mask_id
        z2_in[m0] = self.z2_mask_id

        # mode 1: reveal a, predict z1/z2 for active tokens
        z1_in[m1] = self.z1_mask_id
        z2_in[m1] = self.z2_mask_id
        z1_in[m1 & ~active] = self.z1_null_id
        z2_in[m1 & ~active] = self.z2_null_id

        # mode 2: reveal a,z1, predict z2 for active tokens
        z2_in[m2] = self.z2_mask_id
        z2_in[m2 & ~active] = self.z2_null_id

        # update loss masks
        targets = dict(targets)
        targets["a_loss_mask"] = m0
        targets["z1_loss_mask"] = (m0 | m1) & active
        targets["z2_loss_mask"] = (m0 | m1 | m2) & active

        # keep old name for compatibility if needed
        targets["z_loss_mask"] = targets["z1_loss_mask"] | targets["z2_loss_mask"]

        return a_in, z1_in, z2_in, targets