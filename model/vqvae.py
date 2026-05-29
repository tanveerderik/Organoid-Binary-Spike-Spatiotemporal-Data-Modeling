#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:26:03 2026

@author: derik
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import copy
import numpy as np

from .base import ConvStem3D, ActivePatchEmbed3D, PatchRenderer3D, CtxEmbed
from .base import SparseTokenTransformerEncoder, DecoderCrossAttnBlock, HierarchicalVectorQuantizerEMA
from .spatial_map import SpatialMapPrior
from ..utils.embed import get_3d_sincos_pos_embed


    
class TransformerVQVAE(nn.Module):
    def __init__(
        self,
        img_size,
        full_spatial_size=None,
        patch_size=(16,16,16),
        encoder_embed_dim=256, encoder_depth=8, encoder_num_heads=6, mlp_ratio=4.0,
        
        code_dim=128, num_codes=(16,64),
        vq_decay: float = 0.95,
        vq_beta: float = 0.25,
        usage_loss_weight: float = 1e-3,
        usage_tau: float = 0.5,
        num_quantizers: int = 2,
        
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=6,
        
        
        in_chans=1, out_chans=1,

        # ---- contexts ----

        # This is the raw input global_ctx dim coming from dataset
        # (yours is currently 2 = assay(2) + ...)
        global_ctx_in_dim: int = 2,
        global_emb_dim: int = 16,
        local_ctx_in_dim: int = 5,
        local_emb_dim: int = 16,
        ctx_hidden_mult: float = 2.0,

        drop=0.0, attn_drop=0.0,
        
        # ---- gct-conditioned spatial map bias ----
        use_spatial_map_prior: bool = True,

        # context dropout (helps prevent shortcutting) and CFG knobs
        ctx_drop_p=0.25,
        cfg_ctx_drop_p: float = 0.0,
        
        use_decoder_cross_attn: bool = False,
        decoder_cross_attn_layers: tuple = (),
        
        # ---- decoder attention masking ----
        enc_attn_mask_kind: str = "temporal_band_bi",  # "none" | "temporal_causal" | "temporal_band" | "temporal_band_bi"
        enc_attn_window: int = 5,                      # window length in patch-time tokens
        dec_attn_mask_kind: str = "temporal_band_bi",  # "none" | "temporal_causal" | "temporal_band" | "temporal_band_bi"
        dec_attn_window: int = 5,                      # window length in patch-time tokens
        
        
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.img_size = img_size
        self.full_spatial_size = None if full_spatial_size is None else tuple(map(int, full_spatial_size))
        
        self.encoder_embed_dim = int(encoder_embed_dim)
        self.decoder_embed_dim = int(decoder_embed_dim)
        
        imgT, imgH, imgW = map(int, self.img_size)
        pT, pH, pW = map(int, self.patch_size)
        
        if self.full_spatial_size is None:
            self.full_spatial_size = (imgH, imgW)
        
        if (imgT % pT) != 0 or (imgH % pH) != 0 or (imgW % pW) != 0:
            raise ValueError(
                f"img_size {self.img_size} must be divisible by patch_size {self.patch_size}"
            )
        
        self.token_grid = (imgT // pT, imgH // pH, imgW // pW)   # (t_tok, h_tok, w_tok)
        self.num_tokens = self.token_grid[0] * self.token_grid[1] * self.token_grid[2]

        # ---- remember ctx settings ----
        self.global_ctx_in_dim = int(global_ctx_in_dim)
        self.global_emb_dim = int(global_emb_dim)
        self.local_ctx_in_dim = int(local_ctx_in_dim)
        self.local_emb_dim = int(local_emb_dim)
        
        self.dec_attn_mask_kind = str(dec_attn_mask_kind)
        self.dec_attn_window = int(dec_attn_window) if dec_attn_window is not None else None
        
        

        # --- patchify & positional encodings ---
        
        self.use_stem = True

        if self.use_stem:
            self.stem = ConvStem3D(in_chans=in_chans, out_chans=8)
        else:
            self.stem = nn.Identity()
        
        
        self.patch_embed = ActivePatchEmbed3D(
            in_chans=8,
            embed_dim=encoder_embed_dim,
            patch_size=patch_size,
            bias=True,
        )
        pT, pH, pW = patch_size
        

        # we keep patch_size only for patchify/unpatchify compatibility        
        self.register_buffer("_pos_cache_enc", torch.empty(0), persistent=False)
        self.register_buffer("_pos_cache_dec", torch.empty(0), persistent=False)
        self._pos_cache_key = None

        # --- encoder ---
        
        self.sparse_encoder = SparseTokenTransformerEncoder(
            dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            drop=0.0,
            attn_drop=0.0,
            activation="gelu",
            norm_first=True,
        )

        # --- code space & VQ ---
        self.to_code = nn.Sequential(
            nn.Linear(encoder_embed_dim, code_dim, bias=True),
            nn.LayerNorm(code_dim),
        )
        
        self.num_quantizers = int(num_quantizers)
        
        self.vq = HierarchicalVectorQuantizerEMA(
            num_codes=num_codes,
            code_dim=code_dim,
            decay=vq_decay,
            eps=1e-5,
            beta=vq_beta,
            blank_code=-1,
            blank_token_std=0.02,
            usage_loss_weight=usage_loss_weight,
            usage_tau=usage_tau,
            num_quantizers=self.num_quantizers,
        
            # hierarchy control
            active_quantizers=2,
            ema_norm_cap = 15.0,
        
            # duplicate restart
            duplicate_restart_every = 200,
            duplicate_rel_dist_thresh = 0.05,
            duplicate_restart_noise_std = 0.01,
        
            # dead-code restart
            dead_code_restart_every=100,
            dead_code_usage_thresh=0.05,
            dead_restart_noise_std=0.01,
        )

        # --- decoder ---
        self.code_to_dec = nn.Linear(code_dim, decoder_embed_dim, bias=False)
        self.activity_type_offset = nn.Parameter(torch.zeros(decoder_embed_dim))
        self.offset_scale = 0.5
        nn.init.normal_(self.activity_type_offset, mean=0.0, std=0.02)
        
        self.dec_blocks = nn.ModuleList([
            DecoderCrossAttnBlock(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                attn_mask_kind=dec_attn_mask_kind,
            )
            for _ in range(decoder_depth)
        ])
        self.dec_norm = nn.LayerNorm(decoder_embed_dim)

        pT, pH, pW = patch_size
        self.patch_dim = pT * pH * pW * out_chans
        
        self.patch_renderer = PatchRenderer3D(
            token_dim=decoder_embed_dim,
            out_chans=out_chans,
            patch_size=patch_size,
            refine_layers=0,      # start with 0 first
            refine_hidden=32,
        )
        
        # Optional inverse-like initialization from patch embed
        # self.patch_renderer.init_from_patch_embed(self.patch_embed)

        # ---- Context paths ----


        # 1) globaland local embedding branch (assay-only or assay+task)
        #    This produces a compact vector you can concat into the FiLM ctx.
        self.local_embedder  = CtxEmbed(self.local_ctx_in_dim,  self.local_emb_dim,  mlp_ratio=2.0, drop=0.0, alpha_init=0.0)
        self.global_embedder = CtxEmbed(self.global_ctx_in_dim, self.global_emb_dim, mlp_ratio=2.0, drop=0.0, alpha_init=0.1)

        # 2) Cross attention setup
        # context -> decoder cross-attn tokens
        self.ctx_dropout = nn.Dropout(ctx_drop_p)
        
        self.local_to_dec_ctx = nn.Linear(self.local_emb_dim, decoder_embed_dim, bias=True)
        self.global_to_dec_ctx = nn.Linear(self.global_emb_dim, decoder_embed_dim, bias=True)
        
        # use cross-attn only in early decoder blocks
        self.use_decoder_cross_attn = bool(use_decoder_cross_attn)
        self.decoder_cross_attn_layers = set(decoder_cross_attn_layers) if self.use_decoder_cross_attn else set()


        # global bias toward sparsity
        self.register_buffer("global_logit_bias", torch.tensor(0.0, dtype=torch.float32))
        
        # global context based spatial support prior
        self.use_spatial_map_prior = bool(use_spatial_map_prior)
        self.use_learned_spatial_prior_diagnostics = False
        
        if self.use_spatial_map_prior:
            self.spatial_map_prior = SpatialMapPrior(
                global_emb_dim=self.global_emb_dim,
                patch_dim=self.patch_dim,
                full_spatial_size=self.full_spatial_size,
                patch_size_hw=(pH, pW),
                drop=0.1,
                basis_k=32,
            )
        else:
            self.spatial_map_prior = None
            

        self.best_thr = 0.5
        
        self.cfg_ctx_drop_p = float(cfg_ctx_drop_p)

    
    def set_logit_bias(self, value: float):
        self.global_logit_bias.fill_(float(value))
    
        
    def _global_emb_only(self, global_ctx: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if global_ctx is None:
            return None

        if global_ctx.dim() != 2 or global_ctx.size(1) != self.global_ctx_in_dim:
            raise ValueError(
                f"global_ctx must be (B,{self.global_ctx_in_dim}), got {tuple(global_ctx.shape)}"
            )

        g_param = next(self.global_embedder.parameters())
        g_in = global_ctx.to(dtype=g_param.dtype, device=global_ctx.device)
        g_emb = self.global_embedder(g_in)

        # cast back to input dtype for consistency
        return g_emb.to(dtype=global_ctx.dtype, device=global_ctx.device)
    
    def _prepare_ctx_tokens(
        self,
        local_ctx: Optional[torch.Tensor],
        global_ctx: Optional[torch.Tensor],
        *,
        target_dtype: torch.dtype,
        target_device: torch.device,
        cfg_ctx_drop_p: Optional[float] = None,
        cfg_ctx_force_unc: Optional[bool] = None,
    ):
        """
        Build small context token set for decoder cross-attention.

        Returns:
          ctx_tokens: (B, M, D_dec), where M in {0,1,2}
          ctx_key_padding_mask: (B, M) bool, True = mask out / ignore token

          token order:
            0: local context token (if present)
            1: global context token (if present)
        """
        parts = []
        B = None

        if local_ctx is not None:
            if local_ctx.dim() != 2 or local_ctx.size(1) != self.local_ctx_in_dim:
                raise ValueError(f"local_ctx must be (B,{self.local_ctx_in_dim}), got {tuple(local_ctx.shape)}")
            B = local_ctx.size(0)

            l_param = next(self.local_embedder.parameters())
            lc_in = local_ctx.to(device=target_device, dtype=l_param.dtype)
            l_emb = self.local_embedder(lc_in)
            l_tok = self.local_to_dec_ctx(l_emb).to(device=target_device, dtype=target_dtype)
            parts.append(l_tok.unsqueeze(1))  # (B,1,D)

        if global_ctx is not None:
            if global_ctx.dim() != 2 or global_ctx.size(1) != self.global_ctx_in_dim:
                raise ValueError(f"global_ctx must be (B,{self.global_ctx_in_dim}), got {tuple(global_ctx.shape)}")
            B = global_ctx.size(0) if B is None else B

            g_param = next(self.global_embedder.parameters())
            gc_in = global_ctx.to(device=target_device, dtype=g_param.dtype)
            g_emb = self.global_embedder(gc_in)
            g_tok = self.global_to_dec_ctx(g_emb).to(device=target_device, dtype=target_dtype)
            parts.append(g_tok.unsqueeze(1))  # (B,1,D)

        if not parts:
            return None, None

        ctx_tokens = torch.cat(parts, dim=1)  # (B,M,D)

        if self.ctx_dropout is not None:
            ctx_tokens = self.ctx_dropout(ctx_tokens)

        M = ctx_tokens.size(1)
        ctx_key_padding_mask = torch.zeros((ctx_tokens.size(0), M), device=target_device, dtype=torch.bool)

        p_drop = self.cfg_ctx_drop_p if cfg_ctx_drop_p is None else float(cfg_ctx_drop_p)

        if cfg_ctx_force_unc is True:
            # fully unconditional: hide all context tokens
            ctx_key_padding_mask[:] = True

        elif cfg_ctx_force_unc is False:
            # fully conditional: keep all context tokens
            pass

        elif self.training and p_drop > 0.0:
            # classifier-free dropout per sample:
            # with probability p_drop, hide all context tokens for that sample
            drop_rows = (torch.rand((ctx_tokens.size(0),), device=target_device) < p_drop)
            ctx_key_padding_mask[drop_rows] = True

        # Safety:
        # if every ctx token is masked for every sample, return no context at all
        if bool(ctx_key_padding_mask.all().item()):
            return None, None

        return ctx_tokens, ctx_key_padding_mask
    

    def _apply_output_biases(
        self,
        pred_patches: torch.Tensor,
        grid: Tuple[int, int, int],
        global_ctx: Optional[torch.Tensor] = None,
        roi_hw=None,
        pad_hw=None,
    ):
        # existing global scalar sparsity prior
        pred_patches = pred_patches + self.global_logit_bias.to(dtype=pred_patches.dtype, device=pred_patches.device)
    
        spatial_diag = None
    
        # assay-conditioned spatial prior
        if (
            self.spatial_map_prior is not None
            and global_ctx is not None
            and getattr(self, "use_learned_spatial_prior_diagnostics", False)
        ):
            g_emb = self._global_emb_only(global_ctx)
            sp = self.spatial_map_prior(
                g_emb,
                grid=grid,
                roi_hw=roi_hw,
                pad_hw=pad_hw,
            )
    
            spatial_diag = sp
    
        return pred_patches, spatial_diag
        
    def _set_best_thr(self, best_thr):
        self.best_thr = best_thr

    # ----- reuse your pos-embed cache helper -----
    def _get_pos_embed(self, grid, device, dtype):
        t, h, w = grid
        key = (t, h, w, self.encoder_embed_dim, self.decoder_embed_dim)
    
        if self._pos_cache_key != key or self._pos_cache_enc.numel() == 0:
            pe_enc = get_3d_sincos_pos_embed(
                self.encoder_embed_dim, t, h, w, device=device, dtype=dtype
            )
            pe_dec = get_3d_sincos_pos_embed(
                self.decoder_embed_dim, t, h, w, device=device, dtype=dtype
            )
            self._pos_cache_enc = pe_enc
            self._pos_cache_dec = pe_dec
            self._pos_cache_key = key
    
        return (
            self._pos_cache_enc.to(device=device, dtype=dtype),
            self._pos_cache_dec.to(device=device, dtype=dtype),
        )


    def patchify(self, vol: torch.Tensor, grid: Tuple[int,int,int]) -> torch.Tensor:
        """
        Inverse of unpatchify. Takes (B,C,T,H,W) -> (B,N,PD) using self.patch_size and grid.
        Assumes T = t*pT, H = h*pH, W = w*pW.
        """
        pT, pH, pW = self.patch_size
        B, C, T, H, W = vol.shape
        t, h, w = grid
        assert T == t * pT and H == h * pH and W == w * pW, \
            f"vol shape {(T,H,W)} incompatible with grid {grid} and patch {self.patch_size}"
    
        x = vol.view(B, C, t, pT, h, pH, w, pW)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()   # (B,t,h,w,pT,pH,pW,C)
        return x.view(B, t*h*w, pT*pH*pW*C)

    def compute_blank_mask(self, x: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
        """
        Compute token-level blank mask from the ORIGINAL binary spike input.

        A patch is blank iff sum over all voxels in that patch == 0.

        Args:
          x:    (B, 1, T, H, W)
          grid: (t_tok, h_tok, w_tok)

        Returns:
          blank_mask: (B, N) bool
        """
        patch_tokens = self.patchify(x[:, :1], grid=grid)   # (B, N, patch_dim)
        blank_mask = patch_tokens.sum(dim=-1).eq(0)
        return blank_mask
    

    def unpatchify(self, patches, grid):
        pT,pH,pW = self.patch_size
        B,N,PD = patches.shape
        t,h,w = grid
        x = patches.view(B, t, h, w, pT, pH, pW, self.out_chans).permute(0,7,1,4,2,5,3,6).contiguous()
        return x.view(B, self.out_chans, t*pT, h*pH, w*pW)

    
    def _decode_quantized_latent(
        self,
        z_q: torch.Tensor,                 # (B,N,D_code)
        active_mask: torch.Tensor,         # (B,N) bool
        grid,
        global_ctx=None,
        local_ctx=None,
        cfg_ctx_drop_p=None,
        cfg_ctx_force_unc=None,
        roi_hw=None,
        pad_hw=None,
    ):
        _, pos_dec = self._get_pos_embed(grid, z_q.device, z_q.dtype)
    
        # ---------------------------------------------------------
        # Decoder-space latent states
        # ---------------------------------------------------------
        # 1) Base decoder latent, no type offset, no positional encoding.
        z_dec_base_no_pos = self.code_to_dec(z_q)
    
        # 2) Signed decoder-side activity axis.
        #    blank  = base - offset_scale * offset
        #    active = base + offset_scale * offset
        type_offset = self.activity_type_offset.to(
            device=z_dec_base_no_pos.device,
            dtype=z_dec_base_no_pos.dtype,
        )
    
        signed_type_offset = torch.where(
            active_mask.unsqueeze(-1),
            type_offset.view(1, 1, -1),
            -type_offset.view(1, 1, -1),
        )
    
        # 3) Type-offset decoder latent, still no positional encoding.
        z_dec_no_pos = z_dec_base_no_pos + self.offset_scale*signed_type_offset
    
        # 4) Full decoder input: type-offset latent + positional encoding.
        z_decoder_input = z_dec_no_pos + pos_dec
        z_d = z_decoder_input
    
        # ---------------------------------------------------------
        # Optional decoder context cross-attention
        # ---------------------------------------------------------
        ctx_tokens, ctx_key_padding_mask = self._prepare_ctx_tokens(
            local_ctx=local_ctx,
            global_ctx=global_ctx,
            target_dtype=z_d.dtype,
            target_device=z_d.device,
            cfg_ctx_drop_p=cfg_ctx_drop_p,
            cfg_ctx_force_unc=cfg_ctx_force_unc,
        )
    
        self._ensure_dec_masks(grid, device=z_d.device)
        for i, blk in enumerate(self.dec_blocks):
            use_cross = (i in self.decoder_cross_attn_layers)
            z_d = blk(
                z_d,
                ctx_tokens=ctx_tokens if use_cross else None,
                ctx_key_padding_mask=ctx_key_padding_mask if use_cross else None,
            )
    
        z_d = self.dec_norm(z_d)
        logits_vol_raw, pred_patches_raw = self.patch_renderer(
            z_d,
            grid,
            return_patches=True,
        )
    
        pred_patches, spatial_diag = self._apply_output_biases(
            pred_patches_raw,
            grid=grid,
            global_ctx=global_ctx,
            roi_hw=roi_hw,
            pad_hw=pad_hw,
        )
    
        logits_vol = self.unpatchify(pred_patches, grid)
    
        return {
            "z_q": z_q,
            "active_mask": active_mask,
    
            # no offset, no position
            "z_dec_base_no_pos": z_dec_base_no_pos,
    
            # signed offset tensors
            "activity_type_offset": type_offset,
            "signed_type_offset": signed_type_offset,
    
            # offset applied, no position
            "z_dec_no_pos": z_dec_no_pos,
    
            # offset + position, actual decoder input
            "z_decoder_input": z_decoder_input,
    
            "logits_vol": logits_vol,
            "logits_vol_raw": logits_vol_raw,
            "pred_patches": pred_patches,
            "pred_patches_raw": pred_patches_raw,
            "spatial_diag": spatial_diag,
        }
                
    
    def decode_from_codes(
        self,
        codes,
        grid,
        global_ctx=None,
        local_ctx=None,
        cfg_ctx_drop_p=None,
        cfg_ctx_force_unc=None,
        roi_hw=None,
        pad_hw=None,
        return_all_refinements: bool = False,
    ):
        """
        Decode hierarchical codes.
    
        Args:
          codes: (B,N,L) long or backward-compatible (B,N)
          return_all_refinements:
            False -> return final cumulative decode only
            True  -> return list of cumulative refinement decodes
        """
        
        L_active = int(self.vq.active_quantizers)
        
        refinements = []
        
        if return_all_refinements:
            
            z_q_list, active_mask_list = self._codes_to_quantized_cumulative(codes)
            z_q_list = z_q_list[:L_active]
            active_mask_list = active_mask_list[:L_active]

            for lvl_idx, (z_q_i, active_mask_i) in enumerate(
                zip(z_q_list[:-1], active_mask_list[:-1]),
                start=1,
            ):
                dec_i = self._decode_quantized_latent(
                    z_q=z_q_i,
                    active_mask=active_mask_i,
                    grid=grid,
                    global_ctx=global_ctx,
                    local_ctx=local_ctx,
                    cfg_ctx_drop_p=cfg_ctx_drop_p,
                    cfg_ctx_force_unc=cfg_ctx_force_unc,
                    roi_hw=roi_hw,
                    pad_hw=pad_hw,
                )
                dec_i["level"] = lvl_idx
                refinements.append(dec_i)
                
            z_q_final = z_q_list[-1]
            active_mask_final = active_mask_list[-1]
            
        else:
            z_q_final, active_mask_final = self._codes_to_quantized_final(codes)
    
        final_dec = self._decode_quantized_latent(
            z_q=z_q_final,
            active_mask=active_mask_final,
            grid=grid,
            global_ctx=global_ctx,
            local_ctx=local_ctx,
            cfg_ctx_drop_p=cfg_ctx_drop_p,
            cfg_ctx_force_unc=cfg_ctx_force_unc,
            roi_hw=roi_hw,
            pad_hw=pad_hw,
        )
        final_dec["level"] = L_active
        final_dec["grid"] = grid
        
        if return_all_refinements:
            final_dec["refinements"] = refinements + [final_dec]
    
        return final_dec
        
    @torch.no_grad()
    def predict_mask_from_spec(
        self,
        mask_spec: Union[Dict[str, Any], List[Dict[str, Any]]],
        grid: Tuple[int, int, int],
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Build token-level prediction/supervision mask from dataloader mask_spec.

        Returns:
          pmask_tok: (B, N, 1) float32 where 1 = supervise token, 0 = ignore token
        Notes:
          - Token flatten order matches PatchEmbed3D: Conv3d -> (B,D,T',H',W') -> flatten(2)
            i.e., w fastest, then h, then t.
          - prefix_frames / time_spans are in *frame units* after any pooling/cropping in dataset.
          - spatial_box is in *pixel units* (H/W) after any spatial cropping in dataset.
        """
        if device is None:
            device = next(self.parameters()).device

        if isinstance(mask_spec, dict):
            specs = [mask_spec]
        else:
            specs = list(mask_spec)

        t_tok, h_tok, w_tok = map(int, grid)
        N = t_tok * h_tok * w_tok

        pt, ph, pw = map(int, getattr(self, "patch_size", (1, 1, 1)))

        def _ones() -> torch.Tensor:
            return torch.ones((t_tok, h_tok, w_tok), device=device, dtype=dtype)

        pmasks: List[torch.Tensor] = []

        for spec in specs:
            stype = (spec.get("type", "recon") if isinstance(spec, dict) else "recon")

            if stype == "recon":
                pm = _ones()

            elif stype == "causal":
                # supervise suffix tokens, given prefix tokens
                pf_frames = int(spec.get("prefix_frames", 0))
                if t_tok <= 1:
                    pm = _ones()
                else:
                    # Map frames -> token time index.
                    # Use FLOOR to avoid "given" including partially-covered future token.
                    prefix_toks = pf_frames // max(1, pt)
                    prefix_toks = max(1, min(t_tok - 1, int(prefix_toks)))

                    pm_t = torch.zeros((t_tok,), device=device, dtype=dtype)
                    pm_t[prefix_toks:] = 1.0  # supervise suffix
                    pm = pm_t.view(t_tok, 1, 1).expand(t_tok, h_tok, w_tok).contiguous()

                    # Safety: avoid all-zero mask
                    if float(pm_t.sum().item()) <= 0.0:
                        pm = _ones()

            elif stype == "noncausal":
                # supervise masked time span tokens only
                spans = spec.get("time_spans", [])
                pm_t = torch.zeros((t_tok,), device=device, dtype=dtype)

                for (a, b) in spans:
                    a = int(a); b = int(b)
                    if b <= a:
                        continue

                    # token indices that overlap [a,b) in frames
                    # overlap if token range [k*pt,(k+1)*pt) intersects [a,b)
                    tok_a = a // max(1, pt)
                    tok_b = (b - 1) // max(1, pt)  # inclusive
                    tok_a = max(0, min(t_tok - 1, tok_a))
                    tok_b = max(0, min(t_tok - 1, tok_b))
                    if tok_b >= tok_a:
                        pm_t[tok_a:tok_b + 1] = 1.0

                pm = pm_t.view(t_tok, 1, 1).expand(t_tok, h_tok, w_tok).contiguous()

                # Safety: avoid all-zero mask
                if float(pm_t.sum().item()) <= 0.0:
                    pm = _ones()

            elif stype == "spatial":
                # supervise spatial box tokens for all times
                box = spec.get("spatial_box", None)
                if box is None:
                    pm = _ones()
                else:
                    y0, y1, x0, x1 = map(int, box)
                    # token indices that overlap [y0,y1) and [x0,x1)
                    ty0 = y0 // max(1, ph)
                    ty1 = (y1 - 1) // max(1, ph)
                    tx0 = x0 // max(1, pw)
                    tx1 = (x1 - 1) // max(1, pw)

                    ty0 = max(0, min(h_tok - 1, ty0))
                    ty1 = max(0, min(h_tok - 1, ty1))
                    tx0 = max(0, min(w_tok - 1, tx0))
                    tx1 = max(0, min(w_tok - 1, tx1))

                    pm = torch.zeros((t_tok, h_tok, w_tok), device=device, dtype=dtype)
                    if (ty1 >= ty0) and (tx1 >= tx0):
                        pm[:, ty0:ty1 + 1, tx0:tx1 + 1] = 1.0
                    else:
                        pm = _ones()

            else:
                # unknown spec => supervise all
                pm = _ones()

            pmasks.append(pm.reshape(N, 1))

        return torch.stack(pmasks, dim=0)  # (B, N, 1)


            
    def _ensure_dec_masks(self, grid: Tuple[int,int,int], device: torch.device):
        key_grid = tuple(grid)
        key_kind = getattr(self, "dec_attn_mask_kind", "temporal_band_bi")
        key_win  = getattr(self, "dec_attn_window", None)

        if (
            getattr(self, "_dec_mask_grid", None) != key_grid
            or getattr(self, "_dec_mask_kind", None) != key_kind
            or getattr(self, "_dec_mask_window", None) != key_win
        ):
            for blk in self.dec_blocks:
                blk.set_attn_mask(
                    grid,
                    attn_mask_kind=key_kind,
                    window=key_win,
                    device=device,
                    dtype=torch.float32
                )
            self._dec_mask_grid = key_grid
            self._dec_mask_kind = key_kind
            self._dec_mask_window = key_win

    def save_checkpoint(self, path, optimizer=None, scheduler=None, epoch=None):
        ckpt = {
            "model": self.state_dict(),
            "best_thr": float(getattr(self, "best_thr", 0.5)),
        }
        if optimizer is not None:
            ckpt["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            ckpt["scheduler"] = scheduler.state_dict()
        if epoch is not None:
            ckpt["epoch"] = epoch
        torch.save(ckpt, path)
        
    def load_checkpoint(self, path, map_location=None, optimizer=None, scheduler=None):
        ckpt = torch.load(path, map_location=map_location)
        self.load_state_dict(ckpt["model"])
        if "best_thr" in ckpt:
            self._set_best_thr(float(ckpt["best_thr"]))
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])


    def _codes_to_quantized_cumulative(
        self,
        codes: torch.Tensor,   # (B,N,L) or (B,N)
    ):
        """
        Build cumulative quantized latents for all hierarchy levels.
    
        Args:
          codes:
            (B,N,L) hierarchical code indices
            or backward-compatible (B,N), treated as L=1
    
        Returns:
          z_q_list:
            list of length L
            z_q_list[i] is cumulative sum up to level i
            each item has shape (B,N,D_code)
    
          active_mask_list:
            list of length L
            active_mask_list[i] is (B,N) bool
            True where at least one code up to that level is non-blank
        """
        if codes.dim() == 2:
            codes = codes.unsqueeze(-1)
    
        B, N, L = codes.shape
        D_code = self.vq.code_dim
    
        if L != self.vq.num_quantizers:
            raise ValueError(
                f"Expected codes last dim = num_quantizers = {self.vq.num_quantizers}, got {L}"
            )
    
        blank_token = self.vq.blank_token.to(device=codes.device)
        running_sum = torch.zeros((B, N, D_code), device=codes.device, dtype=blank_token.dtype)
    
        z_q_list = []
        active_mask_list = []
    
        seen_active = torch.zeros((B, N), device=codes.device, dtype=torch.bool)
        
        L_active = int(self.vq.active_quantizers)

        if not (1 <= L_active <= self.vq.num_quantizers):
            raise ValueError(
                f"Invalid active_quantizers={L_active}; expected 1..{self.vq.num_quantizers}"
            )
        
        if L_active > L:
            raise ValueError(
                f"codes has only L={L} levels, but active_quantizers={L_active}"
            )
    
        for lvl in range(L_active):
            codes_l = codes[..., lvl]                          # (B,N)
            active_l = (codes_l != self.vq.blank_code)        # (B,N)
            seen_active = seen_active | active_l
    
            z_level = torch.zeros((B, N, D_code), device=codes.device, dtype=running_sum.dtype)
            if active_l.any():
                path_l = codes[..., :lvl + 1]                    # (B,N,lvl+1)
                z_level[active_l] = self.vq.get_codebook_entry(
                    lvl=lvl,
                    codes_prefix=path_l[active_l],
                )
    
            scale = float(self.vq.level_scales[lvl])
            running_sum = running_sum + scale * z_level
    
            z_q = blank_token.view(1, 1, D_code).expand(B, N, D_code).clone()
            if seen_active.any():
                z_q[seen_active] = running_sum[seen_active]
    
            z_q_list.append(z_q)
            active_mask_list.append(seen_active.clone())
    
        return z_q_list, active_mask_list
    
    def _codes_to_quantized_final(
        self,
        codes: torch.Tensor,   # (B,N,L) or (B,N)
    ):
        """
        Build only the final cumulative quantized latent.
    
        Returns:
          z_q_final:          (B,N,D_code)
          active_mask_final:  (B,N) bool
        """
        if codes.dim() == 2:
            codes = codes.unsqueeze(-1)
    
        B, N, L = codes.shape
        D_code = self.vq.code_dim
    
        if L != self.vq.num_quantizers:
            raise ValueError(
                f"Expected codes last dim = num_quantizers = {self.vq.num_quantizers}, got {L}"
            )
    
        L_active = int(self.vq.active_quantizers)
        
        if not (1 <= L_active <= self.vq.num_quantizers):
            raise ValueError(
                f"Invalid active_quantizers={L_active}; expected 1..{self.vq.num_quantizers}"
            )
        
        if L_active > L:
            raise ValueError(
                f"codes has only L={L} levels, but active_quantizers={L_active}"
            )
    
        blank_token = self.vq.blank_token.to(device=codes.device)
        running_sum = torch.zeros(
            (B, N, D_code),
            device=codes.device,
            dtype=blank_token.dtype,
        )
    
        seen_active = torch.zeros((B, N), device=codes.device, dtype=torch.bool)
    
        for lvl in range(L_active):
            codes_l = codes[..., lvl]
            active_l = codes_l != self.vq.blank_code
            seen_active = seen_active | active_l
    
            if active_l.any():
                path_l = codes[..., :lvl + 1]
    
                z_level_active = self.vq.get_codebook_entry(
                    lvl=lvl,
                    codes_prefix=path_l[active_l],
                )
    
                scale = float(self.vq.level_scales[lvl])
                running_sum[active_l] = running_sum[active_l] + scale * z_level_active
    
        z_q_final = blank_token.view(1, 1, D_code).expand(B, N, D_code).clone()
    
        if seen_active.any():
            z_q_final[seen_active] = running_sum[seen_active]
    
        return z_q_final, seen_active
    
    
    def forward(
        self,
        x: torch.Tensor,                                   # (B,1,T,H,W)
        global_ctx: Optional[torch.Tensor] = None,          # (B,G)
        local_ctx:  Optional[torch.Tensor] = None,          # (B,L)
        cfg_ctx_drop_p: Optional[float] = None,  # CFG: per-sample drop prob
        cfg_ctx_force_unc: Optional[bool] = None,    # CFG: force drop/keep (True=drop, False=keep)
        predict_mask_tok: Optional[torch.Tensor] = None,    # (B,N) or (B,N,1)
        predict_mask_spec: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,  # NEW
        roi_hw=None,
        pad_hw=None,
        return_all_refinements=False,
    ):
        B, C, T, H, W = x.shape
        
        x_orig = x
        x_stem = self.stem(x_orig)
        
        tokens, grid, _, _ = self.patch_embed(x_stem)
        
        blank_mask = self.compute_blank_mask(x_orig, grid)
        active_mask = ~blank_mask
        
        if tuple(grid) != tuple(self.token_grid):
            raise ValueError(
                f"Input produced token grid {tuple(grid)}, but model was initialized for "
                f"{tuple(self.token_grid)}. Check dataset padding/cropping vs model img_size."
            )
        
        t_tok, h_tok, w_tok = grid
        
        pos_enc, pos_dec = self._get_pos_embed(grid, tokens.device, tokens.dtype)
            
        
        # ---------------------------------------------------------
        # Sparse encoder output -> code projection -> active-only VQ
        # ---------------------------------------------------------
        
        # active_mask: (B, N) bool, True = active
        # tokens:      (B, N, D_enc)
        # pos_enc:     (1, N, D_enc) or (B, N, D_enc)
        
        x_enc_full, _, _, _ = self.sparse_encoder(
            tokens=tokens,
            active_mask=active_mask,
            pos_embed=pos_enc,
            fill_value=0.0,
        )                                                               # (B, N, D_enc)
        
        
        z_e_full = self.to_code(x_enc_full)                             # (B, N, D_code)
        # z_e_full = self.code_norm(z_e_full)

        
        B, N, D_code = z_e_full.shape
        active_flat = active_mask.reshape(B * N)                        # (B*N,)
        z_e_flat = z_e_full.reshape(B * N, D_code)                      # (B*N, D_code)
        z_e_active = z_e_flat[active_flat]                              # (M, D_code)
        
        
        
        # --- Hierarchical version ----
        z_q_flat, vq_loss, codes_flat, vq_aux = self.vq.quantize_active_only(
            z_e_active=z_e_active,
            active_flat=active_flat,
            num_total_tokens=B * N,
            return_logits=False,
            return_aux=True,
        )
        
        z_q_ste = z_q_flat.view(B, N, D_code)
        codes = codes_flat.view(B, N, self.vq.num_quantizers)
        
        # ----- hard-code cumulative hierarchy refinements for diagnostics -----
        refinements = []

        if return_all_refinements:
            
            z_q_list, active_mask_list = self._codes_to_quantized_cumulative(codes)
            
            L_active = int(self.vq.active_quantizers)
            z_q_list = z_q_list[:L_active]
            active_mask_list = active_mask_list[:L_active]
            
            for lvl_idx, (z_q_i, active_mask_i) in enumerate(
                zip(z_q_list[:-1], active_mask_list[:-1]),
                start=1,
            ):
                dec_i = self._decode_quantized_latent(
                    z_q=z_q_i,
                    active_mask=active_mask_i,
                    grid=grid,
                    global_ctx=global_ctx,
                    local_ctx=local_ctx,
                    cfg_ctx_drop_p=cfg_ctx_drop_p,
                    cfg_ctx_force_unc=cfg_ctx_force_unc,
                    roi_hw=roi_hw,
                    pad_hw=pad_hw,
                )
                dec_i["level"] = lvl_idx
                refinements.append(dec_i)
        
        # ----- final training decode uses STE path -----
        final_dec = self._decode_quantized_latent(
            z_q=z_q_ste,
            active_mask=active_mask,
            grid=grid,
            global_ctx=global_ctx,
            local_ctx=local_ctx,
            cfg_ctx_drop_p=cfg_ctx_drop_p,
            cfg_ctx_force_unc=cfg_ctx_force_unc,
            roi_hw=roi_hw,
            pad_hw=pad_hw,
        )
        L_active = int(self.vq.active_quantizers)
        final_dec["level"] = L_active
        refinements.append(final_dec)
        # Replace final hard-code refinement with STE refinement.
        # This keeps earlier coarse refinements diagnostic/supervised,
        # but makes the final reconstruction loss send gradients through z_q_ste -> encoder.
        # if len(refinements) > 0:
        #     refinements[-1] = final_dec
        
        z_q = final_dec["z_q"]
        active_mask = final_dec["active_mask"]
        pred_patches_raw = final_dec["pred_patches_raw"]
        logits_vol_raw = final_dec["logits_vol_raw"]
        pred_patches = final_dec["pred_patches"]
        logits_vol = final_dec["logits_vol"]
        spatial_diag = final_dec["spatial_diag"]
        
        z_dec_no_pos = final_dec["z_dec_no_pos"]
        z_decoder_input = final_dec["z_decoder_input"]
        
        z_dec_base_no_pos = final_dec["z_dec_base_no_pos"]
        signed_type_offset = final_dec["signed_type_offset"]
        activity_type_offset = final_dec["activity_type_offset"]
        
        # ----- Token-level supervision mask -----
        N = t_tok * h_tok * w_tok
    
        if predict_mask_tok is not None:
            pm = predict_mask_tok
            if pm.dim() == 2:
                pm = pm.unsqueeze(-1)
            if pm.dim() != 3 or pm.size(0) != B or pm.size(1) != N or pm.size(2) != 1:
                raise ValueError(f"predict_mask_tok must be (B,{N},1) or (B,{N}), got {tuple(predict_mask_tok.shape)}")
            predict_mask = pm.to(device=pred_patches.device, dtype=torch.float32)
    
        elif predict_mask_spec is not None:
            # Build from metadata (list[dict]) -> (B,N,1)
            predict_mask = self.predict_mask_from_spec(
                predict_mask_spec, grid, device=pred_patches.device, dtype=torch.float32
            )
            # safety check
            if predict_mask.shape != (B, N, 1):
                raise ValueError(f"predict_mask_from_spec returned {tuple(predict_mask.shape)}; expected {(B,N,1)}")
    
        else:
            predict_mask = torch.ones((B, N, 1), device=pred_patches.device, dtype=torch.float32)

        out = {
            "pred_patches_raw": pred_patches_raw,
            "logits_vol_raw": logits_vol_raw,
            "pred_patches": pred_patches,
            "logits_vol": logits_vol,
            "vq_loss": vq_loss,
            "codes": codes,
            "vq_aux": vq_aux,
            "blank_mask": blank_mask,
            "grid": grid,
            "predict_mask": predict_mask,
        
            "z_e_full": z_e_full,
            "z_e_active": z_e_active,
            "x_enc_full": x_enc_full,
            "active_mask": active_mask,
            
            "z_dec_no_pos": z_dec_no_pos,
            "z_decoder_input": z_decoder_input,
            
            "z_dec_base_no_pos": z_dec_base_no_pos,
            "signed_type_offset": signed_type_offset,
            "activity_type_offset": activity_type_offset,
        
            "refinements": refinements,
        }


        if spatial_diag is not None:
            # spatial prior for loss / diagnostics only; not added to decoder logits
        
            # soft support maps for loss / visualization
            out["assay_spatial_pix2d_support"] = spatial_diag["hw_support"]
            out["assay_spatial_full_pix2d_support"] = spatial_diag["full_hw_support"]  

            # optional raw logits for diagnostics
            out["assay_spatial_pix2d_logits"] = spatial_diag["hw_logits"]
            out["assay_spatial_full_pix2d_logits"] = spatial_diag["full_hw_logits"]
            
        return out