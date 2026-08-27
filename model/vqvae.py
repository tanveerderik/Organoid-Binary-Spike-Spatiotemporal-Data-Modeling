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
from .base import (
    SparseTokenTransformerEncoder,
    DecoderBlock,
    HierarchicalVectorQuantizerEMA,
)
from .spatial_map import SpatialMapPrior
from ..utils.embed import get_3d_sincos_pos_embed

from ..utils.constants import (
    DEFAULT_GAP_BINS,
    normalize_gap_bins,
)
    
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
        local_ctx_in_dim: int = 9,
        local_emb_dim: int = 16,
        ctx_hidden_mult: float = 2.0,

        drop=0.0, attn_drop=0.0,
        
        # ---- gct-conditioned spatial map bias ----
        use_spatial_map_prior: bool = True,
        gap_bins = None,

        # ---- decoder attention masking ----
        # ---- ABLATION ONLY: dense patch embed, no blank routing ----
        # Default False reproduces the shipped model bit-for-bit. When True,
        # every token is declared active, so blank patches are projected,
        # encoded and QUANTIZED like any other -- i.e. the EMA codebook sees
        # the ~92% blank mass it was designed to be shielded from, and the
        # learned `blank_token` is never used. This is the arm that tests
        # whether the sparse encoder is load-bearing.
        dense_ablation: bool = False,

        enc_attn_mask_kind: str = "temporal_band_bi",  # "none" | "temporal_causal" | "temporal_band" | "temporal_band_bi"
        enc_attn_window: int = 5,                      # window length in patch-time tokens
        dec_attn_mask_kind: str = "temporal_band_bi",  # "none" | "temporal_causal" | "temporal_band" | "temporal_band_bi"
        dec_attn_window: int = 5,                      # window length in patch-time tokens
        
        
    ):
        super().__init__()
        self.dense_ablation = bool(dense_ablation)
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
        # self.to_code = nn.Sequential(
        #     nn.Linear(encoder_embed_dim, code_dim, bias=True),
        #     nn.LayerNorm(code_dim),
        # )
        self.to_code = nn.Sequential(
            # Stabilize the unnormalized residual stream produced by the
            # pre-norm transformer, without normalizing the final code vector.
            nn.LayerNorm(encoder_embed_dim),
        
            # No output LayerNorm: code-space vectors may now use both direction
            # and magnitude. Bias is unnecessary because the EMA codebook can
            # represent a nonzero center itself.
            nn.Linear(
                encoder_embed_dim,
                code_dim,
                bias=False,
            ),
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
            # Was hardcoded to 2, which silently capped a 3-level model at two
            # levels: base.py:1154 takes L = min(active_quantizers,
            # num_quantizers), so level 3 was allocated (1024 entries) but never
            # assigned, never supervised, and reported perplexity 0.
            active_quantizers=self.num_quantizers,
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
            DecoderBlock(
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

        # ---- Context path ----
        #
        # The decoder is dense: context never enters it directly.  Adherence
        # comes from the output-space loss (utils.losses.ctx_loss_soft), which
        # reads context features back out of the reconstructed logits volume.
        #
        # global_embedder stays here because it is half of the Stage-1 gct
        # module pair -- it feeds spatial_map_prior via _global_emb_only and is
        # trained with it.  The lct mapper is a standalone Stage-3 artifact and
        # is no longer part of this model.
        # alpha_init must be nonzero: CtxEmbed gates its MLP branch with
        # alpha_max * tanh(alpha_raw), and at alpha_raw == 0 the gradient of the
        # entire MLP is exactly zero, so the embedder collapses to its linear
        # projection and never recovers.
        self.global_embedder = CtxEmbed(self.global_ctx_in_dim, self.global_emb_dim, mlp_ratio=2.0, drop=0.0, alpha_init=0.1)

        # global context based spatial support prior
        self.use_spatial_map_prior = bool(use_spatial_map_prior)
        self.use_learned_spatial_prior_diagnostics = False
        
        self.gap_bins = normalize_gap_bins(
            gap_bins
            if gap_bins is not None
            else DEFAULT_GAP_BINS
        )

        
        if self.use_spatial_map_prior:
            self.spatial_map_prior = SpatialMapPrior(
                global_emb_dim=self.global_emb_dim,
                patch_dim=self.patch_dim,
                full_spatial_size=self.full_spatial_size,
                patch_size_hw=(pH, pW),
                drop=0.1,
                basis_k=32,
                num_adj_bins=len(self.gap_bins),
            )
        else:
            self.spatial_map_prior = None
            

        self.register_buffer(
            "best_thr_exact",
            torch.tensor(0.5, dtype=torch.float32),
        )
        
        self.register_buffer(
            "best_thr_tol",
            torch.tensor(0.5, dtype=torch.float32),
        )
        
        # Threshold currently used by threshold-aware training losses.
        self.register_buffer(
            "training_prob_threshold",
            torch.tensor(0.5, dtype=torch.float32),
        )
        
    
        
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
    
    

    def _apply_output_biases(
        self,
        pred_patches: torch.Tensor,
        grid: Tuple[int, int, int],
        global_ctx: Optional[torch.Tensor] = None,
        roi_hw=None,
        pad_hw=None,
    ):
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
    
    @torch.no_grad()
    @torch.no_grad()
    def _set_training_prob_threshold(self, value):
        value = float(value)
    
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"training_prob_threshold must be in [0, 1], got {value}"
            )
    
        self.training_prob_threshold.fill_(value)
        
    @torch.no_grad()
    def _set_best_thresholds(
        self,
        *,
        exact=None,
        tolerant=None,
    ):
        if exact is not None:
            self.best_thr_exact.fill_(float(exact))
    
        if tolerant is not None:
            self.best_thr_tol.fill_(float(tolerant))

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
        self._ensure_dec_masks(grid, device=z_d.device)
        for blk in self.dec_blocks:
            z_d = blk(z_d)
    
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
            True  -> NOT IMPLEMENTED here; raises. Use
                     forward(..., return_all_refinements=True) instead.
        """
        
        L_active = int(self.vq.active_quantizers)
        
        if return_all_refinements:
            # This branch never worked. Its per-level loop called `_apply_hole`,
            # a closure defined inside `forward` and therefore out of scope in
            # this method, so the branch raised NameError on every call --
            # which is why nothing ever noticed it was here.
            #
            # `forward(..., return_all_refinements=True)` takes the same option
            # and IS exercised (training/train_vqvae.py, training/eval_vqvae.py),
            # so it is the supported route. See the note in
            # ablations/ladder_decode_depth.py, which already routes around this.
            raise NotImplementedError(
                "decode_from_codes(..., return_all_refinements=True) is not "
                "implemented; use forward(..., return_all_refinements=True), "
                "which offers the same option and has test coverage behind it."
            )

        z_q_final, active_mask_final = self._codes_to_quantized_final(codes)
    
        final_dec = self._decode_quantized_latent(
            z_q=z_q_final,
            active_mask=active_mask_final,
            grid=grid,
            global_ctx=global_ctx,
            local_ctx=local_ctx,
            roi_hw=roi_hw,
            pad_hw=pad_hw,
        )
        final_dec["level"] = L_active
        final_dec["grid"] = grid
        
        return final_dec
        
    @torch.no_grad()
    def build_latent_hidden_mask(
        self,
        predict_mask_spec,
        grid: Tuple[int, int, int],
        *,
        device: torch.device,
        recon_drop_p: float = 0.5,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Token-level mask of latents to HIDE from the decoder. True = hidden.

        Stage 2C originally trained the context branch under full, unmasked
        autoencoding.  Under that objective the context is provably redundant:
        local_ctx is nine deterministic summary statistics of the very same x
        that the encoder already compressed into every one of the N latent
        tokens, and the decoder renders each patch from its own code.  There
        is nothing left for a clip-level descriptor to contribute, so the
        cross-attention branch has no gradient signal to grow into.

        Hiding a subset of the latents restores the information asymmetry the
        context is supposed to fill, and matches how the decoder is actually
        used in Stage 3, where the prior supplies codes it may get wrong.

        The dataloader's mask_spec already describes causal / noncausal /
        spatial holes; those are reused directly.  Plain "recon" samples get a
        random Bernoulli hole so every batch carries some asymmetry.
        """
        pmask = self.predict_mask_from_spec(
            predict_mask_spec, grid, device=device, dtype=torch.float32
        ).squeeze(-1)                                    # (B, N), 1 = supervised

        hidden = pmask > 0.5

        # "recon" spec marks every token as supervised; substitute a random hole.
        all_supervised = hidden.all(dim=1)
        if bool(all_supervised.any()) and recon_drop_p > 0.0:
            rand = torch.rand(
                hidden.shape, device=device, generator=generator
            )
            random_hole = rand < float(recon_drop_p)
            hidden = torch.where(
                all_supervised.unsqueeze(1), random_hole, hidden
            )

        return hidden

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
        checkpoint_state = ckpt["model"]
        model_state = self.state_dict()

        # Stage-1 and earlier Stage-2 checkpoints may contain the previous
        # K2+1 alpha adapter (with a zero anchor), no context gate, or other
        # obsolete continuous-projector tensors.  Those parameters never define
        # the frozen Stage-1 encoder/codebook geometry, so incompatible entries
        # are safely reinitialized while all core VQVAE weights remain strict.
        # The decoder context branch (slot projections, slot embeddings, and
        # the per-channel cross-attention gate) is retrained from scratch in
        # Stage 2C and was reshaped when the single-token context bank was
        # replaced by a multi-slot bank.  Older checkpoints therefore carry
        # incompatible or absent entries for it; those are reinitialized.
        removed_module_prefixes = (
            "continuous_residual_projector.",
            "local_embedder.",
            "local_to_dec_ctx.",
            "global_to_dec_ctx.",
            "ctx_slot_embed",
            "local_ctx_mean",
            "local_ctx_scale",
        )

        # Cross-attention submodules removed from DecoderBlock. Scoped to
        # dec_blocks. so the encoder's own norm2 is never caught.
        removed_dec_block_parts = (".ctx_gate", ".norm2.", ".norm_ctx.", ".cross_attn.")

        def _is_removed_legacy(key: str) -> bool:
            return (
                key.startswith(removed_module_prefixes)
                or (
                    key.startswith("dec_blocks.")
                    and any(part in key for part in removed_dec_block_parts)
                )
            )

        filtered_state = {}
        ignored_incompatible = []
        for key, value in checkpoint_state.items():
            current = model_state.get(key)
            if current is not None and tuple(current.shape) == tuple(value.shape):
                filtered_state[key] = value
                continue

            if _is_removed_legacy(key):
                ignored_incompatible.append(key)
                continue

            raise RuntimeError(
                "Checkpoint/model shape mismatch while loading "
                f"{path}: key={key}, checkpoint_shape={tuple(value.shape)}, "
                f"model_shape={None if current is None else tuple(current.shape)}"
            )

        incompatible = self.load_state_dict(filtered_state, strict=False)

        disallowed_missing = [
            key for key in incompatible.missing_keys if not _is_removed_legacy(key)
        ]

        unexpected = [
            key for key in incompatible.unexpected_keys if not _is_removed_legacy(key)
        ]

        if disallowed_missing or unexpected:
            raise RuntimeError(
                "Checkpoint/model mismatch while loading "
                f"{path}: missing={disallowed_missing}, unexpected={unexpected}"
            )

        initialized = list(incompatible.missing_keys) + ignored_incompatible
        if initialized:
            print(
                "Initialized corrected Stage-2 projection/context parameters "
                "that are absent or incompatible in the checkpoint: "
                + ", ".join(sorted(set(initialized)))
            )

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
        predict_mask_tok: Optional[torch.Tensor] = None,    # (B,N) or (B,N,1)
        predict_mask_spec: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,  # NEW
        roi_hw=None,
        pad_hw=None,
        return_all_refinements=False,
        mask_latents: bool = False,
        latent_recon_drop_p: float = 0.5,
        latent_hidden_mask: Optional[torch.Tensor] = None,   # (B,N) bool, True = hide
    ):
        B, C, T, H, W = x.shape
        
        x_orig = x
        x_stem = self.stem(x_orig)
        
        tokens, grid, _, _ = self.patch_embed(x_stem)
        
        blank_mask = self.compute_blank_mask(x_orig, grid)
        if self.dense_ablation:
            # Declare every token active. Downstream this is the ONLY change:
            # the sparse encoder, the code projection and the residual VQ all
            # key off `active_mask`, so forcing it True routes blank patches
            # through the codebook instead of to `self.blank_token`.
            blank_mask = torch.zeros_like(blank_mask)
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
        
        z_q_hard_ste = z_q_flat.view(B, N, D_code)
        codes = codes_flat.view(B, N, self.vq.num_quantizers)

        z_q_decode = z_q_hard_ste

        # ----- optional latent masking (Stage 2C context conditioning) -----
        # Hidden latents are replaced by the VQ blank token and marked
        # inactive, exactly as an unpredicted token looks at generation time.
        # The encoder, codebook and loss targets are untouched.
        if latent_hidden_mask is None and mask_latents and predict_mask_spec is not None:
            latent_hidden_mask = self.build_latent_hidden_mask(
                predict_mask_spec,
                grid,
                device=z_q_decode.device,
                recon_drop_p=float(latent_recon_drop_p),
            )

        decoder_active_mask = active_mask

        if latent_hidden_mask is not None:
            if latent_hidden_mask.shape != (B, N):
                raise ValueError(
                    f"latent_hidden_mask must be (B,{N}), got {tuple(latent_hidden_mask.shape)}"
                )
            hidden = latent_hidden_mask.to(device=z_q_decode.device, dtype=torch.bool)
            blank = self.vq.blank_token.to(
                device=z_q_decode.device, dtype=z_q_decode.dtype
            )
            z_q_decode = torch.where(
                hidden.unsqueeze(-1),
                blank.view(1, 1, -1).expand_as(z_q_decode),
                z_q_decode,
            )
            decoder_active_mask = active_mask & (~hidden)

        def _apply_hole(z, mask_active):
            """Apply the same hole to a cumulative-refinement latent."""
            if latent_hidden_mask is None:
                return z, mask_active
            z_holed = torch.where(
                hidden.unsqueeze(-1),
                blank.to(dtype=z.dtype).view(1, 1, -1).expand_as(z),
                z,
            )
            return z_holed, mask_active & (~hidden)

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
                    roi_hw=roi_hw,
                    pad_hw=pad_hw,
                )
                dec_i["level"] = lvl_idx
                refinements.append(dec_i)
        
        # ----- final training decode uses hard STE or continuous residual path -----
        final_dec = self._decode_quantized_latent(
            z_q=z_q_decode,
            active_mask=decoder_active_mask,
            grid=grid,
            global_ctx=global_ctx,
            local_ctx=local_ctx,
            roi_hw=roi_hw,
            pad_hw=pad_hw,
        )
        L_active = int(self.vq.active_quantizers)
        final_dec["level"] = L_active
        refinements.append(final_dec)
        # Earlier cumulative hard-code refinements remain diagnostic/supervised.
        # The final refinement uses either the original STE latent or the bounded
        # continuous residual latent selected above.
        # if len(refinements) > 0:
        #     refinements[-1] = final_dec
        
        z_q = final_dec["z_q"]
        # Keep the encoder-side active_mask for downstream losses; the decoder
        # may have seen a masked variant of it.
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
            "latent_hidden_mask": latent_hidden_mask,
            "decoder_active_mask": decoder_active_mask,
            "z_e_full": z_e_full,
            "z_e_active": z_e_active,
            "x_enc_full": x_enc_full,
            "active_mask": active_mask,
            
            "z_dec_no_pos": z_dec_no_pos,
            "z_decoder_input": z_decoder_input,
            
            "z_dec_base_no_pos": z_dec_base_no_pos,
            "signed_type_offset": signed_type_offset,
            "activity_type_offset": activity_type_offset,

            "z_q_hard_ste": z_q_hard_ste,
            "z_q_decode": z_q_decode,
        
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
