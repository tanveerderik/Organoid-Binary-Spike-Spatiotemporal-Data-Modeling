#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:13:33 2026

@author: derik
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class GlobalContextSpatialBank:
    """
    Stores a soft union / EMA full support map per stable global context vector.
    Key is a rounded tuple of gct values.
    """

    def __init__(self, momentum: float = 0.95, round_decimals: int = 6, union_mode: str = "ema"):
        self.momentum = float(momentum)
        self.round_decimals = int(round_decimals)
        self.union_mode = str(union_mode)
        self._bank = {}

    def _key(self, gct_row: torch.Tensor):
        vals = gct_row.detach().cpu().float().tolist()
        return tuple(round(float(v), self.round_decimals) for v in vals)

    def update(self, gct: torch.Tensor, target_full_bhw: torch.Tensor):
        """
        gct: (B,D)
        target_full_bhw: (B,H,W) in [0,1]
        """
        B = gct.shape[0]
        for b in range(B):
            k = self._key(gct[b])
            t = target_full_bhw[b].detach().cpu()

            if k not in self._bank:
                self._bank[k] = t.clone()
                continue

            old = self._bank[k]

            if self.union_mode == "max":
                new = torch.maximum(self.momentum * old, t)
            else:  # "ema"
                new = self.momentum * old + (1.0 - self.momentum) * t

            self._bank[k] = new.clamp(0.0, 1.0)

    def get(self, gct: torch.Tensor, device=None, dtype=None):
        """
        returns: (B,H,W)
        """
        outs = []
        for b in range(gct.shape[0]):
            k = self._key(gct[b])
            if k not in self._bank:
                raise KeyError(f"Missing target memory for gct key={k}")
            x = self._bank[k]
            if device is not None:
                x = x.to(device=device)
            if dtype is not None:
                x = x.to(dtype=dtype)
            outs.append(x)
        return torch.stack(outs, dim=0)

    def has(self, gct_row: torch.Tensor) -> bool:
        return self._key(gct_row) in self._bank

    def state_dict(self):
        return {
            "momentum": self.momentum,
            "round_decimals": self.round_decimals,
            "union_mode": self.union_mode,
            "bank": self._bank,
        }

    def load_state_dict(self, state):
        self.momentum = float(state["momentum"])
        self.round_decimals = int(state["round_decimals"])
        self.union_mode = str(state["union_mode"])
        self._bank = state["bank"]
        
        

class GlobalContextAdjacencyBank:
    def __init__(self, max_gap=3, gap_bins=None, alpha=1.0, beta=20.0, round_decimals=6):
        if gap_bins is None:
            gap_bins = [(g, g) for g in range(1, int(max_gap) + 1)]
        self.gap_bins = [(int(a), int(b)) for a, b in gap_bins]
        self.max_gap = max(b for _, b in self.gap_bins)
        self.num_bins = len(self.gap_bins)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.round_decimals = int(round_decimals)
        self._num = {}
        self._den = {}

    def _key(self, gct_row):
        vals = gct_row.detach().cpu().float().tolist()
        return tuple(round(float(v), self.round_decimals) for v in vals)

    def update_from_x(self, gct, x):
        x_bin = (x > 0).float()
        B, _, T, H, W = x_bin.shape

        for b in range(B):
            k = self._key(gct[b])

            if k not in self._num:
                self._num[k] = torch.zeros(self.num_bins)
                self._den[k] = torch.zeros(self.num_bins)

            xb = x_bin[b:b+1]

            for bi, (lo, hi) in enumerate(self.gap_bins):
                for gap in range(lo, hi + 1):
                    if T <= gap:
                        continue
            
                    joint = (xb[:, :, :-gap] * xb[:, :, gap:]).sum().detach().cpu()
                    base = xb[:, :, :-gap].sum().detach().cpu()
            
                    self._num[k][bi] += joint
                    self._den[k][bi] += base

    def get(self, gct, device=None, dtype=None):
        outs = []
        for b in range(gct.shape[0]):
            k = self._key(gct[b])
            if k not in self._num:
                raise KeyError(f"Missing adjacency memory for gct key={k}")

            num = self._num[k]
            den = self._den[k]

            rate = (num + self.alpha) / (den + self.alpha + self.beta)

            if device is not None:
                rate = rate.to(device)
            if dtype is not None:
                rate = rate.to(dtype)

            outs.append(rate)

        return torch.stack(outs, dim=0)
    
    def get_den(self, gct, device=None, dtype=None):
        outs = []
        for b in range(gct.shape[0]):
            k = self._key(gct[b])
            if k not in self._den:
                raise KeyError(f"Missing adjacency denominator for gct key={k}")
    
            den = self._den[k]
            if device is not None:
                den = den.to(device)
            if dtype is not None:
                den = den.to(dtype)
            outs.append(den)
    
        return torch.stack(outs, dim=0)
    
    def get_confidence(self, gct, device=None, dtype=None, den_scale: float = 100.0):
        den = self.get_den(gct, device=device, dtype=dtype)
        return (den / (den + float(den_scale))).clamp(0.0, 1.0)

    def state_dict(self):
        return {
            "max_gap": self.max_gap,
            "gap_bins": self.gap_bins,
            "num_bins": self.num_bins,
            "alpha": self.alpha,
            "beta": self.beta,
            "round_decimals": self.round_decimals,
            "num": self._num,
            "den": self._den,
        }

    def load_state_dict(self, state):
        self.gap_bins = state.get("gap_bins", [(g, g) for g in range(1, int(state["max_gap"]) + 1)])
        self.gap_bins = [(int(a), int(b)) for a, b in self.gap_bins]
        self.max_gap = max(b for _, b in self.gap_bins)
        self.num_bins = len(self.gap_bins)
        self.alpha = float(state["alpha"])
        self.beta = float(state["beta"])
        self.round_decimals = int(state["round_decimals"])
        self._num = state["num"]
        self._den = state["den"]
        
    def debug_print(self, max_items=5):
        print("\n=== Adjacency Memory Bank (sample) ===")
    
        keys = list(self._num.keys())
    
        for i, k in enumerate(keys[:max_items]):
            num = self._num[k].detach().cpu()
            den = self._den[k].detach().cpu()
            rate = (num + self.alpha) / (den + self.alpha + self.beta)
    
            print(f"\nKey {i}: {k}")
            print(f"  den: {float(den.mean()):.2f}")
    
            for bi, (lo, hi) in enumerate(self.gap_bins):
                label = f"{lo}" if lo == hi else f"{lo}-{hi}"
                print(
                    f"  gap {label}: "
                    f"num={float(num[bi]):.2f} "
                    f"rate={float(rate[bi]):.6f}"
                )
    
        print("====================================\n")
        
        
# ----- Spatial map prior class ----
class SpatialMapPrior(nn.Module):
    """
    Global context-conditioned pixelwise spatial support prior.

    g_emb is expected to be produced upstream by CtxEmbed.

    Learns:
        token_logits:     (B, Hf_tok, Wf_tok) coarse support
        full_hw_logits:   (B, full_H, full_W) pixelwise support logits

    Samples:
        hw_logits:        (B, h_tok*ph, w_tok*pw) current crop/padded pixel map
    """
    def __init__(
        self,
        global_emb_dim: int,
        patch_dim: int,
        full_spatial_size: tuple[int, int],   # full assay size in pixels
        patch_size_hw: tuple[int, int],
        drop: float = 0.1,
        basis_k: int = 32,
        max_gap: int = 3,
        num_adj_bins: int | None = None,
    ):
        super().__init__()
        self.global_emb_dim = int(global_emb_dim)
        self.patch_dim = int(patch_dim)

        self.full_H = int(full_spatial_size[0])
        self.full_W = int(full_spatial_size[1])

        self.ph = int(patch_size_hw[0])
        self.pw = int(patch_size_hw[1])

        # Use ceil so arbitrary full sizes are still representable
        self.full_h_tok = (self.full_H + self.ph - 1) // self.ph
        self.full_w_tok = (self.full_W + self.pw - 1) // self.pw

        # g_emb is already produced by CtxEmbed upstream.
        # Keep this module mostly linear/interpretable.
        self.dropout = nn.Dropout(drop)
        
        self.token_head = nn.Linear(
            self.global_emb_dim,
            self.full_h_tok * self.full_w_tok,
            bias=False,
        )
        
        self.basis_k = int(basis_k)
        self.coeff_head = nn.Linear(
            self.global_emb_dim,
            self.basis_k,
            bias=False,
        )
        
        self.pixel_basis = nn.Parameter(
            torch.randn(self.basis_k, self.full_H, self.full_W) * 0.01
        )
        
        self.residual_scale_raw = nn.Parameter(torch.tensor(-2.0))
        
        
        self.adj_head = nn.Sequential(
            nn.Linear(self.global_emb_dim, self.global_emb_dim),
            nn.ReLU(),
            nn.Linear(self.global_emb_dim, int(num_adj_bins or max_gap))
        )

        self.reset_parameters()

    
    def reset_parameters(self):
        nn.init.normal_(self.token_head.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.coeff_head.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.pixel_basis, mean=0.0, std=0.01)
        
        for m in self.adj_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            

    def _sample_from_full_map(
        self,
        full_hw_map: torch.Tensor,   # now (B, full_H, full_W), pixelwise
        grid: tuple[int, int, int],
        roi_hw,
        pad_hw,
    ) -> torch.Tensor:
        """
        Sample crop-aware runtime pixel map from the learned full pixel map.
    
        Returns:
            crop_hw_map : (B, h_tok*ph, w_tok*pw)
        """
        B, Hf, Wf = full_hw_map.shape
        _, h_tok, w_tok = map(int, grid)
    
        out_H = h_tok * self.ph
        out_W = w_tok * self.pw
    
        if roi_hw is None:
            if (out_H, out_W) == (Hf, Wf):
                return full_hw_map
        
            raise ValueError(
                f"roi_hw is None but runtime spatial size {(out_H, out_W)} "
                f"!= full spatial size {(Hf, Wf)}. Pass roi_hw/pad_hw so the "
                f"spatial prior is padded/cropped, not interpolated."
            )
    
        if len(roi_hw) != B:
            raise ValueError(f"roi_hw length {len(roi_hw)} != batch size {B}")
    
        if pad_hw is None:
            pad_hw = [(0, 0, 0, 0)] * B
    
        if len(pad_hw) != B:
            raise ValueError(f"pad_hw length {len(pad_hw)} != batch size {B}")
    
        crops = []
    
        for b in range(B):
            y0, x0, y1, x1 = map(int, roi_hw[b])
            ptop, pbot, pleft, pright = map(int, pad_hw[b])
    
            y0_clip = max(0, y0)
            x0_clip = max(0, x0)
            y1_clip = min(self.full_H, y1)
            x1_clip = min(self.full_W, x1)
    
            valid = full_hw_map[b:b+1, None, y0_clip:y1_clip, x0_clip:x1_clip]
    
            extra_top = y0_clip - y0
            extra_left = x0_clip - x0
            extra_bottom = y1 - y1_clip
            extra_right = x1 - x1_clip
    
            valid = F.pad(
                valid,
                (
                    pleft + extra_left,
                    pright + extra_right,
                    ptop + extra_top,
                    pbot + extra_bottom,
                ),
                value=0.0,
            )
    
            if valid.shape[-2:] != (out_H, out_W):
                valid = F.interpolate(
                    valid,
                    size=(out_H, out_W),
                    mode="bilinear",
                    align_corners=False,
                )
    
            crops.append(valid[:, 0])
    
        return torch.cat(crops, dim=0)

    def forward(
        self,
        g_emb: torch.Tensor,
        grid: tuple[int, int, int],
        roi_hw=None,
        pad_hw=None,
        gain: float = 2.0,
    ):
        """
        Returns
        -------
        hw_support : (B, h_tok*ph, w_tok*pw)
            Current crop runtime soft support map in [0,1].
        full_hw_support : (B, full_H, full_W)
            Full assay soft support map in [0,1].
        hw_logits : (B, h_tok*ph, w_tok*pw)
            Current crop runtime raw map logits.
        full_hw_logits : (B, full_H, full_W)
            Full assay raw map logits.
        """
        if g_emb is None:
            raise ValueError("SpatialMapPrior requires g_emb, got None.")
    
        if g_emb.dim() != 2 or g_emb.size(1) != self.global_emb_dim:
            raise ValueError(
                f"g_emb must be (B,{self.global_emb_dim}), got {tuple(g_emb.shape)}"
            )
    
        B = g_emb.size(0)
        t_tok, h_tok, w_tok = map(int, grid)
    
        # ---- raw full-map logits from global embedding ----
        # g_emb is already CtxEmbed output upstream.
        g = self.dropout(g_emb)
        
        # Coarse token map.
        token_logits = self.token_head(g).view(
            B,
            self.full_h_tok,
            self.full_w_tok,
        )
        
        token_up = F.interpolate(
            token_logits[:, None],
            size=(self.full_H, self.full_W),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        
        # Pixelwise basis residual.
        coeff = self.coeff_head(g)
        pixel_residual = torch.einsum("bk,khw->bhw", coeff, self.pixel_basis)
        
        residual_scale = torch.sigmoid(self.residual_scale_raw)
        full_hw_logits = token_up + residual_scale * pixel_residual
    
    
        # ---- crop-aware sampling of RAW logits ----
        hw_logits = self._sample_from_full_map(
            full_hw_map=full_hw_logits,
            grid=grid,
            roi_hw=roi_hw,
            pad_hw=pad_hw,
        )
    
        expected_H = h_tok * self.ph
        expected_W = w_tok * self.pw
        
        if hw_logits.shape[1:] != (expected_H, expected_W):
            raise ValueError(
                f"Sampled hw_logits shape {tuple(hw_logits.shape[1:])} does not match "
                f"runtime pixel size {(expected_H, expected_W)}"
            )
    
        # ---- convert logits -> support in [0,1] ----
        full_hw_support = torch.sigmoid(float(gain) * full_hw_logits)
        hw_support = torch.sigmoid(float(gain) * hw_logits)
        token_support = torch.sigmoid(float(gain) * token_logits)
        
        adj_logits = self.adj_head(g)
        adj_probs = torch.sigmoid(adj_logits)
    
        return {
            "hw_support": hw_support,               # (B,h_tok*ph,w_tok*pw), in [0,1]
            "full_hw_support": full_hw_support,     # (B,full_H,full_W), in [0,1]
            "hw_logits": hw_logits,                 # raw sampled logits
            "full_hw_logits": full_hw_logits,       # raw full pixel logits
            "token_support": token_support,         # (B,h_tok,w_tok), in [0,1]
            "token_logits": token_logits,           # raw full tokenwise logits
            "residual_scale": residual_scale.detach(),
            
            "adjacency_logits": adj_logits,
            "adjacency_probs": adj_probs,
        }
        
    
    
