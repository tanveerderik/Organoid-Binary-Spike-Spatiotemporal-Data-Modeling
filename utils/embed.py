#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 13:33:41 2025

@author: derik
"""
import numpy as np
import torch


# 3D sin-cos positional embedding (grid-based, no parameters)
def get_1d_sincos_pos_embed(embed_dim: int, length: int, device=None, dtype=None):
    """Return (length, embed_dim) 1D sin-cos embedding."""
    omega = torch.arange(embed_dim, device=device, dtype=dtype)
    omega = 1. / (10000 ** (omega / embed_dim))
    pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)  # (L,1)
    out = torch.einsum('ln,d->ld', pos, omega)  # (L,D)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)[:, :embed_dim]  # (L,D)

def get_3d_sincos_pos_embed(embed_dim: int, nt: int, nh: int, nw: int, device=None, dtype=None):
    """
    Split embed_dim across T/H/W (roughly equally), then concat 1D embeddings along last dim.
    Returns (1, nt*nh*nw, embed_dim).
    """
    # split dims as evenly as possible
    d_t = embed_dim // 3
    d_h = (embed_dim - d_t) // 2
    d_w = embed_dim - d_t - d_h
    pe_t = get_1d_sincos_pos_embed(d_t, nt, device, dtype)  # (nt,d_t)
    pe_h = get_1d_sincos_pos_embed(d_h, nh, device, dtype)  # (nh,d_h)
    pe_w = get_1d_sincos_pos_embed(d_w, nw, device, dtype)  # (nw,d_w)

    # combine into 3D grid
    pe_t = pe_t[:, None, None, :]  # (nt,1,1,d_t)
    pe_h = pe_h[None, :, None, :]  # (1,nh,1,d_h)
    pe_w = pe_w[None, None, :, :]  # (1,1,nw,d_w)
    pe = torch.cat([
        pe_t.expand(nt, nh, nw, -1),
        pe_h.expand(nt, nh, nw, -1),
        pe_w.expand(nt, nh, nw, -1),
    ], dim=-1)  # (nt,nh,nw,embed_dim)
    pe = pe.reshape(1, nt*nh*nw, embed_dim)
    return pe

def fourier_embed_2d(idx, n_items, dtype=np.float32):
    """
    Map id(s) in [0, n_items-1] to 2D [sin(theta), cos(theta)],
    where theta = 2π * idx / n_items.

    Args:
        idx: int or array-like of ints
        n_items: total number of categories (>0)
        dtype: output dtype (default float32)

    Returns:
        np.ndarray of shape (2,) if scalar input, else (N, 2) for vector input.
    """
    if n_items <= 0:
        raise ValueError("n_items must be > 0")
    idx_arr = np.asarray(idx, dtype=np.float32)
    theta = 2.0 * np.pi * idx_arr / float(n_items)
    s = np.sin(theta)
    c = np.cos(theta)
    out = np.stack([s, c], axis=-1).astype(dtype, copy=False)
    return out
