#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 13:45:37 2025

@author: derik
"""
import numpy as np
from typing import Optional, Tuple
import torch
import torch.nn.functional as F



def temporal_pool_max(thw: np.ndarray, k: int) -> np.ndarray:
    """
    Non-overlapping max-pool along time. thw: (T,H,W) uint8
    """
    if k is None or k <= 1:
        return thw
    T, H, W = thw.shape
    T_trim = (T // k) * k
    if T_trim <= 0:
        # If too short, just keep original
        return thw
    thw = thw[:T_trim]
    # reshape (T_trim//k, k, H, W) and max over axis=1
    thw = thw.reshape(T_trim // k, k, H, W).max(axis=1)
    # now shape (T_trim//k, H, W)
    return thw

def random_spatial_crop(thw: np.ndarray, crop_hw: Tuple[int, int], rng: np.random.Generator) -> tuple[np.ndarray, tuple[int,int,int,int]]:
    """
    Crop thw (T,H,W) to (T,Hc,Wc)
    """
    T, H, W = thw.shape
    Hc, Wc = crop_hw
    if (Hc is None) or (Wc is None):
        return thw
    Hc = int(Hc); Wc = int(Wc)
    if Hc <= 0 or Wc <= 0 or Hc > H or Wc > W:
        return thw
    y0 = int(rng.integers(0, H - Hc + 1))
    x0 = int(rng.integers(0, W - Wc + 1))
    y1, x1 = y0 + Hc, x0 + Wc
    return thw[:, y0:y0 + Hc, x0:x0 + Wc], (y0, x0, y1, x1)

def pick_temporal_span(T: int, span: Optional[int], rng: np.random.Generator) -> Tuple[int, int]:
    """
    Pick a start index and end index (exclusive) in [0, T) for a span of length 'span'.
    If span is None or invalid, return full range.
    """
    if span is None or span <= 0 or span > T:
        return 0, T
    t0 = int(rng.integers(0, T - span + 1))
    return t0, t0 + span
