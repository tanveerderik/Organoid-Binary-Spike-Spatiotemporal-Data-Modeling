#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:06:59 2026

@author: derik
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def unpack_packed_npz_array(packed: np.ndarray, true_shape) -> np.ndarray:
    """
    packed: uint8 array containing bit-packed binary volume
    true_shape: iterable giving the logical shape, e.g. (H, W, T) or (T, H, W)

    Returns:
        uint8 array with shape=true_shape and values in {0,1}
    """
    true_shape = tuple(int(x) for x in true_shape)
    n = int(np.prod(true_shape))

    packed = np.asarray(packed, dtype=np.uint8)
    bits = np.unpackbits(packed.reshape(-1))[:n]
    arr = bits.reshape(true_shape).astype(np.uint8)
    return arr


def npz_load_volume(path: str) -> np.ndarray:
    """
    Robust loader for .npz or .npy.

    Supports:
      - ordinary arrays under arr_0 / data / first key
      - packed binary NPZs with keys:
            'packed' : bit-packed uint8 array
            'shape'  : logical array shape

    Returns:
      numpy uint8 array with values in {0,1}
      shape may be (H,W,T), (T,H,W), or (H,W) depending on source.
    """
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:

            # --- packed binary format ---
            if ("packed" in z) and ("shape" in z):
                packed = z["packed"]
                true_shape = z["shape"]
                arr = unpack_packed_npz_array(packed, true_shape)

            # --- standard keys ---
            elif "arr_0" in z:
                arr = z["arr_0"]
            elif "data" in z:
                arr = z["data"]
            else:
                k0 = list(z.files)[0]
                arr = z[k0]
    else:
        arr = np.load(path, allow_pickle=False)

    arr = np.asarray(arr)

    if arr.ndim == 2:
        H, W = arr.shape
        arr = arr.reshape(H, W, 1)

    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
    else:
        arr = (arr > 0).astype(np.uint8)

    return arr


def ensure_thw(arr: np.ndarray, axis_order: str) -> np.ndarray:
    """
    Convert volume to canonical (T,H,W).

    axis_order:
      - 'HWT' : input is (H,W,T)
      - 'THW' : input is already (T,H,W)
    """
    axis_order = axis_order.upper()

    if axis_order == "HWT":
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array for HWT, got shape {arr.shape}")
        return np.transpose(arr, (2, 0, 1))

    if axis_order == "THW":
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array for THW, got shape {arr.shape}")
        return arr

    raise ValueError("axis_order must be 'HWT' or 'THW'")


def load_thw_raw_only(path: str, axis_order: str) -> np.ndarray:
    """
    Load raw file and return canonical THW (no caching, no pooling, no crop).
    """
    arr = npz_load_volume(path)
    thw = ensure_thw(arr, axis_order)
    return thw.astype(np.uint8, copy=False)