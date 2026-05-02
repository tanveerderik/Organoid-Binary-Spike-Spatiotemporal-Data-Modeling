from typing import Tuple
import torch
import numpy as np


# --------------------------- Utilities ---------------------------

def crop_time_to_multiple(thw: np.ndarray, pt: int) -> np.ndarray:
    # thw: (T,H,W)
    if pt is None or pt <= 1:
        return thw
    T = thw.shape[0]
    T2 = (T // pt) * pt
    if T2 <= 0:
        # not enough frames for even 1 token; leave as-is (or raise if you prefer)
        return thw
    return thw[:T2]


def pad_hw_symmetric_to_multiple(thw: np.ndarray, ph: int, pw: int) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    # returns padded_thw and (pad_top, pad_bottom, pad_left, pad_right)
    if (ph is None or ph <= 1) and (pw is None or pw <= 1):
        return thw, (0,0,0,0)

    T, H, W = thw.shape
    ph = 1 if (ph is None or ph <= 1) else ph
    pw = 1 if (pw is None or pw <= 1) else pw

    pad_h = (ph - (H % ph)) % ph
    pad_w = (pw - (W % pw)) % pw

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_h == 0 and pad_w == 0:
        return thw, (0,0,0,0)

    thw_pad = np.pad(
        thw,
        pad_width=((0,0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )
    return thw_pad, (pad_top, pad_bottom, pad_left, pad_right)