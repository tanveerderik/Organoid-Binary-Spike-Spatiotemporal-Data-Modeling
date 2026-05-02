#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:27:13 2026

@author: derik
"""

# visualize/report_rollout.py


import os
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..utils.recon import compute_activity_ctx
from ..utils.metrics import f1_from_bool



_CTX_NAMES = (
    "mean_firing_density",
    "frame_mean_std",
    "pixel_mean_std",
    "active_site_ratio",
    "temporal_slope",
)

def ctx_from_hw_t_binary(hw_t: np.ndarray) -> np.ndarray:
    return compute_activity_ctx((hw_t > 0).astype(np.uint8).transpose(2, 0, 1))

def ctx_dict(ctx_vec: np.ndarray) -> Dict[str, float]:
    ctx_vec = np.asarray(ctx_vec, dtype=np.float32).reshape(-1)
    return {k: float(v) for k, v in zip(_CTX_NAMES, ctx_vec.tolist())}

def summary_stats(prob_hw_t: np.ndarray, bin_hw_t: np.ndarray) -> Dict[str, float]:
    bin_bool = np.asarray(bin_hw_t) > 0
    frame_counts = bin_bool.sum(axis=(0, 1)).astype(np.float64)
    return {
        "mean_prob": float(np.asarray(prob_hw_t, dtype=np.float32).mean()),
        "mean_bin": float(bin_bool.mean()),
        "total_spikes_bin": float(bin_bool.sum()),
        "mean_spikes_per_frame_bin": float(frame_counts.mean()),
        "std_spikes_per_frame_bin": float(frame_counts.std(ddof=0)),
    }


# ---------- Report building ----------------

def build_full_generation_report(
    gen_out: Dict[str, Any],
    local_ctx: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    prob_hw_t = gen_out["prob_hw_t"]
    bin_hw_t = gen_out["bin_hw_t"]

    report = {
        "mode": "full_generation",
        "stats": summary_stats(prob_hw_t, bin_hw_t),
    }

    ctx_pred = ctx_from_hw_t_binary(bin_hw_t)
    report["ctx_pred_bin"] = ctx_pred
    report["ctx_pred_bin_dict"] = ctx_dict(ctx_pred)

    if local_ctx is not None:
        local_ctx = np.asarray(local_ctx, dtype=np.float32).reshape(-1)
        report["ctx_target"] = local_ctx
        report["ctx_target_dict"] = ctx_dict(local_ctx)
        report["ctx_abs_err_bin"] = np.abs(ctx_pred - local_ctx).astype(np.float32)
        report["ctx_mae_bin"] = float(np.abs(ctx_pred - local_ctx).mean())

    return report


def build_masked_generation_report(
    gen_out: Dict[str, Any],
    local_ctx: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    prob_hw_t = gen_out["prob_hw_t"]
    bin_hw_t = gen_out["bin_hw_t"]
    ref_hw_t = gen_out["ref_hw_t"]
    mask_hw_t = gen_out["predict_mask_hw_t"] > 0.5

    ref_bool = ref_hw_t > 0.5
    pred_bool = bin_hw_t > 0.5
    n_mask = int(mask_hw_t.sum())

    report = {
        "mode": "masked_generation",
        "stats": summary_stats(prob_hw_t, bin_hw_t),
        "masked_voxels": float(n_mask),
    }

    if n_mask > 0:
        report["masked_f1"] = float(f1_from_bool(ref_bool[mask_hw_t], pred_bool[mask_hw_t]))
        report["masked_acc"] = float((ref_bool[mask_hw_t] == pred_bool[mask_hw_t]).mean())
    else:
        report["masked_f1"] = float("nan")
        report["masked_acc"] = float("nan")

    ctx_pred = ctx_from_hw_t_binary(bin_hw_t)
    report["ctx_pred_bin"] = ctx_pred
    report["ctx_pred_bin_dict"] = ctx_dict(ctx_pred)

    if local_ctx is not None:
        local_ctx = np.asarray(local_ctx, dtype=np.float32).reshape(-1)
        report["ctx_target"] = local_ctx
        report["ctx_target_dict"] = ctx_dict(local_ctx)
        report["ctx_abs_err_bin"] = np.abs(ctx_pred - local_ctx).astype(np.float32)
        report["ctx_mae_bin"] = float(np.abs(ctx_pred - local_ctx).mean())

    return report


def build_causal_rollout_report(
    rollout_out: Dict[str, Any],
    *,
    prefix_frames: int,
    local_ctx_mode: str,
    fixed_local_ctx: Optional[np.ndarray] = None,
    gt_future_hw_t: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    vol = rollout_out["rollout_bin_hw_t"]
    stats = summary_stats(rollout_out["rollout_prob_hw_t"], vol)

    report_rows = []
    suffix_frames = int(rollout_out["suffix_frames"])
    start_t = int(rollout_out["window_frames"])

    for step_idx in range(int(rollout_out["rollout_steps"])):
        a = start_t + step_idx * suffix_frames
        b = a + suffix_frames
        pred_chunk = vol[:, :, a:b]

        if local_ctx_mode == "prefix_recomputed":
            prefix_a = a - prefix_frames
            prefix_b = a
            target_ctx = ctx_from_hw_t_binary(vol[:, :, prefix_a:prefix_b])
        else:
            target_ctx = np.asarray(fixed_local_ctx, dtype=np.float32).reshape(-1)

        pred_ctx = ctx_from_hw_t_binary(pred_chunk)

        row = {
            "step_idx": int(step_idx),
            "t0": int(a),
            "t1": int(b),
            "ctx_mae_new": float(np.abs(pred_ctx - target_ctx).mean()),
            **summary_stats(pred_chunk.astype(np.float32), pred_chunk),
        }

        for i, name in enumerate(_CTX_NAMES):
            row[f"ctx_target_{name}"] = float(target_ctx[i])
            row[f"ctx_pred_{name}"] = float(pred_ctx[i])
            row[f"ctx_abs_err_{name}"] = float(abs(pred_ctx[i] - target_ctx[i]))

        if gt_future_hw_t is not None and b <= gt_future_hw_t.shape[2]:
            gt_chunk = gt_future_hw_t[:, :, step_idx * suffix_frames:(step_idx + 1) * suffix_frames]
            gt_bool = gt_chunk > 0.5
            pred_bool = pred_chunk > 0.5
            row["future_f1"] = float(f1_from_bool(gt_bool, pred_bool))
            row["future_acc"] = float((gt_bool == pred_bool).mean())

        report_rows.append(row)

    return {
        "mode": "causal_long_rollout",
        "final_stats": stats,
        "rows": report_rows,
    }


# --------------- Report saving for plotting -----------------

def rows_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)

def save_report_table_csv(rows: List[Dict[str, Any]], path: str) -> None:
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def save_report_table_xlsx(rows: List[Dict[str, Any]], path: str) -> None:
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_excel(path, index=False)

def save_report_json(report: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.int32, np.int64)):
            return int(x)
        return x

    with open(path, "w") as f:
        json.dump(report, f, default=_convert, indent=2)
        
        
def plot_ctx_target_vs_pred(rows: List[Dict[str, Any]], out_path: str) -> None:
    df = pd.DataFrame(rows)
    n = len(_CTX_NAMES)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, _CTX_NAMES):
        x = df[f"ctx_target_{name}"].to_numpy()
        y = df[f"ctx_pred_{name}"].to_numpy()
        ax.scatter(x, y, s=20)
        mn = min(x.min(), y.min())
        mx = max(x.max(), y.max())
        ax.plot([mn, mx], [mn, mx], "--")
        ax.set_title(name)
        ax.set_xlabel("target")
        ax.set_ylabel("pred")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    
def plot_rollout_ctx_mae(rows: List[Dict[str, Any]], out_path: str) -> None:
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["step_idx"], df["ctx_mae_new"])
    ax.set_xlabel("rollout step")
    ax.set_ylabel("ctx MAE")
    ax.set_title("Local context drift across rollout")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    
def plot_rollout_activity(rows: List[Dict[str, Any]], out_path: str) -> None:
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["step_idx"], df["mean_spikes_per_frame_bin"])
    ax.set_xlabel("rollout step")
    ax.set_ylabel("mean spikes / frame")
    ax.set_title("Generated activity across rollout")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    

