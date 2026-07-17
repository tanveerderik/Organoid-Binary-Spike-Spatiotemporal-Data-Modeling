#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:03:53 2026

@author: derik
"""
import os
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .reports_data import (
    _load_json,
    _ensure_dir,
    _is_number,
    _safe_name,
    SampleResult,
    compute_pr_from_arrays,
    _collect_numeric_keys,
    _get_series,
    find_results,
    save_samples_csv,
    per_assay_stats,
    save_per_assay_csv,
    summarize_ctx_agreement,
    summarize_adjacency_agreement,
    save_agreement_csv,
)


LOCAL_CTX_NAMES = [
    "log_mean_firing_density",
    "var_x",
    "var_y",
    "var_t",
    "cov_xy",
    "cov_xt",
    "cov_yt",
    "active_site_ratio",
    "temporal_trend",
]

TASK_NAMES = {
    0: "exact reconstruction",
    1: "causal temporal prediction",
    2: "noncausal temporal completion",
    3: "spatial completion",
}

TASK_MODE_TO_NAME = {
    "recon": "exact reconstruction",
    "causal": "causal temporal prediction",
    "noncausal": "noncausal temporal completion",
    "spatial": "spatial completion",
}

ADJ_GAP_BINS = [(1, 1), (2, 2), (3, 3), (4, 6), (7, 12), (13, 24), (25, 48)]

def _gap_bin_label(gap_idx, gap_bins=None):
    gap_bins = gap_bins or ADJ_GAP_BINS
    lo, hi = gap_bins[gap_idx]
    return f"{lo}" if lo == hi else f"{lo}-{hi}"

def _plot_series(x, y, out_path, title, ylabel):
    if y is None or len(y) == 0:
        return
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    
    
def _plot_listdict_all_numeric(listdict, out_dir, prefix, x_key="epoch", skip_keys=None):
    """
    Plots all numeric keys in a list[dict] against x_key (default 'epoch').
    Saves: f"{prefix}_{key}.png" in out_dir
    """
    if not isinstance(listdict, list) or len(listdict) == 0:
        return

    skip_keys = set(skip_keys or [])
    # Build x-axis
    if isinstance(listdict[0], dict) and x_key in listdict[0]:
        x = []
        for i, d in enumerate(listdict):
            if isinstance(d, dict) and d.get(x_key, None) is not None:
                x.append(int(d.get(x_key)))
            else:
                x.append(i + 1)
        x = np.asarray(x)
    else:
        x = np.arange(1, len(listdict) + 1)

    # Collect all keys
    keys = set()
    for d in listdict:
        if isinstance(d, dict):
            keys.update(d.keys())
    keys = sorted(keys)

    # Plot each numeric key
    for k in keys:
        if k in skip_keys:
            continue

        y = []
        any_numeric = False
        for d in listdict:
            v = d.get(k, np.nan) if isinstance(d, dict) else np.nan
            if _is_number(v):
                any_numeric = True
                y.append(float(v))
            else:
                y.append(np.nan)

        if not any_numeric:
            continue

        fn = f"{prefix}_{_safe_name(k)}.png"
        out_path = os.path.join(out_dir, fn)
        _plot_series(
            x=x,
            y=np.asarray(y, dtype=float),
            out_path=out_path,
            title=f"{prefix}: {k}",
            ylabel=str(k),
        )

def plot_report_json(train_report_path: str, out_dir: str):
    out_dir = _ensure_dir(out_dir)
    rep = _load_json(train_report_path)
    hist = rep.get("history", {})

    # --- New schema: history has train_log and val_metrics ---
    train_log = hist.get("train_log", [])
    val_metrics = hist.get("val_metrics", [])

    # Plot EVERYTHING in train_log
    _plot_listdict_all_numeric(
        listdict=train_log,
        out_dir=out_dir,
        prefix="train",
        x_key="epoch",
        skip_keys={"epoch", "time_sec"},
    )

    # Plot EVERYTHING in val_metrics
    # If your val report dicts include "epoch", it will use it; otherwise it uses 1..N.
    _plot_listdict_all_numeric(
        listdict=val_metrics,
        out_dir=out_dir,
        prefix="val",
        x_key="epoch",
        skip_keys={"epoch"},
    )

    # Optional: print summary
    if (not isinstance(train_log, list) or len(train_log) == 0) and (not isinstance(val_metrics, list) or len(val_metrics) == 0):
        print("[plot_report_json] Nothing to plot (no train_log or val_metrics).")

def plot_f1_box_by_assay(samples: List[SampleResult], out_dir: str, min_count: int = 1):
    out_dir = _ensure_dir(out_dir)

    # collect values + a numeric sort key if we have an assay_id
    groups: Dict[str, Dict[str, Any]] = {}
    for s in samples:
        if s.f1 is None:
            continue
        label = str(s.assay_id) if s.assay_id is not None else str(s.assay_name)
        g = groups.setdefault(label, {"vals": [], "sort_key": None})
        g["vals"].append(float(s.f1))
        if g["sort_key"] is None and s.assay_id is not None:
            g["sort_key"] = int(s.assay_id)

    # keep all non-empty groups (or enforce min_count if you want)
    items = [(lbl, info["vals"], info["sort_key"])
             for lbl, info in groups.items() if len(info["vals"]) >= min_count]

    if not items:
        return

    # sort: first by “has numeric key?”, then by numeric key, then by label
    items.sort(key=lambda x: (x[2] is None, x[2] if x[2] is not None else 0, x[0]))

    labels = [lbl for lbl, _, _ in items]
    data   = [vals for _, vals, _ in items]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(max(6, 0.3*len(labels)+2), 4.5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("F1 (volume)")
    plt.title("Per-assay F1 distribution (sorted by assay ID)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "f1_box_by_assay.png"), dpi=160)
    plt.close()


def plot_ctx_scatter(
    samples,
    out_dir,
    dim=0,
    ctx_names=None,
    task_names=None,
    color_by_task=True,
    title_prefix="Local context agreement",
    filename_prefix="ctx_scatter",
):
    """
    Scatter of ctx_pred[dim] vs ctx_ref[dim].
    Optionally colored by dataset task:
      recon / causal / noncausal / spatial.
    """
    out_dir = _ensure_dir(out_dir)
    ctx_names = ctx_names or LOCAL_CTX_NAMES
    task_names = task_names or TASK_NAMES

    ctx_name = ctx_names[dim] if dim < len(ctx_names) else f"ctx[{dim}]"

    groups = {}

    for s in samples:
        if s.ctx_ref is None or s.ctx_pred is None:
            continue
        if len(s.ctx_ref) <= dim or len(s.ctx_pred) <= dim:
            continue

        xr = float(s.ctx_ref[dim])
        yp = float(s.ctx_pred[dim])

        mode = (s.mode or "").lower().strip()

        if s.task_id is not None and int(s.task_id) in task_names:
            label = task_names[int(s.task_id)]
        elif mode in TASK_MODE_TO_NAME:
            label = TASK_MODE_TO_NAME[mode]
        elif mode:
            label = mode
        else:
            label = "unknown task"

        groups.setdefault(label, {"x": [], "y": []})
        groups[label]["x"].append(xr)
        groups[label]["y"].append(yp)

    if not groups:
        return

    all_x = np.asarray(
        [v for g in groups.values() for v in g["x"]],
        dtype=float,
    )
    all_y = np.asarray(
        [v for g in groups.values() for v in g["y"]],
        dtype=float,
    )

    vmin = float(min(all_x.min(), all_y.min()))
    vmax = float(max(all_x.max(), all_y.max()))
    pad = 0.02 * (vmax - vmin + 1e-12)
    lim = (vmin - pad, vmax + pad)

    plt.figure(figsize=(4.5, 4.5))

    if color_by_task:
        for label, g in groups.items():
            plt.scatter(
                g["x"],
                g["y"],
                s=10,
                alpha=0.7,
                label=label,
            )
    else:
        plt.scatter(all_x, all_y, s=10, alpha=0.7, label="samples")

    plt.plot(lim, lim, linewidth=1.0, linestyle="--")
    plt.xlim(lim)
    plt.ylim(lim)

    plt.xlabel(f"Reference: {ctx_name}")
    plt.ylabel(f"Predicted: {ctx_name}")
    plt.title(f"{title_prefix}: {ctx_name}")

    if color_by_task:
        plt.legend(loc="best", frameon=False, fontsize=8)

    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, f"{filename_prefix}_{dim}_{_safe_name(ctx_name)}.png"),
        dpi=160,
    )
    plt.close()
    
    
def plot_ctx_scatter_stage_overlay(
    samples_stage1,
    samples_stage2,
    out_dir,
    dim=0,
    ctx_names=None,
    stage1_label="Stage 1",
    stage2_label="Stage 2",
):
    out_dir = _ensure_dir(out_dir)
    ctx_names = ctx_names or LOCAL_CTX_NAMES
    ctx_name = ctx_names[dim] if dim < len(ctx_names) else f"ctx[{dim}]"

    def collect(samples):
        xs, ys = [], []
        for s in samples:
            if s.ctx_ref is None or s.ctx_pred is None:
                continue
            if len(s.ctx_ref) <= dim or len(s.ctx_pred) <= dim:
                continue
            xs.append(float(s.ctx_ref[dim]))
            ys.append(float(s.ctx_pred[dim]))
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    x1, y1 = collect(samples_stage1)
    x2, y2 = collect(samples_stage2)

    if len(x1) + len(x2) == 0:
        return

    all_x = np.concatenate([x1, x2]) if len(x1) and len(x2) else (x1 if len(x1) else x2)
    all_y = np.concatenate([y1, y2]) if len(y1) and len(y2) else (y1 if len(y1) else y2)

    vmin = float(min(all_x.min(), all_y.min()))
    vmax = float(max(all_x.max(), all_y.max()))
    pad = 0.02 * (vmax - vmin + 1e-12)
    lim = (vmin - pad, vmax + pad)

    plt.figure(figsize=(4.5, 4.5))

    if len(x1):
        plt.scatter(x1, y1, s=12, alpha=0.45, label=stage1_label)

    if len(x2):
        plt.scatter(x2, y2, s=12, alpha=0.45, label=stage2_label)

    plt.plot(lim, lim, linewidth=1.0, linestyle="--")
    plt.xlim(lim)
    plt.ylim(lim)

    plt.xlabel(f"Reference: {ctx_name}")
    plt.ylabel(f"Predicted: {ctx_name}")
    plt.title(f"Local context: {ctx_name}")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, f"ctx_stage_overlay_{dim}_{_safe_name(ctx_name)}.png"),
        dpi=180,
    )
    plt.close()
    
    
def plot_adjacency_scatter(
    samples,
    out_dir,
    gap_idx=0,
    adj_margin=0.25,
    title_prefix="Adjacency / short-gap agreement",
    filename_prefix="adj_scatter",
):
    out_dir = _ensure_dir(out_dir)
    gap_bins = getattr(samples[0], "adj_gap_bins", None) if len(samples) else None
    gap_label = _gap_bin_label(gap_idx, gap_bins)

    xs, ys = [], []

    for s in samples:
        if s.adj_target is None or s.adj_pred is None:
            continue
        if len(s.adj_target) <= gap_idx or len(s.adj_pred) <= gap_idx:
            continue

        xs.append(float(s.adj_target[gap_idx]))
        ys.append(float(s.adj_pred[gap_idx]))

    if len(xs) == 0:
        return

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)

    vmax = float(max(x.max(), y.max(), 1e-8))
    vmax = max(vmax, float(vmax * (1.0 + adj_margin)))
    lim = (0.0, vmax * 1.08)

    plt.figure(figsize=(4.6, 4.6))
    plt.scatter(x, y, s=14, alpha=0.65, label="samples")

    plt.plot(lim, lim, linestyle="--", linewidth=1.0, label="target")

    plt.plot(
        lim,
        [v * (1.0 + adj_margin) for v in lim],
        linestyle=":",
        linewidth=1.0,
        label=f"allowed (+{adj_margin:.0%})",
    )

    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel(f"Reference gap {gap_label} rate")
    plt.ylabel(f"Predicted gap {gap_label} rate")
    plt.title(f"{title_prefix}: gap {gap_label}")
    
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"{filename_prefix}_gap{gap_label.replace('-', '_to_')}.png"),
        dpi=180,
    )
    plt.close()


def plot_adjacency_scatter_stage_overlay(
    samples_stage1,
    samples_stage2,
    out_dir,
    gap_idx=0,
    adj_margin=0.25,
    stage1_label="Stage 1",
    stage2_label="Stage 2",
    title_prefix="Adjacency / short-gap agreement",
    filename_prefix="adj_stage_overlay",
):
    out_dir = _ensure_dir(out_dir)
    
    gap_bins = None
    for _s in list(samples_stage1) + list(samples_stage2):
        gap_bins = getattr(_s, "adj_gap_bins", None)
        if gap_bins is not None:
            break
    gap_label = _gap_bin_label(gap_idx, gap_bins)
    
    def collect(samples):
        xs, ys = [], []
        for s in samples:
            if s.adj_target is None or s.adj_pred is None:
                continue
            if len(s.adj_target) <= gap_idx or len(s.adj_pred) <= gap_idx:
                continue

            xs.append(float(s.adj_target[gap_idx]))
            ys.append(float(s.adj_pred[gap_idx]))

        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    x1, y1 = collect(samples_stage1)
    x2, y2 = collect(samples_stage2)

    if len(x1) + len(x2) == 0:
        return

    all_x = np.concatenate([a for a in [x1, x2] if len(a)])
    all_y = np.concatenate([a for a in [y1, y2] if len(a)])

    vmax = float(max(all_x.max(), all_y.max(), 1e-8))
    vmax = max(vmax, float(vmax * (1.0 + adj_margin)))
    lim = (0.0, vmax * 1.08)

    plt.figure(figsize=(4.6, 4.6))

    if len(x1):
        plt.scatter(x1, y1, s=14, alpha=0.45, label=stage1_label)

    if len(x2):
        plt.scatter(x2, y2, s=14, alpha=0.45, label=stage2_label)

    plt.plot(lim, lim, linestyle="--", linewidth=1.0, label="target")

    plt.plot(
        lim,
        [v * (1.0 + adj_margin) for v in lim],
        linestyle=":",
        linewidth=1.0,
        label=f"allowed (+{adj_margin:.0%})",
    )

    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel(f"Reference gap {gap_label} rate")
    plt.ylabel(f"Predicted gap {gap_label} rate")
    plt.title(f"{title_prefix}: gap {gap_label}")
    
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"{filename_prefix}_gap{gap_label.replace('-', '_to_')}.png"),
        dpi=180,
    )
    plt.close()
    
    
    
def sweep_all(npzs: List[str], out_dir: str, max_items: int = 200):
    out_dir = _ensure_dir(out_dir)
    curves = []
    for p in npzs[:max_items]:
        try: curves.append(compute_pr_from_arrays(p))
        except Exception: pass
    if not curves: return
    thr = curves[0]["thr"]
    prec = np.stack([c["precision"] for c in curves], 0)
    rec  = np.stack([c["recall"]    for c in curves], 0)
    f1   = np.stack([c["f1"]        for c in curves], 0)
    # Mean PR
    plt.figure(); plt.plot(rec.mean(0), prec.mean(0))
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Mean PR curve"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_pr.png"), dpi=160); plt.close()
    # Mean F1 vs thr
    plt.figure(); plt.plot(thr, f1.mean(0))
    plt.xlabel("Threshold"); plt.ylabel("F1"); plt.title("Mean F1 vs Threshold"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_f1_vs_thr.png"), dpi=160); plt.close()
    
    
def _plot_overlay(
    x_base, y_base,
    x_ft, y_ft,
    *,
    out_path_png,
    out_path_pdf=None,
    title="",
    ylabel="",
    base_label="Base (no prior)",
    ft_label="Prior-guided finetune",
    finetune_start_epoch=None,
    base_best_epoch=None,
    ft_best_epoch_global=None,
):
    if y_base is None and y_ft is None:
        return

    plt.figure(figsize=(6.2, 4.2))

    if y_base is not None:
        plt.plot(x_base, y_base, label=base_label, linewidth=2)

    if y_ft is not None:
        plt.plot(x_ft, y_ft, label=ft_label, linewidth=2)

    if finetune_start_epoch is not None:
        plt.axvline(finetune_start_epoch, linestyle="--", linewidth=1.5, color="black", label="Finetune start")

    # markers at best epochs (optional)
    if base_best_epoch is not None and y_base is not None:
        if 1 <= base_best_epoch <= len(y_base):
            plt.scatter([base_best_epoch], [y_base[base_best_epoch - 1]], s=45, zorder=5, marker="o")

    if ft_best_epoch_global is not None and y_ft is not None:
        idx = np.where(np.asarray(x_ft) == ft_best_epoch_global)[0]
        if idx.size > 0:
            j = int(idx[0])
            plt.scatter([x_ft[j]], [y_ft[j]], s=55, zorder=6, marker="*")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=300)
    if out_path_pdf is not None:
        plt.savefig(out_path_pdf)  # vector
    plt.close()
    
    
    
def run_plotter(train_report: Optional[str],
        eval_roots: List[str],
        out_dir: str = "figs_vqvae",
        do_threshold_sweep: bool = False):
    out_dir = _ensure_dir(out_dir)

    # 1) training curves
    if train_report and os.path.isfile(train_report):
        plot_report_json(train_report, out_dir=os.path.join(out_dir, "training"))

    # 2) eval samples
    samples = find_results(eval_roots)
    save_samples_csv(samples, out_dir, "samples.csv")
    
    ctx_rows = summarize_ctx_agreement(samples, ctx_names=LOCAL_CTX_NAMES)
    save_agreement_csv(ctx_rows, out_dir, "local_context_feature_consistency.csv")
    
    adj_rows = summarize_adjacency_agreement(samples)
    save_agreement_csv(adj_rows, out_dir, "short_gap_adjacency_consistency.csv")

    # 3) per-assay stats + plots
    stats = per_assay_stats(samples)
    save_per_assay_csv(stats, out_dir, "per_assay.csv")
    plot_f1_box_by_assay(samples, os.path.join(out_dir, "eval"))
    for dim in range(len(LOCAL_CTX_NAMES)):  # compute_activity_ctx returns 9 dims in utils/recon.py
        plot_ctx_scatter(
            samples,
            out_dir=os.path.join(out_dir, "eval"),
            dim=dim,
            ctx_names=LOCAL_CTX_NAMES,
            task_names=TASK_NAMES,
            color_by_task=False,
        )
    adj_dir = _ensure_dir(os.path.join(out_dir, "eval", "adjacency"))
    
    n_adj = max((len(s.adj_target) for s in samples if s.adj_target is not None), default=0)
    for gap_idx in range(n_adj):
        plot_adjacency_scatter(
            samples,
            out_dir=adj_dir,
            gap_idx=gap_idx,
        )

    # you can call plot_ctx_scatter for other dims manually if you want

    # 4) optional: threshold sweep (needs arrays.npz saved)
    if do_threshold_sweep:
        npzs = [s.npz_path for s in samples if s.npz_path]
        sweep_all(npzs, os.path.join(out_dir, "sweep"))

    print(f"[analyze_results_vqvae] Done. Output in: {out_dir}")




def plot_base_then_finetune(
    report_base_path: str,
    report_ft_path: str,
    out_dir: str,
    *,
    mode: str = "union",              # "union" or "shared"
    truncate_base_at_best: bool = True,
    save_pdf: bool = True,
    # key skipping
    train_skip=("epoch", "time_sec"),
    val_skip=("epoch",),
    # labels
    base_label="Stage 1: context-agnostic VQ-VAE",
    ft_label="Stage 2: decoder context adaptation",
    base_eval_roots=None,
    ft_eval_roots=None,
):
    out_dir = _ensure_dir(out_dir)

    base = _load_json(report_base_path)
    ft   = _load_json(report_ft_path)

    base_best = int(base.get("best_epoch", 0) or 0)
    ft_best   = int(ft.get("best_epoch", 0) or 0)
    ft_best_global = (base_best + ft_best) if (base_best > 0 and ft_best > 0) else None

    base_hist = base.get("history", {})
    ft_hist   = ft.get("history", {})

    base_train = base_hist.get("train_log", [])
    base_val   = base_hist.get("val_metrics", [])
    ft_train   = ft_hist.get("train_log", [])
    ft_val     = ft_hist.get("val_metrics", [])

    # collect keys
    base_train_keys = _collect_numeric_keys(base_train, skip_keys=train_skip)
    base_val_keys   = _collect_numeric_keys(base_val,   skip_keys=val_skip)
    ft_train_keys   = _collect_numeric_keys(ft_train,   skip_keys=train_skip)
    ft_val_keys     = _collect_numeric_keys(ft_val,     skip_keys=val_skip)

    mode = (mode or "union").lower().strip()
    if mode == "shared":
        train_keys = sorted(base_train_keys & ft_train_keys)
        val_keys   = sorted(base_val_keys   & ft_val_keys)
    else:
        train_keys = sorted(base_train_keys | ft_train_keys)
        val_keys   = sorted(base_val_keys   | ft_val_keys)

    # X axes
    x_base_train = np.arange(1, len(base_train) + 1)
    x_base_val   = np.arange(1, len(base_val) + 1)

    # finetune shift so its epoch 1 becomes base_best+1
    # (if base_best is 0/missing, it still works: shift=0)
    shift = base_best if base_best > 0 else 0
    x_ft_train = shift + np.arange(1, len(ft_train) + 1)
    x_ft_val   = shift + np.arange(1, len(ft_val) + 1)

    # optionally truncate base at best_epoch
    if truncate_base_at_best and base_best > 0:
        x_base_train = x_base_train[:min(base_best, len(x_base_train))]
        x_base_val   = x_base_val[:min(base_best, len(x_base_val))]
        base_train   = base_train[:len(x_base_train)]
        base_val     = base_val[:len(x_base_val)]

    finetune_start_epoch = base_best if base_best > 0 else None

    # ---- Plot ALL validation overlays ----
    val_dir = _ensure_dir(os.path.join(out_dir, "val"))
    for k in val_keys:
        yb = _get_series(base_val, k)
        yf = _get_series(ft_val, k)

        out_png = os.path.join(val_dir, f"val_overlay_{_safe_name(k)}.png")
        out_pdf = os.path.join(val_dir, f"val_overlay_{_safe_name(k)}.pdf") if save_pdf else None

        _plot_overlay(
            x_base=x_base_val, y_base=yb,
            x_ft=x_ft_val,     y_ft=yf,
            out_path_png=out_png,
            out_path_pdf=out_pdf,
            title=f"Validation {k}: base → finetune",
            ylabel=str(k),
            base_label=base_label,
            ft_label=ft_label,
            finetune_start_epoch=finetune_start_epoch,
            base_best_epoch=None if truncate_base_at_best else base_best,
            ft_best_epoch_global=ft_best_global,
        )

    # ---- Plot ALL train overlays ----
    train_dir = _ensure_dir(os.path.join(out_dir, "train"))
    for k in train_keys:
        yb = _get_series(base_train, k)
        yf = _get_series(ft_train, k)

        out_png = os.path.join(train_dir, f"train_overlay_{_safe_name(k)}.png")
        out_pdf = os.path.join(train_dir, f"train_overlay_{_safe_name(k)}.pdf") if save_pdf else None

        _plot_overlay(
            x_base=x_base_train, y_base=yb,
            x_ft=x_ft_train,     y_ft=yf,
            out_path_png=out_png,
            out_path_pdf=out_pdf,
            title=f"Train {k}: base → finetune",
            ylabel=str(k),
            base_label=base_label,
            ft_label=ft_label,
            finetune_start_epoch=finetune_start_epoch,
            base_best_epoch=None if truncate_base_at_best else base_best,
            ft_best_epoch_global=None,  # usually not needed for train
        )


        # ---- Local context scatter overlays: base vs finetune ----
    if base_eval_roots is not None and ft_eval_roots is not None:
        samples_base = find_results(base_eval_roots)
        samples_ft = find_results(ft_eval_roots)
        
        base_ctx_rows = summarize_ctx_agreement(samples_base, ctx_names=LOCAL_CTX_NAMES)
        ft_ctx_rows = summarize_ctx_agreement(samples_ft, ctx_names=LOCAL_CTX_NAMES)
        
        base_ctx_rows["stage"] = "Stage 1"
        ft_ctx_rows["stage"] = "Stage 2"
        
        save_agreement_csv(
            pd.concat([base_ctx_rows, ft_ctx_rows], ignore_index=True),
            out_dir,
            "local_context_feature_consistency_stage1_stage2.csv",
        )
        
        base_adj_rows = summarize_adjacency_agreement(samples_base, max_gaps=len(ADJ_GAP_BINS))
        ft_adj_rows = summarize_adjacency_agreement(samples_ft, max_gaps=len(ADJ_GAP_BINS))
        
        base_adj_rows["stage"] = "Stage 1"
        ft_adj_rows["stage"] = "Stage 2"
        
        save_agreement_csv(
            pd.concat([base_adj_rows, ft_adj_rows], ignore_index=True),
            out_dir,
            "short_gap_adjacency_consistency_stage1_stage2.csv",
        )

        ctx_dir = _ensure_dir(os.path.join(out_dir, "ctx_stage_overlay"))

        for dim in range(len(LOCAL_CTX_NAMES)):
            plot_ctx_scatter_stage_overlay(
                samples_base,
                samples_ft,
                out_dir=ctx_dir,
                dim=dim,
                ctx_names=LOCAL_CTX_NAMES,
                stage1_label=base_label,
                stage2_label=ft_label,
            )
            
        adj_dir = _ensure_dir(os.path.join(out_dir, "adj_stage_overlay"))
        
        n_adj = max(
            [len(s.adj_target) for s in list(samples_base) + list(samples_ft) if s.adj_target is not None],
            default=0,
        )
        for gap_idx in range(n_adj):
            plot_adjacency_scatter_stage_overlay(
                samples_base,
                samples_ft,
                out_dir=adj_dir,
                gap_idx=gap_idx,
                stage1_label=base_label,
                stage2_label=ft_label,
            )

    # Quick audit print
    only_base_val = sorted(base_val_keys - ft_val_keys)
    only_ft_val   = sorted(ft_val_keys - base_val_keys)
    only_base_tr  = sorted(base_train_keys - ft_train_keys)
    only_ft_tr    = sorted(ft_train_keys - base_train_keys)

    print(f"[plot_base_then_finetune] wrote overlays to: {out_dir}")
    print(f"  base best_epoch={base_best}, ft best_epoch={ft_best} (global={ft_best_global})")
    print(f"  val keys:   {len(val_keys)}  (only_base={len(only_base_val)}, only_ft={len(only_ft_val)})")
    print(f"  train keys: {len(train_keys)} (only_base={len(only_base_tr)}, only_ft={len(only_ft_tr)})")
    
    
