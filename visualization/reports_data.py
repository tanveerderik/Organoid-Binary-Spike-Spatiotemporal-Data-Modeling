#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:03:40 2026

@author: derik
"""

import os
import re
import csv
import json
import glob
from typing import Any, Dict, List, Optional

import numpy as np


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True); return p

def _is_number(x):
    try:
        if x is None:
            return False
        v = float(x)
        return np.isfinite(v)
    except Exception:
        return False

def _safe_name(s: str) -> str:
    # safe filename: letters, numbers, underscore, dash, dot
    return re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", str(s))

class SampleResult:
    __slots__ = ("assay_name","assay_id","mode","task_id","f1","thr","ctx_ref","ctx_pred","json_path","npz_path")
    def __init__(self, **kw): 
        for k,v in kw.items(): setattr(self, k, v)
        
        

def find_results(eval_roots: List[str]) -> List[SampleResult]:
    out: List[SampleResult] = []
    for root in eval_roots:
        pattern = os.path.join(root, "**", "sample_*", "results.json")
        for jpath in glob.glob(pattern, recursive=True):
            try: j = _load_json(jpath)
            except Exception: continue
            npz_path = j.get("arrays_npz", None)
            if npz_path and not os.path.isabs(npz_path):
                npz_path = os.path.join(os.path.dirname(jpath), os.path.basename(npz_path))
            if npz_path and not os.path.isfile(npz_path): npz_path = None
            out.append(SampleResult(
                assay_name=str(j.get("assay_name","")),
                assay_id=(int(j["assay_id"]) if "assay_id" in j and j["assay_id"] is not None else None),
                mode=(str(j["mode"]) if "mode" in j and j["mode"] is not None else None),
                task_id=(int(j["task_id"]) if "task_id" in j and j["task_id"] is not None else None),
                f1=(float(j["f1_volume"]) if "f1_volume" in j and j["f1_volume"] is not None else None),
                thr=(float(j["threshold"]) if "threshold" in j and j["threshold"] is not None else None),
                ctx_ref=j.get("ctx_ref_on_window"),
                ctx_pred=j.get("ctx_pred_on_window"),
                json_path=jpath,
                npz_path=npz_path
            ))
    return out

# ---------------------- Aggregations & summaries -----------------------
def save_samples_csv(samples: List[SampleResult], out_dir: str, name="samples.csv"):
    out_dir = _ensure_dir(out_dir)
    path = os.path.join(out_dir, name)
    cols = ["assay_name","assay_id","mode","task_id","f1","thr","json_path","npz_path"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for s in samples:
            w.writerow([s.assay_name, s.assay_id, s.mode, s.task_id, s.f1, s.thr, s.json_path, s.npz_path])
            
            
def per_assay_stats(samples: List[SampleResult]) -> Dict[Any, Dict[str, float]]:
    groups: Dict[Any, List[float]] = {}
    for s in samples:
        key = s.assay_id if s.assay_id is not None else s.assay_name
        if s.f1 is not None:
            groups.setdefault(key, []).append(float(s.f1))
    stats = {}
    for k, vals in groups.items():
        vals = np.array(vals, np.float32)
        stats[k] = {
            "n": int(vals.size),
            "f1_mean": float(vals.mean()),
            "f1_std": float(vals.std(ddof=0)),
            "f1_min": float(vals.min()),
            "f1_max": float(vals.max()),
        }
    return stats


def save_per_assay_csv(stats: Dict[Any, Dict[str, float]], out_dir: str, name="per_assay.csv"):
    out_dir = _ensure_dir(out_dir)
    path = os.path.join(out_dir, name)
    cols = ["assay","n","f1_mean","f1_std","f1_min","f1_max"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for k, d in stats.items():
            w.writerow([k, d["n"], d["f1_mean"], d["f1_std"], d["f1_min"], d["f1_max"]])
            

# -------- Optional: threshold sweep from arrays.npz (if saved) ---------
def compute_pr_from_arrays(npz_path: str, thr_grid: Optional[np.ndarray] = None) -> Dict[str, Any]:
    d = np.load(npz_path)
    ref = (d["ref_u8"] > 0).astype(np.uint8)
    if "prob_u8" in d:
        p = d["prob_u8"].astype(np.float32) / 255.0
    elif "rollout_prob" in d:
        p = d["rollout_prob"].astype(np.float32)
    else:
        raise ValueError(f"{npz_path} missing prob array (prob_u8/rollout_prob).")
    if thr_grid is None: thr_grid = np.linspace(0.01, 0.99, 99)
    prec, rec, f1 = [], [], []
    for thr in thr_grid:
        pred = (p >= thr).astype(np.uint8)
        tp = np.logical_and(ref==1, pred==1).sum(dtype=np.int64)
        fp = np.logical_and(ref==0, pred==1).sum(dtype=np.int64)
        fn = np.logical_and(ref==1, pred==0).sum(dtype=np.int64)
        precision = tp / max(1, tp+fp); recall = tp / max(1, tp+fn)
        f1_val = 2*precision*recall / max(1e-8, precision+recall)
        prec.append(precision); rec.append(recall); f1.append(f1_val)
    f1 = np.array(f1); thr_grid = np.array(thr_grid)
    best = int(np.argmax(f1))
    return {"thr": thr_grid, "precision": np.array(prec), "recall": np.array(rec),
            "f1": f1, "best_idx": best, "best_thr": float(thr_grid[best]), "best_f1": float(f1[best])}



def _collect_numeric_keys(listdict, skip_keys=None):
    """
    Returns set of keys that have at least one numeric value in listdict.
    Uses your existing _is_number().
    """
    skip_keys = set(skip_keys or [])
    keys = set()
    if not isinstance(listdict, list):
        return keys
    for d in listdict:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if k in skip_keys:
                continue
            if _is_number(v):
                keys.add(k)
    return keys

def _get_series(listdict, key):
    """Return np.array(float) with NaNs where missing/non-numeric; or None if never numeric."""
    if not isinstance(listdict, list) or len(listdict) == 0:
        return None
    y = []
    any_ok = False
    for d in listdict:
        v = d.get(key, np.nan) if isinstance(d, dict) else np.nan
        if _is_number(v):
            any_ok = True
            y.append(float(v))
        else:
            y.append(np.nan)
    return np.asarray(y, dtype=float) if any_ok else None


#  ----------- Global data file that combines everything together ------------

import pandas as pd


def _to_df_with_epoch(records):
    df = pd.DataFrame(records)
    if df.empty:
        return df
    if "epoch" not in df.columns:
        df["epoch"] = range(1, len(df) + 1)
    return df


def _prefix_except(df, prefix, keep=("epoch",)):
    if df.empty:
        return df
    rename = {}
    for c in df.columns:
        if c not in keep:
            rename[c] = f"{prefix}{c}"
    return df.rename(columns=rename)


def _merge_train_val_history(train_log, val_metrics, stage_name, global_epoch_offset=0):
    """
    Returns one wide dataframe:
      one row per local epoch
      train_* and val_* columns side by side
    """
    df_train = _to_df_with_epoch(train_log)
    df_val   = _to_df_with_epoch(val_metrics)

    if df_train.empty and df_val.empty:
        return pd.DataFrame()

    df_train = _prefix_except(df_train, "train_")
    df_val   = _prefix_except(df_val, "val_")

    if df_train.empty:
        df = df_val.copy()
    elif df_val.empty:
        df = df_train.copy()
    else:
        df = pd.merge(df_train, df_val, on="epoch", how="outer", sort=True)

    df = df.sort_values("epoch").reset_index(drop=True)

    df.insert(0, "stage", stage_name)
    df = df.rename(columns={"epoch": "local_epoch"})
    df.insert(1, "global_epoch", df["local_epoch"] + global_epoch_offset)

    return df


def export_base_finetune_flat_xlsx(
    report_base_path: str,
    report_ft_path: str,
    out_xlsx: str,
    shift_finetune_by: str = "best",   # "best" or "full"
):
    """
    Export stage-1 and stage-3 histories into one flat/wide Excel sheet.

    Output:
      - one sheet named 'all_epochs'
      - one row per epoch
      - train_* and val_* side by side
    """

    with open(report_base_path, "r") as f:
        base = json.load(f)

    with open(report_ft_path, "r") as f:
        ft = json.load(f)

    base_hist = base.get("history", {})
    ft_hist   = ft.get("history", {})

    base_train = base_hist.get("train_log", [])
    base_val   = base_hist.get("val_metrics", [])
    ft_train   = ft_hist.get("train_log", [])
    ft_val     = ft_hist.get("val_metrics", [])

    base_best = int(base.get("best_epoch", 0) or 0)

    if shift_finetune_by == "best":
        ft_offset = base_best
    elif shift_finetune_by == "full":
        ft_offset = max(len(base_train), len(base_val))
    else:
        raise ValueError("shift_finetune_by must be 'best' or 'full'")

    df_base = _merge_train_val_history(
        base_train,
        base_val,
        stage_name="base",
        global_epoch_offset=0,
    )

    df_ft = _merge_train_val_history(
        ft_train,
        ft_val,
        stage_name="finetune",
        global_epoch_offset=ft_offset,
    )

    df_all = pd.concat([df_base, df_ft], ignore_index=True, sort=False)

    # helpful flags
    df_all["is_base"] = (df_all["stage"] == "base").astype(int)
    df_all["is_finetune"] = (df_all["stage"] == "finetune").astype(int)

    # optional summary columns repeated across rows for convenience
    df_all["base_best_epoch"] = base_best
    df_all["ft_shift_offset"] = ft_offset

    os.makedirs(os.path.dirname(out_xlsx) or ".", exist_ok=True)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="all_epochs", index=False)
        
    out_csv = out_xlsx.replace(".xlsx", ".csv")
    df_all.to_csv(out_csv, index=False)

    print(f"Saved flat combined report to: {out_xlsx} and {out_csv}")
    return df_all