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

from ..utils.constants import (
    ACTIVITY_CTX_NAMES,
)
CTX_NAMES = list(ACTIVITY_CTX_NAMES)

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
    __slots__ = (
        "assay_name", "assay_id", "mode", "task_id",
        "f1", "thr",
        "ctx_ref", "ctx_pred",
        "adj_target", "adj_pred", "adj_allowed", "adj_confidence", "adj_gap_bins",
        "json_path", "npz_path",
    )
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
            adj = j.get("adjacency_rates", {}) or {}
            out.append(SampleResult(
                assay_name=str(j.get("assay_name","")),
                assay_id=(int(j["assay_id"]) if "assay_id" in j and j["assay_id"] is not None else None),
                mode=(str(j["mode"]) if "mode" in j and j["mode"] is not None else None),
                task_id=(int(j["task_id"]) if "task_id" in j and j["task_id"] is not None else None),
                f1=(float(j["f1_volume"]) if "f1_volume" in j and j["f1_volume"] is not None else None),
                thr=(float(j["threshold"]) if "threshold" in j and j["threshold"] is not None else None),
                ctx_ref=j.get("ctx_ref_on_window"),
                ctx_pred=j.get("ctx_pred_on_window"),
                adj_target=adj.get("target_gap_rates"),
                adj_pred=adj.get("pred_gap_rates"),
                adj_allowed=adj.get("allowed_gap_rates"),
                adj_confidence=adj.get("confidence"),
                adj_gap_bins=adj.get("gap_bins"),
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


# --- Quantitative tables for samplewise visualization results ---



def _safe_float(x):
    try:
        if x is None:
            return np.nan
        y = float(x)
        return y if np.isfinite(y) else np.nan
    except Exception:
        return np.nan


def _as_list(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return None


def _pearson_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    if x.size < 2:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def _rankdata_average(a):
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)

    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j

    return ranks


def _spearman_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    if x.size < 2:
        return np.nan

    return _pearson_safe(_rankdata_average(x), _rankdata_average(y))


def _mean_std_str(mean, std, digits=4):
    if not np.isfinite(mean):
        return ""
    if not np.isfinite(std):
        return f"{mean:.{digits}g}"
    return f"{mean:.{digits}g} ± {std:.{digits}g}"


def _fmt_num(x, digits=4):
    return "" if not np.isfinite(x) else f"{x:.{digits}g}"


def build_viz_quant_raw_tables(eval_roots, stage_names=None, ctx_names=None):
    """
    Reads sample_*/results.json files and returns:
      samples_df, ctx_df, adj_df
    """

    ctx_names = ctx_names or CTX_NAMES

    if isinstance(eval_roots, (str, os.PathLike)):
        eval_roots = [eval_roots]

    if stage_names is None:
        stage_names = [os.path.basename(os.path.abspath(str(r))) for r in eval_roots]

    sample_rows = []
    ctx_rows = []
    adj_rows = []

    for stage, root in zip(stage_names, eval_roots):
        pattern = os.path.join(str(root), "**", "sample_*", "results.json")
        paths = sorted(glob.glob(pattern, recursive=True))

        print(f"[quant-tables] {stage}: found {len(paths)} results.json files under {root}")

        for json_path in paths:
            try:
                j = _load_json(json_path)
            except Exception as e:
                print(f"[skip] Could not read {json_path}: {e}")
                continue

            sample_dir = os.path.dirname(json_path)
            sample_name = os.path.basename(sample_dir)

            arrays_npz = j.get("arrays_npz", None)
            npz_path = None
            if isinstance(arrays_npz, str):
                npz_path = arrays_npz if os.path.isabs(arrays_npz) else os.path.join(sample_dir, arrays_npz)
                if not os.path.isfile(npz_path):
                    npz_path = None

            base = {
                "stage": stage,
                "sample_name": sample_name,
                "assay_name": str(j.get("assay_name", "")),
                "assay_id": j.get("assay_id", None),
                "mode": j.get("mode", None),
                "task_id": j.get("task_id", None),
                "threshold": _safe_float(j.get("threshold", np.nan)),
                "f1_volume": _safe_float(j.get("f1_volume", np.nan)),
                "json_path": json_path,
                "npz_path": npz_path,
            }

            sample_rows.append(dict(base))

            ctx_ref = _as_list(j.get("ctx_ref_on_window", None))
            ctx_pred = _as_list(j.get("ctx_pred_on_window", None))

            if ctx_ref is not None and ctx_pred is not None:
                n_ctx = min(len(ctx_ref), len(ctx_pred), len(ctx_names))
                for d in range(n_ctx):
                    ref = _safe_float(ctx_ref[d])
                    pred = _safe_float(ctx_pred[d])
                    err = pred - ref if np.isfinite(ref) and np.isfinite(pred) else np.nan

                    row = dict(base)
                    row.update({
                        "ctx_dim": d,
                        "ctx_name": ctx_names[d],
                        "ctx_ref": ref,
                        "ctx_pred": pred,
                        "ctx_error": err,
                    })
                    ctx_rows.append(row)

            adj = j.get("adjacency_rates", {}) or {}
            if isinstance(adj, dict):
                adj_target = _as_list(adj.get("target_gap_rates", None))
                adj_pred = _as_list(adj.get("pred_gap_rates", None))
                adj_allowed = _as_list(adj.get("allowed_gap_rates", None))
                adj_conf = _as_list(adj.get("confidence", None))

                if adj_target is not None and adj_pred is not None:
                    n_gap = min(len(adj_target), len(adj_pred))
                    for g in range(n_gap):
                        target = _safe_float(adj_target[g])
                        pred = _safe_float(adj_pred[g])
                        allowed = _safe_float(adj_allowed[g]) if adj_allowed is not None and g < len(adj_allowed) else np.nan
                        conf = _safe_float(adj_conf[g]) if adj_conf is not None and g < len(adj_conf) else np.nan

                        err = pred - target if np.isfinite(target) and np.isfinite(pred) else np.nan
                        violation = max(0.0, pred - allowed) if np.isfinite(pred) and np.isfinite(allowed) else np.nan

                        row = dict(base)
                        
                        gap_bins = adj.get("gap_bins", None)
                        if gap_bins is not None and g < len(gap_bins):
                            lo, hi = gap_bins[g]
                            gap_label = str(lo) if int(lo) == int(hi) else f"{int(lo)}-{int(hi)}"
                        else:
                            gap_label = str(g + 1)
                        
                        
                        
                        row.update({
                            "gap_idx": g,
                            "gap": gap_label,
                            "adj_target": target,
                            "adj_pred": pred,
                            "adj_allowed": allowed,
                            "adj_confidence": conf,
                            "adj_error": err,
                            "adj_violation": violation,
                        })
                        adj_rows.append(row)

    return pd.DataFrame(sample_rows), pd.DataFrame(ctx_rows), pd.DataFrame(adj_rows)


def _summarize_pairwise(df, group_cols, ref_col, pred_col, allowed_col=None, violation_col=None):
    rows = []

    if df.empty:
        return pd.DataFrame()

    for keys, d in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = dict(zip(group_cols, keys))

        ref = pd.to_numeric(d[ref_col], errors="coerce").to_numpy(float)
        pred = pd.to_numeric(d[pred_col], errors="coerce").to_numpy(float)

        ok = np.isfinite(ref) & np.isfinite(pred)
        ref = ref[ok]
        pred = pred[ok]
        diff = pred - ref

        row = dict(base)
        row.update({
            "n_total": int(len(d)),
            "n_valid": int(ok.sum()),
            "pearson_r": _pearson_safe(ref, pred),
            "spearman_r": _spearman_safe(ref, pred),
            "ref_mean": float(np.mean(ref)) if ref.size else np.nan,
            "ref_std": float(np.std(ref, ddof=0)) if ref.size else np.nan,
            "pred_mean": float(np.mean(pred)) if pred.size else np.nan,
            "pred_std": float(np.std(pred, ddof=0)) if pred.size else np.nan,
            "diff_mean": float(np.mean(diff)) if diff.size else np.nan,
            "diff_std": float(np.std(diff, ddof=0)) if diff.size else np.nan,
        })

        if allowed_col is not None and allowed_col in d.columns:
            allowed = pd.to_numeric(d.loc[ok, allowed_col], errors="coerce").to_numpy(float)
            row["allowed_mean"] = float(np.nanmean(allowed)) if np.isfinite(allowed).any() else np.nan
            row["allowed_std"] = float(np.nanstd(allowed, ddof=0)) if np.isfinite(allowed).any() else np.nan

        if violation_col is not None and violation_col in d.columns:
            viol = pd.to_numeric(d.loc[ok, violation_col], errors="coerce").to_numpy(float)
            row["violation_mean"] = float(np.nanmean(viol)) if np.isfinite(viol).any() else np.nan
            row["violation_std"] = float(np.nanstd(viol, ddof=0)) if np.isfinite(viol).any() else np.nan
            row["violation_fraction"] = float(np.nanmean(viol > 0)) if np.isfinite(viol).any() else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def make_viz_quant_manuscript_tables(ctx_summary, adj_summary):
    ctx_ms = pd.DataFrame()
    adj_ms = pd.DataFrame()

    if not ctx_summary.empty:
        ctx_ms = ctx_summary.copy()
        ctx_ms.insert(0, "feature_type", "local_context")
        ctx_ms["feature"] = ctx_ms["ctx_name"]

        ctx_ms["target_mean_std"] = [
            _mean_std_str(m, s) for m, s in zip(ctx_ms["ref_mean"], ctx_ms["ref_std"])
        ]
        ctx_ms["pred_mean_std"] = [
            _mean_std_str(m, s) for m, s in zip(ctx_ms["pred_mean"], ctx_ms["pred_std"])
        ]
        ctx_ms["diff_mean_std"] = [
            _mean_std_str(m, s) for m, s in zip(ctx_ms["diff_mean"], ctx_ms["diff_std"])
        ]

        ctx_ms["pearson_r_fmt"] = ctx_ms["pearson_r"].map(_fmt_num)
        ctx_ms["spearman_r_fmt"] = ctx_ms["spearman_r"].map(_fmt_num)

        ctx_ms = ctx_ms[[
            "stage",
            "feature_type",
            "feature",
            "n_valid",
            "target_mean_std",
            "pred_mean_std",
            "pearson_r_fmt",
            "spearman_r_fmt",
            "diff_mean_std",
        ]]

    if not adj_summary.empty:
        adj_ms = adj_summary.copy()
        adj_ms.insert(0, "feature_type", "short_gap_adjacency")
        adj_ms["feature"] = adj_ms["gap"].map(lambda g: f"gap_{str(g).replace('-', '_to_')}")

        adj_ms["target_mean_std"] = [
            _mean_std_str(m, s) for m, s in zip(adj_ms["ref_mean"], adj_ms["ref_std"])
        ]
        adj_ms["pred_mean_std"] = [
            _mean_std_str(m, s) for m, s in zip(adj_ms["pred_mean"], adj_ms["pred_std"])
        ]
        adj_ms["diff_mean_std"] = [
            _mean_std_str(m, s) for m, s in zip(adj_ms["diff_mean"], adj_ms["diff_std"])
        ]

        adj_ms["pearson_r_fmt"] = adj_ms["pearson_r"].map(_fmt_num)
        adj_ms["spearman_r_fmt"] = adj_ms["spearman_r"].map(_fmt_num)

        adj_ms = adj_ms[[
            "stage",
            "feature_type",
            "feature",
            "n_valid",
            "target_mean_std",
            "pred_mean_std",
            "pearson_r_fmt",
            "spearman_r_fmt",
            "diff_mean_std",
            "allowed_mean",
            "allowed_std",
            "violation_mean",
            "violation_std",
            "violation_fraction",
        ]]

    combined = pd.concat([ctx_ms, adj_ms], ignore_index=True, sort=False)
    return ctx_ms, adj_ms, combined



def summarize_ctx_agreement(records, ctx_names=None):
    ctx_names = list(ctx_names or CTX_NAMES)

    rows = []
    for s in records:
        if isinstance(s, dict):
            rows.append(s)
            continue

        ctx_ref = getattr(s, "ctx_ref", None)
        ctx_pred = getattr(s, "ctx_pred", None)
        if ctx_ref is None or ctx_pred is None:
            continue

        n = min(len(ctx_ref), len(ctx_pred), len(ctx_names))
        for d in range(n):
            rows.append({
                "stage": "eval",
                "ctx_dim": d,
                "ctx_name": ctx_names[d],
                "ctx_ref": ctx_ref[d],
                "ctx_pred": ctx_pred[d],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    if "stage" not in df.columns:
        df["stage"] = "eval"

    if "ctx_name" not in df.columns and "ctx_dim" in df.columns:
        df["ctx_name"] = df["ctx_dim"].map(
            lambda d: ctx_names[int(d)]
            if pd.notna(d) and int(d) < len(ctx_names)
            else f"ctx_{d}"
        )

    return _summarize_pairwise(
        df,
        group_cols=["stage", "ctx_dim", "ctx_name"],
        ref_col="ctx_ref",
        pred_col="ctx_pred",
    )

def summarize_adjacency_agreement(records, max_gaps=None):
    rows = []
    for s in records:
        if isinstance(s, dict):
            rows.append(s)
            continue

        tgt = getattr(s, "adj_target", None)
        pred = getattr(s, "adj_pred", None)
        allowed = getattr(s, "adj_allowed", None)
        conf = getattr(s, "adj_confidence", None)

        if tgt is None or pred is None:
            continue

        n = min(len(tgt), len(pred))
        
        gap_bins = getattr(s, "adj_gap_bins", None)

        for g in range(n):
            if gap_bins is not None and g < len(gap_bins):
                lo, hi = gap_bins[g]
                gap_label = str(lo) if int(lo) == int(hi) else f"{int(lo)}-{int(hi)}"
            else:
                gap_label = str(g + 1)
            
            rows.append({
                "stage": "eval",
                "gap_idx": g,
                "gap": gap_label,
                "adj_target": tgt[g],
                "adj_pred": pred[g],
                "adj_allowed": allowed[g] if allowed is not None and g < len(allowed) else np.nan,
                "adj_confidence": conf[g] if conf is not None and g < len(conf) else np.nan,
                "adj_error": pred[g] - tgt[g],
                "adj_violation": max(0.0, pred[g] - allowed[g]) if allowed is not None and g < len(allowed) else np.nan,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    if "stage" not in df.columns:
        df["stage"] = "eval"
        
    if max_gaps is not None and "gap_idx" in df.columns:
        df = df[pd.to_numeric(df["gap_idx"], errors="coerce") < int(max_gaps)]


    if df.empty:
        return pd.DataFrame()

    return _summarize_pairwise(
        df,
        group_cols=["stage", "gap_idx", "gap"],
        ref_col="adj_target",
        pred_col="adj_pred",
        allowed_col="adj_allowed",
        violation_col="adj_violation",
    )

def save_agreement_csv(rows, out_dir, name):
    out_dir = _ensure_dir(out_dir)
    path = os.path.join(out_dir, name)

    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return path

    rows = list(rows)
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return path

    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def export_viz_quant_tables(eval_roots, out_dir, stage_names=None, ctx_names=None):
    """
    Main entry point.

    Example:
      export_viz_quant_tables(
          eval_roots=["reports/reconstruction_stage2a"],
          stage_names=["stage2a"],
          out_dir="reports/reconstruction_stage2a/quant_tables_ctx_adj",
      )
    """

    out_dir = _ensure_dir(out_dir)

    samples_df, ctx_df, adj_df = build_viz_quant_raw_tables(
        eval_roots=eval_roots,
        stage_names=stage_names,
        ctx_names=ctx_names,
    )

    samples_df.to_csv(os.path.join(out_dir, "00_samples.csv"), index=False)
    ctx_df.to_csv(os.path.join(out_dir, "01_ctx_raw_points.csv"), index=False)
    adj_df.to_csv(os.path.join(out_dir, "02_adj_raw_points.csv"), index=False)

    if ctx_df.empty:
        print("[quant-tables] No context rows found.")
        ctx_summary = pd.DataFrame()
    else:
        ctx_summary = _summarize_pairwise(
            ctx_df,
            group_cols=["stage", "ctx_dim", "ctx_name"],
            ref_col="ctx_ref",
            pred_col="ctx_pred",
        )
        ctx_summary.to_csv(os.path.join(out_dir, "03_ctx_summary_long.csv"), index=False)

    if adj_df.empty:
        print("[quant-tables] No adjacency rows found.")
        adj_summary = pd.DataFrame()
    else:
        adj_summary = _summarize_pairwise(
            adj_df,
            group_cols=["stage", "gap_idx", "gap"],
            ref_col="adj_target",
            pred_col="adj_pred",
            allowed_col="adj_allowed",
            violation_col="adj_violation",
        )
        adj_summary.to_csv(os.path.join(out_dir, "04_adj_summary_long.csv"), index=False)

    ctx_ms, adj_ms, combined = make_viz_quant_manuscript_tables(ctx_summary, adj_summary)

    ctx_ms.to_csv(os.path.join(out_dir, "05_ctx_summary_manuscript.csv"), index=False)
    adj_ms.to_csv(os.path.join(out_dir, "06_adj_summary_manuscript.csv"), index=False)
    combined.to_csv(os.path.join(out_dir, "07_combined_manuscript_table.csv"), index=False)

    md_path = os.path.join(out_dir, "08_combined_manuscript_table.md")
    with open(md_path, "w") as f:
        if combined.empty:
            f.write("No valid context or adjacency rows found.\n")
        else:
            f.write(combined.to_markdown(index=False))
            f.write("\n")

    print(f"[quant-tables] Saved tables to: {out_dir}")

    return {
        "samples": samples_df,
        "ctx_raw": ctx_df,
        "adj_raw": adj_df,
        "ctx_summary": ctx_summary,
        "adj_summary": adj_summary,
        "ctx_manuscript": ctx_ms,
        "adj_manuscript": adj_ms,
        "combined_manuscript": combined,
    }
