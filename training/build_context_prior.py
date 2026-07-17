#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:52:44 2026

@author: derik

Build context_prior.pkl from a dataloader.

Saves:
    {
        "version": 1,
        "assay_counts": dict[int, int],
        "global_bank": dict[int, np.ndarray],   # assay_id -> (Ng, G)
        "global_mean": dict[int, np.ndarray],   # assay_id -> (G,)
        "local_models": dict[int, model_dict],  # assay_id -> GMM/Gaussian fallback
        "local_feature_mean": np.ndarray,
        "local_feature_std": np.ndarray,
        "local_feature_min": np.ndarray,
        "local_feature_max": np.ndarray,
        "feature_names": list[str] | None,
    }

Expected batch keys:
    batch["global_ctx"] : (B, G)
    batch["local_ctx"]  : (B, L)
    batch["assay_idx"]  : (B,)

Fallback assay key:
    batch["global_ctx_ids"][:, 0]
"""

import os
import json
import pickle
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from ..utils.constants import (
    ACTIVITY_CTX_NAMES,
    ACTIVITY_CTX_DIM,
)

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _extract_assay(batch: Dict[str, Any], assay_key: str = "assay_idx") -> np.ndarray:
    if assay_key in batch:
        assay = _to_numpy(batch[assay_key]).reshape(-1)
        return assay.astype(np.int64)

    if "global_ctx_ids" in batch:
        ids = _to_numpy(batch["global_ctx_ids"])
        if ids.ndim != 2 or ids.shape[1] < 1:
            raise ValueError(f'Expected batch["global_ctx_ids"] shape (B, >=1), got {ids.shape}')
        return ids[:, 0].reshape(-1).astype(np.int64)

    raise KeyError(
        f'Could not find assay key "{assay_key}" or fallback "global_ctx_ids" in batch.'
    )


def collect_context_arrays(
    loader,
    *,
    global_key: str = "global_ctx",
    local_key: str = "local_ctx",
    assay_key: str = "assay_idx",
    max_batches: Optional[int] = None,
    num_passes: int = 1,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    global_list = []
    local_list = []
    assay_list = []
    
    seen_batches = 0

    for pass_idx in range(num_passes):
        if verbose:
            print(f"[context_prior] collection pass {pass_idx + 1}/{num_passes}")
    
        for bidx, batch in enumerate(loader):
            if max_batches is not None and bidx >= max_batches:
                break

    
            if global_key not in batch:
                raise KeyError(f'Batch missing "{global_key}"')
            if local_key not in batch:
                raise KeyError(f'Batch missing "{local_key}"')
    
            g = _to_numpy(batch[global_key]).astype(np.float32)
            l = _to_numpy(batch[local_key]).astype(np.float32)
            a = _extract_assay(batch, assay_key=assay_key)
    
            if g.ndim == 1:
                g = g[None, :]
            if l.ndim == 1:
                l = l[None, :]
    
            if g.ndim != 2:
                raise ValueError(f"global_ctx must be (B,G), got {g.shape}")
            if l.ndim != 2:
                raise ValueError(f"local_ctx must be (B,L), got {l.shape}")
    
            B = len(a)
            if len(g) != B or len(l) != B:
                raise ValueError(
                    f"Batch length mismatch: len(global)={len(g)}, "
                    f"len(local)={len(l)}, len(assay)={B}"
                )
    
            global_list.append(g)
            local_list.append(l)
            assay_list.append(a)
    
            seen_batches += 1

            if verbose and seen_batches % 50 == 0:
                print(f"[context_prior] collected {seen_batches} batches total")
                

    if not global_list:
        raise ValueError("No batches collected.")

    global_all = np.concatenate(global_list, axis=0).astype(np.float32)
    local_all = np.concatenate(local_list, axis=0).astype(np.float32)
    assay_all = np.concatenate(assay_list, axis=0).astype(np.int64)

    return global_all, local_all, assay_all


@torch.no_grad()
def embed_context_arrays(
    model,
    global_all: np.ndarray,
    local_all: np.ndarray,
    *,
    device,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    if model is None:
        raise ValueError("model is required to build embedding-space context prior.")

    model.eval()

    g_embs = []
    l_embs = []

    for i in range(0, len(global_all), batch_size):
        g = torch.from_numpy(global_all[i:i + batch_size]).float().to(device)
        l = torch.from_numpy(local_all[i:i + batch_size]).float().to(device)

        ge = model.global_embedder(g)
        le = model.local_embedder(l)

        g_embs.append(ge.detach().cpu().numpy().astype(np.float32))
        l_embs.append(le.detach().cpu().numpy().astype(np.float32))

    return (
        np.concatenate(g_embs, axis=0).astype(np.float32),
        np.concatenate(l_embs, axis=0).astype(np.float32),
    )


def normalize_bank(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0, ddof=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    z = ((x - mean) / std).astype(np.float32)
    return z, mean, std



def build_context_prior(
    loader,
    *,
    save_path: str,
    model,
    device,
    global_key: str = "global_ctx",
    local_key: str = "local_ctx",
    assay_key: str = "assay_idx",
    max_batches: Optional[int] = None,
    embed_batch_size: int = 512,
    feature_names: Optional[Sequence[str]] = None,
    verbose: bool = True,
    num_passes: int = 1,
) -> Dict[str, Any]:
    global_all, local_all, assay_all = collect_context_arrays(
        loader,
        global_key=global_key,
        local_key=local_key,
        assay_key=assay_key,
        max_batches=max_batches,
        verbose=verbose,
        num_passes=num_passes,
    )
    
    if feature_names is None:
        feature_names = ACTIVITY_CTX_NAMES
    
    feature_names = list(feature_names)
    
    if len(feature_names) != 9:
        raise ValueError(
            f"feature_names must contain 9 names, "
            f"got {len(feature_names)}"
        )

    global_emb_all, local_emb_all = embed_context_arrays(
        model,
        global_all,
        local_all,
        device=device,
        batch_size=embed_batch_size,
    )

    global_emb_norm, global_emb_mean, global_emb_std = normalize_bank(global_emb_all)
    local_emb_norm, local_emb_mean, local_emb_std = normalize_bank(local_emb_all)

    global_bank_by_assay: Dict[int, np.ndarray] = {}
    global_emb_bank_by_assay: Dict[int, np.ndarray] = {}
    index_by_assay: Dict[int, np.ndarray] = {}
    assay_counts: Dict[int, int] = {}

    for assay_id in sorted(np.unique(assay_all).tolist()):
        assay_id = int(assay_id)
        idx = np.where(assay_all == assay_id)[0].astype(np.int64)

        global_bank_by_assay[assay_id] = global_all[idx].astype(np.float32)
        global_emb_bank_by_assay[assay_id] = global_emb_norm[idx].astype(np.float32)
        index_by_assay[assay_id] = idx
        assay_counts[assay_id] = int(len(idx))

    artifact: Dict[str, Any] = {
        "version": 2,
        "mode": "embedding_retrieval",

        # Raw contexts to feed back into prior/VQVAE.
        "raw_global_ctx": global_all.astype(np.float32),
        "raw_local_ctx": local_all.astype(np.float32),
        "assay_ids": assay_all.astype(np.int64),

        # Frozen context-head embeddings for retrieval.
        "global_emb": global_emb_all.astype(np.float32),
        "local_emb": local_emb_all.astype(np.float32),
        "global_emb_norm": global_emb_norm.astype(np.float32),
        "local_emb_norm": local_emb_norm.astype(np.float32),
        "global_emb_mean": global_emb_mean.astype(np.float32),
        "global_emb_std": global_emb_std.astype(np.float32),
        "local_emb_mean": local_emb_mean.astype(np.float32),
        "local_emb_std": local_emb_std.astype(np.float32),

        # Assay-conditioned banks.
        "global_bank_by_assay": global_bank_by_assay,
        "global_emb_bank_by_assay": global_emb_bank_by_assay,
        "index_by_assay": index_by_assay,
        "assay_counts": assay_counts,

        # Raw local stats only for sanity checks/clipping, not for sampling.
        "local_feature_min": local_all.min(axis=0).astype(np.float32),
        "local_feature_max": local_all.max(axis=0).astype(np.float32),
        "local_feature_mean": local_all.mean(axis=0).astype(np.float32),
        "local_feature_std": local_all.std(axis=0, ddof=0).astype(np.float32),

        "feature_names": None if feature_names is None else list(feature_names),

        "config": {
            "uses_pretrained_context_heads": True,
            "global_embedder": "model.global_embedder",
            "local_embedder": "model.local_embedder",
            "embed_batch_size": int(embed_batch_size),
            "max_batches": max_batches,
            "num_passes": int(num_passes),
        },
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(artifact, f)

    summary = {
        "save_path": save_path,
        "version": 2,
        "mode": "embedding_retrieval",
        "n_samples": int(len(assay_all)),
        "n_assays": int(len(assay_counts)),
        "global_dim": int(global_all.shape[1]),
        "local_dim": int(local_all.shape[1]),
        "global_emb_dim": int(global_emb_all.shape[1]),
        "local_emb_dim": int(local_emb_all.shape[1]),
        "assay_counts": assay_counts,
        "num_passes": int(num_passes),
    }
    
    summary["retrieval_eval"] = evaluate_context_prior_retrieval(artifact)
    summary["random_baseline_eval"] = evaluate_random_context_baseline(artifact)

    json_path = os.path.splitext(save_path)[0] + "_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"[context_prior] saved: {save_path}")
        print(f"[context_prior] summary: {json_path}")
        print(json.dumps(summary, indent=2))

    return artifact


def load_context_prior(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        artifact = pickle.load(f)

    if not isinstance(artifact, dict):
        raise TypeError(f"Expected dict artifact, got {type(artifact)}")
    if artifact.get("version", None) not in (1, 2):
        raise ValueError(f"Unsupported context prior version: {artifact.get('version', None)}")

    return artifact



def evaluate_context_prior_retrieval(
    artifact: Dict[str, Any],
    *,
    k_values=(1, 5, 10, 32),
    n_queries: Optional[int] = None,
    random_state: int = 0,
) -> Dict[str, Any]:
    rng = np.random.default_rng(random_state)

    G = artifact["global_emb_norm"]
    L = artifact["raw_local_ctx"]
    A = artifact["assay_ids"]

    N = len(A)
    if n_queries is None or n_queries > N:
        qidx = np.arange(N)
    else:
        qidx = rng.choice(N, size=n_queries, replace=False)

    results = {}

    for k in k_values:
        same_assay_rates = []
        local_l2 = []
        local_l1 = []

        for i in qidx:
            d = ((G - G[i:i + 1]) ** 2).sum(axis=1)

            # exclude exact self
            d[i] = np.inf

            nn = np.argsort(d)[:k]

            same_assay_rates.append(np.mean(A[nn] == A[i]))

            diff = L[nn] - L[i:i + 1]
            local_l2.append(np.sqrt((diff ** 2).sum(axis=1)).mean())
            local_l1.append(np.abs(diff).mean())

        results[f"k{k}_same_assay_frac"] = float(np.mean(same_assay_rates))
        results[f"k{k}_local_l2"] = float(np.mean(local_l2))
        results[f"k{k}_local_l1"] = float(np.mean(local_l1))

    return results


def evaluate_random_context_baseline(
    artifact: Dict[str, Any],
    *,
    k_values=(1, 5, 10, 32),
    n_queries: Optional[int] = None,
    random_state: int = 0,
) -> Dict[str, Any]:
    rng = np.random.default_rng(random_state)

    L = artifact["raw_local_ctx"]
    A = artifact["assay_ids"]

    N = len(A)
    if n_queries is None or n_queries > N:
        qidx = np.arange(N)
    else:
        qidx = rng.choice(N, size=n_queries, replace=False)

    results = {}

    for k in k_values:
        same_assay_rates = []
        local_l2 = []
        local_l1 = []

        for i in qidx:
            candidates = np.delete(np.arange(N), i)
            nn = rng.choice(candidates, size=min(k, len(candidates)), replace=False)

            same_assay_rates.append(np.mean(A[nn] == A[i]))

            diff = L[nn] - L[i:i + 1]
            local_l2.append(np.sqrt((diff ** 2).sum(axis=1)).mean())
            local_l1.append(np.abs(diff).mean())

        results[f"k{k}_same_assay_frac"] = float(np.mean(same_assay_rates))
        results[f"k{k}_local_l2"] = float(np.mean(local_l2))
        results[f"k{k}_local_l1"] = float(np.mean(local_l1))

    return results