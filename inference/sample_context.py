#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:49:37 2026

@author: derik
"""


import pickle
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float]]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_2d_float(x: ArrayLike) -> np.ndarray:
    x = _to_numpy(x).astype(np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")
    return x


def _normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None, :]) / (std[None, :] + 1e-8)).astype(np.float32)


def _weighted_choice(rng: np.random.Generator, idx: np.ndarray, scores: np.ndarray, temperature: float):
    if len(idx) == 1:
        return int(idx[0])

    scores = scores.astype(np.float64)
    scores = scores - scores.min()

    if temperature <= 0:
        return int(idx[np.argmin(scores)])

    w = np.exp(-scores / float(temperature))
    w = w / (w.sum() + 1e-12)
    return int(rng.choice(idx, p=w))


class ContextBankSampler:
    """
    Empirical context sampler for Stage-3 generative inference.

    It does NOT learn p(context). It retrieves realistic contexts from the
    saved context bank built by build_context_prior.py.

    Supports:
      1. unconditional sampling
      2. sampling local_ctx given global_ctx / assay_id
      3. sampling full local_ctx given partial local constraints
    """

    def __init__(
        self,
        artifact: Dict[str, Any],
        *,
        model=None,
        device: Optional[Union[str, torch.device]] = None,
        seed: int = 0,
    ):
        self.artifact = artifact
        self.model = model
        self.device = torch.device(device) if device is not None else None
        self.rng = np.random.default_rng(seed)

        self.raw_g = artifact["raw_global_ctx"].astype(np.float32)
        self.raw_l = artifact["raw_local_ctx"].astype(np.float32)
        self.assay_ids = artifact["assay_ids"].astype(np.int64)

        self.global_emb_norm = artifact.get("global_emb_norm", None)
        self.local_emb_norm = artifact.get("local_emb_norm", None)

        self.global_emb_mean = artifact.get("global_emb_mean", None)
        self.global_emb_std = artifact.get("global_emb_std", None)
        self.local_emb_mean = artifact.get("local_emb_mean", None)
        self.local_emb_std = artifact.get("local_emb_std", None)

        self.local_min = artifact.get("local_feature_min", self.raw_l.min(axis=0)).astype(np.float32)
        self.local_max = artifact.get("local_feature_max", self.raw_l.max(axis=0)).astype(np.float32)
        self.local_mean = artifact.get("local_feature_mean", self.raw_l.mean(axis=0)).astype(np.float32)
        self.local_std = artifact.get("local_feature_std", self.raw_l.std(axis=0)).astype(np.float32)
        self.local_std = np.where(self.local_std < 1e-8, 1.0, self.local_std).astype(np.float32)

        self.feature_names = artifact.get("feature_names", None)

        self.N = len(self.assay_ids)
        if len(self.raw_g) != self.N or len(self.raw_l) != self.N:
            raise ValueError("Context artifact has inconsistent array lengths.")

    @classmethod
    def from_file(
        cls,
        path: str,
        *,
        model=None,
        device: Optional[Union[str, torch.device]] = None,
        seed: int = 0,
    ):
        with open(path, "rb") as f:
            artifact = pickle.load(f)
        return cls(artifact, model=model, device=device, seed=seed)

    def _candidate_indices(self, assay_id: Optional[int] = None) -> np.ndarray:
        if assay_id is None:
            return np.arange(self.N, dtype=np.int64)
        return np.where(self.assay_ids == int(assay_id))[0].astype(np.int64)

    @torch.no_grad()
    def _embed_global_query(self, global_ctx: ArrayLike) -> np.ndarray:
        g = _as_2d_float(global_ctx)

        if self.model is None:
            # Raw-space fallback.
            mean = self.raw_g.mean(axis=0)
            std = self.raw_g.std(axis=0)
            std = np.where(std < 1e-8, 1.0, std)
            return _normalize(g, mean, std)

        self.model.eval()
        dev = self.device or next(self.model.parameters()).device
        gt = torch.from_numpy(g).float().to(dev)
        ge = self.model.global_embedder(gt).detach().cpu().numpy().astype(np.float32)

        if self.global_emb_mean is not None and self.global_emb_std is not None:
            ge = _normalize(ge, self.global_emb_mean, self.global_emb_std)

        return ge.astype(np.float32)

    def _global_distance(self, global_ctx: ArrayLike, idx: np.ndarray) -> np.ndarray:
        q = self._embed_global_query(global_ctx)

        if self.model is None or self.global_emb_norm is None:
            bank = self.raw_g[idx]
            mean = self.raw_g.mean(axis=0)
            std = self.raw_g.std(axis=0)
            std = np.where(std < 1e-8, 1.0, std)
            bank = _normalize(bank, mean, std)
        else:
            bank = self.global_emb_norm[idx]

        d = ((bank[None, :, :] - q[:, None, :]) ** 2).sum(axis=-1)
        return d.min(axis=0).astype(np.float32)
                
    def _partial_local_distance(
        self,
        partial_local: Optional[
            Union[Dict[Union[int, str], float], ArrayLike]
        ],
        idx: np.ndarray,
    ) -> np.ndarray:
        if partial_local is None:
            return np.zeros(
                len(idx),
                dtype=np.float32,
            )

        n_features = int(self.raw_l.shape[1])

        if n_features != 9:
            raise ValueError(
                f"Context bank local dimension must be 9, "
                f"got {n_features}"
            )

        if isinstance(partial_local, dict):
            target = np.full(
                (n_features,),
                np.nan,
                dtype=np.float32,
            )

            for key, value in partial_local.items():
                if isinstance(key, str):
                    if self.feature_names is None:
                        raise ValueError(
                            "Named partial-local fields require a "
                            "context bank rebuilt with feature_names."
                        )

                    if key not in self.feature_names:
                        raise KeyError(
                            f"Unknown local feature name {key!r}. "
                            f"Valid names: {self.feature_names}"
                        )

                    j = self.feature_names.index(key)
                else:
                    j = int(key)

                    if j < 0 or j >= n_features:
                        raise IndexError(
                            f"Local feature index {j} is outside "
                            f"[0, {n_features - 1}]"
                        )

                value = float(value)

                if not np.isfinite(value):
                    raise ValueError(
                        f"Partial-local value for {key!r} "
                        f"must be finite, got {value}"
                    )

                target[j] = value

        else:
            target = _to_numpy(
                partial_local
            ).astype(
                np.float32
            ).reshape(-1)

            if len(target) != n_features:
                raise ValueError(
                    f"partial_local vector must have length "
                    f"{n_features}, got {len(target)}"
                )

        known = np.isfinite(target)

        if not known.any():
            raise ValueError(
                "partial_local does not contain any finite constraints."
            )

        known_idx = np.flatnonzero(known)

        below = target[known] < self.local_min[known]
        above = target[known] > self.local_max[known]
        outside = below | above

        if outside.any():
            if self.feature_names is None:
                names = [
                    f"local_ctx[{j}]"
                    for j in range(n_features)
                ]
            else:
                names = list(self.feature_names)

            messages = []

            for j in known_idx[outside]:
                messages.append(
                    f"{names[j]}={target[j]:.6g} is outside "
                    f"[{self.local_min[j]:.6g}, "
                    f"{self.local_max[j]:.6g}]"
                )

            raise ValueError(
                "Partial-local request is outside the "
                "observed context-bank range: "
                + "; ".join(messages)
            )

        scale = np.maximum(
            self.local_std[known],
            1e-8,
        )

        diff = (
            self.raw_l[idx][:, known]
            - target[None, known]
        ) / scale[None, :]

        return (
            diff ** 2
        ).mean(
            axis=1
        ).astype(
            np.float32
        )


    def sample_unconditional(
        self,
        *,
        assay_id: Optional[int] = None,
        batch_size: int = 1,
        replace: bool = True,
        return_index: bool = False,
    ):
        idx_pool = self._candidate_indices(assay_id)
        if len(idx_pool) == 0:
            raise ValueError(f"No contexts found for assay_id={assay_id}")

        chosen = self.rng.choice(idx_pool, size=batch_size, replace=replace)
        return self._pack(chosen, return_index=return_index)

    def sample_given_global(
        self,
        global_ctx: ArrayLike,
        *,
        assay_id: Optional[int] = None,
        k: int = 64,
        temperature: float = 0.05,
        batch_size: int = 1,
        return_index: bool = False,
    ):
        idx_pool = self._candidate_indices(assay_id)
        if len(idx_pool) == 0:
            raise ValueError(f"No contexts found for assay_id={assay_id}")

        d = self._global_distance(global_ctx, idx_pool)
        top = np.argsort(d)[: min(k, len(idx_pool))]
        top_idx = idx_pool[top]
        top_d = d[top]

        chosen = [
            _weighted_choice(self.rng, top_idx, top_d, temperature)
            for _ in range(batch_size)
        ]
        return self._pack(np.asarray(chosen, dtype=np.int64), return_index=return_index)

    def sample_given_partial_local(
        self,
        *,
        global_ctx: Optional[ArrayLike] = None,
        assay_id: Optional[int] = None,
        partial_local: Optional[Union[Dict[Union[int, str], float], ArrayLike]] = None,
        k: int = 64,
        global_weight: float = 1.0,
        local_weight: float = 1.0,
        temperature: float = 0.05,
        batch_size: int = 1,
        return_index: bool = False,
    ):
        idx_pool = self._candidate_indices(assay_id)
        if len(idx_pool) == 0:
            raise ValueError(f"No contexts found for assay_id={assay_id}")

        score = np.zeros(len(idx_pool), dtype=np.float32)

        if global_ctx is not None:
            score += float(global_weight) * self._global_distance(global_ctx, idx_pool)

        if partial_local is not None:
            score += float(local_weight) * self._partial_local_distance(partial_local, idx_pool)

        top = np.argsort(score)[: min(k, len(idx_pool))]
        top_idx = idx_pool[top]
        top_score = score[top]

        chosen = [
            _weighted_choice(self.rng, top_idx, top_score, temperature)
            for _ in range(batch_size)
        ]
        return self._pack(np.asarray(chosen, dtype=np.int64), return_index=return_index)

    def sample(
        self,
        *,
        global_ctx: Optional[ArrayLike] = None,
        assay_id: Optional[int] = None,
        partial_local: Optional[Union[Dict[Union[int, str], float], ArrayLike]] = None,
        batch_size: int = 1,
        k: int = 64,
        temperature: float = 0.05,
        return_index: bool = False,
    ):
        if global_ctx is None and partial_local is None:
            return self.sample_unconditional(
                assay_id=assay_id,
                batch_size=batch_size,
                return_index=return_index,
            )

        if global_ctx is not None and partial_local is None:
            return self.sample_given_global(
                global_ctx,
                assay_id=assay_id,
                k=k,
                temperature=temperature,
                batch_size=batch_size,
                return_index=return_index,
            )

        return self.sample_given_partial_local(
            global_ctx=global_ctx,
            assay_id=assay_id,
            partial_local=partial_local,
            k=k,
            temperature=temperature,
            batch_size=batch_size,
            return_index=return_index,
        )

    def _pack(self, idx: np.ndarray, *, return_index: bool = False):
        out = {
            "global_ctx": self.raw_g[idx].astype(np.float32),
            "local_ctx": self.raw_l[idx].astype(np.float32),
            "assay_id": self.assay_ids[idx].astype(np.int64),
        }
        if return_index:
            out["index"] = idx.astype(np.int64)
        return out

    def to_torch(
        self,
        sample: Dict[str, np.ndarray],
        *,
        device: Union[str, torch.device],
        task_id: int = 1,
    ) -> Dict[str, torch.Tensor]:
        B = sample["global_ctx"].shape[0]
        return {
            "global_ctx": torch.from_numpy(sample["global_ctx"]).float().to(device),
            "local_ctx": torch.from_numpy(sample["local_ctx"]).float().to(device),
            "assay_id": torch.from_numpy(sample["assay_id"]).long().to(device),
            "task_id": torch.full((B,), int(task_id), dtype=torch.long, device=device),
        }