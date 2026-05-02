#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:58:42 2026

@author: derik
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from sklearn.mixture import GaussianMixture


@dataclass
class _AssayPrior:
    kind: str                             # "gmm" or "gaussian"
    mean: np.ndarray                      # standardized-space mean, shape (D,)
    cov: np.ndarray                       # standardized-space cov, shape (D,D)
    model: Optional[GaussianMixture] = None


class LCTPrior:
    """
    Simple assay-conditioned prior p(LCT | assay_id).

    Fits one prior per assay:
      - GMM if enough samples are available
      - otherwise a single full-covariance Gaussian fallback

    All fitting is done in globally standardized feature space.
    """

    def __init__(
        self,
        priors: Dict[int, _AssayPrior],
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        feature_min: np.ndarray,
        feature_max: np.ndarray,
        assay_counts: Dict[int, int],
        feature_names: Optional[Sequence[str]] = None,
    ):
        self.priors = priors
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)
        self.feature_min = np.asarray(feature_min, dtype=np.float32)
        self.feature_max = np.asarray(feature_max, dtype=np.float32)
        self.assay_counts = {int(k): int(v) for k, v in assay_counts.items()}
        self.feature_names = None if feature_names is None else list(feature_names)

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.feature_mean) / self.feature_std

    def _destandardize(self, z: np.ndarray) -> np.ndarray:
        return z * self.feature_std + self.feature_mean

    def available_assays(self) -> list[int]:
        return sorted(self.priors.keys())

    def feature_dim(self) -> int:
        return int(self.feature_mean.shape[0])

    def summary(self) -> Dict[str, Any]:
        return {
            "n_assays": len(self.priors),
            "feature_dim": self.feature_dim(),
            "feature_names": self.feature_names,
            "assay_counts": dict(self.assay_counts),
            "prior_kind_by_assay": {
                int(assay_id): prior.kind
                for assay_id, prior in self.priors.items()
            },
        }

    def sample(
        self,
        assay_id: int,
        n_samples: int = 1,
        random_state: Optional[int] = None,
        clip_to_train_range: bool = True,
    ) -> np.ndarray:
        """
        Sample LCT vector(s) for a given assay.

        Returns:
            shape (D,) if n_samples == 1
            shape (n_samples, D) otherwise
        """
        assay_id = int(assay_id)
        if assay_id not in self.priors:
            raise KeyError(
                f"assay_id={assay_id} not found in fitted prior. "
                f"Available assays: {self.available_assays()}"
            )

        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        prior = self.priors[assay_id]
        rng = np.random.default_rng(random_state)

        if prior.kind == "gmm":
            # sklearn handles its own RNG through model.random_state,
            # so for repeatability per call we temporarily set it if requested.
            if random_state is not None:
                old_rs = prior.model.random_state
                prior.model.random_state = int(random_state)
                z, _ = prior.model.sample(n_samples=n_samples)
                prior.model.random_state = old_rs
            else:
                z, _ = prior.model.sample(n_samples=n_samples)

        elif prior.kind == "gaussian":
            z = rng.multivariate_normal(
                mean=prior.mean,
                cov=prior.cov,
                size=n_samples,
            )
        else:
            raise ValueError(f"Unknown prior kind: {prior.kind}")

        x = self._destandardize(np.asarray(z, dtype=np.float32))

        if clip_to_train_range:
            x = np.clip(x, self.feature_min, self.feature_max)

        if n_samples == 1:
            return x[0].astype(np.float32)
        return x.astype(np.float32)

    def sample_many_assays(
        self,
        assay_ids: Sequence[int],
        random_state: Optional[int] = None,
        clip_to_train_range: bool = True,
    ) -> np.ndarray:
        """
        Sample one LCT per assay_id in the provided list.

        Returns:
            array of shape (N, D)
        """
        rng = np.random.default_rng(random_state)
        out = []
        for assay_id in assay_ids:
            rs_i = None if random_state is None else int(rng.integers(0, 2**31 - 1))
            out.append(
                self.sample(
                    assay_id=int(assay_id),
                    n_samples=1,
                    random_state=rs_i,
                    clip_to_train_range=clip_to_train_range,
                )
            )
        return np.stack(out, axis=0).astype(np.float32)


def _safe_cov(X: np.ndarray, reg_covar: float) -> np.ndarray:
    """
    Full covariance with diagonal regularization.
    """
    X = np.asarray(X, dtype=np.float32)
    D = X.shape[1]

    if len(X) <= 1:
        return np.eye(D, dtype=np.float32) * float(reg_covar)

    cov = np.cov(X, rowvar=False)
    cov = np.asarray(cov, dtype=np.float32)

    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=np.float32)

    cov = cov + np.eye(D, dtype=np.float32) * float(reg_covar)
    return cov.astype(np.float32)


def fit_lct_priors(
    lct: np.ndarray,
    assay_ids: np.ndarray,
    *,
    n_components: int = 3,
    min_samples_for_gmm: int = 30,
    reg_covar: float = 1e-4,
    random_state: int = 0,
    feature_names: Optional[Sequence[str]] = None,
) -> LCTPrior:
    """
    Fit assay-conditioned priors p(LCT | assay_id).

    Args
    ----
    lct:
        array of shape (N, D)
    assay_ids:
        array of shape (N,)
    n_components:
        maximum number of GMM components per assay
    min_samples_for_gmm:
        if an assay has fewer samples than this, use a single Gaussian fallback
    reg_covar:
        covariance regularization
    random_state:
        random seed for sklearn GMM
    feature_names:
        optional names for LCT dimensions

    Returns
    -------
    Fitted LCTPrior
    """
    X = np.asarray(lct, dtype=np.float32)
    A = np.asarray(assay_ids).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"lct must have shape (N, D), got {X.shape}")
    if A.ndim != 1:
        raise ValueError(f"assay_ids must have shape (N,), got {A.shape}")
    if len(X) != len(A):
        raise ValueError(f"Length mismatch: len(lct)={len(X)} vs len(assay_ids)={len(A)}")
    if len(X) < 2:
        raise ValueError("Need at least 2 total samples to fit LCT priors.")

    N, D = X.shape
    if feature_names is not None and len(feature_names) != D:
        raise ValueError(
            f"feature_names length must match feature dim D={D}, got {len(feature_names)}"
        )

    # Global standardization stats
    feature_mean = X.mean(axis=0).astype(np.float32)
    feature_std = X.std(axis=0, ddof=0).astype(np.float32)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std).astype(np.float32)

    feature_min = X.min(axis=0).astype(np.float32)
    feature_max = X.max(axis=0).astype(np.float32)

    Z = ((X - feature_mean) / feature_std).astype(np.float32)

    priors: Dict[int, _AssayPrior] = {}
    assay_counts: Dict[int, int] = {}

    unique_assays = np.unique(A)
    for assay_id in unique_assays:
        assay_id = int(assay_id)
        Za = Z[A == assay_id]
        n_a = len(Za)
        assay_counts[assay_id] = int(n_a)

        if n_a == 0:
            continue

        # Degenerate case: only one point
        if n_a == 1:
            mean = Za[0].astype(np.float32)
            cov = np.eye(D, dtype=np.float32) * float(reg_covar)
            priors[assay_id] = _AssayPrior(
                kind="gaussian",
                mean=mean,
                cov=cov,
                model=None,
            )
            continue

        # Prefer GMM if enough samples exist
        if n_a >= min_samples_for_gmm and n_components >= 2:
            k = min(int(n_components), int(n_a))
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    reg_covar=float(reg_covar),
                    random_state=int(random_state),
                )
                gmm.fit(Za)

                mean = Za.mean(axis=0).astype(np.float32)
                cov = _safe_cov(Za, reg_covar)

                priors[assay_id] = _AssayPrior(
                    kind="gmm",
                    mean=mean,
                    cov=cov,
                    model=gmm,
                )
                continue

            except Exception:
                # Fall back to single Gaussian if GMM fit fails
                pass

        mean = Za.mean(axis=0).astype(np.float32)
        cov = _safe_cov(Za, reg_covar)
        priors[assay_id] = _AssayPrior(
            kind="gaussian",
            mean=mean,
            cov=cov,
            model=None,
        )

    return LCTPrior(
        priors=priors,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_min=feature_min,
        feature_max=feature_max,
        assay_counts=assay_counts,
        feature_names=feature_names,
    )


def collect_lct_arrays(
    loader,
    *,
    lct_key: str = "local_ctx",
    assay_key: str = "assay_idx",
    max_batches: Optional[int] = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect all (LCT, assay_idx) pairs from a dataloader.

    Expected batch keys:
      - batch[lct_key]   -> shape (B, D) or (D,)
      - batch[assay_key] -> shape (B,) or scalar

    Returns
    -------
    lct_all:
        shape (N, D)
    assay_idx_all:
        shape (N,)
    """
    lct_list = []
    assay_list = []

    for bidx, batch in enumerate(loader):
        if max_batches is not None and bidx >= max_batches:
            break

        if lct_key not in batch:
            raise KeyError(f"Batch does not contain key '{lct_key}'")
        if assay_key not in batch:
            raise KeyError(f"Batch does not contain key '{assay_key}'")

        lct = batch[lct_key]
        assay = batch[assay_key]

        if isinstance(lct, torch.Tensor):
            lct = lct.detach().cpu().numpy()
        else:
            lct = np.asarray(lct)

        if isinstance(assay, torch.Tensor):
            assay = assay.detach().cpu().numpy()
        else:
            assay = np.asarray(assay)

        lct = np.asarray(lct, dtype=np.float32)
        assay = np.asarray(assay).reshape(-1)

        if lct.ndim == 1:
            lct = lct[None, :]
        elif lct.ndim != 2:
            raise ValueError(f"Expected LCT batch shape (B,D) or (D,), got {lct.shape}")

        if len(lct) != len(assay):
            raise ValueError(
                f"Batch length mismatch: lct batch has {len(lct)} rows, "
                f"but assay batch has {len(assay)} rows"
            )

        lct_list.append(lct.astype(np.float32))
        assay_list.append(assay.astype(np.int64))

        if verbose and ((bidx + 1) % 50 == 0):
            print(f"[collect_lct_arrays] processed {bidx + 1} batches")

    if len(lct_list) == 0:
        raise ValueError("No batches collected. Check loader or max_batches.")

    lct_all = np.concatenate(lct_list, axis=0).astype(np.float32)
    assay_idx_all = np.concatenate(assay_list, axis=0).astype(np.int64)
    return lct_all, assay_idx_all


def fit_lct_priors_from_loader(
    loader,
    *,
    lct_key: str = "local_ctx",
    assay_key: str = "assay_idx",
    max_batches: Optional[int] = None,
    n_components: int = 3,
    min_samples_for_gmm: int = 30,
    reg_covar: float = 1e-4,
    random_state: int = 0,
    feature_names: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> LCTPrior:
    """
    One-call fit from dataloader.
    """
    lct_all, assay_idx_all = collect_lct_arrays(
        loader,
        lct_key=lct_key,
        assay_key=assay_key,
        max_batches=max_batches,
        verbose=verbose,
    )

    return fit_lct_priors(
        lct=lct_all,
        assay_ids=assay_idx_all,
        n_components=n_components,
        min_samples_for_gmm=min_samples_for_gmm,
        reg_covar=reg_covar,
        random_state=random_state,
        feature_names=feature_names,
    )


def save_lct_prior(prior: LCTPrior, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(prior, f)


def load_lct_prior(path: str) -> LCTPrior:
    with open(path, "rb") as f:
        prior = pickle.load(f)
    if not isinstance(prior, LCTPrior):
        raise TypeError(f"Object loaded from {path} is not an LCTPrior")
    return prior


def fit_and_save_lct_prior(
    loader,
    save_path: str,
    *,
    lct_key: str = "local_ctx",
    assay_key: str = "assay_idx",
    max_batches: Optional[int] = None,
    n_components: int = 3,
    min_samples_for_gmm: int = 30,
    reg_covar: float = 1e-4,
    random_state: int = 0,
    feature_names: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> LCTPrior:
    """
    Convenience helper:
      1) collect arrays from loader
      2) fit prior
      3) save pkl
      4) return fitted object
    """
    prior = fit_lct_priors_from_loader(
        loader,
        lct_key=lct_key,
        assay_key=assay_key,
        max_batches=max_batches,
        n_components=n_components,
        min_samples_for_gmm=min_samples_for_gmm,
        reg_covar=reg_covar,
        random_state=random_state,
        feature_names=feature_names,
        verbose=verbose,
    )
    save_lct_prior(prior, save_path)

    if verbose:
        print(f"[LCTPrior] saved to: {save_path}")
        print(f"[LCTPrior] summary: {prior.summary()}")

    return prior


def generate_lct(
    prior: LCTPrior,
    assay_id: int,
    *,
    random_state: Optional[int] = None,
    clip_to_train_range: bool = True,
) -> np.ndarray:
    """
    Convenience wrapper for generating one LCT vector.
    """
    return prior.sample(
        assay_id=assay_id,
        n_samples=1,
        random_state=random_state,
        clip_to_train_range=clip_to_train_range,
    )