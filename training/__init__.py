#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:08:19 2026

@author: derik
"""

"""Training entry points."""

from .train_vqvae import fit_vqvae
from .train_prior import train_motif_prior_mgit
from .stage4_activity import (
    configure_stage4c_event_calibration,
    evaluate_true_stage4_generation,
    train_maskgit_activity_prior,
    train_activity_prior_with_frozen_motif,
)
from .eval_vqvae import evaluate_vqvae
from .train_spatial_prior import fit_spatial_prior_pretrain
from .build_context_prior import build_context_prior, load_context_prior
from .baselines import (
    build_motif_null_baselines,
    load_motif_null_baselines,
    motif_null_predictions,
    build_null_baselines,
    load_null_baselines,
    predict_count_from_density,
)

__all__ = [
    "fit_spatial_prior_pretrain",
    "fit_vqvae",
    "train_motif_prior_mgit",
    "train_maskgit_activity_prior",
    "train_activity_prior_with_frozen_motif",
    "configure_stage4c_event_calibration",
    "evaluate_true_stage4_generation",
    "evaluate_vqvae",
    "build_context_prior",
    "load_context_prior",
    "build_null_baselines",
    "build_motif_null_baselines",
    "load_motif_null_baselines",
    "motif_null_predictions",
    "load_null_baselines",
    "predict_count_from_density",
]
