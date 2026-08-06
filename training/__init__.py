#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:08:19 2026

@author: derik
"""

"""Training entry points."""

from .train_vqvae import fit_vqvae
from .train_prior import train_motif_prior_mgit
from .stage3_activity import (
    configure_stage3c_event_calibration,
    evaluate_true_stage3_generation,
    train_activity_prior_detr,
    train_activity_prior_with_frozen_motif,
)
from .eval_vqvae import evaluate_vqvae
from .train_spatial_prior import fit_spatial_prior_pretrain
from .build_context_prior import build_context_prior, load_context_prior

__all__ = [
    "fit_spatial_prior_pretrain",
    "fit_vqvae",
    "train_motif_prior_mgit",
    "train_activity_prior_detr",
    "train_activity_prior_with_frozen_motif",
    "configure_stage3c_event_calibration",
    "evaluate_true_stage3_generation",
    "evaluate_vqvae",
    "build_context_prior",
    "load_context_prior",
]
