import os
import sys

# Allow Spyder/runfile execution while keeping package-relative imports
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(pkg_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    __package__ = "MAGVIT_project"

# Limit common math backends to 1 thread each
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from glob import glob
import copy
import json
from pathlib import Path
from typing import Iterable, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    # Spyder/IPython can sometimes initialize threads before this file runs.
    pass

from .dataset import make_loaders_for_assays, burst_collate
from .model import TransformerVQVAE, DETRActivityPrior, MaskGITMotifPrior, HierarchicalCodebookPrior
from .model.prior import (
    build_activity_targets_from_codes,
    detr_activity_loss,
    infer_activity_coordinate_mode_from_state_dict,
)
from .model.spatial_map import GlobalContextSpatialBank, GlobalContextAdjacencyBank
from .training import (
    fit_vqvae, evaluate_vqvae, fit_spatial_prior_pretrain, build_context_prior,
    train_motif_prior_mgit,
    train_activity_prior_detr,
    train_activity_prior_with_frozen_motif,
    configure_stage3c_event_calibration,
)
from .training.train_prior import (
    _batch_to_device,
    _make_activity_in_from_codes,
    _vq_codes_alpha_and_pmask,
    _vq_codes_and_pmask,
    distance_neighborhood_ce_loss,
    expected_code_distance_loss,
)
from .inference import (
    ContextBankSampler,
    decode_codes_to_xgen,
    evaluate_generation_global_metrics,
    save_generated_batch_outputs,
    save_generation_metrics_json,
    sample_hierarchical_roi,
)
from .visualization import (
    make_model_videos_vqvae,
    run_plotter,
    plot_base_then_finetune,
    save_assaywise_spatial_maps,
    save_assaywise_adjacency_diagnostics,
)
from .visualization.codebook_diag import (
    plot_blank_active_tsne_l1,
    plot_blank_active_pca_l1,
)
from .visualization.reports_data import (
    export_base_finetune_flat_xlsx,
    export_viz_quant_tables,
)

from .utils.constants import (
    ACTIVITY_CTX_NAMES,
    DEFAULT_GAP_BINS,
    normalize_gap_bins,
    max_gap_from_bins,
)

# =============================================================================
# User configuration
# =============================================================================

# Run any subset sequentially. Examples:
#   (1,)       -> train context-agnostic VQVAE only
#   (2,)       -> load stage-1 ckpt, learn exact continuous hull decoding + context
#   (1, 2, 3)  -> run the whole pipeline sequentially
#   (0,)     -> run spatial-map pretraining only
TRAIN_STAGES = ()        # 0,1,2,3
EVAL_STAGES  = (3,)   # 0,1,2,3

# Stage-2 continuous quantization.
#
# Stage 2A:
#   project the frozen encoder residual onto the convex hull of the selected
#   z1 parent's frozen z2 children; train the decoder on that fixed geometry.
# Stage 2B:
#   train the optional alpha adapter to reproduce the same geometric projection.
#   The decoder is frozen and does not define the adapter target.
# Stage 2C:
#   freeze the continuous mapping and decoder backbone; train only zero-gated
#   decoder cross-attention and context projections.
STAGE2_PHASES = ("2a", "2c")

# Independently evaluate any saved Stage-2 substages. Evaluation order is
# always 2A -> 2B -> 2C, regardless of tuple order.
STAGE2_EVAL_PHASES = ("2a", "2c")

STAGE2_CONTINUOUS_RESIDUAL = True
STAGE2A_EPOCHS = 150
STAGE2B_EPOCHS = 50
STAGE2C_EPOCHS = 50

# Stage-3 training substages. Any subset of ("3a", "3b", "3c") is valid;
# execution order remains 3A -> 3B -> 3C. Missing prerequisites are loaded
# from their best checkpoints.
STAGE3_PHASES = ("3b", "3c")
STAGE3A_EPOCHS = 200
STAGE3B_EPOCHS = 200
STAGE3C_EPOCHS = 100

# Stage 3B/3C activity-coordinate parameterization. Use "factorized" for the
# original axis-head ablation or "joint_dense" for one categorical THW head.
STAGE3_COORDINATE_MODE = "joint_dense"

# Stage 3B is selected by a deterministic, hard expected-count top-K metric.
STAGE3B_HYPERPARAMETERS = {
    "lr": 2e-4,
    "lambda_count": 1.0,
    "lambda_count_neighbor": 0.25,
    "lambda_count_distance": 0.05,
    "lambda_obj": 1.0,
    "lambda_coord": 1.0,
    "lambda_soft_count": 0.10,
    "lambda_soft_grid": 1.0,
    "lambda_dup": 0.10,
    "no_object_weight": 0.10,
    "count_teacher_epochs": 15,
    "count_transition_epochs": 45,
    "deterministic_validation_masks": True,
    "hard_activity_mode": "expected-count unique grid top-K",
}

# Stage 3C is a low-LR event-placement calibration, not a second activity-prior
# training stage. Its true-generation validation is deliberately limited to a
# fixed subset because every validation pass runs iterative MaskGIT + decoding.
STAGE3C_HYPERPARAMETERS = {
    "lr": 1e-5,
    "lambda_detr": 1.0,
    "lambda_count": 1.0,
    "lambda_count_neighbor": 0.25,
    "lambda_count_distance": 0.05,
    "lambda_obj": 1.0,
    "lambda_coord": 1.0,
    "lambda_soft_count": 0.10,
    "lambda_soft_grid": 1.0,
    "lambda_dup": 0.10,
    "no_object_weight": 0.10,
    "lambda_ctx": 0.25,
    "lambda_ctx_field": 0.05,
    "lambda_adj": 0.25,
    "lambda_spatial": 0.25,
    "auxiliary_ramp_epochs": 10,
    "generation_val_max_batches": 4,
    "generation_motif_steps": 12,
    "deterministic_validation_masks": True,
    "hard_activity_mode": "expected-count unique grid top-K straight-through",
}

# Reuse saved Kmax metadata when possible so independently rerun substages use
# the same activity-head shape. Set True only when the data/token grid changed.
STAGE3_RECOMPUTE_KMAX = False
STAGE3_KMAX_PASSES = 5
STAGE3_KMAX_MARGIN = 1.25

# Independently evaluate any saved Stage-3 substages. Evaluation order is
# always 3A -> 3B -> 3C, regardless of tuple order.
#
# 3A: held-out teacher-forced activity / masked motif prediction metrics.
# 3B: held-out activity-count and coordinate metrics.
# 3C: held-out refined activity metrics.
STAGE3_EVAL_PHASES = ("3c",)

# Stable Stage-3A evaluation starts from a fully masked motif ROI. Set this to
# 0.15 to reproduce the mixed masking regime used during training validation.
STAGE3A_EVAL_FULL_MASK_PROB = 1.0

# For selected 3B/3C evaluation phases, also run the expensive decoded
# generation evaluations. Set False to evaluate only the prior heads.
RUN_STAGE3_GENERATION_EVAL = True

RUN_EVAL = True
RUN_VIZ  = True
RUN_VIDEO_GEN = True
RUN_PLOTTER = True
RUN_CODEBOOK_DEBUG = True
RUN_SKIP_MISSING_EVAL = True

# Stage-0 spatial map checkpoint. If this file exists, it will be loaded before
# stages 1/2/3. If TRAIN_STAGES contains 0.5, it will be overwritten/trained first.
SPATIAL_CKPT = Path("ckpts/spatial_bias_pretrain.pt")

CKPT_DIR = Path("ckpts")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CKPTS = {
    "stage1_best": CKPT_DIR / "vqvae_stage1_best.pt",
    "stage1_last": CKPT_DIR / "vqvae_stage1_last.pt",
    "stage2a_best": CKPT_DIR / "vqvae_stage2a_convex_best.pt",
    "stage2a_last": CKPT_DIR / "vqvae_stage2a_convex_last.pt",
    "stage2b_best": CKPT_DIR / "vqvae_stage2b_projector_best.pt",
    "stage2b_last": CKPT_DIR / "vqvae_stage2b_projector_last.pt",
    "stage2_best": CKPT_DIR / "vqvae_stage2_convex_best.pt",
    "stage2_last": CKPT_DIR / "vqvae_stage2_convex_last.pt",
    
    "motif_prior_best": CKPT_DIR / "motif_prior_best.pt",
    # Legacy alias retained for external scripts. New Stage 3B runs write this
    # alias from the generation-aligned hard-metric checkpoint.
    "activity_prior_best": CKPT_DIR / "activity_prior_best.pt",
    "activity_prior_best_loss": CKPT_DIR / "activity_prior_best_loss.pt",
    "activity_prior_best_hard_metric": CKPT_DIR / "activity_prior_best_hard_metric.pt",
    "activity_prior_refined_best": CKPT_DIR / "activity_prior_refined_best.pt",
}

REPORTS = {
    "stage1": Path("reports/training_report_vqvae_stage1.json"),
    "stage2a": Path("reports/training_report_vqvae_stage2a_convex.json"),
    "stage2b": Path("reports/training_report_vqvae_stage2b_projector.json"),
    "stage2": Path("reports/training_report_vqvae_stage2c_convex.json"),
    "stage2a_eval": Path("reports/evaluation_report_vqvae_stage2a_convex.json"),
    "stage2b_eval": Path("reports/evaluation_report_vqvae_stage2b_projector.json"),
    "stage2c_eval": Path("reports/evaluation_report_vqvae_stage2c_convex.json"),

    "prior_motif": Path("reports/training_report_prior_3A_motif.json"),
    "prior_motif_eval": Path("reports/evaluation_report_prior_3A_motif.json"),
    "prior_activity_eval": Path("reports/evaluation_report_prior_3B_activity.json"),
    "prior_refine_eval": Path("reports/evaluation_report_prior_3C_refine.json"),
    "prior_activity": Path("reports/training_report_prior_3B_activity.json"),
    "prior_refine": Path("reports/training_report_prior_3C_refine.json"),
}

VIZ_ROOTS = {
    1: Path("../viz_out_vqvae/vqvae_stage1"),
    2: Path("../viz_out_vqvae/vqvae_stage2"),
}

STAGE2_VIZ_ROOTS = {
    "2a": Path("../viz_out_vqvae/vqvae_stage2/stage2a"),
    "2b": Path("../viz_out_vqvae/vqvae_stage2/stage2b"),
    "2c": Path("../viz_out_vqvae/vqvae_stage2/stage2c"),
}

# Data
patch_size = (6, 15, 14)
temporal_crop = 6000
temporal_pool = 120
batch_size = 4
grad_accum_steps = 8
num_workers = 2
per_assay_quota_stage12 = 30

cache_dir = "../_cache_spike_thw_run1"
cache_mode = "uint8"
cache_max_gb = 80
cache_write_prob = 1.0

# Model
num_codes = (32, 8)



num_assays_for_emb = 1000
dim_assay_for_emb = 64
max_viz_samples = 1000

# ISI adjacency firing constraint
gap_bins = list(
    normalize_gap_bins(
        DEFAULT_GAP_BINS
    )
)

# Tolerance for spike location in a voxel (for training loss and val metrics)
recon_tolerance = (1, 1, 1)
metric_tolerance = (1, 1, 1)

# Spatial map usage after stage 0.5.
# True means the pretrained assay-specific spatial suppressive bias remains active.
# False means stages 1/2 do not use the spatial bias at all.
USE_GCT_PRETRAIN_MODULE = True


# =============================================================================
# Helpers
# =============================================================================

def find_assays() -> dict:
    assay_paths = glob("../output_data/*/*/*_ecephys")
    assay_dict = {}

    for idx, assay_path in enumerate(assay_paths):
        assay_name = os.path.basename(assay_path)
        npz_files = glob(os.path.join(assay_path, "*/binary_unit_burst_*.npz"))
        if len(npz_files) > 0:
            assay_dict[idx] = {"assay_name": assay_name, "files": npz_files}

    for k, v in assay_dict.items():
        print(f"{k}: {v['assay_name']} -> {len(v['files'])} files")

    if len(assay_dict) == 0:
        raise RuntimeError("No assay folders with binary_unit_burst_*.npz files were found.")
    return assay_dict


def make_loaders(assay_dict: dict, assay_indices: list[int], per_assay_quota: int):
    return make_loaders_for_assays(
        assay_indices=assay_indices,
        assay_dict=assay_dict,
        batch_size=batch_size,
        temporal_crop=temporal_crop,
        temporal_pool=temporal_pool,
        spatial_crop=None,
        val_frac=0.2,
        test_frac=0.3,
        seed=42,
        task_probs={"recon": 0.25, "causal": 0.25, "noncausal": 0.25, "spatial": 0.25},
        patch_size=patch_size,
        n_assays=num_assays_for_emb,
        dim_assays=dim_assay_for_emb,
        per_assay_quota=per_assay_quota,
        num_workers=num_workers,
        cache_dir=cache_dir,
        cache_mode=cache_mode,
        cache_max_gb=cache_max_gb,
        cache_write_prob=cache_write_prob,
    )


@torch.no_grad()
def compute_p0_from_loader(loader, max_batches: Optional[int] = 100) -> float:
    total_ones = 0.0
    total_voxels = 0.0
    for i, batch in enumerate(loader):
        x = batch["x"].float()
        total_ones += x.sum().item()
        total_voxels += x.numel()
        if max_batches is not None and (i + 1) >= max_batches:
            break
    return float(total_ones / max(total_voxels, 1.0))

@torch.no_grad()
def compute_short_gap_target_rates_from_loader(
    loader,
    gap_bins = None,
    max_batches: Optional[int] = None,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes conditional same-site short-gap rates after pooling/cropping.

    target_rates[g-1] =
        P(x[t+g,h,w]=1 | x[t,h,w]=1)

    This is much better scaled than voxel-pair probability.
    """
    if gap_bins is None:
        gap_bins = [(1, 1), (2, 2), (3, 3)]
    gap_bins = [(int(a), int(b)) for a, b in gap_bins]
    
    nums = torch.zeros(len(gap_bins), dtype=torch.float64)
    dens = torch.zeros(len(gap_bins), dtype=torch.float64)

    for i, batch in enumerate(loader):
        x = batch["x"].float()

        # Accept (B,1,T,H,W) or (B,T,H,W)
        if x.ndim == 5:
            assert x.shape[1] == 1, f"Expected (B,1,T,H,W), got {tuple(x.shape)}"
            x = x[:, 0]
        elif x.ndim != 4:
            raise ValueError(f"Unexpected x shape: {tuple(x.shape)}")

        x = (x > threshold).float()  # (B,T,H,W)

        for bi, (lo, hi) in enumerate(gap_bins):
            for g in range(lo, hi + 1):
                if x.shape[1] <= g:
                    continue
        
                x0 = x[:, :-g]
                xg = x[:, g:]
        
                nums[bi] += (x0 * xg).sum().double().cpu()
                dens[bi] += x0.sum().double().cpu()

        if max_batches is not None and (i + 1) >= max_batches:
            break

    rates = nums / dens.clamp(min=eps)
    return rates.float()



def make_vqvae(img_size, device: str, *, full_spatial_size=None, use_decoder_cross_attn: bool, decoder_cross_attn_layers: tuple[int, ...]):
    model = TransformerVQVAE(
        img_size=img_size,
        full_spatial_size=full_spatial_size,
        patch_size=patch_size,

        encoder_embed_dim=64,
        encoder_depth=2,
        encoder_num_heads=4,
        code_dim=64,
        num_codes=num_codes,
        decoder_embed_dim=64,
        decoder_depth=2,
        decoder_num_heads=4,
        in_chans=1,
        out_chans=1,

        # Context args
        local_ctx_in_dim=9,
        local_emb_dim=32,
        global_ctx_in_dim=dim_assay_for_emb,
        global_emb_dim=32,

        use_spatial_map_prior=USE_GCT_PRETRAIN_MODULE,
        gap_bins=gap_bins,

        enc_attn_mask_kind="none",
        dec_attn_mask_kind="temporal_causal",

        cfg_ctx_drop_p=0.0,
        use_decoder_cross_attn=use_decoder_cross_attn,
        decoder_cross_attn_layers=decoder_cross_attn_layers,
    ).to(device)

    # Keep norms in fp32 for stability.
    for m in model.modules():
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.float()

    return model


def set_requires_grad(module: Optional[nn.Module], flag: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = bool(flag)


def set_all_trainable(model: nn.Module, flag: bool):
    for p in model.parameters():
        p.requires_grad = bool(flag)


def set_decoder_cross_attention(model: nn.Module, enabled: bool, layers: Iterable[int] = (0,)):
    # Requires the small vqvae.py edit: use_decoder_cross_attn and decoder_cross_attn_layers.
    model.use_decoder_cross_attn = bool(enabled)
    model.decoder_cross_attn_layers = set(int(i) for i in layers) if enabled else set()
    print(f"decoder cross-attn enabled={enabled}, layers={sorted(model.decoder_cross_attn_layers)}")


def freeze_for_stage(model: nn.Module, stage: float):
    """
    Centralized stage policy.

    Stage 1: context-agnostic motif learning.
      - Train stem/encoder/to_code/VQ/decoder.
      - Freeze context embedders and spatial map.
      - Decoder cross-attn OFF.

    Stage 2A: context-free continuous decoder calibration.
      - Train the decoder on the exact geometric convex projection.
      - Keep decoder cross-attention and context branches disabled/frozen.
      - Keep encoder, to_code, and hierarchical EMA codebooks fixed.
      - Keep the optional amortized alpha adapter frozen.
      - Keep the pretrained global embedder and spatial map fixed.

    Stage 3: prior learning.
      - Freeze VQVAE completely.
    """
    set_all_trainable(model, False)

    if stage == 1:
        model.use_continuous_residual = False
        model.continuous_residual_sample_mix = 0.0
        set_decoder_cross_attention(model, enabled=False, layers=())
        for name in ["stem", "patch_embed", "sparse_encoder", "to_code", "vq", "code_to_dec",
                     "dec_blocks", "dec_norm", "patch_renderer"]:
            set_requires_grad(getattr(model, name, None), True)

        if hasattr(model, "activity_type_offset"):
            model.activity_type_offset.requires_grad = True

        # Context paths stay frozen in stage 1.
        set_requires_grad(getattr(model, "local_embedder", None), False)
        set_requires_grad(getattr(model, "global_embedder", None), False)
        set_requires_grad(getattr(model, "local_to_dec_ctx", None), False)
        set_requires_grad(getattr(model, "global_to_dec_ctx", None), False)
        set_requires_grad(getattr(model, "spatial_map_prior", None), False)
        
        model.vq.freeze_codebook_updates = False
        
        # EMA codebook entries are updated manually, not by AdamW.
        # Keep blank_token trainable; freeze only hierarchical codebook tensors.
        if hasattr(model, "vq") and hasattr(model.vq, "tree_embeds"):
            for p in model.vq.tree_embeds:
                p.requires_grad = False

    elif stage == 2:
        set_decoder_cross_attention(
            model,
            enabled=False,
            layers=(),
        )
    
        # Stage 2A decoder components. Context injection remains off.
        for name in [
            "dec_blocks",
            "dec_norm",
            "patch_renderer",
        ]:
            set_requires_grad(
                getattr(model, name, None),
                True,
            )
    
        model.use_continuous_residual = bool(STAGE2_CONTINUOUS_RESIDUAL)
        model.continuous_residual_sample_mix = 0.0
        model.continuous_residual_projector.use_alpha_adapter = False
        model.continuous_residual_projector.decode_with_projection_target = True

        # Decoder adaptation to continuous points includes the first linear
        # code-space interface.  The z1/z2 codebook geometry itself stays fixed.
        set_requires_grad(
            getattr(model, "code_to_dec", None),
            True,
        )

        set_requires_grad(
            getattr(model, "global_embedder", None),
            False,
        )
        set_requires_grad(
            getattr(model, "local_embedder", None),
            False,
        )
        set_requires_grad(
            getattr(model, "local_to_dec_ctx", None),
            False,
        )
        set_requires_grad(
            getattr(model, "global_to_dec_ctx", None),
            False,
        )
        set_requires_grad(
            getattr(model, "spatial_map_prior", None),
            False,
        )
    
        if hasattr(model, "activity_type_offset"):
            model.activity_type_offset.requires_grad = False
    
        # Freeze the complete Stage-1 codebook geometry.  Continuousness is
        # introduced between quantization and decoding, not by moving centroids.
        for p in model.vq.tree_embeds:
            p.requires_grad = False

        model.vq.freeze_codebook_updates = True
        model.vq.dead_code_restart_every = 0
        model.vq.duplicate_restart_every = 0

    elif stage == 3:
        # Stage 3 uses the corrected exact convex projection to produce alpha
        # training targets. The VQVAE remains completely frozen.
        model.use_continuous_residual = True
        model.continuous_residual_sample_mix = 0.0
        model.continuous_residual_projector.use_alpha_adapter = False
        model.continuous_residual_projector.decode_with_projection_target = True
        set_decoder_cross_attention(model, enabled=True, layers=(0,))
        set_all_trainable(model, False)
        model.eval()

    else:
        raise ValueError(f"Unsupported stage={stage}")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Stage {stage}: trainable params = {n_trainable:,} / {n_total:,}")


def freeze_for_stage2b(model: nn.Module):
    """
    Train only the optional amortized alpha projector against the fixed
    geometric convex projection.  The decoder and codebook geometry are frozen.
    """
    freeze_for_stage(model, 2)

    projector = model.continuous_residual_projector
    projector.use_alpha_adapter = True
    projector.decode_with_projection_target = True

    # The exact projection remains the training-time decoder input.  Only the
    # adapter learns to reproduce it through loss_cont_projection.
    set_requires_grad(model.code_to_dec, False)
    set_requires_grad(model.dec_blocks, False)
    set_requires_grad(model.dec_norm, False)
    set_requires_grad(model.patch_renderer, False)
    set_requires_grad(projector.alpha_adapter, True)
    set_requires_grad(projector.hull_scale_adapter, False)

    set_requires_grad(model.stem, False)
    set_requires_grad(model.patch_embed, False)
    set_requires_grad(model.sparse_encoder, False)
    set_requires_grad(model.to_code, False)
    for parameter in model.vq.parameters():
        parameter.requires_grad = False

    model.vq.freeze_codebook_updates = True
    model.vq.dead_code_restart_every = 0
    model.vq.duplicate_restart_every = 0

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Stage 2B: trainable params = {n_trainable:,} / {n_total:,}")


def freeze_for_stage2c(model: nn.Module):
    """
    Freeze encoder, codebooks, alpha projector, and decoder backbone.  Train
    only the context projections and zero-gated cross-attention at decoder
    layer 0.
    """
    freeze_for_stage(model, 2)

    projector = model.continuous_residual_projector
    projector.use_alpha_adapter = False
    projector.decode_with_projection_target = True
    set_requires_grad(projector.alpha_adapter, False)
    set_requires_grad(projector.hull_scale_adapter, False)

    set_decoder_cross_attention(model, enabled=True, layers=(0,))

    # Freeze the complete Stage-2A decoder, then reopen only the context branch.
    set_requires_grad(model.code_to_dec, False)
    set_requires_grad(model.dec_blocks, False)
    set_requires_grad(model.dec_norm, False)
    set_requires_grad(model.patch_renderer, False)

    set_requires_grad(model.local_embedder, True)
    set_requires_grad(model.local_to_dec_ctx, True)
    set_requires_grad(model.global_to_dec_ctx, True)
    set_requires_grad(model.global_embedder, False)
    set_requires_grad(model.spatial_map_prior, False)

    block0 = model.dec_blocks[0]
    set_requires_grad(block0.norm2, True)
    set_requires_grad(block0.norm_ctx, True)
    set_requires_grad(block0.cross_attn, True)
    block0.ctx_gate.requires_grad = True

    set_requires_grad(model.stem, False)
    set_requires_grad(model.patch_embed, False)
    set_requires_grad(model.sparse_encoder, False)
    set_requires_grad(model.to_code, False)
    for parameter in model.vq.parameters():
        parameter.requires_grad = False

    model.vq.freeze_codebook_updates = True
    model.vq.dead_code_restart_every = 0
    model.vq.duplicate_restart_every = 0

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Stage 2C: trainable params = {n_trainable:,} / {n_total:,}")


def make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("No trainable parameters. Did freeze_for_stage freeze everything?")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def load_spatial_pretrain_if_available(model: nn.Module):
    if not USE_GCT_PRETRAIN_MODULE:
        print("Spatial bias disabled for VQVAE stages.")
        return

    if model.spatial_map_prior is None:
        print("Model has no spatial_map_prior branch.")
        return

    if not SPATIAL_CKPT.exists():
        print(f"WARNING: spatial checkpoint not found: {SPATIAL_CKPT}")
        return

    spatial_ckpt = torch.load(SPATIAL_CKPT, map_location="cpu")
    model.global_embedder.load_state_dict(spatial_ckpt["global_embedder"], strict=True)
    
    try:
        model.spatial_map_prior.load_state_dict(spatial_ckpt["spatial_map_prior"], strict=True)
    except RuntimeError as e:
        print("[warn] strict spatial_map_prior load failed, likely adj_head bin-count mismatch.")
        print(e)
        sd = spatial_ckpt["spatial_map_prior"]
        cur = model.spatial_map_prior.state_dict()
        sd = {k: v for k, v in sd.items() if k in cur and tuple(v.shape) == tuple(cur[k].shape)}
        missing, unexpected = model.spatial_map_prior.load_state_dict(sd, strict=False)
        print("[warn] partial spatial_map_prior load:", "missing=", missing, "unexpected=", unexpected)
        
    if "memory_tok" in spatial_ckpt:
        model.memory_tok = GlobalContextSpatialBank()
        model.memory_tok.load_state_dict(spatial_ckpt["memory_tok"])
    else:
        model.memory_tok = None
    
    if "memory_pix" in spatial_ckpt:
        model.memory_pix = GlobalContextSpatialBank()
        model.memory_pix.load_state_dict(spatial_ckpt["memory_pix"])
    else:
        model.memory_pix = None
    
    if "memory_adj" in spatial_ckpt:
        model.memory_adj = GlobalContextAdjacencyBank()
        model.memory_adj.load_state_dict(spatial_ckpt["memory_adj"])
    else:
        model.memory_adj = None
    
    print(
        "Loaded memory banks:",
        f"tok={model.memory_tok is not None}",
        f"pix={model.memory_pix is not None}",
        f"adj={model.memory_adj is not None}",
    )
    
    print(f"Loaded spatial pretrain: {SPATIAL_CKPT}")


def load_stage0_memories_for_eval(model):
    if not SPATIAL_CKPT.exists():
        raise FileNotFoundError(
            f"Stage 3 global metrics require {SPATIAL_CKPT}"
        )

    ckpt = torch.load(SPATIAL_CKPT, map_location="cpu")

    required = ("memory_tok", "memory_pix", "memory_adj")
    missing = [name for name in required if name not in ckpt]
    if missing:
        raise RuntimeError(
            f"Stage 0 checkpoint is missing required memories: {missing}"
        )

    model.memory_tok = GlobalContextSpatialBank()
    model.memory_tok.load_state_dict(ckpt["memory_tok"])

    model.memory_pix = GlobalContextSpatialBank()
    model.memory_pix.load_state_dict(ckpt["memory_pix"])

    model.memory_adj = GlobalContextAdjacencyBank()
    model.memory_adj.load_state_dict(ckpt["memory_adj"])

    expected_bins = tuple(normalize_gap_bins(gap_bins))
    if tuple(model.memory_adj.gap_bins) != expected_bins:
        raise RuntimeError(
            "Stale Stage 0 adjacency checkpoint: "
            f"saved={model.memory_adj.gap_bins}, "
            f"current={expected_bins}"
        )

    expected_tok = (
        int(np.ceil(model.full_spatial_size[0] / model.patch_size[1])),
        int(np.ceil(model.full_spatial_size[1] / model.patch_size[2])),
    )
    expected_pix = tuple(model.full_spatial_size)

    for key, value in model.memory_tok._bank.items():
        if tuple(value.shape) != expected_tok:
            raise RuntimeError(
                f"Stale memory_tok entry {key}: "
                f"{tuple(value.shape)} != {expected_tok}"
            )

    for key, value in model.memory_pix._bank.items():
        if tuple(value.shape) != expected_pix:
            raise RuntimeError(
                f"Stale memory_pix entry {key}: "
                f"{tuple(value.shape)} != {expected_pix}"
            )

def run_stage0_spatial_pretrain(model, train_loader, device):
    if model.spatial_map_prior is None:
        raise RuntimeError("Cannot run stage 0 because model.spatial_map_prior is None.")

    print("\n" + "=" * 80)
    print("STAGE 0: GCT embedding pretraining")
    print("=" * 80)

    set_all_trainable(model, False)
    set_requires_grad(model.global_embedder, True)
    set_requires_grad(model.spatial_map_prior, True)

    optimizer = torch.optim.AdamW(
        list(model.global_embedder.parameters()) + list(model.spatial_map_prior.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )

    report = fit_spatial_prior_pretrain(
        model=model,
        train_loader=train_loader,
        spatial_ckpt_path=str(SPATIAL_CKPT),
        optimizer=optimizer,
        epochs=200,
        device=device,
        memory_momentum=0.98,
        memory_mode="max",
        lambda_sep=1e-4,
        early_stop_patience=20,
        adj_gap_bins=gap_bins,
    )

    set_all_trainable(model, True)
    return report


def common_fit_kwargs(model):
    return dict(
        use_amp=True,
        grad_accum_steps=grad_accum_steps,
        recon_tolerance=recon_tolerance,
        metric_tolerance=metric_tolerance,

        isi_gap_bins=gap_bins,
        isi_tau=0.25,
        isi_margin=0.25,
        isi_lower_margin=0.20,
        isi_lower_weight=0.50,
        lambda_isi=1e-2,
        
        memory_tok=getattr(model, "memory_tok", None),
        memory_pix=getattr(model, "memory_pix", None),
        memory_adj=getattr(model, "memory_adj", None),
        memory_adj_conf_den_scale=100.0,

        level2_start_epoch=1,
        level2_full_loss_epoch=20,
        lambda_sp_token=1e-3,
        lambda_sp_pixel=1e-4,
        sp_pixel_start_epoch=40,
        sp_pixel_warmup_epochs=60,
    )

def select_ckpt(stage: int, prefer_best: bool = True) -> Path:
    if stage == 1:
        best = CKPTS["stage1_best"]
        last = CKPTS["stage1_last"]
    elif stage == 2:
        best = CKPTS["stage2_best"]
        last = CKPTS["stage2_last"]
    else:
        raise ValueError(f"No VQVAE checkpoint defined for stage={stage}")

    if prefer_best and best.exists():
        return best
    if last.exists():
        return last
    if best.exists():
        return best

    # Earlier Stage-2 phases remain valid fallbacks when later phases have not
    # run. Prefer the most advanced available phase.
    if stage == 2:
        if prefer_best and CKPTS["stage2b_best"].exists():
            return CKPTS["stage2b_best"]
        if CKPTS["stage2b_last"].exists():
            return CKPTS["stage2b_last"]
        if CKPTS["stage2b_best"].exists():
            return CKPTS["stage2b_best"]
        if prefer_best and CKPTS["stage2a_best"].exists():
            return CKPTS["stage2a_best"]
        if CKPTS["stage2a_last"].exists():
            return CKPTS["stage2a_last"]
        if CKPTS["stage2a_best"].exists():
            return CKPTS["stage2a_best"]

    raise FileNotFoundError(f"No checkpoint found for stage {stage}: {best} or {last}")


def _normalize_substage_phases(phases, allowed, *, name):
    normalized = tuple(str(phase).lower() for phase in phases)
    invalid = [phase for phase in normalized if phase not in allowed]
    if invalid:
        raise ValueError(f"Unsupported {name} entries: {invalid}; allowed={allowed}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} contains duplicates: {normalized}")
    return tuple(phase for phase in allowed if phase in normalized)


def _select_stage2_phase_ckpt(phase: str, *, prefer_best: bool = True) -> Path:
    phase = str(phase).lower()
    mapping = {
        "2a": (CKPTS["stage2a_best"], CKPTS["stage2a_last"]),
        "2b": (CKPTS["stage2b_best"], CKPTS["stage2b_last"]),
        "2c": (CKPTS["stage2_best"], CKPTS["stage2_last"]),
    }
    if phase not in mapping:
        raise ValueError(f"Unsupported Stage-2 phase={phase!r}")

    best, last = mapping[phase]
    if prefer_best and best.exists():
        return best
    if last.exists():
        return last
    if best.exists():
        return best
    raise FileNotFoundError(
        f"No checkpoint found for Stage {phase.upper()}: {best} or {last}"
    )


def _stage2_phase_report(phase: str) -> Path:
    phase = str(phase).lower()
    return {
        "2a": REPORTS["stage2a"],
        "2b": REPORTS["stage2b"],
        "2c": REPORTS["stage2"],
    }[phase]


def _stage2_phase_eval_report(phase: str) -> Path:
    phase = str(phase).lower()
    return {
        "2a": REPORTS["stage2a_eval"],
        "2b": REPORTS["stage2b_eval"],
        "2c": REPORTS["stage2c_eval"],
    }[phase]


def _json_safe(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_json_report(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(obj), f, indent=4)
    print(f"Saved report: {path}")
   
    
def run_stage1(model, train_loader, val_loader, blank_logit_threshold):
    print("\n" + "=" * 80)
    print("STAGE 1: context-agnostic VQVAE motif learning")
    print("=" * 80)

    freeze_for_stage(model, 1)

    # Fixed surrogate boundary for all threshold-aware training losses.
    # Validation Best-F1 thresholds are recorded but never fed back.
    model._set_training_prob_threshold(0.5)

    n_epoch = 300
    
    optimizer = make_optimizer(model, lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch, eta_min=1e-5)

    report = fit_vqvae(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        epochs=n_epoch,
        ckpt_best_path=str(CKPTS["stage1_best"]),
        ckpt_last_path=str(CKPTS["stage1_last"]),
        early_stop_patience=40,
        val_metric_name="AUPRC_tol_cond",
        val_metric_goal="max",
        use_ROI_mask=False,

        # Stage 1: context loss ON, decoder context injection OFF.
        # This lets ctx losses shape encoder/codebook/decoder motifs,
        # without allowing cross-attention shortcuts.
        lambda_ctx=1e-1,
        lambda_ctx_field=5e-2,
        ctx_start_epoch=5,
        ctx_warmup_epochs=15,
        ctx_epoch_schedule={
            0: 5,   # log_mean_firing_density
            7: 10,   # active_site_ratio
            1: 20,   # var_x
            2: 20,   # var_y
            3: 20,   # var_t
            8: 40,   # temporal_trend
            4: 30,  # cov_xy
            5: 40,  # cov_xt
            6: 40,  # cov_yt
        },
        
        # CFG context dropout is irrelevant in Stage 1 because cross-attn is OFF.
        cfg_ctx_drop_start=0.0,
        cfg_ctx_drop_end=0.0,
        cfg_ctx_start_epoch=10**9,
        cfg_ctx_warmup_epochs=1,
        blank_logit_margin=blank_logit_threshold,
        
        lambda_code_norm=0.10,
        code_norm_rms_ceiling=10.0,
        code_norm_token_ceiling=14.0,

        # Spike discovery but avoid permanent overactivation.
        pos_weight_start=100.0,
        pos_weight_end=1.0,
        pos_decay_epochs=100,

        save_start_epoch = 125,
        **common_fit_kwargs(model),
    )

    with open(REPORTS["stage1"], "w") as f:
        json.dump(report, f, indent=4)
    print(f"Saved report: {REPORTS['stage1']}")
    return report


def _stage2_ctx_schedule():
    return {
        0: 1,   # log_mean_firing_density
        7: 1,   # active_site_ratio
        1: 5,   # var_x
        2: 5,   # var_y
        3: 5,   # var_t
        8: 10,  # temporal_trend
        4: 5,   # cov_xy
        5: 5,   # cov_xt
        6: 5,   # cov_yt
    }

def _make_stage2_scheduler(optimizer, n_epoch):
    def multiplier(epoch):
        progress = min(
            1.0,
            max(0.0, epoch / float(max(1, n_epoch))),
        )
        cosine = 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
        return 0.10 + 0.90 * cosine

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            multiplier
            for _ in optimizer.param_groups
        ],
    )


def run_stage2a(model, train_loader, val_loader, blank_logit_threshold):
    print("\n" + "=" * 80)
    print("STAGE 2A: exact convex projection decoder calibration")
    print("=" * 80)

    map_location = next(model.parameters()).device
    stage1_best_ckpt = select_ckpt(1, prefer_best=True)
    model.load_checkpoint(
        str(stage1_best_ckpt),
        map_location=map_location,
    )
    model._set_training_prob_threshold(0.5)
    freeze_for_stage(model, 2)

    projector = model.continuous_residual_projector
    projector.use_alpha_adapter = False
    projector.decode_with_projection_target = True

    print(
        f"Loaded Stage 1 best weights from: {stage1_best_ckpt}\n"
        "Stage 2A uses the Euclidean projection of z_e-z1 onto "
        "conv{z2_1,...,z2_K}. Encoder, to_code, z1, and z2 are frozen; "
        "the decoder adapts to this fixed continuous geometry."
    )

    n_epoch = int(STAGE2A_EPOCHS)
    stage2a_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not stage2a_params:
        raise RuntimeError("Stage 2A has no trainable decoder parameters.")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": stage2a_params,
                "lr": 1e-5,
                "weight_decay": 1e-4,
            },
        ]
    )
    scheduler = _make_stage2_scheduler(optimizer, n_epoch)

    report = fit_vqvae(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        epochs=n_epoch,
        ckpt_best_path=str(CKPTS["stage2a_best"]),
        ckpt_last_path=str(CKPTS["stage2a_last"]),
        early_stop_patience=30,
        val_metric_name="AUPRC_tol_cond",
        val_metric_goal="max",
        use_ROI_mask=False,
        lambda_vq=0.0,
        lambda_ctx=1e-1,
        lambda_ctx_field=1e-2,
        ctx_start_epoch=0,
        ctx_warmup_epochs=20,
        ctx_epoch_schedule=_stage2_ctx_schedule(),
        cfg_ctx_drop_start=0.0,
        cfg_ctx_drop_end=0.0,
        cfg_ctx_start_epoch=10**9,
        cfg_ctx_warmup_epochs=1,
        pos_weight_start=1.0,
        pos_weight_end=1.0,
        pos_decay_epochs=1,
        blank_logit_margin=blank_logit_threshold,
        lambda_cont_projection=0.0,
        lambda_cont_parent_margin=0.0,
        continuous_sample_start_epoch=10**9,
        continuous_sample_warmup_epochs=1,
        continuous_sample_mix_start=0.0,
        continuous_sample_mix_end=0.0,
        continuous_gumbel_tau_start=1.0,
        continuous_gumbel_tau_end=1.0,
        continuous_posterior_temperature=0.35,
        save_start_epoch=1,
        **{
            **common_fit_kwargs(model),
            "lambda_isi": 1e-1,
            "lambda_sp_pixel": 1e-3,
            "lambda_enc_var": 0.0,
            "lambda_code_norm": 0.0,
            "level2_start_epoch": 1,
            "level2_full_loss_epoch": 1,
        },
    )

    save_json_report(report, REPORTS["stage2a"])
    return report


def _select_stage2a_ckpt():
    for path in (
        CKPTS["stage2a_best"],
        CKPTS["stage2a_last"],
    ):
        if path.exists():
            return path

    raise FileNotFoundError(
        "The exact-convex Stage 2A checkpoint is missing. Run with "
        "STAGE2_PHASES=('2a', '2c') first. Old decoder-defined Stage-2 "
        "checkpoints are intentionally not reused for this geometry."
    )


def run_stage2b(model, train_loader, val_loader, blank_logit_threshold):
    print("\n" + "=" * 80)
    print("STAGE 2B: amortized convex projector fitting")
    print("=" * 80)

    map_location = next(model.parameters()).device
    stage2a_ckpt = _select_stage2a_ckpt()
    model.load_checkpoint(
        str(stage2a_ckpt),
        map_location=map_location,
    )
    model._set_training_prob_threshold(0.5)
    freeze_for_stage2b(model)

    print(
        f"Loaded Stage 2A best weights from: {stage2a_ckpt}\n"
        "Only the alpha adapter is trainable. Its target is the fixed exact "
        "convex projection from Stage 2A; reconstruction loss does not define "
        "the latent coordinates."
    )

    n_epoch = int(STAGE2B_EPOCHS)
    adapter_params = [
        parameter
        for parameter in (
            model.continuous_residual_projector
            .alpha_adapter
            .parameters()
        )
        if parameter.requires_grad
    ]
    if not adapter_params:
        raise RuntimeError("Stage 2B has no trainable alpha-adapter parameters.")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": adapter_params,
                "lr": 1e-4,
                "weight_decay": 1e-4,
            },
        ]
    )
    scheduler = _make_stage2_scheduler(optimizer, n_epoch)

    report = fit_vqvae(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        epochs=n_epoch,
        ckpt_best_path=str(CKPTS["stage2b_best"]),
        ckpt_last_path=str(CKPTS["stage2b_last"]),
        early_stop_patience=15,
        val_metric_name="cont_adapter_projection_mse",
        val_metric_goal="min",
        use_ROI_mask=False,
        lambda_vq=0.0,
        lambda_ctx=0.0,
        lambda_ctx_field=0.0,
        ctx_start_epoch=10**9,
        ctx_warmup_epochs=1,
        ctx_epoch_schedule=_stage2_ctx_schedule(),
        cfg_ctx_drop_start=0.0,
        cfg_ctx_drop_end=0.0,
        cfg_ctx_start_epoch=10**9,
        cfg_ctx_warmup_epochs=1,
        pos_weight_start=1.0,
        pos_weight_end=1.0,
        pos_decay_epochs=1,
        blank_logit_margin=blank_logit_threshold,
        lambda_cont_projection=1.0,
        lambda_cont_parent_margin=0.0,
        continuous_sample_start_epoch=10**9,
        continuous_sample_warmup_epochs=1,
        continuous_sample_mix_start=0.0,
        continuous_sample_mix_end=0.0,
        continuous_gumbel_tau_start=1.0,
        continuous_gumbel_tau_end=1.0,
        continuous_posterior_temperature=0.35,
        save_start_epoch=1,
        **{
            **common_fit_kwargs(model),
            "lambda_isi": 0.0,
            "lambda_sp_pixel": 0.0,
            "lambda_enc_var": 0.0,
            "lambda_code_norm": 0.0,
            "level2_start_epoch": 1,
            "level2_full_loss_epoch": 1,
        },
    )

    save_json_report(report, REPORTS["stage2b"])
    return report


def run_stage2c(model, train_loader, val_loader, blank_logit_threshold):
    print("\n" + "=" * 80)
    print("STAGE 2C: context-conditioned projected-latent calibration")
    print("=" * 80)

    map_location = next(model.parameters()).device
    stage2a_ckpt = _select_stage2a_ckpt()
    model.load_checkpoint(
        str(stage2a_ckpt),
        map_location=map_location,
    )
    model._set_training_prob_threshold(0.5)
    freeze_for_stage2c(model)
    model.continuous_residual_sample_mix = 0.0

    print(
        f"Loaded Stage 2A best weights from: {stage2a_ckpt}\n"
        "The exact convex projection remains active. Encoder/codebooks and "
        "decoder backbone are frozen. Only context projections, cross-attention, "
        "and its zero-initialized residual gate are trainable."
    )

    context_params = [
        parameter
        for module in (
            model.local_embedder,
            model.local_to_dec_ctx,
            model.global_to_dec_ctx,
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    ]

    block0 = model.dec_blocks[0]
    cross_params = [
        parameter
        for module in (
            block0.norm2,
            block0.norm_ctx,
            block0.cross_attn,
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if block0.ctx_gate.requires_grad:
        cross_params.append(block0.ctx_gate)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": context_params,
                "lr": 1e-4,
                "weight_decay": 1e-4,
            },
            {
                "params": cross_params,
                "lr": 5e-5,
                "weight_decay": 1e-4,
            },
        ]
    )

    n_epoch = int(STAGE2C_EPOCHS)
    scheduler = _make_stage2_scheduler(optimizer, n_epoch)

    report = fit_vqvae(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        epochs=n_epoch,
        ckpt_best_path=str(CKPTS["stage2_best"]),
        ckpt_last_path=str(CKPTS["stage2_last"]),
        early_stop_patience=20,
        val_metric_name="AUPRC_tol_cond",
        val_metric_goal="max",
        use_ROI_mask=False,
        lambda_vq=0.0,
        lambda_ctx=1e-1,
        lambda_ctx_field=1e-2,
        ctx_start_epoch=0,
        ctx_warmup_epochs=10,
        ctx_epoch_schedule=_stage2_ctx_schedule(),
        cfg_ctx_drop_start=0.0,
        cfg_ctx_drop_end=0.0,
        cfg_ctx_start_epoch=10**9,
        cfg_ctx_warmup_epochs=1,
        pos_weight_start=1.0,
        pos_weight_end=1.0,
        pos_decay_epochs=1,
        blank_logit_margin=blank_logit_threshold,
        lambda_cont_projection=0.0,
        lambda_cont_parent_margin=0.0,
        continuous_sample_start_epoch=10**9,
        continuous_sample_warmup_epochs=1,
        continuous_sample_mix_start=0.0,
        continuous_sample_mix_end=0.0,
        continuous_gumbel_tau_start=1.0,
        continuous_gumbel_tau_end=1.0,
        continuous_posterior_temperature=0.35,
        save_start_epoch=1,
        **{
            **common_fit_kwargs(model),
            "lambda_isi": 1e-1,
            "lambda_sp_pixel": 1e-3,
            "lambda_enc_var": 0.0,
            "level2_start_epoch": 1,
            "level2_full_loss_epoch": 1,
        },
    )

    save_json_report(report, REPORTS["stage2"])
    return report


def run_stage2(model, train_loader, val_loader, blank_logit_threshold):
    phases = _normalize_substage_phases(
        STAGE2_PHASES,
        ("2a", "2b", "2c"),
        name="STAGE2_PHASES",
    )

    reports = {}
    if "2a" in phases:
        reports["2a"] = run_stage2a(
            model,
            train_loader,
            val_loader,
            blank_logit_threshold,
        )

    if "2b" in phases:
        reports["2b"] = run_stage2b(
            model,
            train_loader,
            val_loader,
            blank_logit_threshold,
        )

    if "2c" in phases:
        reports["2c"] = run_stage2c(
            model,
            train_loader,
            val_loader,
            blank_logit_threshold,
        )

    return reports


@torch.no_grad()
def measure_stage3_active_token_counts(
    model,
    loader,
    device,
    *,
    num_passes=5,
):
    model.eval()

    all_counts = []

    num_passes = int(num_passes)

    if num_passes < 1:
        raise ValueError(
            f"num_passes must be at least 1, got {num_passes}"
        )

    for pass_idx in range(num_passes):
        pass_counts = []

        for batch in loader:
            x = batch["x"].to(
                device,
                non_blocking=True,
            ).float()

            gct = batch["global_ctx"].to(
                device,
                non_blocking=True,
            ).float()

            lct = batch["local_ctx"].to(
                device,
                non_blocking=True,
            ).float()

            out = model(
                x,
                global_ctx=gct,
                local_ctx=lct,
                predict_mask_spec=None,
            )

            codes = out["codes"].long()

            counts = (
                codes[..., 0]
                .ne(-1)
                .sum(dim=1)
            )

            counts_cpu = counts.detach().cpu()

            pass_counts.append(
                counts_cpu
            )
            all_counts.append(
                counts_cpu
            )

        if not pass_counts:
            raise RuntimeError(
                f"Kmax estimation pass {pass_idx + 1} "
                "produced no samples."
            )

        pass_counts_np = torch.cat(
            pass_counts,
            dim=0,
        ).numpy()

        print(
            f"Stage 3 Kmax estimation pass "
            f"{pass_idx + 1}/{num_passes}: "
            f"n={pass_counts_np.size}, "
            f"mean={pass_counts_np.mean():.2f}, "
            f"maximum={pass_counts_np.max()}"
        )

    if not all_counts:
        raise RuntimeError(
            "Cannot determine Stage 3 Kmax: "
            "the training loader produced no samples."
        )

    counts = torch.cat(
        all_counts,
        dim=0,
    ).numpy()

    stats = {
        "num_passes": int(num_passes),
        "num_samples": int(counts.size),
        "minimum": int(counts.min()),
        "maximum": int(counts.max()),
        "mean": float(counts.mean()),
        "median": float(np.median(counts)),
        "p95": float(np.percentile(counts, 95)),
        "p99": float(np.percentile(counts, 99)),
        "p99_5": float(np.percentile(counts, 99.5)),
        "zero_fraction": float(
            np.mean(counts == 0)
        ),
    }

    print(
        "Stage 3 active-token count statistics:"
    )
    print(
        json.dumps(
            stats,
            indent=2,
        )
    )

    return stats


def build_prior_from_model(
    model,
    device,
    *,
    Kmax,
    coordinate_mode=None,
):
    gct_mapper = copy.deepcopy(model.global_embedder).eval()
    lct_mapper = copy.deepcopy(model.local_embedder).eval()

    K1 = int(model.vq.num_codes_per_level[0])
    K2 = int(model.vq.num_codes_per_level[1])

    T, H, W = model.img_size
    pT, pH, pW = model.patch_size
    token_grid = (T // pT, H // pH, W // pW)

    activity_prior = DETRActivityPrior(
        global_dim=dim_assay_for_emb,
        local_dim=9,
        num_tasks=4,
        token_grid=token_grid,
        Kmax=int(Kmax),
        d_model=128,
        n_layer=4,
        n_head=4,
        dropout=0.1,
        coordinate_mode=(
            STAGE3_COORDINATE_MODE
            if coordinate_mode is None
            else coordinate_mode
        ),
    ).to(device)

    motif_prior = MaskGITMotifPrior(
        K1=K1,
        K2=K2,
        num_tasks=4,
        gct_mapper=gct_mapper,
        lct_mapper=lct_mapper,
        gct_latent_dim=model.global_emb_dim,
        lct_latent_dim=model.local_emb_dim,
        d_model=128,
        n_layer=4,
        n_head=4,
        max_len=token_grid[0] * token_grid[1] * token_grid[2],
        dropout=0.1,
        z1_codebook=model.vq.tree_embeds[0],
        z2_codebook=model.vq.tree_embeds[1],
        z1_scale=float(model.vq.level_scales[0]),
        z2_scale=float(model.vq.level_scales[1]),
        hull_margin_fraction=float(
            model.continuous_residual_projector.hull_margin_fraction
        ),
    ).to(device)

    return HierarchicalCodebookPrior(
        activity_prior=activity_prior,
        motif_prior=motif_prior,
    ).to(device)

def _stage3_token_grid(model):
    return tuple(
        int(model.img_size[i] // model.patch_size[i])
        for i in range(3)
    )


def _read_stage3_activity_metadata(path, model):
    path = Path(path)
    meta = torch.load(path, map_location="cpu")

    if "Kmax" not in meta or "token_grid" not in meta:
        raise RuntimeError(
            f"Activity checkpoint {path} must contain Kmax and token_grid metadata."
        )

    saved_grid = tuple(map(int, meta["token_grid"]))
    expected_grid = _stage3_token_grid(model)
    if saved_grid != expected_grid:
        raise RuntimeError(
            f"Activity checkpoint token grid {saved_grid} does not match "
            f"the current VQ-VAE token grid {expected_grid}. Set "
            "STAGE3_RECOMPUTE_KMAX=True and retrain 3B."
        )

    state = _activity_state_from_checkpoint(meta)
    inferred_mode = infer_activity_coordinate_mode_from_state_dict(state)
    saved_mode = str(meta.get("coordinate_mode", inferred_mode)).lower()
    if saved_mode != inferred_mode:
        raise RuntimeError(
            f"Activity checkpoint {path} declares coordinate_mode={saved_mode!r}, "
            f"but its state dict is {inferred_mode!r}."
        )

    saved_ntok = int(meta.get("Ntok", np.prod(saved_grid)))
    expected_ntok = int(np.prod(expected_grid))
    if saved_ntok != expected_ntok:
        raise RuntimeError(
            f"Activity checkpoint Ntok={saved_ntok} does not match token-grid "
            f"product {expected_ntok}."
        )

    saved_named_grid = (
        int(meta.get("Ttok", saved_grid[0])),
        int(meta.get("Htok", saved_grid[1])),
        int(meta.get("Wtok", saved_grid[2])),
    )
    if saved_named_grid != saved_grid:
        raise RuntimeError(
            f"Activity checkpoint {path} has inconsistent coordinate metadata: "
            f"token_grid={saved_grid}, named grid={saved_named_grid}."
        )

    if "coordinate_mode" not in meta:
        print(
            f"Activity checkpoint {path} has no coordinate_mode metadata; "
            f"inferred legacy mode={saved_mode!r} from its head weights."
        )

    return {
        "Kmax": int(meta["Kmax"]),
        "token_grid": saved_grid,
        "coordinate_mode": saved_mode,
        "Ttok": saved_named_grid[0],
        "Htok": saved_named_grid[1],
        "Wtok": saved_named_grid[2],
        "Ntok": saved_ntok,
    }


def _best_stage3b_checkpoint_path():
    hard_path = CKPTS["activity_prior_best_hard_metric"]
    if hard_path.exists():
        return hard_path
    return CKPTS["activity_prior_best"]


def _activity_state_from_checkpoint(checkpoint, state_key=None):
    if state_key is not None:
        if state_key not in checkpoint:
            raise KeyError(
                f"Activity checkpoint is missing requested state key {state_key!r}."
            )
        return checkpoint[state_key]
    for key in ("model", "activity_prior"):
        if key in checkpoint and checkpoint[key] is not None:
            return checkpoint[key]
    raise KeyError("Activity checkpoint contains neither 'model' nor 'activity_prior'.")


def _load_activity_state_compat(
    activity_prior,
    checkpoint,
    *,
    path,
    state_key=None,
    allow_coordinate_partial=False,
):
    state = _activity_state_from_checkpoint(checkpoint, state_key=state_key)
    checkpoint_mode = str(
        checkpoint.get(
            "coordinate_mode",
            infer_activity_coordinate_mode_from_state_dict(state),
        )
    ).lower()
    model_mode = str(activity_prior.coordinate_mode).lower()
    print(
        f"Activity checkpoint coordinate mode: saved={checkpoint_mode}, "
        f"model={model_mode}, path={path}"
    )
    coordinate_partial = checkpoint_mode != model_mode
    if coordinate_partial:
        if not allow_coordinate_partial:
            raise RuntimeError(
                f"Cannot load {checkpoint_mode!r} activity coordinates into a "
                f"{model_mode!r} model from {path}. Stage 3B must be retrained."
            )
        skipped_prefixes = ("grid_head.", "t_head.", "h_head.", "w_head.")
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith(skipped_prefixes)
        }
        print(
            "PARTIAL LOAD: compatible activity backbone/count/event weights were "
            "loaded, while coordinate heads remain newly initialized. This is "
            "not a trained coordinate model; retrain Stage 3B before evaluation."
        )
    incompatible = activity_prior.load_state_dict(state, strict=False)
    allowed_missing = {
        "count_event_film.weight",
        "count_event_film.bias",
    }
    if coordinate_partial:
        if model_mode == "joint_dense":
            allowed_missing.update({"grid_head.weight", "grid_head.bias"})
        else:
            allowed_missing.update({
                "t_head.weight", "t_head.bias",
                "h_head.weight", "h_head.bias",
                "w_head.weight", "w_head.bias",
            })
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    disallowed_missing = missing - allowed_missing
    if disallowed_missing or unexpected:
        raise RuntimeError(
            f"Incompatible activity checkpoint {path}: "
            f"missing={sorted(disallowed_missing)}, unexpected={sorted(unexpected)}"
        )
    if missing:
        print(
            f"Loaded legacy activity checkpoint {path} without {sorted(missing)}. "
            "The new count-to-event FiLM remains at its zero-initialized identity. "
            "Retrain Stage 3B before treating this as a calibrated checkpoint."
        )
    else:
        print(f"Loaded activity checkpoint weights: {path}")


def _print_stage3_startup(label, hyperparameters, module, *, checkpoint_paths=()):
    trainable = [
        (name, int(parameter.numel()))
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    print(f"{label} hyperparameters:")
    print(json.dumps(hyperparameters, indent=2, sort_keys=True))
    dense_elements = int(batch_size) * int(module.Kmax) * int(module.Ntok)
    coordinate_logit_elements = (
        dense_elements
        if module.coordinate_mode == "joint_dense"
        else int(batch_size)
        * int(module.Kmax)
        * int(module.Ttok + module.Htok + module.Wtok)
    )
    grid_head_parameters = (
        sum(parameter.numel() for parameter in module.grid_head.parameters())
        if module.grid_head is not None
        else 0
    )
    print(
        f"{label} coordinate mode: {module.coordinate_mode}\n"
        f"  token grid: ({module.Ttok}, {module.Htok}, {module.Wtok})\n"
        f"  Ntok: {module.Ntok}\n"
        f"  grid-head parameters: {grid_head_parameters:,}\n"
        f"  coordinate-logit tensor at batch_size={batch_size}: "
        f"{coordinate_logit_elements:,} elements, "
        f"{coordinate_logit_elements * 2 / (1024 ** 2):.2f} MiB BF16/FP16, "
        f"{coordinate_logit_elements * 4 / (1024 ** 2):.2f} MiB FP32\n"
        f"  dense B*K*N occupancy intermediate: {dense_elements:,} elements, "
        f"{dense_elements * 2 / (1024 ** 2):.2f} MiB BF16/FP16, "
        f"{dense_elements * 4 / (1024 ** 2):.2f} MiB FP32"
    )
    print(
        f"{label} trainable parameters: "
        f"{sum(count for _, count in trainable):,} across {len(trainable)} tensors"
    )
    for name, count in trainable:
        print(f"  trainable {name}: {count:,}")
    for checkpoint_path in checkpoint_paths:
        print(f"  checkpoint path: {checkpoint_path}")
    print(
        f"  deterministic validation masks: "
        f"{bool(hyperparameters.get('deterministic_validation_masks', False))}"
    )
    print(f"  hard activity mode: {hyperparameters.get('hard_activity_mode', 'n/a')}")


def _resolve_stage3_kmax(model, train_loader, device):
    if not STAGE3_RECOMPUTE_KMAX:
        for path in (
            CKPTS["activity_prior_best_hard_metric"],
            CKPTS["activity_prior_best"],
            CKPTS["activity_prior_refined_best"],
        ):
            if path.exists():
                meta = _read_stage3_activity_metadata(path, model)
                print(
                    f"Reusing Stage 3 Kmax={meta['Kmax']} from {path}. "
                    "Only metadata is reused; activity weights are not loaded."
                )
                return int(meta["Kmax"])

    count_stats = measure_stage3_active_token_counts(
        model=model,
        loader=train_loader,
        device=device,
        num_passes=int(STAGE3_KMAX_PASSES),
    )

    token_grid = _stage3_token_grid(model)
    Ntok = int(np.prod(token_grid))
    observed_max = int(count_stats["maximum"])
    stage3_kmax = min(
        Ntok,
        max(
            1,
            int(np.ceil(float(STAGE3_KMAX_MARGIN) * observed_max)),
        ),
    )

    print(
        "Selected Stage 3 Kmax:\n"
        f"  estimation passes = {count_stats['num_passes']}\n"
        f"  observed maximum = {observed_max}\n"
        f"  safety multiplier = {float(STAGE3_KMAX_MARGIN):.2f}\n"
        f"  selected Kmax = {stage3_kmax}\n"
        f"  Ntok = {Ntok}\n"
        f"  capacity ratio = {stage3_kmax / Ntok:.6f}"
    )
    return int(stage3_kmax)


def _load_stage3_motif_best(prior, device):
    path = CKPTS["motif_prior_best"]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Stage 3A checkpoint: {path}. Run with "
            "STAGE3_PHASES=('3a',) first."
        )
    ckpt = torch.load(path, map_location=device)
    prior.motif_prior.load_state_dict(ckpt["model"], strict=True)
    print(f"Loaded Stage 3A motif checkpoint: {path}")


def _load_stage3_activity_best(prior, model, device):
    path = _best_stage3b_checkpoint_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Stage 3B checkpoint: {path}. Run with "
            "STAGE3_PHASES=('3b',) first."
        )
    meta = _read_stage3_activity_metadata(path, model)
    if int(meta["Kmax"]) != int(prior.activity_prior.Kmax):
        raise RuntimeError(
            f"Built activity prior Kmax={prior.activity_prior.Kmax}, but "
            f"checkpoint {path} uses Kmax={meta['Kmax']}."
        )
    checkpoint = torch.load(path, map_location=device)
    _load_activity_state_compat(
        prior.activity_prior,
        checkpoint,
        path=path,
        state_key="model",
    )
    print(f"Loaded Stage 3B hard-metric activity checkpoint: {path}")


def run_stage3a(prior, model, train_loader, val_loader, device):
    print("[3A] Training motif prior.")
    opt_motif = torch.optim.AdamW(
        [p for p in prior.motif_prior.parameters() if p.requires_grad],
        lr=3e-4,
        weight_decay=0.01,
    )

    history = train_motif_prior_mgit(
        motif_prior=prior.motif_prior,
        vqvae=model,
        opt=opt_motif,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(STAGE3A_EPOCHS),
        grad_clip=1.0,
        ckpt_out=str(CKPTS["motif_prior_best"]),
        early_stop_patience=30,
        grad_accum_steps=grad_accum_steps,
        full_mask_prob=0.15,
        lambda_z1_distance=0.05,
        lambda_z1_neighbor_ce=0.25,
        z1_neighbor_tau=0.25,
        topk=(5, 2),
        lambda_ctx=1.0,
        lambda_ctx_field=0.05,
        lambda_adj=1.0,
        lambda_spatial=1.0,
        ctx_tau=0.25,
        ctx_field_tau=0.25,
        memory_tok=getattr(model, "memory_tok", None),
        memory_adj=getattr(model, "memory_adj", None),
        isi_gap_bins=gap_bins,
        isi_max_gap=max_gap_from_bins(gap_bins),
    )
    save_json_report(history, REPORTS["prior_motif"])
    _load_stage3_motif_best(prior, device)
    return history


def run_stage3b(prior, model, train_loader, val_loader, device):
    print("[3B] Training activity prior.")
    for parameter in prior.activity_prior.parameters():
        parameter.requires_grad_(True)

    config = dict(STAGE3B_HYPERPARAMETERS)
    config.update({
        "coordinate_mode": prior.activity_prior.coordinate_mode,
        "token_grid": [
            prior.activity_prior.Ttok,
            prior.activity_prior.Htok,
            prior.activity_prior.Wtok,
        ],
        "Ntok": prior.activity_prior.Ntok,
        "Kmax": prior.activity_prior.Kmax,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "optimizer": "AdamW",
        "amp_dtype": "CUDA autocast default FP16",
    })
    opt_activity = torch.optim.AdamW(
        [parameter for parameter in prior.activity_prior.parameters() if parameter.requires_grad],
        lr=float(config["lr"]),
        weight_decay=0.01,
    )
    _print_stage3_startup(
        "Stage 3B",
        config,
        prior.activity_prior,
        checkpoint_paths=(
            CKPTS["activity_prior_best_loss"],
            CKPTS["activity_prior_best_hard_metric"],
            CKPTS["activity_prior_best"],
        ),
    )

    history = train_activity_prior_detr(
        activity_prior=prior.activity_prior,
        vqvae=model,
        opt=opt_activity,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(STAGE3B_EPOCHS),
        grad_clip=1.0,
        lambda_count=float(config["lambda_count"]),
        lambda_count_neighbor=float(config["lambda_count_neighbor"]),
        lambda_count_distance=float(config["lambda_count_distance"]),
        lambda_obj=float(config["lambda_obj"]),
        lambda_coord=float(config["lambda_coord"]),
        lambda_soft_count=float(config["lambda_soft_count"]),
        lambda_soft_grid=float(config["lambda_soft_grid"]),
        lambda_dup=float(config["lambda_dup"]),
        no_object_weight=float(config["no_object_weight"]),
        count_neighbor_k=11,
        count_neighbor_tau=2.0,
        count_distance_scale=5.0,
        soft_count_beta=5.0,
        count_teacher_epochs=int(config["count_teacher_epochs"]),
        count_transition_epochs=int(config["count_transition_epochs"]),
        hard_tolerance=(1, 1, 1),
        memory_tok=getattr(model, "memory_tok", None),
        deterministic_val_masks=bool(config["deterministic_validation_masks"]),
        ckpt_out=str(CKPTS["activity_prior_best"]),
        ckpt_loss_out=str(CKPTS["activity_prior_best_loss"]),
        ckpt_hard_out=str(CKPTS["activity_prior_best_hard_metric"]),
        early_stop_patience=30,
        grad_accum_steps=grad_accum_steps,
    )
    save_json_report(history, REPORTS["prior_activity"])
    _load_stage3_activity_best(prior, model, device)
    return history


def run_stage3c(prior, model, train_loader, val_loader, device):
    print("[3C] Training inference-aligned event-placement calibration.")
    config = dict(STAGE3C_HYPERPARAMETERS)
    config.update({
        "coordinate_mode": prior.activity_prior.coordinate_mode,
        "token_grid": [
            prior.activity_prior.Ttok,
            prior.activity_prior.Htok,
            prior.activity_prior.Wtok,
        ],
        "Ntok": prior.activity_prior.Ntok,
        "Kmax": prior.activity_prior.Kmax,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "optimizer": "AdamW",
        "amp_dtype": "CUDA autocast default FP16",
    })
    trainable_names = configure_stage3c_event_calibration(prior.activity_prior)
    opt_refine = torch.optim.AdamW(
        [
            parameter
            for name, parameter in prior.activity_prior.named_parameters()
            if name in trainable_names
        ],
        lr=float(config["lr"]),
        weight_decay=0.01,
    )
    _print_stage3_startup(
        "Stage 3C",
        config,
        prior.activity_prior,
        checkpoint_paths=(
            CKPTS["motif_prior_best"],
            _best_stage3b_checkpoint_path(),
            CKPTS["activity_prior_refined_best"],
        ),
    )

    history = train_activity_prior_with_frozen_motif(
        activity_prior=prior.activity_prior,
        motif_prior=prior.motif_prior,
        vqvae=model,
        opt=opt_refine,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(STAGE3C_EPOCHS),
        grad_clip=1.0,
        ckpt_out=str(CKPTS["activity_prior_refined_best"]),
        early_stop_patience=20,
        grad_accum_steps=grad_accum_steps,
        freeze_motif=True,
        lambda_detr=float(config["lambda_detr"]),
        lambda_count=float(config["lambda_count"]),
        lambda_count_neighbor=float(config["lambda_count_neighbor"]),
        lambda_count_distance=float(config["lambda_count_distance"]),
        lambda_obj=float(config["lambda_obj"]),
        lambda_coord=float(config["lambda_coord"]),
        lambda_soft_count=float(config["lambda_soft_count"]),
        lambda_soft_grid=float(config["lambda_soft_grid"]),
        lambda_dup=float(config["lambda_dup"]),
        no_object_weight=float(config["no_object_weight"]),
        count_neighbor_k=11,
        count_neighbor_tau=2.0,
        count_distance_scale=5.0,
        soft_count_beta=5.0,
        lambda_ctx=float(config["lambda_ctx"]),
        lambda_ctx_field=float(config["lambda_ctx_field"]),
        lambda_adj=float(config["lambda_adj"]),
        lambda_spatial=float(config["lambda_spatial"]),
        auxiliary_ramp_epochs=int(config["auxiliary_ramp_epochs"]),
        ctx_field_tau=0.25,
        memory_tok=getattr(model, "memory_tok", None),
        memory_adj=getattr(model, "memory_adj", None),
        isi_gap_bins=gap_bins,
        isi_max_gap=max_gap_from_bins(gap_bins),
        generation_val_max_batches=int(config["generation_val_max_batches"]),
        generation_motif_steps=int(config["generation_motif_steps"]),
        deterministic_val_masks=bool(config["deterministic_validation_masks"]),
        hard_tolerance=(1, 1, 1),
    )
    save_json_report(history, REPORTS["prior_refine"])
    return history


def run_stage3_prior(model, train_loader, val_loader, device):
    print("\n" + "=" * 80)
    print("STAGE 3: staged prior learning")
    print("=" * 80)

    phases = _normalize_substage_phases(
        STAGE3_PHASES,
        ("3a", "3b", "3c"),
        name="STAGE3_PHASES",
    )
    if not phases:
        print("STAGE3_PHASES is empty; no Stage 3 training was requested.")
        return {}

    ckpt = select_ckpt(2, prefer_best=True)
    model.load_checkpoint(str(ckpt), map_location=device)
    print(f"Loaded VQVAE checkpoint for Stage 3 prior: {ckpt}")
    freeze_for_stage(model, 3)

    stage3_kmax = _resolve_stage3_kmax(model, train_loader, device)
    stage3_coordinate_mode = STAGE3_COORDINATE_MODE
    if "3c" in phases and "3b" not in phases:
        stage3b_path = _best_stage3b_checkpoint_path()
        stage3b_meta = _read_stage3_activity_metadata(stage3b_path, model)
        stage3_coordinate_mode = stage3b_meta["coordinate_mode"]
        print(
            "Stage 3C-only run will use the coordinate mode stored in its "
            f"Stage 3B checkpoint: {stage3_coordinate_mode!r}."
        )
    prior = build_prior_from_model(
        model,
        device,
        Kmax=stage3_kmax,
        coordinate_mode=stage3_coordinate_mode,
    )

    reports = {}

    # Fixed dependency order, matching Stage 2's phase driver.
    if "3a" in phases:
        reports["3a"] = run_stage3a(
            prior, model, train_loader, val_loader, device
        )

    if "3b" in phases:
        reports["3b"] = run_stage3b(
            prior, model, train_loader, val_loader, device
        )

    if "3c" in phases:
        if "3a" not in phases:
            _load_stage3_motif_best(prior, device)
        if "3b" not in phases:
            _load_stage3_activity_best(prior, model, device)
        reports["3c"] = run_stage3c(
            prior, model, train_loader, val_loader, device
        )

    return reports


def _select_stage3_activity_checkpoint(phase: str):
    phase = str(phase).lower()
    if phase == "3b":
        path = _best_stage3b_checkpoint_path()
        state_key = "model"
    elif phase == "3c":
        refined_path = CKPTS["activity_prior_refined_best"]
        if refined_path.exists():
            refined = torch.load(refined_path, map_location="cpu")
            if refined.get("accepted", False):
                return refined_path, "activity_prior"
            print(
                f"Stage 3C checkpoint {refined_path} was not accepted by the "
                "hard-generation gate; evaluating the unrefined Stage 3B checkpoint."
            )
        path = _best_stage3b_checkpoint_path()
        state_key = "model"
    else:
        raise ValueError(
            f"Stage-3 activity checkpoint phase must be '3b' or '3c', got {phase!r}."
        )

    if not path.exists():
        raise FileNotFoundError(
            f"Requested Stage {phase.upper()} checkpoint is missing: {path}"
        )
    return path, state_key


@torch.no_grad()
def _load_stage3_eval_prior(
    model,
    device,
    *,
    phase: str,
    load_generation_memories: bool = False,
    load_motif: bool = True,
):
    phase = str(phase).lower()
    if phase not in ("3a", "3b", "3c"):
        raise ValueError(f"Unsupported Stage-3 evaluation phase={phase!r}")

    vq_ckpt = select_ckpt(2, prefer_best=True)
    model.load_checkpoint(str(vq_ckpt), map_location=device)
    if load_generation_memories:
        load_stage0_memories_for_eval(model)
    freeze_for_stage(model, 3)

    if phase == "3a":
        # The activity branch is unused for 3A predictive evaluation.
        stage3_kmax = 1
        activity_ckpt_path = None
        activity_state_key = None
        activity_coordinate_mode = STAGE3_COORDINATE_MODE
    else:
        activity_ckpt_path, activity_state_key = (
            _select_stage3_activity_checkpoint(phase)
        )
        activity_meta = _read_stage3_activity_metadata(
            activity_ckpt_path,
            model,
        )
        stage3_kmax = int(activity_meta["Kmax"])
        activity_coordinate_mode = activity_meta["coordinate_mode"]

    prior = build_prior_from_model(
        model,
        device,
        Kmax=stage3_kmax,
        coordinate_mode=activity_coordinate_mode,
    )
    if load_motif:
        _load_stage3_motif_best(prior, device)

    if activity_ckpt_path is not None:
        activity_ckpt = torch.load(
            activity_ckpt_path,
            map_location=device,
        )
        _load_activity_state_compat(
            prior.activity_prior,
            activity_ckpt,
            path=activity_ckpt_path,
            state_key=activity_state_key,
        )

    prior.eval()
    print(f"Loaded VQVAE for Stage {phase.upper()} evaluation: {vq_ckpt}")
    if activity_ckpt_path is not None:
        print(
            f"Loaded Stage {phase.upper()} activity prior for evaluation: "
            f"{activity_ckpt_path}"
        )
    return prior


@torch.no_grad()
def load_stage3_prior(model, device, *, activity_phase: str):
    activity_phase = str(activity_phase).lower()
    if activity_phase not in ("3b", "3c"):
        raise ValueError(
            "Complete hierarchical generation requires activity_phase='3b' or '3c'."
        )
    return _load_stage3_eval_prior(
        model,
        device,
        phase=activity_phase,
        load_generation_memories=True,
        load_motif=True,
    )


@torch.no_grad()
def evaluate_stage3a_predictive(model, test_loader, device):
    prior = _load_stage3_eval_prior(
        model,
        device,
        phase="3a",
        load_generation_memories=False,
        load_motif=True,
    )
    motif_prior = prior.motif_prior
    motif_prior.eval()
    model.eval()

    totals = {
        "loss_total": 0.0,
        "loss_motif_objective": 0.0,
        "loss_z1_neighbor_ce": 0.0,
        "loss_z1_expected_distance": 0.0,
        "loss_z1": 0.0,
        "loss_alpha": 0.0,
        "z1_correct": 0.0,
        "z1_top5_correct": 0.0,
        "z1_tokens": 0.0,
        "alpha_argmax_correct": 0.0,
        "alpha_top2_correct": 0.0,
        "alpha_mae_sum": 0.0,
        "alpha_target_entropy_sum": 0.0,
        "alpha_pred_entropy_sum": 0.0,
        "alpha_tokens": 0.0,
        "samples": 0.0,
    }

    blank_code = getattr(model.vq, "blank_code", -1)

    for batch in test_loader:
        x, gct, lct, task_id, mask_spec = _batch_to_device(
            batch,
            device,
        )
        codes, alpha_target, pmask, _ = _vq_codes_alpha_and_pmask(
            model,
            x,
            gct,
            lct,
            mask_spec,
            device,
        )

        targets = motif_prior.make_targets_from_codes(
            codes=codes,
            predict_mask=pmask,
            blank_code=blank_code,
            alpha=alpha_target,
        )
        a_in, z1_in, z2_in, alpha_in, targets = (
            motif_prior.corrupt_inputs_from_targets(
                targets,
                ensure_at_least_one_mask=True,
                full_mask_prob=float(STAGE3A_EVAL_FULL_MASK_PROB),
            )
        )
        targets["z1_teacher_prob"] = 0.0

        logits, motif_loss, aux = motif_prior(
            a_in,
            z1_in,
            z2_in,
            alpha_in=alpha_in,
            global_ctx=gct,
            local_ctx=lct,
            task_id=task_id,
            targets=targets,
            loss_weights=(1.0, 1.0),
        )

        neighbor_loss = distance_neighborhood_ce_loss(
            logits["z1"],
            targets["z1"],
            targets["z1_loss_mask"],
            motif_prior.z1_distance_matrix,
            k=5,
            tau=0.25,
        )
        distance_loss = expected_code_distance_loss(
            logits["z1"],
            targets["z1"],
            targets["z1_loss_mask"],
            motif_prior.z1_distance_matrix,
        )
        total_loss = (
            motif_loss
            + 0.25 * neighbor_loss
            + 0.05 * distance_loss
        )

        batch_size_current = float(x.size(0))
        totals["samples"] += batch_size_current
        totals["loss_total"] += float(total_loss.item()) * batch_size_current
        totals["loss_motif_objective"] += float(motif_loss.item()) * batch_size_current
        totals["loss_z1_neighbor_ce"] += float(neighbor_loss.item()) * batch_size_current
        totals["loss_z1_expected_distance"] += float(distance_loss.item()) * batch_size_current

        z1_mask = targets["z1_loss_mask"].bool()
        n_z1 = float(z1_mask.sum().item())
        if n_z1 > 0:
            z1_logits = logits["z1"][z1_mask]
            z1_target = targets["z1"][z1_mask].long()
            totals["loss_z1"] += float(aux["loss_z1"].item()) * n_z1
            totals["z1_tokens"] += n_z1
            totals["z1_correct"] += float(
                z1_logits.argmax(dim=-1).eq(z1_target).sum().item()
            )
            z1_top5 = z1_logits.topk(
                min(5, z1_logits.shape[-1]),
                dim=-1,
            ).indices
            totals["z1_top5_correct"] += float(
                z1_top5.eq(z1_target.unsqueeze(-1)).any(dim=-1).sum().item()
            )

        alpha_mask = targets["alpha_loss_mask"].bool()
        n_alpha = float(alpha_mask.sum().item())
        if n_alpha > 0:
            alpha_pred = logits["alpha_mean"][alpha_mask]
            alpha_true = targets["alpha"][alpha_mask]
            alpha_true_idx = alpha_true.argmax(dim=-1)
            totals["loss_alpha"] += float(aux["loss_alpha"].item()) * n_alpha
            totals["alpha_tokens"] += n_alpha
            totals["alpha_argmax_correct"] += float(
                alpha_pred.argmax(dim=-1).eq(alpha_true_idx).sum().item()
            )
            alpha_top2 = alpha_pred.topk(
                min(2, alpha_pred.shape[-1]),
                dim=-1,
            ).indices
            totals["alpha_top2_correct"] += float(
                alpha_top2.eq(alpha_true_idx.unsqueeze(-1)).any(dim=-1).sum().item()
            )
            totals["alpha_mae_sum"] += float(
                (alpha_pred - alpha_true).abs().mean(dim=-1).sum().item()
            )
            totals["alpha_target_entropy_sum"] += float(
                (-(alpha_true * alpha_true.clamp_min(1e-12).log()).sum(dim=-1)).sum().item()
            )
            totals["alpha_pred_entropy_sum"] += float(
                (-(alpha_pred * alpha_pred.clamp_min(1e-12).log()).sum(dim=-1)).sum().item()
            )

    sample_den = max(totals["samples"], 1.0)
    z1_den = max(totals["z1_tokens"], 1.0)
    alpha_den = max(totals["alpha_tokens"], 1.0)
    report = {
        "phase": "3a",
        "loss_total": totals["loss_total"] / sample_den,
        "loss_motif_objective": totals["loss_motif_objective"] / sample_den,
        "loss_z1": totals["loss_z1"] / z1_den,
        "loss_alpha": totals["loss_alpha"] / alpha_den,
        "loss_z1_neighbor_ce": totals["loss_z1_neighbor_ce"] / sample_den,
        "loss_z1_expected_distance": totals["loss_z1_expected_distance"] / sample_den,
        "z1_acc": totals["z1_correct"] / z1_den,
        "z1_top5_acc": totals["z1_top5_correct"] / z1_den,
        "alpha_argmax_acc_proxy": totals["alpha_argmax_correct"] / alpha_den,
        "alpha_argmax_top2_acc_proxy": totals["alpha_top2_correct"] / alpha_den,
        "alpha_mae": totals["alpha_mae_sum"] / alpha_den,
        "alpha_target_entropy": totals["alpha_target_entropy_sum"] / alpha_den,
        "alpha_pred_entropy": totals["alpha_pred_entropy_sum"] / alpha_den,
        "supervised_z1_tokens": int(totals["z1_tokens"]),
        "supervised_alpha_tokens": int(totals["alpha_tokens"]),
        "samples": int(totals["samples"]),
        "full_mask_prob": float(STAGE3A_EVAL_FULL_MASK_PROB),
    }
    save_json_report(report, REPORTS["prior_motif_eval"])
    print("Stage 3A held-out predictive evaluation:", report)
    return report


@torch.no_grad()
def evaluate_stage3_activity_predictive(
    model,
    test_loader,
    device,
    *,
    phase: str,
):
    phase = str(phase).lower()
    if phase not in ("3b", "3c"):
        raise ValueError("Activity predictive evaluation requires phase '3b' or '3c'.")

    prior = _load_stage3_eval_prior(
        model,
        device,
        phase=phase,
        load_generation_memories=False,
        load_motif=False,
    )
    activity_prior = prior.activity_prior
    activity_prior.eval()
    model.eval()

    token_grid = (
        activity_prior.Ttok,
        activity_prior.Htok,
        activity_prior.Wtok,
    )
    blank_code = getattr(model.vq, "blank_code", -1)
    total = {}
    total_samples = 0.0

    for batch in test_loader:
        x, gct, lct, task_id, mask_spec = _batch_to_device(
            batch,
            device,
        )
        codes, pmask, _ = _vq_codes_and_pmask(
            model,
            x,
            gct,
            lct,
            mask_spec,
            device,
        )
        targets = build_activity_targets_from_codes(
            codes=codes,
            token_grid=token_grid,
            Kmax=activity_prior.Kmax,
            blank_code=blank_code,
            predict_mask=pmask,
        )
        a_in = _make_activity_in_from_codes(
            codes,
            pmask,
            blank_code=blank_code,
            a_mask_id=activity_prior.a_mask_id,
        )
        out = activity_prior(
            global_ctx=gct,
            local_ctx=lct,
            task_id=task_id,
            a_in=a_in,
            roi_mask=pmask,
        )
        _, aux = detr_activity_loss(
            out,
            targets,
            activity_prior,
            lambda_count=1.0,
            lambda_count_neighbor=0.25,
            lambda_count_distance=0.05,
            lambda_obj=1.0,
            lambda_coord=1.0,
            lambda_soft_count=0.1,
            lambda_soft_grid=1.0,
            lambda_dup=0.01,
            no_object_weight=0.1,
            count_neighbor_k=11,
            count_neighbor_tau=2.0,
            count_distance_scale=5.0,
            soft_count_beta=5.0,
        )

        batch_size_current = float(x.size(0))
        total_samples += batch_size_current
        for key, value in aux.items():
            if torch.is_tensor(value):
                value = float(value.detach().item())
            total[key] = total.get(key, 0.0) + float(value) * batch_size_current

    den = max(total_samples, 1.0)
    report = {key: value / den for key, value in total.items()}
    report["phase"] = phase
    report["samples"] = int(total_samples)
    report_path = (
        REPORTS["prior_activity_eval"]
        if phase == "3b"
        else REPORTS["prior_refine_eval"]
    )
    save_json_report(report, report_path)
    print(f"Stage {phase.upper()} held-out activity evaluation:", report)
    return report


@torch.no_grad()
def collect_generation_diagnostics(
    model,
    sampled,
    gen,
    grid,
):
    activity_out = sampled["activity_out"]
    activity = sampled["activity"].bool()
    codes = sampled["codes"].long()

    count_prob = torch.softmax(
        activity_out["count_logits"],
        dim=-1,
    )

    count_values = torch.arange(
        count_prob.shape[-1],
        device=count_prob.device,
        dtype=count_prob.dtype,
    )

    prob = gen["prob"]
    x_gen = gen["x_gen"]

    # (B,N,P), then count thresholded voxels per latent patch.
    generated_patches = model.patchify(
        x_gen,
        grid,
    )

    patch_spike_count = generated_patches.sum(
        dim=-1
    )

    rows = []

    for b in range(activity.shape[0]):
        active_mask = activity[b]
        active_count = int(
            active_mask.sum().item()
        )

        blank_active_count = int(
            (
                active_mask
                & patch_spike_count[b].eq(0)
            ).sum().item()
        )

        p = prob[b].float().reshape(-1)

        z1 = codes[b, :, 0]
        z2 = codes[b, :, 1]

        invalid = (
            (z1 < -1)
            | (z1 >= int(model.vq.num_codes_per_level[0]))
            | (z2 < -1)
            | (z2 >= int(model.vq.num_codes_per_level[1]))
            | (z1.eq(-1) ^ z2.eq(-1))
        )

        rows.append({
            "threshold_used": float(
                gen["threshold"]
            ),
            "patch_size": [
                int(v)
                for v in model.patch_size
            ],
            "token_grid_shape": [
                int(v)
                for v in grid
            ],
            "Ntok": int(activity.shape[1]),
            "Kmax": int(
                activity_out["count_logits"].shape[-1] - 1
            ),
            "sampled_active_token_count": active_count,
            "count_argmax": int(
                count_prob[b].argmax().item()
            ),
            "expected_count": float(
                (
                    count_prob[b]
                    * count_values
                ).sum().item()
            ),
            "p_count_zero": float(
                count_prob[b, 0].item()
            ),
            "count_entropy": float(
                -(
                    count_prob[b]
                    * count_prob[b]
                    .clamp_min(1e-12)
                    .log()
                ).sum().item()
            ),
            "generated_nonblank_motif_tokens": int(
                z1.ne(-1).sum().item()
            ),
            "invalid_motif_codes": int(
                invalid.sum().item()
            ),
            "decoded_probability_mean": float(
                p.mean().item()
            ),
            "decoded_probability_max": float(
                p.max().item()
            ),
            "decoded_probability_p50": float(
                torch.quantile(p, 0.50).item()
            ),
            "decoded_probability_p90": float(
                torch.quantile(p, 0.90).item()
            ),
            "decoded_probability_p95": float(
                torch.quantile(p, 0.95).item()
            ),
            "decoded_probability_p99": float(
                torch.quantile(p, 0.99).item()
            ),
            "decoded_probability_p99_9": float(
                torch.quantile(p, 0.999).item()
            ),
            "voxels_above_threshold": int(
                x_gen[b].sum().item()
            ),
            "final_generated_spike_count": int(
                x_gen[b].sum().item()
            ),
            "fraction_active_latents_decoding_zero_spikes": (
                float(
                    blank_active_count
                    / active_count
                )
                if active_count > 0
                else None
            ),
        })

    return rows


@torch.no_grad()
def evaluate_stage3_prior(
    model,
    test_loader,
    device,
    *,
    activity_phase="3c",
    out_dir="../viz_out_vqvae/vqvae_stage3/stage3_prior_gen",
    max_batches=20,
    samples_per_context=4,
    steps=12,
    temperature=1.0,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prior = load_stage3_prior(model, device, activity_phase=activity_phase)

    model.eval()
    prior.eval()

    rows = []

    for bidx, batch in enumerate(test_loader):
        if bidx >= max_batches:
            break

        x = batch["x"].to(device).float()
        gct = batch["global_ctx"].to(device).float()
        lct = batch["local_ctx"].to(device).float()

        # Each real test video provides only the requested global/local context.
        # Stage 3 then performs complete task-0 generation over all tokens.
        B = x.shape[0]

        gct_rep = gct.repeat_interleave(
            samples_per_context,
            dim=0,
        )
        lct_rep = lct.repeat_interleave(
            samples_per_context,
            dim=0,
        )

        roi_hw = batch.get("roi_hw", None)
        pad_hw = batch.get("pad_hw", None)

        # Preserve the exact crop/padding metadata for every repeated sample.
        roi_hw_rep = (
            None
            if roi_hw is None
            else [
                roi
                for roi in roi_hw
                for _ in range(samples_per_context)
            ]
        )

        pad_hw_rep = (
            None
            if pad_hw is None
            else [
                pad
                for pad in pad_hw
                for _ in range(samples_per_context)
            ]
        )

        # Encode the ground-truth video only to obtain its VQ codes,
        # reconstruction and token-grid geometry.
        # Do not inherit the randomly sampled dataset mask.
        tok_out = model(
            x,
            global_ctx=gct,
            local_ctx=lct,
            predict_mask_spec=None,
            roi_hw=roi_hw,
            pad_hw=pad_hw,
        )

        grid = tok_out["grid"]
        full_codes = tok_out["codes"].long()

        N = int(
            grid[0]
            * grid[1]
            * grid[2]
        )

        # Full generation: every token is inside the prediction ROI.
        generation_roi = torch.ones(
            (B * samples_per_context, N),
            dtype=torch.bool,
            device=device,
        )

        # Task 0 = full/exact generation.
        task_id = torch.zeros(
            (B * samples_per_context,),
            dtype=torch.long,
            device=device,
        )
        
        sampled = sample_hierarchical_roi(
            prior=prior,
            global_ctx=gct_rep,
            local_ctx=lct_rep,
            task_id=task_id,
            roi_mask=generation_roi,
            visible_codes=None,
            activity_count_temperature=1.0,
            activity_coord_temperature=temperature,
            motif_steps=steps,
            motif_temperature=temperature,
        )

        # This is a completely generated latent field.
        codes = sampled["codes"]

        gen = decode_codes_to_xgen(
            model,
            codes,
            alpha=sampled.get("alpha", None),
            grid=grid,
            global_ctx=gct_rep,
            local_ctx=lct_rep,
            roi_hw=roi_hw_rep,
            pad_hw=pad_hw_rep,
        )
        
        x_gen = gen["x_gen"]

        global_metric_rows = evaluate_generation_global_metrics(
            model=model,
            logits=gen["logits"],
            x_hard=x_gen,
            global_ctx=gct_rep,
            roi_hw=roi_hw_rep,
            pad_hw=pad_hw_rep,
            gap_bins=gap_bins,
            prob_threshold=float(gen["threshold"]),
        )

        diagnostic_rows = collect_generation_diagnostics(
            model=model,
            sampled=sampled,
            gen=gen,
            grid=grid,
        )

        batch_rows = save_generated_batch_outputs(
            x_gen=x_gen,
            out_dir=out_dir,
            prefix=f"stage3_testctx_b{bidx:04d}",
            intended_local_ctx=lct_rep,
            fps=30,
        )
        
        for i, r in enumerate(batch_rows):
            source_i = i // samples_per_context

            decoder_support = diagnostic_rows[i]
            global_metrics = global_metric_rows[i]

            # Keep the old flat fields temporarily for compatibility.
            r.update(decoder_support)

            r["local_context_metrics"] = {
                key: value
                for key, value in r.items()
                if (
                    key.startswith("gen_")
                    or key.startswith("target_")
                    or key.startswith("err_")
                    or key.startswith("match_")
                    or key == "all_context_matches"
                )
            }

            r["global_spatial_metrics"] = global_metrics[
                "global_spatial_metrics"
            ]
            r["global_adjacency_metrics"] = global_metrics[
                "global_adjacency_metrics"
            ]
            r["decoder_support_metrics"] = decoder_support

            r["generation_mode"] = "test_context"
            r["batch"] = int(bidx)
            r["sample_within_context"] = int(
                i % samples_per_context
            )
            r["task_id"] = int(task_id[i].item())

            r["assay_id"] = int(
                batch["assay_idx"][source_i].item()
            )

            r["global_ctx"] = (
                gct_rep[i]
                .detach()
                .cpu()
                .tolist()
            )

            r["roi_hw"] = (
                None
                if roi_hw_rep is None
                else [
                    int(v)
                    for v in roi_hw_rep[i]
                ]
            )

            r["pad_hw"] = (
                None
                if pad_hw_rep is None
                else [
                    int(v)
                    for v in pad_hw_rep[i]
                ]
            )

            r["full_spatial_size"] = [
                int(v)
                for v in model.full_spatial_size
            ]

            r["target_active_token_count"] = int(
                full_codes[source_i, :, 0]
                .ne(-1)
                .sum()
                .item()
            )

            rows.append(r)

        # torch.save(
        #     {
        #         "codes": codes.detach().cpu(),
        #         "logits": logits.detach().cpu(),
        #         "prob": prob.detach().cpu(),
        #         "x_gen": x_gen.detach().cpu(),
        #         "global_ctx": gct_rep.detach().cpu(),
        #         "local_ctx": lct_rep.detach().cpu(),
        #         "task_id": task_id.detach().cpu(),
        #         "grid": tuple(map(int, grid)),
        #     },
        #     out_dir / f"stage3_gen_batch_{bidx:04d}.pt",
        # )

    with open(out_dir / "stage3_generation_metrics.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Saved Stage 3 generated samples/metrics to: {out_dir}")
    return rows

@torch.no_grad()
def evaluate_stage3_prior_sampled_contexts(
    model,
    ref_loader,
    device,
    *,
    activity_phase="3c",
    out_dir="../viz_out_vqvae/vqvae_stage3/stage3_prior_sampled_ctx",
    context_bank_path="ckpts/context_prior.pkl",
    mode="random_full",
    fixed_global_ctx=None,
    partial_local=None,
    assay_id=None,
    k=64,
    max_samples=64,
    batch_size_gen=4,
    task_id=0,
    steps=12,
    temperature=1.0,
    ctx_temperature=0.05,
):
    """
    Stage-3 free-generation eval with controllable context sampling.

    Modes:
      random_full:
          sample realistic (global_ctx, local_ctx) pairs from context bank

      fixed_global:
          keep/query global_ctx fixed, retrieve realistic matching local_ctx

      partial_local:
          retrieve realistic full contexts matching partial local constraints

      fixed_global_partial_local:
          retrieve realistic local_ctx matching both fixed global_ctx and partial local constraints
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prior = load_stage3_prior(model, device, activity_phase=activity_phase)
    model.eval()
    prior.eval()

    ctx_sampler = ContextBankSampler.from_file(
        context_bank_path,
        model=model,
        device=device,
        seed=0,
    )

    # Only used to infer token grid geometry. This does not provide generation context.
    ref_batch = next(iter(ref_loader))
    x_ref = ref_batch["x"].to(device).float()
    g_ref = ref_batch["global_ctx"].to(device).float()
    l_ref = ref_batch["local_ctx"].to(device).float()

    tok_out = model(x_ref, global_ctx=g_ref, local_ctx=l_ref)
    grid = tok_out["grid"]
    N = int(grid[0] * grid[1] * grid[2])

    fixed_global_np = None
    if fixed_global_ctx is not None:
        fixed_global_np = fixed_global_ctx
        if isinstance(fixed_global_np, torch.Tensor):
            fixed_global_np = fixed_global_np.detach().cpu().numpy()

    rows = []
    n_done = 0
    sample_id = 0

    while n_done < max_samples:
        B = min(batch_size_gen, max_samples - n_done)

        if mode == "random_full":
            ctx_np = ctx_sampler.sample(
                assay_id=assay_id,
                batch_size=B,
                return_index=True,
            )

        elif mode == "fixed_global":
            if fixed_global_np is None:
                raise ValueError("mode='fixed_global' requires fixed_global_ctx.")
            ctx_np = ctx_sampler.sample(
                global_ctx=fixed_global_np,
                assay_id=assay_id,
                batch_size=B,
                k=k,
                temperature=ctx_temperature,
                return_index=True,
            )

        elif mode == "partial_local":
            if partial_local is None:
                raise ValueError("mode='partial_local' requires partial_local.")
            ctx_np = ctx_sampler.sample(
                partial_local=partial_local,
                assay_id=assay_id,
                batch_size=B,
                k=k,
                temperature=ctx_temperature,
                return_index=True,
            )

        elif mode == "fixed_global_partial_local":
            if fixed_global_np is None:
                raise ValueError("mode='fixed_global_partial_local' requires fixed_global_ctx.")
            if partial_local is None:
                raise ValueError("mode='fixed_global_partial_local' requires partial_local.")
            ctx_np = ctx_sampler.sample(
                global_ctx=fixed_global_np,
                partial_local=partial_local,
                assay_id=assay_id,
                batch_size=B,
                k=k,
                temperature=ctx_temperature,
                return_index=True,
            )

        else:
            raise ValueError(f"Unknown context sampling mode: {mode}")

        if mode in (
            "fixed_global",
            "fixed_global_partial_local",
        ):
            fixed_global_array = np.asarray(
                fixed_global_np,
                dtype=np.float32,
            ).reshape(1, -1)

            ctx_np["global_ctx"] = np.repeat(
                fixed_global_array,
                B,
                axis=0,
            )

        ctx_t = ctx_sampler.to_torch(
            ctx_np,
            device=device,
            task_id=task_id,
        )

        full_H, full_W = map(
            int,
            model.full_spatial_size,
        )

        runtime_H = int(
            grid[1] * model.patch_size[1]
        )
        runtime_W = int(
            grid[2] * model.patch_size[2]
        )

        pad_h = runtime_H - full_H
        pad_w = runtime_W - full_W

        if pad_h < 0 or pad_w < 0:
            raise RuntimeError(
                "Runtime spatial size is smaller than the "
                "full spatial memory: "
                f"runtime={(runtime_H, runtime_W)}, "
                f"full={(full_H, full_W)}"
            )

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        generation_roi_hw = [
            (0, 0, full_H, full_W)
            for _ in range(B)
        ]

        generation_pad_hw = [
            (
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
            )
            for _ in range(B)
        ]

        roi_mask = torch.ones(
            (B, N),
            device=device,
            dtype=torch.bool,
        )
        
        sampled = sample_hierarchical_roi(
            prior=prior,
            global_ctx=ctx_t["global_ctx"],
            local_ctx=ctx_t["local_ctx"],
            task_id=ctx_t["task_id"],
            roi_mask=roi_mask,
            activity_count_temperature=1.0,
            activity_coord_temperature=temperature,
            motif_steps=steps,
            motif_temperature=temperature,
        )
        
        codes = sampled["codes"]

        gen = decode_codes_to_xgen(
            model,
            codes,
            alpha=sampled.get("alpha", None),
            grid=grid,
            global_ctx=ctx_t["global_ctx"],
            local_ctx=ctx_t["local_ctx"],
            roi_hw=generation_roi_hw,
            pad_hw=generation_pad_hw,
        )

        x_gen = gen["x_gen"]

        global_metric_rows = evaluate_generation_global_metrics(
            model=model,
            logits=gen["logits"],
            x_hard=x_gen,
            global_ctx=ctx_t["global_ctx"],
            roi_hw=generation_roi_hw,
            pad_hw=generation_pad_hw,
            gap_bins=gap_bins,
        )

        diagnostic_rows = collect_generation_diagnostics(
            model=model,
            sampled=sampled,
            gen=gen,
            grid=grid,
        )

        batch_rows = save_generated_batch_outputs(
            x_gen=x_gen,
            out_dir=out_dir,
            prefix=f"{mode}_gen_{sample_id:04d}",
            intended_local_ctx=ctx_t["local_ctx"],
            fps=30,
            local_min=ctx_sampler.local_min,
            local_max=ctx_sampler.local_max,
            local_mean=ctx_sampler.local_mean,
            local_std=ctx_sampler.local_std,
            partial_local=partial_local,
        )
        
        for i, r in enumerate(batch_rows):
            decoder_support = diagnostic_rows[i]
            global_metrics = global_metric_rows[i]

            # Keep existing flat fields for compatibility.
            r.update(decoder_support)

            r["local_context_metrics"] = {
                key: value
                for key, value in r.items()
                if (
                    key.startswith("gen_")
                    or key.startswith("target_")
                    or key.startswith("err_")
                    or key.startswith("match_")
                    or key.startswith("partial_")
                    or key.startswith("in_bank_")
                    or key.startswith("z_from_bank_")
                    or key == "all_context_matches"
                    or key == "all_features_in_bank_range"
                )
            }

            r["global_spatial_metrics"] = global_metrics[
                "global_spatial_metrics"
            ]
            r["global_adjacency_metrics"] = global_metrics[
                "global_adjacency_metrics"
            ]
            r["decoder_support_metrics"] = decoder_support

            r["mode"] = str(mode)
            r["generation_mode"] = str(mode)
            r["sample_id"] = int(sample_id)

            r["context_bank_index"] = int(
                ctx_np["index"][i]
            )
            r["assay_id"] = int(
                ctx_np["assay_id"][i]
            )
            r["task_id"] = int(
                ctx_t["task_id"][i].item()
            )

            r["global_ctx"] = (
                ctx_t["global_ctx"][i]
                .detach()
                .cpu()
                .tolist()
            )

            r["roi_hw"] = [
                int(v)
                for v in generation_roi_hw[i]
            ]
            r["pad_hw"] = [
                int(v)
                for v in generation_pad_hw[i]
            ]
            r["full_spatial_size"] = [
                full_H,
                full_W,
            ]

            rows.append(r)
            sample_id += 1

        n_done += B
        
        
    save_generation_metrics_json(rows, out_dir /  f"{mode}_generation_metrics.json")


    print(f"Saved sampled-context Stage 3 generations to: {out_dir} | mode={mode}")
    return rows


def make_viz_loader(test_loader):
    return DataLoader(
        test_loader.dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=burst_collate,
        num_workers=0,
        pin_memory=True,
    )


def evaluate_and_visualize(
    model,
    test_loader,
    stage: int,
    assay_indices,
    assay_codebook,
    *,
    stage2_phase: Optional[str] = None,
):
    if stage not in (1, 2):
        return
    
    try:
        if stage == 2:
            if stage2_phase is None:
                raise ValueError("stage2_phase is required when evaluating Stage 2.")
            stage2_phase = str(stage2_phase).lower()
            ckpt = _select_stage2_phase_ckpt(stage2_phase, prefer_best=True)
        else:
            ckpt = select_ckpt(stage, prefer_best=True)
    except FileNotFoundError as e:
        if RUN_SKIP_MISSING_EVAL:
            print(f"Skipping eval/viz: {e}")
            return None
        raise

    device = next(model.parameters()).device
    model.load_checkpoint(str(ckpt), map_location=device)

    # Runtime latent mode is not part of the state_dict.
    model.use_continuous_residual = bool(
        stage == 2 and STAGE2_CONTINUOUS_RESIDUAL
    )

    using_alpha_adapter = bool(
        stage == 2 and stage2_phase == "2b"
    )
    using_stage2c_context = bool(
        stage == 2 and stage2_phase == "2c"
    )

    model.continuous_residual_projector.use_alpha_adapter = (
        using_alpha_adapter
    )
    model.continuous_residual_projector.decode_with_projection_target = True
    model.continuous_residual_sample_mix = 0.0
    set_decoder_cross_attention(
        model,
        enabled=using_stage2c_context,
        layers=(0,) if using_stage2c_context else (),
    )


    if RUN_EVAL:
        eval_label = stage2_phase.upper() if stage == 2 else str(stage)
        print(f"Evaluating VQVAE stage {eval_label} using {ckpt} ...")
        test_metrics = evaluate_vqvae(
            model,
            test_loader,
            pos_weight=2.0,
            use_amp=True,
            use_ROI_mask=False,
        )
        print(f"TEST stage {eval_label}:", test_metrics)
        if stage == 2:
            save_json_report(
                _json_safe(test_metrics),
                _stage2_phase_eval_report(stage2_phase),
            )

    if RUN_VIZ:
        out_root = (
            STAGE2_VIZ_ROOTS[stage2_phase]
            if stage == 2
            else VIZ_ROOTS[stage]
        )
        out_root.mkdir(parents=True, exist_ok=True)
        viz_loader = make_viz_loader(test_loader)
        
        if RUN_VIDEO_GEN:
            make_model_videos_vqvae(
                model,
                viz_loader,
                out_root=str(out_root),
                max_samples=max_viz_samples,
                fps=30,
                pool_t=1,
                thr=None,
                cmap_name="viridis",
                isi_gap_bins=gap_bins,
                isi_max_gap=max_gap_from_bins(
                    gap_bins
                ),
            )
            
        if RUN_PLOTTER:
            if stage == 1:
                report_path = REPORTS["stage1"]
            else:
                report_path = _stage2_phase_report(stage2_phase)
            if report_path.exists():
                run_plotter(
                    train_report=str(report_path),
                    eval_roots=[str(VIZ_ROOTS[stage])],
                    out_dir=str(VIZ_ROOTS[stage] / "figs"),
                    do_threshold_sweep=True,
                )

            if model.spatial_map_prior is not None:
                save_assaywise_spatial_maps(
                    model,
                    assay_indices=assay_indices,
                    n_assays=num_assays_for_emb,
                    out_dir=str(out_root / "viz_spatial_bias"),
                    assay_codebook=assay_codebook,
                )
                save_assaywise_adjacency_diagnostics(
                    model,
                    assay_indices=assay_indices,
                    n_assays=num_assays_for_emb,
                    out_dir=str(out_root / "viz_spatial_bias"),
                    assay_codebook=assay_codebook,
                )

            if RUN_CODEBOOK_DEBUG:
                plot_blank_active_tsne_l1(
                    viz_root=str(out_root),
                    out_png=str(out_root / "blank_active_tsne_l1.png"),
                )
                plot_blank_active_pca_l1(
                    viz_root=str(out_root),
                    out_png=str(out_root / "blank_active_pca_l1.png"),
                )


    return True


@torch.no_grad()
def debug_vq_codebooks(
    model,
    near_zero_thresh=1e-6,
    duplicate_cos_thresh=0.995,
    print_topk_pairs=10,
    blank_topk=10,
    save_txt_path=None,
    save_json_path=None,
):
    if not hasattr(model, "vq"):
        print("No hierarchical VQ codebooks found.")
        return

    lines = []
    out_json = {
        "levels": [],
    }

    def log(msg):
        print(msg)
        lines.append(str(msg))

    log("=" * 80)
    log("VQ CODEBOOK DEBUG")
    log("=" * 80)

    blank = model.vq.blank_token.detach().cpu().flatten()

    log("\n[Blank token]")
    log(f"norm: {blank.norm().item():.6g}")

    out_json["blank"] = {
        "norm": float(blank.norm().item()),
    }

    for lvl in range(int(model.vq.num_quantizers)):

        if hasattr(model.vq, "get_effective_codebook_weight"):
            cb = model.vq.get_effective_codebook_weight(lvl).detach().cpu()
        else:
            cb = model.vq.embeds[lvl].weight.detach().cpu()

        K, D = cb.shape

        norms = cb.norm(dim=1)
        alive = norms >= near_zero_thresh
        alive_count = int(alive.sum())

        log(f"\n[Level {lvl}] shape=({K},{D}) alive={alive_count}/{K}")
        log(
            f"norm min={norms.min().item():.6g} "
            f"max={norms.max().item():.6g} "
            f"mean={norms.mean().item():.6g}"
        )

        level_json = {
            "level": int(lvl),
            "shape": [int(K), int(D)],
            "alive": alive_count,
            "dead": int(K - alive_count),
            "norm_min": float(norms.min().item()),
            "norm_max": float(norms.max().item()),
            "norm_mean": float(norms.mean().item()),
            "top_pairs": [],
        }

        if alive_count <= 1:
            out_json["levels"].append(level_json)
            continue

        cb_alive = cb[alive]
        alive_idx = torch.nonzero(alive, as_tuple=False).squeeze(1)

        cb_alive_n = torch.nn.functional.normalize(cb_alive, dim=1)
        sim = cb_alive_n @ cb_alive_n.T

        eye = torch.eye(sim.size(0), dtype=torch.bool)
        off = sim[~eye]

        log(
            f"cos alive min={off.min().item():+.4f} "
            f"max={off.max().item():+.4f} "
            f"mean={off.mean().item():+.4f}"
        )

        level_json["cos_min"] = float(off.min().item())
        level_json["cos_max"] = float(off.max().item())
        level_json["cos_mean"] = float(off.mean().item())

        pairs = []

        for i in range(sim.size(0)):
            for j in range(i + 1, sim.size(0)):
                pairs.append((
                    abs(sim[i, j].item()),
                    sim[i, j].item(),
                    int(alive_idx[i]),
                    int(alive_idx[j]),
                ))

        pairs.sort(reverse=True, key=lambda z: z[0])

        for rank, (_, s, i, j) in enumerate(
            pairs[:print_topk_pairs],
            start=1,
        ):
            log(f"  {rank:2d}. codes ({i},{j}) cos={s:+.6f}")

            level_json["top_pairs"].append({
                "rank": int(rank),
                "code_i": int(i),
                "code_j": int(j),
                "cosine": float(s),
                "duplicate_like": bool(abs(s) >= duplicate_cos_thresh),
            })

        # ------------------------------------------------------------
        # Parent-wise child separation for hierarchical levels
        # ------------------------------------------------------------
        if lvl > 0:
            parent_child_json = []

            num_parent = int(model.vq.num_codes_per_level[0])
            num_child = int(model.vq.num_codes_per_level[lvl])

            if K == num_parent * num_child:
                cb_tree = cb.reshape(num_parent, num_child, D)
                norms_tree = norms.reshape(num_parent, num_child)
                alive_tree = alive.reshape(num_parent, num_child)

                log("\n[parent-wise child separation]")
                log(
                    "parent  alive_child  child_norm_mean  "
                    "cos_min  cos_max  cos_mean"
                )

                for p in range(num_parent):
                    child_alive = alive_tree[p]
                    n_alive_child = int(child_alive.sum())

                    if n_alive_child == 0:
                        parent_child_json.append({
                            "parent": int(p),
                            "alive_children": 0,
                            "child_norm_mean": 0.0,
                            "cos_min": None,
                            "cos_max": None,
                            "cos_mean": None,
                        })
                        continue

                    child_norm_mean = float(norms_tree[p, child_alive].mean().item())

                    if n_alive_child < 2:
                        log(
                            f"{p:6d}  {n_alive_child:11d}  "
                            f"{child_norm_mean:15.6g}  "
                            f"{'NA':>7}  {'NA':>7}  {'NA':>8}"
                        )

                        parent_child_json.append({
                            "parent": int(p),
                            "alive_children": int(n_alive_child),
                            "child_norm_mean": child_norm_mean,
                            "cos_min": None,
                            "cos_max": None,
                            "cos_mean": None,
                        })
                        continue

                    child = cb_tree[p, child_alive]
                    child_n = torch.nn.functional.normalize(child, dim=1)
                    child_sim = child_n @ child_n.T

                    child_eye = torch.eye(
                        child_sim.size(0),
                        dtype=torch.bool,
                    )
                    child_off = child_sim[~child_eye]

                    cmin = float(child_off.min().item())
                    cmax = float(child_off.max().item())
                    cmean = float(child_off.mean().item())

                    log(
                        f"{p:6d}  {n_alive_child:11d}  "
                        f"{child_norm_mean:15.6g}  "
                        f"{cmin:+7.3f}  {cmax:+7.3f}  {cmean:+8.3f}"
                    )

                    parent_child_json.append({
                        "parent": int(p),
                        "alive_children": int(n_alive_child),
                        "child_norm_mean": child_norm_mean,
                        "cos_min": cmin,
                        "cos_max": cmax,
                        "cos_mean": cmean,
                    })

                level_json["parent_child_separation"] = parent_child_json
            else:
                log(
                    "\n[parent-wise child separation skipped] "
                    f"K={K} does not match num_parent*num_child="
                    f"{num_parent * num_child}"
                )

        out_json["levels"].append(level_json)

    # ---------------- save txt ----------------

    if save_txt_path is not None:
        save_txt_path = Path(save_txt_path)
        save_txt_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_txt_path, "w") as f:
            f.write("\n".join(lines))

    # ---------------- save json ----------------

    if save_json_path is not None:
        save_json_path = Path(save_json_path)
        save_json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_json_path, "w") as f:
            json.dump(out_json, f, indent=2)


def run_stage_evaluation(
    stage,
    model,
    train_loader,
    test_loader,
    device,
    assay_indices,
):
    print("\n" + "=" * 80)
    print(f"STAGE {stage}: evaluation and visualization")
    print("=" * 80)

    # ============================================================
    # Stage 0
    # ============================================================
    if stage == 0:
        if not RUN_VIZ:
            print("Skipping Stage 0 visualization because RUN_VIZ=False.")
            return

        if model.spatial_map_prior is None:
            print(
                "Skipping Stage 0 evaluation/visualization: "
                "model has no spatial_map_prior."
            )
            return

        # Always evaluate the best saved Stage 0 checkpoint,
        # not the final in-memory early-stopping epoch.
        load_spatial_pretrain_if_available(model)

        out_dir = Path(
            "../viz_out_vqvae/pre_stage1_spatial_maps"
        )
        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        assay_codebook = (
            train_loader
            .dataset
            .dataset
            .assay_codebook
        )

        save_assaywise_spatial_maps(
            model,
            assay_indices=assay_indices,
            n_assays=num_assays_for_emb,
            out_dir=str(out_dir),
            assay_codebook=assay_codebook,
        )

        save_assaywise_adjacency_diagnostics(
            model,
            assay_indices=assay_indices,
            n_assays=num_assays_for_emb,
            out_dir=str(out_dir),
            assay_codebook=assay_codebook,
        )

        if (
            hasattr(model, "memory_adj")
            and model.memory_adj is not None
        ):
            model.memory_adj.debug_print(
                max_items=5
            )

        print(
            f"Saved Stage 0 visualizations to: "
            f"{out_dir.resolve()}"
        )
        return

    # ============================================================
    # Stage 1
    # ============================================================
    if stage == 1:
        evaluate_and_visualize(
            model,
            test_loader,
            stage=1,
            assay_indices=assay_indices,
            assay_codebook=(
                test_loader.dataset.dataset.assay_codebook
            ),
        )

        if RUN_CODEBOOK_DEBUG:
            out_root = VIZ_ROOTS[1]
            debug_vq_codebooks(
                model,
                save_txt_path=str(out_root / "codebook_debug.txt"),
                save_json_path=str(out_root / "codebook_debug.json"),
            )
        return

    # ============================================================
    # Stage 2 substages
    # ============================================================
    if stage == 2:
        eval_phases = _normalize_substage_phases(
            STAGE2_EVAL_PHASES,
            ("2a", "2b", "2c"),
            name="STAGE2_EVAL_PHASES",
        )
        if not eval_phases:
            print("STAGE2_EVAL_PHASES is empty; skipping Stage 2 evaluation.")
            return

        for phase in eval_phases:
            evaluated = evaluate_and_visualize(
                model,
                test_loader,
                stage=2,
                stage2_phase=phase,
                assay_indices=assay_indices,
                assay_codebook=(
                    test_loader.dataset.dataset.assay_codebook
                ),
            )
            if evaluated is None:
                continue

            if RUN_CODEBOOK_DEBUG:
                out_root = STAGE2_VIZ_ROOTS[phase]
                debug_vq_codebooks(
                    model,
                    save_txt_path=str(out_root / "codebook_debug.txt"),
                    save_json_path=str(out_root / "codebook_debug.json"),
                )
        return

    # ============================================================
    # Stage 3 substages
    # ============================================================
    if stage == 3:
        eval_phases = _normalize_substage_phases(
            STAGE3_EVAL_PHASES,
            ("3a", "3b", "3c"),
            name="STAGE3_EVAL_PHASES",
        )
        if not eval_phases:
            print("STAGE3_EVAL_PHASES is empty; skipping Stage 3 evaluation.")
            return

        if "3a" in eval_phases:
            try:
                evaluate_stage3a_predictive(
                    model,
                    test_loader,
                    device,
                )
            except FileNotFoundError as exc:
                if RUN_SKIP_MISSING_EVAL:
                    print(f"Skipping Stage 3A evaluation: {exc}")
                else:
                    raise

        for phase in ("3b", "3c"):
            if phase not in eval_phases:
                continue

            try:
                evaluate_stage3_activity_predictive(
                    model,
                    test_loader,
                    device,
                    phase=phase,
                )
            except FileNotFoundError as exc:
                if RUN_SKIP_MISSING_EVAL:
                    print(f"Skipping Stage {phase.upper()} evaluation: {exc}")
                    continue
                raise

            if not RUN_STAGE3_GENERATION_EVAL:
                continue

            phase_root = Path(
                "../viz_out_vqvae/vqvae_stage3"
            ) / phase

            # A. Held-out exact-context generation.
            evaluate_stage3_prior(
                model,
                test_loader,
                device,
                activity_phase=phase,
                out_dir=str(phase_root / "stage3_prior_test_ctx"),
                max_batches=20,
                samples_per_context=4,
                steps=12,
                temperature=1.0,
            )

            ref_batch = next(iter(test_loader))
            fixed_gctx = ref_batch["global_ctx"][0:1]
            fixed_assay_id = int(ref_batch["assay_idx"][0].item())

            evaluate_stage3_prior_sampled_contexts(
                model,
                test_loader,
                device,
                activity_phase=phase,
                out_dir=str(phase_root / "stage3_prior_random_full"),
                context_bank_path="ckpts/context_prior.pkl",
                mode="random_full",
                max_samples=64,
            )
            evaluate_stage3_prior_sampled_contexts(
                model,
                test_loader,
                device,
                activity_phase=phase,
                out_dir=str(phase_root / "stage3_prior_fixed_global"),
                context_bank_path="ckpts/context_prior.pkl",
                mode="fixed_global",
                fixed_global_ctx=fixed_gctx,
                assay_id=fixed_assay_id,
                max_samples=64,
            )
            evaluate_stage3_prior_sampled_contexts(
                model,
                test_loader,
                device,
                activity_phase=phase,
                out_dir=str(phase_root / "stage3_prior_partial_local"),
                context_bank_path="ckpts/context_prior.pkl",
                mode="partial_local",
                partial_local={
                    "log_mean_firing_density": -9.1,
                    "temporal_trend": 0.0,
                },
                max_samples=64,
            )
            evaluate_stage3_prior_sampled_contexts(
                model,
                test_loader,
                device,
                activity_phase=phase,
                out_dir=str(
                    phase_root / "stage3_prior_fixed_global_partial_local"
                ),
                context_bank_path="ckpts/context_prior.pkl",
                mode="fixed_global_partial_local",
                fixed_global_ctx=fixed_gctx,
                assay_id=fixed_assay_id,
                partial_local={
                    "log_mean_firing_density": -9.1,
                    "temporal_trend": 0.0,
                },
                max_samples=64,
            )

        return

    raise ValueError(
        f"Unsupported EVAL_STAGE={stage}"
    )



# =============================================================================
# Main
# =============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    assay_dict = find_assays()
    assay_indices = list(assay_dict.keys())

    train_loader, val_loader, test_loader, meta = make_loaders(
        assay_dict=assay_dict,
        assay_indices=assay_indices,
        per_assay_quota=per_assay_quota_stage12,
    )

    batch0 = next(iter(train_loader))
    x0 = batch0["x"]
    _, _, T0, H0, W0 = x0.shape
    img_size = (T0, H0, W0)

    full_hw0 = tuple(map(int, batch0["full_hw"][0]))
    meta["img_size"] = img_size
    meta["full_spatial_size"] = full_hw0

    baseline_prob = compute_p0_from_loader(train_loader, max_batches=100)
    meta["spike_voxel_prob"] = baseline_prob
    baseline_prob_clip = float(np.clip(baseline_prob, 1e-8, 1.0 - 1e-8))
    logit_baseline = float(np.log(baseline_prob_clip / (1.0 - baseline_prob_clip)))

    print("meta =", meta)
    print("baseline_prob =", baseline_prob, "logit_baseline =", logit_baseline)

    isi_target_gap_rates = compute_short_gap_target_rates_from_loader(
        train_loader,
        gap_bins=gap_bins,
        max_batches=None,
    )

    print("isi_target_gap_rates:", isi_target_gap_rates.tolist())

    # Build once. Cross-attn layer set is controlled by freeze_for_stage().
    model = make_vqvae(
        img_size=img_size,
        device=device,
        full_spatial_size=full_hw0,
        use_decoder_cross_attn=False,
        decoder_cross_attn_layers=(),
    )

    # ============================================================
    # Sequential training followed immediately by stage evaluation
    # ============================================================
    
    evaluated_stages = set()
    
    # When Stage 0 is not being retrained, load its saved state before
    # any downstream VQVAE/prior training begins.
    needs_spatial_pretrain = (
        any(
            stage in TRAIN_STAGES
            for stage in (1, 2, 3)
        )
        or any(
            stage in EVAL_STAGES
            for stage in (0, 1, 2, 3)
        )
    )
    
    if 0 not in TRAIN_STAGES and needs_spatial_pretrain:
        load_spatial_pretrain_if_available(model)
    
    
    for stage in TRAIN_STAGES:
        # ========================================================
        # Train one stage
        # ========================================================
    
        if stage == 0:
            run_stage0_spatial_pretrain(
                model,
                train_loader,
                device,
            )
    
            # Restore the best Stage 0 checkpoint rather than using
            # the final early-stopping epoch.
            load_spatial_pretrain_if_available(model)
    
            # Build the retrieval bank using the best Stage 0
            # global-context embedding.
            build_context_prior(
                train_loader,
                save_path="ckpts/context_prior.pkl",
                model=model,
                device=device,
                feature_names=ACTIVITY_CTX_NAMES,
            )
    
        elif stage == 1:
            run_stage1(
                model,
                train_loader,
                val_loader,
                blank_logit_threshold=1.05*logit_baseline
            )
    
        elif stage == 2:
            run_stage2(
                model,
                train_loader,
                val_loader,
                blank_logit_threshold=1.05*logit_baseline
            )
    
        elif stage == 3:
            run_stage3_prior(
                model,
                train_loader,
                val_loader,
                device,
            )
    
        else:
            raise ValueError(
                f"Unsupported TRAIN_STAGE={stage}"
            )
    
        # ========================================================
        # Immediately evaluate the stage that just finished
        # ========================================================
    
        if stage in EVAL_STAGES:
            run_stage_evaluation(
                stage=stage,
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                assay_indices=assay_indices,
            )
    
            evaluated_stages.add(stage)
    
    
    # ============================================================
    # Evaluation-only stages
    # ============================================================
    #
    # This preserves configurations such as:
    #
    # TRAIN_STAGES = ()
    # EVAL_STAGES = (1, 2)
    #
    # or:
    #
    # TRAIN_STAGES = (2,)
    # EVAL_STAGES = (1, 2)
    #
    for stage in EVAL_STAGES:
        if stage in evaluated_stages:
            continue
    
        run_stage_evaluation(
            stage=stage,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            assay_indices=assay_indices,
        )

    # ============================================================
    # Combined Stage 1 → Stage 2 plots/reports
    # ============================================================
    
    stage2_report_for_plots = (
        REPORTS["stage2"]
        if REPORTS["stage2"].exists()
        else REPORTS["stage2a"]
    )

    if REPORTS["stage1"].exists() and stage2_report_for_plots.exists():
        plot_base_then_finetune(
            report_base_path=str(REPORTS["stage1"]),
            report_ft_path=str(stage2_report_for_plots),
            out_dir="../viz_out_vqvae/plots_stage1_stage2_overlays",
            mode="shared",                 # use "union" if you want every metric possible
            truncate_base_at_best=False,    # shows full stage 1 curve
            save_pdf=True,
            base_label="Stage 1: context-agnostic VQVAE",
            ft_label="Stage 2: context-conditioned decoder",
            base_eval_roots=[str(VIZ_ROOTS[1])],
            ft_eval_roots=[str(VIZ_ROOTS[2])],
        )
    
        export_base_finetune_flat_xlsx(
            report_base_path=str(REPORTS["stage1"]),
            report_ft_path=str(stage2_report_for_plots),
            out_xlsx="reports/training_report_stage1_stage2_flat.xlsx",
            shift_finetune_by="best",
        )
        
        export_viz_quant_tables(
            eval_roots=[str(VIZ_ROOTS[1]), str(VIZ_ROOTS[2])],
            stage_names=["stage1", "stage2"],
            out_dir="../viz_out_vqvae/quant_tables_ctx_adj",
        )

if __name__ == "__main__":
    main()