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
from .model import TransformerVQVAE, MaskGITActivityPrior, MaskGITMotifPrior, HierarchicalCodebookPrior
from .model.base import CtxEmbed, LctMapper
from .model.prior import (
    build_activity_targets_from_codes,
    maskgit_activity_loss,
    infer_activity_coordinate_mode_from_state_dict,
)
from .model.spatial_map import GlobalContextSpatialBank, GlobalContextAdjacencyBank
from .training import (
    fit_vqvae, evaluate_vqvae, fit_spatial_prior_pretrain, build_context_prior,
    build_null_baselines, load_null_baselines, predict_count_from_density,
    build_motif_null_baselines, load_motif_null_baselines, motif_null_predictions,
    train_motif_prior_mgit,
    train_maskgit_activity_prior,
    train_activity_prior_with_frozen_motif,
    configure_stage4c_event_calibration,
)
from .training.stage4_activity import token_frequency_for_batch
from .training.stage2b_flatten import run_stage2b_flatten
from .training.stage3_lct import run_stage3_lct
from .training.train_prior import (
    _batch_to_device,
    _make_activity_in_from_codes,
    _vq_codes_and_pmask_for_prior,
    _vq_codes_and_pmask,
    distance_neighborhood_ce_loss,
    expected_code_distance_loss,
)
from .inference import (
    ContextBankSampler,
    decode_codes_to_xgen,
    decode_flat_ids_to_xgen,
    generate_rate_surrogate,
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

from .utils.losses import local_moment_field_loss
from .utils.constants import (
    ACTIVITY_CTX_NAMES,
    DEFAULT_GAP_BINS,
    normalize_gap_bins,
    max_gap_from_bins,
)

# =============================================================================
# User configuration
# =============================================================================

# Pipeline stages. Run any subset sequentially.
#
#   1   gct mapper      -- CtxEmbed + spatial_map_prior pretraining
#   2   VQ-VAE          -- 2a hierarchical training, 2b flatten + dedupe
#   3   lct mapper      -- standalone local-context trunk (needs 2b)
#   4   prior           -- MaskGIT, phases 4a/4b/4c (needs 1, 2b, 3)
#
# Examples:
#   (2,)          -> train the VQ-VAE and flatten its codebook
#   (1, 2, 3, 4)  -> run the whole pipeline sequentially
TRAIN_STAGES = (4,)        # 1,2,3,4
EVAL_STAGES  = ()          # 1,2,3,4

# Stage-2 continuous quantization.
#
# Stage 2A:
#   project the frozen encoder residual onto the convex hull of the selected
#   z1 parent's frozen z2 children; train the decoder on that fixed geometry.
# Stage 2B:
#   train the optional alpha adapter to reproduce the same geometric projection.
#   The decoder is frozen and does not define the adapter target.
# Stage 2B:
#   freeze the continuous mapping and decoder backbone; train only zero-gated
#   decoder cross-attention and context projections.
# Stage-1 training substages. Any subset of ("1a", "1b") is valid; execution
# order is always 1A -> 1B.
# Stage 1A:
#   joint context-agnostic training of encoder/to_code/VQ/decoder. The discrete
#   hierarchy (z1 parents and z2 children) is learned here. This is the stage the
#   z1-level results characterise: codebook perplexity, eta^2(code->activity),
#   motif syntax, and the causal code->context response.
# Stage 1B:
#   refit ONLY the z2 children against the frozen encoder and z1, at a new
#   children-per-parent count. Residuals depend solely on the encoder and z1, so
#   this is well posed and leaves every z1-level Stage 1A result intact. Used to
#   widen the child hull without paying for a full Stage 1A retrain.
STAGE2_PHASES = ("2a", "2b")

# ---- Stage 3: standalone lct mapper ----
STAGE3_EPOCHS = 300
STAGE3_BATCHES_PER_EPOCH = 120
STAGE3_NUM_TEXTONS = 128
# Descriptor space, not decoder space: z1 carries reusable motif identity while
# z2/z3 carry exactness detail lct cannot predict. Measured z1 > z12 > flat.
STAGE3_TEXTON_BASIS = "z1"


# Independently evaluate any saved Stage-2 substages. Evaluation order is
# always 2A -> 2B, regardless of tuple order.


# Stage-4 training substages. Any subset of ("4a", "4b", "4c") is valid;
# execution order remains 3A -> 3B -> 3C. Missing prerequisites are loaded
# from their best checkpoints.
STAGE4_PHASES = ("4a",)
# Warm-start 3A from ckpts/motif_prior_warmstart.pt when present. The z1 trunk
# from the previous run reached 0.288 top-1; retraining from scratch would spend
# ~400 epochs re-earning it. strict=False because the alpha head is now a
# Dirichlet concentration head (same shape, new interpretation).
# The only warm-start checkpoint on disk predates the flat alphabet, so this
# must stay off until a flat-alphabet 4A run produces one.
STAGE4A_WARM_START = False
# Literal path: CKPT_DIR is defined further down in this config block.
STAGE4A_WARM_START_PATH = Path("ckpts") / "motif_prior_warmstart.pt"
STAGE4A_EPOCHS = 600
STAGE4B_EPOCHS = 200
STAGE4C_EPOCHS = 100

# Stage 4B/3C activity-coordinate parameterization. Use "factorized" for the
# original axis-head ablation or "joint_dense" for one categorical THW head.
STAGE4_COORDINATE_MODE = "joint_dense"

# Stage 4B activity-prior architecture.
#   "sparse_region" : SparseRegionActivityPrior. Visible-active tokens enter a
#                     sparse-memory encoder (no mean pool), Kmax queries
#                     cross-attend to that memory with region anchors, and the
#                     coordinate head factorizes as p(region) * p(token|region).
# Activity placement is scored exactly. In token space a +/-1-token tolerance is
# not a near miss: one token is 6 frames x 15 rows x 14 cols, the dilated target
# covers ~49% of the grid, and a uniform ROI draw scores ~0.49 for free.
STAGE4_HARD_TOLERANCE = (0, 0, 0)

STAGE4B_REGION_GRID = None        # None = derive from the token grid (region extent ~4
                                  # tokens/axis). (8,8,16) -> (2,2,4) = 16 regions of 64,
                                  # identical to the hand-picked value. Pin a tuple only to
                                  # override; it must tile the token grid exactly.
STAGE4B_DECODER_LAYERS = 4
# Ablation: False replaces the region/within factorization with a flat Ntok-way
# grid head, isolating the region head from the sparse-memory encoder.
STAGE4B_REGION_FACTORIZATION = True
# Per-assay adjacency target for the 3B structural loss, keyed by global_ctx --
# the same statistic Stage 0 supervised the global embedder on.
STAGE4B_USE_TOKEN_ADJ_BANK = False
STAGE4B_TOKEN_ADJ_VARIANT = "token"

# Depth of the per-cell readout decoder.
STAGE4B_MASKGIT_LAYERS = 3
STAGE4B_MASKGIT_HYPERPARAMETERS = {
    "lr": 3e-4,
    "epochs": 120,
    "lambda_bce": 1.0,
    # 1.0, not the class-balancing ratio. Up-weighting the ~5% positives helps
    # F1/AUPRC (both rank-based, blind to it) and destroys calibration, which is
    # the property a sampled prior actually needs.
    "pos_weight": 1.0,
    "lambda_count": 1.0,
    "lambda_adj_t": 0.10,
    "lambda_adj_s": 0.10,
    # loss_spatial lands near 8.0 against ~1.6 for BCE, so its weight is scaled
    # to make the contribution comparable rather than dominant.
    "lambda_spatial": 0.02,
    # Fraction of samples given a scattered random-ratio token mask instead of
    # the structured recon/causal/noncausal/spatial mask. Rounds 2+ of MaskGIT
    # decoding produce exactly this distribution and the shipped specs never do.
    "random_mask_prob": 0.5,
    "random_mask_ratio": (0.15, 1.0),
    # NLL, not F1: F1 is maximized by emitting the mode.
    "select_on": "nll",
    "save_start_epoch": 10,
    "early_stop_patience": 30,
}
# MaskGIT decoding schedule for sampling from the dense prior.
# Tuned by sweep: total |error| across rate, temporal persistence lags 1-7 and
# spatial co-activation fell 1.31 -> 0.18 going from (1.0, 1.0, 10) to these, and
# the mean-field baseline sits at 0.73. Greedy decoding over-produces persistence
# (0.77 vs 0.61 real at lag 1) because a committed cell raises its temporal
# neighbours and zero-noise late rounds then take them deterministically; raising
# temperature and holding noise longer fixes it.
# Selected on VALIDATION (reports/evaluation_report_3B_sampler_sweep_VAL.json);
# test is measured once at this setting and never used for selection.
STAGE4B_SAMPLE_STEPS = 10
STAGE4B_SAMPLE_TEMPERATURE = 1.5
STAGE4B_SAMPLE_GUMBEL = 4.0

# Stage 4B is selected by a deterministic, hard expected-count top-K metric.
STAGE4B_HYPERPARAMETERS = {
    "lr": 2e-4,
    "lambda_count": 1.0,
    "lambda_count_neighbor": 0.25,
    "lambda_count_distance": 0.05,
    # Token-space structural terms (see maskgit_activity_loss).
    "lambda_adj_t": 0.0,
    "lambda_adj_s": 0.0,
    # Coordinate-neighbourhood partial credit, added to (not replacing) the
    # exact-cell CE. Temporal-only by default: one token spatially is 15 rows
    # x 14 cols of electrodes, which is a different site, not a near miss.
    "deterministic_validation_masks": True,
    "hard_activity_mode": "expected-count unique grid top-K",
    # Early stopping must not fire during the count-teacher curriculum: the
    # handoff degrades validation by construction, so a patience shorter than
    # count_teacher_epochs + count_transition_epochs kills the run mid-transition
    # before it ever trains teacher-free.
    "early_stop_patience": 30,
    "early_stop_start_epoch": 60,
    # No checkpoint is saved or tracked until the count handoff completes at
    # count_teacher_epochs + count_transition_epochs. A teacher-forced model is
    # solving an easier problem, so its score must not set the selection bar.
    "save_start_epoch": 60,
}

# Stage 4C is a low-LR event-placement calibration, not a second activity-prior
# training stage. Its true-generation validation is deliberately limited to a
# fixed subset because every validation pass runs iterative MaskGIT + decoding.
STAGE4C_HYPERPARAMETERS = {
    "lr": 1e-5,
    "lambda_activity": 1.0,
    "lambda_count": 1.0,
    "lambda_count_neighbor": 0.25,
    "lambda_count_distance": 0.05,
    "lambda_ctx": 0.25,
    "lambda_ctx_field": 0.05,
    "lambda_adj": 0.25,
    "lambda_spatial": 0.25,
    "auxiliary_ramp_epochs": 10,
    # No candidate accepted or tracked until the auxiliary losses finish ramping.
    "save_start_epoch": 10,
    # Raised from a hard-coded 20. Stage 4C moves 1.2% of parameters at 1e-5, so
    # twenty epochs is very little actual movement and "no improvement" is weaker
    # evidence of convergence here than the same count would be in Stage 4B.
    "early_stop_patience": 45,
    "generation_val_max_batches": 4,
    "generation_motif_steps": 12,
    "deterministic_validation_masks": True,
    "hard_activity_mode": "expected-count unique grid top-K straight-through",
}

# Reuse saved Kmax metadata when possible so independently rerun substages use
# the same activity-head shape. Set True only when the data/token grid changed.
STAGE4_RECOMPUTE_KMAX = False
STAGE4_KMAX_PASSES = 5
STAGE4_KMAX_MARGIN = 1.25

# Independently evaluate any saved Stage-4 substages. Evaluation order is
# always 3A -> 3B -> 3C, regardless of tuple order.
#
# 3A: held-out teacher-forced activity / masked motif prediction metrics.
# 3B: held-out activity-count and coordinate metrics.
# 3C: held-out refined activity metrics.
STAGE4_EVAL_PHASES = ("4c",)

# Stable Stage-3A evaluation starts from a fully masked motif ROI. Set this to
# 0.15 to reproduce the mixed masking regime used during training validation.
STAGE4A_EVAL_FULL_MASK_PROB = 1.0

# For selected 3B/3C evaluation phases, also run the expensive decoded
# generation evaluations. Set False to evaluate only the prior heads.
RUN_STAGE4_GENERATION_EVAL = True

# --- Null-baseline / leakage-audit entry points (opt-in; default path unchanged) ---
#
# Recommended order:
#   1. RUN_BUILD_NULL_BASELINES   (one pass over train, ~10 min)
#   2. RUN_LEAKAGE_DELTA_EVAL     (existing ckpts on the new temporal test split)
#   3. RUN_COUNT_NULL_EVAL + RUN_GENERATION_BASELINE_EVAL
#   4. only then retrain on the clean split
RUN_BUILD_NULL_BASELINES = False
RUN_LEAKAGE_DELTA_EVAL = False
RUN_COUNT_NULL_EVAL = False
RUN_GENERATION_BASELINE_EVAL = False
RUN_MOTIF_NULL_EVAL = False

NULL_BASELINE_PATH = Path("ckpts/null_baselines.pkl")

# Populated in main() when the file exists. Stage 4B/3C read this so their
# hard-activity reports carry matched-null reference scores.
_NULL_BASELINES = None

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
    # --- Stage 2A: hierarchical VQ-VAE ---
    "stage2a_best": CKPT_DIR / "vqvae_stage2a_best.pt",
    "stage2a_last": CKPT_DIR / "vqvae_stage2a_last.pt",

    # --- Stage 2B: flattened + deduped codebook ---
    "stage2b_flat": CKPT_DIR / "stage2b_flat_codebook.pt",

    # --- Stage 3: standalone lct mapper ---
    "stage3_lct": CKPT_DIR / "stage3_lct_mapper.pt",

    # --- Stage 4: prior ---
    "motif_prior_best": CKPT_DIR / "motif_prior_best.pt",
    # Legacy alias retained for external scripts. New Stage 4B runs write this
    # alias from the generation-aligned hard-metric checkpoint.
    "activity_prior_best": CKPT_DIR / "activity_prior_best.pt",
    "activity_prior_best_loss": CKPT_DIR / "activity_prior_best_loss.pt",
    "activity_prior_best_hard_metric": CKPT_DIR / "activity_prior_best_hard_metric.pt",
    "activity_prior_refined_best": CKPT_DIR / "activity_prior_refined_best.pt",

}

REPORTS = {
    "stage1_gct": Path("reports/training_report_stage1_gct.json"),
    "stage2a": Path("reports/training_report_vqvae_stage2a.json"),
    "stage2a_eval": Path("reports/evaluation_report_vqvae_stage2a.json"),
    "stage2b": Path("reports/analysis_stage2b_flatten.json"),
    "stage3_lct": Path("reports/analysis_stage3_lct.json"),

    "prior_motif": Path("reports/training_report_prior_3A_motif.json"),
    "prior_motif_eval": Path("reports/evaluation_report_prior_3A_motif.json"),
    "prior_activity_eval": Path("reports/evaluation_report_prior_3B_activity.json"),
    "prior_refine_eval": Path("reports/evaluation_report_prior_3C_refine.json"),
    "prior_activity": Path("reports/training_report_prior_3B_activity.json"),
    "prior_refine": Path("reports/training_report_prior_3C_refine.json"),

    "leakage_delta": Path("reports/evaluation_report_leakage_delta.json"),
    "count_nulls": Path("reports/evaluation_report_count_nulls.json"),
    "generation_baselines": Path("reports/evaluation_report_generation_baselines.json"),
    "motif_nulls": Path("reports/evaluation_report_motif_nulls.json"),
}

MOTIF_NULL_BASELINE_PATH = Path("ckpts/motif_null_baselines.pkl")

VIZ_ROOTS = {
    2: Path("../viz_out_vqvae/vqvae_stage2a"),
}


# Data
patch_size = (6, 15, 14)
temporal_crop = 6000
temporal_pool = 120
batch_size = 4
grad_accum_steps = 8
num_workers = 2
# (train, val, test) per-assay sampling quotas. 31 assays -> 930/186/279,
# matching the item counts of every previously generated report so the
# temporal-split numbers stay directly comparable to the old random-split ones.
per_assay_quota_stage12 = (30, 6, 9)

cache_dir = "../_cache_spike_thw_run1"
cache_mode = "uint8"
cache_max_gb = 80
cache_write_prob = 1.0

# Model
#
# THREE codebook levels with a full tolerance ladder:
#   z1 (1,1,1)  coarse
#   z2 (0,1,1)  spatial-only tolerance
#   z3 (0,0,0)  exact
#
# The levels exist to PRODUCE centroids under the ladder. Stage 2B sums them
# into one flat entry per token, so the deliverable is a single discrete
# codebook, not a hierarchy at inference time.
#
# Alphabet 32*8*4 = 1024 nominal (935 observed after dedupe). Storage is
# 32 + 256 + 1024 = 1312 vectors; the binding constraint is occupancy, not
# memory. Watch the vq_l3 dead fractions: if the tail dies, drop to (16,8,4)
# or (32,4,4) rather than raising K3.
#
# These are the values every shipped checkpoint was trained with. They used to
# live only in a scratch config, which meant a plain `python main.py` built a
# two-level model that could not load its own checkpoints.
num_codes = (32, 8, 4)
# Number of VQ levels. Must equal len(num_codes).
num_quantizers = 3



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


def make_loaders(assay_dict: dict, assay_indices: list[int], per_assay_quota):
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



def make_vqvae(img_size, device: str, *, full_spatial_size=None):
    model = TransformerVQVAE(
        img_size=img_size,
        full_spatial_size=full_spatial_size,
        patch_size=patch_size,

        encoder_embed_dim=64,
        encoder_depth=2,
        encoder_num_heads=4,
        code_dim=64,
        num_codes=num_codes,
        num_quantizers=num_quantizers,
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

        # Element-wise context dropout off: on a low-rank control signal it is
        # noise that teaches the decoder to ignore context.  Whole-token CFG
        # dropout is the intended mechanism.
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


def freeze_for_stage(model: nn.Module, stage: int):
    """
    Centralized stage policy.

    Stage 1: gct mapper.
      - Train global_embedder + spatial_map_prior only.
      - The rest of the VQ-VAE is untouched.

    Stage 2: hierarchical VQ-VAE (2a; 2b is a deterministic flatten, no grads).
      - Train stem/encoder/to_code/VQ/decoder.
      - Freeze the Stage-1 gct pair.

    Stage 3: lct mapper.
      - Nothing in the VQ-VAE trains; the standalone trunk owns its own
        optimizer. The model is frozen and in eval so the encoder pass that
        produces the targets is deterministic.

    Stage 4: prior learning.
      - Freeze the VQ-VAE completely.
    """
    set_all_trainable(model, False)

    if stage == 1:
        set_requires_grad(getattr(model, "global_embedder", None), True)
        set_requires_grad(getattr(model, "spatial_map_prior", None), True)

    elif stage == 2:
        for name in ["stem", "patch_embed", "sparse_encoder", "to_code", "vq",
                     "code_to_dec", "dec_blocks", "dec_norm", "patch_renderer"]:
            set_requires_grad(getattr(model, name, None), True)

        if hasattr(model, "activity_type_offset"):
            model.activity_type_offset.requires_grad = True

        # The Stage-1 gct pair stays frozen here.
        set_requires_grad(getattr(model, "global_embedder", None), False)
        set_requires_grad(getattr(model, "spatial_map_prior", None), False)

        model.vq.freeze_codebook_updates = False

        # EMA codebook entries are updated manually, not by AdamW.
        # Keep blank_token trainable; freeze only hierarchical codebook tensors.
        if hasattr(model, "vq") and hasattr(model.vq, "tree_embeds"):
            for p in model.vq.tree_embeds:
                p.requires_grad = False

    elif stage in (3, 4):
        set_all_trainable(model, False)
        model.vq.freeze_codebook_updates = True
        model.vq.dead_code_restart_every = 0
        model.vq.duplicate_restart_every = 0
        model.eval()

    else:
        raise ValueError(f"Unsupported stage={stage}")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Stage {stage}: trainable params = {n_trainable:,} / {n_total:,}")


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


def load_stage1_gct_for_eval(model):
    if not SPATIAL_CKPT.exists():
        raise FileNotFoundError(
            f"Stage 4 global metrics require {SPATIAL_CKPT}"
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

def run_stage1_gct_pretrain(model, train_loader, device):
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

        # STAGED ACTIVATION, not merely staged loss weights.
        #
        # These are the values the shipped 2A checkpoint was trained with. The
        # defaults used to be start_epoch=1 for both, which assigns all three
        # quantizers from epoch 1: z3 then quantizes a residual whose upstream
        # levels are still moving. That arm is on disk as
        # reports/stage1a_3level_simultaneous_epochs.jsonl and reached
        # AUPRC_tol 0.04499 at epoch 134, against 0.15244 for the staged run at
        # the same epoch -- a 3.4x gap from activation timing alone, since both
        # arms moved the loss weights on the identical 20/35/50/65 schedule.
        #
        # Each level gets 15 epochs to populate via EMA between ACTIVATION and
        # carrying full loss weight, so it is never asked to bear the objective
        # from randomly initialised codes:
        #   1-20   z1 only     [1.0]
        #   20-35  z1,z2       [1.0, 0.5]
        #   35-50  z1,z2       [0.75, 1.0]
        #   50-65  z1,z2,z3    [0.75, 1.0, 0.5]
        #   65+    z1,z2,z3    [0.5, 0.75, 1.0]
        level2_start_epoch=20,
        level2_full_loss_epoch=35,
        level3_start_epoch=50,
        level3_full_loss_epoch=65,
        lambda_sp_token=1e-3,
        lambda_sp_pixel=1e-4,
        sp_pixel_start_epoch=40,
        sp_pixel_warmup_epochs=60,
    )

def select_ckpt(stage: int = 2, prefer_best: bool = True) -> Path:
    if stage != 2:
        raise ValueError(f"No VQVAE checkpoint defined for stage={stage}")
    best = CKPTS["stage2a_best"]
    last = CKPTS["stage2a_last"]

    if prefer_best and best.exists():
        return best
    if last.exists():
        return last
    if best.exists():
        return best
    raise FileNotFoundError(f"No checkpoint found for stage {stage}: {best} or {last}")


def _normalize_substage_phases(phases, allowed, *, name):
    normalized = tuple(str(phase).lower() for phase in phases)
    invalid = [phase for phase in normalized if phase not in allowed]
    if invalid:
        raise ValueError(f"Unsupported {name} entries: {invalid}; allowed={allowed}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} contains duplicates: {normalized}")
    return tuple(phase for phase in allowed if phase in normalized)


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
   
    
# Stage 1 context-regularizer balance.
#
# Measured on ckpts/vqvae_stage1_best.pt, the unweighted ctx loss was 73.3%
# log_mean_firing_density and 23.0% temporal_trend; the five spatial and
# covariance moments together contributed ~0.3%.  They were supervised in name
# only, and ctx_loss_soft backprops into the encoder and codebook in Stage 1,
# so the learned motifs carry that bias.
#
# STAGE2_BALANCED_CTX turns on 1/var per-dim weighting.  lambda is rescaled by
# the measured weighted/unweighted ratio (0.036) so the regularizer keeps the
# same TOTAL share of the objective (~0.68%) and only its internal
# distribution changes -- otherwise the run would confound two edits.
STAGE2_BALANCED_CTX = True
STAGE2_LAMBDA_CTX = 1e-1
STAGE2_LAMBDA_CTX_BALANCED = 2.80

# Epochs actually run.  STAGE2_SCHED_T_MAX stays at the full 300 so a
# truncated probe follows the SAME cosine LR trajectory as the stored
# 300-epoch run -- shortening T_max would compress the schedule and make
# matched-epoch comparison meaningless.
STAGE2_EPOCHS = 300
STAGE2_SCHED_T_MAX = 300
STAGE2_SAVE_START_EPOCH = 125


def run_stage2a(model, train_loader, val_loader, blank_logit_threshold):
    stage2_ctx_weights = None
    if STAGE2_BALANCED_CTX:
        _, _, _norm = build_local_ctx_bank(train_loader)
        stage2_ctx_weights = (
            1.0 / _norm[1].pow(2).clamp_min(1e-12)
        ).to(next(model.parameters()).device)
        globals()["STAGE2_LAMBDA_CTX"] = STAGE2_LAMBDA_CTX_BALANCED
        print(
            f"Stage 2A balanced ctx weighting ON, lambda_ctx="
            f"{STAGE2_LAMBDA_CTX_BALANCED} (rescaled from 0.1 by the measured "
            f"weighted/unweighted ratio 0.036)"
        )

    print("\n" + "=" * 80)
    print("STAGE 2A: hierarchical VQ-VAE, context-agnostic motif learning")
    print("=" * 80)

    freeze_for_stage(model, 2)

    # Fixed surrogate boundary for all threshold-aware training losses.
    # Validation Best-F1 thresholds are recorded but never fed back.
    model._set_training_prob_threshold(0.5)

    n_epoch = int(STAGE2_EPOCHS)

    optimizer = make_optimizer(model, lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(STAGE2_SCHED_T_MAX), eta_min=1e-5
    )

    report = fit_vqvae(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        epochs=n_epoch,
        ckpt_best_path=str(CKPTS["stage2a_best"]),
        ckpt_last_path=str(CKPTS["stage2a_last"]),
        early_stop_patience=40,
        # Select on EXACT AUPRC, not tolerant. The tolerant metric's 27-voxel
        # slack rewards a diffuse field: the 3-level model measured exact/tol
        # 0.909 against the 2-level reference's 0.737, i.e. it is the sharper
        # model and was being ranked on the axis that penalises sharpness.
        val_metric_name="AUPRC",
        val_metric_goal="max",
        use_ROI_mask=False,

        # Stage 1: context loss ON, decoder context injection OFF.
        # This lets ctx losses shape encoder/codebook/decoder motifs,
        # without allowing cross-attention shortcuts.
        lambda_ctx=float(STAGE2_LAMBDA_CTX),
        ctx_dim_weights=stage2_ctx_weights,
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

        save_start_epoch = int(STAGE2_SAVE_START_EPOCH),
        **common_fit_kwargs(model),
    )

    with open(REPORTS["stage2a"], "w") as f:
        json.dump(report, f, indent=4)
    print(f"Saved report: {REPORTS['stage1']}")
    return report


def build_local_ctx_bank(loader, max_batches=None):
    """Bank of observed (global, local) context pairs, keyed by assay.

    Mirrors inference/sample_context.py: local contexts live in pools keyed by
    global context, and a request is either a local drawn from a chosen
    assay's pool or a whole (global, local) pair drawn together.  Training the
    context head against requests drawn the same way keeps every
    counterfactual jointly plausible without modelling the structure of the
    nine dimensions explicitly.

    Returns:
      bank:  {assay_id: {"local": (n_a, L), "global": (G,)}}
      std:   (L,) within-assay per-dimension std, used as the fallback jitter
             scale for assays with fewer than two entries and as the unit for
             reporting counterfactual displacement.
      norm:  (mean, scale) pooled over the whole training split, used to
             standardize the embedder's input.

    Built from the training loader only.
    """
    local_rows = {}
    global_row = {}
    for i, batch in enumerate(loader):
        lct = batch.get("local_ctx", None)
        gct = batch.get("global_ctx", None)
        aidx = batch.get("assay_idx", None)
        if lct is None:
            return None, None, None

        lct = lct.detach().float().cpu()
        gct = None if gct is None else gct.detach().float().cpu()

        if aidx is None:
            local_rows.setdefault(-1, []).append(lct)
            if gct is not None:
                global_row.setdefault(-1, gct[0])
        else:
            aidx = torch.as_tensor(aidx).reshape(-1)
            for a in aidx.unique():
                key = int(a)
                sel = aidx == a
                local_rows.setdefault(key, []).append(lct[sel])
                if gct is not None and key not in global_row:
                    # global_ctx is the assay codebook row: constant per assay.
                    global_row[key] = gct[sel][0]

        if max_batches is not None and (i + 1) >= max_batches:
            break

    if not local_rows:
        return None, None, None

    bank = {}
    for key, rows in local_rows.items():
        bank[key] = {
            "local": torch.cat(rows, 0),
            "global": global_row.get(key, None),
        }
    bank = {k: v for k, v in bank.items() if v["global"] is not None}
    if not bank:
        return None, None, None

    all_rows = torch.cat([v["local"] for v in bank.values()], 0)
    pooled_std = all_rows.std(dim=0, unbiased=False)

    var_sum = torch.zeros(all_rows.shape[1])
    weight = 0.0
    servable = 0
    for entry in bank.values():
        rows = entry["local"]
        if rows.shape[0] < 2:
            continue
        servable += 1
        var_sum += rows.var(dim=0, unbiased=True) * (rows.shape[0] - 1)
        weight += rows.shape[0] - 1

    if weight <= 0:
        print("local_ctx: no assay has >1 sample; falling back to pooled std.")
        std = pooled_std
    else:
        std = (var_sum / weight).sqrt()

    std = std.clamp_min(0.01 * float(std.max().clamp_min(1e-6)))

    sizes = sorted(v["local"].shape[0] for v in bank.values())
    print(
        f"context bank: {all_rows.shape[0]} (global, local) pairs over "
        f"{len(bank)} assays ({servable} servable, pool sizes min={sizes[0]} "
        f"median={sizes[len(sizes)//2]} max={sizes[-1]})"
    )
    print("  within-assay std: " + ", ".join(f"{v:.4f}" for v in std.tolist()))
    print("  pooled std:       " + ", ".join(f"{v:.4f}" for v in pooled_std.tolist()))
    print(
        "  within/pooled:    "
        + ", ".join(f"{v:.2f}" for v in (std / pooled_std.clamp_min(1e-8)).tolist())
        + "   (low = assay identity dominates that dimension)"
    )
    pooled_mean = all_rows.mean(dim=0)
    return bank, std, (pooled_mean, pooled_std.clamp_min(1e-6))


def measure_stage4_active_token_counts(
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
            f"Stage 4 Kmax estimation pass "
            f"{pass_idx + 1}/{num_passes}: "
            f"n={pass_counts_np.size}, "
            f"mean={pass_counts_np.mean():.2f}, "
            f"maximum={pass_counts_np.max()}"
        )

    if not all_counts:
        raise RuntimeError(
            "Cannot determine Stage 4 Kmax: "
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
        "Stage 4 active-token count statistics:"
    )
    print(
        json.dumps(
            stats,
            indent=2,
        )
    )

    return stats


def build_context_mappers(model, device):
    """Rebuild the two context mappers as standalone frozen modules.

    Neither lives inside the VQ-VAE any more:

      gct  -- CtxEmbed, half of the Stage-1 gct pair. The VQ-VAE still owns a
              copy for spatial_map_prior, but SPATIAL_CKPT is the source of
              truth, so the prior loads from disk rather than from whatever
              stage happens to be resident in `model`.
      lct  -- LctMapper, a Stage-4 artifact trained with the encoder frozen.
              It was never part of the VQ-VAE.

    Both are returned in eval() with requires_grad False: the prefix must be a
    fixed function of the request, not something prior training can drift.
    """
    gct_mapper = CtxEmbed(
        int(model.global_ctx_in_dim),
        int(model.global_emb_dim),
        mlp_ratio=2.0,
        drop=0.0,
        alpha_init=0.1,
    )
    if not SPATIAL_CKPT.exists():
        raise FileNotFoundError(
            f"gct mapper source missing: {SPATIAL_CKPT}. Run stage 1 first."
        )
    gct_state = torch.load(str(SPATIAL_CKPT), map_location="cpu")["global_embedder"]
    gct_mapper.load_state_dict(gct_state, strict=True)

    lct_path = Path(CKPTS["stage3_lct"])
    if not lct_path.exists():
        raise FileNotFoundError(
            f"lct mapper source missing: {lct_path}. Run stage 3 first."
        )
    lct_mapper = LctMapper.from_stage3_checkpoint(lct_path, map_location="cpu")

    for mapper in (gct_mapper, lct_mapper):
        mapper.to(device).eval()
        for parameter in mapper.parameters():
            parameter.requires_grad = False

    print(
        f"context mappers: gct {model.global_ctx_in_dim}->{model.global_emb_dim} "
        f"from {SPATIAL_CKPT.name} | lct 9->{lct_mapper.out_dim} from {lct_path.name}"
    )
    return gct_mapper, lct_mapper


def build_prior_from_model(
    model,
    device,
    *,
    Kmax,
    coordinate_mode=None,
):
    gct_mapper, lct_mapper = build_context_mappers(model, device)

    K1 = int(model.vq.num_codes_per_level[0])
    K2 = int(model.vq.num_codes_per_level[1])

    T, H, W = model.img_size
    pT, pH, pW = model.patch_size
    token_grid = (T // pT, H // pH, W // pW)

    activity_kwargs = dict(
        global_dim=dim_assay_for_emb,
        local_dim=9,
        num_tasks=4,
        token_grid=token_grid,
        Kmax=int(Kmax),
        d_model=128,
        n_layer=4,
        n_head=4,
        dropout=0.1,
    )
    activity_prior = MaskGITActivityPrior(
        **activity_kwargs,
        region_grid=STAGE4B_REGION_GRID,
        n_decoder_layer=STAGE4B_DECODER_LAYERS,
        n_maskgit_layer=STAGE4B_MASKGIT_LAYERS,
    ).to(device)

    flat_path = Path(CKPTS["stage2b_flat"])
    if not flat_path.exists():
        raise FileNotFoundError(
            f"Stage-2B flat codebook missing: {flat_path}. Run stage 2 phase 2b."
        )
    flat_cb = torch.load(str(flat_path), map_location="cpu")

    motif_prior = MaskGITMotifPrior(
        flat_codebook=flat_cb["embed"],
        merge_map=flat_cb["merge_map"],
        token_grid=token_grid,
        num_tasks=4,
        gct_mapper=gct_mapper,
        lct_mapper=lct_mapper,
        gct_latent_dim=model.global_emb_dim,
        lct_latent_dim=lct_mapper.out_dim,
        d_model=128,
        n_layer=4,
        n_head=4,
        dropout=0.1,
    ).to(device)

    return HierarchicalCodebookPrior(
        activity_prior=activity_prior,
        motif_prior=motif_prior,
    ).to(device)

def _stage4_token_grid(model):
    return tuple(
        int(model.img_size[i] // model.patch_size[i])
        for i in range(3)
    )


def _read_stage4_activity_metadata(path, model):
    path = Path(path)
    meta = torch.load(path, map_location="cpu")

    if "Kmax" not in meta or "token_grid" not in meta:
        raise RuntimeError(
            f"Activity checkpoint {path} must contain Kmax and token_grid metadata."
        )

    saved_grid = tuple(map(int, meta["token_grid"]))
    expected_grid = _stage4_token_grid(model)
    if saved_grid != expected_grid:
        raise RuntimeError(
            f"Activity checkpoint token grid {saved_grid} does not match "
            f"the current VQ-VAE token grid {expected_grid}. Set "
            "STAGE4_RECOMPUTE_KMAX=True and retrain 3B."
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


def _best_stage4b_checkpoint_path():
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
                f"{model_mode!r} model from {path}. Stage 4B must be retrained."
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
            "not a trained coordinate model; retrain Stage 4B before evaluation."
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
            "Retrain Stage 4B before treating this as a calibrated checkpoint."
        )
    else:
        print(f"Loaded activity checkpoint weights: {path}")


def _print_stage4_startup(label, hyperparameters, module, *, checkpoint_paths=()):
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
    # SparseRegionActivityPrior has no monolithic grid head; its placement
    # parameters live in the region/within pair.
    coordinate_head_modules = [
        getattr(module, name, None)
        for name in ("grid_head", "region_head", "within_head")
    ]
    grid_head_parameters = sum(
        sum(parameter.numel() for parameter in head.parameters())
        for head in coordinate_head_modules
        if head is not None
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


def _resolve_stage4_kmax(model, train_loader, device):
    if not STAGE4_RECOMPUTE_KMAX:
        for path in (
            CKPTS["activity_prior_best_hard_metric"],
            CKPTS["activity_prior_best"],
            CKPTS["activity_prior_refined_best"],
        ):
            if path.exists():
                meta = _read_stage4_activity_metadata(path, model)
                print(
                    f"Reusing Stage 4 Kmax={meta['Kmax']} from {path}. "
                    "Only metadata is reused; activity weights are not loaded."
                )
                return int(meta["Kmax"])

    count_stats = measure_stage4_active_token_counts(
        model=model,
        loader=train_loader,
        device=device,
        num_passes=int(STAGE4_KMAX_PASSES),
    )

    token_grid = _stage4_token_grid(model)
    Ntok = int(np.prod(token_grid))
    observed_max = int(count_stats["maximum"])
    stage4_kmax = min(
        Ntok,
        max(
            1,
            int(np.ceil(float(STAGE4_KMAX_MARGIN) * observed_max)),
        ),
    )

    print(
        "Selected Stage 4 Kmax:\n"
        f"  estimation passes = {count_stats['num_passes']}\n"
        f"  observed maximum = {observed_max}\n"
        f"  safety multiplier = {float(STAGE4_KMAX_MARGIN):.2f}\n"
        f"  selected Kmax = {stage4_kmax}\n"
        f"  Ntok = {Ntok}\n"
        f"  capacity ratio = {stage4_kmax / Ntok:.6f}"
    )
    return int(stage4_kmax)


def _load_stage4_motif_best(prior, device):
    path = CKPTS["motif_prior_best"]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Stage 4A checkpoint: {path}. Run with "
            "STAGE4_PHASES=('4a',) first."
        )
    ckpt = torch.load(path, map_location=device)
    prior.motif_prior.load_state_dict(ckpt["model"], strict=True)
    print(f"Loaded Stage 4A motif checkpoint: {path}")


def _load_stage4_activity_best(prior, model, device):
    path = _best_stage4b_checkpoint_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Stage 4B checkpoint: {path}. Run with "
            "STAGE4_PHASES=('4b',) first."
        )
    meta = _read_stage4_activity_metadata(path, model)
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
    print(f"Loaded Stage 4B hard-metric activity checkpoint: {path}")


def run_stage4a(prior, model, train_loader, val_loader, device):
    print("[4A] Training motif prior.")

    if STAGE4A_WARM_START and STAGE4A_WARM_START_PATH.exists():
        warm = torch.load(str(STAGE4A_WARM_START_PATH), map_location=device)
        warm_state = warm.get("model", warm)
        missing, unexpected = prior.motif_prior.load_state_dict(
            warm_state, strict=False
        )
        # strict=False will happily load NOTHING. A checkpoint from the old
        # two-level protocol (z1_emb / z2_emb / alpha_mu_head) matches zero
        # keys against the flat-alphabet prior, which would look like a warm
        # start in the log while actually being a cold one.
        matched = len(warm_state) - len(unexpected)
        if matched == 0:
            raise RuntimeError(
                f"Warm start {STAGE4A_WARM_START_PATH} matched 0 of "
                f"{len(warm_state)} keys -- it predates the flat-alphabet "
                f"prior. Set STAGE4A_WARM_START=False or point it at a "
                f"flat-alphabet checkpoint."
            )
        print(
            f"[4A] warm start from {STAGE4A_WARM_START_PATH} "
            f"(epoch {warm.get('epoch', '?')}); "
            f"matched={matched} missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
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
        epochs=int(STAGE4A_EPOCHS),
        grad_clip=1.0,
        ckpt_out=str(CKPTS["motif_prior_best"]),
        # Was 30. Both train and val z1 accuracy were still climbing at the old
        # 200-epoch limit with no overfit gap, so the run was budget-limited.
        early_stop_patience=150,
        grad_accum_steps=grad_accum_steps,
        full_mask_prob=0.15,

        # Loss rebalancing. At the end of the previous run the budget was:
        #   z1 CE 53%, alpha 23%, z1 neighbour-CE 15%, ctx 8%, rest ~1%.
        # Measured against matched nulls, alpha is unpredictable by ANY method
        # (model 0.168, per-position lookup 0.164, uniform simplex 0.170), so
        # 23% of the gradient was spent on a target carrying no signal. Its
        # weight is dropped to 0.1 -- not zero, because the alpha head still
        # feeds decoding and should not drift.
        loss_weights=(1.0, 0.1),

        # The neighbourhood-CE term spreads target mass over the 5 codebook-
        # nearest codes. It was included so near-misses are not punished, but
        # the model loses to the null on codebook DISTANCE as badly as on exact
        # identity (0.475 vs 0.395), so the smoothing is not buying anything and
        # is suppressing commitment. Reduced from 0.25.
        lambda_z1_distance=0.05,
        lambda_z1_neighbor_ce=0.05,
        z1_neighbor_tau=0.25,
        topk=(5, 2),

        # Voxel-domain auxiliary losses reduced from 1.0 so the motif objective
        # dominates while z1 is still improving. Stage 4C performs the
        # inference-aligned statistical calibration; 3A's job is motif identity.
        lambda_ctx=0.25,
        lambda_ctx_field=0.05,
        lambda_adj=0.25,
        lambda_spatial=0.25,
        ctx_tau=0.25,
        ctx_field_tau=0.25,
        memory_tok=getattr(model, "memory_tok", None),
        memory_adj=getattr(model, "memory_adj", None),
        isi_gap_bins=gap_bins,
        isi_max_gap=max_gap_from_bins(gap_bins),
    )
    save_json_report(history, REPORTS["prior_motif"])
    _load_stage4_motif_best(prior, device)
    return history


def run_stage4b(prior, model, train_loader, val_loader, device):
    print("[4B] Training activity prior.")
    for parameter in prior.activity_prior.parameters():
        parameter.requires_grad_(True)

    config = dict(STAGE4B_HYPERPARAMETERS)
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
    _print_stage4_startup(
        "Stage 4B",
        config,
        prior.activity_prior,
        checkpoint_paths=(
            CKPTS["activity_prior_best_loss"],
            CKPTS["activity_prior_best_hard_metric"],
            CKPTS["activity_prior_best"],
        ),
    )

    token_adj_bank = None
    _tab = Path("ckpts/token_adjacency_bank.pt")
    if STAGE4B_USE_TOKEN_ADJ_BANK and _tab.exists():
        from .model.spatial_map import GlobalContextAdjacencyBank
        _p = torch.load(str(_tab), map_location="cpu")
        token_adj_bank = GlobalContextAdjacencyBank()
        token_adj_bank.load_state_dict(_p[STAGE4B_TOKEN_ADJ_VARIANT])
        print(
            f"[4B] token adjacency bank: {_tab} variant={STAGE4B_TOKEN_ADJ_VARIANT} "
            f"bands={token_adj_bank.gap_bins} assays={len(token_adj_bank._num)}"
        )
    elif STAGE4B_USE_TOKEN_ADJ_BANK:
        raise FileNotFoundError(f"{_tab} missing; build it before enabling the bank.")

    # Selected on NLL rather than F1:
    # F1 is maximized by emitting the mode, which is the wrong target for a
    # checkpoint that exists to be sampled.
    dcfg = dict(STAGE4B_MASKGIT_HYPERPARAMETERS)
    history = train_maskgit_activity_prior(
        activity_prior=prior.activity_prior,
        vqvae=model,
        opt=opt_activity,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(dcfg["epochs"]),
        grad_clip=1.0,
        blank_code=getattr(model.vq, "blank_code", -1),
        lambda_bce=float(dcfg["lambda_bce"]),
        pos_weight=float(dcfg["pos_weight"]),
        lambda_count=float(dcfg["lambda_count"]),
        token_adj_bank=token_adj_bank,
        lambda_adj_t=float(dcfg["lambda_adj_t"]),
        lambda_adj_s=float(dcfg["lambda_adj_s"]),
        lambda_spatial=float(dcfg["lambda_spatial"]),
        random_mask_prob=float(dcfg["random_mask_prob"]),
        random_mask_ratio=tuple(dcfg["random_mask_ratio"]),
        select_on=str(dcfg["select_on"]),
        save_start_epoch=int(dcfg["save_start_epoch"]),
        early_stop_patience=int(dcfg["early_stop_patience"]),
        deterministic_val_masks=bool(config["deterministic_validation_masks"]),
        ckpt_out=str(CKPTS["activity_prior_best_hard_metric"]),
    )
    save_json_report(history, REPORTS["prior_activity"])
    _load_stage4_activity_best(prior, model, device)
    return history



def run_stage4c(prior, model, train_loader, val_loader, device):
    print("[4C] Training inference-aligned event-placement calibration.")
    config = dict(STAGE4C_HYPERPARAMETERS)
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
    trainable_names = configure_stage4c_event_calibration(prior.activity_prior)
    opt_refine = torch.optim.AdamW(
        [
            parameter
            for name, parameter in prior.activity_prior.named_parameters()
            if name in trainable_names
        ],
        lr=float(config["lr"]),
        weight_decay=0.01,
    )
    _print_stage4_startup(
        "Stage 4C",
        config,
        prior.activity_prior,
        checkpoint_paths=(
            CKPTS["motif_prior_best"],
            _best_stage4b_checkpoint_path(),
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
        epochs=int(STAGE4C_EPOCHS),
        grad_clip=1.0,
        ckpt_out=str(CKPTS["activity_prior_refined_best"]),
        early_stop_patience=int(config["early_stop_patience"]),
        grad_accum_steps=grad_accum_steps,
        freeze_motif=True,
        lambda_activity=float(config["lambda_activity"]),
        lambda_count=float(config["lambda_count"]),
        lambda_count_neighbor=float(config["lambda_count_neighbor"]),
        lambda_count_distance=float(config["lambda_count_distance"]),
        count_neighbor_k=11,
        count_neighbor_tau=2.0,
        count_distance_scale=5.0,
        soft_count_beta=5.0,
        lambda_ctx=float(config["lambda_ctx"]),
        lambda_ctx_field=float(config["lambda_ctx_field"]),
        lambda_adj=float(config["lambda_adj"]),
        lambda_spatial=float(config["lambda_spatial"]),
        auxiliary_ramp_epochs=int(config["auxiliary_ramp_epochs"]),
        save_start_epoch=int(config["save_start_epoch"]),
        ctx_field_tau=0.25,
        memory_tok=getattr(model, "memory_tok", None),
        memory_adj=getattr(model, "memory_adj", None),
        isi_gap_bins=gap_bins,
        isi_max_gap=max_gap_from_bins(gap_bins),
        generation_val_max_batches=int(config["generation_val_max_batches"]),
        generation_motif_steps=int(config["generation_motif_steps"]),
        deterministic_val_masks=bool(config["deterministic_validation_masks"]),
        hard_tolerance=STAGE4_HARD_TOLERANCE,
        null_baselines=_NULL_BASELINES,
    )
    save_json_report(history, REPORTS["prior_refine"])
    return history


def run_stage4_prior(model, train_loader, val_loader, device):
    print("\n" + "=" * 80)
    print("STAGE 4: staged prior learning")
    print("=" * 80)

    phases = _normalize_substage_phases(
        STAGE4_PHASES,
        ("4a", "4b", "4c"),
        name="STAGE4_PHASES",
    )
    if not phases:
        print("STAGE4_PHASES is empty; no Stage 4 training was requested.")
        return {}

    ckpt = select_ckpt(2, prefer_best=True)
    model.load_checkpoint(str(ckpt), map_location=device)
    print(f"Loaded VQVAE checkpoint for Stage 4 prior: {ckpt}")
    freeze_for_stage(model, 4)

    stage4_kmax = _resolve_stage4_kmax(model, train_loader, device)
    stage4_coordinate_mode = STAGE4_COORDINATE_MODE
    if "4c" in phases and "4b" not in phases:
        stage4b_path = _best_stage4b_checkpoint_path()
        stage4b_meta = _read_stage4_activity_metadata(stage4b_path, model)
        stage4_coordinate_mode = stage4b_meta["coordinate_mode"]
        print(
            "Stage 4C-only run will use the coordinate mode stored in its "
            f"Stage 4B checkpoint: {stage4_coordinate_mode!r}."
        )
    prior = build_prior_from_model(
        model,
        device,
        Kmax=stage4_kmax,
        coordinate_mode=stage4_coordinate_mode,
    )

    reports = {}

    # Fixed dependency order, matching Stage 2's phase driver.
    if "4a" in phases:
        reports["4a"] = run_stage4a(
            prior, model, train_loader, val_loader, device
        )

    if "4b" in phases:
        reports["4b"] = run_stage4b(
            prior, model, train_loader, val_loader, device
        )

    if "4c" in phases:
        if "4a" not in phases:
            _load_stage4_motif_best(prior, device)
        if "4b" not in phases:
            _load_stage4_activity_best(prior, model, device)
        reports["4c"] = run_stage4c(
            prior, model, train_loader, val_loader, device
        )

    return reports


def _select_stage4_activity_checkpoint(phase: str):
    phase = str(phase).lower()
    if phase == "4b":
        path = _best_stage4b_checkpoint_path()
        state_key = "model"
    elif phase == "4c":
        refined_path = CKPTS["activity_prior_refined_best"]
        if refined_path.exists():
            refined = torch.load(refined_path, map_location="cpu")
            if refined.get("accepted", False):
                return refined_path, "activity_prior"
            print(
                f"Stage 4C checkpoint {refined_path} was not accepted by the "
                "hard-generation gate; evaluating the unrefined Stage 4B checkpoint."
            )
        path = _best_stage4b_checkpoint_path()
        state_key = "model"
    else:
        raise ValueError(
            f"Stage-4 activity checkpoint phase must be '4b' or '4c', got {phase!r}."
        )

    if not path.exists():
        raise FileNotFoundError(
            f"Requested Stage {phase.upper()} checkpoint is missing: {path}"
        )
    return path, state_key


@torch.no_grad()
def _load_stage4_eval_prior(
    model,
    device,
    *,
    phase: str,
    load_generation_memories: bool = False,
    load_motif: bool = True,
):
    phase = str(phase).lower()
    if phase not in ("4a", "4b", "4c"):
        raise ValueError(f"Unsupported Stage-4 evaluation phase={phase!r}")

    vq_ckpt = select_ckpt(2, prefer_best=True)
    model.load_checkpoint(str(vq_ckpt), map_location=device)
    if load_generation_memories:
        load_stage1_gct_for_eval(model)
    freeze_for_stage(model, 4)

    if phase == "4a":
        # The activity branch is unused for 3A predictive evaluation.
        stage4_kmax = 1
        activity_ckpt_path = None
        activity_state_key = None
        activity_coordinate_mode = STAGE4_COORDINATE_MODE
    else:
        activity_ckpt_path, activity_state_key = (
            _select_stage4_activity_checkpoint(phase)
        )
        activity_meta = _read_stage4_activity_metadata(
            activity_ckpt_path,
            model,
        )
        stage4_kmax = int(activity_meta["Kmax"])
        activity_coordinate_mode = activity_meta["coordinate_mode"]

    prior = build_prior_from_model(
        model,
        device,
        Kmax=stage4_kmax,
        coordinate_mode=activity_coordinate_mode,
    )
    if load_motif:
        _load_stage4_motif_best(prior, device)

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
def load_stage4_prior(model, device, *, activity_phase: str):
    activity_phase = str(activity_phase).lower()
    if activity_phase not in ("4b", "4c"):
        raise ValueError(
            "Complete hierarchical generation requires activity_phase='4b' or '4c'."
        )
    return _load_stage4_eval_prior(
        model,
        device,
        phase=activity_phase,
        load_generation_memories=True,
        load_motif=True,
    )


@torch.no_grad()
def evaluate_stage4a_predictive(model, test_loader, device, train_loader=None):
    prior = _load_stage4_eval_prior(
        model,
        device,
        phase="4a",
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
        "samples": 0.0,
    }

    blank_code = getattr(model.vq, "blank_code", -1)

    # ---- null ladder -------------------------------------------------------
    # flat_acc against 936 classes is uninterpretable on its own.  The relevant
    # question is whether the prior beats a lookup table: the empirical flat
    # code frequency at this assay and this grid position.  The same comparison
    # already showed the activity head losing to an assay-frequency null, so
    # 4A does not get to report a raw accuracy without one.
    #
    # Fit on TRAINING data only.  The tables are keyed to codebook identity, so
    # they must be rebuilt whenever Stage 2 is retrained.
    null_levels = ("uniform", "global", "assay", "assay_position")
    null_payload = None
    if train_loader is not None:
        V = int(motif_prior.V) + 1
        if MOTIF_NULL_BASELINE_PATH.exists():
            try:
                null_payload = load_motif_null_baselines(str(MOTIF_NULL_BASELINE_PATH))
            except RuntimeError as exc:
                print(f"[4A nulls] {exc}")
                null_payload = None
            if null_payload is not None and int(null_payload.get("V", -1)) != V:
                print(
                    f"[4A nulls] cached payload has V={null_payload.get('V')} "
                    f"but model has V={V}; rebuilding."
                )
                null_payload = None
        if null_payload is None:
            print("[4A nulls] building motif null baselines from the training split...")
            null_payload = build_motif_null_baselines(
                train_loader, model, motif_prior, device=device,
                save_path=str(MOTIF_NULL_BASELINE_PATH),
            )
    else:
        print("[4A nulls] no train_loader supplied; skipping the null ladder.")
        null_levels = ()

    null_totals = {
        lvl: {"correct": 0.0, "top5": 0.0, "ce": 0.0} for lvl in null_levels
    }

    for batch in test_loader:
        x, gct, lct, task_id, mask_spec = _batch_to_device(
            batch,
            device,
        )
        assay_idx_batch = batch.get("assay_idx", None)
        codes, pmask, _ = _vq_codes_and_pmask_for_prior(
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
        )
        a_in, f_in, targets = motif_prior.corrupt_inputs_from_targets(
            targets,
            ensure_at_least_one_mask=True,
            full_mask_prob=float(STAGE4A_EVAL_FULL_MASK_PROB),
        )

        logits, motif_loss, aux = motif_prior(
            a_in,
            f_in,
            global_ctx=gct,
            local_ctx=lct,
            task_id=task_id,
            targets=targets,
            loss_weights=(1.0,),
        )

        flat_logits_nooov = logits["flat"][..., : motif_prior.V]
        flat_valid = targets["f_loss_mask"] & targets["f"].lt(motif_prior.V)
        flat_tgt = targets["f"].clamp(0, motif_prior.V - 1)
        neighbor_loss = distance_neighborhood_ce_loss(
            flat_logits_nooov,
            flat_tgt,
            flat_valid,
            motif_prior.flat_distance_matrix,
            k=5,
            tau=0.25,
        )
        distance_loss = expected_code_distance_loss(
            flat_logits_nooov,
            flat_tgt,
            flat_valid,
            motif_prior.flat_distance_matrix,
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

        z1_mask = targets["f_loss_mask"].bool()
        n_z1 = float(z1_mask.sum().item())
        if n_z1 > 0:
            z1_logits = logits["flat"][z1_mask]
            z1_target = targets["f"][z1_mask].long()
            totals["loss_z1"] += float(aux["loss_flat"].item()) * n_z1
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

            if null_payload is not None and assay_idx_batch is not None:
                import numpy as _np
                sel = z1_mask.nonzero(as_tuple=False)          # (m, 2): sample, position
                sample_of = sel[:, 0].detach().cpu().numpy()
                pos_of = sel[:, 1].detach().cpu().numpy()
                assay_np = (
                    torch.as_tensor(assay_idx_batch).reshape(-1).detach().cpu().numpy()
                )
                assay_of = assay_np[sample_of]
                tgt_np = z1_target.detach().cpu().numpy()

                for lvl in null_levels:
                    z1_prob = motif_null_predictions(
                        null_payload, assay_of, pos_of, level=lvl
                    )
                    pred = z1_prob.argmax(axis=1)
                    null_totals[lvl]["correct"] += float((pred == tgt_np).sum())
                    k5 = min(5, z1_prob.shape[1])
                    top5 = _np.argpartition(-z1_prob, k5 - 1, axis=1)[:, :k5]
                    null_totals[lvl]["top5"] += float(
                        (top5 == tgt_np[:, None]).any(axis=1).sum()
                    )
                    p_true = z1_prob[_np.arange(len(tgt_np)), tgt_np]
                    null_totals[lvl]["ce"] += float(
                        -_np.log(_np.clip(p_true, 1e-12, None)).sum()
                    )

    z1_den = max(totals["z1_tokens"], 1.0)
    report = {
        "phase": "4a",
        "loss_total": totals["loss_total"] / sample_den,
        "loss_motif_objective": totals["loss_motif_objective"] / sample_den,
        "loss_flat": totals["loss_z1"] / z1_den,
        "loss_flat_neighbor_ce": totals["loss_z1_neighbor_ce"] / sample_den,
        "loss_flat_expected_distance": totals["loss_z1_expected_distance"] / sample_den,
        "flat_acc": totals["z1_correct"] / z1_den,
        "flat_top5_acc": totals["z1_top5_correct"] / z1_den,
        "supervised_flat_tokens": int(totals["z1_tokens"]),
        "samples": int(totals["samples"]),
        "full_mask_prob": float(STAGE4A_EVAL_FULL_MASK_PROB),
    }

    if null_payload is not None:
        for lvl in null_levels:
            acc = null_totals[lvl]["correct"] / z1_den
            top5 = null_totals[lvl]["top5"] / z1_den
            ce = null_totals[lvl]["ce"] / z1_den
            report[f"null_{lvl}_z1_acc"] = acc
            report[f"null_{lvl}_z1_top5_acc"] = top5
            report[f"null_{lvl}_z1_ce"] = ce
            report[f"null_{lvl}_z1_acc_margin"] = report["z1_acc"] - acc
            report[f"null_{lvl}_z1_top5_margin"] = report["z1_top5_acc"] - top5
            report[f"null_{lvl}_z1_ce_margin"] = ce - report["loss_flat"]
        strongest = max(
            null_levels, key=lambda L: report[f"null_{L}_z1_acc"]
        )
        report["null_strongest_level"] = strongest
        report["beats_strongest_null_acc"] = bool(
            report["flat_acc"] > report[f"null_{strongest}_z1_acc"]
        )

    save_json_report(report, REPORTS["prior_motif_eval"])
    print("Stage 4A held-out predictive evaluation:", report)
    return report


@torch.no_grad()


@torch.no_grad()
def collect_generation_diagnostics(
    model,
    sampled,
    gen,
    grid,
):
    activity_out = sampled["activity_out"]
    activity = sampled["activity"].bool()
    codes = sampled["flat_ids"].long().unsqueeze(-1)

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
def evaluate_stage4_prior(
    model,
    test_loader,
    device,
    *,
    activity_phase="4c",
    out_dir="../viz_out_vqvae/vqvae_stage4/stage4_prior_gen",
    max_batches=20,
    samples_per_context=4,
    steps=12,
    temperature=1.0,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prior = load_stage4_prior(model, device, activity_phase=activity_phase)

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
        # Stage 4 then performs complete task-0 generation over all tokens.
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
        # Completely generated latent field, in the flat Stage-2B alphabet.
        flat_ids = sampled["flat_ids"]

        gen = decode_flat_ids_to_xgen(
            model,
            flat_ids,
            flat_codebook=prior.motif_prior.flat_codebook,
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
            prefix=f"stage4_testctx_b{bidx:04d}",
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
        #     out_dir / f"stage4_gen_batch_{bidx:04d}.pt",
        # )

    with open(out_dir / "stage4_generation_metrics.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Saved Stage 4 generated samples/metrics to: {out_dir}")
    return rows

@torch.no_grad()
def evaluate_stage4_prior_sampled_contexts(
    model,
    ref_loader,
    device,
    *,
    activity_phase="4c",
    out_dir="../viz_out_vqvae/vqvae_stage4/stage4_prior_sampled_ctx",
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
    Stage-4 free-generation eval with controllable context sampling.

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

    prior = load_stage4_prior(model, device, activity_phase=activity_phase)
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
        
        # Completely generated latent field, in the flat Stage-2B alphabet.
        flat_ids = sampled["flat_ids"]

        gen = decode_flat_ids_to_xgen(
            model,
            flat_ids,
            flat_codebook=prior.motif_prior.flat_codebook,
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


    print(f"Saved sampled-context Stage 4 generations to: {out_dir} | mode={mode}")
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


# =============================================================================
# Null baselines and leakage audit
# =============================================================================

def _load_null_baselines_if_available():
    """Return the null-baseline payload, or None with an explanatory message."""
    if not NULL_BASELINE_PATH.exists():
        print(
            f"[nulls] {NULL_BASELINE_PATH} not found. "
            "Set RUN_BUILD_NULL_BASELINES=True once to create it. "
            "Null reference scores will be skipped."
        )
        return None
    try:
        return load_null_baselines(str(NULL_BASELINE_PATH))
    except Exception as exc:
        print(f"[nulls] could not load {NULL_BASELINE_PATH}: {exc}")
        return None


@torch.no_grad()
def evaluate_existing_checkpoints_on_temporal_split(model, test_loader, device):
    """Reconstruction on the temporal test split, using the CURRENT checkpoints.

    Those checkpoints were fit under the old index-level split, where the same
    burst file appeared in train and test. Running them unchanged against a
    genuinely held-out temporal split measures how much of the reported
    performance depended on that overlap. Run this before committing to a full
    retrain: the delta is informative either way, and it is one evaluation pass.
    """
    print("\n" + "=" * 80)
    print("LEAKAGE DELTA: existing checkpoints, temporal held-out test split")
    print("=" * 80)

    rows = {}
    for stage_label, checkpoint in (
        ("stage1", CKPTS["stage2a_best"]),
    ):
        if not checkpoint.exists():
            print(f"  [{stage_label}] missing {checkpoint}; skipped.")
            continue

        model.load_checkpoint(str(checkpoint), map_location=device)

        report = evaluate_vqvae(
            model,
            test_loader,
            use_ROI_mask=False,
            recon_tolerance=recon_tolerance,
            metric_tolerance=metric_tolerance,
        )
        rows[stage_label] = report
        print(
            f"  [{stage_label}] AUPRC={report['AUPRC']:.5f} "
            f"BestF1={report['BestF1']:.5f} "
            f"AUPRC_tol={report['AUPRC_tol']:.5f}"
        )

    payload = {
        "note": (
            "Existing checkpoints (trained under the old index-level split, which "
            "placed the same burst files in train and test) evaluated on the new "
            "temporal within-assay held-out test split. Compare against the "
            "corresponding val_metrics entries in the stage training reports."
        ),
        "split": "temporal_within_assay",
        "results": rows,
    }
    save_json_report(payload, REPORTS["leakage_delta"])
    return payload


@torch.no_grad()
def evaluate_count_nulls(model, test_loader, device, null_baselines):
    """Compare the activity count head against trivial count predictors.

    Total spike count is analytically recoverable from local context feature 0
    (log mean firing density), so a closed-form map from that feature is the
    trivial solution available to any reader. A constant predictor emitting the
    training mean is the weaker reference. If the count head does not beat both,
    it has not earned its parameters and the manuscript should say so.
    """
    print("\n" + "=" * 80)
    print("COUNT NULLS: activity count head vs trivial predictors")
    print("=" * 80)

    prior = load_stage4_prior(model, device, activity_phase="4c")
    activity_prior = prior.activity_prior
    activity_prior.eval()
    model.eval()

    token_grid = (activity_prior.Ttok, activity_prior.Htok, activity_prior.Wtok)
    blank_code = getattr(model.vq, "blank_code", -1)

    global_stats = null_baselines["count_stats"]["global"]
    per_assay_stats = null_baselines["count_stats"]["per_assay"]
    density_fit = null_baselines["density_to_count"]
    density_fit_per_assay = null_baselines.get("density_to_count_per_assay", {})

    errors = {"model": [], "constant_global": [], "constant_per_assay": [],
              "density_global": [], "density_per_assay": []}
    targets = []

    for batch in test_loader:
        x, gct, lct, task_id, mask_spec = _batch_to_device(batch, device)
        codes, pmask, _ = _vq_codes_and_pmask(model, x, gct, lct, mask_spec, device)
        activity_targets = build_activity_targets_from_codes(
            codes=codes,
            token_grid=token_grid,
            Kmax=activity_prior.Kmax,
            blank_code=blank_code,
            predict_mask=pmask,
        )
        a_in = _make_activity_in_from_codes(
            codes, pmask, blank_code=blank_code, a_mask_id=activity_prior.a_mask_id
        )
        out = activity_prior(
            global_ctx=gct, local_ctx=lct, task_id=task_id,
            a_in=a_in, roi_mask=pmask,
        )

        true_count = activity_targets["count_target"].float().cpu().numpy()
        predicted = activity_prior._expected_count_from_logits(
            out["count_logits"]
        ).float().cpu().numpy()

        assay_idx = np.asarray(batch["assay_idx"]).reshape(-1).astype(np.int64)
        log_density = lct[:, 0].float().cpu().numpy()

        # Targets count active tokens INSIDE the ROI, while the null statistics
        # are whole-grid counts, so each null must be scaled to the ROI.
        #
        # Scaling by ROI area would assume activity is spread uniformly over the
        # grid, which is wrong for the causal task (the ROI is a time suffix) and
        # for the spatial task (a box over a non-uniform electrode support). Use
        # the assay's measured token-activation map to compute the fraction of
        # expected activity that actually falls inside this ROI, and fall back to
        # the area fraction only when the assay is missing from the bank.
        roi_bool = (pmask.squeeze(-1) if pmask.dim() == 3 else pmask).bool()
        token_frequency = token_frequency_for_batch(
            null_baselines, batch, token_grid, device
        )
        if token_frequency is None:
            roi_fraction = (
                roi_bool.sum(dim=1).float() / float(activity_prior.Ntok)
            ).cpu().numpy()
        else:
            flat_frequency = token_frequency.reshape(roi_bool.shape[0], -1)
            inside = (flat_frequency * roi_bool.float()).sum(dim=1)
            total = flat_frequency.sum(dim=1).clamp_min(1e-8)
            roi_fraction = (inside / total).cpu().numpy()

        constant_global = np.full_like(true_count, float(global_stats["mean"])) * roi_fraction
        constant_per_assay = np.array([
            float(per_assay_stats.get(int(a), global_stats)["mean"]) for a in assay_idx
        ]) * roi_fraction
        density_global = predict_count_from_density(density_fit, log_density) * roi_fraction
        density_per_assay = np.array([
            predict_count_from_density(
                density_fit_per_assay.get(int(a), density_fit), [d]
            )[0]
            for a, d in zip(assay_idx, log_density)
        ]) * roi_fraction

        targets.extend(true_count.tolist())
        errors["model"].extend(np.abs(predicted - true_count).tolist())
        errors["constant_global"].extend(np.abs(constant_global - true_count).tolist())
        errors["constant_per_assay"].extend(np.abs(constant_per_assay - true_count).tolist())
        errors["density_global"].extend(np.abs(density_global - true_count).tolist())
        errors["density_per_assay"].extend(np.abs(density_per_assay - true_count).tolist())

    target_arr = np.asarray(targets, dtype=np.float64)
    payload = {
        "samples": int(target_arr.size),
        "target_count_mean": float(target_arr.mean()),
        "target_count_mad": float(np.abs(target_arr - target_arr.mean()).mean()),
        "mae": {k: float(np.mean(v)) for k, v in errors.items()},
    }
    payload["model_beats"] = {
        k: bool(payload["mae"]["model"] < payload["mae"][k])
        for k in errors if k != "model"
    }

    print(f"  target count mean={payload['target_count_mean']:.2f} "
          f"MAD={payload['target_count_mad']:.2f}  (n={payload['samples']})")
    for name, value in payload["mae"].items():
        marker = "" if name == "model" else (
            "  <-- model WORSE" if value < payload["mae"]["model"] else ""
        )
        print(f"    MAE[{name:20s}] = {value:7.3f}{marker}")

    save_json_report(payload, REPORTS["count_nulls"])
    return payload


@torch.no_grad()
def evaluate_generation_baselines(model, test_loader, device, null_baselines,
                                  max_batches: int = 8):
    """Model generation vs an independent-rate surrogate, plus a context swap.

    Surrogate: per-voxel independent Bernoulli using the assay's empirical
    firing-rate map, rescaled to the requested density. It reproduces marginal
    rates and nothing else, so any metric on which the model does not beat it is
    not evidence of learned spatiotemporal structure.

    Context swap: generate sample i's volume using sample j's local context from
    the same assay. If the generated statistics track j (the supplied context)
    rather than i, conditioning is real; if they track neither, the context is
    being ignored.
    """
    print("\n" + "=" * 80)
    print("GENERATION BASELINES: model vs independent-rate surrogate")
    print("=" * 80)

    prior = load_stage4_prior(model, device, activity_phase="4c")
    model.eval()
    prior.eval()

    generator = torch.Generator(device=device).manual_seed(20240917)
    rows = {"model": [], "surrogate": [], "swapped_context": []}

    import time as _time
    start_time = _time.time()

    for batch_index, batch in enumerate(test_loader):
        if batch_index >= int(max_batches):
            break

        if batch_index and batch_index % 5 == 0:
            done = batch_index
            rate = (_time.time() - start_time) / max(done, 1)
            remaining = (min(int(max_batches), len(test_loader)) - done) * rate
            print(
                f"  [genbaselines] batch {done}/{min(int(max_batches), len(test_loader))} "
                f"({rate:.1f}s/batch, ~{remaining/60:.1f} min left)",
                flush=True,
            )

        x, gct, lct, task_id, mask_spec = _batch_to_device(batch, device)
        batch_size = x.shape[0]
        tok_out = model(x, global_ctx=gct, local_ctx=lct, predict_mask_spec=None)
        grid = tok_out["grid"]
        n_tokens = int(grid[0] * grid[1] * grid[2])

        roi = torch.ones((batch_size, n_tokens), dtype=torch.bool, device=device)
        zeros_task = torch.zeros((batch_size,), dtype=torch.long, device=device)

        # Roll the local context within the batch so each sample is generated
        # under a different sample's requested activity descriptor.
        swapped_lct = torch.roll(lct, shifts=1, dims=0)

        for label, context in (("model", lct), ("swapped_context", swapped_lct)):
            sampled = sample_hierarchical_roi(
                prior=prior,
                global_ctx=gct,
                local_ctx=context,
                task_id=zeros_task,
                roi_mask=roi,
                visible_codes=None,
                motif_steps=12,
                motif_temperature=1.0,
            )
            generated = decode_flat_ids_to_xgen(
                model, sampled["flat_ids"],
                flat_codebook=prior.motif_prior.flat_codebook, grid=grid,
                global_ctx=gct, local_ctx=context,
                roi_hw=batch.get("roi_hw", None), pad_hw=batch.get("pad_hw", None),
            )
            rows[label].append(
                _generation_row(generated["x_gen"], x, context, model)
            )

        surrogate = generate_rate_surrogate(
            null_baselines=null_baselines,
            assay_idx=batch["assay_idx"],
            local_ctx=lct,
            shape=x.shape,
            device=device,
            generator=generator,
        )
        rows["surrogate"].append(_generation_row(surrogate["x_gen"], x, lct, model))

    payload = {
        "batches": int(min(max_batches, batch_index + 1)),
        "results": {
            name: {
                key: float(np.mean([r[key] for r in batch_rows]))
                for key in batch_rows[0]
            }
            for name, batch_rows in rows.items()
            if batch_rows
        },
        "note": (
            "context_mae for 'swapped_context' is measured against the SUPPLIED "
            "(rolled) context, so a low value means the generator tracks the "
            "requested descriptor rather than the assay average."
        ),
    }

    print(f"{'':20s}{'count_rel_err':>14s}{'ctx_mae':>10s}{'field_err':>11s}{'gap_mae':>10s}")
    for name, values in payload["results"].items():
        print(f"{name:20s}{values['count_relative_error']:14.4f}"
              f"{values['context_mae']:10.4f}{values['local_field_error']:11.4f}"
              f"{values['short_gap_mae']:10.4f}")

    save_json_report(payload, REPORTS["generation_baselines"])
    return payload


def _generation_row(x_gen, x_target, requested_ctx, model):
    """Shared metric row so model and surrogate are scored by identical code."""
    from .training.stage4_activity import _activity_ctx_torch, _hard_gap_rates

    _, _, t_dec, h_dec, w_dec = x_gen.shape
    target = x_target[:, :1, :t_dec, :h_dec, :w_dec]

    generated_count = x_gen.sum(dim=(1, 2, 3, 4)).float()
    target_count = target.sum(dim=(1, 2, 3, 4)).float()

    generated_ctx = _activity_ctx_torch(x_gen[:, 0])
    context_error = (generated_ctx - requested_ctx[:, :9].to(generated_ctx)).abs()

    generated_gap = _hard_gap_rates(x_gen, gap_bins)
    target_gap = _hard_gap_rates(target, gap_bins)

    probability = x_gen.clamp(1e-4, 1.0 - 1e-4)
    logits = torch.log(probability) - torch.log1p(-probability)
    field_error = local_moment_field_loss(
        logits_b1thw=logits,
        target_b1thw=target,
        patch_size=model.patch_size,
        tau=0.25,
        prob_threshold=float(model.best_thr_tol.item()),
    )

    return {
        "count_relative_error": float(
            ((generated_count - target_count).abs()
             / target_count.clamp_min(1.0)).mean().item()
        ),
        "context_mae": float(context_error.mean().item()),
        "local_field_error": float(field_error.item()),
        "short_gap_mae": float((generated_gap - target_gap).abs().mean().item()),
        "generated_spike_count": float(generated_count.mean().item()),
        "target_spike_count": float(target_count.mean().item()),
    }


@torch.no_grad()
def evaluate_motif_nulls(model, train_loader, test_loader, device, *, rebuild: bool = True):
    """Motif prior vs empirical motif nulls, in the prior's most favourable regime.

    Activity is teacher-forced from ground truth and every active ROI motif is
    masked, matching STAGE4A_EVAL_FULL_MASK_PROB=1.0. The model additionally
    sees true motifs at visible (non-ROI) positions, which the nulls do not, so
    the comparison is conservative in the model's favour.

    Reported per active ROI token, over the flat Stage-2B alphabet:
      top-1 / top-5    flat code identity (chance = 1/V)
      latent MSE       ||e_flat[f_pred] - e_flat[f_true]||^2, which is what the
                       decoder actually consumes, reported both absolutely and
                       relative to substituting the blank token as a scale
                       reference
    """
    prior = load_stage4_prior(model, device, activity_phase="4c")
    motif_prior = prior.motif_prior
    motif_prior.eval()
    model.eval()
    V = int(motif_prior.V)

    if rebuild or not MOTIF_NULL_BASELINE_PATH.exists():
        payload = build_motif_null_baselines(
            train_loader, model, motif_prior, device=device,
            save_path=str(MOTIF_NULL_BASELINE_PATH),
        )
    else:
        payload = load_motif_null_baselines(str(MOTIF_NULL_BASELINE_PATH))
        if int(payload.get("V", -1)) != V + 1:
            payload = build_motif_null_baselines(
                train_loader, model, motif_prior, device=device,
                save_path=str(MOTIF_NULL_BASELINE_PATH),
            )

    E = motif_prior.flat_codebook.detach().float()
    names = ["model", "uniform", "global", "assay", "assay_position"]
    totals = {n: {"t1": 0.0, "t5": 0.0, "latent_mse": 0.0} for n in names}
    n_tokens = 0.0
    blank_reference = 0.0
    blank_code = getattr(model.vq, "blank_code", -1)

    for batch in test_loader:
        x, gct, lct, task_id, mask_spec = _batch_to_device(batch, device)
        codes, pmask, _ = _vq_codes_and_pmask_for_prior(
            model, x, gct, lct, mask_spec, device
        )
        targets = motif_prior.make_targets_from_codes(
            codes=codes, predict_mask=pmask, blank_code=blank_code
        )
        a_in, f_in, targets = motif_prior.corrupt_inputs_from_targets(
            targets, ensure_at_least_one_mask=True, full_mask_prob=1.0
        )
        logits, _, _ = motif_prior(
            a_in, f_in,
            global_ctx=gct, local_ctx=lct, task_id=task_id, targets=None,
        )
        selected = targets["f_loss_mask"].bool() & targets["f"].lt(V)
        if not bool(selected.any()):
            continue

        f_true = targets["f"][selected].long()
        batch_pos, token_pos = torch.nonzero(selected, as_tuple=True)
        assay_idx = np.asarray(batch["assay_idx"]).reshape(-1)[batch_pos.cpu().numpy()]
        positions = token_pos.cpu().numpy()

        true_latent = E[f_true]
        blank_reference += float(
            (model.vq.blank_token.detach().float().unsqueeze(0) - true_latent)
            .pow(2).sum().item()
        )

        predictions = {"model": logits["flat"][selected][:, :V].float()}
        for level in ("uniform", "global", "assay", "assay_position"):
            prob = motif_null_predictions(
                payload, assay_idx, positions, level=level
            )[:, :V]
            predictions[level] = (
                torch.from_numpy(np.log(prob + 1e-12)).float().to(device)
            )

        for name, flat_logits in predictions.items():
            top_k = flat_logits.topk(min(5, V), dim=-1).indices
            totals[name]["t1"] += float((flat_logits.argmax(-1) == f_true).sum().item())
            totals[name]["t5"] += float(
                (top_k == f_true.unsqueeze(-1)).any(-1).sum().item()
            )
            pred_latent = E[flat_logits.argmax(-1)]
            totals[name]["latent_mse"] += float(
                (pred_latent - true_latent).pow(2).sum().item()
            )

        n_tokens += float(f_true.numel())

    den = max(n_tokens, 1.0)
    report = {"active_roi_tokens": int(n_tokens), "V": V, "chance_top1": 1.0 / V}
    for name in names:
        report[f"{name}_top1"] = totals[name]["t1"] / den
        report[f"{name}_top5"] = totals[name]["t5"] / den
        report[f"{name}_latent_mse"] = totals[name]["latent_mse"] / den
    report["blank_latent_mse"] = blank_reference / den
    for name in names[1:]:
        report[f"model_minus_{name}_top1"] = (
            report["model_top1"] - report[f"{name}_top1"]
        )

    save_json_report(report, REPORTS["motif_nulls"])
    print("Motif null comparison:", report)
    return report



def evaluate_and_visualize(
    model,
    test_loader,
    stage: int,
    assay_indices,
    assay_codebook,
):
    if stage != 2:
        return

    try:
        ckpt = select_ckpt(2, prefer_best=True)
    except FileNotFoundError as e:
        if RUN_SKIP_MISSING_EVAL:
            print(f"Skipping eval/viz: {e}")
            return None
        raise

    device = next(model.parameters()).device
    model.load_checkpoint(str(ckpt), map_location=device)


    if RUN_EVAL:
        print(f"Evaluating VQVAE stage {stage} using {ckpt} ...")
        test_metrics = evaluate_vqvae(
            model,
            test_loader,
            pos_weight=2.0,
            use_amp=True,
            use_ROI_mask=False,
        )
        print(f"TEST stage {stage}:", test_metrics)
        save_json_report(_json_safe(test_metrics), REPORTS["stage2a_eval"])

    if RUN_VIZ:
        out_root = VIZ_ROOTS[stage]
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
            report_path = REPORTS["stage2a"]
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
    # Stage 1: gct mapper
    # ============================================================
    if stage == 1:
        if not RUN_VIZ:
            print("Skipping Stage 1 visualization because RUN_VIZ=False.")
            return

        if model.spatial_map_prior is None:
            print(
                "Skipping Stage 1 evaluation/visualization: "
                "model has no spatial_map_prior."
            )
            return

        # Always evaluate the best saved Stage 1 checkpoint,
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
    # Stage 2: hierarchical VQ-VAE
    # ============================================================
    if stage == 2:
        evaluate_and_visualize(
            model,
            test_loader,
            stage=2,
            assay_indices=assay_indices,
            assay_codebook=(
                test_loader.dataset.dataset.assay_codebook
            ),
        )

        if RUN_CODEBOOK_DEBUG:
            out_root = VIZ_ROOTS[2]
            debug_vq_codebooks(
                model,
                save_txt_path=str(out_root / "codebook_debug.txt"),
                save_json_path=str(out_root / "codebook_debug.json"),
            )
        return

    # ============================================================
    # Stage 4 substages
    # ============================================================
    if stage == 4:
        eval_phases = _normalize_substage_phases(
            STAGE4_EVAL_PHASES,
            ("4a", "4b", "4c"),
            name="STAGE4_EVAL_PHASES",
        )
        if not eval_phases:
            print("STAGE4_EVAL_PHASES is empty; skipping Stage 4 evaluation.")
            return

        if "4a" in eval_phases:
            try:
                evaluate_stage4a_predictive(
                    model,
                    test_loader,
                    device,
                    train_loader=train_loader,
                )
            except FileNotFoundError as exc:
                if RUN_SKIP_MISSING_EVAL:
                    print(f"Skipping Stage 4A evaluation: {exc}")
                else:
                    raise

        for phase in ("4b", "4c"):
            if phase not in eval_phases:
                continue

            # The old predictive evaluator went with the set-prediction
            # readout it scored. Stage 4B is now scored by NLL/AUPRC during
            # training and by sample-based generative metrics afterwards.
            try:
                pass
            except FileNotFoundError as exc:
                if RUN_SKIP_MISSING_EVAL:
                    print(f"Skipping Stage {phase.upper()} evaluation: {exc}")
                    continue
                raise

            if not RUN_STAGE4_GENERATION_EVAL:
                continue

            phase_root = Path(
                "../viz_out_vqvae/vqvae_stage4"
            ) / phase

            # A. Held-out exact-context generation.
            evaluate_stage4_prior(
                model,
                test_loader,
                device,
                activity_phase=phase,
                out_dir=str(phase_root / "stage4_prior_test_ctx"),
                max_batches=20,
                samples_per_context=4,
                steps=12,
                temperature=1.0,
            )

            ref_batch = next(iter(test_loader))
            fixed_gctx = ref_batch["global_ctx"][0:1]
            fixed_assay_id = int(ref_batch["assay_idx"][0].item())

            evaluate_stage4_prior_sampled_contexts(
                model,
                test_loader,
                device,
                activity_phase=phase,
                out_dir=str(phase_root / "stage4_prior_random_full"),
                context_bank_path="ckpts/context_prior.pkl",
                mode="random_full",
                max_samples=64,
            )
            evaluate_stage4_prior_sampled_contexts(
                model,
                test_loader,
                device,
                activity_phase=phase,
                out_dir=str(phase_root / "stage4_prior_fixed_global"),
                context_bank_path="ckpts/context_prior.pkl",
                mode="fixed_global",
                fixed_global_ctx=fixed_gctx,
                assay_id=fixed_assay_id,
                max_samples=64,
            )
            evaluate_stage4_prior_sampled_contexts(
                model,
                test_loader,
                device,
                activity_phase=phase,
                out_dir=str(phase_root / "stage4_prior_partial_local"),
                context_bank_path="ckpts/context_prior.pkl",
                mode="partial_local",
                partial_local={
                    "log_mean_firing_density": -9.1,
                    "temporal_trend": 0.0,
                },
                max_samples=64,
            )
            evaluate_stage4_prior_sampled_contexts(
                model,
                test_loader,
                device,
                activity_phase=phase,
                out_dir=str(
                    phase_root / "stage4_prior_fixed_global_partial_local"
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
    )

    # ============================================================
    # Sequential training followed immediately by stage evaluation
    # ============================================================
    
    evaluated_stages = set()
    
    # When Stage 1 is not being retrained, load its saved state before
    # any downstream VQVAE/prior training begins.
    needs_spatial_pretrain = (
        any(stage in TRAIN_STAGES for stage in (2, 3, 4))
        or any(stage in EVAL_STAGES for stage in (1, 2, 3, 4))
    )
    
    if 1 not in TRAIN_STAGES and needs_spatial_pretrain:
        load_spatial_pretrain_if_available(model)

    # ============================================================
    # Null baselines and leakage audit (opt-in, no training)
    # ============================================================
    global _NULL_BASELINES

    if RUN_BUILD_NULL_BASELINES:
        build_null_baselines(
            train_loader,
            model,
            device=device,
            save_path=str(NULL_BASELINE_PATH),
        )

    _NULL_BASELINES = _load_null_baselines_if_available()

    if RUN_LEAKAGE_DELTA_EVAL:
        evaluate_existing_checkpoints_on_temporal_split(model, test_loader, device)
        # The checkpoint loads above leave the model in the last-evaluated
        # stage's configuration; restore Stage 0 state before anything else.
        load_spatial_pretrain_if_available(model)

    if RUN_COUNT_NULL_EVAL:
        if _NULL_BASELINES is None:
            print("[nulls] skipping count nulls: no baseline file.")
        else:
            evaluate_count_nulls(model, test_loader, device, _NULL_BASELINES)

    if RUN_MOTIF_NULL_EVAL:
        evaluate_motif_nulls(model, train_loader, test_loader, device)

    if RUN_GENERATION_BASELINE_EVAL:
        if _NULL_BASELINES is None:
            print("[nulls] skipping generation baselines: no baseline file.")
        else:
            evaluate_generation_baselines(
                model, test_loader, device, _NULL_BASELINES
            )

    
    for stage in TRAIN_STAGES:
        # ========================================================
        # Train one stage
        # ========================================================
    
        if stage == 1:
            run_stage1_gct_pretrain(
                model,
                train_loader,
                device,
            )
    
            # Restore the best Stage 1 checkpoint rather than using
            # the final early-stopping epoch.
            load_spatial_pretrain_if_available(model)
    
            # Build the retrieval bank using the best Stage 1
            # global-context embedding.
            _, _lct_mapper = build_context_mappers(model, device)
            build_context_prior(
                train_loader,
                save_path="ckpts/context_prior.pkl",
                model=model,
                device=device,
                feature_names=ACTIVITY_CTX_NAMES,
                lct_mapper=_lct_mapper,
            )
    
        elif stage == 2:
            phases = _normalize_substage_phases(
                STAGE2_PHASES,
                ("2a", "2b"),
                name="STAGE2_PHASES",
            )
            if "2a" in phases:
                run_stage2a(
                    model,
                    train_loader,
                    val_loader,
                    blank_logit_threshold=1.05*logit_baseline
                )
            if "2b" in phases:
                model.load_checkpoint(
                    str(select_ckpt(2, prefer_best=True)), map_location=device
                )
                run_stage2b_flatten(
                    model,
                    train_loader,
                    device,
                    batch_to_device=_batch_to_device,
                    out_path=CKPTS["stage2b_flat"],
                    report_path=REPORTS["stage2b"],
                    source_ckpt=select_ckpt(2, prefer_best=True),
                )
    
        elif stage == 3:
            model.load_checkpoint(
                str(select_ckpt(2, prefer_best=True)), map_location=device
            )
            run_stage3_lct(
                model,
                train_loader,
                test_loader,
                device,
                batch_to_device=_batch_to_device,
                flat_codebook_path=CKPTS["stage2b_flat"],
                out_path=CKPTS["stage3_lct"],
                report_path=REPORTS["stage3_lct"],
                epochs=STAGE3_EPOCHS,
                batches_per_epoch=STAGE3_BATCHES_PER_EPOCH,
                n_textons=STAGE3_NUM_TEXTONS,
                tex_basis=STAGE3_TEXTON_BASIS,
            )
    
        elif stage == 4:
            run_stage4_prior(
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

if __name__ == "__main__":
    main()