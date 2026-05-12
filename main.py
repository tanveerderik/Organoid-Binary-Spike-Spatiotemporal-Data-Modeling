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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    # Spyder/IPython can sometimes initialize threads before this file runs.
    pass

from .dataset import make_loaders_for_assays, burst_collate
from .model import TransformerVQVAE, TokenMGITTransformer
from .model.spatial_map import GlobalContextSpatialBank, GlobalContextAdjacencyBank
from .training import fit_vqvae, train_prior_mgit, evaluate_vqvae, fit_spatial_prior_pretrain
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
from .visualization.reports_data import export_base_finetune_flat_xlsx

# =============================================================================
# User configuration
# =============================================================================

# Run any subset sequentially. Examples:
#   (1,)       -> train context-agnostic VQVAE only
#   (2,)       -> load stage-1 ckpt, train context-conditioned decoder only
#   (1, 2, 3)  -> run the whole pipeline sequentially
#   (0,)     -> run spatial-map pretraining only
TRAIN_STAGES = (1,2)        # 0,1,2,3
EVAL_STAGES  = (1,2)   # 0,1,2,3

RUN_EVAL = False
RUN_VIZ  = True
RUN_VIDEO_GEN = False
RUN_PLOTTER = True
RUN_CODEBOOK_DEBUG = True
RUN_SKIP_MISSING_EVAL = True

# Stage-0 spatial map checkpoint. If this file exists, it will be loaded before
# stages 1/2/3. If RUN_STAGES contains 0.5, it will be overwritten/trained first.
SPATIAL_CKPT = Path("../ckpts/spatial_bias_pretrain.pt")

CKPT_DIR = Path("../ckpts")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CKPTS = {
    "stage1_best": CKPT_DIR / "vqvae_stage1_best.pt",
    "stage1_last": CKPT_DIR / "vqvae_stage1_last.pt",
    "stage2_best": CKPT_DIR / "vqvae_stage2_best.pt",
    "stage2_last": CKPT_DIR / "vqvae_stage2_last.pt",
    "prior_best": CKPT_DIR / "mgit_prior_best.pt",
}

REPORTS = {
    "stage1": Path("../training_report_vqvae_stage1.json"),
    "stage2": Path("../training_report_vqvae_stage2.json"),
    "prior": Path("../training_report_mgit_prior.json"),
}

VIZ_ROOTS = {
    1: Path("../viz_out_vqvae/vqvae_stage1"),
    2: Path("../viz_out_vqvae/vqvae_stage2"),
}

# Data
patch_size = (4, 8, 8)
temporal_crop = 6000
temporal_pool = 120
batch_size = 1
grad_accum_steps = 32
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
max_gap = 3

# Tolerance for spike location in a voxel (for training loss and val metrics)
recon_tolerance = (2, 2, 2)
metric_tolerance = (0, 1, 1)

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
    max_gap: int = 3,
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
    nums = torch.zeros(max_gap, dtype=torch.float64)
    dens = torch.zeros(max_gap, dtype=torch.float64)

    for i, batch in enumerate(loader):
        x = batch["x"].float()

        # Accept (B,1,T,H,W) or (B,T,H,W)
        if x.ndim == 5:
            assert x.shape[1] == 1, f"Expected (B,1,T,H,W), got {tuple(x.shape)}"
            x = x[:, 0]
        elif x.ndim != 4:
            raise ValueError(f"Unexpected x shape: {tuple(x.shape)}")

        x = (x > threshold).float()  # (B,T,H,W)

        for g in range(1, max_gap + 1):
            if x.shape[1] <= g:
                continue

            x0 = x[:, :-g]   # activity at t
            xg = x[:, g:]    # activity at t+g

            nums[g - 1] += (x0 * xg).sum().double().cpu()
            dens[g - 1] += x0.sum().double().cpu()

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
        local_ctx_in_dim=5,
        local_emb_dim=32,
        global_ctx_in_dim=dim_assay_for_emb,
        global_emb_dim=32,

        use_spatial_map_prior=USE_GCT_PRETRAIN_MODULE,

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

    Stage 2: context-conditioned decoder calibration.
      - Freeze encoder/codebook.
      - Train decoder + local/global ctx-to-dec branches.
      - Keep pretrained spatial map frozen.
      - Decoder cross-attn ON in selected layer(s).

    Stage 3: prior learning.
      - Freeze VQVAE completely.
    """
    set_all_trainable(model, False)

    if stage == 1:
        set_decoder_cross_attention(model, enabled=False, layers=())
        for name in ["stem", "patch_embed", "sparse_encoder", "to_code", "vq", "code_to_dec",
                     "token_type_embed", "dec_blocks", "dec_norm", "patch_renderer"]:
            set_requires_grad(getattr(model, name, None), True)

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
        set_decoder_cross_attention(model, enabled=True, layers=(0,))
        for name in ["code_to_dec", "token_type_embed", "dec_blocks", "dec_norm", "patch_renderer",
                     "local_embedder", "local_to_dec_ctx", "global_to_dec_ctx"]:
            set_requires_grad(getattr(model, name, None), True)

        # Keep global embedder frozen if it came from spatial-map pretraining. This avoids moving
        # the representation that spatial_map_prior expects. If you later want to adapt it, unfreeze
        # it together with spatial_map_prior or add a copied decoder-only global embedder.
        set_requires_grad(getattr(model, "global_embedder", None), False)
        set_requires_grad(getattr(model, "spatial_map_prior", None), False)
        
        model.vq.freeze_codebook_updates = True
        
        if hasattr(model, "vq") and hasattr(model.vq, "tree_embeds"):
            for p in model.vq.tree_embeds:
                p.requires_grad = False


    elif stage == 3:
        set_decoder_cross_attention(model, enabled=True, layers=(0,))
        set_all_trainable(model, False)
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
    model.spatial_map_prior.load_state_dict(spatial_ckpt["spatial_map_prior"], strict=True)
    
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
        adj_max_gap=max_gap,
    )

    set_all_trainable(model, True)
    return report


def common_fit_kwargs(model):
    return dict(
        use_amp=True,
        grad_accum_steps=grad_accum_steps,
        recon_tolerance=recon_tolerance,
        metric_tolerance=metric_tolerance,

        isi_max_gap=max_gap,
        isi_tau=0.25,
        isi_margin=0.25,
        lambda_isi=1e-4,
        
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

    raise FileNotFoundError(f"No checkpoint found for stage {stage}: {best} or {last}")
    
    
def run_stage1(model, train_loader, val_loader, baseline_prob, logit_baseline):
    print("\n" + "=" * 80)
    print("STAGE 1: context-agnostic VQVAE motif learning")
    print("=" * 80)

    freeze_for_stage(model, 1)

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
        ctx_start_epoch=10,
        ctx_warmup_epochs=10,
        ctx_epoch_schedule={
            0: 10,
            1: 30,
            2: 30,
            3: 20,
            4: 50,
        },
        
        # CFG context dropout is irrelevant in Stage 1 because cross-attn is OFF.
        cfg_ctx_drop_start=0.0,
        cfg_ctx_drop_end=0.0,
        cfg_ctx_start_epoch=10**9,
        cfg_ctx_warmup_epochs=1,

        # Spike discovery but avoid permanent overactivation.
        pos_weight_start=100.0,
        pos_weight_end=5.0,
        pos_decay_epochs=100,

        use_logit_bias_schedule=False,
        logit_bias_start=logit_baseline,
        logit_bias_end=-0.01,
        logit_bias_decay_epochs=5,
        save_start_epoch = 125,
        **common_fit_kwargs(model),
    )

    with open(REPORTS["stage1"], "w") as f:
        json.dump(report, f, indent=4)
    print(f"Saved report: {REPORTS['stage1']}")
    return report


def run_stage2(model, train_loader, val_loader):
    print("\n" + "=" * 80)
    print("STAGE 2: context-conditioned decoder calibration")
    print("=" * 80)

    stage1_ckpt = select_ckpt(1, prefer_best=True)
    model.load_checkpoint(str(stage1_ckpt), map_location=next(model.parameters()).device)
    print(f"Loaded Stage 1 checkpoint for Stage 2: {stage1_ckpt}")

    freeze_for_stage(model, 2)

    n_epoch = 150
    optimizer = make_optimizer(model, lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch, eta_min=1e-5)

    report = fit_vqvae(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        epochs=n_epoch,
        ckpt_best_path=str(CKPTS["stage2_best"]),
        ckpt_last_path=str(CKPTS["stage2_last"]),
        early_stop_patience=30,
        val_metric_name="AUPRC_tol_cond",
        val_metric_goal="max",
        use_ROI_mask=False,

        # Stage 2: all context losses active quickly; cross-attn on layer 0 only.
        lambda_ctx=1e-1,
        ctx_start_epoch=0,
        ctx_warmup_epochs=20,
        ctx_epoch_schedule={
            0: 1,    # mean rate
            3: 1,    # active site ratio
            1: 1,   # temporal std
            2: 1,   # spatial std
            4: 1,   # temporal slope
        },
        
        cfg_ctx_drop_start=0.6,
        cfg_ctx_drop_end=0.25,
        cfg_ctx_start_epoch=1,
        cfg_ctx_warmup_epochs=5,

        # Calibration phase. Keep >1 first; test 1.0 only later if recall survives.
        pos_weight_start=5.0,
        pos_weight_end=1.0,
        pos_decay_epochs=100,

        use_logit_bias_schedule=False,
        
        save_start_epoch = 50,
        **{
            **common_fit_kwargs(model),
            "lambda_isi": 1e-3,
            "lambda_enc_var": 0.0,
            "level2_start_epoch": 1,
            "level2_full_loss_epoch": 1,
        },
    )

    with open(REPORTS["stage2"], "w") as f:
        json.dump(report, f, indent=4)
    print(f"Saved report: {REPORTS['stage2']}")
    return report


def build_prior_from_model(model, device):
    """
    Current placeholder compatible with your existing TokenMGITTransformer call.
    NOTE: this is not yet the final hierarchical prior. For hierarchy, you probably need
    either one prior head per VQ level or a flattened level-aware vocabulary.
    """
    gct_mapper = copy.deepcopy(model.global_embedder).eval()
    lct_mapper = copy.deepcopy(model.local_embedder).eval()

    # Existing code used vocab_size=num_codes + 1, which breaks if num_codes is a list.
    # This fallback uses max(K_l)+1 so the current prior can at least run for one-level style code prediction.
    # Replace train_prior_mgit before using true hierarchical tokens.
    vocab_size = int(max(num_codes)) + 1
    mask_id = vocab_size - 1

    prior = TokenMGITTransformer(
        vocab_size=vocab_size,
        mask_id=mask_id,
        num_tasks=4,
        gct_mapper=gct_mapper,
        gct_dim=model.global_ctx_in_dim,
        gct_latent_dim=model.global_emb_dim,
        lct_mapper=lct_mapper,
        lct_dim=model.local_ctx_in_dim,
        lct_latent_dim=model.local_emb_dim,
    ).to(device)
    return prior


def run_stage3_prior(model, train_loader, val_loader, device):
    print("\n" + "=" * 80)
    print("STAGE 3: MAGVIT/MaskGIT prior learning")
    print("=" * 80)

    try:
        ckpt = select_ckpt(2, prefer_best=True)
    except FileNotFoundError:
        ckpt = select_ckpt(1, prefer_best=True)
    
    model.load_checkpoint(str(ckpt), map_location=device)
    print(f"Loaded VQVAE checkpoint for Stage 3 prior: {ckpt}")
    
    freeze_for_stage(model, 3)

    prior = build_prior_from_model(model, device)
    opt_prior = torch.optim.AdamW(prior.parameters(), lr=3e-4, weight_decay=0.01)

    history_prior = train_prior_mgit(
        prior=prior,
        vqvae=model,
        opt=opt_prior,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        grad_clip=1.0,
        ckpt_out=str(CKPTS["prior_best"]),
        early_stop_patience=10,
        recon_task_id=0,
        recon_mask_ratio=0.60,
    )

    with open(REPORTS["prior"], "w") as f:
        json.dump(history_prior, f, indent=4)
    print(f"Saved report: {REPORTS['prior']}")
    return history_prior


def make_viz_loader(test_loader):
    return DataLoader(
        test_loader.dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=burst_collate,
        num_workers=0,
        pin_memory=True,
    )


def evaluate_and_visualize(model, test_loader, stage: int, assay_indices, assay_codebook):
    if stage not in (1, 2):
        return
    
    try:
        ckpt = select_ckpt(stage, prefer_best=True)
    except FileNotFoundError as e:
        if RUN_SKIP_MISSING_EVAL:
            print(f"Skipping eval/viz: {e}")
            return
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
            )
            
        if RUN_PLOTTER:
            report_path = REPORTS["stage1"] if stage == 1 else REPORTS["stage2"]
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
    logit_baseline = max(logit_baseline, -5.0)

    print("meta =", meta)
    print("baseline_prob =", baseline_prob, "logit_baseline =", logit_baseline)

    isi_target_gap_rates = compute_short_gap_target_rates_from_loader(
        train_loader,
        max_gap=max_gap,
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

    # =========================
    # TRAINING
    # =========================
    if 0 in TRAIN_STAGES:
        run_stage0_spatial_pretrain(model, train_loader, device)
    
    # Load Stage-0 pretrained GCT/spatial/memory banks before VQVAE/prior training or evaluation.
    if any(s in TRAIN_STAGES for s in (1, 2, 3)) or any(s in EVAL_STAGES for s in (0, 1, 2)):
        load_spatial_pretrain_if_available(model)
    
    for stage in TRAIN_STAGES:
        if stage == 0:
            continue
    
        elif stage == 1:
            run_stage1(model, train_loader, val_loader, baseline_prob, logit_baseline)
    
        elif stage == 2:
            run_stage2(model, train_loader, val_loader)
    
        elif stage == 3:
            run_stage3_prior(model, train_loader, val_loader, device)
    
        else:
            raise ValueError(f"Unsupported TRAIN_STAGE={stage}")
    
    
    # =========================
    # EVALUATION / VISUALIZATION
    # =========================
    for stage in EVAL_STAGES:
    
        if stage == 0:
            if not RUN_VIZ:
                continue
    
            if model.spatial_map_prior is None:
                print("Skipping stage 0 eval/viz: model has no spatial_map_prior.")
                continue
    
            out_dir = Path("../viz_out_vqvae/pre_stage1_spatial_maps")
            out_dir.mkdir(parents=True, exist_ok=True)
    
            save_assaywise_spatial_maps(
                model,
                assay_indices=assay_indices,
                n_assays=num_assays_for_emb,
                out_dir=str(out_dir),
                assay_codebook=train_loader.dataset.dataset.assay_codebook,
            )
            save_assaywise_adjacency_diagnostics(
                model,
                assay_indices=assay_indices,
                n_assays=num_assays_for_emb,
                out_dir=str(out_dir),
                assay_codebook=train_loader.dataset.dataset.assay_codebook,
            )
    
            if hasattr(model, "memory_adj") and model.memory_adj is not None:
                model.memory_adj.debug_print(max_items=5)
    
            print(f"Saved stage 0 spatial maps to: {out_dir}")
            continue
    
        if stage in (1, 2):
            evaluate_and_visualize(
                model,
                test_loader,
                stage=stage,
                assay_indices=assay_indices,
                assay_codebook=test_loader.dataset.dataset.assay_codebook,
            )
    
            if RUN_CODEBOOK_DEBUG:
                if stage == 1:
                    save_txt_path="../viz_out_vqvae/vqvae_stage1/codebook_debug.txt"
                    save_json_path="../viz_out_vqvae/vqvae_stage1/codebook_debug.json"
                elif stage == 2:
                    save_txt_path="../viz_out_vqvae/vqvae_stage2/codebook_debug.txt"
                    save_json_path="../viz_out_vqvae/vqvae_stage2/codebook_debug.json"
                debug_vq_codebooks(
                    model,
                    save_txt_path=save_txt_path,
                    save_json_path=save_json_path,
                )
    
        else:
            raise ValueError(f"Unsupported EVAL_STAGE={stage}")

    # ============================================================
    # Combined Stage 1 → Stage 2 plots/reports
    # ============================================================
    
    if REPORTS["stage1"].exists() and REPORTS["stage2"].exists():
        plot_base_then_finetune(
            report_base_path=str(REPORTS["stage1"]),
            report_ft_path=str(REPORTS["stage2"]),
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
            report_ft_path=str(REPORTS["stage2"]),
            out_xlsx="../training_report_stage1_stage2_flat.xlsx",
            shift_finetune_by="best",
        )

if __name__ == "__main__":
    main()
