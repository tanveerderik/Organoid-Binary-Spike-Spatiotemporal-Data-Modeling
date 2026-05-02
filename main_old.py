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
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


#%%
# Optional: reduce PyTorch CPU thread usage too (set after importing torch)
from glob import glob
import json
import numpy as np

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from torch.utils.data import DataLoader

from .dataset import make_loaders_for_assays, burst_collate
from .model import TransformerVQVAE, TokenMGITTransformer
from .utils.losses import compute_min_gap_frames
from .training import fit_vqvae, train_prior_mgit, evaluate_vqvae, fit_spatial_bias_pretrain
from .visualization import make_model_videos_vqvae, run_plotter, plot_base_then_finetune, save_assaywise_spatial_maps
from .visualization.reports_data import export_base_finetune_flat_xlsx

from .visualization.rollout import generate_full_video
from .visualization.reports_rollout import (
    build_full_generation_report,
    save_report_json,
)


#%%

# find all assay folders
assay_paths = glob("../output_data/*/*/*_ecephys")

assay_dict = {}

for idx, assay_path in enumerate(assay_paths):
    assay_name = os.path.basename(assay_path)  # e.g. "assayA_ecephys"

    # find all npz files inside subfolders of this assay
    npz_files = glob(os.path.join(assay_path, "*/binary_unit_burst_*.npz"))

    # store both assay name and its files
    if len(npz_files) > 0:  # only keep non-empty assays
        assay_dict[idx] = {
            "assay_name": assay_name,
            "files": npz_files
     }

num_assays_for_emb = 1000 #max(assay_dict.keys()) + 1  # instead of len(assay_dict)
dim_assay_for_emb = 64

# check result
for k, v in assay_dict.items():
    print(f"{k}: {v['assay_name']} -> {len(v['files'])} files")
    
    
#%%
TRAIN_STAGE = 1
# 1 = context-agnostic VQVAE
# 2 = context-conditioned decoder calibration
# 3 = prior training
# 4 = optional decoder/prior finetune

spatial_map_pretraining_mode = False

vqvae_training_mode = (TRAIN_STAGE == 1)
vqvae_context_calib_mode = (TRAIN_STAGE == 2)
prior_training_mode = (TRAIN_STAGE == 3)
vqvae_finetuning_mode = (TRAIN_STAGE == 4)

base_visualization_flag = True
base_plotter_flag = True

base_generation_flag = False
base_rollout_flag = False

sft_visualization_flag = False
sft_generation_flag = False
sft_rollout_flag = False
sft_plotter_flag = False

    
#%%
# -----------------------
# Data + Model + Training
# -----------------------
patch_size = (16, 16, 16)

# assay_indices = [0]  # 21 highest
assay_indices = list(assay_dict.keys())

temporal_crop = 6000
temporal_pool = 120
img_time = min(12000, temporal_crop)

cache_dir="../_cache_spike_thw_run1"
cache_mode="uint8"
cache_max_gb=80         # pick a safe cap for your drive
cache_write_prob=1.0

num_codes=[32, 128]                      # must match VQ codebook size

max_viz_samples = 1000


train_loader, val_loader, test_loader, meta = make_loaders_for_assays(
    assay_indices=assay_indices,
    assay_dict=assay_dict,
    batch_size=8,
    temporal_crop=temporal_crop,
    temporal_pool=temporal_pool,
    spatial_crop=None,
    val_frac=0.2,
    test_frac=0.3,
    seed=42,
    task_probs={"recon": 0.25, "causal": 0.25, "noncausal": 0.25, "spatial": 0.25},  # mix of tasks
    patch_size=patch_size,     
    n_assays=num_assays_for_emb,
    dim_assays=dim_assay_for_emb,           
    per_assay_quota=10,
    num_workers=2,
    cache_dir=cache_dir,
    cache_mode=cache_mode,
    cache_max_gb=cache_max_gb,
    cache_write_prob=cache_write_prob,
)


batch0 = next(iter(train_loader))
x0 = batch0["x"]          # (B,1,T,H,W)
_, _, T0, H0, W0 = x0.shape
img_size = (T0, H0, W0)

full_hw0 = tuple(map(int, batch0["full_hw"][0]))
meta["img_size"] = img_size
meta["full_spatial_size"] = full_hw0

@torch.no_grad()
def compute_p0_from_loader(loader, max_batches=None):
    total_ones = 0.0
    total_voxels = 0.0

    for i, batch in enumerate(loader):
        x = batch["x"]  # (B,1,T,H,W)

        # ensure float for safety
        x = x.float()

        total_ones += x.sum().item()
        total_voxels += x.numel()

        if (max_batches is not None) and (i + 1 >= max_batches):
            break

    p0 = total_ones / max(total_voxels, 1.0)
    return p0

baseline_prob = compute_p0_from_loader(train_loader, max_batches=100)
meta["spike_voxel_prob"] = baseline_prob

baseline_prob = float(np.clip(baseline_prob, 1e-8, 1.0 - 1e-8))
logit_baseline = float(np.log(baseline_prob / (1.0 - baseline_prob)))
logit_baseline = max(logit_baseline, -5.0)

print(meta)



device = "cuda" if torch.cuda.is_available() else "cpu"


if TRAIN_STAGE == 1:
    use_decoder_cross_attn = False
    decoder_cross_attn_layers = ()
    cfg_ctx_drop_start = 0.0
    cfg_ctx_drop_end = 0.0

elif TRAIN_STAGE == 2:
    use_decoder_cross_attn = True
    decoder_cross_attn_layers = (0,)   # only first decoder block
    cfg_ctx_drop_start = 0.0
    cfg_ctx_drop_end = 0.3

else:
    use_decoder_cross_attn = True
    decoder_cross_attn_layers = (0,)
    cfg_ctx_drop_start = 0.3
    cfg_ctx_drop_end = 0.3

#%%
model = TransformerVQVAE(
    img_size=img_size,
    full_spatial_size=None,
    patch_size=patch_size,

    encoder_embed_dim=64, encoder_depth=4, encoder_num_heads=4,
    code_dim=64, num_codes=num_codes,
    decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=4,
    in_chans=1, out_chans=1,
    

    # ---- Context args ----
    local_ctx_in_dim=5,
    local_emb_dim=32,
    global_ctx_in_dim=dim_assay_for_emb,      # dataset gives 2D [assay2] for now
    global_emb_dim=32,
    
    spatial_bias_lim = 0.01,
    use_spatial_map_bias=True,
    spatial_map_bias_hidden_dim=64,
    
    enc_attn_mask_kind="none",
    dec_attn_mask_kind="temporal_causal",
    
    cfg_ctx_drop_p=0.0,
    use_decoder_cross_attn=use_decoder_cross_attn,
    decoder_cross_attn_layers=decoder_cross_attn_layers,
    
).to(device)



# keep norms in fp32 for stability (recommended)
for m in model.modules():
    if isinstance(m, (torch.nn.LayerNorm, torch.nn.GroupNorm,
                      torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
        m.float()

n_epoch = 500
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=n_epoch, eta_min=1e-5
)

# Example: set once before training
Fs = 20000         # <-- your raw sampling rate (Hz). Put the real number here.
refractory_ms = 5.0  # <-- pick what you want as a hard minimum

recon_tolerance=(2, 2, 2) # rt, rh, rw
metric_tolerance=(0, 1, 1)

pos_weight_start = 100.0
pos_weight_end = 5.0
pos_decay_epochs = 100        # reach pos_weight by epoch 200

lambda_isi = 1
lambda_ctx = 1e-1
lambda_cycle = 1e-4

ctx_start_epoch = 15         # start ctx after 25
ctx_warmup_epochs = 25       # ramp duration (or shorter, like 10–15)

ctx_epoch_schedule = {
    0: ctx_start_epoch,    # dim 0 starts at epoch 25
    1: 60,    # dim 1 starts at epoch 60
    2: 60,    # dim 2 starts at epoch 60
    3: 90,   # dim 3 starts at epoch 25
    4: 120,   # dim 4 starts at epoch 100
}

cfg_ctx_drop_start = 0.0
cfg_ctx_drop_end = 0.3
cfg_ctx_start_epoch = 120
cfg_ctx_warmup_epochs = 100

min_gap_frames = compute_min_gap_frames(refractory_ms, Fs, temporal_pool)
print("min_gap_frames =", min_gap_frames)


#%%

# optimizer only for global_embedder + spatial_map_bias
params_spatial_pretrain = (
    list(model.global_embedder.parameters()) +
    list(model.spatial_map_bias.parameters())
)
optimizer_spatial_pretrain = torch.optim.AdamW(params_spatial_pretrain, lr=1e-3, weight_decay=1e-4)
spatial_ckpt_path = "../ckpts/spatial_bias_pretrain.pt"

if spatial_map_pretraining_mode:
    print("Pretraining spatial map...")
    
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze only stage-0.5 branch
    for p in model.global_embedder.parameters():
        p.requires_grad = True
    for p in model.spatial_map_bias.parameters():
        p.requires_grad = True
    
    
    stage05 = fit_spatial_bias_pretrain(
        model=model,
        train_loader=train_loader,
        spatial_ckpt_path = spatial_ckpt_path,
        val_loader=None,
        optimizer=optimizer_spatial_pretrain,
        epochs=100,
        device=device,
        memory_momentum=0.98,
        memory_mode="max",   # I would start here
        lambda_size=1e-2,
        lambda_sep=1e-4,
        early_stop_patience=10
    )
    
    # Unfreeze everything
    for p in model.parameters():
        p.requires_grad = True

spatial_ckpt = torch.load(spatial_ckpt_path, map_location="cpu")
model.global_embedder.load_state_dict(spatial_ckpt["global_embedder"], strict=True)
model.spatial_map_bias.load_state_dict(spatial_ckpt["spatial_map_bias"], strict=True)


for p in model.global_embedder.parameters():
    p.requires_grad = False
for p in model.spatial_map_bias.parameters():
    p.requires_grad = False
    


# --- Adjusting learning rates for biases ---
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=n_epoch, eta_min=1e-5
)


#%%

ckpt_best_path = "../ckpts/vqvae_best_new.pt"
ckpt_last_path = "../ckpts/vqvae_last_new.pt"
report_path = "../training_report_vqvae_no_prior.json"


if vqvae_training_mode:
    print("Training VQVAE...")
    report = fit_vqvae(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        epochs=n_epoch,
        use_amp=True,
        grad_accum_steps=4,
        ckpt_best_path=ckpt_best_path,      # <-- renamed (old was ckpt_path)
        ckpt_last_path=ckpt_last_path,
        early_stop_patience=40,
        val_metric_name="AUPRC_tol_cond",
        val_metric_goal="max",
        use_ROI_mask=False,
        
        recon_tolerance=recon_tolerance,
        metric_tolerance=metric_tolerance,
        
        min_gap_frames = min_gap_frames,
        pos_weight_start = pos_weight_start,
        pos_weight_end = pos_weight_end,
        pos_decay_epochs = pos_decay_epochs,        # reach pos_weight by epoch 50 during training
        lambda_isi = lambda_isi,
        lambda_ctx = lambda_ctx,
        ctx_start_epoch = ctx_start_epoch,         # start ctx after 20
        ctx_warmup_epochs = ctx_warmup_epochs,       # ramp duration (or shorter, like 10–15)
        ctx_epoch_schedule = ctx_epoch_schedule,
        
        use_logit_bias_schedule=True,
        logit_bias_start=logit_baseline,
        logit_bias_end=-0.01,
        logit_bias_decay_epochs=15,
        
        level2_start_epoch = 10,
        level2_full_loss_epoch = 20,
        
        lambda_sp_consistency = 1e-3,
        
        cfg_ctx_drop_start=cfg_ctx_drop_start,
        cfg_ctx_drop_end=cfg_ctx_drop_end,
        cfg_ctx_start_epoch=cfg_ctx_start_epoch,
        cfg_ctx_warmup_epochs=cfg_ctx_warmup_epochs,
    
        # NEW optional args you can safely leave at defaults:
        prior=None,                 # put your trained prior here when finetuning
        prior_weight_cycle=0.0,     # e.g. 0.1 when doing decoder finetune
        prior_cycle_steps=0,        # e.g. 64 for decoder finetune
        prior_top_k=None,
        enable_cycle_path=False,
    )

    print("Train report:", report)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Training report saved to {report_path}")



#%%
model.load_checkpoint(ckpt_last_path, map_location=device)

print("model.best_thr before viz =", model.best_thr)


# ---- Test evaluation ----
print("Evaluating VQVAE...")

test_metrics = evaluate_vqvae(
    model,
    test_loader,
    pos_weight=pos_weight_end,
    use_amp=True,
    prior=None,                         # Optional: if you're using a trained prior
    prior_cycle_steps=0,               # Or 0 if not needed
    prior_top_k=None,                    # Or set (e.g. 100) if sampling restriction
    use_ROI_mask=False,
)

print("TEST:", test_metrics)


#%%
# deterministic order for reproducibility

viz_loader = DataLoader(
    test_loader.dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=burst_collate,   # <-- key change
    num_workers=0,
    pin_memory=True,
)


    
#%%
if base_visualization_flag:
    save_assaywise_spatial_maps(
        model,
        assay_indices=assay_indices,
        n_assays=num_assays_for_emb,
        out_dir="../viz_out_vqvae/vqvae_results_base/viz_spatial_bias",
        assay_codebook=test_loader.dataset.dataset.assay_codebook,
    )
    
    make_model_videos_vqvae(
        model,
        viz_loader,
        out_root="../viz_out_vqvae/vqvae_results_base",
        max_samples=max_viz_samples,
        fps=30,
        pool_t=1,
        thr=None,          # uses model.best_thr if present
        cmap_name="viridis"
    )
    

#%%
if base_plotter_flag:
    run_plotter(
        train_report=report_path,
        eval_roots=["../viz_out_vqvae/vqvae_results_base"],
        out_dir="../viz_out_vqvae/vqvae_results_base/figs",
        do_threshold_sweep=True,
    )

#%% Codebook health check

import torch
import torch.nn.functional as F

@torch.no_grad()
def debug_vq_codebooks(
    model,
    near_zero_thresh: float = 1e-6,
    duplicate_cos_thresh: float = 0.995,
    print_topk_pairs: int = 10,
    print_matrix_preview: int = 10,
    blank_topk: int = 10,
):
    """
    Debug hierarchical VQ codebooks for your current implementation.

    Assumes:
      - model.vq.embeds: ModuleList[nn.Embedding]
      - model.vq.blank_token: nn.Parameter of shape (D,)

    Prints for each level:
      - shape
      - alive/dead counts
      - norm stats
      - cosine similarity summary over alive codes
      - top highly similar code pairs
      - effective code groups (based on positive cosine duplicate threshold)
      - estimated effective count

    Also prints:
      - blank_token norm / health
      - nearest codebook entries to blank_token for each level
    """

    def _connected_components_from_adj(adj: torch.Tensor):
        """
        adj: (N,N) bool adjacency matrix
        returns: list[list[int]] of component node indices
        """
        n = adj.size(0)
        visited = torch.zeros(n, dtype=torch.bool)
        groups = []

        for i in range(n):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            comp = []

            while stack:
                u = stack.pop()
                comp.append(u)
                nbrs = torch.nonzero(adj[u], as_tuple=False).squeeze(1).tolist()
                for v in nbrs:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)

            groups.append(sorted(comp))
        return groups

    if not hasattr(model, "vq"):
        print("Model has no attribute 'vq'.")
        return

    if not hasattr(model.vq, "embeds"):
        print("model.vq has no attribute 'embeds'.")
        return

    if not hasattr(model.vq, "blank_token"):
        print("model.vq has no attribute 'blank_token'.")
        return

    print("=" * 80)
    print("VQ CODEBOOK DEBUG")
    print("=" * 80)

    # -----------------------------
    # Blank token health
    # -----------------------------
    blank = model.vq.blank_token.detach().cpu().flatten()
    blank_norm = blank.norm().item()
    blank_alive = blank_norm >= near_zero_thresh

    print("\n[Blank token]")
    print(f"shape: ({blank.numel()},)")
    print(f"norm : {blank_norm:.6g}")
    print(f"alive: {bool(blank_alive)}")

    for lvl, emb in enumerate(model.vq.embeds):
        cb = emb.weight.detach().cpu()  # (K,D)
        K, D = cb.shape

        print(f"\n[Level {lvl}]")
        print(f"shape: ({K}, {D})")

        norms = cb.norm(dim=1)
        alive_mask = norms >= near_zero_thresh
        dead_mask = ~alive_mask

        num_alive = int(alive_mask.sum().item())
        num_dead = int(dead_mask.sum().item())
        dead_idx = torch.nonzero(dead_mask, as_tuple=False).squeeze(1)

        print(f"alive: {num_alive} / {K}")
        print(f"dead : {num_dead} / {K}")
        print(
            "norm stats:",
            f"min={norms.min().item():.6g}",
            f"max={norms.max().item():.6g}",
            f"mean={norms.mean().item():.6g}",
            f"std={norms.std().item():.6g}",
        )

        if num_dead > 0:
            print("dead code indices:", dead_idx.tolist())

        # ----------------------------------
        # Blank token similarity to codebook
        # ----------------------------------
        cbn_all = F.normalize(cb, dim=1)
        blank_n = F.normalize(blank.unsqueeze(0), dim=1).squeeze(0)
        blank_sim = cbn_all @ blank_n   # (K,)

        topk_blank = min(blank_topk, K)
        vals, idxs = torch.topk(blank_sim, k=topk_blank, largest=True)

        print(f"top {topk_blank} codes most similar to blank_token:")
        for rank, (i, s) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
            print(
                f"  {rank:2d}. code {i:3d}  cos(blank, code)={s:+.6f}  "
                f"norm={norms[i].item():.6g}  alive={bool(alive_mask[i].item())}"
            )

        print(
            "blank-token similarity summary:",
            f"min={blank_sim.min().item():+.6f}",
            f"max={blank_sim.max().item():+.6f}",
            f"mean={blank_sim.mean().item():+.6f}",
            f"std={blank_sim.std().item():.6f}",
        )

        if num_alive <= 1:
            print("Not enough alive codes for cosine/effective analysis.")
            continue

        cb_alive = cb[alive_mask]
        alive_idx = torch.nonzero(alive_mask, as_tuple=False).squeeze(1)

        cbn = F.normalize(cb_alive, dim=1)
        sim = cbn @ cbn.T  # (A,A)

        A = sim.size(0)
        eye = torch.eye(A, dtype=torch.bool)
        offdiag = sim[~eye]

        print(
            "cosine(sim) over alive codes:",
            f"min={offdiag.min().item():.6f}",
            f"max={offdiag.max().item():.6f}",
            f"mean={offdiag.mean().item():.6f}",
            f"std={offdiag.std().item():.6f}",
        )

        num_dup_abs = int((offdiag.abs() > duplicate_cos_thresh).sum().item())
        num_dup_pos = int((offdiag > duplicate_cos_thresh).sum().item())
        num_dup_neg = int((offdiag < -duplicate_cos_thresh).sum().item())

        print(f"num offdiag pairs with |cos| > {duplicate_cos_thresh}: {num_dup_abs}")
        print(f"num offdiag pairs with  cos  > {duplicate_cos_thresh}: {num_dup_pos}")
        print(f"num offdiag pairs with  cos  < -{duplicate_cos_thresh}: {num_dup_neg}")

        preview = min(print_matrix_preview, A)
        print(f"cosine preview ({preview}x{preview}):")
        print(sim[:preview, :preview])

        # ----------------------------------
        # Effective code groups
        # Only positive near-duplicates count
        # ----------------------------------
        dup_adj = sim > duplicate_cos_thresh
        dup_adj.fill_diagonal_(True)

        groups_local = _connected_components_from_adj(dup_adj)
        groups_global = [[int(alive_idx[j].item()) for j in g] for g in groups_local]
        groups_global = sorted(groups_global, key=lambda g: (-len(g), g))

        num_effective = len(groups_global)
        grouped_sizes = [len(g) for g in groups_global]

        print(f"effective entries (@ cos > {duplicate_cos_thresh}): {num_effective}")
        print("effective groups:", groups_global)
        print("group sizes     :", grouped_sizes)

        duplicate_groups = [g for g in groups_global if len(g) > 1]
        singleton_groups = [g for g in groups_global if len(g) == 1]

        print(f"duplicate groups ({len(duplicate_groups)}): {duplicate_groups}")
        print(f"singleton groups ({len(singleton_groups)}): {singleton_groups}")

        # ----------------------------------
        # Top similar pairs
        # ----------------------------------
        pair_sims = []
        for i in range(A):
            for j in range(i + 1, A):
                pair_sims.append((abs(sim[i, j].item()), sim[i, j].item(), i, j))

        pair_sims.sort(reverse=True, key=lambda x: x[0])

        print(f"top {min(print_topk_pairs, len(pair_sims))} most similar alive-code pairs:")
        for rank, (_, s, i, j) in enumerate(pair_sims[:print_topk_pairs], start=1):
            gi = int(alive_idx[i].item())
            gj = int(alive_idx[j].item())
            print(
                f"  {rank:2d}. codes ({gi}, {gj})  cos={s:+.6f}  "
                f"norms=({norms[gi].item():.6g}, {norms[gj].item():.6g})"
            )

    print("\n" + "=" * 80)
    
debug_vq_codebooks(model)
    
 #%% Prior Training

d_model=512
n_layer=8
n_head=8
max_len=4096                       # must be ≥ T'*H'*W' from your patch embed
num_tasks=4


import copy
gct_mapper = copy.deepcopy(model.global_embedder).eval()
lct_mapper = copy.deepcopy(model.local_embedder).eval()

prior = TokenMGITTransformer(
    vocab_size=num_codes + 1,
    mask_id=num_codes,                 # usually the last id is mask
    num_tasks=num_tasks,
    gct_mapper=gct_mapper,  # <-- frozen mapper
    gct_dim=model.global_ctx_in_dim,                 # your global context input dim dim (or None)
    gct_latent_dim=model.global_emb_dim,  # safest
    lct_mapper=lct_mapper,  # <-- frozen mapper
    lct_dim=model.local_ctx_in_dim,                 # your Fourier dim (or None)
    lct_latent_dim=model.local_emb_dim,  # safest
)
prior = prior.to(device)

lr=3e-4
opt_prior = torch.optim.AdamW(prior.parameters(), lr=lr, weight_decay=0.01)

#%%
ckpt_prior_path = "../ckpts/mgit_prior_best.pt"
n_epoch_prior = 100

if prior_training_mode:
    # history_prior = train_prior(prior = prior,
    #     vqvae=model,   # path to your trained VQVAE
    #     opt = opt_prior,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     epochs=100,
    #     grad_clip=1.0,
    #     ckpt_out=ckpt_prior_path,
    #     early_stop_patience=10
    # )
    

    history_prior = train_prior_mgit(
        prior=prior,
        vqvae=model,
        opt=opt_prior,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=n_epoch_prior,
        grad_clip=1.0,
        ckpt_out=ckpt_prior_path,
        early_stop_patience=10,
        recon_task_id=0,
        recon_mask_ratio=0.60,
    )

    
    filename_prior = "../training_report_mgit_prior.json"
    
    # Open the file in write mode ('w') and use json.dump() to write the dictionary
    with open(filename_prior, 'w') as f:
        json.dump(history_prior, f, indent=4) # indent for pretty printing
    
    print(f"Prior dictionary saved to {filename_prior}")

#%%
checkpoint_prior = torch.load(ckpt_prior_path)
prior.load_state_dict(checkpoint_prior["model"])



#%%

# ----------------- Generation after finetuning the decoder ------------------------

n_demo = 10
fullgen_dir = "../viz_out_vqvae/generation_base/fullgen_demo/"
rollout_dir = "../viz_out_vqvae/generation_base/rollout_demo/"

os.makedirs(fullgen_dir, exist_ok=True)

# choose the generated video shape
# easiest choice: use same shape as x in viz_loader
if base_generation_flag:
    for n, batch in enumerate(viz_loader):
        if n == n_demo:
            break

        x = batch["x"]
        assay_id = int(batch["assay_idx"][0])

        # ---- get output shape (T,H,W) from current sample ----
        if x.ndim == 5:   # (B,1,T,H,W)
            _, _, T, H, W = x.shape
        elif x.ndim == 4: # (B,T,H,W)
            _, T, H, W = x.shape
        else:
            raise RuntimeError(f"Unexpected x shape: {x.shape}")

        out_shape_thw = (int(T), int(H), int(W))

        # ---- contexts from the real batch ----
        # adjust these names if your batch uses different keys
        gct = batch["global_ctx"][0]
        lct = batch["local_ctx"][0]

        sample_dir = os.path.join(fullgen_dir, f"demo_{n:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        print(f"{n+1}/{n_demo}  Assay: {assay_id}  Shape: {out_shape_thw}")

        gen_out = generate_full_video(
            vqvae=model,          # or vqvae=vqvae if you separated them
            prior=prior,          # your trained MaskGIT / prior model
            out_shape_thw=out_shape_thw,
            global_ctx=gct,
            local_ctx=lct,
            task_id=0,
            steps=24,
            top_k=None,
            thr=None,
            out_dir=sample_dir,
            fps=30,
            pool_t=None,
        )

        report = build_full_generation_report(
            gen_out,
            local_ctx=lct.detach().cpu().numpy() if isinstance(lct, torch.Tensor) else np.asarray(lct),
        )

        save_report_json(report, os.path.join(sample_dir, "report_full.json"))
        
        



#%%
ckpt_best_path2 = "../ckpts/vqvae_best_ft.pt"
ckpt_last_path2 = "../ckpts/vqvae_last_ft.pt"
report_path2 = "../training_report_vqvae_with_mgit_prior.json"

n_epoch_ft = 200
# ---- UNFREEZE MODEL ----
for p in model.parameters():
    p.requires_grad = True

# ---- Freeze encoder/codebook if doing decoder-only finetune ----
freeze_modules = [
    model.stem,
    model.sparse_encoder,
    model.to_code,
    model.vq,
    model.global_embedder,
    model.local_embedder,
    model.spatial_map_bias,
]

for m in freeze_modules:
    for p in m.parameters():
        p.requires_grad = False


# ---- REBUILD OPTIMIZER if you change requires_grad flags ----
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-4,           # smaller LR for finetune
    weight_decay=0.05
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=n_epoch_ft,          # e.g. 50–100 finetune epochs
    eta_min=1e-5
)


# Before finetune, build a finetune loader with NEXT-heavy mix if needed
train_loader_ft, val_loader_ft, _, meta_ft = make_loaders_for_assays(
    assay_indices=assay_indices,
    assay_dict=assay_dict,
    batch_size=8,
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
    per_assay_quota=30,
    num_workers=2,
    cache_dir=cache_dir,
    cache_mode=cache_mode,
    cache_max_gb=cache_max_gb,
    cache_write_prob=cache_write_prob,
)

ctx_epoch_schedule_ft = {
    0: 1,    # dim 0 starts at epoch 1
    1: 1,    # dim 1 starts at epoch 1
    2: 1,    # dim 2 starts at epoch 1
    3: 1,   # dim 3 starts at epoch 1
    4: 1,   # dim 4 starts at epoch 1
}

cfg_ctx_start_epoch_ft=25
cfg_ctx_warmup_epochs_ft=100

if vqvae_finetuning_mode:
    print("Finetuning VQVAE...")
    report = fit_vqvae(
        model,
        train_loader_ft,
        val_loader_ft,
        optimizer,
        scheduler,
        epochs=n_epoch_ft,
        use_amp=True,
        grad_accum_steps=4,
        ckpt_best_path=ckpt_best_path2,      # <-- renamed (old was ckpt_path)
        ckpt_last_path=ckpt_last_path2,
        early_stop_patience=10,
        val_metric_name="AUPRC_uncond",
        val_metric_goal="max",
        use_ROI_mask=True,
        
        recon_tolerance=recon_tolerance,
        metric_tolerance=metric_tolerance,
        
        min_gap_frames = min_gap_frames,
        pos_weight_start = pos_weight_end,
        pos_weight_end = pos_weight_end,
        pos_decay_epochs = 0,        # No need for any decay I guess
        lambda_isi = lambda_isi,
        lambda_ctx = lambda_ctx,
        ctx_start_epoch = 0,         # start ctx from the beginning 
        ctx_warmup_epochs = 5,       # tiny ramp in case initial instability during finetuning
        ctx_epoch_schedule=ctx_epoch_schedule_ft,
        
        use_logit_bias_schedule=False,
        
        cfg_ctx_drop_start=0.3,
        cfg_ctx_drop_end=1.0,
        cfg_ctx_start_epoch=cfg_ctx_start_epoch_ft,
        cfg_ctx_warmup_epochs=cfg_ctx_warmup_epochs_ft,

        # NEW optional args you can safely leave at defaults:
        prior=prior,                 # put your trained prior here when finetuning
        prior_weight_cycle=lambda_cycle,     # e.g. 0.1 when doing decoder finetune
        prior_cycle_steps=32,        # e.g. 64 for decoder finetune
        prior_top_k=None,
        enable_cycle_path=True,
    )

    print("Finetuning report:", report)

    with open(report_path2, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Finetuning report saved to {report_path2}")

#%%
model.load_checkpoint(ckpt_best_path2, map_location=device)
print("model.best_thr before viz =", model.best_thr)

# ---- Test evaluation ----
print("Evaluating VQVAE Finetuned Model...")

test_metrics = evaluate_vqvae(
    model,
    test_loader,
    pos_weight=pos_weight_end,
    use_amp=True,
    prior=prior,                         # Optional: if you're using a trained prior
    prior_cycle_steps=32,               # Or 0 if not needed
    prior_top_k=None,                    # Or set (e.g. 100) if sampling restriction
    use_ROI_mask=True,
)

print("TEST:", test_metrics)


#%%
# deterministic order for reproducibility

viz_loader = DataLoader(
    test_loader.dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=burst_collate,   # <-- key change
    num_workers=0,
    pin_memory=True,
)


if sft_visualization_flag:
    make_model_videos_vqvae(
        model,
        viz_loader,
        out_root="../viz_out_vqvae/vqvae_results_sft",
        max_samples=max_viz_samples,
        fps=30,
        pool_t=1,
        thr=None,          # uses model.best_thr if present
        cmap_name="viridis"
    )
    
    save_assaywise_spatial_maps(
        model,
        assay_indices=assay_indices,
        n_assays=num_assays_for_emb,
        out_dir="../viz_out_vqvae/vqvae_results_sft/viz_spatial_bias",
        assay_codebook=test_loader.dataset.dataset.assay_codebook,
    )



#%%

# ----------------- Generation after finetuning the decoder ------------------------
# Assume you already have a sample (H,W,T0) from your dataset or a file
# and that your model has model.patch_size and unpatchify helpers available.
        
n_demo = 10
fullgen_dir = "../viz_out_vqvae/generation_sft/fullgen_demo/"
rollout_dir = "../viz_out_vqvae/generation_sft/rollout_demo/"

os.makedirs(fullgen_dir, exist_ok=True)

# choose the generated video shape
# easiest choice: use same shape as x in viz_loader
if sft_generation_flag:
    for n, batch in enumerate(viz_loader):
        if n == n_demo:
            break

        x = batch["x"]
        assay_id = int(batch["assay_idx"][0])

        # ---- get output shape (T,H,W) from current sample ----
        if x.ndim == 5:   # (B,1,T,H,W)
            _, _, T, H, W = x.shape
        elif x.ndim == 4: # (B,T,H,W)
            _, T, H, W = x.shape
        else:
            raise RuntimeError(f"Unexpected x shape: {x.shape}")

        out_shape_thw = (int(T), int(H), int(W))

        # ---- contexts from the real batch ----
        # adjust these names if your batch uses different keys
        gct = batch["global_ctx"][0]
        lct = batch["local_ctx"][0]

        sample_dir = os.path.join(fullgen_dir, f"demo_{n:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        print(f"{n+1}/{n_demo}  Assay: {assay_id}  Shape: {out_shape_thw}")

        gen_out = generate_full_video(
            vqvae=model,          # or vqvae=vqvae if you separated them
            prior=prior,          # your trained MaskGIT / prior model
            out_shape_thw=out_shape_thw,
            global_ctx=gct,
            local_ctx=lct,
            task_id=0,
            steps=24,
            top_k=None,
            thr=None,
            out_dir=sample_dir,
            fps=30,
            pool_t=None,
        )

        report = build_full_generation_report(
            gen_out,
            local_ctx=lct.detach().cpu().numpy() if isinstance(lct, torch.Tensor) else np.asarray(lct),
        )

        save_report_json(report, os.path.join(sample_dir, "report_full.json"))
        
        
        
        
#%%
if sft_plotter_flag:
    run_plotter(
        train_report=report_path2,
        eval_roots=["../viz_out_vqvae/vqvae_results_sft"],
        out_dir="../viz_out_vqvae/vqvae_results_sft/figs",
        do_threshold_sweep=True,
    )


#%%

plot_base_then_finetune(
    report_base_path=report_path,
    report_ft_path=report_path2,
    out_dir="../viz_out_vqvae/plots_all_overlays",
    mode="shared",                 # union saves everything possible
    truncate_base_at_best=False,   # base curve ends at best epoch
    save_pdf=True,
)


#%%

export_base_finetune_flat_xlsx(
    report_base_path=report_path,
    report_ft_path=report_path2,
    out_xlsx="../training_report_base_finetune_flat.xlsx",
    shift_finetune_by="best",   # or "full"
)