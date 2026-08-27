# ckpts/

Every checkpoint on disk, what produced it, and whether it ships.

**Nothing here is renamed or deleted.** Experiment checkpoints are kept because a suffix is often the only record of what a run was, and the cost of keeping them is disk, not correctness.

The **shipped** rows are derived from `main.py`'s `CKPTS` dict and `external_baselines/param_census.py`, not typed by hand, so this table cannot drift from the code without the generator noticing.


## Top-level (132 files)

| file | status | MB | tracked | note |
|---|---|---:|:---:|---|
| `_SMOKE_4d_best_loss.pt` | experiment | 8.7 | no | smoke test, not a result |
| `_SMOKE_4d_best_z1acc.pt` | experiment | 8.7 | no | smoke test, not a result |
| `activity_prior_PILOT_qctx.pt` | experiment | 7.7 | no |  |
| `activity_prior_best.pt` | **shipped** (`CKPTS['activity_prior_best']`) | 4.7 | yes | val-selected |
| `activity_prior_best_PRE_V961.pt` | experiment | 4.7 | no | pre-flattening snapshot, 1024-entry ladder alphabet |
| `activity_prior_best_hard_metric.pt` | **shipped** (`CKPTS['activity_prior_best_hard_metric']`) | 7.7 | yes | val-selected |
| `activity_prior_best_hard_metric_COMPSEL.pt` | experiment | 7.7 | no | selected on the generation composite (superseded: 4B now selects on val NLL) |
| `activity_prior_best_hard_metric_GUMBELRUN1.pt` | experiment | 7.7 | no | gumbel top-K readout run |
| `activity_prior_best_hard_metric_NLLRUN3.pt` | experiment | 7.7 | no | val-NLL selection run |
| `activity_prior_best_hard_metric_PRE4C_042127.pt` | experiment | 7.7 | no | before Stage 4C adaptation |
| `activity_prior_best_hard_metric_PREQCTX.pt` | experiment | 7.7 | no | before the quantised-context change |
| `activity_prior_best_hard_metric_PRE_V961.pt` | experiment | 4.7 | no | pre-flattening snapshot, 1024-entry ladder alphabet |
| `activity_prior_best_hard_metric_last.pt` | experiment | 7.7 | no | final epoch, not val-selected |
| `activity_prior_best_hard_metric_last_COMPSEL.pt` | experiment | 7.7 | no | selected on the generation composite (superseded: 4B now selects on val NLL) |
| `activity_prior_best_hard_metric_last_NLLRUN3.pt` | experiment | 7.7 | no | val-NLL selection run |
| `activity_prior_best_loss.pt` | **shipped** (`CKPTS['activity_prior_best_loss']`) | 4.7 | yes | val-selected |
| `activity_prior_best_loss_PRE_V961.pt` | experiment | 4.7 | no | pre-flattening snapshot, 1024-entry ladder alphabet |
| `activity_prior_refined_best.pt` | **shipped** (`CKPTS['activity_prior_refined_best']`) | 7.7 | yes | val-selected |
| `activity_prior_refined_best_PREQCTX.pt` | experiment | 7.7 | no | before the quantised-context change |
| `activity_prior_refined_best_PRE_V961.pt` | experiment | 4.7 | no | pre-flattening snapshot, 1024-entry ladder alphabet |
| `activity_prior_v2adj_best_PARTIAL.pt` | experiment | 8.4 | no | val-selected |
| `activity_prior_v2adj_best_hard_PARTIAL.pt` | experiment | 8.4 | no | val-selected |
| `activity_prior_v2adj_best_loss_PARTIAL.pt` | experiment | 8.4 | no | val-selected |
| `activity_prior_v2bankadj_best.pt` | experiment | 8.4 | no | val-selected |
| `activity_prior_v2bankadj_best_hard.pt` | experiment | 8.4 | no | val-selected |
| `activity_prior_v2bankadj_best_loss.pt` | experiment | 8.4 | no | val-selected |
| `activity_prior_v2bankadjce_best.pt` | experiment | 8.4 | no | val-selected |
| `activity_prior_v2bankadjce_best_hard.pt` | experiment | 8.4 | no | val-selected |
| `activity_prior_v2bankadjce_best_loss.pt` | experiment | 8.4 | no | val-selected |
| `activity_prior_v2dense_best_hard.pt` | experiment | 7.7 | no | val-selected |
| `activity_prior_v2flathead_best.pt` | experiment | 8.9 | no | val-selected |
| `activity_prior_v2flathead_best_hard.pt` | experiment | 8.9 | no | val-selected |
| `activity_prior_v2flathead_best_loss.pt` | experiment | 8.9 | no | val-selected |
| `activity_prior_v2lossw_best.pt` | experiment | 8.4 | no | val-selected |
| `activity_prior_v2lossw_best_hard.pt` | experiment | 8.4 | no | val-selected |
| `activity_prior_v2lossw_best_loss.pt` | experiment | 8.4 | no | val-selected |
| `activity_prior_v2maskgit_best_hard.pt` | experiment | 7.7 | no | val-selected |
| `activity_prior_v2maskgit_refined_best.pt` | experiment | 7.7 | no | val-selected |
| `activity_prior_v2sparse_best.pt` | experiment | 8.4 | no | val-selected |
| `activity_prior_v2sparse_best_EP15KILLED.pt` | experiment | 8.4 | no | run killed at epoch 15 |
| `activity_prior_v2sparse_best_hard.pt` | experiment | 8.4 | no | val-selected |
| `activity_prior_v2sparse_best_hard_EP15KILLED.pt` | experiment | 8.4 | no | run killed at epoch 15 |
| `activity_prior_v2sparse_best_loss.pt` | experiment | 8.4 | no | val-selected |
| `activity_prior_v2sparse_best_loss_EP15KILLED.pt` | experiment | 8.4 | no | run killed at epoch 15 |
| `dense_activity_head_heldout.pt` | experiment | 3.7 | no |  |
| `dense_activity_head_temporal.pt` | experiment | 3.7 | no |  |
| `motif_prior_adapt_SEED101.pt` | experiment | 8.7 | no | seed replicate for variance estimation |
| `motif_prior_adapt_SEED101_best_loss.pt` | experiment | 8.7 | no | seed replicate for variance estimation |
| `motif_prior_adapt_SEED101_best_z1acc.pt` | experiment | 8.7 | no | seed replicate for variance estimation |
| `motif_prior_adapt_SEED202.pt` | experiment | 8.7 | no | seed replicate for variance estimation |
| `motif_prior_adapt_SEED202_best_loss.pt` | experiment | 8.7 | no | seed replicate for variance estimation |
| `motif_prior_adapt_SEED202_best_z1acc.pt` | experiment | 8.7 | no | seed replicate for variance estimation |
| `motif_prior_adapt_SEED303.pt` | experiment | 8.7 | no | seed replicate for variance estimation |
| `motif_prior_adapt_SEED303_best_loss.pt` | experiment | 8.7 | no | seed replicate for variance estimation |
| `motif_prior_adapt_SEED303_best_z1acc.pt` | experiment | 8.7 | no | seed replicate for variance estimation |
| `motif_prior_adapt_best.pt` | **shipped** (`CKPTS['motif_prior_ship']`) | 8.7 | yes | val-selected |
| `motif_prior_adapt_best_best_loss.pt` | experiment | 8.7 | no | val-selected |
| `motif_prior_adapt_best_best_z1acc.pt` | experiment | 8.7 | no | val-selected |
| `motif_prior_best.pt` | **shipped** (`CKPTS['motif_prior_best']`) | 8.7 | yes | val-selected |
| `motif_prior_best_4A_TEACHERFORCED.pt` | experiment | 8.7 | yes | val-selected |
| `motif_prior_best_PRE3AFIX.pt` | experiment | 4.2 | no | before the Stage-4A fix |
| `motif_prior_best_PREFLAT.pt` | experiment | 4.2 | no | before the flat alphabet |
| `motif_prior_best_PRE_V961.pt` | experiment | 8.5 | no | pre-flattening snapshot, 1024-entry ladder alphabet |
| `motif_prior_best_V961_CONSTLR_CONTROL.pt` | experiment | 8.7 | no | val-selected |
| `motif_prior_best_best_loss.pt` | experiment | 8.7 | no | val-selected |
| `motif_prior_best_best_z1acc.pt` | experiment | 8.7 | no | val-selected |
| `motif_prior_pilot_best.pt` | experiment | 8.5 | no | val-selected |
| `motif_prior_v2_best.pt` | experiment | 4.2 | no | val-selected |
| `motif_prior_warmstart.pt` | **referenced** (`main.py`) | 4.2 | no | warm-start source |
| `spatial_bias_pretrain.pt` | **referenced** (`main.py`) | 6.8 | yes |  |
| `stage1b_flat_codebook.pt` | experiment | 0.3 | no |  |
| `stage1c_lct_final.pt` | experiment | 1.6 | no |  |
| `stage1c_lct_mn.pt` | experiment | 1.4 | no |  |
| `stage1c_lct_trunk.pt` | experiment | 0.4 | no |  |
| `stage1c_lct_trunk_trained.pt` | experiment | 0.3 | no |  |
| `stage1c_texton_basis.pt` | experiment | 1.7 | no |  |
| `stage1c_texton_trunk.pt` | experiment | 0.4 | no |  |
| `stage2b_flat_codebook.pt` | **shipped** (`CKPTS['stage2b_flat']`) | 0.3 | yes |  |
| `stage2b_flat_codebook_OCCFILTER.pt` | experiment | 0.3 | no |  |
| `stage3_lct_mapper.pt` | **shipped** (`CKPTS['stage3_lct']`) | 1.6 | yes |  |
| `stage3_lct_mapper_V936.pt` | experiment | 1.6 | no |  |
| `token_adjacency_bank.pt` | **referenced** (`main.py`) | 0.1 | no |  |
| `vqvae_regress2level_last.pt` | experiment | 17.7 | no | final epoch, not val-selected |
| `vqvae_stage1_balanced_best.pt` | experiment | 17.7 | no | val-selected |
| `vqvae_stage1_balanced_best_64ch.pt` | experiment | 18.6 | no | superseded 64-children-per-parent config |
| `vqvae_stage1_balanced_last.pt` | experiment | 17.7 | no | final epoch, not val-selected |
| `vqvae_stage1_best.pt` | **referenced** (`main.py`) | 17.6 | yes | val-selected |
| `vqvae_stage1_last.pt` | experiment | 17.6 | yes | final epoch, not val-selected |
| `vqvae_stage1a_3level_best.pt` | experiment | 18.2 | no | val-selected |
| `vqvae_stage1a_3level_last.pt` | experiment | 18.2 | no | final epoch, not val-selected |
| `vqvae_stage1a_single_k256_best.pt` | experiment | 17.7 | no | val-selected |
| `vqvae_stage1a_single_k256_last.pt` | experiment | 17.7 | no | final epoch, not val-selected |
| `vqvae_stage1a_single_k32_best.pt` | experiment | 17.6 | no | val-selected |
| `vqvae_stage1a_single_k32_last.pt` | experiment | 17.6 | no | final epoch, not val-selected |
| `vqvae_stage1a_single_k32_nocurriculum_best.pt` | experiment | 17.6 | no | val-selected |
| `vqvae_stage1a_single_k32_tolonly_last.pt` | experiment | 17.6 | no | final epoch, not val-selected |
| `vqvae_stage1b_64ch_best.pt` | experiment | 18.6 | no | superseded 64-children-per-parent config |
| `vqvae_stage1b_64ch_kmeansonly.pt` | experiment | 18.6 | no | superseded 64-children-per-parent config |
| `vqvae_stage1b_64ch_last.pt` | experiment | 18.6 | no | superseded 64-children-per-parent config |
| `vqvae_stage1c_ctxhead_best.pt` | experiment | 18.2 | no | val-selected |
| `vqvae_stage1c_ctxhead_last.pt` | experiment | 18.2 | no | final epoch, not val-selected |
| `vqvae_stage2_convex_best.pt` | experiment | 17.6 | yes | val-selected |
| `vqvae_stage2_convex_last.pt` | experiment | 17.6 | yes | final epoch, not val-selected |
| `vqvae_stage2a_64ch_best.pt` | experiment | 18.6 | no | superseded 64-children-per-parent config |
| `vqvae_stage2a_64ch_last.pt` | experiment | 18.6 | no | superseded 64-children-per-parent config |
| `vqvae_stage2a_best.pt` | **shipped** (`CKPTS['stage2a_best']`) | 18.2 | yes | val-selected |
| `vqvae_stage2a_convex_best.pt` | experiment | 17.6 | yes | val-selected |
| `vqvae_stage2a_convex_last.pt` | experiment | 17.6 | yes | final epoch, not val-selected |
| `vqvae_stage2a_last.pt` | **shipped** (`CKPTS['stage2a_last']`) | 18.2 | yes | final epoch, not val-selected |
| `vqvae_stage2a_v2_best.pt` | experiment | 17.7 | no | val-selected |
| `vqvae_stage2a_v2_last.pt` | experiment | 17.7 | no | final epoch, not val-selected |
| `vqvae_stage2a_v3_best.pt` | experiment | 17.7 | no | val-selected |
| `vqvae_stage2a_v3_last.pt` | experiment | 17.7 | no | final epoch, not val-selected |
| `vqvae_stage2c_balanced_best.pt` | experiment | 17.7 | no | val-selected |
| `vqvae_stage2c_balanced_last.pt` | experiment | 17.7 | no | final epoch, not val-selected |
| `vqvae_stage2c_ctxfix2_best.pt` | experiment | 17.7 | no | val-selected |
| `vqvae_stage2c_ctxfix2_last.pt` | experiment | 17.7 | no | final epoch, not val-selected |
| `vqvae_stage2c_ctxfix3_best.pt` | experiment | 17.7 | no | val-selected |
| `vqvae_stage2c_ctxfix3_last.pt` | experiment | 17.7 | no | final epoch, not val-selected |
| `vqvae_stage2c_ctxfix_best.pt` | experiment | 17.7 | no | val-selected |
| `vqvae_stage2c_ctxfix_last.pt` | experiment | 17.7 | no | final epoch, not val-selected |
| `vqvae_stage2c_v2_best.pt` | experiment | 17.7 | no | val-selected |
| `vqvae_stage2c_v2_last.pt` | experiment | 17.7 | no | final epoch, not val-selected |
| `z2_centroids_1A_original.pt` | experiment | 0.1 | no |  |
| `z2_hull_vertices_1B_64ch.pt` | experiment | 0.5 | no | superseded 64-children-per-parent config |
| `z2_kmeans_seed_1B_64ch.pt` | experiment | 0.5 | no | superseded 64-children-per-parent config |
| `context_prior.pkl` | **referenced** (`analysis/generate_regimes.py`) | 1.1 | yes |  |
| `context_prior_OLDSPLIT.pkl` | experiment | 0.6 | no |  |
| `motif_null_baselines.pkl` | **referenced** (`main.py`) | 244.6 | no |  |
| `motif_null_baselines_V936.pkl` | experiment | 238.2 | no |  |
| `null_baselines.pkl` | **referenced** (`inference/metrics_gen.py`) | 3.5 | no |  |
| `null_baselines_v2.pkl` | experiment | 3.5 | no |  |

## Subdirectories

| dir | size | contents |
|---|---:|---|
| `ablations/` | 171M | sparse/dense encoder arms and the token-entropy dose-response run |
| `cache/` | 117M | derived caches moved out of reports/; regenerable, gitignored |
| `external_baselines/` | 318M | fitted baseline weights (DG, GLM, MaskGIT-flat, U-Net, CVAE); gitignored |
| `v2_baseline/` | 83M | pre-v3-rebuild snapshot, kept as the restore point for the peak-radius fix |

## Snapshot trees outside ckpts/

Gitignored local restore points, listed so their purpose is recorded rather than guessed. They are **not** empty.

| dir | size | what it predates |
|---|---:|---|
| `ckpts_oldsplit_backup/` | 134M | the current train/val/test split |
| `ckpts_pre2Cfix/` | 34M | the Stage-2C cross-attention investigation |
| `ckpts_pre_v2/` | 10M | the v2 baseline |
| `reports_oldsplit_backup/` | 6.5M | the current split |
| `reports_pre2Cfix/` | 312K | the Stage-2C investigation |
