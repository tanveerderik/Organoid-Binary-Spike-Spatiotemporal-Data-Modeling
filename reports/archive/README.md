# reports/archive/

Results from configurations that no longer exist. **Not comparable to any
shipped number** — kept because they are the only surviving trace of these
runs, and deleting them would make the decisions they informed unauditable.

Nothing in the repo reads these files.

| file | superseded configuration |
|---|---|
| `evaluation_report_plan_64.json` | 64-children-per-parent codebook |
| `evaluation_report_refit_children.json` | 64-children refit |
| `evaluation_report_parents_vs_children.json` | 64-children parent/child split |
| `training_report_vqvae_stage1b_64ch.json` | 64-children Stage 1B |
| `training_report_vqvae_stage2a_64ch.json` | 64-children Stage 2A |
| `stage2c_sampling_calibration.json` | Stage 2C decoder cross-attention (abandoned: the gate stayed pinned at 2.6e-4 and cond == uncond to five decimals) |
| `stage2c_sampling_temperature.json` | same |
| `oracle_activity_composite.json` | generation-composite decomposition, from when the composite was the paper metric and epoch selector |
| `oracle_activity_composite_decomp.json` | same |

The shipped configuration is a 3-level 32/8/4 ladder flattened and deduped to
V=961. See the top-level `README.md`.
