# Interpretable diagnostics

Two eval budgets, not interchangeable: the task axis uses 12 batches (48 clips), seed 20260822; everything else uses 8 batches (32 clips), seed 20260821. Identical clips for every model within a budget. Never carry a number between them.

Bold marks the best model for each metric. ↑ higher is better, ↓ lower is better, ≈REAL means the target is the real value itself, so both over- and under-shooting are failures. Rows with no arrow are descriptive.

Full test listings, conditioning ladders and provenance are in `diagnostics_appendix.md`.


## Scalability: what each arm stores per assay

The dataset here has 31 preparations. The question the paper has to answer is what happens at 1000. An arm whose capacity lives in a per-assay table does not have a scaling problem in principle -- it has one in practice, because every new preparation adds a full site map that must be estimated from that preparation's own data and stored forever.

| arm | fitted, shared | shared maps | stored PER ASSAY | memorised : fitted |
|---|---|---|---|---|
| Ours (4C+soft) | 8,620,946 | 0 | **0** | -- |
| MaskGIT-flat | 13,232,961 | 0 | **0** | -- |
| 3D U-Net (det.) | 8,372,673 | 0 | **0** | -- |
| 3D CVAE | 9,360,601 | 0 | **0** | -- |
| _ref_ Dich. Gaussian | 26 | 26,880 | 26,881 | 32,050 : 1 |
| _ref_ Coupled GLM | 5,194 | 53,792 | 26,880 | 160 : 1 |

**Training budget of the learned arms.** `val slope` is the mean validation improvement per epoch over the last five: near zero means the arm converged, visibly positive means the schedule ran out before the model did and its number is a lower bound on what the method can do.

| arm | train clips | epochs | selected at | val slope, last 5 | converged |
|---|---|---|---|---|---|
| 3D U-Net (det.) | 480 | 40 | 40 | +0.00172 | **budget-limited** |
| 3D CVAE | 480 | 40 | 39 | +0.00090 | yes, early-stopped |

| per-assay storage at | 31 assays | 100 assays | 1000 assays | growth |
|---|---|---|---|---|
| Ours (4C+soft) | 0 | 0 | 0 | **flat** |
| MaskGIT-flat | 0 | 0 | 0 | **flat** |
| 3D U-Net (det.) | 0 | 0 | 0 | **flat** |
| 3D CVAE | 0 | 0 | 0 | **flat** |
| _ref_ Dich. Gaussian | 833,311 | 2,688,100 | 26,881,000 | linear |
| _ref_ Coupled GLM | 833,280 | 2,688,000 | 26,880,000 | linear |

The learned arms store nothing per assay: `gct` is a fixed random +/-1 code regenerated from seed 0 (`dataset.py:403-415`), a handle rather than a table. The lookup arms store a full (H, W) site map each, so their footprint grows linearly and is already ~3x our entire model at 1000 preparations -- while the part of them that is shared across assays is 26 fitted values (DG) and 5,194 (GLM).

The claim here is scalability and nothing wider. A random code is an assay HANDLE, so a new preparation still needs training exposure and no number in this report measures transfer to an unseen preparation; the split is temporal within assay. What is measured is that parameter cost does not grow with the number of preparations. `gct` also reaches the priors through exactly one frozen mapper (`CtxEmbed`, `main.py:1492`), so substituting measured descriptors -- unit map, ISI distribution, stimulation protocol, DIV, cell type -- for the random code is a change to that module alone. That is stated as future work, not as a result.

**Withhold the table and the capability goes with it.** Same models, same clips, per-assay site map replaced by the global one:

| _ref_ arm | within-assay gap: with map → without | adherence: with map → without |
|---|---|---|
| Dich. Gaussian | +0.0186 → +0.0027  (**7x worse**) | +0.5314 → +0.2075  (**3x worse**) |
| Coupled GLM | +0.0272 → +0.0008  (**34x worse**) | +0.4524 → -0.0381  (**collapses past zero**) |

The GLM's adherence goes NEGATIVE: without a per-assay map it does not merely degrade, it stops tracking the requested context at all. That is the sense in which these arms are ceilings rather than methods -- what they score is the map, and the map is exactly the thing that does not scale.

**What this does and does not claim.** Zero per-assay storage is not zero memorisation: assay-specific information can live in shared weights. The checkable claim is that parameter count does not grow with the number of assays. gct is a SEEDED RANDOM code, so a new assay still needs training exposure -- this is not a zero-shot transfer claim. We therefore make the narrow claim: parameter cost is flat in the number of preparations, and the fitted capacity is shared rather than per-assay. We do not claim zero-shot transfer to an unseen preparation, and no table here measures it.

## Task axis

Per-clip average precision inside the masked region, 48 clips, every clip under every task so the columns are paired. `model` is the Monte-Carlo mean over 8 samplings. Scored on ROI voxels only.

`recon (0)` is free generation, not completion: its ROI is all-TRUE, so nothing is visible. Read it as the ZERO-CONTEXT reference, not a fourth peer.

**Do not read across task columns.** The ROI fraction differs by task (1.00 / 0.60 / 0.41 / 0.33), so the columns have different denominators and different base rates. Comparisons are valid WITHIN a column.

⚠ **Dich. Gaussian** has no completion mechanism: its clip-level output is a static per-site probability, so it cannot read the visible remainder. Its row is FREE GENERATION scored on the hole -- a capability statement, not a like-for-like score.

⚠ **3D U-Net (det.)** is the DIRECT-SUPERVISION reference, not a peer generator: it is trained with this exact loss on this exact hole distribution, and AP is a ranking metric, so a conditional-mean regressor is the best answer the table can contain. It is here to show what the generative arms have to buy their way past, and what it costs -- it carries no latent, no samples and no reusable representation. Compare it with `3D CVAE`, which is the same network with a latent variable and nothing else changed.


### Spatiotemporal AP ↑

**Which VOXEL fires, and when.** Each arm emits a score for every voxel inside the hole; the voxels are ranked by that score and scored against the truth with step-wise average precision, one clip at a time, then averaged over clips. A candidate is one (t, y, x) voxel of the ROI; a positive is a voxel that really spikes. This is the full completion problem -- an arm must get the electrode AND the frame right to earn credit.

| model ↑ | recon (0) = free gen | causal (1) | noncausal (2) | spatial (3) |
|---|---|---|---|---|
| Ours (4C+soft) | 0.0174 (115×) | 0.0163 (124×) | 0.0201 (128×) | **0.0244** (118×) |
| MaskGIT-flat | **0.0193** (129×) | 0.0184 (140×) | 0.0209 (134×) | 0.0181 (87×) |
| 3D U-Net (det.) | 0.0034 (22×) | 0.0074 (56×) | 0.0095 (61×) | 0.0047 (23×) |
| 3D CVAE | 0.0042 (28×) | **0.0190** (144×) | **0.0231** (148×) | 0.0070 (34×) |
| | | | | |
| _ref_ Dich. Gaussian | 0.0715 (475×) | 0.0741 (563×) | 0.0787 (503×) | 0.0830 (402×) |
| _ref_ Coupled GLM | 0.0892 (593×) | 0.1005 (763×) | 0.1029 (658×) | 0.1028 (497×) |
| | | | | |
| _chance_ uniformly random score | 1.50e-04 | 1.32e-04 | 1.56e-04 | 2.07e-04 |
| | | | | |
| _ceiling_ Ours (4C+soft) true tokens | **0.3254** | **0.4146** | **0.3490** | **0.3620** |
| _ceiling_ MaskGIT-flat true tokens | 0.0628 | 0.0692 | 0.0635 | 0.0859 |
| | | | | |
| _null_ assay site map, SEEN | **0.0869** | **0.0884** | **0.0954** | **0.1000** |
| _null_ assay site map, UNSEEN | 0.0026 | 0.0027 | 0.0028 | 0.0049 |
| _null_ visible profile x site map | -- | **0.0884** | **0.0954** | 0.0957 |
| _null_ persistence | -- | 0.0328 | 0.0449 | -- |

A candidate is one voxel of the ROI. Row definitions (`_chance_`, `_ceiling_`, `_null_`) follow the next table.


## Count calibration -- how many spikes, per clip

The per-task battery (appendix) scores binarised volumes, and it sets the spike count to to `round(assay_train_rate x |ROI|)`. That is ONE NUMBER PER ASSAY. Real clips inside an assay differ a lot -- true ROI counts have sd 103 spikes on the first task below -- so the readout cannot express per-clip activity even in principle, and it gives every model the same count regardless of what the model predicted.

`within-assay r` is the correlation with the TRUE ROI count after removing each assay's mean. `--` means the quantity is a constant within the assay and has no correlation to compute; that is the shared readout's row, and it is the point of this table.

| within-assay r with true ROI count | recon (0) = free gen | causal (1) | noncausal (2) | spatial (3) |
|---|---|---|---|---|
| _null_ assay train rate (the shared readout) | -- | +0.6172 | +0.1449 | +0.2477 |
| _null_ lct arithmetic, no model | +1.0000 | +0.8274 | +0.5783 | +0.5753 |
| Ours (4C+soft), own predicted count | **+0.9505** | **+0.9724** | **+0.9145** | **+0.9560** |
| MaskGIT-flat, own predicted count | +0.8525 | +0.1983 | +0.1721 | -0.0229 |
| 3D U-Net (det.), own predicted count | +0.9364 | +0.8197 | +0.0921 | +0.5774 |
| 3D CVAE, own predicted count | +0.8590 | +0.9137 | +0.8402 | +0.7476 |
| | | | | |
| _ref_ Dich. Gaussian, own predicted count | +0.9192 | +0.7729 | +0.3837 | +0.6772 |
| _ref_ Coupled GLM, own predicted count | +0.2762 | +0.6365 | +0.2265 | +0.7587 |

**Read this against the `lct arithmetic` row, not against zero.** Every arm is handed the TRUE clip's lct at generation time, and its first feature is `log_mean_firing_density` -- the clip's total count. So `exp(lct[0]) x |ROI|` predicts the ROI count with no model whatsoever. On `recon` that control scores +1.0000: the ROI is the whole volume there, so lct[0] IS the answer and no count head can beat arithmetic. On the three tasks where the ROI is a strict subset the control is much weaker, and that gap is where a count head actually earns its place.


Absolute accuracy of the same counts -- MAE in spikes, and bias as a fraction of the true mean:

| count MAE / bias | recon (0) = free gen | causal (1) | noncausal (2) | spatial (3) |
|---|---|---|---|---|
| _readout_ assay train rate (shared)  **calibrated** | 56.6 / +12.2% | 46.7 / +22.4% | 30.4 / +11.8% | 37.5 / -16.6% |
| Ours (4C+soft), own predicted count | 1116.5 / +575.8% | 548.2 / +527.0% | 493.4 / +609.9% | 503.1 / +587.3% |
| MaskGIT-flat, own predicted count | 6973.9 / +3596.3% | 3987.2 / +3827.7% | 3226.1 / +3986.7% | 1715.1 / +1997.5% |
| 3D U-Net (det.), own predicted count | 68.9 / +34.7% | 66.4 / +57.0% | 100.0 / +118.3% | 54.6 / +55.5% |
| 3D CVAE, own predicted count | 41.6 / +8.0% | 29.1 / +22.3% | 42.5 / +48.5% | 25.2 / +6.8% |
| _ref_ Dich. Gaussian, own predicted count | 15726.4 / +8109.9% | 9361.5 / +8999.6% | 6412.5 / +7926.8% | 5234.3 / +6110.1% |
| _ref_ Coupled GLM, own predicted count  **calibrated** | 55.1 / -2.0% | 41.6 / +8.0% | 29.0 / -1.7% | 26.1 / -1.6% |

**calibrated** marks an arm whose count bias stays within +/-25% on every task. Nothing is bolded on value: the best learned arm here is still ~6x over-count, and the only arm that is calibrated is a per-assay lookup.

**What the count actually buys: `log_mean_firing_density` under both readouts.** This feature is `log(mean + 1e-6)` over a fixed volume, so it measures the spike COUNT and nothing else -- it is the one lct feature that a count prediction can move. MAE against the clip's own lct:

| `log_mean_firing_density` MAE ↓ | recon (0) = free gen | causal (1) | noncausal (2) | spatial (3) |
|---|---|---|---|---|
| rate-matched (identical for every model) | 0.3697 | 0.3127 | 0.2068 | 0.2568 |
| own count: Ours (4C+soft) | 1.8865 | 1.2377 | 1.1951 | 1.1866 |
| own count: MaskGIT-flat | 2.9209 | 2.4077 | 2.3443 | 1.7851 |
| own count: 3D U-Net (det.) | 0.2586 | 0.2922 | 0.3437 | 0.2289 |
| own count: 3D CVAE | **0.1990** | **0.1531** | **0.1806** | **0.1561** |
| own count: Dich. Gaussian | 4.6016 | 4.0561 | 3.7096 | 3.5200 |
| own count: Coupled GLM | 0.3687 | 0.2978 | 0.2081 | 0.2151 |

Under the rate-matched readout the row is a CONSTANT: every model writes the same number of spikes, so the count error is the readout's, not the model's. Let each model supply its own count and the row separates them completely -- which is the point of having both readouts, and the direct answer to what count prediction is worth here.


**Discrimination and calibration are separate capabilities, and no arm has both.** The two tables above disagree about who wins, which is the finding rather than a contradiction: ranking which clip is busier within an assay, and knowing how many spikes that means, are different problems. An arm built on a decode probability field ranks well and is scaled badly, because the ROI sum of such a field is not a calibrated expected count; a point process fitted with an explicit rate gets the level right and discriminates poorly. The shared readout in the battery above is a third point: correct level, zero discrimination. That is why the battery uses it -- it is the only one comparable across arms -- and why this table exists separately.


**Reading the ceilings.** At voxel level the two priors score within 0.002 of each other, but they sit on tokenizers that are not comparable: on `recon` the ceiling is **0.3254** for Ours (4C+soft) against **0.0628** for MaskGIT-flat, a factor of **5.2**. At SITE level the same two ceilings are 0.2548 and 0.2619 -- the hierarchical alphabet buys resolution in TIME, not in space. What each prior then recovers of its own ceiling:

| model_mc / own oracle | recon (0) = free gen | causal (1) | noncausal (2) | spatial (3) |
|---|---|---|---|---|
| voxel: Ours (4C+soft) | 5% | 4% | 6% | 7% |
| voxel: MaskGIT-flat | **31%** | **27%** | **33%** | **21%** |
| site: Ours (4C+soft) | **103%** | **85%** | **91%** | **88%** |
| site: MaskGIT-flat | 50% | 48% | 51% | 43% |

The two rows point opposite ways. Our prior recovers most of what its alphabet allows for WHERE and almost none of it for WHEN, so the binding constraint on us is the prior's timing, not the tokenizer. MaskGIT-flat recovers a larger share at voxel level but of a ceiling five times lower, and only about half at site level: its binding constraint is the flat alphabet. Above 100% is possible because `model_mc` averages 8 samplings into a smoother ranking while the oracle is a single decode of the true tokens; the oracle bounds what the ALPHABET can express, not what an averaged field can score.


## Reconstruction

Step-wise average precision, not the trapezoid AUPRC in `utils/metrics.py` -- trapezoid interpolates the PR curve linearly, which is invalid (Davis & Goadrich 2006) and inflated the saturating MaskGIT tokenizer by +0.22 off a single voxel.

| | Ours (4C+soft) | MaskGIT-flat | 3D U-Net (det.) | 3D CVAE | _ref_ Dich. Gaussian | _ref_ Coupled GLM |
|---|---|---|---|---|---|---|
| AP step-wise, exact ↑ | **0.2564** | 0.0269 | -- | -- | -- | -- |
| AP step-wise, tolerant ↑ | **0.2680** | 0.0492 | -- | -- | -- | -- |
| best F1, exact ↑ | **0.3055** | 0.0678 | -- | -- | -- | -- |
| best F1, tolerant ↑ | **0.3074** | 0.1046 | -- | -- | -- | -- |
| trapezoid inflation ↓ | **+0.0015** | +0.2239 | -- | -- | -- | -- |
| codebook used | 619 | 844 | -- | -- | -- | -- |
| codebook perplexity | 436.9 | 312.2 | -- | -- | -- | -- |

Empty cells are a capability fact, not a missing run. This section measures what an ALPHABET can represent, so it needs a tokenizer: DG and the GLM are point processes and have none, and the two 3-D conv arms are inpainters with no autoencoding path -- they never see the clip they are asked to complete, so there is nothing for them to reconstruct. The comparison that does include them is **Task completion**.

`_ref_` rows are per-assay LOOKUP TABLES, not peer models: they store one site map per preparation and are shown as memorisation ceilings. Bold competes between the learned generative models only. See **Scalability** for what each arm stores per assay.


## Generation

Four families, no composite. Each cell is the metric at full context (`glob+full`) with that same model's `random`-context value in brackets -- its own null, so the bracket says how much of the number is conditioning rather than the model's default behaviour. The full five-rung ladders are in the appendix.

| metric | Ours (4C+soft) | MaskGIT-flat | 3D U-Net (det.) | 3D CVAE | _ref_ Dich. Gaussian | _ref_ Coupled GLM |
|---|---|---|---|---|---|---|
| A. Conditional accuracy ↓ | **0.6078** _[1.260]_ | 0.7403 _[1.399]_ | 1.0809 _[1.344]_ | 0.9644 _[1.362]_ | 0.6571 _[0.911]_ | 0.6666 _[0.710]_ |
| B. Adherence ↑ | **0.7764** _[0.657]_ | 0.5490 _[0.418]_ | 0.3856 _[0.526]_ | 0.4585 _[0.620]_ | 0.5314 _[0.250]_ | 0.4524 _[0.094]_ |
| C. Spatial placement, lookup-proof ↑ | **0.0204** _[0.003]_ | 0.0036 _[0.002]_ | -0.0062 _[0.001]_ | 0.0006 _[-0.003]_ | 0.0186 _[0.009]_ | 0.0272 _[-0.001]_ |
| D. Marginal realism ↓ | **0.1879** _[0.243]_ | 0.4028 _[0.306]_ | 0.7658 _[0.726]_ | 0.8529 _[0.849]_ | 0.0365 _[0.047]_ | 0.0794 _[0.088]_ |

`value _[null]_`, the null being the same arm given random context. An arm whose value and null are close is not using its context -- which is what the `_ref_` lookups do once their per-assay map is withheld (see **Scalability**).

**3D U-Net (det.) is marked `det.` because it has no sampling distribution.** It is trained to predict the conditional mean of the hole, so one context gives one field and every "sample" from it is the same field re-thresholded. That is the optimal answer to a ranking question (AP, F1) and vacuous as an answer to a distributional one (avalanche, ISI, marginal realism), so its two kinds of column must not be read the same way. `3D CVAE` is the same backbone with a latent variable added and nothing else changed, so the gap between the two arms is what stochasticity costs and buys here.

`_ref_` rows are per-assay LOOKUP TABLES, not peer models: they store one site map per preparation and are shown as memorisation ceilings. Bold competes between the learned generative models only. See **Scalability** for what each arm stores per assay.

