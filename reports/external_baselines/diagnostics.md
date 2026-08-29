# Interpretable diagnostics

Pinned evaluation protocol. task axis: 70 batches, 8 MC samples, seed 20260822; everything else: 70 batches, seed 20260821. The test loader is not shuffled, so a batch count is a prefix of the split and every model within a budget sees identical clips. Never carry a number between budgets.

Bold marks the best model for each metric. ↑ higher is better, ↓ lower is better, ≈REAL means the target is the real value itself, so both over- and under-shooting are failures. Rows with no arrow are descriptive.

Full test listings, conditioning ladders and provenance are in `diagnostics_appendix.md`.


## Scalability: what each arm stores per assay

The dataset here has 31 preparations. The question the paper has to answer is what happens at 1000. An arm whose capacity lives in a per-assay table does not have a scaling problem in principle -- it has one in practice, because every new preparation adds a full site map that must be estimated from that preparation's own data and stored forever.

| arm | fitted, shared | shared maps | stored PER ASSAY | memorised : fitted |
|---|---|---|---|---|
| Ours (4C+soft) | 8,620,946 | 0 | **0** | -- |
| MaskGIT-flat | 13,232,961 | 0 | **0** | -- |
| 3D U-Net (det.)† | 8,372,673 | 860,160 | **0** | -- |
| 3D CVAE | 9,360,601 | 860,160 | **0** | -- |
| _ref_ Dich. Gaussian | 26 | 26,880 | 26,881 | 32,050 : 1 |
| _ref_ Coupled GLM | 5,194 | 53,792 | 26,880 | 160 : 1 |

**Training budget of the learned arms.** `val slope` is the mean validation improvement per epoch over the last five: near zero means the arm converged, visibly positive means the schedule ran out before the model did and its number is a lower bound on what the method can do.

| arm | train clips | epochs | selected at | val slope, last 5 | converged |
|---|---|---|---|---|---|
| 3D U-Net (det.)† | 480 | 42 | 32 | -0.00658 | yes, early-stopped |
| 3D CVAE | 480 | 31 | 21 | -0.00008 | yes, early-stopped |

| per-assay storage at | 31 assays | 100 assays | 1000 assays | growth |
|---|---|---|---|---|
| Ours (4C+soft) | 0 | 0 | 0 | **flat** |
| MaskGIT-flat | 0 | 0 | 0 | **flat** |
| 3D U-Net (det.)† | 0 | 0 | 0 | **flat** |
| 3D CVAE | 0 | 0 | 0 | **flat** |
| _ref_ Dich. Gaussian | 833,311 | 2,688,100 | 26,881,000 | linear |
| _ref_ Coupled GLM | 833,280 | 2,688,000 | 26,880,000 | linear |

The learned arms store nothing per assay: `gct` is a fixed random +/-1 code regenerated from seed 0 (`dataset.py:403-415`), a handle rather than a table. The lookup arms store a full (H, W) site map each, so their footprint grows linearly and is already ~3x our entire model at 1000 preparations -- while the part of them that is shared across assays is 26 fitted values (DG) and 5,194 (GLM).

The claim here is scalability and nothing wider. A random code is an assay HANDLE, so a new preparation still needs training exposure and no number in this report measures transfer to an unseen preparation; the split is temporal within assay. What is measured is that parameter cost does not grow with the number of preparations. `gct` also reaches the priors through exactly one frozen mapper (`CtxEmbed`, `main.py:1492`), so substituting measured descriptors -- unit map, ISI distribution, stimulation protocol, DIV, cell type -- for the random code is a change to that module alone. That is stated as future work, not as a result.

**Withhold the table and the capability goes with it.** Same models, same clips, per-assay site map replaced by the global one:

| _ref_ arm | within-assay gap: with map → without | adherence: with map → without |
|---|---|---|
| Dich. Gaussian | +0.0058 → +0.0011  (**5x worse**) | +0.5472 → +0.2076  (**3x worse**) |
| Coupled GLM | -0.0049 → -0.0004  (**11x worse**) | +0.5012 → +0.0188  (**27x worse**) |

The GLM's adherence goes NEGATIVE: without a per-assay map it does not merely degrade, it stops tracking the requested context at all. That is the sense in which these arms are ceilings rather than methods -- what they score is the map, and the map is exactly the thing that does not scale.

**What this does and does not claim.** Zero per-assay storage is not zero memorisation: assay-specific information can live in shared weights. The checkable claim is that parameter count does not grow with the number of assays. gct is a SEEDED RANDOM code, so a new assay still needs training exposure -- this is not a zero-shot transfer claim. We therefore make the narrow claim: parameter cost is flat in the number of preparations, and the fitted capacity is shared rather than per-assay. We do not claim zero-shot transfer to an unseen preparation, and no table here measures it.

## Task axis

Per-clip average precision inside the masked region, 279 clips, every clip under every task so the columns are paired. `model` is the Monte-Carlo mean over 8 samplings. Scored on ROI voxels only.

`recon (0)` is free generation, not completion: its ROI is all-TRUE, so nothing is visible. Read it as the ZERO-CONTEXT reference, not a fourth peer.

**Do not read across task columns.** The ROI fraction differs by task (1.00 / 0.59 / 0.40 / 0.34), so the columns have different denominators and different base rates. Comparisons are valid WITHIN a column.

⚠ **Dich. Gaussian** has no completion mechanism: its clip-level output is a static per-site probability, so it cannot read the visible remainder. Its row is FREE GENERATION scored on the hole -- a capability statement, not a like-for-like score.

⚠ **3D U-Net (det.)†** is the DIRECT-SUPERVISION reference, not a peer generator: it is trained with this exact loss on this exact hole distribution, and AP is a ranking metric, so a conditional-mean regressor is the best answer the table can contain. It is here to show what the generative arms have to buy their way past, and what it costs -- it carries no latent, no samples and no reusable representation. Compare it with `3D CVAE`, which is the same network with a latent variable and nothing else changed.


### Spatiotemporal AP ↑

**Which VOXEL fires, and when.** Each arm emits a score for every voxel inside the hole; the voxels are ranked by that score and scored against the truth with step-wise average precision, one clip at a time, then averaged over clips. A candidate is one (t, y, x) voxel of the ROI; a positive is a voxel that really spikes. This is the full completion problem -- an arm must get the electrode AND the frame right to earn credit.

| model ↑ | recon (0) = free gen | causal (1) | noncausal (2) | spatial (3) |
|---|---|---|---|---|
| Ours (4C+soft) | 0.0127 | 0.0137 | 0.0154 | 0.0200 |
| MaskGIT-flat | 0.0193 | 0.0184 | 0.0209 | 0.0181 |
| 3D U-Net (det.)† | **0.0269** | **0.0359** | **0.0367** | **0.0255** |
| 3D CVAE | 0.0250 | 0.0252 | 0.0257 | 0.0209 |
| | | | | |
| _ref_ Dich. Gaussian | 0.0545 | 0.0574 | 0.0611 | 0.0682 |
| _ref_ Coupled GLM | 0.0688 | 0.0737 | 0.0777 | 0.0806 |
| _ceiling_ Ours (4C+soft) true tokens | **0.2535** | **0.3046** | **0.2725** | **0.3289** |
| _ceiling_ MaskGIT-flat true tokens | 0.0628 | 0.0692 | 0.0635 | 0.0859 |
| | | | | |
| _null_ assay site map, SEEN | **0.0670** | **0.0695** | **0.0719** | **0.0835** |
| _null_ assay site map, UNSEEN | 0.0022 | 0.0021 | 0.0030 | 0.0034 |
| _null_ visible profile x site map | -- | 0.0685 | **0.0719** | 0.0732 |
| _null_ persistence | -- | 0.0273 | 0.0406 | -- |

A candidate is one voxel of the ROI. Row definitions (`_chance_`, `_ceiling_`, `_null_`) follow the next table.

**Every learned arm loses to a static per-assay site map.** The row below is the same canonical `_null_ assay site map, SEEN` already in the table above, now scored PAIRED against each arm on the same clips. It contains no clip-specific information whatsoever.

| median delta, lookup - arm (% of clips lookup wins) | recon (0) = free gen | causal (1) | noncausal (2) | spatial (3) |
|---|---|---|---|---|
| 3D U-Net (det.)† | +0.0298 (99%) | +0.0286 (96%) | +0.0294 (94%) | +0.0424 (96%) |
| 3D CVAE | +0.0352 (99%) | +0.0359 (99%) | +0.0369 (96%) | +0.0461 (99%) |

All 8 comparisons favour the lookup, BH-FDR q <= 1.2e-38. Positive means the lookup wins. Null from the canonical model-free block; pairing verified by NaN-mask equality per task, since the per-model reports carry no clip keys.


## Count calibration -- how many spikes, per clip

The per-task battery (appendix) scores binarised volumes, and it sets the spike count to to `round(assay_train_rate x |ROI|)`. That is ONE NUMBER PER ASSAY. Real clips inside an assay differ a lot -- true ROI counts have sd 100 spikes on the first task below -- so the readout cannot express per-clip activity even in principle, and it gives every model the same count regardless of what the model predicted.

`within-assay r` is the correlation with the TRUE ROI count after removing each assay's mean. `--` means the quantity is a constant within the assay and has no correlation to compute; that is the shared readout's row, and it is the point of this table.

| within-assay r with true ROI count | recon (0) = free gen | causal (1) | noncausal (2) | spatial (3) |
|---|---|---|---|---|
| _null_ assay train rate (the shared readout) | -- | +0.6015 | +0.2062 | +0.1209 |
| _null_ lct arithmetic, no model | +1.0000 | +0.8328 | +0.6263 | +0.4550 |
| Ours (4C+soft), own predicted count | **+0.9506** | **+0.9638** | **+0.9138** | **+0.9550** |
| MaskGIT-flat, own predicted count | +0.7577 | +0.3077 | +0.2394 | -0.1147 |
| 3D U-Net (det.)†, own predicted count | +0.4811 | +0.6683 | +0.3590 | +0.5303 |
| 3D CVAE, own predicted count | +0.4686 | +0.7374 | +0.3885 | +0.2091 |
| | | | | |
| _ref_ Dich. Gaussian, own predicted count | +0.9221 | +0.7371 | +0.5222 | +0.6014 |
| _ref_ Coupled GLM, own predicted count | +0.0362 | +0.6071 | +0.2806 | +0.8531 |

**Read this against the `lct arithmetic` row, not against zero.** Every arm is handed the TRUE clip's lct at generation time, and its first feature is `log_mean_firing_density` -- the clip's total count. So `exp(lct[0]) x |ROI|` predicts the ROI count with no model whatsoever. On `recon` that control scores +1.0000: the ROI is the whole volume there, so lct[0] IS the answer and no count head can beat arithmetic. On the three tasks where the ROI is a strict subset the control is much weaker, and that gap is where a count head actually earns its place.


Absolute accuracy of the same counts -- MAE in spikes, and bias as a fraction of the true mean:

| count MAE / bias | recon (0) = free gen | causal (1) | noncausal (2) | spatial (3) |
|---|---|---|---|---|
| _readout_ assay train rate (shared)  **calibrated** | 41.9 / -0.7% | 30.8 / +4.9% | 20.9 / +1.0% | 36.7 / -15.3% |
| Ours (4C+soft), own predicted count | 1001.9 / +638.5% | 531.5 / +613.2% | 397.0 / +645.2% | 405.5 / +651.6% |
| MaskGIT-flat, own predicted count | 182.2 / -94.0% | 96.0 / -92.3% | 74.1 / -91.6% | 82.9 / -96.3% |
| 3D U-Net (det.)†, own predicted count | 153.9 / +82.0% | 108.5 / +119.0% | 130.8 / +211.7% | 215.1 / +345.0% |
| 3D CVAE, own predicted count | 112.5 / +55.9% | 67.8 / +69.8% | 75.8 / +119.9% | 206.2 / +326.8% |
| _ref_ Dich. Gaussian, own predicted count | 14858.5 / +9470.1% | 8691.3 / +10028.0% | 5891.0 / +9573.1% | 5070.4 / +8147.4% |
| _ref_ Coupled GLM, own predicted count  **calibrated** | 42.6 / -5.5% | 29.9 / +1.6% | 20.1 / -2.9% | 16.6 / -4.2% |

**calibrated** marks an arm whose count bias stays within +/-25% on every task. Nothing is bolded on value: the best learned arm here is still ~6x over-count, and the only arm that is calibrated is a per-assay lookup.

**MaskGIT-flat's own-count row is not a meaningful measurement and should not be read as one.** It has no count head; the number is the ROI sum of `sigmoid` over its tokenizer decoder, which is trained with `pos_weight`. Raw, that decoder overstates the log-odds by `log w` ~ 9.0 nats and the row read about +3800%. The cost-sensitive inversion (Elkan, IJCAI 2001) that fixes the two 3-D conv arms -- bias +16.7% and +97.3% on `recon` -- overshoots here to about -94%, because that inversion assumes a head near the weighted-BCE minimiser and this one is a SATURATING VQ decoder applied to sampled tokens. Both numbers are wrong in different directions and neither is quoted as its calibration. The correction is applied for consistency with the other arms, not because it calibrates this one; choosing between +3800% and -94% on which looks better would be tuning to the metric. Every binarised column is unaffected either way -- the readout solves for a per-clip shift that absorbs any constant offset exactly.

**What the count actually buys: `log_mean_firing_density` under both readouts.** This feature is `log(mean + 1e-6)` over a fixed volume, so it measures the spike COUNT and nothing else -- it is the one lct feature that a count prediction can move. MAE against the clip's own lct:

| `log_mean_firing_density` MAE ↓ | recon (0) = free gen | causal (1) | noncausal (2) | spatial (3) |
|---|---|---|---|---|
| rate-matched (identical for every model) | 0.3440 | 0.2491 | 0.1710 | 0.3284 |
| own count: Ours (4C+soft) | 1.9781 | 1.4100 | 1.2260 | 1.0765 |
| own count: MaskGIT-flat | 3.1868 | 0.9863 | 0.5292 | 0.6668 |
| own count: 3D U-Net (det.)† | 0.7758 | 0.5434 | 0.6122 | 0.8451 |
| own count: 3D CVAE | 0.5177 | 0.4305 | 0.4584 | 0.7453 |
| own count: Dich. Gaussian | 4.8111 | 4.2497 | 3.8947 | 3.7506 |
| own count: Coupled GLM | **0.3378** | **0.2461** | **0.1648** | **0.1452** |

Under the rate-matched readout the row is a CONSTANT: every model writes the same number of spikes, so the count error is the readout's, not the model's. Let each model supply its own count and the row separates them completely -- which is the point of having both readouts, and the direct answer to what count prediction is worth here.


**Discrimination and calibration are separate capabilities, and no arm has both.** The two tables above disagree about who wins, which is the finding rather than a contradiction: ranking which clip is busier within an assay, and knowing how many spikes that means, are different problems. An arm built on a decode probability field ranks well and is scaled badly, because the ROI sum of such a field is not a calibrated expected count; a point process fitted with an explicit rate gets the level right and discriminates poorly. The shared readout in the battery above is a third point: correct level, zero discrimination. That is why the battery uses it -- it is the only one comparable across arms -- and why this table exists separately.


**Reading the ceilings.** At voxel level the two priors score within 0.007 of each other, but they sit on tokenizers that are not comparable: on `recon` the ceiling is **0.2535** for Ours (4C+soft) against **0.0628** for MaskGIT-flat, a factor of **4.0**. At SITE level the same two ceilings are 0.2296 and 0.2619 -- the hierarchical alphabet buys resolution in TIME, not in space. What each prior then recovers of its own ceiling:

| model_mc / own oracle | recon (0) = free gen | causal (1) | noncausal (2) | spatial (3) |
|---|---|---|---|---|
| voxel: Ours (4C+soft) | 5% | 4% | 6% | 6% |
| voxel: MaskGIT-flat | **31%** | **27%** | **33%** | **21%** |
| site: Ours (4C+soft) | **98%** | **84%** | **80%** | **97%** |
| site: MaskGIT-flat | 50% | 48% | 51% | 43% |

The two rows point opposite ways. Our prior recovers most of what its alphabet allows for WHERE and almost none of it for WHEN, so the binding constraint on us is the prior's timing, not the tokenizer. MaskGIT-flat recovers a larger share at voxel level but of a ceiling five times lower, and only about half at site level: its binding constraint is the flat alphabet. Above 100% is possible because `model_mc` averages 8 samplings into a smoother ranking while the oracle is a single decode of the true tokens; the oracle bounds what the ALPHABET can express, not what an averaged field can score.


## Reconstruction

Step-wise average precision, not the trapezoid AUPRC in `utils/metrics.py` -- trapezoid interpolates the PR curve linearly, which is invalid (Davis & Goadrich 2006) and inflated the saturating MaskGIT tokenizer by +0.22 off a single voxel.

| | Ours (4C+soft) | MaskGIT-flat | 3D U-Net (det.)† | 3D CVAE | _ref_ Dich. Gaussian | _ref_ Coupled GLM |
|---|---|---|---|---|---|---|
| AP step-wise, exact ↑ | **0.2310** | 0.0257 | -- | -- | -- | -- |
| AP step-wise, tolerant ↑ | **0.2428** | 0.0466 | -- | -- | -- | -- |
| best F1, exact ↑ | **0.2888** | 0.0635 | -- | -- | -- | -- |
| best F1, tolerant ↑ | **0.2908** | 0.0976 | -- | -- | -- | -- |
| trapezoid inflation ↓ | **+0.0019** | +0.0368 | -- | -- | -- | -- |
| codebook used | 846 | 923 | -- | -- | -- | -- |
| codebook perplexity | 522.8 | 249.5 | -- | -- | -- | -- |

Empty cells are a capability fact, not a missing run. This section measures what an ALPHABET can represent, so it needs a tokenizer: DG and the GLM are point processes and have none, and the two 3-D conv arms are inpainters with no autoencoding path -- they never see the clip they are asked to complete, so there is nothing for them to reconstruct. The comparison that does include them is **Task completion**.

`_ref_` rows are per-assay LOOKUP TABLES, not peer models: they store one site map per preparation and are shown as memorisation ceilings. Bold competes between the learned generative models only. See **Scalability** for what each arm stores per assay.


## Generation

Four families, no composite. Each cell is the metric at full context (`glob+full`) with that same model's `random`-context value in brackets -- its own null, so the bracket says how much of the number is conditioning rather than the model's default behaviour. The full five-rung ladders are in the appendix.

| metric | Ours (4C+soft) | MaskGIT-flat | 3D U-Net (det.)† | 3D CVAE | _ref_ Dich. Gaussian | _ref_ Coupled GLM |
|---|---|---|---|---|---|---|
| A. Conditional accuracy ↓ | **0.5350** _[1.062]_ | 0.7362 _[1.189]_ | 0.6803 _[1.074]_ | 0.7441 _[1.139]_ | 0.5137 _[0.724]_ | 0.5610 _[0.545]_ |
| B. Adherence ↑ | **0.6992** _[0.718]_ | 0.5429 _[0.561]_ | 0.5165 _[0.505]_ | 0.4925 _[0.487]_ | 0.5472 _[0.178]_ | 0.5012 _[0.006]_ |
| C. Spatial placement, lookup-proof ↑ | **0.0101** _[0.002]_ | 0.0027 _[-0.001]_ | 0.0035 _[-0.001]_ | 0.0052 _[-0.001]_ | 0.0058 _[-0.001]_ | -0.0049 _[0.005]_ |
| D. Marginal realism ↓ | 0.2242 _[0.247]_ | 0.3661 _[0.368]_ | **0.1025** _[0.076]_ | 0.3484 _[0.345]_ | 0.0640 _[0.145]_ | 0.0268 _[0.031]_ |

`value _[null]_`, the null being the same arm given random context. An arm whose value and null are close is not using its context -- which is what the `_ref_` lookups do once their per-assay map is withheld (see **Scalability**).

**3D U-Net (det.) is marked `det.` because it has no sampling distribution.** It is trained to predict the conditional mean of the hole, so one context gives one field and every "sample" from it is the same field re-thresholded. That is the optimal answer to a ranking question (AP, F1) and vacuous as an answer to a distributional one (avalanche, ISI, marginal realism), so its two kinds of column must not be read the same way. `3D CVAE` is the same backbone with a latent variable added and nothing else changed, so the gap between the two arms is what stochasticity costs and buys here.

† **3D U-Net (det.) is not a representation learner, and its reconstruction numbers are not comparable to a tokenizer's.** It is a U-Net: `forward` concatenates the full-resolution stem output back in on the way up, so an uncompressed path runs from input to output and the decoder reads AROUND the compressed layer. Per clip it carries 54.8M floats across its skips against an input of 1.29M binary voxels -- the full-resolution skip alone holds 32x more floats than the volume has voxels. It produces **no codebook, no discrete index, and no reusable latent**: nothing is shared across clips, nothing is indexable, and there is no bottleneck a prior could be trained over. Our alphabet is 1024 tokens over V=961, i.e. 1.24 KB per clip. The skips are what win it the ranking columns and are precisely what disqualify it as a tokenizer; the two cannot be had together, because a tokenizer's value comes from forcing everything through the code.

`_ref_` rows are per-assay LOOKUP TABLES, not peer models: they store one site map per preparation and are shown as memorisation ceilings. Bold competes between the learned generative models only. See **Scalability** for what each arm stores per assay.


## Design choices: why this tokenizer

### Patch size -- why (6,15,14)

Model-free: one pass over 48 val clips at voxel rate 1.622e-04, nothing trained. Two quantities move in OPPOSITE directions as the patch changes, and the shipped size is where both are still tolerable.

- **blank** is the fraction of tokens containing no spike. It is what the prior is trained against: once nearly every target is blank, predicting blank is close to optimal and the prior collapses.
- **capture@32** is the fraction of active-patch variance that 32 centroids can describe (k-means on the raw patches, K = the parent codebook size). It is the ALPHABET side: if 32 entries cannot describe the patch distribution, no amount of training fixes it.

| patch | voxels | tokens | blank  ↓ | active tok/clip | spikes/active | spike sd | capture@32  ↑ |
|---|---|---|---|---|---|---|---|
| (24, 15, 14) | 5040 | 256 | 79.6% | 52 | 4.01 | 3.77 | 0.060 |
| (12, 30, 14) | 5040 | 256 | 76.0% | 61 | 3.41 | 3.37 | 0.062 |
| (6, 30, 28) | 5040 | 256 | 73.1% | 69 | 3.04 | 3.08 | 0.070 |
| (12, 15, 14) | 2520 | 512 | 84.5% | 79 | 2.65 | 2.32 | 0.061 |
| (6, 30, 14) | 2520 | 512 | 81.8% | 93 | 2.25 | 2.05 | 0.068 |
| (24, 15, 7) | 2520 | 512 | 86.4% | 69 | 3.01 | 2.71 | 0.056 |
| **(6, 15, 14)** _<- shipped_ | 1260 | 1024 | 88.9% | 114 | 1.84 | 1.43 | 0.087 |
| (3, 15, 14) | 630 | 2048 | 92.7% | 149 | 1.41 | 0.87 | 0.138 |
| (6, 15, 7) | 630 | 2048 | 93.4% | 134 | 1.56 | 1.05 | 0.106 |
| (6, 10, 7) | 420 | 3072 | 95.3% | 143 | 1.46 | 0.91 | 0.126 |
| (3, 15, 7) | 315 | 4096 | 95.9% | 167 | 1.26 | 0.64 | 0.181 |
| (3, 10, 7) | 210 | 6144 | 97.2% | 174 | 1.20 | 0.55 | 0.228 |
| (2, 8, 7) | 112 | 11520 | 98.4% | 189 | 1.11 | 0.38 | 0.380 |

**Smaller patches: the alphabet gets easier and the prior gets impossible.** From (6, 15, 14) to (2, 8, 7) capture@32 rises 0.087 -> 0.380, because a patch holding 1.11 spikes is easy to describe. But blank goes 88.9% -> 98.4%, and active tokens barely move (114 -> 189) while the grid grows 11x. The same signal is spread over far more positions, each almost always empty. Reconstruction IMPROVES here -- finer patches decode more sharply -- while generation collapses.

**Larger patches: the prior gets easier and the alphabet cannot keep up.** At (24, 15, 14) blank falls to 79.6%, which is what the prior wants, but a non-blank token now holds 4.01 spikes with sd 3.77 against 1.84 / 1.43 shipped, and capture@32 drops 0.087 -> 0.060. Thirty-two entries cannot span that much heterogeneity, so the cost reappears as quantization error.

**The quantizer has no mechanism to absorb that.** Every codebook tensor is created with `requires_grad = False` (`model/base.py:200`) and updated only by EMA, so the code vectors receive no gradient from any loss. The one usage term, `max_entropy - entropy` at weight 1e-3 (`model/base.py:424`), is computed from the distances between ENCODER outputs and a no-grad codebook, so its gradient reaches the encoder alone: it is an encoder-side anti-collapse pressure that spreads which entry gets hit, an INDIRECT influence on the entry point rather than anything acting on the entries. It also pushes usage toward UNIFORM, which is the opposite of a sparsity prior. Nothing in the objective can enlarge what 32 entries are able to span.

So the patch is chosen for the PRIOR and the ALPHABET jointly, not for the decoder: reconstruction alone would pick the smallest patch on the table. The shipped size also fixes the token budget at 1024, which is what makes a clip fit in memory, and keeps enough spikes per token for the variance and covariance context features to be legible. Blank fraction is the same quantity that binds in **Sparse encoder** below.

_capture@32 is k-means on RAW binary patches, so it measures the intrinsic diversity of the content at each patch size. It is not a measurement of our tokenizer, which quantizes 64-d encoder outputs rather than raw voxels; read it as a relative comparison ACROSS patch sizes, never as an absolute ceiling._

### Ladder depth -- what each hierarchy level buys

Encode ONCE with all three levels, then decode the same codes at each cumulative depth. Nothing is retrained and nothing is re-encoded, so the only thing that varies is how much of the residual sum reaches the decoder. Exact step-wise AP, paired over the same 64 clips.

| decode depth | AP  ↑ | gain over previous |
|---|---|---|
| `z1` | 0.0772 | -- |
| `z1+z2` | 0.1604 | **+0.0832** (100% of clips, p=3.5e-12) |
| `z1+z2+z3` | 0.2806 | **+0.1202** (100% of clips, p=3.5e-12) |

**Every level earns its place, and z3 earns the most.** It adds more than z2 and wins on every clip. A separate content analysis -- eta^2 of the ladder path on spike count and temporal spread -- puts z3 near zero; that measurement is blind here, because those summary statistics cannot resolve exact within-patch PLACEMENT, which is what z3 carries. Where the question is what the decoder can render, it is measured at the decoder.

### Sparse encoder and the blank token

91.7% of tokens are blank. The shipped encoder routes those to one learned `blank_token` and quantizes only the ~8% carrying a spike; the `dense` arm declares every token active, so blank patches -- which are exactly the zero vector -- are projected and quantized like any other. Identical seed, data order, optimizer, schedule and losses otherwise.

Measured on CONTENT tokens only, using the true blank mask recomputed from the volume. The shipped `active_codes_nonblank` counters cannot be compared across these two arms: they mean *declared active*, and the dense arm declares everything, so its counters cover all 1024 tokens while the sparse arm's cover the ~8% with content.

| level-1 parent codebook | sparse | dense |
|---|---|---|
| codes used on content (of 32) | **32** | 25 |
| usage entropy, nats | **3.054** | 2.731 |
| effective alphabet (perplexity) | **21.2** | 15.4 |
| codes ALSO used by blank patches | **0** | 13 |
| eta^2 code -> spike count | 0.304 | **0.316** |
| val AUPRC, exact | **0.0501** | 0.0358 |

**The sparse encoder keeps the whole parent alphabet for content and keeps every code unambiguous about emptiness.** The dense arm spends 7 parent codes on nothing and leaves 13 of its remaining 25 (52%) shared between spiking and empty patches, so the code no longer says whether the patch is occupied. Note eta^2 is NOT worse for the dense arm -- the codes it does use still describe their patches; there are simply fewer of them and 52% are ambiguous.

Both arms ran 80 epochs against the shipped tokenizer's 300, so these are early-training numbers and valid only as a PAIRED comparison. At 300 epochs the shipped level-1 codebook reaches perplexity 29.8 of 32 and effective rank 8.29, against 21.2 and 3.67 here.

