"""Stage 4D: adapt the motif prior to the activity maps Stage 4B actually emits.

WHY THIS AND NOT STAGE 4C

4C froze 4A and tuned 4B. That direction is handicapped three ways: the sampled
activity map is discrete, so the gradient into 4B is a straight-through
surrogate through a top-K selection whose true Jacobian is zero almost
everywhere; the trainable surface (cell_head / maskgit_norm / count_head) cannot
change the RANKING that decides which cells light; and "make the map closer to
truth" is exactly what 4B was already trained to do directly, with full gradient
access and a proper scoring rule. Measured outcome: 56 epochs, every candidate
inside one seed sd, motif MRR declining at t = -10.45.

This module runs the other direction. 4B is FROZEN and 4A is fine-tuned on the
maps 4B emits. Now the sampling distribution does not depend on the parameters
being optimised, so this is ordinary supervised learning with unbiased
gradients over a fixed distribution.

WHAT THERE IS TO LEARN

4A trained on 100% teacher-forced ground-truth activity. The Bayes-optimal
predictor given a noisy map marginalises over the map's error,

    p(f | a_hat, c) = sum_a* p(f | a*, c) p(a* | a_hat, c)

while 4A computes p(f | a* = a_hat, c) -- it treats the map as exact. Closing
that gap needs NO new information: the correction depends only on (a_hat, c),
and 4A observes both. Note the prefix c reaches 4A twice, directly and through
a_hat, and that redundancy is precisely what lets it learn to discount a cell
the map lit but the context contradicts.

Concretely, measured at 4B's ROI recall of 0.72: when 4B misses a truly-active
cell, `f_in` there is set to `f_null_id` -- the conditioning asserts "this cell
is off" -- and under teacher forcing that input configuration has probability
EXACTLY ZERO. This is not a refinement of an estimate; it is a region of input
space the model has never visited.

THE SOFT FIELD

4B computes a calibrated probability per cell and the generation interface then
does `activity.long().clamp(0, 1)` (inference/sample_prior.py), throwing the
uncertainty away. But the marginalisation above needs exactly that uncertainty.
So the adapted arm passes the SOFT field via the `activity_prob` argument added
to MaskGITMotifPrior.forward, which reduces to the hard `a_in` path bit-exactly
at p in {0,1} (verified: max |logit delta| = 0.000e+00).

WHAT IS GRADED

Only `m & gt_active` -- masked positions where a true code exists. Cells 4B lit
that are not truly active have no target, so they are force-masked (never shown
a fake code) and excluded from the loss. Cells 4B missed are not filled at
generation time either; they carry `f_null_id` and contribute conditioning
noise, which is the thing being adapted to.
"""
from typing import Dict, Optional, Tuple

import torch


@torch.no_grad()
def build_adapted_motif_io(
    *,
    motif_prior,
    activity_prior,
    codes: torch.Tensor,
    predict_mask: torch.Tensor,
    global_ctx: torch.Tensor,
    local_ctx: torch.Tensor,
    task_id: torch.Tensor,
    blank_code: int,
    p_model: float,
    readout: str = "gumbel",
    readout_tau: float = 1.0,
    full_mask_prob: float = 0.15,
    ensure_at_least_one_mask: bool = True,
    use_soft_field: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """Motif inputs built from 4B's activity map for a Bernoulli(p_model) subset.

    Returns (a_in, f_in, targets, activity_prob). At p_model = 0 this matches
    ``corrupt_inputs_from_targets`` in distribution, so a ramp starting at 0
    begins from Stage 4A's own training regime.
    """
    from .stage4_activity import _make_activity_in_from_codes

    targets = motif_prior.make_targets_from_codes(
        codes=codes, predict_mask=predict_mask, blank_code=blank_code
    )
    gt_active = targets["active"].bool()
    roi = targets["predict_mask"].bool()
    f_true = targets["f"].long().clamp(0, motif_prior.V - 1)
    B, N = gt_active.shape
    device = gt_active.device

    # ---- frozen 4B -------------------------------------------------------
    activity_in = _make_activity_in_from_codes(
        codes, predict_mask, blank_code=blank_code, a_mask_id=activity_prior.a_mask_id
    )
    out = activity_prior(
        global_ctx=global_ctx, local_ctx=local_ctx, task_id=task_id,
        a_in=activity_in, roi_mask=predict_mask,
        count_target=None, count_teacher_prob=0.0,
    )
    hard = activity_prior.sample_hard_activity_for_generation(
        out, roi_mask=predict_mask, readout=readout, tau=readout_tau,
        count_mode="expected",
    ).bool()
    soft = torch.sigmoid(out["cell_logits"].float())
    if soft.dim() == 3:
        soft = soft.squeeze(-1)

    # ---- per-sample ramp -------------------------------------------------
    use_model = (torch.rand((B, 1), device=device) < float(p_model))

    # Outside the ROI the activity is genuinely observed, so it stays true in
    # both arms; only the predicted region is substituted.
    active = torch.where(roi & use_model, hard, gt_active)

    if use_soft_field:
        prob = active.float()
        blend = roi & use_model
        prob = torch.where(blend, soft.clamp(0.0, 1.0), prob)
    else:
        prob = None

    # ---- MaskGIT corruption over the ADAPTED activity --------------------
    gamma = torch.rand((B, 1), device=device)
    if full_mask_prob > 0:
        full = torch.rand((B, 1), device=device) < float(full_mask_prob)
        gamma = torch.where(full, torch.ones_like(gamma), gamma)

    valid = roi & active
    m = (torch.rand((B, N), device=device) < gamma) & valid
    # A cell 4B lit that is not truly active has no ground-truth code, so it can
    # never be shown as "visible" -- there is nothing truthful to show. Force it
    # masked, which is also what inference does with every predicted-active cell.
    m = m | (valid & ~gt_active)

    if ensure_at_least_one_mask:
        for b in range(B):
            idx = torch.where(valid[b])[0]
            if idx.numel() > 0 and not m[b].any():
                m[b, idx[torch.randint(idx.numel(), (1,), device=device)]] = True

    f_in = f_true.clone()
    f_in[~active] = motif_prior.f_null_id
    f_in[m] = motif_prior.f_mask_id
    a_in = active.long()

    targets = dict(targets)
    targets["a"] = a_in
    targets["active"] = active
    targets["a_loss_mask"] = m
    # Graded only where a true code exists. Excludes 4B's false positives, which
    # have no target, and includes nothing 4A cannot in principle get right.
    targets["f_loss_mask"] = m & gt_active
    targets["z_loss_mask"] = targets["f_loss_mask"]
    # The trainer logs the realised corruption rate from this.
    targets["gamma"] = gamma
    targets["gt_active"] = gt_active
    targets["adapted_fraction"] = use_model.float().mean()

    return a_in, f_in, targets, prob


def adapt_probability(epoch: int, ramp_epochs: int, max_p: float) -> float:
    """Linear ramp, 0 at epoch 0 and max_p from ramp_epochs on.

    Starting at 0 means epoch 1 trains in Stage 4A's own regime, so any early
    divergence is the substitution and not a change of setup. Stopping at
    max_p < 1 keeps some teacher forcing as a regulariser and preserves the
    oracle-activity capability the inpainting tasks still need.
    """
    if ramp_epochs <= 0:
        return float(max_p)
    return float(max_p) * min(1.0, max(0.0, (epoch - 1) / float(ramp_epochs)))
