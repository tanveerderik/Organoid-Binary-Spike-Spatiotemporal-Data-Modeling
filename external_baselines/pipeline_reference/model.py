"""The shipped pipeline, wrapped in the external-baseline protocol.

Not a baseline. This is the model under test, dressed so that
`external_baselines/diagnose.py` can measure it with *the same code* that
measured DG, the coupled GLM and MaskGIT-flat. Without it the diagnostic table
has three columns and a blank where the answer should be, and any comparison
has to be made by eye across two scripts.

Two things this adapter is careful about.

**It is the same generation path.** `sample()` reproduces
`analysis/generate_regimes.py::generate` call for call -- activity prior ->
`sample_hard_activity_for_generation` -> `iterative_unmask_motif_given_activity`
with the soft field -> `decode_flat_ids_to_xgen` at `best_thr_tol`. If that file
changes, this one has to change with it; there is no second implementation of
the model here, only a second way of calling it.

**It cannot see the clip.** `ConditioningBatch` carries no `x`, but the
pipeline's generate() takes an `x_ref` to encode. Under free generation those
codes are provably discarded:

  * the ROI is all-True, and `_make_activity_in_from_codes` overwrites every
    position inside the ROI with `a_mask_id`, so `a_in` is constant;
  * `iterative_unmask_motif_given_activity` computes `visible_active` as
    `vis_active & ~roi`, which is empty for an all-True ROI.

So this adapter passes a ZERO volume as `x_ref` -- structurally the same
guarantee the other baselines get, rather than a promise that the codes go
unused. `tests/test_pipeline_reference.py` checks the two paths agree bitwise,
which is what turns the argument above into a measurement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import torch

from ..common.protocol import BaselineMeta, ConditioningBatch, SpikeVolumeBaseline
from ..common.pipeline import build_pipeline
from ..registry import register

from MAGVIT_project.inference.sample_prior import iterative_unmask_motif_given_activity
from MAGVIT_project.inference.decode import decode_flat_ids_to_xgen
from MAGVIT_project.training.train_prior import (
    _vq_codes_and_pmask_for_prior, _make_activity_in_from_codes)


class _TokenizerShim(torch.nn.Module):
    """`(logits, flat_ids, None)` from the VQ-VAE, matching the baseline tokenizers.

    The pipeline's encoder is context-conditioned, so unlike the flat tokenizers
    this one needs gct/lct. `diagnose.reconstruction` passes them when
    `tokenizer_wants_cond` is set.
    """

    def __init__(self, vqvae, motif_prior, blank_code: int):
        super().__init__()
        self.vqvae = vqvae
        self.motif_prior = motif_prior
        self.blank_code = int(blank_code)

    @torch.no_grad()
    def forward(self, x, cond: Optional[ConditioningBatch] = None):
        gct = None if cond is None else cond.global_ctx
        lct = None if cond is None else cond.local_ctx
        out = self.vqvae(
            x, global_ctx=gct, local_ctx=lct,
            # No mask spec and no latent hole: the question in section 1 is
            # "can the tokenizer represent this clip", so nothing is hidden from
            # the decoder. This is deliberately NOT the reported AUPRC_cond,
            # which is scored inside a predict mask with recon_drop_p=0.5.
            predict_mask_spec=None,
            roi_hw=None if cond is None else cond.roi_hw,
            pad_hw=None if cond is None else cond.pad_hw,
        )
        logits = out["logits_vol"]
        flat, active = self.motif_prior.flat_ids_from_codes(
            out["codes"].long(), blank_code=self.blank_code)
        # Inactive tokens are the blank slot, not a codebook entry; give them
        # their own bin so codebook-usage counts aren't inflated by blanks.
        V = int(self.motif_prior.flat_codebook.shape[0])
        ids = torch.where(active, flat.clamp(0, V - 1),
                          torch.full_like(flat, V))
        return logits, ids, None


class PipelineReference(SpikeVolumeBaseline):

    meta = BaselineMeta(
        name="pipeline-4C-soft",
        citation="this work",
        family="voxel+token",
        conditioning=(
            "gct (64-d) and lct (9-d) enter the activity prior and the motif "
            "prior as prefix context, and gct additionally reaches the decoder "
            "through the Stage-1 spatial memory bank."),
        notes=(
            "Free generation: all 1024 tokens masked, so no visible code from "
            "the held-out clip is read. Binarised at the F1-selected "
            "best_thr_tol, the shipped readout."),
    )

    def __init__(self, phase: str = "4b", activity_field: str = "soft",
                 motif_steps: int = 12, motif_temperature: float = 1.0,
                 motif_top_k: int = 5, activity_readout: str = "gumbel",
                 activity_readout_tau: float = 1.0,
                 activity_count_scale: float = 1.0, task_id: int = 0):
        self.cfg: Dict[str, Any] = dict(
            phase=phase, activity_field=activity_field, motif_steps=motif_steps,
            motif_temperature=motif_temperature, motif_top_k=motif_top_k,
            activity_readout=activity_readout,
            activity_readout_tau=activity_readout_tau,
            activity_count_scale=activity_count_scale, task_id=task_id)
        self._p = None
        self._last_cond = None
        self._last_logits = None

    # -- lifecycle -------------------------------------------------------
    def fit(self, clips: Iterator[Dict[str, Any]], *, device: str) -> None:
        raise NotImplementedError(
            "The pipeline is trained by main.py's stages, not by the baseline "
            "harness. Use `load()`.")

    def save(self, path: Path) -> None:
        raise NotImplementedError(
            "Refuses to write: the pipeline checkpoints are the shipped "
            "artifacts and this adapter must never be able to touch them.")

    def load(self, path: Optional[Path] = None, *, device: str = "cuda") -> None:
        """`path` is ignored -- the checkpoints come from main.CKPTS, so this
        adapter always reflects whatever is actually shipped."""
        self.device = device
        self._p = build_pipeline(device, phase=self.cfg["phase"])
        self.cfg["n_codes"] = int(self._p["motif_prior"].flat_codebook.shape[0])
        self.tokenizer = _TokenizerShim(
            self._p["vqvae"], self._p["motif_prior"], self._p["blank_code"])
        self.tokenizer_wants_cond = True
        self.meta.extra.update({k: self._p[k] for k in
                                ("vq_ckpt", "motif_ckpt", "activity_ckpt",
                                 "best_thr_tol")})

    # -- generation ------------------------------------------------------
    @torch.no_grad()
    def _generate(self, cond: ConditioningBatch, generator=None, x_ref=None):
        """`x_ref` exists ONLY for tests/test_pipeline_reference.py, which feeds
        the real clip and requires a bitwise-identical result. Generation never
        passes it, so nothing in the measured path can read a held-out volume."""
        p = self._p
        vqvae, activity_prior, motif_prior = (
            p["vqvae"], p["activity_prior"], p["motif_prior"])
        dev = torch.device(self.device)
        gct = cond.global_ctx.to(dev).float()
        lct = cond.local_ctx.to(dev).float()
        B = gct.shape[0]
        T, H, W = cond.shape

        # Zero x_ref: see the module docstring. Every code it produces is
        # discarded downstream, and the test proves it.
        if x_ref is None:
            x_ref = torch.zeros((B, 1, T, H, W), device=dev)
        task_id = torch.full((B,), int(self.cfg["task_id"]),
                             device=dev, dtype=torch.long)
        recon_spec = [{"type": "recon"}] * B

        codes, pmask, grid = _vq_codes_and_pmask_for_prior(
            vqvae, x_ref, gct, lct, recon_spec, dev)
        token_grid = tuple(map(int, grid)) if grid is not None else (
            activity_prior.Ttok, activity_prior.Htok, activity_prior.Wtok)
        if not bool((pmask > 0.5).all()):
            raise RuntimeError(
                "recon spec did not yield a full ROI; free generation requires "
                "every token masked.")

        roi = torch.ones(codes.shape[0], codes.shape[1], device=dev, dtype=torch.bool)
        predict_mask = roi.unsqueeze(-1) if pmask.dim() == 3 else roi

        a_in = _make_activity_in_from_codes(
            codes, predict_mask, blank_code=p["blank_code"],
            a_mask_id=activity_prior.a_mask_id)
        a_out = activity_prior(global_ctx=gct, local_ctx=lct, task_id=task_id,
                               a_in=a_in, roi_mask=predict_mask,
                               count_target=None, count_teacher_prob=0.0)
        hard = activity_prior.sample_hard_activity_for_generation(
            a_out, roi_mask=predict_mask, readout=self.cfg["activity_readout"],
            tau=self.cfg["activity_readout_tau"], count_mode="expected",
            count_scale=self.cfg["activity_count_scale"]).long()

        a_prob = None
        if self.cfg["activity_field"] == "soft":
            a_prob = torch.sigmoid(a_out["cell_logits"].float())
            if a_prob.dim() == 3 and a_prob.size(-1) == 1:
                a_prob = a_prob.squeeze(-1)

        motif = iterative_unmask_motif_given_activity(
            motif_prior, activity=hard, global_ctx=gct, local_ctx=lct,
            task_id=task_id, roi_mask=roi, visible_codes=codes,
            steps=self.cfg["motif_steps"],
            temperature=self.cfg["motif_temperature"],
            top_k=self.cfg["motif_top_k"], activity_prob=a_prob)
        dec = decode_flat_ids_to_xgen(
            vqvae, motif["flat_ids"], flat_codebook=motif_prior.flat_codebook,
            grid=token_grid, global_ctx=gct, local_ctx=lct,
            roi_hw=cond.roi_hw, pad_hw=cond.pad_hw)
        return dec

    @torch.no_grad()
    def sample(self, cond, *, generator=None) -> torch.Tensor:
        dec = self._generate(cond, generator)
        # Cached so sample_intensity() reports the field this very sample came
        # from, rather than an independent second draw -- the degeneracy check
        # compares a field to the clip its sample was scored on.
        self._last_cond, self._last_logits = cond, dec["logits"]
        x = dec["x_gen"]
        return x.squeeze(1) if x.dim() == 5 else x

    @torch.no_grad()
    def sample_intensity(self, cond, *, generator=None):
        if cond is not self._last_cond or self._last_logits is None:
            dec = self._generate(cond, generator)
            self._last_cond, self._last_logits = cond, dec["logits"]
        lg = self._last_logits
        return lg.squeeze(1) if lg.dim() == 5 else lg


@register("pipeline")
def _build(**kw) -> PipelineReference:
    return PipelineReference(**kw)
