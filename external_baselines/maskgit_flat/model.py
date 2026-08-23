"""MaskGIT-flat: single-level 3D VQ tokenizer + vanilla MaskGIT prior.

The architectural baseline. It answers the reviewer who asks whether the
hierarchical alphabet and the where/what factorisation are doing any work, or
whether an ordinary masked video transformer on the same data would have got
there. Trained here from scratch on the current split -- the repository does
hold older single-level checkpoints from an internal ablation arm, but they
predate the current temporal split and reusing them would risk contamination.

Two fitting stages, in order:

  1. tokenizer -- Bernoulli reconstruction + VQ commitment.
  2. prior -- masked-token cross-entropy on the frozen tokenizer's codes.

Generation is free: every token starts masked, the prior fills them in by
parallel iterative decoding conditioned on gct/lct, the tokenizer decodes to
per-voxel logits, and those are binarised at the same rate-calibrated threshold
every other baseline gets.

Unlike the two statistical baselines this one lives in a token space, so it can
implement `tokenize()` and enter the token-family metrics as well -- though its
codebook is its own, so token-level numbers are comparable to the pipeline's
only in distribution, not code for code.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.protocol import BaselineMeta, ConditioningBatch, SpikeVolumeBaseline
from ..registry import register
from .prior import MaskGITPrior
from .tokenizer import GRID, FlatVQTokenizer, PATCH


class MaskGITFlat(SpikeVolumeBaseline):

    def __init__(
        self,
        *,
        n_codes: int = 1024,
        dim: int = 64,
        width: int = 128,
        d_model: int = 256,
        layers: int = 6,
        tok_epochs: int = 12,
        prior_epochs: int = 30,
        tok_lr: float = 2e-4,
        prior_lr: float = 3e-4,
        commit_weight: float = 0.25,
        steps: int = 12,
        temperature: float = 1.0,
        device: str = "cuda",
    ):
        self.cfg = dict(n_codes=n_codes, dim=dim, width=width, d_model=d_model,
                        layers=layers, tok_epochs=tok_epochs,
                        prior_epochs=prior_epochs, tok_lr=tok_lr,
                        prior_lr=prior_lr, commit_weight=commit_weight,
                        steps=steps, temperature=temperature)
        self.device = device
        self.tokenizer: Optional[FlatVQTokenizer] = None
        self.prior: Optional[MaskGITPrior] = None
        self.rate_coef: Optional[np.ndarray] = None
        self.fit_report: Dict[str, Any] = {}
        self._last_intensity: Optional[torch.Tensor] = None
        # -log(pos_weight); set in fit(), restored in load(). 0.0 means an
        # uncorrected legacy checkpoint. See _draw_logits for why this cannot
        # move a sampled output.
        self.logit_offset: float = 0.0

        self.meta = BaselineMeta(
            name="maskgit_flat",
            citation=("Chang, Zhang, Jiang, Liu & Freeman, MaskGIT, CVPR 2022; "
                      "Yu et al., MAGVIT, CVPR 2023"),
            family="voxel+token",
            conditioning="gct and lct as MaskGIT prefix conditioning tokens",
            notes=("Single flat codebook of 1024 on the same 8x8x16 grid and the "
                   "same (6,15,14) patch as the pipeline, so token budget and "
                   "alphabet size are matched. No ladder, no activity prior, no "
                   "soft field, no adaptation stage. Voxel readout is Bernoulli "
                   "sampling of the decoder probability with a scalar shift "
                   "fixing the expected count, not rank thresholding."),
            extra=self.cfg,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _vol(batch) -> torch.Tensor:
        x = batch["x"]
        return x if x.dim() == 5 else x.unsqueeze(1)

    def fit(self, clips: Iterator[Dict[str, Any]], *, device: str = "cuda") -> None:
        self.device = device
        c = self.cfg

        # Sparse cache: ~200 spikes in 1.29M voxels, so indices are three orders
        # of magnitude smaller than the dense volumes and exactly equivalent.
        cached, lct_rows, rate_rows = [], [], []
        shape = None
        for batch in clips:
            v = (self._vol(batch) > 0.5)
            shape = tuple(v.shape[2:])
            cached.append({
                "nz": v.nonzero().to(torch.int16),
                "B": int(v.shape[0]),
                "gct": batch["global_ctx"].clone(),
                "lct": batch["local_ctx"].clone(),
            })
            for i in range(v.shape[0]):
                lct_rows.append(batch["local_ctx"][i].cpu().numpy())
                rate_rows.append(max(float(v[i].float().mean()), 1e-8))
        if not cached:
            raise RuntimeError("MaskGITFlat.fit saw no clips")
        T, H, W = shape

        def dense(b):
            v = torch.zeros(b["B"], 1, T, H, W, device=device)
            nz = b["nz"].to(device).long()
            if nz.numel():
                v[nz[:, 0], nz[:, 1], nz[:, 2], nz[:, 3], nz[:, 4]] = 1.0
            return v

        # -- stage 1: tokenizer -----------------------------------------
        self.tokenizer = FlatVQTokenizer(c["n_codes"], c["dim"], c["width"]).to(device)
        opt = torch.optim.Adam(self.tokenizer.parameters(), lr=c["tok_lr"])
        # Bernoulli, not MSE: the volume is 99.98% zeros and a squared error is
        # minimised by predicting zero everywhere. pos_weight lifts the spike
        # class enough that the codebook has something to encode.
        pw = torch.tensor([1.0 / max(float(np.mean(rate_rows)), 1e-8)], device=device)
        self.logit_offset = -float(np.log(float(pw)))
        tok_hist = []
        for ep in range(c["tok_epochs"]):
            tot = n = 0.0
            for b in cached:
                x = dense(b)
                logits, idx, commit = self.tokenizer(x)
                rec = F.binary_cross_entropy_with_logits(logits, x, pos_weight=pw)
                loss = rec + c["commit_weight"] * commit
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.tokenizer.parameters(), 5.0)
                opt.step()
                tot += float(rec); n += 1
            tok_hist.append(tot / max(n, 1))
            used = int((self.tokenizer.vq.cluster_size > 1e-2).sum())
            print(f"  tokenizer epoch {ep+1}/{c['tok_epochs']}  bce={tok_hist[-1]:.5f}  "
                  f"codes_used={used}/{c['n_codes']}", flush=True)

        # -- stage 2: prior on frozen codes ------------------------------
        self.tokenizer.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            for b in cached:
                b["ids"] = self.tokenizer.tokens(dense(b)).cpu()

        n_tok = int(np.prod(GRID))
        self.prior = MaskGITPrior(
            c["n_codes"], n_tok, c["d_model"], c["layers"],
            gct_dim=cached[0]["gct"].shape[1], lct_dim=cached[0]["lct"].shape[1],
        ).to(device)
        popt = torch.optim.Adam(self.prior.parameters(), lr=c["prior_lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(popt, c["prior_epochs"])
        pri_hist = []
        for ep in range(c["prior_epochs"]):
            tot = n = 0.0
            for b in cached:
                loss, _ = self.prior.loss(
                    b["ids"].to(device), b["gct"].to(device), b["lct"].to(device))
                popt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.prior.parameters(), 5.0)
                popt.step()
                tot += float(loss); n += 1
            sched.step()
            pri_hist.append(tot / max(n, 1))
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  prior epoch {ep+1}/{c['prior_epochs']}  "
                      f"ce={pri_hist[-1]:.5f}", flush=True)

        # -- lct -> log rate, the same calibration every baseline uses ---
        A = np.concatenate([np.asarray(lct_rows, np.float64),
                            np.ones((len(lct_rows), 1))], axis=1)
        y = np.log(np.asarray(rate_rows, np.float64))
        self.rate_coef, *_ = np.linalg.lstsq(A, y, rcond=None)

        ids_all = torch.cat([b["ids"] for b in cached])
        self.fit_report = {
            "tokenizer_bce_per_epoch": tok_hist,
            "prior_ce_per_epoch": pri_hist,
            "n_clips": int(sum(b["B"] for b in cached)),
            "codes_used": int(len(torch.unique(ids_all))),
            "n_codes": c["n_codes"],
            "token_entropy_nats": float(
                -(lambda p: (p[p > 0] * p[p > 0].log()).sum())(
                    torch.bincount(ids_all.reshape(-1),
                                   minlength=c["n_codes"]).float()
                    / ids_all.numel())),
            "config": c,
        }

    # ------------------------------------------------------------------
    def _target_rate(self, cond) -> torch.Tensor:
        lct = cond.local_ctx.detach().cpu().numpy().astype(np.float64)
        A = np.concatenate([lct, np.ones((lct.shape[0], 1))], axis=1)
        return torch.tensor(np.clip(np.exp(A @ self.rate_coef), 1e-7, 1e-2),
                            dtype=torch.float32)

    def _decode_field(self, grid: torch.Tensor) -> torch.Tensor:
        """Decode token ids to a per-voxel logit field, in HONEST units.

        Every field this class hands out -- free generation, completion and the
        oracle -- comes from this one decoder head, so the calibration
        correction belongs here. It was first applied in `_draw_logits` alone,
        which was wrong in a way that produced a silent no-op: `task_eval._field`
        prefers `complete()` whenever a baseline implements it, and this one
        does, so the corrected path was never executed and the re-scored report
        came back byte-identical.

        Cost-sensitive correction (Elkan, IJCAI 2001). The head is trained with
        pos_weight=w, whose minimiser is

            sigma(z) = w*r / (w*r + 1 - r),

        so the raw logit overstates the log-odds by log w ~ 9.0 nats. Left
        uncorrected, sigma(z) is not a probability, and `own_hard` in
        task_eval.py sums exactly that to get the arm's own expected spike
        count -- reporting a units error as if it were a modelling failure.

        This CANNOT change a sampled output. `_bernoulli_at_rate` bisects for a
        per-clip scalar shift b that hits the target rate, and a constant added
        to every logit is absorbed exactly into b. AP, F1, top-k and every
        binarised metric are invariant by construction; only `own_*` may move.
        That invariance is the test -- and, as above, it is also why a no-op
        cannot be distinguished from a correct fix by the binarised columns.
        """
        return self.tokenizer.decode_ids(grid)[:, 0] + self.logit_offset

    @torch.no_grad()
    def _draw_logits(self, cond: ConditioningBatch, generator=None) -> torch.Tensor:
        ids = self.prior.generate(
            cond.global_ctx, cond.local_ctx,
            steps=self.cfg["steps"], temperature=self.cfg["temperature"],
            generator=generator, device=cond.global_ctx.device)
        grid = ids.reshape(ids.shape[0], *GRID)
        return self._decode_field(grid)

    @staticmethod
    def _bernoulli_at_rate(logits: torch.Tensor, target: torch.Tensor,
                           generator=None) -> torch.Tensor:
        """Sample Bernoulli(sigmoid(logits + b)) with b set so E[rate] = target.

        NOT top-k. The tokenizer's decoder is trained with binary cross-entropy,
        so its output is a per-voxel probability and the faithful readout is to
        draw from it. Rank-thresholding a smooth probability field instead
        selects whichever contiguous voxels sit inside the hottest patches, and
        since each token expands to a 6x15x14 block the result is blobs rather
        than isolated spike events -- measured at spatial_coact 11x the real
        value and persist1 6.9x, from a tokenizer that reconstructs at BCE
        0.071. That was an artefact of the readout, not of the architecture.

        The scalar shift b is the maximum-entropy way to hit a target count
        without distorting the relative ordering the model produced, and it is
        found per clip by bisection on a monotone function.
        """
        B = logits.shape[0]
        flat = logits.reshape(B, -1)
        lo = torch.full((B, 1), -20.0, device=flat.device)
        hi = torch.full((B, 1), 20.0, device=flat.device)
        tgt = target.to(flat.device).view(B, 1)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            rate = torch.sigmoid(flat + mid).mean(dim=1, keepdim=True)
            too_low = rate < tgt
            lo = torch.where(too_low, mid, lo)
            hi = torch.where(too_low, hi, mid)
        p = torch.sigmoid(flat + 0.5 * (lo + hi))
        u = torch.rand(p.shape, generator=generator).to(p.device)
        return (u < p).float().reshape(logits.shape)

    @torch.no_grad()
    def sample(self, cond: ConditioningBatch, *, generator=None) -> torch.Tensor:
        logits = self._draw_logits(cond, generator)
        self._last_intensity = logits
        return self._bernoulli_at_rate(
            logits, self._target_rate(cond), generator)

    @torch.no_grad()
    def sample_intensity(self, cond: ConditioningBatch, *, generator=None):
        if self._last_intensity is not None:
            out, self._last_intensity = self._last_intensity, None
            return out
        return self._draw_logits(cond, generator)

    @torch.no_grad()
    def complete(self, cond: ConditioningBatch, real: torch.Tensor,
                 roi: torch.Tensor, *, generator=None):
        """Token-level inpainting: pin the visible tokens, decode the ROI.

        This is MaskGIT's native strength and the reason it is the baseline
        that matters on the task axis -- unlike DG it can actually read the
        visible remainder.

        The hole is zeroed BEFORE tokenising. The patch embed is a
        PATCH-strided Conv3d and so is token-local, but the residual blocks
        that follow are 3x3x3 and mix neighbouring tokens, so a "visible"
        token's id can otherwise carry hidden voxels.
        """
        dev = self.device
        x = real.to(dev).float()
        if x.dim() == 4:
            x = x.unsqueeze(1)
        m = roi.to(dev).float()
        if m.dim() == 4:
            m = m.unsqueeze(1)

        ids = self.tokenizer.tokens(x * (1.0 - m))
        B = ids.shape[0]
        ids = ids.reshape(B, -1)

        roi_tok = F.max_pool3d(m[:, :1], kernel_size=PATCH, stride=PATCH)
        roi_tok = roi_tok.reshape(B, -1) > 0.5
        if roi_tok.shape[1] != ids.shape[1]:
            raise RuntimeError(
                f"token ROI {tuple(roi_tok.shape)} does not match ids "
                f"{tuple(ids.shape)}; ROI must be patch-aligned")

        out = self.prior.complete(
            ids, roi_tok, cond.global_ctx.to(dev), cond.local_ctx.to(dev),
            steps=self.cfg["steps"], temperature=self.cfg["temperature"],
            generator=generator)
        return self._decode_field(out.reshape(B, *GRID))

    @torch.no_grad()
    def oracle_field(self, real: torch.Tensor):
        """Decode the TRUE tokens of the full clip -- this tokenizer's ceiling."""
        dev = self.device
        x = real.to(dev).float()
        if x.dim() == 4:
            x = x.unsqueeze(1)
        ids = self.tokenizer.tokens(x)
        B = ids.shape[0]
        return self._decode_field(ids.reshape(B, *GRID))

    @torch.no_grad()
    def tokenize(self, vols: torch.Tensor) -> Optional[torch.Tensor]:
        x = vols if vols.dim() == 5 else vols.unsqueeze(1)
        return self.tokenizer.tokens(x.to(self.device))

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"tokenizer": self.tokenizer.state_dict(),
                    "prior": self.prior.state_dict(),
                    "rate_coef": self.rate_coef, "cfg": self.cfg,
                    "logit_offset": self.logit_offset,
                    "fit_report": self.fit_report}, path)
        path.with_suffix(".fit.json").write_text(
            json.dumps(self.fit_report, indent=2, default=float))

    def load(self, path: Path, *, device: str = "cuda") -> None:
        d = torch.load(Path(path), map_location="cpu")
        c = self.cfg = d["cfg"]
        self.tokenizer = FlatVQTokenizer(c["n_codes"], c["dim"], c["width"])
        self.tokenizer.load_state_dict(d["tokenizer"])
        self.prior = MaskGITPrior(c["n_codes"], int(np.prod(GRID)),
                                  c["d_model"], c["layers"])
        self.prior.load_state_dict(d["prior"])
        self.tokenizer = self.tokenizer.to(device).eval()
        self.prior = self.prior.to(device).eval()
        self.rate_coef = d["rate_coef"]
        self.fit_report = d.get("fit_report", {})
        # Legacy checkpoints predate the correction and store no offset. They
        # load with 0.0 rather than a guessed value, so an uncorrected file
        # keeps its published (wrong-units) behaviour instead of silently
        # changing meaning; the corrected checkpoint carries the number.
        self.logit_offset = float(d.get("logit_offset", 0.0))
        self.device = device


@register("maskgit_flat")
def _build(**kw) -> MaskGITFlat:
    return MaskGITFlat(**kw)
