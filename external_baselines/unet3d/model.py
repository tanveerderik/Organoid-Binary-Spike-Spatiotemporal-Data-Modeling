"""UNet3D-Inpaint: supervised conditional video completion, no latent code.

The question this baseline answers is the one a reviewer asks first, and it is
not about hierarchy or about alphabets. It is: *the four tasks are masked video
inpainting -- what happens if you simply train a 3-D convolutional net to fill
the hole?* Every other arm in the comparison answers a harder question and is
then scored on this one. This arm answers exactly the question being scored,
with the loss pointed straight at the metric, so it is the strongest challenger
the table can contain and the honest reference for what the generative
machinery has to buy its way past.

What it gets, deliberately:

  * the same gct/lct conditioning, through FiLM;
  * the same four mask families in the same 25/25/25/50 proportions, redrawn
    every epoch from `dataset.py`'s own distribution;
  * holes snapped to the same (6,15,14) patch lattice it is evaluated on;
  * full-resolution output, so unlike every token-based arm it can localise a
    spike to a single electrode;
  * val-split early stopping, the same selection the shipped checkpoints get.

What it does not have, and what the table must therefore say:

  * **no latent variable.** One context gives one field, forever. It cannot be
    sampled from, so "free generation" for this arm is its conditional mean
    thresholded -- which is optimal for a ranking metric like AP and useless
    for a distributional one. Expect it to do well on AP/F1 and badly on the
    avalanche and ISI distances, and do not read the first as a win over
    generative models without reading the second.
  * **no reusable representation.** There are no motifs to inspect, no token
    sequence to condition anything else on, and nothing transfers to a task the
    net was not trained on.

Storage is amortised exactly like ours and MaskGIT-flat's: nothing is kept per
assay, so it belongs in the learned class of the scalability table rather than
with the lookup arms.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.protocol import BaselineMeta, ConditioningBatch, SpikeVolumeBaseline
from ..registry import register
from .masks import PATCH, sample_spec, spec_to_roi
from .net import CondUNet3D


def _pos_hw(sd, key: str = "pos"):
    """(H, W) of the saved spatial embedding, or None if the checkpoint predates
    it. Read from the weights rather than from the config so an older file loads
    as the network it actually is, instead of raising on a shape mismatch."""
    t = sd.get(key)
    return None if t is None else (int(t.shape[-2]), int(t.shape[-1]))


class UNet3DInpainter(SpikeVolumeBaseline):

    def __init__(
        self,
        *,
        widths: tuple = (32, 64, 128, 256),
        ctx_dim: int = 128,
        epochs: int = 40,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        micro_batch: int = 2,
        val_batches: int = 24,
        patience: int = 8,
        seed: int = 20260822,
        amp: bool = True,
        device: str = "cuda",
    ):
        self.cfg = dict(widths=tuple(widths), ctx_dim=ctx_dim, epochs=epochs,
                        lr=lr, weight_decay=weight_decay,
                        micro_batch=micro_batch, val_batches=val_batches,
                        patience=patience, seed=seed, amp=amp)
        self.device = device
        self.net: Optional[CondUNet3D] = None
        self.rate_coef: Optional[np.ndarray] = None
        self.logit_offset: float = 0.0
        self.fit_report: Dict[str, Any] = {}
        self._last_logits: Optional[torch.Tensor] = None

        self.meta = BaselineMeta(
            name="unet3d",
            citation=("Cicek et al., 3D U-Net, MICCAI 2016; Perez et al., FiLM, "
                      "AAAI 2018; cf. Pathak et al., Context Encoders, CVPR 2016"),
            family="voxel",
            conditioning="gct and lct embedded jointly, FiLM on every block",
            notes=("Direct supervision of the completion objective: BCE inside "
                   "the ROI only, on the same four mask families and the same "
                   "patch-aligned holes used at evaluation. Deterministic -- one "
                   "context gives one field -- so it has no sampling "
                   "distribution and no reusable latent. Full-resolution "
                   "output, so unlike the tokenized arms it is not limited to "
                   "the 210-electrode patch. Stores nothing per assay."),
            extra=self.cfg,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _vol(batch) -> torch.Tensor:
        x = batch["x"]
        return x if x.dim() == 5 else x.unsqueeze(1)

    @staticmethod
    def _cache(clips, device) -> tuple[list, tuple]:
        """Sparse-cache the clips. ~200 spikes in 1.29M voxels, so the nonzero
        indices are three orders of magnitude smaller than the dense volume and
        carry exactly the same information."""
        cached, shape = [], None
        for batch in clips:
            v = (UNet3DInpainter._vol(batch) > 0.5)
            shape = tuple(int(s) for s in v.shape[2:])
            cached.append({"nz": v.nonzero().to(torch.int16),
                           "B": int(v.shape[0]),
                           "gct": batch["global_ctx"].float().clone(),
                           "lct": batch["local_ctx"].float().clone()})
        if not cached:
            raise RuntimeError("UNet3DInpainter.fit saw no clips")
        return cached, shape

    @staticmethod
    def _dense(b, shape, device) -> torch.Tensor:
        T, H, W = shape
        v = torch.zeros(b["B"], 1, T, H, W, device=device)
        nz = b["nz"].to(device).long()
        if nz.numel():
            v[nz[:, 0], nz[:, 1], nz[:, 2], nz[:, 3], nz[:, 4]] = 1.0
        return v

    # ------------------------------------------------------------------
    def _roi_bce(self, logits, x, roi, pw) -> torch.Tensor:
        """BCE restricted to the hole, averaged over ROI voxels.

        Averaging over the ROI rather than over the volume is what makes the
        four tasks commensurable: their ROIs span 33% to 100% of the volume, so
        a volume-mean loss would weight `recon` three times as heavily as
        `spatial` for no reason other than hole size.
        """
        per = F.binary_cross_entropy_with_logits(
            logits, x, pos_weight=pw, reduction="none")
        return (per * roi).sum() / roi.sum().clamp(min=1.0)

    def _epoch_specs(self, cached, shape, rng) -> List[List[Dict[str, Any]]]:
        T, H, W = shape
        return [[sample_spec(rng, T, H, W) for _ in range(b["B"])] for b in cached]

    @torch.no_grad()
    def _val_loss(self, cached, shape, specs, pw) -> float:
        self.net.eval()
        tot = n = 0.0
        for b, sp in zip(cached, specs):
            x = self._dense(b, shape, self.device)
            roi = spec_to_roi(sp, shape, self.device)
            with torch.cuda.amp.autocast(enabled=self.cfg["amp"]):
                lg = self.net(x * (1.0 - roi), roi,
                              b["gct"].to(self.device), b["lct"].to(self.device))
                loss = self._roi_bce(lg.float(), x, roi, pw)
            tot += float(loss); n += 1
        self.net.train()
        return tot / max(n, 1)

    def fit(self, clips: Iterator[Dict[str, Any]], *, device: str = "cuda") -> None:
        self.device = device
        c = self.cfg
        train, shape = self._cache(clips, device)

        # Validation clips for early stopping. Pulled here rather than passed in
        # because `fit` is handed TRAIN only by contract -- and selection on val
        # is exactly what the shipped checkpoints get, so taking it is parity
        # rather than an advantage. Never test.
        from ..common import data as bdata
        val, vshape = self._cache(
            bdata.iter_raw("val", batches=c["val_batches"]), device)
        if vshape != shape:
            raise RuntimeError(f"val volume {vshape} != train volume {shape}")

        rates = []
        for b in train:
            x = self._dense(b, shape, device)
            rates += [max(float(x[i].mean()), 1e-8) for i in range(b["B"])]
        base = float(np.mean(rates))
        pw = torch.tensor([1.0 / base], device=device)
        self.logit_offset = -float(np.log(float(pw)))

        # Weight init as well as mask draws. Without this the network
        # initialisation came from the ambient global RNG, so a seed
        # sweep would vary two things at once and could not attribute
        # the spread. Same trap as [train_map_null_rng_contamination].
        torch.manual_seed(c["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(c["seed"])
        self.net = CondUNet3D(
            ctx_in=train[0]["gct"].shape[1] + train[0]["lct"].shape[1],
            ctx_dim=c["ctx_dim"], widths=c["widths"],
            spatial_size=shape[1:]).to(device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=c["lr"],
                                weight_decay=c["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c["epochs"])
        scaler = torch.cuda.amp.GradScaler(enabled=c["amp"])

        # Frozen val holes: resampling them every epoch would make the selection
        # curve noisy for the same reason the Stage-4 val curve was noisy.
        vrng = np.random.default_rng(c["seed"] + 991)
        vspecs = self._epoch_specs(val, shape, vrng)

        rng = np.random.default_rng(c["seed"])
        mb = max(1, int(c["micro_batch"]))
        hist, vhist = [], []
        best, best_ep, best_state = float("inf"), -1, None
        t0 = time.time()
        for ep in range(c["epochs"]):
            specs = self._epoch_specs(train, shape, rng)
            tot = n = 0.0
            for b, sp in zip(train, specs):
                x = self._dense(b, shape, device)
                roi = spec_to_roi(sp, shape, device)
                g, l = b["gct"].to(device), b["lct"].to(device)
                opt.zero_grad(set_to_none=True)
                # Split the collated batch into micro-batches: one width-32
                # activation over 1.29M voxels is 165 MB at batch 4, and the
                # gradient is identical because the loss is a mean over ROI
                # voxels reweighted by each chunk's share of them.
                w = roi.sum().clamp(min=1.0)
                for s in range(0, b["B"], mb):
                    e = min(s + mb, b["B"])
                    with torch.cuda.amp.autocast(enabled=c["amp"]):
                        lg = self.net(x[s:e] * (1.0 - roi[s:e]), roi[s:e],
                                      g[s:e], l[s:e])
                        per = F.binary_cross_entropy_with_logits(
                            lg.float(), x[s:e], pos_weight=pw, reduction="none")
                        loss = (per * roi[s:e]).sum() / w
                    scaler.scale(loss).backward()
                    tot += float(loss) * (e - s) / b["B"]
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                scaler.step(opt); scaler.update()
                n += 1
            sched.step()
            hist.append(tot / max(n, 1))
            vhist.append(self._val_loss(val, shape, vspecs, pw))
            if vhist[-1] < best - 1e-6:
                best, best_ep = vhist[-1], ep
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.net.state_dict().items()}
            print(f"  epoch {ep+1}/{c['epochs']}  roi_bce={hist[-1]:.5f}  "
                  f"val={vhist[-1]:.5f}  best@{best_ep+1}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            if ep - best_ep >= c["patience"]:
                print(f"  early stop: no val improvement in {c['patience']} epochs",
                      flush=True)
                break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()

        # lct -> log rate, the same calibration every other baseline uses, so
        # the free-generation readout is not what separates the arms.
        lct_rows = np.concatenate([b["lct"].cpu().numpy() for b in train])
        A = np.concatenate([lct_rows.astype(np.float64),
                            np.ones((lct_rows.shape[0], 1))], axis=1)
        self.rate_coef, *_ = np.linalg.lstsq(
            A, np.log(np.asarray(rates, np.float64)), rcond=None)

        self.fit_report = {
            "roi_bce_per_epoch": hist,
            "val_roi_bce_per_epoch": vhist,
            "best_epoch": best_ep + 1,
            "best_val_roi_bce": best,
            "epochs_run": len(hist),
            "n_clips": int(sum(b["B"] for b in train)),
            "n_val_clips": int(sum(b["B"] for b in val)),
            "train_voxel_rate": base,
            "pos_weight": float(pw),
            "logit_offset": self.logit_offset,
            "n_params": int(sum(p.numel() for p in self.net.parameters())),
            "patch_alignment": list(PATCH),
            "fit_seconds": time.time() - t0,
            "config": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in c.items()},
        }

    # ------------------------------------------------------------------
    def _target_rate(self, cond) -> torch.Tensor:
        lct = cond.local_ctx.detach().cpu().numpy().astype(np.float64)
        A = np.concatenate([lct, np.ones((lct.shape[0], 1))], axis=1)
        return torch.tensor(np.clip(np.exp(A @ self.rate_coef), 1e-7, 1e-2),
                            dtype=torch.float32)

    @staticmethod
    def _bernoulli_at_rate(logits, target, generator=None) -> torch.Tensor:
        """Bernoulli(sigmoid(logits + b)) with the scalar b set so E[rate] hits
        `target`. Character for character the readout MaskGIT-flat uses, and for
        the same reason -- the decoder is trained with BCE, so its output is a
        probability and the faithful readout is to draw from it. Sharing the
        readout is what makes the two voxel arms comparable."""
        B = logits.shape[0]
        flat = logits.reshape(B, -1)
        lo = torch.full((B, 1), -20.0, device=flat.device)
        hi = torch.full((B, 1), 20.0, device=flat.device)
        tgt = target.to(flat.device).view(B, 1)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            too_low = torch.sigmoid(flat + mid).mean(dim=1, keepdim=True) < tgt
            lo = torch.where(too_low, mid, lo)
            hi = torch.where(too_low, hi, mid)
        p = torch.sigmoid(flat + 0.5 * (lo + hi))
        u = torch.rand(p.shape, generator=generator).to(p.device)
        return (u < p).float().reshape(logits.shape)

    # -- calibration ----------------------------------------------------
    # Training minimises BCE at `pos_weight = 1/rate`, and the minimiser of that
    # loss is NOT the posterior. For a voxel that fires with probability r,
    #
    #     dL/dz = 0  =>  sigma(z*) = w.r / (w.r + 1 - r)  =>  z* = log w + logit(r)
    #
    # so the network's raw output overstates the probability by exactly log w --
    # here log(7153) = 8.9 nats, which puts a base-rate voxel at sigma ~ 0.5
    # instead of 1.4e-4. Subtracting it recovers the unweighted posterior
    # (Elkan, "The Foundations of Cost-Sensitive Learning", IJCAI 2001).
    #
    # This is a units correction, not a tuned knob: there is nothing to choose,
    # and it is rank-preserving, so AP, F1 and any top-k readout are bitwise
    # unchanged. What it fixes is the one readout that reads the field as a
    # probability rather than as a score -- the own-count column, which sums
    # sigma over the ROI. Without it that column reports a model predicting
    # ~50% of all voxels as spikes.
    @torch.no_grad()
    def _logits(self, x_vis, roi, cond) -> torch.Tensor:
        dev = self.device
        with torch.cuda.amp.autocast(enabled=self.cfg["amp"]):
            lg = self.net(x_vis.to(dev), roi.to(dev),
                          cond.global_ctx.to(dev), cond.local_ctx.to(dev))
        return lg.float()[:, 0] + self.logit_offset

    @torch.no_grad()
    def sample(self, cond: ConditioningBatch, *, generator=None) -> torch.Tensor:
        lg = self._free_logits(cond)
        self._last_logits = lg
        return self._bernoulli_at_rate(lg, self._target_rate(cond), generator)

    @torch.no_grad()
    def _free_logits(self, cond: ConditioningBatch) -> torch.Tensor:
        """Free generation is the `recon` task: everything masked, nothing
        visible. Identical by construction to `complete` with an all-ones ROI,
        which is what task 0 of the battery already is."""
        B = cond.batch_size
        T, H, W = cond.shape
        z = torch.zeros(B, 1, T, H, W, device=self.device)
        return self._logits(z, torch.ones_like(z), cond)

    @torch.no_grad()
    def sample_intensity(self, cond: ConditioningBatch, *, generator=None):
        if self._last_logits is not None:
            out, self._last_logits = self._last_logits, None
            return out
        return self._free_logits(cond)

    @torch.no_grad()
    def complete(self, cond: ConditioningBatch, real: torch.Tensor,
                 roi: torch.Tensor, *, generator=None):
        """Per-voxel logits given everything outside `roi`.

        The hole is blanked before the volume ever reaches the network, so the
        held-out content is not merely unused, it is absent -- which is what
        `tests/test_task_completion.py` checks by re-running this with the hole
        filled with noise and demanding a bitwise-identical result.
        """
        dev = self.device
        x = real.to(dev).float()
        if x.dim() == 4:
            x = x.unsqueeze(1)
        m = roi.to(dev).float()
        if m.dim() == 4:
            m = m.unsqueeze(1)
        m = m[:, :1]
        return self._logits(x[:, :1] * (1.0 - m), m, cond)

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"net": self.net.state_dict(), "rate_coef": self.rate_coef,
                    "logit_offset": self.logit_offset,
                    "cfg": self.cfg, "fit_report": self.fit_report}, path)
        path.with_suffix(".fit.json").write_text(
            json.dumps(self.fit_report, indent=2, default=float))

    def load(self, path: Path, *, device: str = "cuda") -> None:
        d = torch.load(Path(path), map_location="cpu")
        c = self.cfg = d["cfg"]
        ctx_in = d["net"]["embed.0.weight"].shape[1]
        self.net = CondUNet3D(ctx_in=ctx_in, ctx_dim=c["ctx_dim"],
                              widths=tuple(c["widths"]),
                              spatial_size=_pos_hw(d["net"]))
        self.net.load_state_dict(d["net"])
        self.net = self.net.to(device).eval()
        self.rate_coef = d["rate_coef"]
        # Checkpoints written before the correction existed carry only the
        # pos_weight in their fit report; recover the offset from it rather than
        # silently defaulting to zero, which would be the miscalibrated field.
        self.logit_offset = float(
            d["logit_offset"] if "logit_offset" in d
            else -np.log(float(d["fit_report"]["pos_weight"])))
        self.fit_report = d.get("fit_report", {})
        self.device = device


@register("unet3d")
def _build(**kw) -> UNet3DInpainter:
    return UNet3DInpainter(**kw)
