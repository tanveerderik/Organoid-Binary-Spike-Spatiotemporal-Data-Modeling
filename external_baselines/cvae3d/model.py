"""CVAE-3D: the same backbone as `unet3d`, plus a latent variable. Nothing else.

The pair is the point. `unet3d` is trained to predict the conditional mean and
therefore has exactly one output per context; this arm is identical in
architecture, conditioning, hole distribution, optimiser and readout, and
differs only by carrying z. So the difference between their columns is
attributable to stochasticity and to nothing else -- which is the cleanest
statement available here about what a generative model buys over a regressor on
this data, and it is a statement the table can make without us asserting it.

Expect the split to run the other way on the two metric families: the
deterministic arm should win the ranking metrics (a posterior mean is the
optimal ranking) and lose the distributional ones (a point estimate has no
spread). If that is what happens, neither arm alone is a fair peer and the
paper should quote both.

Also the continuous-latent answer to MaskGIT-flat's discrete one: same
conditioning, same data, no quantization.
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
from ..unet3d.masks import PATCH, sample_spec, spec_to_roi
from ..unet3d.model import UNet3DInpainter
from .net import CondVAE3D

# Latent grid is the backbone bottleneck: three stride-2 stages.
LATENT_STRIDE = 8


def _pos_hw(sd, key: str = "pos"):
    """(H, W) of the saved spatial embedding, or None if the checkpoint predates
    it. Read from the weights rather than from the config so an older file loads
    as the network it actually is, instead of raising on a shape mismatch."""
    t = sd.get(key)
    return None if t is None else (int(t.shape[-2]), int(t.shape[-1]))


class CVAE3D(SpikeVolumeBaseline):

    def __init__(
        self,
        *,
        widths: tuple = (32, 64, 128, 256),
        ctx_dim: int = 128,
        z_ch: int = 4,
        post_logvar_init: float = -4.0,
        beta: float = 0.05,
        free_bits: float = 0.05,
        kl_warmup_frac: float = 0.3,
        temperature: float = 1.0,
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
        self.cfg = dict(widths=tuple(widths), ctx_dim=ctx_dim, z_ch=z_ch,
                        post_logvar_init=post_logvar_init,
                        beta=beta, free_bits=free_bits,
                        kl_warmup_frac=kl_warmup_frac, temperature=temperature,
                        epochs=epochs, lr=lr, weight_decay=weight_decay,
                        micro_batch=micro_batch, val_batches=val_batches,
                        patience=patience, seed=seed, amp=amp)
        self.device = device
        self.net: Optional[CondVAE3D] = None
        self.rate_coef: Optional[np.ndarray] = None
        self.logit_offset: float = 0.0
        self.fit_report: Dict[str, Any] = {}
        self._last_logits: Optional[torch.Tensor] = None

        self.meta = BaselineMeta(
            name="cvae3d",
            citation=("Sohn, Lee & Yan, Conditional VAE, NIPS 2015; Kingma & "
                      "Welling, Auto-Encoding Variational Bayes, ICLR 2014"),
            family="voxel",
            conditioning=("gct and lct via FiLM, plus a conditional latent prior "
                          "p(z | visible, gct, lct) on a 6x15x28 grid"),
            notes=("Identical backbone, holes, optimiser and readout to unet3d; "
                   "the only difference is the latent variable, so the gap "
                   "between the two arms isolates stochasticity. Continuous "
                   "latent, no quantization -- the non-discrete counterpart to "
                   "MaskGIT-flat. Stores nothing per assay."),
            extra=self.cfg,
        )

    # Borrowed wholesale from the deterministic arm, by reference rather than by
    # copy. The clip cache, the densification, the lct -> rate calibration and
    # the Bernoulli-at-rate readout are the SAME CODE in both arms, so the pair
    # cannot drift apart in a way that would be mistaken for an effect of the
    # latent variable -- which is the only thing the comparison is about.
    _cache = staticmethod(UNet3DInpainter._cache)
    _dense = staticmethod(UNet3DInpainter._dense)
    _bernoulli_at_rate = staticmethod(UNet3DInpainter._bernoulli_at_rate)
    _target_rate = UNet3DInpainter._target_rate
    _epoch_specs = UNet3DInpainter._epoch_specs

    # ------------------------------------------------------------------
    @staticmethod
    def _kl_term(kl, roi, free_bits) -> torch.Tensor:
        """Mean KL per latent dimension, over the latent cells the hole touches.

        Restricted to the hole because that is all the latent has to describe:
        everything outside it reaches the decoder through the skip connections
        already, so charging the latent for it is a tax that buys nothing and
        pushes straight to collapse.

        Free bits (Kingma et al., 2016) are applied per channel, which is where
        collapse actually happens -- a channel either carries information or
        goes to the prior, and clamping the per-channel mean stops the second
        from being free.
        """
        w = F.avg_pool3d(roi, LATENT_STRIDE)                    # (B,1,t,h,w)
        d = (kl * w).sum(dim=(0, 2, 3, 4)) / w.sum().clamp(min=1e-6)
        return d.clamp(min=free_bits).sum()

    def _forward_loss(self, x, roi, g, l, pw, beta):
        lg, kl = self.net(x * (1.0 - roi), roi, g, l, x_full=x)
        per = F.binary_cross_entropy_with_logits(
            lg.float(), x, pos_weight=pw, reduction="none")
        rec = (per * roi).sum() / roi.sum().clamp(min=1.0)
        klt = self._kl_term(kl.float(), roi, self.cfg["free_bits"])
        return rec + beta * klt, rec, klt

    @torch.no_grad()
    def _val_loss(self, cached, shape, specs, pw) -> tuple:
        """Selection is on the RECONSTRUCTION term at beta's final value, drawn
        from the PRIOR -- i.e. on what generation will actually do, not on the
        ELBO the recognition network gets to cheat at."""
        self.net.eval()
        tot = kls = n = 0.0
        for b, sp in zip(cached, specs):
            x = self._dense(b, shape, self.device)
            roi = spec_to_roi(sp, shape, self.device)
            g, l = b["gct"].to(self.device), b["lct"].to(self.device)
            gen = torch.Generator().manual_seed(self.cfg["seed"])
            with torch.cuda.amp.autocast(enabled=self.cfg["amp"]):
                lg, _ = self.net(x * (1.0 - roi), roi, g, l, generator=gen)
                per = F.binary_cross_entropy_with_logits(
                    lg.float(), x, pos_weight=pw, reduction="none")
                rec = (per * roi).sum() / roi.sum().clamp(min=1.0)
                _, kl = self.net(x * (1.0 - roi), roi, g, l, x_full=x)
            tot += float(rec)
            kls += float(self._kl_term(kl.float(), roi, 0.0)) / self.cfg["z_ch"]
            n += 1
        self.net.train()
        return tot / max(n, 1), kls / max(n, 1)

    def fit(self, clips: Iterator[Dict[str, Any]], *, device: str = "cuda") -> None:
        self.device = device
        c = self.cfg
        train, shape = self._cache(clips, device)

        from ..common import data as bdata
        val, vshape = self._cache(
            bdata.iter_raw("val", batches=c["val_batches"]), device)
        if vshape != shape:
            raise RuntimeError(f"val volume {vshape} != train volume {shape}")
        if any(s % LATENT_STRIDE for s in shape):
            raise ValueError(f"volume {shape} is not a multiple of the latent "
                             f"stride {LATENT_STRIDE}")

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
        self.net = CondVAE3D(
            ctx_in=train[0]["gct"].shape[1] + train[0]["lct"].shape[1],
            ctx_dim=c["ctx_dim"], widths=c["widths"], z_ch=c["z_ch"],
            post_logvar_init=c["post_logvar_init"],
            spatial_size=shape[1:]).to(device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=c["lr"],
                                weight_decay=c["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c["epochs"])
        scaler = torch.cuda.amp.GradScaler(enabled=c["amp"])

        vrng = np.random.default_rng(c["seed"] + 991)
        vspecs = self._epoch_specs(val, shape, vrng)
        rng = np.random.default_rng(c["seed"])
        mb = max(1, int(c["micro_batch"]))
        warm = max(1, int(round(c["kl_warmup_frac"] * c["epochs"])))

        hist, vhist, klhist, vklhist = [], [], [], []
        best, best_ep, best_state = float("inf"), -1, None
        t0 = time.time()
        for ep in range(c["epochs"]):
            beta = c["beta"] * min(1.0, (ep + 1) / warm)
            specs = self._epoch_specs(train, shape, rng)
            tot = klt = n = 0.0
            for b, sp in zip(train, specs):
                x = self._dense(b, shape, device)
                roi = spec_to_roi(sp, shape, device)
                g, l = b["gct"].to(device), b["lct"].to(device)
                opt.zero_grad(set_to_none=True)
                for s in range(0, b["B"], mb):
                    e = min(s + mb, b["B"])
                    frac = (e - s) / b["B"]
                    with torch.cuda.amp.autocast(enabled=c["amp"]):
                        loss, rec, kk = self._forward_loss(
                            x[s:e], roi[s:e], g[s:e], l[s:e], pw, beta)
                    scaler.scale(loss * frac).backward()
                    tot += float(rec) * frac
                    klt += float(kk) * frac / c["z_ch"]
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                scaler.step(opt); scaler.update()
                n += 1
            sched.step()
            hist.append(tot / max(n, 1)); klhist.append(klt / max(n, 1))
            vr, vk = self._val_loss(val, shape, vspecs, pw)
            vhist.append(vr); vklhist.append(vk)
            if vr < best - 1e-6:
                best, best_ep = vr, ep
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.net.state_dict().items()}
            print(f"  epoch {ep+1}/{c['epochs']}  roi_bce={hist[-1]:.5f}  "
                  f"kl/dim={klhist[-1]:.4f}  beta={beta:.4f}  val={vr:.5f}  "
                  f"val_kl/dim={vk:.4f}  best@{best_ep+1}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            if ep - best_ep >= c["patience"]:
                print(f"  early stop: no val improvement in {c['patience']} epochs",
                      flush=True)
                break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()

        lct_rows = np.concatenate([b["lct"].cpu().numpy() for b in train])
        A = np.concatenate([lct_rows.astype(np.float64),
                            np.ones((lct_rows.shape[0], 1))], axis=1)
        self.rate_coef, *_ = np.linalg.lstsq(
            A, np.log(np.asarray(rates, np.float64)), rcond=None)

        self.fit_report = {
            "roi_bce_per_epoch": hist,
            "kl_per_dim_per_epoch": klhist,
            "val_roi_bce_per_epoch": vhist,
            "val_kl_per_dim_per_epoch": vklhist,
            "best_epoch": best_ep + 1,
            "best_val_roi_bce": best,
            # Read this before reading anything else: a KL that has gone to the
            # free-bits floor means the latent is unused and this arm has
            # silently become `unet3d`, in which case its columns say nothing
            # about stochasticity.
            "final_val_kl_per_dim": vklhist[-1] if vklhist else None,
            "free_bits": c["free_bits"],
            "latent_collapsed": bool(vklhist and vklhist[-1] <= 1.05 * c["free_bits"]),
            "epochs_run": len(hist),
            "n_clips": int(sum(b["B"] for b in train)),
            "n_val_clips": int(sum(b["B"] for b in val)),
            "train_voxel_rate": base,
            "pos_weight": float(pw),
            "logit_offset": self.logit_offset,
            "n_params": int(sum(p.numel() for p in self.net.parameters())),
            "patch_alignment": list(PATCH),
            "latent_grid": [s // LATENT_STRIDE for s in shape],
            "fit_seconds": time.time() - t0,
            "config": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in c.items()},
        }

    # ------------------------------------------------------------------

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
    def _logits(self, x_vis, roi, cond, generator=None) -> torch.Tensor:
        dev = self.device
        with torch.cuda.amp.autocast(enabled=self.cfg["amp"]):
            lg, _ = self.net(x_vis.to(dev), roi.to(dev),
                             cond.global_ctx.to(dev), cond.local_ctx.to(dev),
                             generator=generator,
                             temperature=self.cfg["temperature"])
        return lg.float()[:, 0] + self.logit_offset

    @torch.no_grad()
    def _free_logits(self, cond, generator=None) -> torch.Tensor:
        B = cond.batch_size
        T, H, W = cond.shape
        z = torch.zeros(B, 1, T, H, W, device=self.device)
        return self._logits(z, torch.ones_like(z), cond, generator)

    @torch.no_grad()
    def sample(self, cond: ConditioningBatch, *, generator=None) -> torch.Tensor:
        lg = self._free_logits(cond, generator)
        self._last_logits = lg
        return self._bernoulli_at_rate(lg, self._target_rate(cond), generator)

    @torch.no_grad()
    def sample_intensity(self, cond: ConditioningBatch, *, generator=None):
        if self._last_logits is not None:
            out, self._last_logits = self._last_logits, None
            return out
        return self._free_logits(cond, generator)

    @torch.no_grad()
    def complete(self, cond: ConditioningBatch, real: torch.Tensor,
                 roi: torch.Tensor, *, generator=None):
        """One draw of z from the conditional prior, decoded.

        A draw, not a mean: the harness already averages `--mc` independent
        calls for the ranking metrics and keeps single draws for the
        distributional ones, so returning the mean here would quietly delete
        the only thing this arm has that `unet3d` does not.

        The hole is blanked before the network sees it, so nothing inside the
        ROI can reach either the prior or the generator -- the recognition
        network, which does see the whole clip, is not on this path at all.
        """
        dev = self.device
        x = real.to(dev).float()
        if x.dim() == 4:
            x = x.unsqueeze(1)
        m = roi.to(dev).float()
        if m.dim() == 4:
            m = m.unsqueeze(1)
        m = m[:, :1]
        return self._logits(x[:, :1] * (1.0 - m), m, cond, generator)

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
        ctx_in = d["net"]["unet.embed.0.weight"].shape[1]
        self.net = CondVAE3D(ctx_in=ctx_in, ctx_dim=c["ctx_dim"],
                             widths=tuple(c["widths"]), z_ch=c["z_ch"],
                             post_logvar_init=c.get("post_logvar_init", 0.0),
                             spatial_size=_pos_hw(d["net"], "unet.pos"))
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


@register("cvae3d")
def _build(**kw) -> CVAE3D:
    return CVAE3D(**kw)
