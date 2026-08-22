"""The four conditioning regimes, applied to an external baseline.

Pooled `stat_error` cannot see conditioning -- a model that reproduces the
population marginals scores well on it whether or not it responds to the clip
it was asked about. The Dichotomized Gaussian demonstrates this emphatically:
it sits ~2sd off the real-vs-real ceiling on pooled statistics while getting
each individual clip's avalanche structure badly wrong.

So the comparison that carries the paper is the *ladder*: give every model the
same monotone sequence of context, and ask whose samples move toward the held-out
clip as context is added. A baseline that ignores context is flat across the
ladder by construction, and that is visible here rather than arguable.

    random                 gct and lct both drawn unconditionally from the TRAIN
                           context bank. Nothing about the held-out clip is used.
    global_only            gct pinned to the clip; lct retrieved conditioned on it.
    global_partial_local   gct pinned; log_mean_firing_density and
                           active_site_ratio pinned; the rest retrieved.
    global_full_local      gct and lct both pinned. Full context.

The retrieval logic is a faithful copy of `analysis/generate_regimes.py::
contexts_for`, including the detail that pinned features are overwritten with
their exact values after retrieval -- the bank returns a real neighbouring clip
whose pinned features are only *close*, and "pinned" has to mean pinned or the
ladder is not monotone in what it claims. If that function changes, this must
change with it; the regime definitions are asserted into every run_config.json
so a drift is detectable after the fact.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch

from MAGVIT_project.inference.sample_context import ContextBankSampler
from MAGVIT_project.utils.constants import ACTIVITY_CTX_NAMES

from .protocol import ConditioningBatch

REGIMES = ("random", "global_only", "global_partial_local", "global_full_local")

# The two an experimenter could actually state up front: how active the culture
# is, and on what fraction of sites. The variances, covariances and trend
# describe the spatiotemporal shape we want the model to invent.
PARTIAL_LOCAL_DEFAULT = ("log_mean_firing_density", "active_site_ratio")

CTX_FLAGS = {
    "random": {"global": False, "local": False},
    "global_only": {"global": True, "local": False},
    "global_partial_local": {"global": True, "local": "partial"},
    "global_full_local": {"global": True, "local": True},
}


class ContextLadder:

    def __init__(
        self,
        bank_path: str = "ckpts/context_prior.pkl",
        *,
        partial: Sequence[str] = PARTIAL_LOCAL_DEFAULT,
        seed: int = 20260821,
    ):
        self.bank = ContextBankSampler.from_file(bank_path, model=None, seed=seed)
        missing = [f for f in partial if f not in ACTIVITY_CTX_NAMES]
        if missing:
            raise ValueError(f"unknown lct features {missing}; "
                             f"available: {ACTIVITY_CTX_NAMES}")
        self.partial_idx = [ACTIVITY_CTX_NAMES.index(f) for f in partial]
        self.partial_names = tuple(partial)
        self.bank_path = bank_path

    def definitions(self) -> Dict[str, object]:
        return {
            "bank": self.bank_path,
            "bank_size": int(self.bank.N),
            "partial_local_pinned": list(self.partial_names),
            "regimes": {
                "random": "gct and lct both drawn unconditionally from the train bank",
                "global_only": "gct pinned; lct retrieved conditioned on gct",
                "global_partial_local":
                    f"gct pinned; {list(self.partial_names)} pinned exactly; rest retrieved",
                "global_full_local": "gct and lct both pinned to the held-out clip",
            },
        }

    def apply(self, cond: ConditioningBatch, regime: str) -> ConditioningBatch:
        if regime == "global_full_local":
            return cond
        if regime not in CTX_FLAGS:
            raise ValueError(f"unknown regime {regime!r}")

        g_np = cond.global_ctx.detach().cpu().numpy().astype(np.float32)
        l_np = cond.local_ctx.detach().cpu().numpy().astype(np.float32)
        g_out, l_out = np.empty_like(g_np), np.empty_like(l_np)

        for i in range(cond.batch_size):
            if regime == "random":
                s = self.bank.sample_unconditional(batch_size=1)
                g_out[i], l_out[i] = s["global_ctx"][0], s["local_ctx"][0]
            elif regime == "global_only":
                s = self.bank.sample_given_global(g_np[i], batch_size=1)
                g_out[i], l_out[i] = g_np[i], s["local_ctx"][0]
            else:  # global_partial_local
                pin = {int(j): float(l_np[i, j]) for j in self.partial_idx}
                s = self.bank.sample_given_partial_local(
                    global_ctx=g_np[i], partial_local=pin, batch_size=1)
                l_ret = np.asarray(s["local_ctx"][0], dtype=np.float32)
                for j in self.partial_idx:
                    l_ret[j] = l_np[i, j]
                g_out[i], l_out[i] = g_np[i], l_ret

        dev = cond.global_ctx.device
        return ConditioningBatch(
            global_ctx=torch.from_numpy(g_out).to(dev),
            local_ctx=torch.from_numpy(l_out).to(dev),
            assay_idx=cond.assay_idx,
            shape=cond.shape,
            assay_name=cond.assay_name,
        )
