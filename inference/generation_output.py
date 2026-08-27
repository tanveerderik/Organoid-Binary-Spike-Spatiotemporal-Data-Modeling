"""Structured output for Stage 4 generation runs.

The previous layout was a flat ``videos/`` directory plus a single JSON of
local-context comparisons. That makes the two things you actually want to do
awkward: compare the same assay across context regimes, and ask which physical
property a given sample got wrong.

Layout written here::

    <root>/
      manifest.json              one row per sample, every regime, flat -- load this
      summary.json               per-regime aggregates and real-data comparison
      real_reference.json        statistics of the real clips, computed once
      <regime>/
        regime_summary.json      aggregates for this regime alone
        videos/  <assay>__s000.mp4
        stats/   <assay>__s000.json

Regime directories are named for the conditioning actually supplied, not for the
mask spec, because "what was the model told" is the axis these runs are compared
along. Files are prefixed with the assay so a directory listing groups by assay
without opening anything.

Every sample carries the full MEA battery (rate, temporal persistence by lag,
spatial co-activation, avalanche sizes, ISI, bursts) plus, where a ground-truth
clip exists, its per-sample deviation from that clip's own statistics. Aggregate
comparison against real data uses relative error so properties on different
scales contribute comparably -- avalanche_mean is ~200 and rate is ~0.08, so a
raw sum would be the avalanche term alone.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import json

import numpy as np
import torch

from .metrics_generative import mea_statistics, compare_statistics, ks_distance
from .decode import save_xgen_video


def _clean(d: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the raw sample arrays the battery carries for KS tests."""
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in d.items() if not k.startswith("_")}


class GenerationWriter:
    """Accumulates samples across regimes and writes the tree on close()."""

    def __init__(self, root: str | Path, fps: int = 30):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.rows: List[Dict[str, Any]] = []
        self.per_regime: Dict[str, List[torch.Tensor]] = {}
        self.real_vols: List[torch.Tensor] = []
        self._counter: Dict[str, int] = {}

    # ------------------------------------------------------------------
    def add_real(self, vols: torch.Tensor) -> None:
        """Real clips, used once to build the reference statistics."""
        self.real_vols.append(vols.detach().cpu())

    def add_batch(
        self,
        regime: str,
        vols: torch.Tensor,                      # (B,T,H,W) binary
        *,
        assay_names: Sequence[str],
        task_ids: Sequence[int],
        context: Dict[str, bool],                # {"global": True, "local": False, ...}
        truth_vols: Optional[torch.Tensor] = None,
        extra: Optional[List[Dict[str, Any]]] = None,
        save_video: bool = True,
    ) -> None:
        rdir = self.root / regime
        (rdir / "videos").mkdir(parents=True, exist_ok=True)
        (rdir / "stats").mkdir(parents=True, exist_ok=True)
        v = vols.detach().cpu()
        self.per_regime.setdefault(regime, []).append(v)

        for i in range(v.shape[0]):
            n = self._counter.get(regime, 0)
            self._counter[regime] = n + 1
            assay = str(assay_names[i]) if i < len(assay_names) else "unknown"
            stem = f"{assay}__s{n:04d}"

            stats = mea_statistics(v[i : i + 1])
            row: Dict[str, Any] = {
                "regime": regime,
                "sample": n,
                "assay": assay,
                "task_id": int(task_ids[i]) if i < len(task_ids) else -1,
                "context_global": bool(context.get("global", False)),
                "context_local": bool(context.get("local", False)),
                "context_visible_tokens": bool(context.get("visible", False)),
                "stats": _clean(stats),
            }

            # Per-sample deviation from this clip's own ground truth, where one
            # exists. Meaningful for inpainting; for free generation there is no
            # per-sample target and this is deliberately absent rather than zero.
            if truth_vols is not None:
                t = truth_vols.detach().cpu()[i : i + 1]
                row["vs_truth"] = _clean(compare_statistics(mea_statistics(t), stats))

            if extra and i < len(extra):
                row.update(extra[i])

            if save_video:
                vp = rdir / "videos" / f"{stem}.mp4"
                # save_xgen_video indexes [0] off a leading channel axis, so it
                # wants (1,T,H,W); the batch here is (B,T,H,W).
                save_xgen_video(v[i : i + 1], vp, fps=self.fps)
                row["video_path"] = str(vp.relative_to(self.root))

            sp = rdir / "stats" / f"{stem}.json"
            sp.write_text(json.dumps(row, indent=2, default=float))
            row["stats_path"] = str(sp.relative_to(self.root))
            self.rows.append(row)

    # ------------------------------------------------------------------
    def close(self) -> Dict[str, Any]:
        real_stats = None
        if self.real_vols:
            real_stats = mea_statistics(torch.cat(self.real_vols))
            (self.root / "real_reference.json").write_text(
                json.dumps(_clean(real_stats), indent=2, default=float)
            )

        summary: Dict[str, Any] = {}
        for regime, chunks in self.per_regime.items():
            allv = torch.cat(chunks)
            gs = mea_statistics(allv)
            entry: Dict[str, Any] = {
                "n_samples": int(allv.shape[0]),
                "stats": _clean(gs),
            }
            if real_stats is not None:
                entry["vs_real"] = _clean(compare_statistics(real_stats, gs))
            per_assay: Dict[str, int] = {}
            for r in self.rows:
                if r["regime"] == regime:
                    per_assay[r["assay"]] = per_assay.get(r["assay"], 0) + 1
            entry["samples_per_assay"] = per_assay
            summary[regime] = entry
            (self.root / regime / "regime_summary.json").write_text(
                json.dumps(entry, indent=2, default=float)
            )

        (self.root / "manifest.json").write_text(
            json.dumps(self.rows, indent=2, default=float)
        )
        (self.root / "summary.json").write_text(
            json.dumps(summary, indent=2, default=float)
        )
        return summary


def summary_table(summary: Dict[str, Any]) -> str:
    """Compact text table over regimes, for logs and quick eyeballing."""
    keys = ["rate", "persist1", "spatial_coact", "avalanche_mean", "isi_mean", "burst_rate"]
    head = f"{'regime':22s} {'n':>5s} " + " ".join(f"{k[:12]:>12s}" for k in keys)
    lines = [head, "-" * len(head)]
    for regime, e in summary.items():
        s = e["stats"]
        lines.append(f"{regime:22s} {e['n_samples']:5d} " +
                     " ".join(f"{s.get(k, float('nan')):12.4f}" for k in keys))
    if any("vs_real" in e for e in summary.values()):
        lines += ["", f"{'regime':22s} {'statErr':>10s} {'KSaval':>8s} {'KSisi':>8s}", "-" * 52]
        for regime, e in summary.items():
            v = e.get("vs_real", {})
            lines.append(f"{regime:22s} {v.get('stat_error', float('nan')):10.4f} "
                         f"{v.get('ks_avalanche', float('nan')):8.3f} "
                         f"{v.get('ks_isi', float('nan')):8.3f}")
    return "\n".join(lines)
