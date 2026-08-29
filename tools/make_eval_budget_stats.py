#!/usr/bin/env python3
"""Measure what the pinned evaluation protocol actually covers.

A reader is entitled to ask what the scored clips are: how they were chosen,
whether the choice could have been made after seeing a result, and how much of
the corpus they touch. None of that should be typed by hand, and the answer is
not obvious from the code.

What this measures, and why the answer matters:

  * The test loader is `shuffle=False` over a `DeterministicSubset`
    (`dataset.py:1134`), so `--batches N` scores a fixed *prefix* of the test
    split in recording order. The harness `--seed` pins the model's sampling --
    Monte-Carlo draws, Gumbel noise, mask draws -- and not which clips are
    scored. That is a stronger guarantee than a seeded random sample, but it
    also means a small N does not sample the corpus: it truncates it. An
    earlier protocol ran 12 and 8 batches, which reached 6 and 4 of the 31
    recordings respectively.
  * Recording coverage is therefore read off `assay_idx` per budget rather
    than assumed from the split counts.

The batch counts are read from the reports themselves, so this cannot claim a
coverage the scored runs did not have.

    python tools/make_eval_budget_stats.py

Writes reports/eval_budget.json, rendered into macros by
tools/make_paper_tables.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from MAGVIT_project.external_baselines.common import data as bdata  # noqa: E402

OUT = ROOT / "reports" / "eval_budget.json"
EB = ROOT / "reports" / "external_baselines"

# Each budget names the report that defines it, so the batch count and seed come
# from the run rather than from this file. `mc` is stated where the harness has
# one; the diagnostics battery draws a single sample per clip per rung.
BUDGETS = {
    "task_axis": {
        "source": EB / "task_eval_pipeline.json",
        "used_by": "the four-setting completion battery",
    },
    "diagnostics": {
        "source": EB / "diagnose_pipeline.json",
        "used_by": "generation families, conditioning ladders and count "
                   "statistics",
    },
}


def _prefix(loader, n_batches: int | None) -> dict:
    assays, clips = [], 0
    for i, batch in enumerate(loader):
        if n_batches is not None and i >= n_batches:
            break
        assays.extend(int(a) for a in batch["assay_idx"].tolist())
        clips += int(batch["x"].shape[0])
    return {"clips": clips,
            "recordings": len(set(assays)),
            "assay_indices": sorted(set(assays))}


def main() -> int:
    test = bdata.loader_for("test")
    n_test_clips = len(test.dataset)

    cfgs = {}
    for name, spec in BUDGETS.items():
        rep = json.loads(Path(spec["source"]).read_text())
        cfgs[name] = {"batches": int(rep["batches"]),
                      "seed": rep.get("seed"),
                      "mc": rep.get("mc"),
                      "used_by": spec["used_by"],
                      "source": str(Path(spec["source"]).relative_to(ROOT))}

    # Two independent passes must agree, or "identical clips for every arm" is
    # not true. Checked rather than asserted in prose.
    widest = max(c["batches"] for c in cfgs.values())
    if _prefix(test, widest) != _prefix(test, widest):
        raise RuntimeError("the test loader is not order-stable across passes; "
                           "the pinned-protocol claim does not hold")

    # What the whole split contains, so "covers every recording" is measured
    # against the split rather than against the other budget. Two budgets that
    # agree with each other can still both be short.
    full = _prefix(test, None)

    out = {"test_clips_available": int(n_test_clips),
           "loader_shuffle": False,
           "split_total": full,
           "budgets": {}}
    for name, cfg in cfgs.items():
        out["budgets"][name] = {**cfg, **_prefix(test, cfg["batches"])}

    out["all_recordings_covered"] = all(
        b["recordings"] == full["recordings"] for b in out["budgets"].values())
    out["all_clips_covered"] = all(
        b["clips"] == full["clips"] for b in out["budgets"].values())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT}")
    print(f"  {'split':<12} {'':>3}         {full['clips']} clips "
          f"from {full['recordings']} recordings")
    for name, b in out["budgets"].items():
        print(f"  {name:<12} {b['batches']:>3} batches -> {b['clips']} clips "
              f"from {b['recordings']} recordings "
              f"(of {n_test_clips} test clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
