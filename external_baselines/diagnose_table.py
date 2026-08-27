#!/usr/bin/env python3
"""Merge the per-model diagnose_*.json into one side-by-side report.

`compare_table.py` answers "which model scores better on the pooled generation
statistics". That question turned out to be nearly unanswerable -- the pooled
statistics cannot see conditioning at all, so a model that ignores its context
and emits the dataset average scores well on them.

This answers the two questions that ARE answerable, and keeps them apart
because no model wins both:

    CONDITIONAL   given this much context, does the sample match THIS clip?
    MARGINAL      does the sample look like real data at all?

Four families, no composite. Each is one number with a stated null, and the
conditional ones are per-clip so they admit a paired test.

    python external_baselines/diagnose_table.py > reports/external_baselines/diagnostics.md
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

DIR = Path("reports/external_baselines")
MODELS = [("pipeline", "Ours (4C+soft)"), ("maskgit_flat", "MaskGIT-flat"),
          ("unet3d", "3D U-Net (det.)\u2020"), ("cvae3d", "3D CVAE"),
          ("dg", "Dich. Gaussian"), ("glm", "Coupled GLM")]
RUNGS = [("random", "random"), ("local_only", "LOCAL only"),
         ("global_only", "GLOBAL only"), ("global_partial_local", "glob+partial"),
         ("global_full_local", "glob+full")]
# `recon` is task_id=0, whose spec yields an ALL-TRUE ROI -- nothing is
# visible, so it is not a completion task at all: it is free generation, and
# measured bitwise identical to the free-generation path. Labelled as such so
# it is never read as a peer of the three real completion tasks.
TASKS = [("recon", "recon (0) = free gen"), ("causal", "causal (1)"),
         ("noncausal", "noncausal (2)"), ("spatial", "spatial (3)")]
# The paper's peer methods are LEARNED GENERATIVE models. DG and the coupled
# GLM store a per-assay site map (26,880 floats each, growing linearly with the
# number of preparations) against 26 and 5,194 fitted values respectively --
# they are memorisation ceilings, not competitors, and rendering them as peers
# put "GLM" in bold as the winner of several columns. Bolding now competes over
# the learned arms only; the lookups render as `_ref_` rows in the same visual
# class as `_ceiling_` and `_null_`.
MODEL_CLASS = {"pipeline": "learned", "maskgit_flat": "learned",
               "unet3d": "learned", "cvae3d": "learned",
               "dg": "lookup", "glm": "lookup"}

# Not every learned arm is a generative model, and the distinction decides how
# a column may be read. `unet3d` is trained to predict the conditional MEAN of
# the hole, so it has exactly one output per context: optimal for a ranking
# metric like AP, and empty on any metric that asks about a distribution. It is
# the direct-supervision reference -- the strongest thing that can be pointed at
# the completion objective -- not a peer generator. `cvae3d` is the same
# backbone WITH a latent, so the pair isolates stochasticity.
DETERMINISTIC = {"unet3d"}

# A dagger rather than an asterisk: model labels appear inside table cells that
# already carry `**bold**`, and an odd number of stray asterisks on a line is
# exactly the input that makes a Markdown renderer start emphasis in the middle
# of a number. The dagger has no Markdown meaning at all.
UNET_MARK = "\u2020"

MARK_NOTE = (
    UNET_MARK + " **3D U-Net (det.) is not a representation learner, and its "
    "reconstruction numbers are not comparable to a tokenizer's.** It is a "
    "U-Net: `forward` concatenates the full-resolution stem output back in on "
    "the way up, so an uncompressed path runs from input to output and the "
    "decoder reads AROUND the compressed layer. Per clip it carries 54.8M "
    "floats across its skips against an input of 1.29M binary voxels -- the "
    "full-resolution skip alone holds 32x more floats than the volume has "
    "voxels. It produces **no codebook, no discrete index, and no reusable "
    "latent**: nothing is shared across clips, nothing is indexable, and there "
    "is no bottleneck a prior could be trained over. Our alphabet is 1024 "
    "tokens over V=961, i.e. 1.24 KB per clip. The skips are what win it the "
    "ranking columns and are precisely what disqualify it as a tokenizer; the "
    "two cannot be had together, because a tokenizer's value comes from "
    "forcing everything through the code.")

DET_NOTE = ("**3D U-Net (det.) is marked `det.` because it has no sampling "
            "distribution.** It is trained to predict the conditional mean of "
            "the hole, so one context gives one field and every \"sample\" from "
            "it is the same field re-thresholded. That is the optimal answer to "
            "a ranking question (AP, F1) and vacuous as an answer to a "
            "distributional one (avalanche, ISI, marginal realism), so its two "
            "kinds of column must not be read the same way. `3D CVAE` is the "
            "same backbone with a latent variable added and nothing else "
            "changed, so the gap between the two arms is what stochasticity "
            "costs and buys here.")


def _is_learned(k) -> bool:
    return MODEL_CLASS.get(k, "learned") == "learned"


def _split_class(have):
    """(learned, lookup) preserving MODELS order."""
    return ([x for x in have if _is_learned(x[0])],
            [x for x in have if not _is_learned(x[0])])


def _class_rows(have, cell, ncol, fmt, better, decorate=None, ref=None,
                spacer_cols=None):
    """Model rows in two classes, for a table with `ncol` value columns.

    Learned arms are bolded among THEMSELVES -- a lookup table winning a column
    is not a result about modelling, and bolding it says the opposite. Lookups
    follow a spacer, tagged `_ref_`, never bolded.

    `cell(k, j)` returns the raw float for model `k`, column `j`.
    `decorate(text, k, j, v)` optionally post-processes a formatted cell.
    """
    learned, lookup = _split_class(have)
    out = []

    def emit(group, bold, tag):
        if not group:
            return
        cols = []
        for j in range(ncol):
            vals = [cell(k, j) for k, _ in group]
            # `near` needs the target; it may vary per column (adjacency has a
            # different REAL rate in every gap bin), so accept either a scalar
            # or one value per column.
            rj = ref[j] if isinstance(ref, (list, tuple)) else ref
            cols.append(_mark(vals, fmt, better, ref=rj) if bold
                        else ["--" if v != v else fmt.format(v) for v in vals])
        for i, (k, lab) in enumerate(group):
            cs = [cols[j][i] for j in range(ncol)]
            if decorate:
                cs = [decorate(c, k, j, cell(k, j)) for j, c in enumerate(cs)]
            out.append(f"| {tag}{lab} | " + " | ".join(cs) + " |")

    emit(learned, True, "")
    if lookup:
        out.append("| |" + " |" * (spacer_cols or ncol))
        emit(lookup, False, "_ref_ ")
    return out


def _class_cols(have):
    """Column order (learned first, then lookup) plus a per-row marker.

    The reconstruction table is transposed -- models are COLUMNS -- so the
    row-wise helper does not apply. Same rule though: bold competes among the
    learned columns only, and lookup headers carry `_ref_`.
    """
    learned, lookup = _split_class(have)
    order = learned + lookup
    heads = [lab for _, lab in learned] + [f"_ref_ {lab}" for _, lab in lookup]

    def mark(vals_by_key, fmt, better):
        lv = [vals_by_key[k] for k, _ in learned]
        cells = _mark(lv, fmt, better) if better else [
            "--" if v != v else fmt.format(v) for v in lv]
        for k, _ in lookup:
            v = vals_by_key[k]
            cells.append("--" if v != v else fmt.format(v))
        return cells

    return order, heads, mark


REF_NOTE = ("`_ref_` rows are per-assay LOOKUP TABLES, not peer models: they "
            "store one site map per preparation and are shown as memorisation "
            "ceilings. Bold competes between the learned generative models "
            "only. See **Scalability** for what each arm stores per assay.")


NULLS = [("marginal", "_null_ assay site map, SEEN", None),
         ("marginal_xa", "_null_ assay site map, UNSEEN", None),
         ("separable", "_null_ visible profile x site map", None),
         ("copy", "_null_ persistence", None)]
# Canonical schema, never restated locally. utils/constants.py exists so
# "training, inference, metrics, reports and visualization cannot silently
# disagree" -- a hand-copied list here is exactly that disagreement waiting to
# happen.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from MAGVIT_project.utils.constants import (        # noqa: E402
    ACTIVITY_CTX_NAMES, gap_bin_labels)

LCT = list(ACTIVITY_CTX_NAMES)



ARROW = {"high": "\u2191", "low": "\u2193", "near": "\u2248REAL", None: ""}


def _cap(fn, *a, **k) -> str:
    """Run a section renderer and capture what it prints.

    The renderers print straight to stdout and are correct as they stand;
    rewriting ~200 print statements to thread a file handle would risk the
    numbers for no gain. Capturing leaves every renderer untouched and lets
    the same function feed either output file.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*a, **k)
    return buf.getvalue()


def _arrow(better):
    """Direction marker for a metric label.

    Three directions, not two. `near` exists because a co-firing probability
    has no good direction: matching the real value is the goal, and both
    over- and under-producing are failures. Labelling that row with an arrow
    would tell the reader the opposite of the truth.
    """
    a = ARROW[better]
    return f" {a}" if a else ""


def _mark(vals, fmt, better, ref=None):
    """Format a list of numbers, bolding the best.

    `better` is "high", "low", or "near" -- "near" means closest to `ref`, which
    is what "best" means for a co-firing probability: matching the real value,
    not maximising or minimising it. Ties are all bolded, and a row where "best"
    is meaningless (codebook size, perplexity) passes better=None.
    """
    out, keys = [], []
    for v in vals:
        ok = isinstance(v, (int, float)) and v == v
        out.append(fmt.format(v) if ok else "--")
        if not ok or better is None:
            keys.append(None)
        elif better == "high":
            keys.append(-float(v))
        elif better == "low":
            keys.append(float(v))
        else:
            keys.append(abs(float(v) - ref))
    live = [k for k in keys if k is not None]
    if live:
        best = min(live)
        out = [f"**{t}**" if k is not None and k == best else t
               for t, k in zip(out, keys)]
    return out


def _mark_learned(have, vals, fmt, better, ref=None):
    """`_mark`, but bolding competes among LEARNED arms only.

    The report's convention is that DG and the coupled GLM are per-assay lookup
    tables shown as memorisation ceilings, not peers, so a bold on one of their
    cells reads as "the lookup won" -- exactly the reviewer trap the `_ref_`
    row class exists to avoid. `_class_rows` already enforces this where rows
    are models; these are the tables where models are COLUMNS, which the rule
    had not been applied to.
    """
    idx = [i for i, (k, _) in enumerate(have) if _is_learned(k)]
    sub = _mark([vals[i] for i in idx], fmt, better, ref=ref)
    out = []
    j = 0
    for i, v in enumerate(vals):
        if i in idx:
            out.append(sub[j]); j += 1
        else:
            out.append(fmt.format(v)
                       if isinstance(v, (int, float)) and v == v else "--")
    return out


def _chance(NJ, mode, key):
    """Chance AP for `mode` -- the positive prevalence among scored candidates.

    A uniformly random score has expected AP equal to the fraction of
    candidates that are positive, so this is the true floor of each column. It
    is NOT one number: the ROI size differs by task, and the voxel and site
    metrics differ by three orders of magnitude (1.5e-4 vs 3.2e-3), which is
    the whole reason a raw 0.0174 and a raw 0.2635 cannot be compared by eye.
    Computed model-free by $JOB/tmp/chance_levels.py into the canonical nulls
    file.
    """
    try:
        t = NJ["chance"]["tasks"][mode]
        return float(t["voxel_chance" if key == "mean" else "site_chance"])
    except (KeyError, TypeError):
        return float("nan")


def _cfmt(v):
    return "--" if v != v else (f"{v:.2e}" if v < 1e-3 else f"{v:.4f}")


def _x_chance(cell, v, c):
    """Append the multiple of chance, keeping any bold already applied."""
    if v != v or c != c or c <= 0:
        return cell
    return f"{cell} ({v / c:,.0f}\u00d7)"


def _legend(key, orc, nrows, ch):
    """Say what every non-model row IS. A row labelled `_null_ separable` that
    is never defined is worse than no row: the reader cannot tell whether
    losing to it matters.
    """
    what = ("voxel of the ROI" if key == "mean"
            else "electrode touched by the ROI")
    L = ["**Rows.** Bold marks the best MODEL in each column; within the "
         "ceiling and null groups it marks the highest value, i.e. the "
         "hardest bar in that column, not a compliment to the arm.", "",
         f"- `_chance_` -- expected AP of a uniformly random ranking, which "
         f"equals the fraction of candidates that are positive. One candidate "
         f"is one {what}. Every model number is given as a multiple of this.",
         ]
    if orc:
        L.append("- `_ceiling_ ... true tokens` -- that model's OWN tokenizer "
                 "handed the TRUE codes for the hole and asked to decode them. "
                 "It is what the model would score if its prior named every "
                 "hidden token correctly, so the gap below it is the PRIOR's "
                 "failure and the gap above it is the ALPHABET's. Per model, "
                 "not shared: the two tokenizers differ and the distance "
                 "between their ceilings is itself a result. DG and the GLM "
                 "are not tokenizers and have none.")
    d = {"_null_ assay site map, SEEN":
         "the clip's own assay's per-site firing rate, measured on that "
         "assay's TRAIN clips and held constant in time. No model, no "
         "completion -- pure lookup. This is what DG and the GLM reproduce.",
         "_null_ assay site map, UNSEEN":
         "the same lookup with the clip's own assay WITHHELD, averaged over "
         "the other 30. What a site map is worth without having seen this "
         "preparation; the gap to the row above it is how much of the SEEN "
         "row is memorisation.",
         "_null_ visible profile x site map":
         "rank-1 separable: the clip's own VISIBLE frames' activity profile "
         "times the assay train site map. The honest 'marginals plus whatever "
         "you can see' competitor -- it uses no hidden voxel. n/a on `recon`, "
         "where nothing is visible.",
         "_null_ persistence":
         "copy the visible frames adjacent to the hole. Defined only for the "
         "two temporal tasks; `spatial`'s hole spans every frame and `recon` "
         "has no visible frame, so both are n/a rather than faked."}
    for lab, _ in nrows:
        if lab in d:
            L.append(f"- `{lab}` -- {d[lab]}")
    return "\n".join(L) + "\n"


def _task_json():
    """Per-model task_eval payloads. The pipeline's lives under the bare name
    because it was produced before `--model` existed."""
    out = {}
    for k, _ in MODELS:
        # Build the candidate list, rather than filtering an empty name out of
        # it afterwards: `DIR / ""` is the DIRECTORY, whose `.name` is truthy
        # and which `.exists()` happily confirms, so the old guard let it
        # through and `read_text` raised IsADirectoryError. Latent until an arm
        # appeared in MODELS before it had been scored.
        # MaskGIT-flat is read from the CALIBRATED checkpoint's report when it
        # exists. The correction is a constant logit offset, which
        # `_bernoulli_at_rate` absorbs exactly, so every binarised column is
        # unchanged by construction and only the own-count block moves -- see
        # the note under Count calibration for why that block is still not
        # meaningful for this arm.
        cands = [DIR / f"task_eval_{k}.json"]
        if k == "maskgit_flat":
            cands.insert(0, DIR / "task_eval_maskgit_flat_cal.json")
        if k == "pipeline":
            cands.append(DIR / "task_eval.json")
        for cand in cands:
            if cand.is_file():
                out[k] = json.loads(cand.read_text())
                break
    return out


def _nulls_json():
    """The ONE canonical null block, shared by every row.

    The null arms contain no model, so they must not vary by model. They did:
    the train-map sampler (dataset.py:1160, shuffle=True, no generator) seeded
    itself from the global RNG, which each model's init had perturbed by a
    different amount, so every run averaged a different 240 train clips.
    `train_site_maps` now pins its own generator, and this file is that pinned
    computation run once with no prior and no baseline loaded.
    """
    f = DIR / "task_eval_nulls.json"
    return json.loads(f.read_text()) if f.exists() else None


def _derived():
    f = DIR / "derived_stats.json"
    return json.loads(f.read_text()) if f.exists() else None


def _lookup_block(modes) -> None:
    """How each learned arm fares against a MODEL-FREE per-assay site map.

    Without this the voxel-AP table reads as a contest the 3D U-Net wins. It
    does win it -- and every learned arm, that one included, loses to a lookup
    table holding no clip-specific information at all. That is the fact which
    tells the reader what the column is actually sensitive to: WHERE this
    preparation fires, not what this clip did. The margin is the memorisation
    we declined to pay for, not headroom any of these models is close to.
    """
    D = _derived()
    if not D:
        return
    rows = [r for r in D["lookup_vs_arms"]["rows"] if "skipped" not in r]
    if not rows:
        return
    by = {(r["task"], r["arm"]): r for r in rows}
    arms = [(k, lab) for k, lab in MODELS
            if any(a == k for _, a in by)]
    print("**Every learned arm loses to a static per-assay site map.** The row "
          "below is the same canonical `_null_ assay site map, SEEN` already "
          "in the table above, now scored PAIRED against each arm on the same "
          "clips. It contains no clip-specific information whatsoever.\n")
    print("| median delta, lookup - arm (% of clips lookup wins) | "
          + " | ".join(lab for _, lab in modes) + " |")
    print("|---" * (len(modes) + 1) + "|")
    for k, lab in arms:
        cells = []
        for m, _ in modes:
            r = by.get((m, k))
            cells.append("--" if r is None else
                         f"+{r['median_delta']:.4f} ({100*r['lookup_win_frac']:.0f}%)")
        print(f"| {lab} | " + " | ".join(cells) + " |")
    qs = [r["q"] for r in rows]
    print(f"\nAll {len(rows)} comparisons favour the lookup, "
          f"BH-FDR q <= {max(qs):.1e}. Positive means the lookup wins. "
          "Null from the canonical model-free block; pairing verified by "
          "NaN-mask equality per task, since the per-model reports carry no "
          "clip keys.\n")


def task_section(parts=("intro", "ap", "calib", "ceiling", "battery",
                        "tests", "notes")) -> None:
    """Table 2 -- the task axis.

    `parts` selects which blocks to emit so the headline report and the
    appendix can be rendered from the same code path. Splitting by re-running
    with a different selection, rather than by slicing the finished text,
    means a section can never end up in neither file.

    Everything else in this report is measured at task_id=0, while the model
    trains on four tasks at 25% each (dataset.py:452). A completion task leaves
    ground truth inside the hole, so unlike free generation it admits a
    per-voxel accuracy that is paired per clip.
    """
    J = _task_json()
    if not J:
        return
    have = [(k, lab) for k, lab in MODELS if k in J]
    ref = J[have[0][0]]
    modes = [(m, s) for m, s in TASKS if m in ref["tasks"]]
    NJ = _nulls_json()

    def arm(k, mode, a, key="mean"):
        try:
            v = J[k]["tasks"][mode]["arms"][a][key]
        except KeyError:
            return float("nan")
        return float("nan") if v is None else float(v)

    def null(mode, a, key="mean"):
        """Null arms come from the canonical block, never from a model run."""
        src = NJ if NJ is not None else J[have[0][0]]
        alt = {"mean": "ap", "site_mean": "site_ap"}
        for kk in ((key, alt.get(key, key)) if NJ is not None
                   else (key,)):
            try:
                v = src["tasks"][mode]["arms"][a][kk]
            except KeyError:
                continue
            if v is not None:
                return float(v)
        return float("nan")

    uncond = [k for k, _ in have
              if not all(J[k]["tasks"][m].get("conditioned", False)
                         for m, _ in modes)]

    if "intro" in parts:
        print("## Task axis\n")
    n = ref["tasks"][modes[0][0]]["n_clips"]
    if "intro" in parts:
        # Headline keeps only what a reader needs in order to READ the two
        # tables correctly: the sample size, the ROI-fraction caveat (without
        # it the columns look like a difficulty ordering, which they are not),
        # and the warning on the arm that cannot condition at all. Everything
        # else is provenance and lives in the appendix.
        print(f"Per-clip average precision inside the masked region, {n} "
              f"clips, every clip under every task so the columns are paired. "
              f"`model` is the Monte-Carlo mean over {ref.get('mc', 1)} "
              "samplings. Scored on ROI voxels only.\n")
        print("`recon (0)` is free generation, not completion: its ROI is "
              "all-TRUE, so nothing is visible. Read it as the ZERO-CONTEXT "
              "reference, not a fourth peer.\n")
        rf = " / ".join(
            f"{ref['tasks'][m].get('roi_token_frac', float('nan')):.2f}"
            for m, _ in modes)
        print("**Do not read across task columns.** The ROI fraction differs "
              f"by task ({rf}), so the columns have different denominators "
              "and different base rates. Comparisons are valid WITHIN a "
              "column.\n")
        if uncond:
            names = ", ".join(dict(MODELS)[k] for k in uncond)
            print(f"\u26a0 **{names}** has no completion mechanism: its "
                  "clip-level output is a static per-site probability, so it "
                  "cannot read the visible remainder. Its row is FREE "
                  "GENERATION scored on the hole -- a capability statement, "
                  "not a like-for-like score.\n")
        det = [k for k, _ in have if k in DETERMINISTIC]
        if det:
            names = ", ".join(dict(MODELS)[k] for k in det)
            print(f"\u26a0 **{names}** is the DIRECT-SUPERVISION reference, "
                  "not a peer generator: it is trained with this exact loss on "
                  "this exact hole distribution, and AP is a ranking metric, "
                  "so a conditional-mean regressor is the best answer the "
                  "table can contain. It is here to show what the generative "
                  "arms have to buy their way past, and what it costs -- it "
                  "carries no latent, no samples and no reusable "
                  "representation. Compare it with `3D CVAE`, which is the "
                  "same network with a latent variable and nothing else "
                  "changed.\n")

    if "notes" in parts:
        print("## Task axis -- provenance\n")
        print("**`recon (0)` is not a completion task.** Its mask spec yields "
              "an all-TRUE ROI, so nothing is visible and there is nothing to "
              "complete. Verified, not assumed: on a shared batch and seed the "
              "pipeline's task-0 output is BITWISE identical to the shipped "
              "free-generation path, as is MaskGIT-flat's, and with a full ROI "
              "both the GLM's and MaskGIT-flat's `complete()` return the same "
              "volume for the true clip and for a random one -- "
              "clip-independent, i.e. free generation. Dich. Gaussian has no "
              "completion mechanism, so all four of its columns are free "
              "generation.\n")
        if NJ is not None:
            print("The null rows are one canonical block "
                  "(`task_eval_nulls.json`), computed once with no model "
                  "loaded. They contain no model and must not vary by model -- "
                  "they did, by up to 5%, because the shared train loader "
                  "takes its shuffle order from the global RNG and reuses "
                  "persistent workers whose per-item crop RNG had already been "
                  "advanced by whatever was built first. `train_site_maps` now "
                  "draws on its own single-process seeded loader, verified "
                  "byte-identical across processes. Scored clips were never "
                  "affected: the test split is a `DeterministicSubset`.\n")

    for key, title, note in (
        ("mean", "Spatiotemporal AP",
         "**Which VOXEL fires, and when.** Each arm emits a score for every "
         "voxel inside the hole; the voxels are ranked by that score and "
         "scored against the truth with step-wise average precision, one clip "
         "at a time, then averaged over clips. A candidate is one (t, y, x) "
         "voxel of the ROI; a positive is a voxel that really spikes. This is "
         "the full completion problem -- an arm must get the electrode AND "
         "the frame right to earn credit."),
        ("site_mean", "Site-level AP",
         "**Which ELECTRODE fires at all, ignoring when.** Time is collapsed "
         "inside the hole: a candidate is one (y, x) electrode touched by the "
         "ROI, a positive is an electrode that spikes in at least one hole "
         "frame, and an arm's score for it is its mean field over the hole "
         "frames. Together with the table above this splits an arm's "
         "performance into WHERE and WHEN -- an arm that finds the right "
         "electrodes but the wrong frames scores well here and badly above, "
         "which is the signature of a sample being graded by a metric that "
         "wants a posterior. Not redundant with the spatial-support violation "
         "in the battery: that statistic is MARGINAL -- it asks only whether "
         "spikes land on electrodes that are ever active, and saturates at "
         "1.0000 for any arm whose output IS a site map. Only AP is paired to "
         "the clip being completed."),
    ):
        # Voxel AP is the headline completion metric; site AP splits
        # where-from-when and lives in the appendix. Site AP is NOT redundant
        # with spatial-support violation -- see the note under the table.
        want = ("ap" if key == "mean" else "ap_site")
        if want not in parts:
            continue
        if all(arm(k, modes[0][0], "model_mc", key) != arm(k, modes[0][0], "model_mc", key)
               for k, _ in have):
            continue
        print(f"\n### {title}{_arrow('high')}\n\n{note}\n")
        print("| model \u2191 | " + " | ".join(s for _, s in modes) + " |")
        print("|---|" + "---|" * len(modes))
        ch_by_col = [_chance(NJ, m, key) for m, _ in modes]
        for row in _class_rows(
                have, lambda k, j: arm(k, modes[j][0], "model_mc", key),
                len(modes), "{:.4f}", "high",
                decorate=lambda t, k, j, v: _x_chance(t, v, ch_by_col[j])):
            print(row)
        # -- chance. AP's chance level is the positive PREVALENCE, and it is
        # not a constant: it differs by task (the ROI size differs) and by
        # three orders of magnitude between the two metrics. Without this row
        # the reader cannot tell that 0.0174 voxel is a LARGER multiple of
        # chance than 0.2635 site.
        ch = [_chance(NJ, m, key) for m, _ in modes]
        if any(v == v for v in ch):
            print("| | | | | |")
            print("| _chance_ uniformly random score | "
                  + " | ".join("--" if v != v else _cfmt(v) for v in ch) + " |")
            print("| | | | | |")

        # Tokenizer ceiling: PER MODEL, not a shared row. The pipeline and
        # MaskGIT-flat quantize with different codebooks, and the distance
        # between their ceilings is a result in its own right -- collapsing
        # them to one number would hide it. DG and the GLM are not tokenizers
        # and have no oracle. Bold the HIGHER ceiling: that comparison is a
        # result (which alphabet can represent more), not decoration.
        orc = [(k, lab) for k, lab in have
               if any(arm(k, m, "oracle", key) == arm(k, m, "oracle", key)
                      for m, _ in modes)]
        if orc:
            ocols = [_mark([arm(k, m, "oracle", key) for k, _ in orc],
                           "{:.4f}", "high") for m, _ in modes]
            for i, (k, lab) in enumerate(orc):
                print(f"| _ceiling_ {lab} true tokens | "
                      + " | ".join(c[i] for c in ocols) + " |")
            print("| | | | | |")

        # Nulls: bold the STRONGEST one per task -- the bar a model actually
        # has to clear. "Best null" is not a compliment to the null, it is the
        # hardest number in the column, so the reader can see at a glance which
        # null is binding and whether any model cleared it.
        nrows = [(alab, [null(m, a, key) for m, _ in modes])
                 for a, alab, _ in NULLS]
        nrows = [(l, v) for l, v in nrows if not all(x != x for x in v)]
        if nrows:
            ncols = [_mark([v[j] for _, v in nrows], "{:.4f}", "high")
                     for j in range(len(modes))]
            for i, (alab, _) in enumerate(nrows):
                print(f"| {alab} | " + " | ".join(c[i] for c in ncols) + " |")
        print()
        # Full legend once, under the SECOND table; the first gets only the
        # line that actually differs between them. Printing the same six
        # bullets twice cost 9 lines and taught the reader nothing new.
        if key == "site_mean":
            print(_legend(key, orc, nrows, ch))
        else:
            print("A candidate is one voxel of the ROI. Row definitions "
                  "(`_chance_`, `_ceiling_`, `_null_`) follow the next "
                  "table.\n")
            _lookup_block(modes)

    if "calib" in parts:
        _count_calibration(J, have, modes)
    if "tests" in parts:
        _task_tests(J, have, modes, NJ)
    if "battery" in parts:
        _task_battery(J, have, modes)
    if "battery_own" in parts:
        _battery_own(J, have, modes)

    if "notes" in parts:
        print("\nThe oracle row decodes each model's TRUE tokens, so it is that "
              "tokenizer's ceiling rather than a competitor: the gap between a "
              "model and its own oracle is what the PRIOR fails to predict, and "
              "the gap between the oracle and the data is what the ALPHABET "
              "cannot represent.\n")
        print("`marginal SEEN` reads the test clip's own assay off a table built "
              "from that assay's train clips. The split is temporal within assay, "
              "so no model here ever faces an unseen assay and that lookup is "
              "never charged for it -- which is exactly what DG and the GLM "
              "memorise. `marginal UNSEEN` is the same null with the clip's own "
              "assay withheld. Neither row establishes transfer for a learned "
              "model; that needs a retrain with assays held out.\n")
        print("`separable` collapses onto `marginal` on the two temporal tasks by "
              "construction: their holes span whole frames, so no visible voxel "
              "lies inside the hole and the clip-derived temporal profile is "
              "constant there. At SITE level it collapses onto `marginal` on every "
              "task, including `spatial`: collapsing time turns the per-frame "
              "profile into one positive per-clip scalar, which cannot reorder "
              "sites.\n")

        # Ranges computed from the same arms the tables above print. Typing
        # them once worked until a re-run moved them; the `d=3` note elsewhere
        # in this file is what that failure looks like when nobody notices.
        # Per model, not pooled. Pooling these hid a sign: the previous
        # hand-written version claimed "+0.002 to +0.012 (voxel)", which was
        # the GLM's range only -- DG is BELOW the seen null at voxel level.
        lookups = [(k, lab) for k, lab in have if k in ("dg", "glm")]
        if lookups:
            nl = [null(m, "marginal", "mean") for m, _ in modes]
            ns = [null(m, "marginal", "site_mean") for m, _ in modes]
            print("**Reading the winners.** DG and the GLM top both tables. "
                  "Their margin over the SEEN site map -- the lookup they "
                  "reproduce -- is:\n")
            print("| margin over `marginal SEEN` | voxel | site |")
            print("|---|---|---|")
            for k, lab in lookups:
                dv = [arm(k, m, "model_mc", "mean") - null(m, "marginal", "mean")
                      for m, _ in modes]
                ds = [arm(k, m, "model_mc", "site_mean")
                      - null(m, "marginal", "site_mean") for m, _ in modes]
                dv = [v for v in dv if v == v]
                ds = [v for v in ds if v == v]
                # No "winner" here -- the table shows how small these
                # margins are. Bold would read as praise, so the mark is the
                # finding itself: a range entirely below zero is a LOSS to
                # the very lookup the arm is accused of reproducing.
                def _cell(v):
                    t = f"{min(v):+.3f} to {max(v):+.3f}"
                    return t + (" **(below the null)**" if max(v) < 0 else "")
                print(f"| {lab} | {_cell(dv)} | {_cell(ds)} |")
            print(f"\nThe nulls they are beating sit at {min(nl):.2f}-"
                  f"{max(nl):.2f} (voxel) and {min(ns):.2f}-{max(ns):.2f} "
                  "(site), so these are small margins on a very strong "
                  "lookup, and several are not significant (see the table "
                  "above). They are not completing the hole; they reproduce a "
                  "per-assay lookup table and add a small temporal correction "
                  "on the tasks where coupling filters can act. Section C "
                  "withholds that table and both fall below their own null.\n")

    if "ceiling" in parts:
        _ceiling_recovery(J, have, modes)




def _scalability(J, have) -> None:
    """Where each arm keeps its capacity, and what that costs at scale.

    This is the load-bearing section for the paper's claim, and it is a fact
    about the checkpoints rather than about any evaluation run, so it is
    measured by `external_baselines/param_census.py` and only rendered here.
    """
    f = DIR / "param_census.json"
    if not f.exists():
        return
    C = json.loads(f.read_text())
    A, proj = C["arms"], C["projection_assays"]
    rows = [(k, lab) for k, lab in have if k in A]
    if not rows:
        return

    print("\n## Scalability: what each arm stores per assay\n")
    print("The dataset here has 31 preparations. The question the paper has to "
          "answer is what happens at 1000. An arm whose capacity lives in a "
          "per-assay table does not have a scaling problem in principle -- it "
          "has one in practice, because every new preparation adds a full "
          "site map that must be estimated from that preparation's own data "
          "and stored forever.\n")

    print("| arm | fitted, shared | shared maps | stored PER ASSAY | "
          "memorised : fitted |")
    print("|---|---|---|---|---|")
    for k, lab in rows:
        a = A[k]
        pa, fit = a["per_assay"], a.get("fitted", a["shared"])
        ratio = (f"{a.get('per_assay_total', 0) / fit:,.0f} : 1"
                 if fit and pa else "--")
        tag = "" if _is_learned(k) else "_ref_ "
        print(f"| {tag}{lab} | {fit:,} | {a.get('shared_maps', 0):,} | "
              f"{'**0**' if pa == 0 else format(pa, ',')} | {ratio} |")

    # Training budget, so "your baseline was undertrained" is answerable from
    # the artifacts. It is the fair-comparison question a reviewer asks the
    # moment a baseline loses, and the honest answer is a curve slope, not an
    # assurance.
    bud = [(k, lab, A[k]["budget"]) for k, lab in rows
           if isinstance(A[k].get("budget"), dict)
           and A[k]["budget"].get("epochs_run")]
    if bud:
        print("\n**Training budget of the learned arms.** `val slope` is the "
              "mean validation improvement per epoch over the last five: near "
              "zero means the arm converged, visibly positive means the "
              "schedule ran out before the model did and its number is a lower "
              "bound on what the method can do.\n")
        print("| arm | train clips | epochs | selected at | val slope, last 5 |"
              " converged |")
        print("|---|---|---|---|---|---|")
        for k, lab, b in bud:
            sl = b.get("val_slope_last5")
            stopped = b.get("early_stopped")
            conv = ("--" if stopped is None else
                    "yes, early-stopped" if stopped else
                    "**budget-limited**")
            print(f"| {lab} | {b.get('n_train_clips') or '--'} | "
                  f"{b['epochs_run']} | {b.get('best_epoch') or '--'} | "
                  f"{'--' if sl is None else f'{sl:+.5f}'} | {conv} |")

    print("\n| per-assay storage at | " + " | ".join(f"{n} assays" for n in proj)
          + " | growth |")
    print("|---|" + "---|" * (len(proj) + 1))
    for k, lab in rows:
        tag = "" if _is_learned(k) else "_ref_ "
        g = "**flat**" if A[k]["per_assay"] == 0 else "linear"
        print(f"| {tag}{lab} | "
              + " | ".join(f"{A[k]['per_assay'] * n:,}" for n in proj)
              + f" | {g} |")

    print("\nThe learned arms store nothing per assay: `gct` is a fixed random "
          "+/-1 code regenerated from seed 0 (`dataset.py:403-415`), a handle "
          "rather than a table. The lookup arms store a full (H, W) site map "
          "each, so their footprint grows linearly and is already ~3x our "
          "entire model at 1000 preparations -- while the part of them that "
          "is shared across assays is 26 fitted values (DG) and 5,194 (GLM).\n"
          "\nThe claim here is scalability and nothing wider. A random code is "
          "an assay HANDLE, so a new preparation still needs training exposure "
          "and no number in this report measures transfer to an unseen "
          "preparation; the split is temporal within assay. What is measured is "
          "that parameter cost does not grow with the number of preparations. "
          "`gct` also reaches the priors through exactly one frozen mapper "
          "(`CtxEmbed`, `main.py:1492`), so substituting measured descriptors -- "
          "unit map, ISI distribution, stimulation protocol, DIV, cell type -- "
          "for the random code is a change to that module alone. That is stated "
          "as future work, not as a result.\n")

    # The ablation that turns a storage argument into a capability argument.
    nom = {}
    for k, lab in rows:
        d = J.get(k + "_NOMAP")
        if d is None:
            continue
        r = d["context_and_space"]["regimes"].get("global_full_local", {})
        b = J[k]["context_and_space"]["regimes"].get("global_full_local", {})
        nom[lab] = (b.get("within_assay_gap"), r.get("within_assay_gap"),
                    b.get("lct_r_mean_vs_used"), r.get("lct_r_mean_vs_used"))
    if nom:
        print("**Withhold the table and the capability goes with it.** Same "
              "models, same clips, per-assay site map replaced by the global "
              "one:\n")
        # Bold marks the COLLAPSE FACTOR, not the smaller number -- bolding a
        # degraded value would invert the convention used everywhere else.
        print("| _ref_ arm | within-assay gap: with map \u2192 without | "
              "adherence: with map \u2192 without |")
        print("|---|---|---|")
        def _fm(v):
            return "--" if v is None else f"{v:+.4f}"
        def _drop(a, b):
            if a is None or b is None:
                return ""
            if b <= 0 < a:
                return "  (**collapses past zero**)"
            if a and b:
                return f"  (**{a / b:.0f}x worse**)"
            return ""
        for lab, (g0, g1, a0, a1) in nom.items():
            print(f"| {lab} | {_fm(g0)} \u2192 {_fm(g1)}{_drop(g0, g1)} | "
                  f"{_fm(a0)} \u2192 {_fm(a1)}{_drop(a0, a1)} |")
        print("\nThe GLM's adherence goes NEGATIVE: without a per-assay map it "
              "does not merely degrade, it stops tracking the requested "
              "context at all. That is the sense in which these arms are "
              "ceilings rather than methods -- what they score is the map, and "
              "the map is exactly the thing that does not scale.\n")

    print("**What this does and does not claim.** " + C["caveat"] + " We "
          "therefore make the narrow claim: parameter cost is flat in the "
          "number of preparations, and the fitted capacity is shared rather "
          "than per-assay. We do not claim zero-shot transfer to an unseen "
          "preparation, and no table here measures it.\n")


def _count_calibration(J, have, modes) -> None:
    """How many spikes does each arm think belong in the hole?

    The battery's shared readout takes the count from the assay's TRAIN rate,
    which makes the arms comparable but also makes the count a CONSTANT within
    an assay -- on `recon`, where the ROI is the whole volume, its within-assay
    variance is exactly zero. That is not a neutral choice for a model with an
    activity count head, whose whole job is to predict per-clip count from
    context. This table restores the comparison the readout removes.

    Reported as WITHIN-ASSAY correlation with the true ROI count. Pooled
    correlation would flatter both arms equally, because both track assay
    identity and assay identity explains most of the variance; the question is
    whether a count moves with the CLIP once the assay is fixed.
    """
    def cc(k, m):
        return J.get(k, {}).get("tasks", {}).get(m, {}).get("count_calibration")

    if not any(cc(k, m) for k, _ in have for m, _ in modes):
        return
    ref = next(cc(k, m) for k, _ in have for m, _ in modes if cc(k, m))

    print("\n## Count calibration -- how many spikes, per clip\n")
    print("The per-task battery (appendix) scores binarised volumes, and it "
          "sets the spike count to "
          "to `round(assay_train_rate x |ROI|)`. That is ONE NUMBER PER ASSAY. "
          "Real clips inside an assay differ a lot -- true ROI counts have sd "
          f"{ref['true_sd']:.0f} spikes on the first task below -- so the "
          "readout cannot express per-clip activity even in principle, and it "
          "gives every model the same count regardless of what the model "
          "predicted.\n")
    print("`within-assay r` is the correlation with the TRUE ROI count after "
          "removing each assay's mean. `--` means the quantity is a constant "
          "within the assay and has no correlation to compute; that is the "
          "shared readout's row, and it is the point of this table.\n")

    NJ = _nulls_json() or {}
    ctrl = (NJ.get("count_control_lct_only") or {}).get("tasks", {})

    print("| within-assay r with true ROI count | "
          + " | ".join(s_ for _, s_ in modes) + " |")
    print("|---|" + "---|" * len(modes))
    # The imposed count is model-free, so it is one row, not four.
    vals = []
    for m, _ in modes:
        c = next((cc(k, m) for k, _ in have if cc(k, m)), None)
        vals.append(None if not c else c.get("assay_r_within_assay"))
    print("| _null_ assay train rate (the shared readout) | "
          + " | ".join("--" if v is None else f"{v:+.4f}" for v in vals) + " |")
    if ctrl:
        # THE control. Every arm is handed the true clip lct at generation
        # time and lct[0] is log_mean_firing_density, so exp(lct[0]) * |ROI|
        # predicts the count with no model at all. A count head only earns
        # credit for what it scores ABOVE this line.
        print("| _null_ lct arithmetic, no model | " + " | ".join(
            ("--" if ctrl.get(m, {}).get("r_within_assay") is None
             else f"{ctrl[m]['r_within_assay']:+.4f}") for m, _ in modes)
            + " |")
    hv = [(k, lab) for k, lab in have if any(cc(k, m) for m, _ in modes)]
    lrn, lkp = _split_class(hv)

    def _r(k, j):
        v = (cc(k, modes[j][0]) or {}).get("own_r_within_assay")
        return float("nan") if v is None else v

    def _emit(group, bold, tag):
        if not group:
            return
        cols = []
        for j in range(len(modes)):
            vals = [_r(k, j) for k, _ in group]
            cols.append(_mark(vals, "{:+.4f}", "high") if bold
                        else ["--" if v != v else f"{v:+.4f}" for v in vals])
        for i, (k, lab) in enumerate(group):
            print(f"| {tag}{lab}, own predicted count | "
                  + " | ".join(c[i] for c in cols) + " |")

    _emit(lrn, True, "")
    if lkp:
        print("| |" + " |" * len(modes))
        _emit(lkp, False, "_ref_ ")
    if ctrl:
        rec = modes[0][0]
        print(f"\n**Read this against the `lct arithmetic` row, not against "
              "zero.** Every arm is handed the TRUE clip's lct at generation "
              "time, and its first feature is `log_mean_firing_density` -- the "
              "clip's total count. So `exp(lct[0]) x |ROI|` predicts the ROI "
              "count with no model whatsoever. On "
              f"`{rec}` that control scores "
              f"{ctrl.get(rec, {}).get('r_within_assay', float('nan')):+.4f}: "
              "the ROI is the whole volume there, so lct[0] IS the answer and "
              "no count head can beat arithmetic. On the three tasks where the "
              "ROI is a strict subset the control is much weaker, and that gap "
              "is where a count head actually earns its place.\n")

    print("\nAbsolute accuracy of the same counts -- MAE in spikes, and bias "
          "as a fraction of the true mean:\n")
    print("| count MAE / bias | " + " | ".join(s_ for _, s_ in modes) + " |")
    print("|---|" + "---|" * len(modes))
    am = [(next((cc(k, m) for k, _ in have if cc(k, m)), None)) for m, _ in modes]
    # Rows: the shared readout first, then each arm's own count. The shared
    # readout COMPETES here rather than sitting outside the comparison -- the
    # whole question is whether a model's own count beats a lookup, and it
    # currently does not on absolute error even though it wins on ranking.
    labels_ = ["_readout_ assay train rate (shared)"]
    keys_ = [None]
    maes = [[c["assay_mae"] if c else float("nan") for c in am]]
    biases = [[c["assay_bias"] if c else float("nan") for c in am]]
    degen = [[False] * len(modes)]
    for k, lab in have:
        if not any(cc(k, m) for m, _ in modes):
            continue
        labels_.append(f"{lab}, own predicted count")
        keys_.append(k)
        row_m, row_b, row_d = [], [], []
        for m, _ in modes:
            c = cc(k, m)
            bad = bool(c) and c["own_sd"] == 0 and c["own_mean"] == 0
            row_d.append(bad or not c)
            row_m.append(float("nan") if (not c or bad) else c["own_mae"])
            row_b.append(float("nan") if (not c or bad) else c["own_bias"])
        maes.append(row_m); biases.append(row_b); degen.append(row_d)

    # Bold among LEARNED arms only. The GLM genuinely has the best absolute
    # count calibration and that is said in prose beneath -- but bolding a
    # per-assay lookup as the winner is the reading this report exists to
    # prevent. Row 0 is the model-free readout and never competes.
    # `keys_` tracks which arm produced each row so bolding can be restricted
    # to the learned ones. Row 0 is the model-free readout and competes with
    # nobody. The GLM has the best absolute calibration and that is stated in
    # prose below -- bolding a per-assay lookup as the winner is exactly the
    # reading this report exists to prevent.
    # No bolding of the numbers here. "Best among the learned arms" would put
    # bold on a 1100-spike MAE at +576% bias, which reads as a win; and bolding
    # the GLM would crown a per-assay lookup. What matters is binary and
    # factual: is this count usable at all? Mark rows within +/-25% bias.
    CAL = 0.25
    for i, lab in enumerate(labels_):
        tag = "" if (keys_[i] is None or _is_learned(keys_[i])) else "_ref_ "
        cells = []
        for j in range(len(modes)):
            if degen[i][j]:
                cells.append("_n/a_")
            else:
                v = maes[i][j]
                cells.append(("--" if v != v else f"{v:.1f}")
                             + f" / {biases[i][j]:+.1%}")
        ok = [abs(b) <= CAL for b, d in zip(biases[i], degen[i]) if not d]
        flag = "  **calibrated**" if ok and all(ok) else ""
        print(f"| {tag}{lab}{flag} | " + " | ".join(cells) + " |")
    print(f"\n**calibrated** marks an arm whose count bias stays within "
          f"+/-{CAL:.0%} on every task. Nothing is bolded on value: the best "
          "learned arm here is still ~6x over-count, and the only arm that is "
          "calibrated is a per-assay lookup.")

    if "maskgit_flat" in J:
        print(
            "\n**MaskGIT-flat's own-count row is not a meaningful measurement "
            "and should not be read as one.** It has no count head; the number "
            "is the ROI sum of `sigmoid` over its tokenizer decoder, which is "
            "trained with `pos_weight`. Raw, that decoder overstates the "
            "log-odds by `log w` ~ 9.0 nats and the row read about +3800%. The "
            "cost-sensitive inversion (Elkan, IJCAI 2001) that fixes the two "
            "3-D conv arms -- bias +16.7% and +97.3% on `recon` -- overshoots "
            "here to about -94%, because that inversion assumes a head near "
            "the weighted-BCE minimiser and this one is a SATURATING VQ "
            "decoder applied to sampled tokens. Both numbers are wrong in "
            "different directions and neither is quoted as its calibration. "
            "The correction is applied for consistency with the other arms, "
            "not because it calibrates this one; choosing between +3800% and "
            "-94% on which looks better would be tuning to the metric. Every "
            "binarised column is unaffected either way -- the readout solves "
            "for a per-clip shift that absorbs any constant offset exactly.")

    # THE demonstration. The whole reason the shared readout needed defending
    # is that it flattens this row to a constant; show what the row does once
    # the model supplies the count.
    def bo(k, m):
        return J.get(k, {}).get("tasks", {}).get(m, {}).get("battery_own")
    def bs(k, m):
        return J.get(k, {}).get("tasks", {}).get(m, {}).get("battery")
    if any(bo(k, m) for k, _ in have for m, _ in modes):
        nm = (bs(have[0][0], modes[0][0]) or {}).get("activity_ctx_names", [])
        if "log_mean_firing_density" in nm:
            i0 = nm.index("log_mean_firing_density")
            print("\n**What the count actually buys: `log_mean_firing_density` "
                  "under both readouts.** This feature is `log(mean + 1e-6)` "
                  "over a fixed volume, so it measures the spike COUNT and "
                  "nothing else -- it is the one lct feature that a count "
                  "prediction can move. MAE against the clip's own lct:\n")
            print("| `log_mean_firing_density` MAE \u2193 | "
                  + " | ".join(s_ for _, s_ in modes) + " |")
            print("|---|" + "---|" * len(modes))
            sh = [(bs(have[0][0], m) or {}).get("local_ctx", {}).get("mae", [float("nan")])[i0]
                  for m, _ in modes]
            print("| rate-matched (identical for every model) | "
                  + " | ".join(f"{v:.4f}" for v in sh) + " |")
            rows2 = [(lab, [(bo(k, m) or {}).get("local_ctx", {}).get("mae", [float("nan")])[i0]
                            for m, _ in modes])
                     for k, lab in have if any(bo(k, m) for m, _ in modes)]
            c2 = [_mark([r[1][j] for r in rows2], "{:.4f}", "low")
                  for j in range(len(modes))]
            for i, (lab, _) in enumerate(rows2):
                print(f"| own count: {lab} | "
                      + " | ".join(c[i] for c in c2) + " |")
            print("\nUnder the rate-matched readout the row is a CONSTANT: "
                  "every model writes the same number of spikes, so the count "
                  "error is the readout's, not the model's. Let each model "
                  "supply its own count and the row separates them completely "
                  "-- which is the point of having both readouts, and the "
                  "direct answer to what count prediction is worth here.\n")

    print("\n**Discrimination and calibration are separate capabilities, and "
          "no arm has both.** The two tables above disagree about who wins, "
          "which is the finding rather than a contradiction: ranking which "
          "clip is busier within an assay, and knowing how many spikes that "
          "means, are different problems. An arm built on a decode "
          "probability field ranks well and is scaled badly, because the ROI "
          "sum of such a field is not a calibrated expected count; a point "
          "process fitted with an explicit rate gets the level right and "
          "discriminates poorly. The shared readout in the battery above is a "
          "third point: correct level, zero discrimination. That is why the "
          "battery uses it -- it is the only one comparable across arms -- and "
          "why this table exists separately.\n")


def _battery_own(J, have, modes) -> None:
    """The per-task battery again, under each arm's OWN predicted count."""
    def bat(k, m):
        return J.get(k, {}).get("tasks", {}).get(m, {}).get("battery_own")

    if not any(bat(k, m) for k, _ in have for m, _ in modes):
        return
    ref = next(bat(k, m) for k, _ in have for m, _ in modes if bat(k, m))
    names = ref["activity_ctx_names"]

    print("\n## Per-task battery under each model's OWN count\n")
    print(f"Readout `{ref.get('readout')}`: k is the arm's own ROI "
          "probability sum instead of the assay train rate. Same three tables "
          "as the headline battery, same canonical schema.\n")
    print("This is reported for completeness, not as the primary result. The "
          "probability sum is not a calibrated count -- it over-produces "
          "several-fold -- so these volumes are too dense and `lct` MAEs are "
          "WORSE here than under the shared readout even though the same "
          "model's count RANKING is near-perfect within an assay. See the "
          "count-calibration table in the headline report.\n")

    for mode, mlab in modes:
        rows = [(k, lab, bat(k, mode)) for k, lab in have if bat(k, mode)]
        if not rows:
            continue
        real = rows[0][2]
        print(f"\n### {mlab}\n")
        print("Local context, MAE against the clip's own lct \u2193  "
              "(own-count readout)\n")
        hv3 = [(k, lab) for k, lab, _ in rows]
        bm3 = {k: b for k, _, b in rows}
        lr3, lk3 = _split_class(hv3)
        print("| feature | " + " | ".join(
            [lab for _, lab in lr3] + [f"_ref_ {lab}" for _, lab in lk3]) + " |")
        print("|---|" + "---|" * len(rows))
        for i, nm in enumerate(names):
            cells = _mark([bm3[k]["local_ctx"]["mae"][i] for k, _ in lr3],
                          "{:.4f}", "low")
            cells += [f"{bm3[k]['local_ctx']['mae'][i]:.4f}" for k, _ in lk3]
            print(f"| {nm} | " + " | ".join(cells) + " |")
        print("\nSpatial consistency \u2248REAL | "
              + " | ".join(f"{lab} {b['spatial_consistency']['model']:.4f}"
                           for _, lab, b in rows)
              + f" | REAL {real['spatial_consistency']['real']:.4f}\n")


def _bh(ps):
    """Benjamini-Hochberg q-values, in the input order."""
    import numpy as _np
    ps = _np.asarray(ps, dtype=float)
    n = ps.size
    o = _np.argsort(ps)
    q = _np.empty(n, dtype=float)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = o[rank]
        prev = min(prev, ps[i] * n / (rank + 1))
        q[i] = prev
    return q


def _task_tests(J, have, modes, NJ) -> None:
    """Paired Wilcoxon of each model against the model-free nulls.

    The tests stored inside each task_eval JSON compare the SINGLE-draw `model`
    arm against that run's own null columns. The tables above report `model_mc`
    against the canonical nulls, so those stored p-values do not test what is
    displayed. These do. Clips are matched by position, which is sound because
    the test loader is a `DeterministicSubset` (dataset.py:1134) -- asserted
    below against the `copy` arm, which is a pure function of the clip and
    therefore must agree across files if the ordering does.
    """
    if NJ is None:
        return
    from scipy.stats import wilcoxon

    # -- alignment proof
    align = []
    for k, _ in have:
        for m, _ in modes:
            a = J[k]["tasks"][m].get("per_clip", {}).get("copy")
            b = NJ["tasks"][m].get("per_clip", {}).get("copy")
            if not a or not b or len(a) != len(b):
                continue
            ok = all((x != x and y != y) or abs(x - y) < 1e-12
                     for x, y in zip(a, b))
            align.append(ok)
    if align and not all(align):
        print("\n> \u26a0 clip alignment check FAILED; paired tests omitted.\n")
        return

    rows, ps = [], []
    for key, klab in (("per_clip", "voxel"), ("per_clip_site", "site")):
        for k, lab in have:
            for a, alab in (("marginal", "SEEN"), ("marginal_xa", "UNSEEN")):
                for m, mlab in modes:
                    u = J[k]["tasks"][m].get(key, {}).get("model_mc")
                    v = NJ["tasks"][m].get(key, {}).get(a)
                    if not u or not v or len(u) != len(v):
                        continue
                    d = [(x - y) for x, y in zip(u, v)
                         if x == x and y == y]
                    if len(d) < 5 or all(abs(x) < 1e-15 for x in d):
                        continue
                    stat = wilcoxon(d)
                    rows.append([klab, lab, alab, mlab,
                                 float(np.mean(d)), float(stat.pvalue)])
                    ps.append(float(stat.pvalue))
    if not rows:
        return
    qs = _bh(ps)

    print("\n### Model vs null, paired\n")
    print("Wilcoxon signed-rank on per-clip differences, every clip under "
          "every task. `delta` is mean(model_mc - null); positive means the "
          "model wins. q is Benjamini-Hochberg across all "
          f"{len(rows)} tests in this table.\n")
    print("| metric | model | vs null | task | delta | q |")
    print("|---|---|---|---|---|---|")
    # Each row is a separate test, so "best in column" is meaningless here.
    # What the reader needs marked is the DIRECTION plus significance: a bold
    # delta is a win over that null that survived BH-FDR. A significant LOSS
    # is left plain -- bolding it would read as an achievement.
    for r, q in zip(rows, qs):
        sig = "" if q < 0.05 else " _(n.s.)_"
        d = f"{r[4]:+.4f}"
        if q < 0.05 and r[4] > 0:
            d = f"**{d}**"
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {d} | {q:.2g}{sig} |")



def _ceiling_recovery(J, have, modes) -> None:
    """model_mc as a fraction of that model's OWN tokenizer ceiling.

    Separates the two failure modes a single AP cannot: a low score because the
    alphabet cannot represent the volume, versus a low score because the prior
    cannot name the right tokens.
    """
    def arm(k, m, a, key):
        try:
            v = J[k]["tasks"][m]["arms"][a][key]
        except KeyError:
            return float("nan")
        return float("nan") if v is None else float(v)

    orc = [(k, lab) for k, lab in have
           if any(arm(k, m, "oracle", "mean") == arm(k, m, "oracle", "mean")
                  for m, _ in modes)]
    if not orc:
        return

    # Computed, never typed. The stale "d=3" note elsewhere in this file
    # survived a data change precisely because it was literal text; any figure
    # quoted in prose has to come from the same JSON as the table under it.
    m0 = modes[0][0]
    _c = {k: (arm(k, m0, "oracle", "mean"), arm(k, m0, "oracle", "site_mean"))
          for k, _ in orc}
    _lab = dict(orc)
    if len(orc) >= 2:
        (k1, k2) = orc[0][0], orc[1][0]
        v1, s1 = _c[k1]
        v2, s2 = _c[k2]
        ratio = v1 / v2 if v2 else float("nan")
        print(f"\n**Reading the ceilings.** At voxel level the two priors "
              f"score within "
              f"{abs(arm(k1, m0, 'model_mc', 'mean') - arm(k2, m0, 'model_mc', 'mean')):.3f} "
              f"of each other, but they sit on tokenizers that are not "
              f"comparable: on `{m0}` the ceiling is **{v1:.4f}** for "
              f"{_lab[k1]} against **{v2:.4f}** for {_lab[k2]}, a factor of "
              f"**{ratio:.1f}**. At SITE level the same two ceilings are "
              f"{s1:.4f} and {s2:.4f} -- the hierarchical alphabet buys "
              f"resolution in TIME, not in space. What each prior then "
              f"recovers of its own ceiling:\n")
    else:
        print("\n**Reading the ceilings.** What each prior recovers of its "
              "own tokenizer ceiling:\n")
    print("| model_mc / own oracle | " + " | ".join(s for _, s in modes) + " |")
    print("|---|" + "---|" * len(modes))
    for key, klab in (("mean", "voxel"), ("site_mean", "site")):
        # Bold WITHIN the metric group: comparing a voxel recovery against a
        # site recovery is meaningless, so the two groups are marked apart.
        pct = []
        for k, lab in orc:
            row = []
            for m, _ in modes:
                mm, oc = arm(k, m, "model_mc", key), arm(k, m, "oracle", key)
                row.append(100.0 * mm / oc if (oc == oc and oc > 0)
                           else float("nan"))
            pct.append(row)
        cols = [_mark([r[j] for r in pct], "{:.0f}%", "high")
                for j in range(len(modes))]
        for i, (k, lab) in enumerate(orc):
            print(f"| {klab}: {lab} | " + " | ".join(c[i] for c in cols) + " |")

    print("\nThe two rows point opposite ways. Our prior recovers most of what "
          "its alphabet allows for WHERE and almost none of it for WHEN, so "
          "the binding constraint on us is the prior's timing, not the "
          "tokenizer. MaskGIT-flat recovers a larger share at voxel level but "
          "of a ceiling five times lower, and only about half at site level: "
          "its binding constraint is the flat alphabet. Above 100% is possible "
          "because `model_mc` averages 8 samplings into a smoother ranking "
          "while the oracle is a single decode of the true tokens; the oracle "
          "bounds what the ALPHABET can express, not what an averaged field "
          "can score.\n")


def _task_battery(J, have, modes) -> None:
    """One uniform block per task, on the canonical schema.

    Same three tables for recon / causal / noncausal / spatial, in the same
    order, with labels taken from utils/constants.py: the short-gap adjacency
    bins (`DEFAULT_GAP_BINS`), spatial consistency (1 - the hard
    spatial-support violation fraction that `evaluate_generation_global_metrics`
    reports), and the nine `ACTIVITY_CTX_NAMES` local-context features.
    """
    def bat(k, mode):
        return J.get(k, {}).get("tasks", {}).get(mode, {}).get("battery")

    if not any(bat(k, m) for k, _ in have for m, _ in modes):
        return

    ref = next(bat(k, m) for k, _ in have for m, _ in modes if bat(k, m))
    labels = ref["gap_labels"]
    names = ref["activity_ctx_names"]

    print("\n### Per-task battery, canonical schema\n")
    print("Same three tables for every task. Bins are `DEFAULT_GAP_BINS` and "
          "features are `ACTIVITY_CTX_NAMES`, both from `utils/constants.py`; "
          "the adjacency rate is `_hard_gap_rates` and the violation fraction "
          "is `evaluate_generation_global_metrics`, the same functions "
          "training reports, imported rather than reimplemented.\n")
    print("**Readout: rate-matched.** These three statistics are defined on "
          "BINARY spike trains, but every arm emits a continuous field, so a "
          "binarisation is required before any of them can be computed. Two "
          "choices exist and both are reported:\n")
    print("- **Rate-matched** (this section). Every arm is given the same "
          "spike budget -- the top `round(assay_train_rate x |ROI|)` voxels of "
          "its own ranking -- so all arms emit an equal number of spikes and "
          "the only thing that can differ is WHERE they go. This isolates "
          "placement from count, and it is what makes arms with incompatible "
          "score scales (our probabilities, MaskGIT's logits, the GLM's "
          "intensities) comparable at all. The budget is taken from TRAIN "
          "clips of the same assay, never from the held-out clip.\n")
    print("- **Own-count** (`Count calibration` above, and the appendix). "
          "Each arm sets its own budget from its own field. This measures the "
          "deployed system, count errors included, and is where a count head "
          "can earn credit.\n")
    print("Rate-matching is the control, not the headline claim: an arm that "
          "wins here wins on placement, and its count accuracy is reported "
          "separately rather than being allowed to leak into every "
          "distribution statistic at once.\n")
    print("Two consequences of rate-matching, true in all four blocks and so "
          "stated once here:\n")
    print("1. **`log_mean_firing_density` is not scored here.** It is "
          "`log(mean + 1e-6)` over a fixed volume, so it depends only on the "
          "spike COUNT -- and the readout sets the count, identically for "
          "every model. Measured directly on one batch (a separate check, not "
          "a number this report re-renders): the arms shared as little as 2% "
          "of their spikes while writing exactly the same number of them. "
          "The per-block figure below is the readout's count error, not "
          "a model's. See **Count calibration** for what the models "
          "themselves predict.\n")
    print("2. **A spatial consistency of 1.0000 is not a win.** An arm whose "
          "output IS a per-assay site map can only place mass where that map "
          "is already active, so it cannot violate the support by "
          "construction. REAL itself does not score 1.0.\n")

    for mode, mlab in modes:
        rows = [(k, lab, bat(k, mode)) for k, lab in have if bat(k, mode)]
        if not rows:
            continue
        real = rows[0][2]
        print(f"\n#### {mlab}\n")

        print("Short-gap adjacency rate by gap bin \u2248REAL\n")
        print("| gap (frames) | " + " | ".join(labels) + " |")
        print("|---|" + "---|" * len(labels))
        rr = real["adjacency_gap_rate"]["real"]
        print("| REAL | " + " | ".join(f"{v:.5f}" for v in rr) + " |")
        # Bold DOWN each gap bin, closest to REAL -- this is a "near" metric,
        # so neither the largest nor the smallest rate is best. Bolding by
        # magnitude here would mark over-production as a win.
        hv = [(k, lab) for k, lab, _ in rows]
        bmap = {k: b for k, _, b in rows}
        for row in _class_rows(
                hv, lambda k, j: bmap[k]["adjacency_gap_rate"]["model"][j],
                len(rr), "{:.5f}", "near", ref=list(rr),
                decorate=lambda t, k, j, v: (f"{t} ({v / rr[j]:.2f}x)"
                                             if rr[j] else t)):
            print(row)

        rc_ = real["spatial_consistency"]["real"]
        print("\nSpatial consistency \u2248REAL  (1 - hard spatial-support "
              "violation fraction)\n")
        print("| model | spatial consistency | violation fraction |")
        print("|---|---|---|")
        print(f"| REAL | {rc_:.4f} | "
              f"{real['spatial_support_violation']['real']:.4f} |")
        sv = [b["spatial_consistency"]["model"] for _, _, b in rows]
        hv = [(k, lab) for k, lab, _ in rows]
        bmap = {k: b for k, _, b in rows}
        for row in _class_rows(
                hv, lambda k, _j: bmap[k]["spatial_consistency"]["model"],
                1, "{:.4f}", "near", ref=rc_, spacer_cols=2,
                decorate=lambda t, k, _j, _v:
                    f"{t} | {bmap[k]['spatial_support_violation']['model']:.4f}"):
            print(row)

        print("\nLocal context, MAE against the clip's own lct \u2193\n")
        hv2 = [(k, lab) for k, lab, _ in rows]
        bm2 = {k: b for k, _, b in rows}
        learned2, lookup2 = _split_class(hv2)
        print("| feature | " + " | ".join(
            [lab for _, lab in learned2] + [f"_ref_ {lab}" for _, lab in lookup2])
            + " |")
        print("|---|" + "---|" * len(rows))
        fixed = []
        for i, nm in enumerate(names):
            vals = [b["local_ctx"]["mae"][i] for _, _, b in rows]
            # A feature the shared readout pins is identical for every model by
            # construction, not a tie on the merits. `roi_topN_at_assay_train_rate`
            # writes the SAME number of spikes into the ROI whatever the model
            # ranked, and truth fills the rest, so total count -- and therefore
            # log_mean_firing_density -- cannot differ. Bolding all four would
            # read as a four-way win. Report it, do not score it.
            if max(vals) - min(vals) < 1e-9:
                fixed.append(nm)
                cells = [f"{v:.4f}" for v in vals]
            else:
                # bold among learned arms only; lookups are reference columns
                lv = [bm2[k]["local_ctx"]["mae"][i] for k, _ in learned2]
                cells = _mark(lv, "{:.4f}", "low") + [
                    f"{bm2[k]['local_ctx']['mae'][i]:.4f}" for k, _ in lookup2]
            print(f"| {nm} | " + " | ".join(cells) + " |")
        if fixed:
            # These grade the READOUT, not the models -- so say what they grade
            # rather than leaving a dead row. log_mean_firing_density is
            # log(x.mean() + 1e-6) over a fixed volume, i.e. a pure function of
            # the spike COUNT, and `roi_topN_at_assay_train_rate` sets that
            # count from the assay's train rate with the model supplying only
            # the ordering. The residual against the clip's own lct is
            # therefore the count calibration of the readout itself.
            import math as _m
            i0 = (names.index("log_mean_firing_density")
                  if "log_mean_firing_density" in names else None)
            d = None
            if i0 is not None:
                d = _m.exp(real["local_ctx"]["model"][i0]
                           - real["local_ctx"]["real"][i0]) - 1.0
            # The explanation is stated ONCE at section level; only the
            # per-task figure varies, so only the figure is repeated.
            tail = "" if d is None else (
                f" Completed volume carries **{d:+.1%}** of the real clip's "
                "total activity.")
            print("\nNot scored: " + ", ".join(f"`{f}`" for f in fixed)
                  + " (readout-fixed -- see the section note)." + tail)

GEN_METRICS = [
    ("A. Conditional accuracy", "low",
     "Per-clip z-scored MAE between the lct recomputed from the sample and the "
     "TRUE clip's lct, each feature divided by its spread across test clips. "
     "Per-clip, so a paired Wilcoxon applies.",
     lambda reg, k, r: float(np.mean(reg(k, r)["lct_z_mae_per_clip"]))),
    ("B. Adherence", "high",
     "Mean over the 9 features of r(realised, **requested**) -- against the lct "
     "handed to the model, not the true one. A property of the model, not of "
     "how much context it got, so an obedient model is flat across the ladder.",
     lambda reg, k, r: float(reg(k, r)["lct_r_mean_vs_used"])),
    ("C. Spatial placement, lookup-proof", "high",
     "Map correlation against the clip's own electrodes MINUS the same "
     "generated map scored against a DIFFERENT clip of the same assay. "
     "`assay_idx` bypasses the ladder, so raw map r is mostly a per-assay "
     "lookup for DG and the GLM; this difference is the part a fixed site map "
     "cannot fake.",
     lambda reg, k, r: float(reg(k, r)["within_assay_gap"])),
]


def _sre_factory(real):
    """D. Marginal realism, over the CANONICAL gap bins.

    The note this replaced described "3 spatial displacements and 7 temporal
    lags" and a zero at "d=3". That profile was deleted when the diagnostics
    moved onto `DEFAULT_GAP_BINS`; `reg(k, r)["adjacency"]` has been the seven
    canonical short-gap rates ever since, and there are no spatial
    displacements in it at all. The prose survived the data change because a
    grep for `space_d`/`time_lag` cannot see the names spelled out in words.
    """
    def sre(reg, k, r):
        a = np.array(reg(k, r)["adjacency"], float)
        return float(np.nanmean(np.abs(a - real) / (a + real)))
    return sre


def emit_generation_families(have, rungs, reg, real, out_dir: Path) -> None:
    """Dump the four generation families so the paper can render them.

    This module is the only place these four are computed. Until now they
    existed solely as rendered markdown, so the only way to get them into a
    LaTeX table was to retype them -- and a retyped number cannot be traced
    back to the run that produced it. This writes the same values the markdown
    prints, from the same lambdas, at every rung.

    Additive: it does not touch what `report` renders, and diagnostics.md is
    byte-identical with or without the flag.
    """
    fams = GEN_METRICS + [("D. Marginal realism", "low", "", _sre_factory(real))]
    out = {"rungs": [r for r, _ in rungs], "families": {}}
    for title, better, _note, fn in fams:
        blk = {"better": better, "arms": {}}
        for key, label in MODELS:
            if not any(key == k for k, _ in have):
                continue
            blk["arms"][label] = {r: float(fn(reg, key, r)) for r, _ in rungs}
        out["families"][title] = blk
    q = out_dir / "generation_families.json"
    q.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {q}", file=sys.stderr)


def report(J, have, rungs, reg, real, labels, parts) -> None:
    """Render the sections named in `parts`, in report order."""
    def ladder(title, note, fn, better, fmt="{:.4f}"):
        print(f"\n### {title}{_arrow(better)}\n\n{note}\n")
        print(f"| model{_arrow(better)} | " + " | ".join(s_ for _, s_ in rungs) + " |")
        print("|---|" + "---|" * len(rungs))
        for row in _class_rows(have, lambda k, j: fn(k, rungs[j][0]),
                               len(rungs), fmt, better):
            print(row)

    sre = _sre_factory(real)
    GEN = GEN_METRICS + [("D. Marginal realism", "low",
                          "Mean symmetric relative error `|gen-real|/(gen+real)` "
                          "of the short-gap rate over the seven canonical gap "
                          "bins (`DEFAULT_GAP_BINS`). Bounded in [0,1]: 0 "
                          "matches real exactly, 1 is a total miss. Bounded on "
                          "purpose -- a log-ratio explodes when a model emits "
                          "exactly zero rate in some bin. Does not depend on "
                          "conditioning.", sre)]

    if "recon" in parts:
        print("\n## Reconstruction\n")
        print("Step-wise average precision, not the trapezoid AUPRC in "
              "`utils/metrics.py` -- trapezoid interpolates the PR curve "
              "linearly, which is invalid (Davis & Goadrich 2006) and inflated "
              "the saturating MaskGIT tokenizer by +0.22 off a single voxel.\n")
        R = {k: (J[k].get("reconstruction") or {}) for k, _ in have}
        order, heads, mark = _class_cols(have)
        print("| | " + " | ".join(heads) + " |")
        print("|---|" + "---|" * len(order))
        for lab, key, f, better in (
                ("AP step-wise, exact", "ap_step_exact", "{:.4f}", "high"),
                ("AP step-wise, tolerant", "ap_step_tol111", "{:.4f}", "high"),
                ("best F1, exact", "best_f1_exact", "{:.4f}", "high"),
                ("best F1, tolerant", "best_f1_tol111", "{:.4f}", "high"),
                ("trapezoid inflation", None, "{:+.4f}", "low"),
                ("codebook used", "codes_used", "{:.0f}", None),
                ("codebook perplexity", "codebook_perplexity", "{:.1f}", None)):
            vals = {k: ((R[k].get("auprc_exact", float("nan"))
                         - R[k].get("ap_step_exact", float("nan")))
                        if key is None else R[k].get(key, float("nan")))
                    for k, _ in have}
            print(f"| {lab}{_arrow(better)} | "
                  + " | ".join(mark(vals, f, better)) + " |")
        print("\nEmpty cells are a capability fact, not a missing run. This "
              "section measures what an ALPHABET can represent, so it needs a "
              "tokenizer: DG and the GLM are point processes and have none, "
              "and the two 3-D conv arms are inpainters with no autoencoding "
              "path -- they never see the clip they are asked to complete, so "
              "there is nothing for them to reconstruct. The comparison that "
              "does include them is **Task completion**.\n\n" + REF_NOTE + "\n")

    if "gen" in parts:
        # Collapsed: the four metrics at FULL context, with `random` as the
        # null beside each. The five-rung ladders are in the appendix -- the
        # headline claim is "at full context, how does each arm do", and four
        # separate 5-column ladders bury that in 44 lines.
        print("\n## Generation\n")
        print("Four families, no composite. Each cell is the metric at full "
              "context (`glob+full`) with that same model's `random`-context "
              "value in brackets -- its own null, so the bracket says how much "
              "of the number is conditioning rather than the model's default "
              "behaviour. The full five-rung ladders are in the appendix.\n")
        order, heads, mark = _class_cols(have)
        print("| metric | " + " | ".join(heads) + " |")
        print("|---|" + "---|" * len(order))
        for title, better, _note, fn in GEN:
            vals = {k: fn(reg, k, "global_full_local") for k, _ in have}
            nul = {k: fn(reg, k, "random") for k, _ in have}
            cells = [f"{c} _[{nul[k]:.3f}]_"
                     for (k, _), c in zip(order, mark(vals, "{:.4f}", better))]
            print(f"| {title}{_arrow(better)} | " + " | ".join(cells) + " |")
        print("\n`value _[null]_`, the null being the same arm given random "
              "context. An arm whose value and null are close is not using "
              "its context -- which is what the `_ref_` lookups do once their "
              "per-assay map is withheld (see **Scalability**).\n")
        print(DET_NOTE + "\n")
        print(MARK_NOTE + "\n")
        print(REF_NOTE + "\n")

    if "gen_ladders" in parts:
        print("\n## Generation, full conditioning ladders\n")
        print("`local_only` is a control, not a rung: it hands the model the "
              "true lct with a MISMATCHED gct, so it is contradictory "
              "information rather than less of it.\n")
        for title, better, note, fn in GEN:
            ladder(title, note, lambda k, r, _f=fn: _f(reg, k, r), better)

    if "adjacency" in parts:
        print("\n## Short-gap adjacency at full context\n")
        print("Rate P(spike at t+g | spike at t) by canonical gap bin "
              "(`DEFAULT_GAP_BINS`), free generation at the "
              "`global_full_local` rung.\n")
        print("Note this is NOT the same measurement as the `recon` block of "
              "the per-task battery, though both report the same quantity on "
              "the same bins: this one is the shipped free-generation path at "
              "a conditioning rung, that one is the task harness with MC "
              "averaging and a top-N readout. Expect the ratios to differ; "
              "quote whichever matches the claim being made, and say which.\n")
        print("Best = closest to REAL, not largest or smallest.\n")
        print(f"| offset{_arrow('near')} | REAL | "
              + " | ".join(lab for _, lab in have) + " |")
        print("|---|---|" + "---|" * len(have))
        for i, lab in enumerate(labels):
            vals = [reg(k, "global_full_local")["adjacency"][i] for k, _ in have]
            marked = _mark_learned(have, vals, "{:.5f}", "near", ref=real[i])
            cells = [m if not real[i] else f"{m} ({v/real[i]:.2f}x)"
                     for m, v in zip(marked, vals)]
            print(f"| {lab} | {real[i]:.5f} | " + " | ".join(cells) + " |")

        # Per-bin closeness is not the same as reproducing the CURVE. The real
        # profile has a definite shape -- suppressed at gap 1, peaking at 3-6 --
        # and a flat curve sitting near the real mean scores well on every row
        # while having no shape at all. The correlation of the mean-centred,
        # unit-scaled curves separates the two claims; `gap1->3` is the single
        # feature that shape is built on.
        D = _derived()
        g = (D or {}).get("gap_shape")
        if g and g.get("arms"):
            print(f"\nShape of the curve, not its level: correlation with the "
                  f"REAL profile after mean-centring and scaling. REAL rises "
                  f"{g['real_gap1_to_gap3_pct']:+.0f}% from gap 1 to gap 3.\n")
            print(f"| | " + " | ".join(lab for _, lab in have) + " |")
            print("|---|" + "---|" * len(have))
            for nm, fk, f_ in (("shape r vs REAL " + _arrow("high"), "shape_r", "{:.3f}"),
                               ("gap 1 -> gap 3 " + _arrow("near"), "gap1_to_gap3_pct", "{:+.0f}%")):
                vals = [g["arms"].get(k, {}).get(fk, float("nan"))
                        for k, _ in have]
                ref = (1.0 if fk == "shape_r" else g["real_gap1_to_gap3_pct"])
                print(f"| {nm} | " + " | ".join(
                    _mark_learned(have, vals, f_,
                                  "high" if fk == "shape_r" else "near",
                                  ref=ref)) + " |")
            print("\nAn arm can match the LEVEL by being featureless. Read this "
                  "with the table above, not instead of it.\n")

    if "lct_feat" in parts:
        print("\n## Per-feature lct, ours\n")
        print(f"| feature{_arrow('high')} | "
              + " | ".join(s_ for _, s_ in rungs) + " |")
        print("|---|" + "---|" * len(rungs))
        CONTROLS = {"random", "local_only"}
        for f in LCT:
            vals = [reg("pipeline", r)["lct_per_feature"][f]["r"] for r, _ in rungs]
            live = [v for (r, _), v in zip(rungs, vals) if r not in CONTROLS]
            top = max(live) if live else None
            cells = [(f"**{v:+.2f}**" if (r not in CONTROLS and v == top)
                      else f"{v:+.2f}") for (r, _), v in zip(rungs, vals)]
            print(f"| {f} | " + " | ".join(cells) + " |")
        print("\nr against the REQUESTED lct. The spatial-shape features "
              "(var_x, var_y, cov_xy) collapse under `LOCAL only` while rate "
              "and trend survive: the electrode layout arrives through gct, so "
              "a mismatched gct makes a requested spatial variance physically "
              "unrealisable.\n")


# Headline answers one question -- does it follow the context, and does it
# scale -- with the fewest metrics that can. Everything supporting is appended.
ROOT_REPORTS = DIR.parent


def _ablations() -> None:
    """Design choices, each answered by a run rather than by an argument.

    Two questions a reviewer asks about the tokenizer and cannot answer from
    any other table: why three hierarchy levels, and why a sparse encoder with
    a dedicated blank token.
    """
    lad = ROOT_REPORTS / "ablation_ladder_decode_depth.json"
    spa = ROOT_REPORTS / "ablation_sparse_encoder.json"
    con = ROOT_REPORTS / "ablation_sparse_encoder_content.json"
    pat = ROOT_REPORTS / "ablation_patch_size.json"
    if not (lad.is_file() or spa.is_file() or pat.is_file()):
        return
    print("\n## Design choices: why this tokenizer\n")

    if pat.is_file():
        P_ = json.loads(pat.read_text())
        rows = [r for r in P_["rows"] if "skipped" not in r]
        rows = sorted(rows, key=lambda r: r["n_tokens"])
        print("### Patch size -- why (6,15,14)\n")
        print("Model-free: one pass over "
              f"{P_['n_clips']} val clips at voxel rate "
              f"{P_['voxel_rate']:.3e}, nothing trained. Two quantities move in "
              "OPPOSITE directions as the patch changes, and the shipped size "
              "is where both are still tolerable.\n")
        print("- **blank** is the fraction of tokens containing no spike. It is "
              "what the prior is trained against: once nearly every target is "
              "blank, predicting blank is close to optimal and the prior "
              "collapses.\n"
              "- **capture@32** is the fraction of active-patch variance that "
              "32 centroids can describe (k-means on the raw patches, K = the "
              "parent codebook size). It is the ALPHABET side: if 32 entries "
              "cannot describe the patch distribution, no amount of training "
              "fixes it.\n")
        print("| patch | voxels | tokens | blank " + _arrow("low")
              + " | active tok/clip | spikes/active | spike sd | capture@32 "
              + _arrow("high") + " |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            ps = tuple(r["patch"])
            lab = (f"**{ps}** _<- shipped_" if r["is_shipped"] else f"{ps}")
            print(f"| {lab} | {r['voxels_per_patch']} | {r['n_tokens']} "
                  f"| {100*r['blank_frac']:.1f}% | {r['active_tokens']:.0f} "
                  f"| {r['spikes_per_active']:.2f} | {r['spikes_sd']:.2f} "
                  f"| {r['capture_k32']:.3f} |")
        sh = next(r for r in rows if r["is_shipped"])
        sm = max(rows, key=lambda r: r["n_tokens"])
        bg = min(rows, key=lambda r: r["n_tokens"])
        print(f"\n**Smaller patches: the alphabet gets easier and the prior "
              f"gets impossible.** From {tuple(sh['patch'])} to "
              f"{tuple(sm['patch'])} capture@32 rises "
              f"{sh['capture_k32']:.3f} -> {sm['capture_k32']:.3f}, because a "
              f"patch holding {sm['spikes_per_active']:.2f} spikes is easy to "
              f"describe. But blank goes {100*sh['blank_frac']:.1f}% -> "
              f"{100*sm['blank_frac']:.1f}%, and active tokens barely move "
              f"({sh['active_tokens']:.0f} -> {sm['active_tokens']:.0f}) while "
              f"the grid grows {sm['n_tokens']/sh['n_tokens']:.0f}x. The same "
              "signal is spread over far more positions, each almost always "
              "empty. Reconstruction IMPROVES here -- finer patches decode "
              "more sharply -- while generation collapses.\n")
        print(f"**Larger patches: the prior gets easier and the alphabet "
              f"cannot keep up.** At {tuple(bg['patch'])} blank falls to "
              f"{100*bg['blank_frac']:.1f}%, which is what the prior wants, but "
              f"a non-blank token now holds {bg['spikes_per_active']:.2f} "
              f"spikes with sd {bg['spikes_sd']:.2f} against "
              f"{sh['spikes_per_active']:.2f} / {sh['spikes_sd']:.2f} shipped, "
              f"and capture@32 drops {sh['capture_k32']:.3f} -> "
              f"{bg['capture_k32']:.3f}. Thirty-two entries cannot span that "
              "much heterogeneity, so the cost reappears as quantization "
              "error.\n")
        print("**The quantizer has no mechanism to absorb that.** Every "
              "codebook tensor is created with `requires_grad = False` "
              "(`model/base.py:200`) and updated only by EMA, so the code "
              "vectors receive no gradient from any loss. The one usage term, "
              "`max_entropy - entropy` at weight 1e-3 (`model/base.py:424`), "
              "is computed from the distances between ENCODER outputs and a "
              "no-grad codebook, so its gradient reaches the encoder alone: it "
              "is an encoder-side anti-collapse pressure that spreads which "
              "entry gets hit, an INDIRECT influence on the entry point rather "
              "than anything acting on the entries. It also pushes usage "
              "toward UNIFORM, which is the opposite of a sparsity prior. "
              "Nothing in the objective can enlarge what 32 entries are able "
              "to span.\n")
        print("So the patch is chosen for the PRIOR and the ALPHABET jointly, "
              "not for the decoder: reconstruction alone would pick the "
              "smallest patch on the table. The shipped size also fixes the "
              "token budget at 1024, which is what makes a clip fit in memory, "
              "and keeps enough spikes per token for the variance and "
              "covariance context features to be legible. Blank fraction is "
              "the same quantity that binds in **Sparse encoder** below.\n")
        print("_capture@32 is k-means on RAW binary patches, so it measures the "
              "intrinsic diversity of the content at each patch size. It is not "
              "a measurement of our tokenizer, which quantizes 64-d encoder "
              "outputs rather than raw voxels; read it as a relative comparison "
              "ACROSS patch sizes, never as an absolute ceiling._\n")

    if lad.is_file():
        L = json.loads(lad.read_text())
        print("### Ladder depth -- what each hierarchy level buys\n")
        print("Encode ONCE with all three levels, then decode the same codes "
              "at each cumulative depth. Nothing is retrained and nothing is "
              "re-encoded, so the only thing that varies is how much of the "
              "residual sum reaches the decoder. Exact step-wise AP, paired "
              f"over the same {L['n_clips']} clips.\n")
        lv = L["levels"]
        print("| decode depth | AP " + _arrow("high") + " | gain over previous |")
        print("|---|---|---|")
        prev = None
        for nm in ("z1", "z1+z2", "z1+z2+z3"):
            if nm not in lv:
                continue
            m = lv[nm]["mean"]
            inc = ""
            for k_, v_ in L["increments"].items():
                if k_.endswith("-> " + nm):
                    inc = (f"**+{v_['mean_delta']:.4f}** "
                           f"({100*v_['win_frac']:.0f}% of clips, "
                           f"p={v_['wilcoxon_p']:.1e})")
            print(f"| `{nm}` | {m:.4f} | {inc or '--'} |")
            prev = m
        print("\n**Every level earns its place, and z3 earns the most.** It "
              "adds more than z2 and wins on every clip. A separate content "
              "analysis -- eta^2 of the ladder path on spike count and "
              "temporal spread -- puts z3 near zero; that measurement is "
              "blind here, because those summary statistics cannot resolve "
              "exact within-patch PLACEMENT, which is what z3 carries. Where "
              "the question is what the decoder can render, it is measured at "
              "the decoder.\n")

    if spa.is_file():
        S = json.loads(spa.read_text())["summary"]
        print("### Sparse encoder and the blank token\n")
        print("91.7% of tokens are blank. The shipped encoder routes those to "
              "one learned `blank_token` and quantizes only the ~8% carrying a "
              "spike; the `dense` arm declares every token active, so blank "
              "patches -- which are exactly the zero vector -- are projected "
              "and quantized like any other. Identical seed, data order, "
              "optimizer, schedule and losses otherwise.\n")
        if con.is_file():
            C = {a["arm"]: a for a in json.loads(con.read_text())["arms"]}
            print("Measured on CONTENT tokens only, using the true blank mask "
                  "recomputed from the volume. The shipped "
                  "`active_codes_nonblank` counters cannot be compared across "
                  "these two arms: they mean *declared active*, and the dense "
                  "arm declares everything, so its counters cover all 1024 "
                  "tokens while the sparse arm's cover the ~8% with content.\n")
            print("| level-1 parent codebook | sparse | dense |")
            print("|---|---|---|")
            def g(arm, f_):
                try: return C[arm]["levels"][0][f_]
                except Exception: return float("nan")
            for lab, f_, fmt, better in (
                    ("codes used on content (of 32)", "content_codes_used", "{:.0f}", "high"),
                    ("usage entropy, nats", "content_entropy_nats", "{:.3f}", "high"),
                    ("effective alphabet (perplexity)", "content_perplexity", "{:.1f}", "high"),
                    ("codes ALSO used by blank patches", "codes_shared_content_and_blank", "{:.0f}", "low")):
                print(f"| {lab} | " + " | ".join(
                    _mark([g("sparse", f_), g("dense", f_)], fmt, better)) + " |")
            e = lambda a: (C[a]["levels"][0].get("eta2_code_to_content", {})
                           .get("spike_count", float("nan")))
            print("| eta^2 code -> spike count | " + " | ".join(
                _mark([e("sparse"), e("dense")], "{:.3f}", "high")) + " |")
        print(f"| val AUPRC, exact | " + " | ".join(
            _mark([S["sparse"]["best_val_auprc"], S["dense"]["best_val_auprc"]],
                  "{:.4f}", "high")) + " |")
        if con.is_file():
            # Derived, not typed: this sentence quoted "seven" and "half" from
            # one run and then silently disagreed with the table when the
            # underlying JSON was regenerated at a different batch count.
            lost = int(g("sparse", "content_codes_used")
                       - g("dense", "content_codes_used"))
            shd = int(g("dense", "codes_shared_content_and_blank"))
            used = int(g("dense", "content_codes_used"))
            frac = shd / used if used else float("nan")
            print(f"\n**The sparse encoder keeps the whole parent alphabet for "
                  f"content and keeps every code unambiguous about emptiness.** "
                  f"The dense arm spends {lost} parent codes on nothing and "
                  f"leaves {shd} of its remaining {used} ({frac:.0%}) shared "
                  f"between spiking and empty patches, so the code no longer "
                  f"says whether the patch is occupied. Note eta^2 is NOT "
                  f"worse for the dense arm -- the codes it does use still "
                  f"describe their patches; there are simply fewer of them and "
                  f"{frac:.0%} are ambiguous.\n")
        print(f"Both arms ran {S['sparse'].get('epochs_logged','?')} epochs "
              "against the shipped tokenizer's 300, so these are "
              "early-training numbers and valid only as a PAIRED comparison. "
              "At 300 epochs the shipped level-1 codebook reaches perplexity "
              "29.8 of 32 and effective rank 8.29, against 21.2 and 3.67 "
              "here.\n")


HEADLINE = ("scal", "intro", "ap", "calib", "ceiling", "recon", "gen")
APPENDIX = ("notes", "ap_site", "tests", "battery", "battery_own",
            "gen_ladders", "adjacency", "lct_feat")


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out-dir", default=str(DIR))
    ap_.add_argument("--stdout", action="store_true",
                     help="print the headline report instead of writing files")
    ap_.add_argument("--emit-families", action="store_true",
                     help="also write generation_families.json, the four "
                          "generation families at every rung, for the paper "
                          "tables. Additive: the markdown is unchanged.")
    args = ap_.parse_args()

    J, have = {}, []
    for k, lab in MODELS:
        p = DIR / f"diagnose_{k}.json"
        if p.exists():
            J[k] = json.loads(p.read_text()); have.append((k, lab))
    # The map-withheld ablations are not models, so they never enter `have`;
    # they are loaded only so Scalability can show what a lookup arm is worth
    # once its per-assay table is taken away.
    for k in list(J):
        p = DIR / f"diagnose_{k}_NOMAP.json"
        if p.exists():
            J[k + "_NOMAP"] = json.loads(p.read_text())
    rungs = [(r, s_) for r, s_ in RUNGS
             if r in J[have[0][0]]["context_and_space"]["regimes"]]

    def reg(k, r):
        return J[k]["context_and_space"]["regimes"][r]

    real = np.array(J[have[0][0]]["context_and_space"]["adjacency_real"], float)
    labels = J[have[0][0]]["context_and_space"]["adjacency_labels"]

    conv = ("Bold marks the best model for each metric. \u2191 higher is "
            "better, \u2193 lower is better, \u2248REAL means the target is "
            "the real value itself, so both over- and under-shooting are "
            "failures. Rows with no arrow are descriptive.\n")
    budgets = ("Two eval budgets, not interchangeable: the task axis uses 12 "
               "batches (48 clips), seed 20260822; everything else uses 8 "
               "batches (32 clips), seed 20260821. Identical clips for every "
               "model within a budget. Never carry a number between them.\n")

    headline = ("# Interpretable diagnostics\n\n" + budgets + "\n" + conv
                + "\nFull test listings, conditioning ladders and provenance "
                  "are in `diagnostics_appendix.md`.\n\n"
                + _cap(_scalability, J, have)
                + _cap(task_section, HEADLINE)
                + _cap(report, J, have, rungs, reg, real, labels, HEADLINE)
                + _cap(_ablations))
    appendix = ("# Interpretable diagnostics -- appendix\n\n"
                "Supporting detail for `diagnostics.md`: full paired-test "
                "listings, the conditioning ladders behind the collapsed "
                "generation table, and measurement provenance. Same runs, "
                "same numbers.\n\n" + budgets + "\n" + MARK_NOTE + "\n\n"
                + _cap(task_section, APPENDIX)
                + _cap(report, J, have, rungs, reg, real, labels, APPENDIX))

    out = Path(args.out_dir)
    if args.emit_families:
        emit_generation_families(have, rungs, reg, real, out)

    if args.stdout:
        print(headline, end="")
        return 0
    (out / "diagnostics.md").write_text(headline)
    (out / "diagnostics_appendix.md").write_text(appendix)
    print(f"wrote {out/'diagnostics.md'} "
          f"({headline.count(chr(10))} lines)", file=sys.stderr)
    print(f"wrote {out/'diagnostics_appendix.md'} "
          f"({appendix.count(chr(10))} lines)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
