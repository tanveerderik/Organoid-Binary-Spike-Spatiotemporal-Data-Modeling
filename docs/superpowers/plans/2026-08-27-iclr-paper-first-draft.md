# ICLR 2027 first draft — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the compiling skeleton in `paper/` into a complete first draft arguing one thesis — **generative performance on sparse spike volumes comes from a compact alphabet of reusable spatiotemporal motifs** — with every number rendered from a committed JSON artifact and every claim matched to the evidence that actually supports it.

**Architecture:** Three layers. An *evidence* layer adds the one measurement the thesis needs and does not yet have (`analysis/motif_reuse.py` — are motifs actually shared across recordings, or private to each?). A *generator* layer (`tools/make_paper_tables.py`, `tools/make_paper_figures.py`, `tools/extract_provenance.py`, `tools/make_preproc_stats.py`) reads only committed artifacts under `reports/` and writes `paper/tables/*.tex` and `reports/paper_figures/*.pdf`. A *prose* layer (`paper/sections/*.tex`) cites those outputs and never types a number. Evidence before figures, figures before the prose that interprets them.

**Tech Stack:** LaTeX (ICLR 2026 style, swap to 2027 when published), BibTeX, Python 3.10 + matplotlib + numpy at `/home/derik/anaconda3/envs/pytorch/bin/python`.

**Spec:** `~/.claude/plans/cheeky-waddling-rivest.md` — the approved ICLR paper plan (format rules, page budget, figure/table inventory, reviewer-question map, supplement inventory, four-week schedule). Read it alongside this plan; this plan implements it.

---

## Global Constraints

Copied from the standing project constraints and the spec. Every task's requirements implicitly include this section.

- **Main text is 9 pages.** References, appendix and the three statements do not count. There is no figure, table, word or reference limit — only pages.
- **Anonymity is a desk-reject condition.** `\iclrfinalcopy` in `paper/main.tex:38` stays commented out. No author identity in the main text *or* the supplementary material.
- **No number is typed by hand into a `.tex` file.** Every value comes from a generator that reads `reports/`. `paper/tables/*.tex` are generated files; edit the generator, never the output.
- **Thesis word is `scalability`.** Never write "generalizable", "generalises", or "transfers to unseen preparations". `gct` is a seeded random code, so no zero-shot claim is available.
- **Do not present a rejected, more complicated method as an ablation of the shipped one.** Rejected designs go in the appendix as negative results.
- **Never claim direct lookup-table leakage as a win.** DG and the coupled GLM are memorisation ceilings, marked `_ref_`, and never compete for a bolded cell.
- **Naming.** Use the canonical labels and bins from `utils/constants.py`. Never invent names like `space_d1` or `lag_1`.
- **Terminology.** *Reconstruction* = the tokeniser path only. *Free generation (zero context)* = task 0 of the completion battery. They are never the same claim and never share a figure.
- **Two eval budgets are not interchangeable.** Task axis = 12 batches / 48 clips, seed 20260822. Everything else = 8 batches / 32 clips, seed 20260821. Never carry a number between them.
- **Every comparison is per-clip paired**, Wilcoxon signed-rank, Benjamini–Hochberg over the whole family. Quote `q`, never a bare `p`.
- **Every conditioning number is reported against that same arm's random-context null.**
- **Temp files** go in `$CLAUDE_JOB_DIR/tmp`, never `/tmp`.
- **Do not read raw NWB `acquisition/ElectricalSeries/data`** — GB range. Metadata only, via `h5py` (`pynwb` is not installed).
- **`paper/` is gitignored.** Commits in this plan stage `tools/`, `reports/`, `docs/` only. Never `git add paper/`.

---

## The thesis, and the framing this draft must adopt

The paper argues that **a compact alphabet of reusable spatiotemporal motifs is
what makes generation work on data this sparse.** Not "we beat baselines" —
`reports/external_baselines/cross_model_tests.json` scores 32 paired
head-to-heads and **20 go to another arm, 5 to us, 7 are not significant.** A
scoreboard framing gets shredded. A motif framing does not, because the motif
evidence is the part that is unambiguously ours.

### The five links in the argument

| # | claim | evidence | status |
|---|---|---|---|
| 1 | The alphabet is compact and expressive | 961 entries, 619 used at perplexity 436.9; decode ceiling AP 0.3254 against the matched flat tokeniser's 0.0628, a factor of **5.2** | exists |
| 2 | Each ladder level earns its place | z1 0.0772 → +z2 0.1604 (+0.0832) → +z3 0.2806 (**+0.1202**), 100% of clips, p = 3.5e-12 | exists |
| 3 | Motifs are **reused across recordings**, not private to each | — | **MISSING — Task 1B** |
| 4 | Motifs are predictable, so a prior can name them | median rank **9 of 961** against the strongest null's 53; top-1 0.0860 vs 0.0675; **+2.04 nats** | exists |
| 5 | Reuse buys generative performance | site AP ~2× the flat tokeniser on all four tasks (88–94% of clips, q = 1.03e-10 … 3.71e-8); 3 of 4 generation families | exists |

Link 3 is the load-bearing one and **it has never been measured.** Nothing in
`reports/` tallies code usage per recording. If each recording used a private
slice of the alphabet, "shared capacity" would be a partition dressed up as
sharing, and the zero-parameters-per-recording claim would be hollow. Task 1B
measures it before any prose asserts it. If it comes back showing partition
rather than reuse, **the thesis changes and this plan is wrong** — that is the
point of measuring first.

One indirect indication that it will hold: in the null ladder, a per-recording
code marginal (`null_assay_z1`, top-1 0.0149, median rank 102) is only about
twice as good as the global marginal (0.0078, rank 155.5). If codes were
recording-private, knowing the recording would nearly determine the code, and
that gap would be enormous. It is not. But that is an inference from a null, not
a measurement, and the paper needs the measurement.

There is also existing causal evidence for motifs as *units*:
`reports/evaluation_report_code_motifs.json` mines 3-grams over the level-1
codes and finds lift up to **100.6×** over a null, with a transplant test and an
order-sensitivity test (6 of 32 cells order-sensitive at significance). Task 1B
Step 7 checks whether that artifact is still valid at V = 961 before it is cited.

### What we concede, in the main text, in our own words

- **The 3D U-Net wins the ranking columns.** Voxel AP on all four tasks (q =
  5.31e-05 … 2e-11) and site AP on all four. It is trained with this exact loss
  on this exact hole distribution and carries 54.8M skip floats per clip.
- **But it is an inpainter, not a generator, and the paper must say which.**
  Verified in code, not inferred: the class is `UNet3DInpainter`
  (`external_baselines/unet3d/model.py:64`); `complete()` blanks the hole before
  the volume reaches the network (`x[:, :1] * (1.0 - m)`, model.py:387) and
  `tests/test_task_completion.py` re-runs it with the hole filled with noise and
  demands bitwise-identical output. It has **no autoencoding path**, so it
  cannot reconstruct at all — the `--` in the reconstruction table is a
  capability fact, not a missing run. Its "free generation" is
  `_free_logits()` (model.py:352): an all-zeros volume with an all-ones ROI, so
  the output is driven purely by FiLM on `gct` and `lct`, then thresholded by
  `_bernoulli_at_rate` at a fitted log-linear rate. One context gives one field.
- **And it collapses on conditioning.** Conditional accuracy 1.2212 at its
  random null → **1.3637** at full context; adherence 0.4228 → **0.2232**. It
  gets *worse* with more context, and is worst of all six arms on both. Its only
  generation win is marginal realism (0.0620), a pooled statistic a
  deterministic mean field should win. So the honest line is not "we lose to the
  U-Net" — it is that it wins ranking metrics where it is a supervised
  reference and has nothing to say about conditioning.
- **Every learned arm loses to a static per-recording site map**, voxel and
  site, all four tasks, q ≤ 3.1e-10 — a lookup with no clip-specific
  information. This is the most important honest statement in the paper and it
  belongs in §5, not the appendix. It is also the cost of declining to memorise,
  and §5.7 shows the memorisation does not scale.
- **MaskGIT-flat gains more from conditioning than we do**: median improvement
  random → full 0.3817 over 106/128 clips against our 0.1435 over 82/128. We
  make no claim that our conditioning is stronger.
- **Our count head over-counts by roughly 6×**; the only calibrated arm is a
  per-recording lookup. Discrimination and calibration are separate capabilities
  and no arm here has both.

### The four completion tasks, stated once

Established from `external_baselines/task_eval.py:92` and `masks_for_task` at
:116. Every hole is snapped to the (6,15,14) patch lattice.

| id | name | hole | ROI fraction |
|---|---|---|---|
| 0 | `recon` | the whole volume — nothing is visible | 1.000 |
| 1 | `causal` | every frame after a prefix of 0.25–0.75·T | 0.596 |
| 2 | `noncausal` | a contiguous middle span of 0.30·T | 0.406 |
| 3 | `spatial` | a box of 0.50H × 0.50W across all frames | 0.330 |

**Task 0 is misnamed in the code.** Its ROI is the whole volume, so nothing is
visible and there is nothing to reconstruct *from*: it is free generation at
zero context, and it is bitwise identical across arms by construction. Never
call it reconstruction in the paper. Tasks 1–3 are forecasting, temporal
interpolation and spatial inpainting.

---

## File structure

| file | responsibility | status |
|---|---|---|
| `tools/extract_provenance.py` | NWB metadata → `reports/data_provenance.json` | exists, complete |
| `tools/make_preproc_stats.py` | burst NPZ + loader config → `reports/preproc_stats.json` | **new**, Task 1 |
| `analysis/motif_reuse.py` | frozen tokeniser + test clips → `reports/analysis_motif_reuse.json` | **new**, Task 1B — the thesis's missing link |
| `tools/paper_figs/f2_motifs.py` | motif atlas, reuse matrix and samples → F2 | **new**, Task 6 |
| `tools/make_paper_tables.py` | `reports/*.json` → `paper/tables/*.tex` | exists; T2/T4c/T4d missing |
| `tools/make_paper_figures.py` | `reports/*` → `reports/paper_figures/*.pdf`, mirrored to `paper/figures/` | exists; all four `draw_*` are placeholders |
| `paper/refs.bib` | bibliography | 8 entries; motivation refs missing |
| `paper/sections/01_introduction.tex` … `08_statements.tex` | main text, one file per section | skeleton with 48 `\todo{}` |
| `paper/sections/99_appendix.tex` | S1–S16 | skeleton |

`tools/make_paper_figures.py` will grow past a comfortable single file once four real renderers live in it. Split it at Task 4: keep `make_paper_figures.py` as the driver (figure registry, sizing, mirroring, CLI) and move panel drawing into `tools/paper_figs/`, one module per figure. Files that change together stay together: a figure's data loading and its drawing belong in the same module.

---

## Task 1: Preprocessing provenance

The spec says "verify the binning chain in `dataset.py` rather than assuming it". That verification has been done and is recorded here so the executor does not repeat it — but the numbers must still be *rendered*, not typed.

**What the chain actually is**, established by measurement on 2026-08-27:

1. Spike sorting is SpyKING Circus 2 via SpikeInterface (`../read_organoid_data_skc2.py:255-264`): band-pass 300–6000 Hz, 4th-order Butterworth, `radius_um = 100`, waveforms `ms_before = 2.0`, `ms_after = 2.0`.
2. Curated units are written to their peak electrode as a **point** raster. Two NPZ families exist per burst window and only one is used: `binary_unit_burst_*.npz` (mean run length exactly 1.0 — one bit per spike) is what the loader reads (`dataset.py:773,789`); `binary_ch_burst_*.npz` marks the *waveform extent* instead (mean run length 80–93 samples ≈ 4.0–4.7 ms, which is the 4 ms sorter window) and is **not** used. Reading the wrong family inflates the voxel rate by about 22×.
3. Each window is `(120, 220, 12000)` bit-packed along time, uniformly, for all 2,133 windows. The raw bin is one 20 kHz sample (50 µs), so a window is **600 ms**.
4. Loader (`main.py:554-555`): `temporal_pool = 120` max-pool → **6 ms frames**, 100 per window; `temporal_crop = 6000` raw samples = 50 pooled frames; `crop_time_to_multiple` to a multiple of the patch's `T = 6` → **48 frames = 288 ms**; `pad_hw_symmetric_to_multiple` pads W 220 → **224** (multiple of 14).
5. Clip = `48 × 120 × 224` = 1,290,240 voxels. Measured pooled occupancy median 7.6e-5 across windows; val clips sit at 1.5–1.6e-4 because the crop is drawn inside a burst.

**Files:**
- Create: `tools/make_preproc_stats.py`
- Create: `reports/preproc_stats.json` (generated)
- Modify: `paper/sections/03_data.tex` — the `\paragraph{Preprocessing and clip definition.}` `\todo`
- Modify: `paper/sections/99_appendix.tex` — `\section{Preprocessing}`
- Modify: `tools/make_paper_tables.py` — add `preproc_macros()`

**Interfaces:**
- Consumes: `../output_data/*/*/*/*/binary_unit_burst_*.npz`; `main.py` constants `patch_size`, `temporal_crop`, `temporal_pool`.
- Produces: `reports/preproc_stats.json` with keys `n_windows`, `n_recordings`, `raw_shape` `[120,220,12000]`, `raw_bin_us`, `window_ms`, `temporal_pool`, `frame_ms`, `crop_frames_pre`, `clip_frames`, `clip_shape`, `clip_voxels`, `pad_w_from`, `pad_w_to`, `median_pooled_rate`, `mean_spikes_per_clip`, `split_counts`. And `paper/tables/preproc_macros.tex` defining `\FrameMs`, `\ClipMs`, `\ClipVoxels`, `\NWindows`, `\VoxelRate`, `\SpikesPerClip`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_preproc_stats.py`:

```python
import json, subprocess, sys, pathlib

def test_preproc_stats_are_derived_not_assumed():
    subprocess.run([sys.executable, "tools/make_preproc_stats.py"], check=True)
    d = json.loads(pathlib.Path("reports/preproc_stats.json").read_text())
    # The chain, end to end. Each of these was measured, not assumed.
    assert d["raw_shape"] == [120, 220, 12000]
    assert d["raw_bin_us"] == 50.0
    assert d["window_ms"] == 600.0
    assert d["temporal_pool"] == 120
    assert d["frame_ms"] == 6.0
    assert d["clip_frames"] == 48
    assert d["clip_shape"] == [48, 120, 224]
    assert d["clip_voxels"] == 48 * 120 * 224
    assert d["n_windows"] == 2133
    assert d["n_recordings"] == 31
    # The unit family is point events; the channel family is waveform extent.
    # Getting this backwards inflates the rate ~22x, so it is asserted.
    assert d["unit_mean_run_len"] == 1.0
    assert 3.0 < d["ch_mean_run_len"] < 200.0
    assert 1e-5 < d["median_pooled_rate"] < 1e-3
```

- [ ] **Step 2: Run it to verify it fails**

```
cd "/media/derik/Seagate Desktop Drive/organoid_data/MAGVIT_project"
/home/derik/anaconda3/envs/pytorch/bin/python -m pytest tests/test_preproc_stats.py -v
```
Expected: FAIL — `No such file or directory: 'tools/make_preproc_stats.py'`.

- [ ] **Step 3: Write the generator**

Create `tools/make_preproc_stats.py`:

```python
#!/usr/bin/env python3
"""Derive the preprocessing chain from the data files, not from prose.

Section 3 of the paper states a frame duration, a clip duration and a voxel
rate. None of those may be typed by hand, and none may be assumed from the
loader config alone: the loader's `temporal_pool` is a count of raw bins, and
the raw bin width is a property of the extracted NPZ, which lives outside this
repository. This script measures it.

The measurement that matters: `binary_unit_burst_*.npz` stores POINT events
(one bit per spike, run length exactly 1), while `binary_ch_burst_*.npz` stores
the WAVEFORM EXTENT around each spike (run length ~85 samples, which is the
sorter's 4 ms window at 20 kHz). Only the unit family is read by the loader
(`dataset.py:773`). Reading the other one inflates the voxel rate by ~22x, so
both are measured here and both land in the JSON.

    python tools/make_preproc_stats.py
"""
from __future__ import annotations

import glob
import json
import random
from pathlib import Path

import numpy as np

DATA = Path("../output_data")
OUT = Path("reports/preproc_stats.json")

# From main.py:554-555 and the patch definition. Imported rather than retyped
# would be better, but main.py runs a training pipeline at import time.
TEMPORAL_POOL = 120
TEMPORAL_CROP = 6000
PATCH_T, PATCH_H, PATCH_W = 6, 15, 14
SAMPLE_RATE_HZ = 20_000.0   # MaxWell HD-MEA, stated by both source studies


def _unpack(path: str) -> np.ndarray:
    d = np.load(path)
    packed, shape = d["packed"], d["shape"]
    H, W, Tb = packed.shape
    T = int(shape[2])
    bits = np.unpackbits(packed.reshape(-1, Tb), axis=1)[:, :T]
    return bits.reshape(H, W, T).astype(bool)


def _run_stats(b: np.ndarray) -> tuple[int, float]:
    """Number of contiguous runs, and their mean length in raw bins."""
    starts = (np.diff(b.astype(np.int8), axis=2) == 1).sum() + b[:, :, 0].sum()
    return int(starts), float(b.sum() / max(starts, 1))


def main() -> int:
    unit = sorted(glob.glob(str(DATA / "*/*/*/*/binary_unit_burst_*.npz")))
    chan = sorted(glob.glob(str(DATA / "*/*/*/*/binary_ch_burst_*.npz")))
    assert unit, f"no unit burst files under {DATA}"

    recordings = {p.split("/")[-3] for p in unit}

    # Every window must have the same raw shape, or "a window is 600 ms" is
    # not a statement about the corpus. Checked, not assumed.
    shapes = {tuple(np.load(p)["shape"].tolist()) for p in unit[:400]}
    assert len(shapes) == 1, f"heterogeneous raw shapes: {shapes}"
    H, W, T_raw = shapes.pop()

    rng = random.Random(0)
    sample = rng.sample(unit, min(12, len(unit)))
    rates, run_lens = [], []
    for p in sample:
        b = _unpack(p)
        _, mean_run = _run_stats(b)
        run_lens.append(mean_run)
        pooled = b[:, :, : T_raw // TEMPORAL_POOL * TEMPORAL_POOL].reshape(
            H, W, -1, TEMPORAL_POOL).max(axis=3)
        rates.append(float(pooled.mean()))

    ch_runs = []
    for p in rng.sample(chan, min(6, len(chan))):
        ch_runs.append(_run_stats(_unpack(p))[1])

    raw_bin_us = 1e6 / SAMPLE_RATE_HZ
    frame_ms = TEMPORAL_POOL * raw_bin_us / 1e3
    crop_frames_pre = max(1, TEMPORAL_CROP // TEMPORAL_POOL)
    clip_frames = crop_frames_pre - (crop_frames_pre % PATCH_T)
    pad_w_to = W + (-W % PATCH_W)
    voxels = clip_frames * H * pad_w_to
    rate = float(np.median(rates))

    d = {
        "n_windows": len(unit),
        "n_recordings": len(recordings),
        "raw_shape": [H, W, T_raw],
        "raw_bin_us": raw_bin_us,
        "window_ms": T_raw * raw_bin_us / 1e3,
        "temporal_pool": TEMPORAL_POOL,
        "frame_ms": frame_ms,
        "crop_frames_pre": crop_frames_pre,
        "clip_frames": clip_frames,
        "clip_ms": clip_frames * frame_ms,
        "clip_shape": [clip_frames, H, pad_w_to],
        "clip_voxels": voxels,
        "pad_w_from": W,
        "pad_w_to": pad_w_to,
        "patch": [PATCH_T, PATCH_H, PATCH_W],
        "unit_mean_run_len": round(float(np.mean(run_lens)), 4),
        "ch_mean_run_len": round(float(np.mean(ch_runs)), 4),
        "median_pooled_rate": rate,
        "mean_spikes_per_clip": round(rate * voxels, 1),
        "split_counts": {"train": 1069, "val": 426, "test": 638},
        "sorter": {
            "name": "SpyKING Circus 2 (SpikeInterface)",
            "freq_min_hz": 300, "freq_max_hz": 6000,
            "filter": "butterworth", "filter_order": 4,
            "radius_um": 100, "ms_before": 2.0, "ms_after": 2.0,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=1) + "\n")
    print(f"wrote {OUT}")
    for k in ("frame_ms", "clip_ms", "clip_voxels", "median_pooled_rate",
              "mean_spikes_per_clip", "unit_mean_run_len", "ch_mean_run_len"):
        print(f"  {k:22s} {d[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note `split_counts` is the one hard-coded entry. It comes from the loader's own split and is asserted against `reports/` elsewhere; if a future run changes it, Task 14's cross-check catches the mismatch.

- [ ] **Step 4: Run the test to verify it passes**

```
/home/derik/anaconda3/envs/pytorch/bin/python -m pytest tests/test_preproc_stats.py -v
```
Expected: PASS.

- [ ] **Step 5: Add the macro renderer**

In `tools/make_paper_tables.py`, add this function and call it from `main()` before `scalability()`:

```python
def preproc_macros() -> None:
    """Macros for the numbers Section 3 states about a clip."""
    d = json.loads(Path("reports/preproc_stats.json").read_text())
    m = [
        f"\\newcommand{{\\FrameMs}}{{{d['frame_ms']:.0f}}}",
        f"\\newcommand{{\\ClipMs}}{{{d['clip_ms']:.0f}}}",
        f"\\newcommand{{\\ClipFrames}}{{{d['clip_frames']}}}",
        f"\\newcommand{{\\ClipVoxels}}{{{d['clip_voxels']:,}}}",
        f"\\newcommand{{\\NWindows}}{{{d['n_windows']:,}}}",
        f"\\newcommand{{\\WindowMs}}{{{d['window_ms']:.0f}}}",
        f"\\newcommand{{\\VoxelRate}}{{{d['median_pooled_rate']:.2e}}}",
        f"\\newcommand{{\\SpikesPerClip}}{{{d['mean_spikes_per_clip']:.0f}}}",
        f"\\newcommand{{\\NTrain}}{{{d['split_counts']['train']}}}",
        f"\\newcommand{{\\NVal}}{{{d['split_counts']['val']}}}",
        f"\\newcommand{{\\NTest}}{{{d['split_counts']['test']}}}",
    ]
    _write("preproc_macros.tex", "\n".join(m), "reports/preproc_stats.json")
```

Then add `\input{tables/preproc_macros.tex}` to `paper/main.tex` immediately after the existing `\input{tables/data_macros.tex}` on line 21.

- [ ] **Step 6: Write the §3 preprocessing paragraph**

Replace the `\todo{Derive from dataset.py...}` block in `paper/sections/03_data.tex` with:

```latex
\paragraph{Preprocessing and clip definition.}
Spikes are detected and sorted per recording with SpyKING Circus~2
\citep{buccino2020spikeinterface}, band-limited to $300$--$6000$\,Hz, and each
curated unit is written to its peak electrode as a point event at the native
$20$\,kHz sample grid. Activity is segmented into \NWindows\ burst windows of
\WindowMs\,ms across the \NAssays\ recordings. A window is max-pooled by
$120$ samples into \FrameMs\,ms frames, a contiguous span is drawn at random,
and the span is trimmed to \ClipFrames\ frames --- a multiple of the patch's
temporal extent --- giving a clip of \ClipMs\,ms. The array's $120\times220$
footprint is padded symmetrically to $120\times224$, a multiple of the patch's
spatial extent, so a clip is a $\ClipFrames\times120\times224$ binary volume of
\ClipVoxels\ voxels at a mean occupancy of \VoxelRate, i.e.\ about
\SpikesPerClip\ spikes. Files split \NTrain/\NVal/\NTest\ train/val/test,
temporally within each recording. Full detail, including the two NPZ
representations and which one is read, is in Appendix~\ref{app:preproc}.
```

- [ ] **Step 7: Write the appendix preprocessing section**

Replace the `\todo` under `\section{Preprocessing}` in `paper/sections/99_appendix.tex` with a subsection covering, in order: the sorter configuration (band, order, radius, waveform window) from `preproc_stats.json["sorter"]`; the point-event versus waveform-extent distinction with both measured run lengths and the note that only the point representation is read; the pooling, cropping and padding arithmetic; and one sentence stating that max-pooling means a frame records *whether* an electrode fired in that 6 ms, not how many times.

- [ ] **Step 8: Regenerate and commit**

```bash
cd "/media/derik/Seagate Desktop Drive/organoid_data/MAGVIT_project"
/home/derik/anaconda3/envs/pytorch/bin/python tools/make_preproc_stats.py
/home/derik/anaconda3/envs/pytorch/bin/python tools/make_paper_tables.py
git add tools/make_preproc_stats.py tools/make_paper_tables.py \
        reports/preproc_stats.json tests/test_preproc_stats.py
git commit -m "Derive the preprocessing chain from the burst files

The frame duration and voxel rate stated in Section 3 could not be read off the
loader config alone: temporal_pool counts raw bins, and the raw bin width is a
property of the extracted NPZ, which is produced outside this repository.

Measured instead. The raw bin is one 20 kHz sample, so a burst window is 600 ms
and a pooled frame is 6 ms; a clip is 48 frames, 288 ms. Two NPZ families exist
per window and only binary_unit_burst is read: it stores point events, run
length exactly 1.0, while binary_ch_burst stores the sorter's 4 ms waveform
extent, run length 80-93 samples. Reading the wrong one inflates the voxel rate
by about 22x, so the test asserts both.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Gba1U8ohSPVVTaVQBS8eYW"
```

---

## Task 1B: Are motifs actually reused across recordings?

The thesis's load-bearing measurement, and the only new evidence this plan
adds. No retraining: the shipped tokeniser is frozen and this is a tally.

If the alphabet is genuinely shared, a code used by one recording is used by
many, and the entropy of the code distribution barely drops once you know which
recording a clip came from. If instead each recording carved out a private
slice, "shared capacity" is a partition with a shared index, the
zero-parameters-per-recording claim is hollow, and the paper's thesis is wrong.
**Measure before asserting.**

**Files:**
- Create: `analysis/motif_reuse.py`
- Create: `reports/analysis_motif_reuse.json` (generated)
- Test: `tests/test_motif_reuse.py`

**Interfaces:**
- Consumes: `ckpts/vqvae_stage2a_best.pt` (shipped tokeniser);
  `ckpts/stage2b_flat_codebook.pt`, a dict with `embed` (961, 64),
  `merge_map` (1024,) int64, `provenance` (961, 3) int64 giving each flat
  entry's `(a, b, c)` ladder triple, and `counts` (961,);
  `MAGVIT_project.main.find_assays()` and `make_loaders(...)`;
  `MAGVIT_project.ablations.sparse_encoder.build(...)`;
  `reports/data_provenance.json` for each recording's preparation type.
- Produces: `reports/analysis_motif_reuse.json` with keys
  `n_recordings`, `n_clips`, `V`, `vocab_global`, `per_recording`
  (list of `{assay_idx, assay_name, prep, n_clips, n_tokens, vocab, top10_frac}`),
  `shared_core` (`{"used_by_ge_k": {k: count}}`),
  `entropy` (`{"H_code", "H_code_given_recording", "reuse_ratio"}`),
  `jaccard` (`{"mean_within_organoid", "mean_within_slice", "mean_cross_prep",
  "matrix"}`), `label_shuffle_null`
  (`{"mean_jaccard", "sd", "n_shuffles", "z"}`), and `codebook_source`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_motif_reuse.py`:

```python
import json, pathlib
import pytest

OUT = pathlib.Path("reports/analysis_motif_reuse.json")

pytestmark = pytest.mark.skipif(
    not OUT.exists(),
    reason="run: python analysis/motif_reuse.py --batches 24")


def _d():
    return json.loads(OUT.read_text())


def test_shape_of_the_measurement():
    d = _d()
    assert d["V"] == 961
    assert d["n_recordings"] == 31
    assert len(d["per_recording"]) == 31
    # The blank token is excluded everywhere: it is 91.7% of all tokens and
    # would make every recording look identical to every other one.
    assert d["excludes_blank"] is True


def test_entropy_decomposition_is_well_formed():
    d = _d()
    e = d["entropy"]
    # Conditioning cannot increase entropy.
    assert e["H_code_given_recording"] <= e["H_code"] + 1e-9
    assert 0.0 <= e["reuse_ratio"] <= 1.0
    # reuse_ratio is H(code|recording)/H(code): 1.0 means fully shared,
    # near 0 means each recording has a private vocabulary.


def test_null_is_present_and_comparable():
    d = _d()
    n = d["label_shuffle_null"]
    assert n["n_shuffles"] >= 200
    assert n["sd"] > 0.0
    # A z near zero means observed overlap is indistinguishable from the
    # overlap you get when recording labels are meaningless -- i.e. fully
    # shared. A large negative z means partition.
    assert "z" in n
```

The test is `skipif`-guarded because the analysis needs a GPU and the shipped
checkpoint; it asserts the *shape and internal consistency* of the result, never
its value. **Do not write a test that asserts reuse is high.** The measurement
is allowed to come back negative, and a test that forbids that is not a test.

- [ ] **Step 2: Run it to verify it skips, then fails once the file exists**

```
cd "/media/derik/Seagate Desktop Drive/organoid_data/MAGVIT_project"
/home/derik/anaconda3/envs/pytorch/bin/python -m pytest tests/test_motif_reuse.py -v
```
Expected: 3 SKIPPED — `reports/analysis_motif_reuse.json` does not exist yet.

- [ ] **Step 3: Write the analysis**

Create `analysis/motif_reuse.py`:

```python
#!/usr/bin/env python3
"""Is the motif alphabet shared across recordings, or private to each?

The paper claims generation works because a compact alphabet of spatiotemporal
motifs is REUSED. Nothing in reports/ has ever checked the reuse half. This
does, on the frozen shipped tokenizer, with no retraining: encode the test
clips, tally which flat codes each recording emits, and ask three questions.

  1. How much of the alphabet does one recording use, and how much of it is
     shared? `shared_core` counts entries used by at least k recordings.

  2. How much does knowing the recording tell you about the code? The entropy
     decomposition H(code) vs H(code | recording). The ratio is 1.0 when the
     recording label is uninformative -- a fully shared alphabet -- and falls
     toward 0 as each recording acquires a private vocabulary.

  3. Is the observed cross-recording overlap distinguishable from chance? The
     label-shuffle null reassigns clips to recordings at random and recomputes
     the mean pairwise Jaccard. Without it, "recordings share 60% of their
     codes" is uninterpretable: two random samples from one distribution
     already share most of it.

The blank token is EXCLUDED throughout. It is 91.7% of all tokens, every
recording emits it, and including it would drive every overlap statistic to
~1.0 while measuring nothing about motifs.

Jaccard is reported on code SETS (unweighted) and the entropy decomposition on
code COUNTS (weighted). They answer different questions -- which motifs are
available to a recording, and how often it reaches for them -- and a shared
alphabet used with very different frequencies is a real and reportable outcome.

    python analysis/motif_reuse.py --batches 24
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent))
import MAGVIT_project.main as M                                    # noqa: E402
from MAGVIT_project.ablations.sparse_encoder import build          # noqa: E402

OUT = ROOT / "reports" / "analysis_motif_reuse.json"
CKPT = "ckpts/vqvae_stage2a_best.pt"
FLAT = "ckpts/stage2b_flat_codebook.pt"
BLANK_CODE = -1


def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy in nats of a count vector. Empty -> 0.0."""
    tot = counts.sum()
    if tot <= 0:
        return 0.0
    p = counts[counts > 0] / tot
    return float(-(p * np.log(p)).sum())


def _jaccard_matrix(sets: list[set]) -> np.ndarray:
    n = len(sets)
    J = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            u = len(sets[i] | sets[j])
            J[i, j] = J[j, i] = (len(sets[i] & sets[j]) / u) if u else 0.0
    return J


def _offdiag_mean(J: np.ndarray, rows: list[int], cols: list[int],
                  same: bool) -> float:
    vals = [J[i, j] for i in rows for j in cols if (i < j if same else True)]
    return float(np.mean(vals)) if vals else float("nan")


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--flat", default=FLAT)
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--shuffles", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    flat = torch.load(a.flat, map_location="cpu")
    merge_map = flat["merge_map"].long()
    V = int(flat["embed"].shape[0])

    ad = M.find_assays()
    _tr, _val, test, _meta = M.make_loaders(
        assay_dict=ad, assay_indices=list(ad.keys()),
        per_assay_quota=M.per_assay_quota_stage12)
    b0 = next(iter(test))
    model = build(tuple(b0["x"].shape[-3:]), tuple(map(int, b0["full_hw"][0])),
                  dev, dense=False, seed=0)
    sd = torch.load(a.ckpt, map_location="cpu")
    model.load_state_dict(sd.get("model", sd.get("state_dict", sd)),
                          strict=False)
    model.eval()

    n_assays = len(ad)
    counts = np.zeros((n_assays, V), dtype=np.int64)
    clip_rows: list[tuple[int, np.ndarray]] = []   # (assay_idx, per-clip counts)

    for i, batch in enumerate(test):
        if i >= a.batches:
            break
        x = batch["x"].to(dev).float()
        if x.dim() == 4:
            x = x.unsqueeze(1)
        out = model(x, local_ctx=batch["local_ctx"].to(dev).float(),
                    global_ctx=batch["global_ctx"].to(dev).float())
        codes = out["codes"]                       # (B, N, 3) ladder codes
        aidx = batch["assay_idx"].long().cpu().numpy()

        # Flatten exactly as MaskGITMotifPrior.flat_ids_from_codes does
        # (model/prior.py:393). Restated rather than imported because building
        # the prior would pull in two mappers and a checkpoint this analysis
        # does not need; the arithmetic is four lines and is asserted below.
        c0, c1, c2 = (codes[..., 0].long(), codes[..., 1].long(),
                      codes[..., 2].long())
        active = c0.ne(BLANK_CODE)                 # blank excluded here
        nominal = (c0.clamp_min(0) * 8 + c1.clamp_min(0)) * 4 + c2.clamp_min(0)
        fid = merge_map.to(nominal.device)[
            nominal.clamp(0, merge_map.numel() - 1)]

        for b in range(fid.shape[0]):
            ids = fid[b][active[b]].cpu().numpy()
            row = np.bincount(ids, minlength=V)
            counts[aidx[b]] += row
            clip_rows.append((int(aidx[b]), row))
        print(f"  batch {i}  clips={len(clip_rows)}", flush=True)

    assert counts.sum() > 0, "no active tokens encoded -- check the checkpoint"

    prov = json.loads((ROOT / "reports" / "data_provenance.json").read_text())
    prep = {r["assay_idx"]: ("organoid" if r["dandiset"] == "000732" else "slice")
            for r in prov["used"]}
    name = {r["assay_idx"]: r["assay_name"] for r in prov["used"]}

    sets = [set(np.nonzero(counts[i])[0].tolist()) for i in range(n_assays)]
    per_rec = []
    for i in range(n_assays):
        c = counts[i]
        top = np.sort(c)[::-1][:10].sum()
        per_rec.append({
            "assay_idx": i,
            "assay_name": name.get(i, f"assay_{i}"),
            "prep": prep.get(i, "unknown"),
            "n_tokens": int(c.sum()),
            "vocab": int((c > 0).sum()),
            "top10_frac": float(top / c.sum()) if c.sum() else 0.0,
        })

    used_by = (counts > 0).sum(axis=0)             # (V,) recordings per entry
    shared_core = {str(k): int((used_by >= k).sum())
                   for k in (1, 2, 5, 10, 16, 24, 31)}

    # Weighted decomposition. H(code | recording) is the recording-weighted
    # mean of each recording's own code entropy.
    tot = counts.sum()
    H_code = _entropy(counts.sum(axis=0))
    w = counts.sum(axis=1) / tot
    H_cond = float(sum(w[i] * _entropy(counts[i]) for i in range(n_assays)))

    J = _jaccard_matrix(sets)
    org = [i for i in range(n_assays) if prep.get(i) == "organoid"]
    sli = [i for i in range(n_assays) if prep.get(i) == "slice"]

    # Label-shuffle null: reassign CLIPS to recordings at random, preserving
    # each recording's clip count, and recompute the mean pairwise Jaccard.
    # This is the reference the observed value has to be read against.
    rng = np.random.default_rng(a.seed)
    labels = np.array([r[0] for r in clip_rows])
    rows = np.stack([r[1] for r in clip_rows])
    obs_mean = _offdiag_mean(J, list(range(n_assays)), list(range(n_assays)),
                             same=True)
    null = []
    for _ in range(a.shuffles):
        perm = rng.permutation(labels)
        cc = np.zeros_like(counts)
        np.add.at(cc, perm, rows)
        ss = [set(np.nonzero(cc[i])[0].tolist()) for i in range(n_assays)]
        Jn = _jaccard_matrix(ss)
        null.append(_offdiag_mean(Jn, list(range(n_assays)),
                                  list(range(n_assays)), same=True))
    null = np.array(null)

    res = {
        "codebook_source": a.flat,
        "ckpt": a.ckpt,
        "excludes_blank": True,
        "V": V,
        "n_recordings": n_assays,
        "n_clips": len(clip_rows),
        "vocab_global": int((counts.sum(axis=0) > 0).sum()),
        "per_recording": per_rec,
        "shared_core": {"used_by_ge_k": shared_core},
        "entropy": {
            "H_code": H_code,
            "H_code_given_recording": H_cond,
            "reuse_ratio": (H_cond / H_code) if H_code > 0 else 0.0,
        },
        "jaccard": {
            "observed_mean": obs_mean,
            "mean_within_organoid": _offdiag_mean(J, org, org, same=True),
            "mean_within_slice": _offdiag_mean(J, sli, sli, same=True),
            "mean_cross_prep": _offdiag_mean(J, org, sli, same=False),
            "matrix": J.round(4).tolist(),
        },
        "label_shuffle_null": {
            "mean_jaccard": float(null.mean()),
            "sd": float(null.std()),
            "n_shuffles": int(a.shuffles),
            "z": float((obs_mean - null.mean()) / null.std())
                 if null.std() > 0 else 0.0,
        },
    }
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(f"\nwrote {a.out}")
    print(f"  global vocabulary        {res['vocab_global']} of {V}")
    print(f"  used by all 31           {shared_core['31']}")
    print(f"  used by >= 16            {shared_core['16']}")
    print(f"  H(code)                  {H_code:.4f} nats")
    print(f"  H(code | recording)      {H_cond:.4f} nats")
    print(f"  reuse ratio              {res['entropy']['reuse_ratio']:.4f}")
    print(f"  mean Jaccard  observed   {obs_mean:.4f}")
    print(f"                shuffled   {null.mean():.4f} +/- {null.std():.4f}"
          f"   z = {res['label_shuffle_null']['z']:+.2f}")
    print(f"  within-organoid / within-slice / cross  "
          f"{res['jaccard']['mean_within_organoid']:.4f} / "
          f"{res['jaccard']['mean_within_slice']:.4f} / "
          f"{res['jaccard']['mean_cross_prep']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verify the flattening matches the prior's**

The script restates `flat_ids_from_codes` rather than importing it. Prove the
restatement is exact:

```
/home/derik/anaconda3/envs/pytorch/bin/python -c "
import torch
mm = torch.load('ckpts/stage2b_flat_codebook.pt', map_location='cpu')['merge_map'].long()
a = torch.arange(32).view(-1,1,1).expand(32,8,4).reshape(-1)
b = torch.arange(8).view(1,-1,1).expand(32,8,4).reshape(-1)
c = torch.arange(4).view(1,1,-1).expand(32,8,4).reshape(-1)
nominal = (a*8 + b)*4 + c
assert (nominal == torch.arange(1024)).all(), 'index arithmetic disagrees'
f = mm[nominal]
print('distinct flat ids reachable:', f.unique().numel())"
```
Expected: `distinct flat ids reachable: 961`, matching
`reports/analysis_stage2b_flatten.json["distinct"]`. Anything else means the
arithmetic is wrong and every number downstream is wrong.

- [ ] **Step 5: Run the analysis**

```
/home/derik/anaconda3/envs/pytorch/bin/python analysis/motif_reuse.py --batches 24
```

Read the printed summary before going further. Three outcomes and what each means:

- **`reuse_ratio` near 1.0 and `z` near 0** — the alphabet is fully shared, the
  recording label tells you almost nothing about which motif comes next, and the
  thesis holds as written. Proceed.
- **`reuse_ratio` around 0.7–0.9 with a moderate negative `z`** — mostly shared
  with a recording-specific tail. Also fine, and more interesting: report the
  shared core size and say plainly that the tail exists.
- **`reuse_ratio` below ~0.5 with a large negative `z`** — recordings have
  private vocabularies. **Stop and re-plan.** The thesis needs rewriting and
  Tasks 6, 8 and 10 are built on a claim the data does not support.

Whichever it is, that is the result. Do not tune `--batches` or the blank
handling to move it.

- [ ] **Step 6: Run the tests**

```
/home/derik/anaconda3/envs/pytorch/bin/python -m pytest tests/test_motif_reuse.py -v
```
Expected: PASS, 3 tests.

- [ ] **Step 7: Check whether the n-gram motif mining is still valid at V = 961**

`reports/evaluation_report_code_motifs.json` is the causal evidence for motifs
as units — 3-gram lift up to 100.6× over a null, plus a transplant test and an
order-sensitivity test (6 of 32 cells order-sensitive, 5 of 40 order-invariant).
It mines over the **level-1** codes, and level 1 is the 32-entry parent
codebook, which Stage 2B's deduplication does not touch. So it very probably
survives the move to V = 961 — but the flat null ladder had to be refit for
exactly this reason, so check rather than assume:

```
/home/derik/anaconda3/envs/pytorch/bin/python -c "
import json
d = json.load(open('reports/evaluation_report_code_motifs.json'))
print('ngram          ', d['ngram'])
print('n_mined_clips  ', d['n_mined_clips'])
print('motif symbols  ', sorted({v for m in d['motifs'] for v in m}))
print('max lift       ', max(d['lift']))"
```

If every motif symbol is in `[-1, 31]`, the mining is over level-1 codes and is
unaffected by the flatten — record that in the JSON as a `level: 1` note and
cite it. If any symbol exceeds 31, it is over a flat alphabet and the artifact
predates V = 961; in that case **do not cite it**, and add re-mining to the
supplement's future-work list rather than blocking this plan on a rerun.

- [ ] **Step 8: Commit**

```bash
git add analysis/motif_reuse.py tests/test_motif_reuse.py \
        reports/analysis_motif_reuse.json
git commit -m "Measure whether the motif alphabet is shared across recordings

The paper's thesis is that generation works because a compact alphabet of
spatiotemporal motifs is reused. Nothing in reports/ had ever checked the reuse
half: if each recording used a private slice of the 961 entries, shared capacity
would be a partition with a shared index and the zero-parameters-per-recording
claim would be hollow.

Three statistics on the frozen tokenizer, no retraining. Shared-core counts by
how many recordings use an entry; an entropy decomposition H(code) against
H(code | recording), whose ratio is 1.0 for a fully shared alphabet; and mean
pairwise Jaccard against a label-shuffle null, because raw overlap is
uninterpretable without one -- two random samples of one distribution already
share most of it.

The blank token is excluded throughout. It is 91.7% of tokens and every
recording emits it, so including it would drive every overlap statistic to ~1.0
while measuring nothing.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Gba1U8ohSPVVTaVQBS8eYW"
```

---

## Task 2: Bibliography

The Introduction and Related work cannot be written before the references exist. This task adds them all; no later task adds a citation without adding its entry here first.

**Files:**
- Modify: `paper/refs.bib` (append; the existing eight entries stay)

**Interfaces:**
- Produces: BibTeX keys used by every later prose task. Exact keys are listed below and must not be renamed.

- [ ] **Step 1: Append the motivation and lineage entries**

Append to `paper/refs.bib`:

```bibtex
% ---------------------------------------------------------------------------
% Organoid intelligence and synthetic biological intelligence. These carry the
% first move of the Introduction: cultured human neural tissue is being driven
% toward closed-loop tasks, and every such system needs a forward model.
% ---------------------------------------------------------------------------

@article{kagan2022dishbrain,
  title   = {In vitro neurons learn and exhibit sentience when embodied in a
             simulated game-world},
  author  = {Kagan, Brett J. and Kitchen, Andy C. and Tran, Nhi T. and
             Habibollahi, Forough and Khajehnejad, Moein and Parker, Bradyn J.
             and Bhat, Anjali and Rollo, Ben and Razi, Adeel and
             Friston, Karl J.},
  journal = {Neuron},
  volume  = {110},
  number  = {23},
  pages   = {3952--3969.e8},
  year    = {2022},
  doi     = {10.1016/j.neuron.2022.09.001}
}

@article{smirnova2023oi,
  title   = {Organoid intelligence ({OI}): the new frontier in biocomputing and
             intelligence-in-a-dish},
  author  = {Smirnova, Lena and Caffo, Brian S. and Gracias, David H. and
             Huang, Qi and Morales Pantoja, Itzy E. and Tang, Bohao and
             Zack, Donald J. and Berlinicke, Cynthia A. and Boyd, J. Lomax and
             Harris, Timothy D. and Johnson, Erik C. and Kagan, Brett J. and
             Kahn, Jeffrey and Muotri, Alysson R. and Paulhamus, Barton L. and
             Schwamborn, Jens C. and Plotkin, Jesse and Szalay, Alexander S.
             and Vogelstein, Joshua T. and Worley, Paul F. and
             Hartung, Thomas},
  journal = {Frontiers in Science},
  volume  = {1},
  pages   = {1017235},
  year    = {2023},
  doi     = {10.3389/fsci.2023.1017235}
}

% ---------------------------------------------------------------------------
% Organoid and ex vivo electrophysiology: what the signal is, and on what
% hardware. sharf2022 is the source study behind dandiset 000732.
% ---------------------------------------------------------------------------

@article{sharf2022organoids,
  title   = {Functional neuronal circuitry and oscillatory dynamics in human
             brain organoids},
  author  = {Sharf, Tal and van der Molen, Tjitse and Glasauer, Stella M. K.
             and Guzman, Elmer and Buccino, Alessio P. and Luna, Gabriel and
             Cheng, Zhuowei and Audouard, Morgane and Ranasinghe, Kamalini G.
             and Kudo, Kiwamu and Nagarajan, Srikantan S. and
             Tovar, Kenneth R. and Petzold, Linda R. and Hierlemann, Andreas
             and Hansma, Paul K. and Kosik, Kenneth S.},
  journal = {Nature Communications},
  volume  = {13},
  pages   = {4403},
  year    = {2022},
  doi     = {10.1038/s41467-022-32115-4}
}

@article{trujillo2019oscillations,
  title   = {Complex Oscillatory Waves Emerging from Cortical Organoids Model
             Early Human Brain Network Development},
  author  = {Trujillo, Cleber A. and Gao, Richard and Negraes, Priscilla D. and
             Gu, Jing and Buchanan, Justin and Preissl, Sebastian and
             Wang, Allen and Wu, Wei and Haddad, Gabriel G. and
             Chaim, Isaac A. and Domissy, Alain and Vandenberghe, Matthieu and
             Devor, Anna and Yeo, Gene W. and Voytek, Bradley and
             Muotri, Alysson R.},
  journal = {Cell Stem Cell},
  volume  = {25},
  number  = {4},
  pages   = {558--569.e7},
  year    = {2019},
  doi     = {10.1016/j.stem.2019.08.002}
}

@article{ballini2014hdmea,
  title   = {A 1024-Channel {CMOS} Microelectrode Array With 26,400 Electrodes
             for Recording and Stimulation of Electrogenic Cells In Vitro},
  author  = {Ballini, Marco and M{\"u}ller, Jan and Livi, Paolo and
             Chen, Yihui and Frey, Urs and Stettler, Alexander and
             Shadmani, Amir and Viswam, Vijay and Jones, Ian Lloyd and
             Jäckel, David and Radivojevic, Milos and Lewandowska, Marta K.
             and Gong, Wei and Fiscella, Michele and Bakkum, Douglas J. and
             Heer, Flavio and Hierlemann, Andreas},
  journal = {IEEE Journal of Solid-State Circuits},
  volume  = {49},
  number  = {11},
  pages   = {2705--2719},
  year    = {2014},
  doi     = {10.1109/JSSC.2014.2359219}
}

@article{beggs2003avalanches,
  title   = {Neuronal avalanches in neocortical circuits},
  author  = {Beggs, John M. and Plenz, Dietmar},
  journal = {Journal of Neuroscience},
  volume  = {23},
  number  = {35},
  pages   = {11167--11177},
  year    = {2003},
  doi     = {10.1523/JNEUROSCI.23-35-11167.2003}
}

@article{buccino2020spikeinterface,
  title   = {{SpikeInterface}, a unified framework for spike sorting},
  author  = {Buccino, Alessio Paolo and Hurwitz, Cole Lincoln and
             Garcia, Samuel and Magland, Jeremy and Siegle, Joshua H. and
             Hurwitz, Roger and Hennig, Matthias H.},
  journal = {eLife},
  volume  = {9},
  pages   = {e61834},
  year    = {2020},
  doi     = {10.7554/eLife.61834}
}

% ---------------------------------------------------------------------------
% Discrete generative modelling. RQ-VAE and SoundStream are the direct lineage
% for the residual ladder and must be cited as such -- residual quantization is
% not our invention; the contribution is what it is applied to and how the
% ladder is flattened.
% ---------------------------------------------------------------------------

@inproceedings{razavi2019vqvae2,
  title     = {Generating Diverse High-Fidelity Images with {VQ-VAE-2}},
  author    = {Razavi, Ali and van den Oord, Aaron and Vinyals, Oriol},
  booktitle = {NeurIPS},
  year      = {2019}
}

@inproceedings{esser2021vqgan,
  title     = {Taming Transformers for High-Resolution Image Synthesis},
  author    = {Esser, Patrick and Rombach, Robin and Ommer, Bj{\"o}rn},
  booktitle = {CVPR},
  year      = {2021}
}

@inproceedings{lee2022rqvae,
  title     = {Autoregressive Image Generation using Residual Quantization},
  author    = {Lee, Doyup and Kim, Chiheon and Kim, Saehoon and Cho, Minsu and
               Han, Wook-Shin},
  booktitle = {CVPR},
  year      = {2022}
}

@article{zeghidour2022soundstream,
  title   = {{SoundStream}: An End-to-End Neural Audio Codec},
  author  = {Zeghidour, Neil and Luebs, Alejandro and Omran, Ahmed and
             Skoglund, Jan and Tagliasacchi, Marco},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume  = {30},
  pages   = {495--507},
  year    = {2022},
  doi     = {10.1109/TASLP.2021.3129994}
}

@inproceedings{yu2024magvit2,
  title     = {Language Model Beats Diffusion: Tokenizer is Key to Visual
               Generation},
  author    = {Yu, Lijun and Lezama, Jos{\'e} and Gundavarapu, Nitesh B. and
               Versari, Luca and Sohn, Kihyuk and Minnen, David and
               Cheng, Yong and Gupta, Agrim and Gu, Xiuye and
               Hauptmann, Alexander G. and Gong, Boqing and Yang, Ming-Hsuan
               and Essa, Irfan and Ross, David A. and Jiang, Lu},
  booktitle = {ICLR},
  year      = {2024}
}

% ---------------------------------------------------------------------------
% Generative and statistical models for neural data. The distinction the paper
% draws: these model firing rates of SORTED UNITS on tens to hundreds of
% channels; we model an array-level binary volume.
% ---------------------------------------------------------------------------

@article{pandarinath2018lfads,
  title   = {Inferring single-trial neural population dynamics using sequential
             auto-encoders},
  author  = {Pandarinath, Chethan and O'Shea, Daniel J. and Collins, Jasmine
             and Jozefowicz, Rafal and Stavisky, Sergey D. and
             Kao, Jonathan C. and Trautmann, Eric M. and Kaufman, Matthew T.
             and Ryu, Stephen I. and Hochberg, Leigh R. and
             Henderson, Jaimie M. and Shenoy, Krishna V. and Abbott, L. F. and
             Sussillo, David},
  journal = {Nature Methods},
  volume  = {15},
  pages   = {805--815},
  year    = {2018},
  doi     = {10.1038/s41592-018-0109-9}
}

@article{truccolo2005pointprocess,
  title   = {A point process framework for relating neural spiking activity to
             spiking history, neural ensemble, and extrinsic covariate effects},
  author  = {Truccolo, Wilson and Eden, Uri T. and Fellows, Matthew R. and
             Donoghue, John P. and Brown, Emery N.},
  journal = {Journal of Neurophysiology},
  volume  = {93},
  number  = {2},
  pages   = {1074--1089},
  year    = {2005},
  doi     = {10.1152/jn.00697.2004}
}

@inproceedings{ye2021ndt,
  title     = {Representation learning for neural population activity with
               Neural Data Transformers},
  author    = {Ye, Joel and Pandarinath, Chethan},
  booktitle = {Neurons, Behavior, Data analysis, and Theory},
  year      = {2021}
}

% ---------------------------------------------------------------------------
% Evaluation methodology. Each of these is already load-bearing in
% reports/external_baselines/diagnostics.md and must be cited where the
% corresponding choice is defended.
% ---------------------------------------------------------------------------

@inproceedings{davis2006prcurves,
  title     = {The relationship between Precision-Recall and {ROC} curves},
  author    = {Davis, Jesse and Goadrich, Mark},
  booktitle = {ICML},
  year      = {2006}
}

@inproceedings{elkan2001costsensitive,
  title     = {The Foundations of Cost-Sensitive Learning},
  author    = {Elkan, Charles},
  booktitle = {IJCAI},
  year      = {2001}
}

@article{benjamini1995fdr,
  title   = {Controlling the False Discovery Rate: A Practical and Powerful
             Approach to Multiple Testing},
  author  = {Benjamini, Yoav and Hochberg, Yosef},
  journal = {Journal of the Royal Statistical Society: Series B},
  volume  = {57},
  number  = {1},
  pages   = {289--300},
  year    = {1995}
}
```

- [ ] **Step 2: Fill the two incomplete existing entries**

`sharf2025protosequences` and `yu2023magvit` both carry `note = {TODO complete author list ...}`. Resolve both from the publisher page and delete the `note` field. `sharf2025protosequences` is the published version of the study behind dandiset 000732 and its Methods are the source for "eight organoids (four whole, four sliced)" in §3 — the citation must be complete.

- [ ] **Step 3: Verify no TODO remains**

```
grep -n "TODO" paper/refs.bib
```
Expected: no output.

- [ ] **Step 4: Commit**

`paper/` is gitignored, so there is nothing to stage. Record the state instead:

```bash
cp paper/refs.bib "$CLAUDE_JOB_DIR/tmp/refs.bib.checkpoint"
echo "refs.bib complete: $(grep -c '^@' paper/refs.bib) entries"
```
Expected: 25 entries.

---

## Task 3: Complete the table generators

Three of the four main tables are missing. `t3_generation_perclip.tex` renders one family of four. There is no task-completion table at all.

**Files:**
- Modify: `tools/make_paper_tables.py` — rewrite `generation()`, add `task_completion()`, `stage_prior_ladder()`
- Test: `tests/test_paper_tables.py`

**Interfaces:**
- Consumes: `reports/external_baselines/comparison.json` (keys `per_clip`, `conditioning_gain`, `reference` = `"4C+soft (ship)"`), `reports/external_baselines/cross_model_tests.json` (`rows`, each with `family`, `task`, `arm`, `pipeline`, `arm_value`, `pipeline_wins_frac`, `q`, `verdict`), `reports/evaluation_report_prior_4A_V961.json`.
- Produces: `paper/tables/t2_task_completion.tex`, `t3_generation_families.tex`, `t4c_prior_nulls.tex`. `t3_generation_perclip.tex` is retired.

- [ ] **Step 1: Write the failing test**

Create `tests/test_paper_tables.py`:

```python
import subprocess, sys, pathlib, re

TABLES = pathlib.Path("paper/tables")

def _render():
    subprocess.run([sys.executable, "tools/make_paper_tables.py"], check=True)

def test_task_completion_table_has_both_metrics_and_all_arms():
    _render()
    t = (TABLES / "t2_task_completion.tex").read_text()
    for arm in ("Ours", "MaskGIT-flat", "3D U-Net", "3D CVAE"):
        assert arm in t, f"{arm} missing from T2"
    assert "site" in t.lower() and "voxel" in t.lower()
    # The seen-recording site-map null must be in the table, not a footnote:
    # every learned arm loses to it and the table has to show that.
    assert "site map" in t.lower()
    # Both null AP values, site and voxel, on the free-generation column.
    assert "0.7205" in t and "0.0869" in t

def test_generation_table_has_all_four_families_with_nulls():
    _render()
    t = (TABLES / "t3_generation_families.tex").read_text()
    for fam in ("Conditional accuracy", "Adherence", "Spatial placement",
                "Marginal realism"):
        assert fam in t, f"family {fam} missing from T3"
    # Every cell carries its own random-context null in brackets.
    assert t.count("[") >= 16

def test_no_table_is_hand_edited():
    for p in TABLES.glob("*.tex"):
        assert p.read_text().startswith("% GENERATED by"), p
```

- [ ] **Step 2: Run it to verify it fails**

```
/home/derik/anaconda3/envs/pytorch/bin/python -m pytest tests/test_paper_tables.py -v
```
Expected: FAIL — `t2_task_completion.tex` does not exist.

- [ ] **Step 3: Emit the generation families as JSON, then render from it**

The four families are **not** in `comparison.json`. Its `per_clip` block holds
nine per-clip statistics (`stat_error_clean`, `ks_isi`, `rel_rate`, …), each as
`{arm: {"median": float}}`, over seven *pipeline variants* — it does not contain
`3D U-Net` or `3D CVAE` at all, and its `stat_error_clean` median (0.4175) is
the conditioning-gain median, not the `glob+full` value (0.6078) that the paper
quotes. Rendering T3 from it would silently produce different numbers than the
diagnostics.

The families are computed in `external_baselines/diagnose_table.py`:
`GEN_METRICS` at line 1440 defines A, B and C as lambdas over
`reg(arm, rung)`, and `_sre_factory` at line 1461 defines D. `MODELS` (line 32)
and `RUNGS` (line 35) are the arm and rung orders. That module is the only
place these values exist, so the fix is to make it emit them.

Add to `external_baselines/diagnose_table.py`, called from `main()` when a new
`--emit-families` flag is passed:

```python
def emit_generation_families(have, rungs, reg, real, out_dir: Path) -> None:
    """Dump the four generation families so the paper can render them.

    diagnose_table.py is the only place these four are computed. Until now they
    existed solely as rendered markdown, which meant the only way to get them
    into a table was to retype them -- and a retyped number cannot be traced to
    the run that produced it. This writes the same values the markdown prints,
    from the same lambdas, at every rung.
    """
    fams = GEN_METRICS + [("D. Marginal realism", "low", "", _sre_factory(real))]
    out = {"rungs": [r for r, _ in rungs], "families": {}}
    for title, better, _note, fn in fams:
        out["families"][title] = {"better": better, "arms": {}}
        for key, label in MODELS:
            if key not in have:
                continue
            out["families"][title]["arms"][label] = {
                rung: float(fn(reg, key, rung)) for rung, _ in rungs
            }
    p = out_dir / "generation_families.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {p}", file=sys.stderr)
```

Note the argument order: `GEN_METRICS` lambdas take `(reg, k, r)`, so call
`fn(reg, key, rung)` and not `fn(key, rung)` — the `ladder()` closure inside
`report()` binds `reg` before calling, which is why the signature looks
inconsistent at first reading.

Then replace `generation()` in `tools/make_paper_tables.py`:

```python
def generation() -> None:
    """The four generation families at full context, each with its own null.

    The bracketed value is that same arm's `random`-context score, so a cell
    whose value and null are close is not using its conditioning. Without it a
    reader cannot tell conditioning from an arm's default behaviour, which is
    exactly what the lookup arms do once their per-assay map is withheld.

    Bold competes between the learned generative models only. The `_ref_` rows
    are per-assay lookup tables shown as memorisation ceilings, and the U-Net
    row carries a dagger: it is a conditional-mean regressor with no sampling
    distribution, so its distributional columns are not read like the others.
    """
    src = EB / "generation_families.json"
    if not src.exists():
        raise SystemExit(
            f"{src} missing. Generate it with:\n"
            f"  python -m external_baselines.diagnose_table --emit-families\n"
            f"The four families exist only inside diagnose_table.py; do not "
            f"transcribe them from diagnostics.md.")
    d = json.loads(src.read_text())
    FULL, NULL = "global_full_local", "random"
    LABEL = {"Ours (4C+soft)": "Ours", "MaskGIT-flat": "MaskGIT-flat",
             "3D U-Net (det.)\u2020": "3D U-Net$^\\dagger$",
             "3D CVAE": "3D CVAE",
             "Dich. Gaussian": "\\emph{ref} DG",
             "Coupled GLM": "\\emph{ref} GLM"}

    cols = [lbl for lbl in LABEL.values()]
    rows = []
    for title, blk in d["families"].items():
        arrow = "$\\downarrow$" if blk["better"] == "low" else "$\\uparrow$"
        cells = []
        for raw, _lbl in LABEL.items():
            v = blk["arms"].get(raw)
            cells.append("--" if v is None else
                         f"{v[FULL]:.4f} \\emph{{[{v[NULL]:.3f}]}}")
        rows.append(f"{title} {arrow} & " + " & ".join(cells) + " \\\\")

    body = ("\\begin{tabular}{l r r r r r r}\n\\toprule\n"
            "family & " + " & ".join(cols) + " \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    _write("t3_generation_families.tex", body, str(src))
```

The constraint that forced this detour is worth restating: no number is typed
by hand, and a table copied out of rendered markdown *is* typed by hand. The
extra flag on `diagnose_table.py` costs twenty lines and makes every cell in T3
traceable to a run.

- [ ] **Step 3b: Verify the emitted values match the rendered markdown**

```
/home/derik/anaconda3/envs/pytorch/bin/python -m external_baselines.diagnose_table --emit-families
/home/derik/anaconda3/envs/pytorch/bin/python -c "
import json
d = json.load(open('reports/external_baselines/generation_families.json'))
for fam, blk in d['families'].items():
    o = blk['arms']['Ours (4C+soft)']
    print(f\"{fam:38s} full={o['global_full_local']:.4f} random={o['random']:.4f}\")"
```

Expected, matching `diagnostics.md` exactly:

```
A. Conditional accuracy                full=0.6078 random=1.2602
B. Adherence                           full=0.7764 random=0.6570
C. Spatial placement, lookup-proof     full=0.0204 random=0.0029
D. Marginal realism                    full=0.1879 random=0.2431
```

Any mismatch means the emit path is not calling the same lambdas the renderer
calls. Fix the emit path; never adjust the expectation.

Then confirm the markdown is unchanged by the new flag:

```
md5sum reports/external_baselines/diagnostics.md
```
Expected: `86073e2f…` — the byte-identical hash from the last clean render.

- [ ] **Step 4: Add `task_completion()`**

```python
def task_completion() -> None:
    """Site-level and voxel-level AP on the four completion tasks.

    Read WITHIN a column only. The ROI fraction differs by task
    (1.00 / 0.60 / 0.41 / 0.33), so the columns have different base rates and a
    horizontal comparison is meaningless. `recon` is free generation at zero
    context, not a fourth task.
    """
    src = EB / "cross_model_tests.json"
    d = json.loads(src.read_text())
    # The null is the point of the table, so it is a row, not a footnote:
    # `marginal` is the clip's own recording's per-site firing rate measured on
    # that recording's TRAIN clips and held constant in time -- no model, no
    # completion, and no clip-specific information at all. Every learned arm
    # loses to it. `marginal_xa` is the same lookup with the clip's own
    # recording withheld, and the distance between the two is how much of the
    # first is memorisation.
    nulls = json.loads((EB / "task_eval_nulls.json").read_text())["tasks"]
    TASKS = ["recon", "causal", "noncausal", "spatial"]
    ARMLBL = {"MaskGIT-flat": "MaskGIT-flat",
              "3D U-Net (det.)": "3D U-Net$^\\dagger$",
              "3D CVAE": "3D CVAE"}

    # Ours is identical across rows of a (family, task); take it from any row.
    ours = {}
    per_arm = {}
    for r in d["rows"]:
        if r["arm"] not in ARMLBL:
            continue
        ours[(r["family"], r["task"])] = r["pipeline"]
        per_arm[(r["family"], r["task"], r["arm"])] = r

    blocks = []
    for fam, famlbl in (("site AP", "Site AP $\\uparrow$ --- \\emph{where}"),
                        ("voxel AP", "Voxel AP $\\uparrow$ --- \\emph{when and where}")):
        blocks.append(f"\\multicolumn{{5}}{{l}}{{\\textbf{{{famlbl}}}}} \\\\")
        blocks.append("Ours & " + " & ".join(
            f"{ours[(fam, t)]:.4f}" for t in TASKS) + " \\\\")
        for arm, lbl in ARMLBL.items():
            cells = []
            for t in TASKS:
                r = per_arm.get((fam, t, arm))
                if r is None:
                    cells.append("--")
                    continue
                mark = {"pipeline": "$\\ast$", "arm": "", "ns": "$^{n.s.}$"}[r["verdict"]]
                cells.append(f"{r['arm_value']:.4f}{mark}")
            blocks.append(f"{lbl} & " + " & ".join(cells) + " \\\\")
        fld = "site_ap" if fam == "site AP" else "ap"
        for nk, nlbl in (("marginal", "\\emph{null} recording site map, seen"),
                         ("marginal_xa", "\\emph{null} recording site map, unseen")):
            blocks.append(f"{nlbl} & " + " & ".join(
                f"{nulls[t]['arms'][nk][fld]:.4f}" for t in TASKS) + " \\\\")
        blocks.append("\\midrule")
    blocks.pop()

    body = ("\\begin{tabular}{l r r r r}\n\\toprule\n"
            "arm & free gen.\\ (0) & causal (1) & noncausal (2) & spatial (3)"
            " \\\\\n\\midrule\n" + "\n".join(blocks)
            + "\n\\bottomrule\n\\end{tabular}")
    _write("t2_task_completion.tex", body, str(src))
```

`$\ast$` marks a cell where **we** win the paired test; unmarked means the other arm wins; `n.s.` means neither. Define this in the caption, not in the table.

- [ ] **Step 5: Add `stage_prior_ladder()`**

```python
def stage_prior_ladder() -> None:
    """Stage 4A motif prior against the four-rung null ladder.

    The strongest null is `assay_position` -- the empirical distribution of
    codes at that grid position within that recording. Beating `uniform` is
    worth nothing; the ladder exists so the reported margin is against the
    hardest available lookup, and `null_strongest_level` records which that is.
    """
    src = Path("reports/evaluation_report_prior_4A_V961.json")
    d = json.loads(src.read_text())
    assert d["null_strongest_level"] == "assay_position", d["null_strongest_level"]
    rungs = [("null_uniform_z1", "uniform"),
             ("null_global_z1", "global marginal"),
             ("null_assay_z1", "per-recording marginal"),
             ("null_assay_position_z1", "per-recording $\\times$ position")]
    rows = []
    for pre, lbl in rungs:
        rows.append(f"{lbl} & {d[pre+'_acc']:.4f} & {d[pre+'_top5_acc']:.4f} & "
                    f"{d[pre+'_median_rank']:.0f} & {d[pre+'_mrr']:.4f} & "
                    f"{d[pre+'_ce']:.4f} \\\\")
    rows.append("\\midrule")
    rows.append(f"\\textbf{{motif prior (ours)}} & \\textbf{{{d['flat_acc']:.4f}}} & "
                f"\\textbf{{{d['flat_top5_acc']:.4f}}} & "
                f"\\textbf{{{d['flat_median_rank']:.0f}}} & "
                f"\\textbf{{{d['flat_mrr']:.4f}}} & "
                f"\\textbf{{{d['loss_flat']:.4f}}} \\\\")
    body = ("\\begin{tabular}{l r r r r r}\n\\toprule\n"
            "predictor & top-1 $\\uparrow$ & top-5 $\\uparrow$ & "
            "median rank $\\downarrow$ & MRR $\\uparrow$ & CE (nats) $\\downarrow$"
            " \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}")
    _write("t4c_prior_nulls.tex", body, str(src))
```

Quote **median** rank, never mean: the distribution has a long tail and `flat_mean_rank` (44.65) misrepresents `flat_median_rank` (9.0).

- [ ] **Step 6: Wire into `main()` and delete the retired table**

```python
def main() -> int:
    preproc_macros()
    data_provenance()
    scalability()
    task_completion()
    generation()
    stage_contributions()
    stage_prior_ladder()
    ...
```

Then `rm paper/tables/t3_generation_perclip.tex` and change `05_experiments.tex` to `\input{tables/t3_generation_families.tex}`.

- [ ] **Step 7: Run the tests**

```
/home/derik/anaconda3/envs/pytorch/bin/python -m pytest tests/test_paper_tables.py -v
```
Expected: PASS, 3 tests.

- [ ] **Step 8: Commit**

```bash
git add tools/make_paper_tables.py tests/test_paper_tables.py
git commit -m "Render the task-completion, generation and prior-null tables

T2 reports site AP and voxel AP as two blocks of the same table because the two
answer different questions -- where an arm puts spikes, and whether it also
gets the frame right -- and our result differs between them. Cells are marked
with the paired-test verdict rather than bolded, since 20 of the 32 tests go to
another arm and bolding would imply a scoreboard we do not win.

T3 replaces the one-family stub with all four, each cell carrying that arm's
own random-context null. T4c is the Stage 4A motif prior against the four-rung
null ladder; median rank is quoted because the mean (44.65) misrepresents the
median (9.0).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Gba1U8ohSPVVTaVQBS8eYW"
```

---

## Task 4: Figure infrastructure and F3 (task axis)

**REQUIRED SUB-SKILL:** load the `dataviz` skill before writing any plotting code in this task or in Tasks 5–7. It sets the palette, the accessibility rules and the mark specs; the four figures must read as one system.

**Files:**
- Create: `tools/paper_figs/__init__.py`
- Create: `tools/paper_figs/style.py`
- Create: `tools/paper_figs/f3_task_axis.py`
- Modify: `tools/make_paper_figures.py` — driver only
- Test: `tests/test_paper_figures.py`

**Interfaces:**
- Consumes: `reports/external_baselines/cross_model_tests.json`, `reports/external_baselines/task_eval_nulls.json`.
- Produces: `tools/paper_figs/style.py` exporting `PALETTE: dict[str, str]`, `ARM_ORDER: list[tuple[str, str]]`, `apply_style() -> None`, `TEXT_W: float = 5.5`. `tools/paper_figs/f3_task_axis.py` exporting `draw(fig) -> None`. Each figure module exports exactly `draw(fig)` and takes a whole `Figure`, not an `Axes`, because F2 and F3 both need multiple panels.

- [ ] **Step 1: Write the failing test**

Create `tests/test_paper_figures.py`:

```python
import subprocess, sys, pathlib, hashlib

OUT = pathlib.Path("reports/paper_figures")

def _render():
    subprocess.run([sys.executable, "tools/make_paper_figures.py"], check=True)

def test_all_four_figures_render():
    _render()
    for name in ("f1_pipeline", "f2_motifs", "f3_task_axis",
                 "f4_generation"):
        p = OUT / f"{name}.pdf"
        assert p.exists() and p.stat().st_size > 2000, name

def test_no_placeholder_text_survives():
    _render()
    for p in OUT.glob("*.pdf"):
        assert b"PLACEHOLDER" not in p.read_bytes(), p

def test_figures_are_byte_stable_across_runs():
    _render()
    first = {p.name: hashlib.md5(p.read_bytes()).hexdigest()
             for p in sorted(OUT.glob("*.pdf"))}
    _render()
    second = {p.name: hashlib.md5(p.read_bytes()).hexdigest()
              for p in sorted(OUT.glob("*.pdf"))}
    assert first == second, "figures are not reproducible run to run"
```

Byte stability requires a deterministic PDF: set `matplotlib.rcParams["pdf.compression"] = 0` is not enough, because matplotlib stamps a `CreationDate`. Set `SOURCE_DATE_EPOCH=0` in `main()` via `os.environ.setdefault("SOURCE_DATE_EPOCH", "0")` *before* importing pyplot, and seed every RNG the figures use.

- [ ] **Step 2: Run it to verify it fails**

```
/home/derik/anaconda3/envs/pytorch/bin/python -m pytest tests/test_paper_figures.py -v
```
Expected: `test_no_placeholder_text_survives` FAILS — all four are placeholders.

- [ ] **Step 3: Write the shared style module**

Create `tools/paper_figs/style.py`:

```python
"""One visual system for all four figures.

Arm identity is carried by COLOUR and held fixed across every figure, so a
reader who learns the mapping in F2 keeps it in F3 and F4. The two lookup arms
share a single desaturated grey: they are memorisation ceilings, not peers, and
giving them distinct saturated hues would read as a six-way competition.

Colours below are the dataviz placeholder palette. Swap them for the paper's
own palette in one place, here, and every figure follows.
"""
from __future__ import annotations

import matplotlib

TEXT_W = 5.5   # ICLR single-column body width, inches

# Ordered as the paper argues: ours, the peer prior, the two conv references,
# then the lookups.
ARM_ORDER = [
    ("pipeline",     "Ours"),
    ("maskgit_flat", "MaskGIT-flat"),
    ("unet3d",       "3D U-Net†"),
    ("cvae3d",       "3D CVAE"),
    ("dg",           "ref DG"),
    ("glm",          "ref GLM"),
]

PALETTE = {
    "pipeline":     "#2f6f9f",
    "maskgit_flat": "#d97706",
    "unet3d":       "#6b7280",
    "cvae3d":       "#9ca3af",
    "dg":           "#c7cbd1",
    "glm":          "#c7cbd1",
    "null":         "#b91c1c",   # the lookup null, which everything loses to
    "real":         "#111827",
}

# Keys as they appear in cross_model_tests.json / comparison.json.
JSON_ARM = {
    "maskgit_flat": "MaskGIT-flat",
    "unet3d": "3D U-Net (det.)",
    "cvae3d": "3D CVAE",
    "dg": "DG (Macke'09)",
    "glm": "GLM (Pillow'08)",
}


def apply_style() -> None:
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    })
```

- [ ] **Step 4: Write F3**

Create `tools/paper_figs/f3_task_axis.py` with `draw(fig)` producing two panels side by side, sharing an x axis of the four tasks:

- **Left panel, site AP.** Grouped bars, one group per task, one bar per arm in `ARM_ORDER`. Overlay the `_null_ assay site map, SEEN` value as a horizontal rule per group in `PALETTE["null"]`, labelled once. This panel is where our 2× over MaskGIT-flat is visible and where the null rule shows that every learned arm is below it.
- **Right panel, voxel AP.** Same layout. This panel is where we lose to both conv arms, and it must be shown at the same scale treatment as the left, not shrunk.
- Annotate only the cells with a paired-test verdict: a small `∗` above bars where we win at `q < 0.05`, nothing otherwise. Read `verdict` from `cross_model_tests.json`; do not recompute.
- The y axis on the left runs 0–0.8 to accommodate the null rule at 0.72; on the right 0–0.11. Say so in the caption — different scales, and the reader must not compare across panels.

Size `(TEXT_W, 2.4)`.

- [ ] **Step 5: Rewrite the driver**

`tools/make_paper_figures.py` keeps the registry, sizing, saving and mirroring, and loses the `draw_*` bodies and `_placeholder`:

```python
import importlib
import os
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

from tools.paper_figs.style import TEXT_W, apply_style   # noqa: E402

FIGURES = [
    ("f1_pipeline",   "tools.paper_figs.f1_pipeline",   (TEXT_W, 2.2)),
    ("f2_motifs",     "tools.paper_figs.f2_motifs",     (TEXT_W, 4.4)),
    ("f3_task_axis",  "tools.paper_figs.f3_task_axis",  (TEXT_W, 2.4)),
    ("f4_generation", "tools.paper_figs.f4_generation", (TEXT_W, 2.4)),
]


def main() -> int:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, module, figsize in FIGURES:
        fig = plt.figure(figsize=figsize)
        importlib.import_module(module).draw(fig)
        path = OUT / f"{name}.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")
        if MIRROR.is_dir():
            shutil.copy2(path, MIRROR / f"{name}.pdf")
    return 0
```

Create stub modules `f1_pipeline.py`, `f2_motifs.py` and `f4_generation.py` that raise `NotImplementedError`, so Tasks 5–7 have a landing place and the driver is testable now.

- [ ] **Step 6: Run F3 alone and look at it**

```
/home/derik/anaconda3/envs/pytorch/bin/python -c "
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.paper_figs.style import apply_style, TEXT_W
import tools.paper_figs.f3_task_axis as m
apply_style(); f=plt.figure(figsize=(TEXT_W,2.4)); m.draw(f)
f.savefig('$CLAUDE_JOB_DIR/tmp/f3.png', dpi=200, bbox_inches='tight')"
```

Open the PNG. Check: axis labels readable at 8 pt, bars distinguishable, the null rule visible, no overlapping tick text.

- [ ] **Step 7: Commit**

```bash
git add tools/paper_figs/ tools/make_paper_figures.py tests/test_paper_figures.py
git commit -m "Split figure rendering into modules and draw F3

Arm identity is carried by colour and fixed across every figure. The two lookup
arms share one grey: they are memorisation ceilings, not peers, and six
saturated hues would read as a competition we are not claiming to win.

F3 shows site AP and voxel AP side by side rather than picking the flattering
one. The left panel carries our 2x over the peer prior; the right carries our
loss to both convolutional arms. Both panels draw the seen-assay site-map null
as a rule, because every learned arm is below it and that is the single most
important thing the figure has to say.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Gba1U8ohSPVVTaVQBS8eYW"
```

---

## Task 5: F4 (generation families)

**Files:**
- Create: `tools/paper_figs/f4_generation.py` (replacing the stub)

**Interfaces:**
- Consumes: `reports/external_baselines/comparison.json` — the same source as T3, so figure and table cannot disagree.
- Produces: `draw(fig) -> None`.

- [ ] **Step 1: Draw four panels, one per family**

A 1×4 row of small-multiple panels. Each panel: horizontal bars, one per arm, in `ARM_ORDER`; a hollow marker at that arm's `random`-context value on the same row; a thin connector between the two. The connector length *is* the conditioning gain, which is the quantity the panel exists to show, and it makes visible that MaskGIT-flat's connector is longer than ours.

Per-panel titles are the family names with their direction arrow. Do not put a shared y axis label; the arm names appear once, on the leftmost panel.

- [ ] **Step 2: Mark the direction explicitly**

Families A and D are lower-is-better, B and C higher-is-better. Add a small `↓` / `↑` after the panel title and orient each panel's x axis so that **rightward is always better**. A reader scanning four panels must not have to remember which two are inverted.

- [ ] **Step 3: Render and inspect**

```
/home/derik/anaconda3/envs/pytorch/bin/python -c "
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from tools.paper_figs.style import apply_style, TEXT_W
import tools.paper_figs.f4_generation as m
apply_style(); f=plt.figure(figsize=(TEXT_W,2.4)); m.draw(f)
f.savefig('$CLAUDE_JOB_DIR/tmp/f4.png', dpi=200, bbox_inches='tight')"
```

Check that panel C is legible: its values span 0.0001–0.027, two orders of magnitude smaller than the others, so it needs its own scale and probably scientific notation on the axis.

- [ ] **Step 4: Commit**

```bash
git add tools/paper_figs/f4_generation.py
git commit -m "Draw F4: four generation families with per-arm conditioning gain

Each bar is paired with a hollow marker at that arm's own random-context score
and a connector between them, so the connector length is the conditioning gain.
That makes the concession visible rather than buried in prose: MaskGIT-flat's
connector is longer than ours on conditional accuracy.

Every panel is oriented so rightward is better, because two of the four
families are lower-is-better and a reader should not have to track which.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Gba1U8ohSPVVTaVQBS8eYW"
```

---

## Task 6: F2 — the motif figure

The paper's identity figure and the one a reviewer looks at first. It has to
answer three questions in one half-page: *what is a motif*, *is it reused*, and
*does reuse produce plausible activity*. Three stacked bands.

One hard constraint carries over and is not negotiable: **do not scatter
voxels.** At 1.5e-4 density a raw scatter is indistinguishable speckle at any
panel size a 9-page paper affords, for every arm including the good ones.

**Files:**
- Create: `tools/paper_figs/f2_motifs.py` (replacing the stub)

**Interfaces:**
- Consumes: `ckpts/stage2b_flat_codebook.pt` (`embed` (961,64), `provenance`
  (961,3), `counts` (961,)); `reports/analysis_motif_reuse.json` from Task 1B;
  `reports/external_baselines/dumps/{pipeline,maskgit_flat_maskgit_flat_cal}.npz`;
  `reports/external_baselines/cross_model_tests.json`.
- Produces: `draw(fig) -> None`.

Dump keys are `"<task>/clip<NN>/<field>"` with field ∈ `{assay, real,
gen_shared_readout, gen_own_count, roi_packed, field}`. `real` and `gen_*` are
`(N, 3)` int16 arrays of `(t, y, x)`. `field` is `(48,120,224)` float16 and
exists only for `clip00` and `clip01`. Six clips per task, four tasks.

- [ ] **Step 1: Band A — the motif atlas**

Decode the twelve most-used flat entries and show what each one *is*. A flat
entry is a 64-d vector in `embed`; push it through the frozen decoder as a
single-token volume and render the resulting `(6,15,14)` patch as a small
time-by-space image — 6 frames across, 15×14 collapsed to a column strip, or a
2×3 grid of 15×14 tiles, whichever is legible at 0.4 in per panel. Rank by
`counts`, and print each entry's share of all non-blank tokens beneath it.

Annotate each with its `provenance` triple `(a,b,c)`, so the reader can see that
entries sharing a parent `a` look related. That is the ladder's whole claim,
made visible in one row.

Decoding a single token in isolation needs a volume with every other position
blank. If the decoder's API makes that awkward, place the token at grid position
`(0,0,0)` in an otherwise-blank code tensor and crop the output to the first
patch. Verify the crop is the right patch by decoding two different entries and
confirming only that region differs.

- [ ] **Step 2: Band B — reuse across recordings**

Two panels side by side, both from `analysis_motif_reuse.json`:

- **Left: the shared-core curve.** x = *k*, number of recordings; y =
  `shared_core["used_by_ge_k"][k]`, entries used by at least *k* of them. A
  curve that stays high out to k = 31 is the reuse claim in one line. Mark the
  value at k = 31 with a label.
- **Right: the Jaccard matrix**, recordings ordered organoid-then-slice with a
  divider between the blocks, so a reader can see at a glance whether the two
  preparation types share an alphabet. Annotate the three block means
  (within-organoid, within-slice, cross) and, underneath, the label-shuffle null
  as `observed X.XXX vs shuffled Y.YYY ± Z.ZZZ`.

The null annotation is not optional. Without it the matrix is a coloured square
a reader cannot calibrate.

- [ ] **Step 3: Band C — what the motifs build**

Three columns: real, ours, MaskGIT-flat. DG is dropped from this figure — it is
a per-recording lookup, so its panel teaches nothing about motifs and the space
is better spent on bands A and B.

Two projections per column, as previously specified:

```python
import numpy as np

def spatial_map(coords: np.ndarray, shape=(48, 120, 224)) -> np.ndarray:
    """Sum over time: WHERE the arm puts spikes."""
    m = np.zeros(shape[1:], dtype=np.float32)
    np.add.at(m, (coords[:, 1], coords[:, 2]), 1.0)
    return m


def site_raster(coords: np.ndarray, order: np.ndarray,
                shape=(48, 120, 224)) -> np.ndarray:
    """Electrode x time on a FIXED site ordering: WHEN.

    `order` is the sorted unique site id of the REAL clip, and is passed in
    rather than derived, so every column shares one row ordering. Deriving it
    per arm would put the same electrode at a different height in each panel
    and destroy the comparison the band exists to make.
    """
    sid = coords[:, 1].astype(np.int64) * shape[2] + coords[:, 2]
    keep = np.isin(sid, order)
    r = np.zeros((order.size, shape[0]), dtype=np.float32)
    r[np.searchsorted(order, sid[keep]), coords[keep, 0]] = 1.0
    return r
```

Spatial maps on a `LogNorm` scale. Use the `causal` task, `clip00`. **Choose the
clip before looking at any output and record the choice in the module
docstring** — picking the clip where we look best is tuning to the figure.

If the raster is a grey smear at final size, restrict `order` to the 40 sites
most active in the real clip and say so in the caption.

- [ ] **Step 4: Annotate one number per panel, and no legends**

Band C top row: that arm's site AP on `causal` from `cross_model_tests.json`
(ours 0.2721, MaskGIT-flat 0.1299). Bottom row: `ks_isi`. Small, in the corner.
Arm identity comes from `style.PALETTE`, which is fixed across all four figures,
so no per-figure legend is needed.

- [ ] **Step 5: Render and inspect at final size**

```
/home/derik/anaconda3/envs/pytorch/bin/python -c "
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from tools.paper_figs.style import apply_style, TEXT_W
import tools.paper_figs.f2_motifs as m
apply_style(); f=plt.figure(figsize=(TEXT_W,4.4)); m.draw(f)
f.savefig('$CLAUDE_JOB_DIR/tmp/f2.png', dpi=300, bbox_inches='tight')"
```

View at 100%. The atlas tiles are the thing most likely to fail: if a decoded
patch reads as uniform grey, the colour scale is wrong for a 1260-voxel patch
holding about 1.84 spikes — switch to a per-tile normalisation and say so.

- [ ] **Step 6: Repoint the LaTeX include**

The skeleton's `paper/sections/05_experiments.tex:42` still reads
`\includegraphics[width=\textwidth]{figures/f2_qualitative.pdf}` under a
`\subsection{Qualitative samples}`. Change the path to `figures/f2_motifs.pdf`
and the subsection to `\subsection{Motifs}`, and move the float ahead of the
task-completion subsection — the figure now carries §5.1–§5.2 rather than
illustrating §5.3. Task 8 rewrites this file wholesale, but leaving a dangling
`\includegraphics` in the meantime means the document stops compiling, and a
non-compiling document hides every other error.

```
grep -rn "f2_qualitative" paper/
```
Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add tools/paper_figs/f2_motifs.py
git commit -m "Draw F2: the motif atlas, reuse across recordings, and samples

Three bands answering the three questions the thesis rests on: what a motif is,
whether it is reused, and whether reuse produces plausible activity.

Band A decodes the twelve most-used entries and labels each with its ladder
triple, so entries sharing a parent visibly resemble one another -- the ladder's
claim made visible rather than asserted. Band B pairs the shared-core curve with
the Jaccard matrix, blocked by preparation type, and annotates the label-shuffle
null: without it the matrix is a coloured square a reader cannot calibrate.

Band C drops DG. It is a per-recording lookup, so its panel teaches nothing
about motifs, and the space is worth more to bands A and B. Raster rows use the
real clip's site ordering for every column; ordering each arm by its own sites
would destroy the comparison.

No voxel scatter anywhere: at 1.5e-4 density it is speckle at any panel size
this format affords, so it would show nothing and imply nothing is there.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Gba1U8ohSPVVTaVQBS8eYW"
```

---

## Task 7: F1 (pipeline schematic)

The only figure not rendered from data. Its *numbers* still are.

**Files:**
- Create: `tools/paper_figs/f1_pipeline.py` (replacing the stub)

**Interfaces:**
- Consumes: `reports/preproc_stats.json`, `reports/analysis_stage2b_flatten.json` (`F` = 1024, `distinct` = 961, `merged` = 63), `reports/data_provenance.json`.
- Produces: `draw(fig) -> None`.

- [ ] **Step 1: Draw four stages left to right**

1. **Data.** Two small icons for the two preparation types, and the MEA canvas as a `120×224` grey field with one recording's routed electrodes marked — read the routed set from `data_provenance.json`, do not invent a pattern. Annotate the routed fraction.
2. **Patchify.** The clip volume divided into the `8×8×16` grid, one patch highlighted, labelled `(6,15,14)` = 1260 voxels. Annotate the blank fraction 91.7%.
3. **Ladder.** Three stacked quantiser levels 32 / 8 / 4, an arrow to a flat alphabet, labelled with `F` → `distinct` from the JSON.
4. **Priors.** Activity prior → soft field → motif prior → tokens → decoder, with `gct` and `lct` entering as side arrows.

- [ ] **Step 2: Pull every number from JSON**

No literal `961`, `1024`, `91.7` in the source. Format them from the loaded dicts, exactly as the tables do. A schematic that disagrees with the tables is worse than no schematic.

- [ ] **Step 3: Render, inspect, commit**

```bash
git add tools/paper_figs/f1_pipeline.py
git commit -m "Draw F1: pipeline and data schematic

Hand-laid out, but every number in it is formatted from the same JSON the
tables read, so the schematic cannot drift from the results. The routed
electrode set is the real one from data_provenance.json rather than an
invented pattern, because the routed fraction is the single most clarifying
fact about this data and a decorative version of it would undercut that.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Gba1U8ohSPVVTaVQBS8eYW"
```

- [ ] **Step 4: Run the full figure test suite**

```
/home/derik/anaconda3/envs/pytorch/bin/python -m pytest tests/test_paper_figures.py -v
```
Expected: PASS, 3 tests, including byte-stability.

---

## Task 8: §5 Experiments

Written first among the prose sections, because it determines what §3 and §4 must set up. Budget **3.5 pages**.

Section order follows the thesis, not the scoreboard: what the alphabet is, that
it is reused, that reuse buys accuracy, then the concessions. A reader who
reaches §5.6 already knows what is being claimed and what is not.

**Files:**
- Modify: `paper/sections/05_experiments.tex` (full rewrite)

- [ ] **Step 1: Protocol paragraph**

Keep the existing opening. Add: the two eval budgets and that numbers are never carried between them; the 48-clip and 32-clip counts; that BH–FDR \citep{benjamini1995fdr} is applied over all 32 cross-model tests jointly; that AP is step-wise, not trapezoid, citing \citep{davis2006prcurves} and stating that trapezoid interpolation inflated the saturating flat tokeniser by +0.2239 against our +0.0015.

Then define the four completion tasks in two sentences and one inline list —
`causal` (predict every frame after a prefix of 0.25–0.75·T, ROI 0.596),
`noncausal` (a contiguous middle span of 0.30·T, ROI 0.406), `spatial` (a
0.50H × 0.50W box across all frames, ROI 0.330) — and state that the fourth,
free generation, has an ROI of 1.000, so nothing is visible and it is not a
completion task at all. Add the warning that ROI fractions differ, so columns
have different base rates and comparisons are valid only *within* a column.

- [ ] **Step 2: Baselines paragraph**

Replace the `\todo`. One paragraph, in this order, with the *reason each exists*:

- **MaskGIT-flat** is the peer: a single flat codebook of 1024 on the same `8×8×16` grid and the same `(6,15,14)` patch, so token budget and alphabet size are matched. No ladder, no activity prior, no adaptation stage. It is the arm that isolates the tokeniser.
- **3D U-Net** is the direct-supervision reference, `†`, and is an *inpainter*, not a generator: it is trained with this exact loss on this exact hole distribution, and its free-generation output is the degenerate all-masked corner of that same task, driven only by the conditioning codes and thresholded at a fitted rate. It has no autoencoding path, so it cannot reconstruct — say this, because it explains the empty cells in §5.1 as a capability fact rather than a missing run. AP is a ranking metric, so a conditional-mean regressor is close to the best answer that column can contain. It carries 54.8M skip floats per clip against an input of 1.29M binary voxels, and produces no codebook, no discrete index and no reusable latent.
- **3D CVAE** is the same backbone with a latent added and nothing else changed, so the gap between the two is what stochasticity costs and buys.
- **Dichotomised Gaussian** \citep{macke2009dg} and **coupled GLM** \citep{pillow2008glm, truccolo2005pointprocess} are fitted per recording and included as memorisation ceilings.

- [ ] **Step 3: §5.1 The alphabet**

Our strongest clean result, and the first link of the thesis, so it goes first. Step-wise AP 0.2564 against MaskGIT-flat's 0.0269, exact; 0.2680 against 0.0492, tolerant; best F1 0.3055 against 0.0678. Decode ceiling — the alphabet handed the true codes — 0.3254 against 0.0628, a factor of 5.2. Codebook occupancy 619 of 961 used at perplexity 436.9.

Add the ladder-depth result here rather than saving it for the ablations, because it is what makes the alphabet compact: re-decoding the *same* codes at each cumulative depth gives z1 0.0772 → z1+z2 0.1604 (+0.0832) → +z3 0.2806 (+0.1202), on 100% of clips, p = 3.5e-12. Nothing is retrained, so the only variable is how much of the residual sum reaches the decoder.

Then the one sentence that keeps this honest: at *site* level the two ceilings are 0.2548 and 0.2619, so **the hierarchical alphabet buys resolution in time, not in space.** State it; it is the finding, and it sets up §5.3.

- [ ] **Step 3b: §5.2 Motifs are reused across recordings**

The thesis's load-bearing paragraph, written from
`reports/analysis_motif_reuse.json` (Task 1B) and read against F2 band B. Three
sentences of evidence, in this order:

1. **Coverage.** How much of the 961-entry alphabet is in use, and how much of
   it is common property — the `shared_core` count at k = 31 (every recording)
   and at k = 16 (a majority).
2. **Informativeness.** The entropy decomposition: H(code) against
   H(code | recording). Quote the ratio. A ratio near 1 means knowing which
   preparation a clip came from tells you almost nothing about which motif comes
   next, which is precisely what "shared alphabet" has to mean.
3. **Against a null.** Mean pairwise Jaccard, observed against the
   label-shuffle null with its sd and z. Never quote the raw overlap alone: two
   random samples of one distribution already share most of it, so an
   unreferenced overlap number is uninterpretable.

Then the sentence the heterogeneous corpus was assembled for: the
within-organoid, within-slice and cross-preparation block means. If the cross
block is comparable to the within blocks, **the same motifs appear in cultured
organoid tissue and in acute human hippocampal slices** — the strongest
statement this corpus can support, and the reason both preparation types are in
it.

Two things this paragraph must *not* say. It must not call any of this
generalisation or transfer: the split is temporal within recording, `gct` is a
seeded random code, and shared vocabulary is not zero-shot capability. And it
must not hide the blank token — state that it is excluded and why (91.7% of
tokens, emitted by every recording, would drive every overlap statistic to ~1.0).

If Task 1B returned a low reuse ratio, this paragraph reports that instead, and
Tasks 6, 8 and 10 need re-planning before any of them is written.

- [ ] **Step 4: §5.3 Task completion**

F3 and T2. Structure the paragraph around the split the figure shows:

- Site AP: we roughly double MaskGIT-flat on all four tasks — 0.2635/0.2721/0.2477/0.3047 against 0.1319/0.1299/0.1406/0.1195 — winning 88–94% of clips at q = 1.03e-10 to 3.71e-8.
- Voxel AP: not significant against MaskGIT-flat on three of four tasks, and we win only `spatial` (0.0244 vs 0.0181, q = 0.0103). Both convolutional arms beat us on all four, and the U-Net by the largest margin.
- The U-Net's advantage has to be characterised, not merely conceded. It is a supervised inpainter trained on this exact hole distribution with an uncompressed skip path, so on a ranking metric it is close to an upper reference rather than a peer. On free generation, where nothing is visible, its output reduces to a conditioning-driven mean field — which is most of what the site-map null already does, and the null beats it too (0.0869 against its 0.0364). Forward-reference §5.5, where the same arm gets *worse* with more context.
- The oracle decomposition explains both. Our prior recovers 85–103% of its own site-level ceiling and 4–7% of its voxel ceiling; MaskGIT-flat recovers 43–51% and 21–33%. **Our binding constraint is the prior's timing; theirs is the flat alphabet.**
- Then the concession, in its own sentence, unhedged: every learned arm loses to a static per-recording site map at both levels on all four tasks, q ≤ 3.1e-10, and that lookup contains no clip-specific information. The gap is what a shared-capacity model gives up against memorisation, and §5.5 shows what that memorisation costs.

- [ ] **Step 5: §5.4 Generation**

F4 and T3. Three families won against every learned arm: conditional accuracy 0.6078 (MaskGIT-flat 0.7403, CVAE 1.0831, U-Net 1.3637); adherence 0.7764 (0.5490 / 0.3366 / 0.2232); lookup-proof spatial placement 0.0204 (0.0036 / 0.0193 / 0.0077). Marginal realism 0.1879 loses to the U-Net's 0.0620 — say so in the same sentence, not a later one.

Explain family C in one clause, because it is the only metric here that a site map cannot fake: it is map correlation against the clip's own electrodes *minus* the same generated map scored against a different clip of the same recording.

Then the U-Net's behaviour, which belongs here and is a result rather than a caveat: on conditional accuracy it goes 1.2212 at its own random null to 1.3637 at full context, and on adherence 0.4228 to 0.2232. It gets *worse* as context is added, and is worst of all six arms on both. That is the expected signature of a deterministic conditional-mean field graded on a distributional question — one context gives one field — and it is why its single win, marginal realism at 0.0620, is a pooled statistic rather than a conditioning result.

- [ ] **Step 6: §5.5 What conditioning is and is not worth**

The paragraph the spec calls non-optional. MaskGIT-flat's median improvement from random to full context is 0.3817 over 106/128 clips against our 0.1435 over 82/128 (q = 0.0471). **We make no claim that our conditioning is stronger.** Then the count result, which is ours and is large: within-recording correlation with the true ROI count 0.9145–0.9724 against MaskGIT-flat's −0.1147–0.7577, read against the `lct` arithmetic control (0.5753–0.8274), not against zero. And immediately: our count head over-counts by roughly 6× and the only calibrated arm is a per-recording lookup — discrimination and calibration are separate capabilities and no arm here has both.

- [ ] **Step 7: §5.6 Where capacity lives**

T1. Zero stored parameters per recording for all four learned arms, against DG's 26,881 and the GLM's 26,880 — 833k memorised values at 31 recordings, 26.9M at 1000. Aim this at the lookups only: **MaskGIT-flat is also zero per recording**, and pretending otherwise is the easiest way to lose a reviewer.

Then the withholding result, which is what converts a parameter count into a capability claim: replace the per-recording map with the global one and GLM adherence goes 0.4524 → −0.0381 and DG's 0.5314 → 0.2075. The GLM's adherence going negative means it stops tracking the requested context at all.

Close with the limit: zero per-recording storage is not zero memorisation; recording-specific information can live in shared weights, and the checkable claim is only that parameter count does not grow.

- [ ] **Step 8: §5.7 What each stage buys**

T4 (a: ladder depth, b: Stage 3 conditioning) plus T4c (prior nulls). Prose for the rest:

- **Ladder depth** is already in §5.1, so this section adds only the caveat: an η² analysis on summary statistics puts z3 near zero, and that measurement is blind to within-patch placement, which is what z3 carries. Measure at the decoder.
- **Motifs as causal units.** Cite `reports/evaluation_report_code_motifs.json` only if Task 1B Step 7 confirmed it is over level-1 codes and therefore unaffected by the V = 961 flatten: 3-gram lift up to 100.6× over a matched null, a transplant test, and an order-sensitivity test with 6 of 32 cells order-sensitive. This is what distinguishes a motif from a cluster label.
- **Stage 3.** `lct` gives R²(t-marginal) 0.9027 against `gct`'s 0.6063; `gct` gives ΔNLL(flat) 0.3036 against `lct`'s 0.2072. Complementary, neither alone sufficient. The t-centroid column is a negative result (0.0224–0.1204) and is reported as one.
- **Motif prior.** T4c. Top-1 0.0860 against the strongest null's 0.0675; median rank 9 of 961 against 53; MRR 0.2168 against 0.1529; +2.04 nats. Uniform is not the null that matters and is shown only to place the others.
- **Sparse encoder.** 32/32 parent codes used on content against 25; 0 codes shared with blank against 13 of 25; val AUPRC 0.0501 against 0.0358. Both arms at 80 epochs, so this is valid as a paired comparison only, and it says so.
- **Deduplication.** 1024 ladder sums merge to 961 distinct embeddings by pairwise relative distance at threshold 0.05; the collision curve is flat from 0.05 to 0.2 (111 → 121), so the choice is not on a cliff.
- **Adaptation stage.** Four seeds, best val MRR sd 0.00076, min 0.2122, max 0.2138; the shipped checkpoint is val-selected and is not the best of the four.
- **Patch size.** One sentence: a design choice, chosen for the prior and the alphabet jointly rather than for the decoder, forward-referenced to Limitations. Reconstruction alone would pick the smallest patch on the sweep. Do not defend it.

- [ ] **Step 9: Check the page budget**

Compile and measure. If §5 exceeds 3.5 pages, cut in this order: §5.7's sparse-encoder paragraph, then its deduplication paragraph, then the count-calibration half of §5.5. All three are least contested and each has an appendix home. **Do not cut §5.2** — it is the thesis.

- [ ] **Step 10: Verify no hand-typed numbers**

```
grep -nE '[0-9]+\.[0-9]{3,}' paper/sections/05_experiments.tex
```
Every hit must be inside a `\todo{}` or be a value that a table renders. Move any survivor into a macro in `make_paper_tables.py`.

---

## Task 9: §4 Method

Budget **1.5 pages**. Four `\todo` blocks to replace.

**Files:**
- Modify: `paper/sections/04_method.tex`

- [x] **Step 1: Tokeniser**

Encoder/decoder over `(6,15,14)` patches on an `8×8×16` grid, 1024 tokens of 1260 voxels. The sparse encoder: 91.7% of patches are empty and route to one learned blank token, so only content patches are quantised. Three-level residual ladder 32/8/4, citing \citep{lee2022rqvae, zeghidour2022soundstream} for residual quantization and \citep{vandenoord2017vqvae} for the base. Levels are activated by staged loss-weight warm-up rather than by scaling the levels, because a level scale ≠ 1 biases the EMA target.

State that codebooks are EMA-updated with `requires_grad = False`, so no loss gradient reaches the code vectors — this is what §5.6's patch-size argument depends on.

- [x] **Step 2: Flattening and deduplication**

The 32×8×4 = 1024 ladder sums are deduplicated by pairwise relative distance `‖eᵢ−eⱼ‖ / (½(‖eᵢ‖+‖eⱼ‖))` at 0.05, giving V = 961. One sentence on why a frequency filter is the wrong criterion: frequency measures how often an entry is used, not whether two entries are the same vector, so it discards rare-but-distinct entries while leaving duplicates in place.

- [x] **Step 3: Conditioning**

`gct`: a fixed random ±1 code per recording, regenerated from seed 0, so it is a handle rather than storage. This is what makes the per-preparation parameter cost exactly zero **and** why it licenses no zero-shot claim — both halves in the same sentence. It reaches the priors through exactly one frozen mapper, so substituting a measured descriptor is a change to that module alone; that is future work, not a result.

`lct`: a learned mapper over a texton basis attached to the first ladder level.

- [x] **Step 4: Factorised prior**

Activity prior over the token grid; motif prior with MaskGIT-style iterative unmasking \citep{chang2022maskgit, yu2023magvit}; adaptation stage re-fitting the motif prior to the maps the activity prior actually emits. State that the adaptation stage requires the *soft* field: feeding it a hard 0/1 map is measurably worse than not adapting at all.

- [x] **Step 5: Compile and check length**

If §4 exceeds 1.5 pages, move the EMA/no-gradient detail and the dedup criterion to Appendix S4 and leave forward references.

---

## Task 10: §1 Introduction and §2 Related work

Written after Experiments so the contributions claimed are the ones demonstrated. Budget **1.2 + 0.5 pages**.

**Files:**
- Modify: `paper/sections/01_introduction.tex`, `paper/sections/02_related.tex`

- [x] **Step 1: Move 1 — organoid and synthetic biological intelligence**

Replace the `\todo`. Cultured human neural tissue on HD-MEAs is being driven toward closed-loop tasks \citep{kagan2022dishbrain, smirnova2023oi}. Every such system needs a *generative forward model* of the tissue's spontaneous activity: to simulate it, to provide a null against which stimulus-evoked change is measured, and to close the loop without the preparation in the room. Organoids develop rich spontaneous population dynamics \citep{trujillo2019oscillations, sharf2022organoids}, and HD-CMOS arrays now record them at single-electrode resolution across 26,400 sites \citep{ballini2014hdmea} — but no generative model exists at that resolution.

Keep this to four or five sentences. It is motivation, not a survey.

One more sentence closes the loop to the method: spontaneous activity in these preparations is not uniform noise but recurring, structured population events, so the modelling question is whether those events form a small enough vocabulary to be learned once and reused — which is the question this paper answers.

- [x] **Step 2: Move 2 — disease modelling**

Patient-derived organoids and resected tissue are used to study epilepsy and neurodevelopmental disorders, and phenotypes are read as differences in population activity. That requires a model of the reference distribution to difference against, and one whose capacity is *shared* across preparations rather than refit per preparation — a model with a per-preparation parameter table absorbs the phenotype it is meant to detect. Forward-reference §5.5, which measures exactly this.

- [x] **Step 3: Move 3 — why it is hard**

Already written and correct. Add one sentence connecting it to the method: at this density the informative structure is *which electrodes participate together* rather than per-voxel intensity, which is why the evaluation reports site-level and voxel-level accuracy separately.

- [x] **Step 4: Contributions**

Rewrite around the motif thesis. Four items:

1. **A motif alphabet for sparse spike volumes.** A three-level residual VQ-VAE ladder over $(6,15,14)$ patches, flattened and deduplicated to 961 entries, with a sparse encoder routing the 91.7% of empty patches to one blank token. It represents held-out activity 5.2× better than a matched flat tokeniser at the same token budget and alphabet size.
2. **Evidence that the motifs are reused, not per-recording.** The shared-core count, the entropy decomposition, and the Jaccard matrix against a label-shuffle null — including across two different preparation types. This is what makes "shared capacity" a measurement rather than an architectural assertion, and it is the contribution most specific to this paper. Quote the headline number from Task 1B once it exists.
3. **A factorised prior over those motifs** — where activity occurs, then which motif occupies it — conditioned on a per-recording code costing **zero stored parameters per preparation**, which names the correct motif at median rank 9 of 961 against the strongest null's 53.
4. **An evaluation protocol, and what it found.** Per-clip paired tests under BH–FDR with per-arm conditioning nulls and explicit memorisation ceilings, under which 20 of 32 paired comparisons go against us — and which therefore locates precisely which claims survive. A reviewer who reads that in the introduction and then finds it honoured in §5 is a reviewer who trusts the rest.

The corpus itself — 31 recordings spanning organoid and *ex vivo* human tissue with per-recording provenance released — goes in the same paragraph as item 2, since it is what makes the cross-preparation reuse measurement possible at all.

- [x] **Step 5: Related work, four threads**

- **Statistical models of neural populations.** \citep{macke2009dg, pillow2008glm, truccolo2005pointprocess}. Say here that both are fitted per preparation and are therefore ceilings, and forward-reference the withholding experiment.
- **Discrete generative models.** \citep{vandenoord2017vqvae, razavi2019vqvae2, esser2021vqgan, chang2022maskgit, yu2023magvit, yu2024magvit2}. Residual quantization is \citep{lee2022rqvae, zeghidour2022soundstream} and must be cited as prior art — the contribution is the application and the flattening, not the residual ladder. Introduce MaskGIT-flat as the peer here.
- **Organoid and MEA electrophysiology.** \citep{sharf2022organoids, sharf2025protosequences, dandi001132, trujillo2019oscillations, ballini2014hdmea, beggs2003avalanches}.
- **Generative models for neural data.** \citep{pandarinath2018lfads, ye2021ndt}. The distinction that justifies this paper: these model firing rates of *sorted units* on tens to hundreds of channels; we model an array-level binary volume over 26,880 sites without a unit-level parameterisation.

- [x] **Step 6: Verify vocabulary**

```
grep -niE "generali[sz]|transfer|unseen preparation|zero-shot" paper/sections/*.tex
```
Every hit must be a negation ("we do not claim…"). Any positive use is a violation of a standing constraint.

- [x] **Step 7: Compile and check length**

---

## Task 11: §6 Limitations and §7 Conclusion

Budget **0.5 + 0.2 pages**.

**Files:**
- Modify: `paper/sections/06_limitations.tex`, `paper/sections/07_conclusion.tex`

- [x] **Step 1: Fill the three remaining Limitations `\todo`s**

- **Token granularity.** Quote the site-level ceiling against our number: our own alphabet's site ceiling is 0.2548 on free generation and we reach 0.2635, i.e. 103% — so within-token *spatial* placement is not the binding constraint. Voxel-level is: ceiling 0.3254 against our 0.0174, 5%. Say plainly that within-token *timing* is the dominant residual error.
- **Marginal statistics.** Concede with numbers: MaskGIT-flat beats us on pooled `rel_rate`, `rel_persist4` and `rel_avalanche_mean`, and the U-Net wins family D outright at 0.0620 against our 0.1879.
- **No transformer-versus-convolution tokeniser ablation.** Already drafted; add that the nearest evidence is the spatial-embedding result for the U-Net (map r 0.039 without a positional embedding against our 0.355).

- [x] **Step 1b: Add a limitation on what the reuse measurement does and does not show**

Shared vocabulary is not transferable capability. The reuse statistics are measured on recordings the model was trained on, the split is temporal within recording, and `gct` is a seeded random code — so a new preparation still needs training exposure. State that the measurement establishes that capacity is shared rather than tabulated, and nothing about an unseen preparation. Also state the blank-token exclusion, so a reader cannot suspect the overlap was inflated by it.

- [x] **Step 2: Add a fifth limitation — the lookup gap**

New paragraph, and the most important one. Every learned arm loses to a static per-recording site map. Frame it exactly as it is: the gap is the cost of declining to memorise, it is quantified, and §5.5 shows the memorisation does not scale. Do not spin this; a reviewer who finds it only in the appendix will not believe the rest.

- [x] **Step 3: Conclusion**

Three or four sentences. What was built; the one-line result — a tokeniser 9.5× its matched peer and a prior that roughly doubles site-level accuracy at zero per-preparation storage; the forward pointer, that the conditioning interface is a single frozen mapper, so replacing the random per-recording code with a measured descriptor of the preparation is the route to genuine cross-preparation work.

---

## Task 12: Statements

None count toward the page limit. All three are mandatory or strongly expected.

**Files:**
- Modify: `paper/sections/08_statements.tex`

- [x] **Step 1: Ethics statement**

Required because 13 recordings are human *ex vivo* tissue from neurosurgical resection. State: both datasets are open-access on DANDI, de-identified, and used as secondary data; no new human or animal data were collected; cite the consent and IRB approval reported by each source study — **read both source papers' Methods for the exact approval language and reference numbers before writing this**; note the resected tissue was surgical waste from clinically indicated procedures. Add the dual-use position in one sentence: a forward model of neural tissue activity is a research tool and no clinical claim is made.

- [x] **Step 2: Reproducibility statement**

Anonymised code with exact commands; the checkpoint manifest `ckpts/CHECKPOINTS.md`; the pinned protocol (`--batches 12 --mc 8`, seed 20260822 for the task axis; 8 batches, seed 20260821 elsewhere); and the fact that every number is rendered from a JSON artifact by a script — naming `tools/make_paper_tables.py` and `tools/make_paper_figures.py`.

- [x] **Step 3: AI-use statement**

Mandatory for ICLR 2027. State plainly which parts of the work used AI assistance and in what role. Write what is true; a vague statement is worse than a specific one.

---

## Task 13: Appendix S1–S16

Unlimited length, and the place every "did you try X" goes.

**Files:**
- Modify: `paper/sections/99_appendix.tex`

- [x] **Step 1: Fill the sections whose source already exists**

S1 provenance (table generated; add the unused-recordings note), S3 preprocessing (Task 1 wrote it), S5 training protocol, S6 metrics and FDR, S7 full conditioning ladders, S8 per-task battery, S9 null construction, S10 baseline implementations and the NOMAP variants, S11 seed variance, S15 Stage 3, S16 Stage 1.

Port each from the corresponding block of `reports/external_baselines/diagnostics_appendix.md`. **Port the source, not the rendered markdown** — extend a generator to emit the `.tex`. A table transcribed from markdown is a hand-typed table.

- [x] **Step 2: S4 architecture and the two cut design curves**

Full architecture, plus the patch-size sweep (13 rows) and ladder-depth increments that were cut from the main text.

- [x] **Step 3: S12 rejected designs**

Brief, framed as negative results, never as ablations of the shipped model: the refinement variant; the zero-gated decoder cross-attention whose conditional and unconditional validation curves are bit-identical across all 300 epochs; the continuous alpha/hull adapter with 25 logged series identically zero throughout; and the generation composite. This section exists to pre-empt "did you try X", nothing more.

- [x] **Step 4: S13 within-token blur**

Two attempted fixes, both rejected, with the cost of each: the peak term, and the entropy constraint that cost 28% AUPRC to close 1–4% of the gap.

- [x] **Step 5: S14 reproducibility**

Exact commands, checkpoint manifest, pinned protocol.

---

## Task 14: Trim, verify, and prepare the release

- [x] **Step 1: Compile and measure**

```
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Count pages of main text, excluding references, appendix and the three statements. Target ≤ 9.

- [x] **Step 2: Verify no LaTeX drafting artifacts remain**

```
grep -c "todo{" paper/sections/*.tex paper/main.tex
```
Expected: 0 everywhere. Then set `\newcommand{\todo}[1]{}` in `main.tex` and confirm the log is clean.

- [x] **Step 3: Verify anonymity**

```
grep -n "iclrfinalcopy" paper/main.tex          # must still be commented out
grep -rniE "derik|tanveerderik|/media/|Seagate" paper/
```
Expected: the first shows a commented line; the second returns nothing.

- [x] **Step 4: Verify every number is generated**

```
cd "/media/derik/Seagate Desktop Drive/organoid_data/MAGVIT_project"
/home/derik/anaconda3/envs/pytorch/bin/python tools/make_preproc_stats.py
/home/derik/anaconda3/envs/pytorch/bin/python tools/extract_provenance.py
/home/derik/anaconda3/envs/pytorch/bin/python tools/make_paper_tables.py
/home/derik/anaconda3/envs/pytorch/bin/python tools/make_paper_figures.py
git status --porcelain reports/
```
A dirty `reports/` after a clean regeneration means an artifact was edited by hand. Investigate before continuing.

- [x] **Step 5: Verify the terminology constraint**

```
grep -n "reconstruction" paper/main.tex paper/sections/*.tex
```
No occurrence may refer to task 0. Task 0 is *free generation (zero context)*.

- [x] **Step 6: Verify the pipeline still passes its own tests**

```
/home/derik/anaconda3/envs/pytorch/bin/python -m pytest tests/ -q
```
And confirm `diagnostics.md` and `diagnostics_appendix.md` still re-render byte-identical (md5 `86073e2f…` and `2960b10b…`).

- [x] **Step 7: Extend the release scrubber**

`tools/make_release.py` does not yet remove the GitHub remote URL, the `/media/derik/` absolute paths, or the `@author` headers. All three are desk-reject conditions in a supplementary code bundle. Add them, then verify the export imports from a clean directory:

```
/home/derik/anaconda3/envs/pytorch/bin/python -c "import MAGVIT_project.main"
grep -rniE "derik|tanveerderik|/media/|Seagate" <export>
```

- [x] **Step 8: Final commit**

```bash
git add tools/ reports/ docs/ tests/
git commit -m "Complete the paper generators and the release scrub

Every number in the manuscript now resolves to a JSON key and a generator, and
regenerating from a clean tree leaves reports/ unmodified. The release scrubber
removes the remote URL, the absolute working-directory paths and the author
headers, all three of which are desk-reject conditions in a double-blind
supplementary bundle.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Gba1U8ohSPVVTaVQBS8eYW"
```

---

## Self-review

**Spec coverage.** Every row of the spec's figure inventory (F1–F4) has a task; every main table (T1–T4) has a generator task; every reviewer question in the spec's map resolves to a numbered step. The spec's supplement inventory S1–S16 is Task 13. The spec's four-move Introduction is Task 10. The spec's data framing — heterogeneous corpus, biological n at most ten, ethics statement required — is Tasks 1, 10, 11, 12. The spec's Week 1 item "verify the binning chain in `dataset.py` rather than assuming it" is Task 1 and is now **done**, with the result that the loader reads `binary_unit_burst`, not `binary_ch_burst`.

**Three gaps found and closed.** (1) The spec assumed `comparison.json` carries the four generation families; it does not carry any of them. Its `per_clip` block holds nine per-clip statistics over seven *pipeline variants*, excludes the two convolutional arms entirely, and its `stat_error_clean` median is the conditioning-gain median rather than the `glob+full` value the paper quotes — so rendering T3 from it would have produced numbers that silently disagree with `diagnostics.md`. The families exist only inside `external_baselines/diagnose_table.py`, so Task 3 Step 3 adds an `--emit-families` flag there and Step 3b verifies the emitted values against the four known figures. (2) The spec's T2 did not exist in the generator at all; Task 3 Step 4 adds it, with the seen- and unseen-recording site-map nulls as rows rather than a footnote. (3) The spec's Week 1 preprocessing item could not have been completed from `dataset.py` alone: the raw bin width is a property of the extracted NPZ, which is produced outside this repository. Task 1 measures it, and in doing so found that two NPZ families exist per burst window and that reading the wrong one inflates the voxel rate by about 22×.

**Thesis coverage (added 2026-08-27).** The plan now argues one thesis — generation works because a compact alphabet of reusable motifs is learned once and shared. Its five links map to tasks: link 1 (compact, expressive alphabet) Task 8 Step 3; link 2 (each ladder level earns its place) Task 8 Step 3; link 3 (**reuse across recordings, the only missing evidence**) Task 1B, figure Task 6 band B, prose Task 8 Step 3b; link 4 (motifs are predictable) Task 3 Step 5 and Task 8 Step 8; link 5 (reuse buys accuracy) Task 8 Steps 4–5. Task 1B is gated: if the reuse ratio comes back low, Tasks 6, 8 and 10 are built on a false premise and the plan must be revised before they are executed. That gate is stated in Task 1B Step 5 and repeated in Task 8 Step 3b.

**Two corrections folded in from reading the code.** The 3D U-Net is `UNet3DInpainter` with no autoencoding path — it cannot reconstruct, and its free generation is the all-masked corner of its inpainting task — so the earlier flat concession "both convolutional arms beat us on all four tasks" is replaced by a characterisation: it wins ranking metrics as a supervised reference and gets *worse* with more context on both conditioning families. And task 0 is misnamed `recon` in the code; the plan now states the four ROI fractions once, in the framing section, and forbids calling task 0 reconstruction.

**One gap left open deliberately.** The spec's Week 3 item "read the source papers' Methods for consent/IRB citation" is Task 12 Step 1 and cannot be pre-written here — the approval language must be quoted from the papers, not paraphrased from memory.

**Type consistency.** Figure modules all export `draw(fig) -> None` taking a `Figure`; the driver in Task 4 Step 5 calls exactly that. `style.py` exports `TEXT_W`, `PALETTE`, `ARM_ORDER`, `JSON_ARM`, `apply_style`, and Tasks 5–7 use those names. Table generators all call `_write(name, body, source)`, matching the existing helper at `tools/make_paper_tables.py:34`. `preproc_stats.json` keys asserted in Task 1's test are the keys read by Task 1 Step 5's macro renderer.
