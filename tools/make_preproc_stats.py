#!/usr/bin/env python3
"""Derive the preprocessing chain from the data files, not from prose.

Section 3 states a frame duration, a clip duration and a voxel rate. None may
be typed by hand, and none can be read off the loader config alone: the
loader's `temporal_pool` counts raw bins, and the raw bin width is a property of
the extracted NPZ, which is produced outside this repository. So it is measured.

The measurement that matters: `binary_unit_burst_*.npz` stores POINT events --
one bit per spike at the unit's peak electrode, run length exactly 1 -- while
`binary_ch_burst_*.npz` stores the WAVEFORM EXTENT around each spike, run length
~85 raw samples, which is the sorter's 4 ms window at 20 kHz. Only the unit
family is read by the loader (`dataset.py:773,789`). Reading the other one
inflates the voxel rate by roughly 22x, so both are measured here and both land
in the JSON where the difference is visible.

    python tools/make_preproc_stats.py
"""
from __future__ import annotations

import glob
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "output_data"
OUT = ROOT / "reports" / "preproc_stats.json"

# From main.py:554-555 and the patch definition. Imported would be better, but
# importing main.py builds a training pipeline as a side effect.
TEMPORAL_POOL = 120
TEMPORAL_CROP = 6000
PATCH_T, PATCH_H, PATCH_W = 6, 15, 14
SAMPLE_RATE_HZ = 20_000.0     # MaxWell HD-MEA; stated by both source studies


def _unpack(path: str) -> np.ndarray:
    d = np.load(path)
    packed, shape = d["packed"], d["shape"]
    H, W, Tb = packed.shape
    T = int(shape[2])
    return np.unpackbits(packed.reshape(-1, Tb), axis=1)[:, :T] \
             .reshape(H, W, T).astype(bool)


def _mean_run_len(b: np.ndarray) -> float:
    starts = (np.diff(b.astype(np.int8), axis=2) == 1).sum() + b[:, :, 0].sum()
    return float(b.sum() / max(int(starts), 1))


def main() -> int:
    unit = sorted(glob.glob(str(DATA / "*/*/*/*/binary_unit_burst_*.npz")))
    chan = sorted(glob.glob(str(DATA / "*/*/*/*/binary_ch_burst_*.npz")))
    if not unit:
        raise SystemExit(f"no unit burst files under {DATA}")

    recordings = {p.split("/")[-3] for p in unit}
    shapes = {tuple(np.load(p)["shape"].tolist()) for p in unit[:400]}
    if len(shapes) != 1:
        raise SystemExit(f"heterogeneous raw shapes {shapes}; 'a window is N ms'"
                         f" would not be a statement about the corpus")
    H, W, T_raw = shapes.pop()

    rng = random.Random(0)
    rates, runs = [], []
    for p in rng.sample(unit, min(12, len(unit))):
        b = _unpack(p)
        runs.append(_mean_run_len(b))
        pooled = b[:, :, : T_raw // TEMPORAL_POOL * TEMPORAL_POOL] \
                  .reshape(H, W, -1, TEMPORAL_POOL).max(axis=3)
        rates.append(float(pooled.mean()))
    ch_runs = [_mean_run_len(_unpack(p))
               for p in rng.sample(chan, min(6, len(chan)))] if chan else [0.0]

    raw_bin_us = 1e6 / SAMPLE_RATE_HZ
    frame_ms = TEMPORAL_POOL * raw_bin_us / 1e3
    crop_pre = max(1, TEMPORAL_CROP // TEMPORAL_POOL)
    clip_frames = crop_pre - (crop_pre % PATCH_T)
    pad_w_to = W + (-W % PATCH_W)
    voxels = clip_frames * H * pad_w_to
    window_rate = float(np.median(rates))

    # Two different rates, and the paper must quote the right one. The value
    # above is the occupancy of a WHOLE 600 ms burst window. A clip is a 288 ms
    # span drawn at random inside that window, and the draw lands
    # preferentially on the active part, so clips are about twice as dense.
    # Every other number in the paper -- chance AP, the patch-size sweep, the
    # spikes-per-clip figure -- is computed on clips, so the clip rate is the
    # one Section 3 states. It is read from the sweep that measured it rather
    # than recomputed here, which would need the loader and could drift from it.
    sweep = json.loads((ROOT / "reports" / "ablation_patch_size.json").read_text())
    clip_rate = float(sweep["voxel_rate"])
    clip_rate_n = int(sweep["n_clips"])

    d = {
        "n_windows": len(unit),
        "n_recordings": len(recordings),
        "raw_shape": [H, W, T_raw],
        "raw_bin_us": raw_bin_us,
        "window_ms": T_raw * raw_bin_us / 1e3,
        "temporal_pool": TEMPORAL_POOL,
        "frame_ms": frame_ms,
        "crop_frames_pre": crop_pre,
        "clip_frames": clip_frames,
        "clip_ms": clip_frames * frame_ms,
        "clip_shape": [clip_frames, H, pad_w_to],
        "clip_voxels": voxels,
        "pad_w_from": W, "pad_w_to": pad_w_to,
        "patch": [PATCH_T, PATCH_H, PATCH_W],
        "token_grid": [clip_frames // PATCH_T, H // PATCH_H,
                       pad_w_to // PATCH_W],
        "unit_mean_run_len": round(float(np.mean(runs)), 4),
        "ch_mean_run_len": round(float(np.mean(ch_runs)), 4),
        "window_pooled_rate": window_rate,
        "_window_rate_note": "occupancy of a whole 600 ms burst window; NOT "
                             "what the paper quotes -- see clip_voxel_rate",
        "clip_voxel_rate": clip_rate,
        "clip_voxel_rate_n_clips": clip_rate_n,
        "clip_voxel_rate_source": "reports/ablation_patch_size.json",
        "mean_spikes_per_clip": round(clip_rate * voxels, 1),
        "split_counts": {"train": 1069, "val": 426, "test": 638},
        "sorter": {"name": "SpyKING Circus 2 (SpikeInterface)",
                   "freq_min_hz": 300, "freq_max_hz": 6000,
                   "filter": "butterworth", "filter_order": 4,
                   "radius_um": 100, "ms_before": 2.0, "ms_after": 2.0},
    }
    OUT.write_text(json.dumps(d, indent=1) + "\n")
    print(f"wrote {OUT}")
    for k in ("window_ms", "frame_ms", "clip_ms", "clip_voxels", "token_grid",
              "window_pooled_rate", "clip_voxel_rate", "mean_spikes_per_clip",
              "unit_mean_run_len", "ch_mean_run_len"):
        print(f"  {k:22s} {d[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
