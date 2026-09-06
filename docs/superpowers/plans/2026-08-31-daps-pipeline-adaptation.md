# DAPS Pipeline Adaptation — Implementation Plan (v4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt the MAGVIT VQ-VAE pipeline (sparse encoder, dense decoder, hierarchical VQ, MaskGIT generation) to work on Cortical Labs DAPS data — 64-channel MEA recordings of stimulation-evoked activity in dense 2D multilayer cortical culture — with learned electrode characterization (from 3x3 neighborhood waveform clustering) replacing the high-density site map.

**Architecture:** Same 4-stage training pipeline as the ICLR submission (Stage 1 = gct mapper, Stage 2 = VQ-VAE, Stage 3 = lct mapper, Stage 4 = prior/generator), with an offline Stage 0 for waveform preprocessing. Input is two-stream: `(x_binary, x_electrode_embed_static)` where `x_binary` is a (1,T,8,8) binary spike raster and `x_electrode_embed_static` is the chip-level marginal electrode characterization (never varies per clip). Stage 0 uses K=128 GMM (GPU, full covariance) on relative-P2P-normalized 3×3 neighborhood features (675 dims = 75-sample waveform × 9 electrodes, PCA to 60 components at 95% variance). All cluster assignments are **soft posteriors** (predict_proba), not hard labels — each electrode is a sparse weighted mixture over the codebook. Electrode characterization enters two ways: (1) dynamic per-clip posterior composition through gct (making gct clip-aware), and (2) static chip-level marginal (`E[dynamic]`) as additive token-level embedding in encoder/decoder. The encoder is context-agnostic: it sees only the static electrode identity, never the per-clip variation. No FiLM — electrode information is additive at the token level. gct = `MLP(concat(stim_embed, elec_char_flat_dynamic, adj_flat)) → 32-d`. Stage 1 pretraining has 3 heads: dynamic electrode embed prediction, adjacency rate prediction, and linear stim param decoder (loss weights swept). Adjacency = temporal autocorrelation P(fire at t+k | fired at t), keyed by gct embedding with rounding — same gap bins as HDMEA. At generation time, the static embed serves double duty (encoder bias AND gct input), which is just the natural marginal — no fallback logic needed. Generation is unified: stim params in gct, null stim = spontaneous (learned null token TBD). Spontaneous clips: random crop from 30s baseline, oversampled at bursty windows. Conv stem: single 3×3×K 3D conv layer (not deep multi-layer). VQ codebook: V=961, same 3-level hierarchy and (111)(011)(000) tolerance. lct design: parked (9+ dims, exact features TBD).

**Tech Stack:** PyTorch (GPU GMM), pytables (HDF5 reading), numpy, scipy, scikit-learn (PCA only), existing MAGVIT codebase

**Spec:** This plan implements the design agreed in the brainstorming sessions of 2026-08-31 through 2026-09-04. The DAPS data lives at `/media/derik/Seagate Desktop Drive/cortical_DAPS_data/`.

## Global Constraints

- Python 3.10+, PyTorch 2.x
- All new code lives under a `daps/` subdirectory to keep the HDMEA pipeline untouched
- Binary spike rasters only — no continuous voltage in the model I/O (raw voltage is read only in Stage 0 for neighbor waveform extraction)
- 2x2 spatial x T/4 temporal patches, stride 2 → 64 tokens per clip
- ctx_field requires pH>=2, pW>=2 — never use single-electrode patches
- No FiLM anywhere — electrode embedding is additive at token level
- Encoder is context-agnostic: receives STATIC electrode embed only (chip-level marginal). Dynamic per-clip characterization goes to gct only.
- Electrode characterization uses 3x3 neighbor waveforms; border/corner electrodes get truncated neighborhoods (zero-padded missing neighbors)
- Held-out stim channels as primary test split
- Reuse existing model classes where possible; subclass or wrap, don't fork
- Dense 2D multilayer cortical culture (NOT organoids) — cell line CLV3.1
- Multi-chip aware: electrode characterization varies per chip (7 chips in full dataset, 1 in current tar)
- Stage 0 clustering runs on training data only — no test-set leakage
- K=128 waveform codebook (BIC knee at D=60, 94.4 effective K, 2 dead codes, all 3 inits converged)
- 675-dim features: full 75-sample waveform × 9 electrodes in 3×3 neighborhood, concatenated
- PCA → 60 components (95% variance); old 83-dim/PCA-20 scheme is superseded
- Soft GMM posteriors throughout — electrodes are weighted mixtures, never hard-assigned (mean dominant weight 70%, ~30% on secondary codes)
- Relative P2P normalization: divide all 675 feature dims by central waveform peak-to-peak amplitude before PCA
- GPU GMM (PyTorch) for tractability — sklearn CPU is too slow
- VQ codebook: V=961, same 3-level hierarchy (32/8/4) and (111)(011)(000) tolerance as HDMEA
- Conv stem: single 3×3×K 3D conv layer (K flexible for temporal axis), not multi-layer deep stem
- gct composition: MLP(concat(stim_embed, elec_char_flat_dynamic, adj_flat)) → 32-d, keyed by gct embedding with rounding
- Stage 1 pretraining: 3 heads (dynamic electrode embed, adjacency rates, linear stim param decoder), loss weights swept
- Adjacency = temporal autocorrelation P(fire at t+k | fired at t), per stim-condition, same gap bins as HDMEA
- Generation: unified model (stim in gct, null stim = spontaneous). Spontaneous clips from random crop of 30s baseline, oversampled at bursty windows.
- Multi-chip: no explicit chip identity — dynamic electrode char encodes it implicitly. Refit Stage 0 per chip.
- Baselines: standard (DG, GLM, conv) + stim-specific (PSTH, linear stim→response, nearest-neighbor)
- Evaluation: standard metrics (AUPRC, conditional accuracy) + stim-specific comparison

## Source Data Summary

- Archive: `cortical_DAPS_data/DAPS_data/DAPS_data` (28GB tar)
- Currently available: sys-019 (MEA serial 40327), 1 chip
- Full dataset: 7 systems (sys-000 through sys-019), different cultures, impedances, treatments (PDMS, AraC, XAV, DAPT)
- 4,130 HDF5 trial files per chip, each ~3.1s, 64 channels, 25kHz, int16
- 3 stim pulses per trial at 0.8s intervals
- 2 baseline recordings (pre/post, 30s each, no stim)
- Per-file: `root.samples` (raw voltage, int16), `root.spikes` (timestamp, channel, 75-pt waveform), `root.stims` (timestamp, channel), `root._v_attrs.application` (stim params, culture info)
- MEA: MCS 60MEA200/30iR-Ti, 8x8 grid, 200um pitch, 30um electrodes
- Full factorial: 59 stim channels x 14 currents (500-3750nA) x 5 pulse widths (40-200us)

## File Structure

```
daps/
├── extract_data.py          # Task 1: tar extraction + H5 → NPZ (with neighbor voltage snippets)
├── waveform_clustering.py   # Task 2: Stage 0 - PCA+GMM on 3x3 neighborhood waveforms
├── dataset.py               # Task 3: DapsDataset (two-stream, 11-dim lct)
├── gct.py                   # Task 4: HybridGCT + StimResponseBank + AdjacencyBank
├── model_adapter.py         # Task 5: model adapter (additive electrode embed, no FiLM)
├── conv_stem.py             # Task 6: single 3×3×K 3D conv stem for 8x8
├── splits.py                # Task 7: held-out channel / current splits
├── train.py                 # Task 8: 4-stage training orchestrator
├── baselines.py             # Task 9: baselines
├── evaluate.py              # Task 10: evaluation + generation metrics
└── tests/
    ├── test_extract.py
    ├── test_clustering.py
    ├── test_dataset.py
    ├── test_gct.py
    ├── test_model.py
    ├── test_stem.py
    ├── test_splits.py
    └── test_baselines.py
```

---

### Task 1: Data Extraction and Neighborhood Waveform Extraction

**Files:**
- Create: `daps/extract_data.py`
- Create: `daps/tests/test_extract.py`

**Interfaces:**
- Consumes: raw HDF5 files from tar archive
- Produces:
  - `extract_all_trials(tar_path, out_dir) -> dict` mapping config keys to NPZ paths. Each NPZ contains `{"spikes_binary": (3, T, 8, 8), "stim_params": dict}`.
  - `extract_neighbor_waveforms(h5_path, fs=25000) -> dict` mapping `(channel, spike_idx) -> {"waveforms": (9, 75)}` where `waveforms[0]` is the central electrode's 75-sample snippet and `waveforms[1:9]` are the 8 spatial neighbors' full 75-sample snippets (zero-padded for missing neighbors at borders/corners). Raw voltage from `root.samples` is read at the spike timestamp.
  - Separate `waveform_db.npz` per chip: all spikes' concatenated waveforms (75 × 9 = 675-dim feature vector per spike), along with electrode ID and clip ID.

**Context:**
- Each H5 file has `root.spikes` table (timestamp, channel, 75-pt waveform) and `root.samples` (raw voltage, shape `(N_samples, 64)`, int16, column-per-channel)
- For each spike at timestamp `t` on channel `ch`: read the 75-sample window `root.samples[t-37:t+38, :]` for the central electrode and all 3×3 spatial neighbors of `ch`. Concatenate all 9 full 75-sample waveforms into a 675-dim feature vector (central first, then NW, N, NE, W, E, SW, S, SE).
- 3x3 neighborhood: for electrode at grid position `(r, c)`, neighbors are all `(r+dr, c+dc)` where `dr, dc in {-1, 0, 1}` and `(dr, dc) != (0, 0)` and the position is within the 8x8 grid.
  - Corner electrodes: 3 neighbors
  - Edge electrodes: 5 neighbors
  - Interior electrodes: 8 neighbors
  - Missing neighbors are zero-padded in the 8-dim neighbor vector (fixed ordering: NW, N, NE, W, E, SW, S, SE)
- Sampling rate: 25kHz. Spike timestamps are in sample units relative to `root._v_attrs.start_timestamp`
- Stim timestamps define epoch boundaries: each epoch is 0.8s (20,000 samples) starting at stim onset
- Time bins: 10ms = 250 samples → 80 bins per 0.8s epoch
- Channel layout (column-major 8x8):
  ```python
  LAYOUT = np.array([
      [0, 8, 16, 24, 32, 40, 48, 56],
      [1, 9, 17, 25, 33, 41, 49, 57],
      [2, 10, 18, 26, 34, 42, 50, 58],
      [3, 11, 19, 27, 35, 43, 51, 59],
      [4, 12, 20, 28, 36, 44, 52, 60],
      [5, 13, 21, 29, 37, 45, 53, 61],
      [6, 14, 22, 30, 38, 46, 54, 62],
      [7, 15, 23, 31, 39, 47, 55, 63],
  ])
  ```
  Channel index → (row, col) via `row = ch % 8`, `col = ch // 8`.
- Baseline recordings: split into non-overlapping 0.8s clips (37 clips from 30s), same binary format but with `stim_params = None`
- The waveform database is saved separately from the binary rasters because Stage 0 runs offline on training data only.

- [ ] **Step 1: Write test for single-trial H5 → binary raster conversion**

```python
# daps/tests/test_extract.py
import numpy as np
import pytest


def test_h5_to_binary_raster_shape():
    """One trial -> 3 binary rasters of shape (80, 8, 8)."""
    from daps.extract_data import h5_to_clips
    test_h5 = (
        "/tmp/claude-1000/-media-derik-Seagate-Desktop-Drive-organoid-data-"
        "MAGVIT-project/58860ec6-d34d-4028-a44b-fb4aaa742cc1/scratchpad/"
        "mnt/labpool/gershom/sys-019/daps/culture/MCS_1_40327_00001/"
        "2024-18-11_00-57-17.697432/002250nA/"
        "2024-11-18_03-27-45.293+00-00_ch_13_current_002250nA_"
        "pulsewidth_0120uS_pol_1_rec_1_iter_2194_of_4130.h5"
    )
    clips, stim_params, spike_data = h5_to_clips(test_h5, bin_ms=10)
    assert clips.shape == (3, 80, 8, 8), f"Got {clips.shape}"
    assert clips.dtype == np.uint8
    assert set(np.unique(clips)).issubset({0, 1})
    assert stim_params["stim_channel"] == 13
    assert stim_params["current_nA"] == 2250
    assert stim_params["pulse_width_us"] == 120


def test_h5_to_binary_raster_has_spikes():
    """The raster should contain spikes (not all zeros)."""
    from daps.extract_data import h5_to_clips
    test_h5 = (
        "/tmp/claude-1000/-media-derik-Seagate-Desktop-Drive-organoid-data-"
        "MAGVIT-project/58860ec6-d34d-4028-a44b-fb4aaa742cc1/scratchpad/"
        "mnt/labpool/gershom/sys-019/daps/culture/MCS_1_40327_00001/"
        "2024-18-11_00-57-17.697432/002250nA/"
        "2024-11-18_03-27-45.293+00-00_ch_13_current_002250nA_"
        "pulsewidth_0120uS_pol_1_rec_1_iter_2194_of_4130.h5"
    )
    clips, _, _ = h5_to_clips(test_h5, bin_ms=10)
    total_spikes = clips.sum()
    assert total_spikes > 0, "No spikes found in raster"
    assert total_spikes > 50, f"Only {total_spikes} spikes, expected more"


def test_channel_to_grid_mapping():
    """Channel 13 should map to (5, 1) in the 8x8 grid."""
    from daps.extract_data import channel_to_grid
    row, col = channel_to_grid(13)
    assert (row, col) == (5, 1), f"Got ({row}, {col})"
    assert channel_to_grid(0) == (0, 0)
    assert channel_to_grid(63) == (7, 7)
    assert channel_to_grid(8) == (0, 1)


def test_neighbor_indices():
    """3x3 neighborhood should handle borders correctly."""
    from daps.extract_data import get_neighbor_channels
    # Corner (0,0): 3 neighbors
    nbrs = get_neighbor_channels(0)
    assert len([n for n in nbrs if n is not None]) == 3
    # Edge (0,3): 5 neighbors
    nbrs = get_neighbor_channels(24)  # ch 24 -> (0, 3)
    assert len([n for n in nbrs if n is not None]) == 5
    # Interior (3,3): 8 neighbors
    nbrs = get_neighbor_channels(27)  # ch 27 -> (3, 3)
    assert len([n for n in nbrs if n is not None]) == 8


def test_neighbor_waveform_extraction_shape():
    """Each spike should produce an 83-dim feature vector."""
    from daps.extract_data import extract_neighbor_waveforms
    test_h5 = (
        "/tmp/claude-1000/-media-derik-Seagate-Desktop-Drive-organoid-data-"
        "MAGVIT-project/58860ec6-d34d-4028-a44b-fb4aaa742cc1/scratchpad/"
        "mnt/labpool/gershom/sys-019/daps/culture/MCS_1_40327_00001/"
        "2024-18-11_00-57-17.697432/002250nA/"
        "2024-11-18_03-27-45.293+00-00_ch_13_current_002250nA_"
        "pulsewidth_0120uS_pol_1_rec_1_iter_2194_of_4130.h5"
    )
    features, channels, clip_ids = extract_neighbor_waveforms(test_h5)
    assert features.shape[1] == 83, f"Got {features.shape[1]} dims, expected 83"
    assert features.shape[0] > 0, "No spikes extracted"
    assert features.shape[0] == len(channels)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/media/derik/Seagate Desktop Drive/organoid_data/MAGVIT_project"
python -m pytest daps/tests/test_extract.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'daps'`

- [ ] **Step 3: Implement h5_to_clips, channel_to_grid, get_neighbor_channels, extract_neighbor_waveforms**

```python
# daps/extract_data.py
"""Convert DAPS HDF5 trial files to binary spike rasters + neighborhood waveform features."""
import numpy as np
import tables
from pathlib import Path

LAYOUT = np.array([
    [0, 8, 16, 24, 32, 40, 48, 56],
    [1, 9, 17, 25, 33, 41, 49, 57],
    [2, 10, 18, 26, 34, 42, 50, 58],
    [3, 11, 19, 27, 35, 43, 51, 59],
    [4, 12, 20, 28, 36, 44, 52, 60],
    [5, 13, 21, 29, 37, 45, 53, 61],
    [6, 14, 22, 30, 38, 46, 54, 62],
    [7, 15, 23, 31, 39, 47, 55, 63],
])

_CH_TO_GRID = {}
for r in range(8):
    for c in range(8):
        _CH_TO_GRID[int(LAYOUT[r, c])] = (r, c)

_GRID_TO_CH = {v: k for k, v in _CH_TO_GRID.items()}

# Neighbor ordering: NW, N, NE, W, E, SW, S, SE
_NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def channel_to_grid(ch: int) -> tuple[int, int]:
    return _CH_TO_GRID[ch]


def get_neighbor_channels(ch: int) -> list[int | None]:
    """Return 8-element list of neighbor channel indices. None for out-of-bounds."""
    r, c = _CH_TO_GRID[ch]
    neighbors = []
    for dr, dc in _NEIGHBOR_OFFSETS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 8 and 0 <= nc < 8:
            neighbors.append(_GRID_TO_CH[(nr, nc)])
        else:
            neighbors.append(None)
    return neighbors


def extract_neighbor_waveforms(
    h5_path: str,
    fs: int = 25000,
    waveform_half: int = 37,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 83-dim feature vectors for all spikes in an H5 file.

    For each spike: central waveform (75) + 8 neighbor trough amplitudes (zero-padded for borders).

    Returns:
        features: (N_spikes, 83) float32
        channels: (N_spikes,) int — which electrode each spike belongs to
        clip_ids: (N_spikes,) int — which epoch (0/1/2) or -1 for baseline
    """
    f = tables.open_file(h5_path, mode="r")
    start_ts = f.root._v_attrs.start_timestamp

    spike_channels = f.root.spikes.col("channel")
    spike_timestamps = f.root.spikes.col("timestamp") - start_ts
    spike_waveforms = f.root.spikes.col("samples")  # (N, 75) int16
    raw_samples = f.root.samples  # (total_samples, 64) int16, on-disk

    stim_ts = np.array([row["timestamp"] - start_ts for row in f.root.stims])

    n_spikes = len(spike_channels)
    features = np.zeros((n_spikes, 83), dtype=np.float32)
    channels_out = spike_channels.astype(np.int32)
    clip_ids = np.full(n_spikes, -1, dtype=np.int32)
    total_raw_samples = raw_samples.shape[0]

    for i in range(n_spikes):
        ch = int(spike_channels[i])
        t = int(spike_timestamps[i])
        wf = spike_waveforms[i].astype(np.float32) * 0.195  # to uV

        features[i, :75] = wf

        # Assign clip_id based on stim timestamps
        for ei, st in enumerate(stim_ts):
            epoch_end = st + int(0.8 * fs)
            if st <= t < epoch_end:
                clip_ids[i] = ei
                break

        # Extract neighbor trough amplitudes from raw voltage
        neighbors = get_neighbor_channels(ch)
        t_start = max(0, t - waveform_half)
        t_end = min(total_raw_samples, t + waveform_half + 1)
        if t_end > t_start:
            raw_snippet = np.array(
                raw_samples[t_start:t_end, :], dtype=np.float32
            ) * 0.195  # to uV

            for ni, nbr_ch in enumerate(neighbors):
                if nbr_ch is not None:
                    features[i, 75 + ni] = raw_snippet[:, nbr_ch].min()
                # else: remains 0.0 (zero-padded)

    f.close()
    return features, channels_out, clip_ids


def h5_to_clips(
    h5_path: str,
    bin_ms: int = 10,
    fs: int = 25000,
    epoch_sec: float = 0.8,
) -> tuple[np.ndarray, dict, dict]:
    """
    Extract binary spike rasters from a DAPS HDF5 trial file.

    Returns:
        clips: (n_stims, T_bins, 8, 8) uint8 binary raster
        stim_params: dict with stim_channel, current_nA, pulse_width_us
        spike_data: dict with per-epoch spike_times and waveforms per channel
    """
    samples_per_bin = int(fs * bin_ms / 1000)
    bins_per_epoch = int(epoch_sec * 1000 / bin_ms)

    f = tables.open_file(h5_path, mode="r")
    attrs = f.root._v_attrs
    start_ts = attrs.start_timestamp
    app = attrs.application

    stim_ts = np.array([row["timestamp"] - start_ts for row in f.root.stims])
    n_stims = len(stim_ts)

    spike_channels = f.root.spikes.col("channel")
    spike_timestamps = f.root.spikes.col("timestamp") - start_ts
    spike_waveforms = f.root.spikes.col("samples")
    f.close()

    clips = np.zeros((n_stims, bins_per_epoch, 8, 8), dtype=np.uint8)
    spike_data = {"times": [], "waveforms": []}

    for ei, st in enumerate(stim_ts):
        epoch_end = st + int(epoch_sec * fs)
        mask = (spike_timestamps >= st) & (spike_timestamps < epoch_end)
        ep_channels = spike_channels[mask]
        ep_times = spike_timestamps[mask] - st
        ep_waveforms = spike_waveforms[mask]

        epoch_spike_times = {}
        epoch_waveforms = {}

        for ch, t, wf in zip(ep_channels, ep_times, ep_waveforms):
            time_bin = int(t / samples_per_bin)
            if time_bin >= bins_per_epoch:
                continue
            row, col = channel_to_grid(int(ch))
            clips[ei, time_bin, row, col] = 1

            ch_int = int(ch)
            if ch_int not in epoch_spike_times:
                epoch_spike_times[ch_int] = []
                epoch_waveforms[ch_int] = []
            epoch_spike_times[ch_int].append(int(t))
            epoch_waveforms[ch_int].append(wf)

        spike_data["times"].append(epoch_spike_times)
        spike_data["waveforms"].append(epoch_waveforms)

    stim_params = {
        "stim_channel": int(app["stim_channel"]),
        "current_nA": int(round(app["stim_current_A"] * 1e9)),
        "pulse_width_us": int(round(app["pulse_width_sec"] * 1e6)),
        "stim_polarity": app.get("stim_polarity", "negative"),
        "recovery": bool(app.get("recovery_bool", True)),
        "culture_info": app.get("culture_information", {}),
    }

    return clips, stim_params, spike_data


def h5_to_baseline_clips(
    h5_path: str,
    bin_ms: int = 10,
    fs: int = 25000,
    epoch_sec: float = 0.8,
) -> tuple[np.ndarray, dict]:
    """Extract non-overlapping binary clips from a baseline recording."""
    samples_per_bin = int(fs * bin_ms / 1000)
    bins_per_epoch = int(epoch_sec * 1000 / bin_ms)
    samples_per_epoch = int(epoch_sec * fs)

    f = tables.open_file(h5_path, mode="r")
    start_ts = f.root._v_attrs.start_timestamp
    total_samples = f.root.samples.shape[0]
    n_clips = total_samples // samples_per_epoch

    spike_channels = f.root.spikes.col("channel")
    spike_timestamps = f.root.spikes.col("timestamp") - start_ts
    f.close()

    clips = np.zeros((n_clips, bins_per_epoch, 8, 8), dtype=np.uint8)

    for ci in range(n_clips):
        t_start = ci * samples_per_epoch
        t_end = t_start + samples_per_epoch
        mask = (spike_timestamps >= t_start) & (spike_timestamps < t_end)
        ep_channels = spike_channels[mask]
        ep_times = spike_timestamps[mask] - t_start

        for ch, t in zip(ep_channels, ep_times):
            time_bin = int(t / samples_per_bin)
            if time_bin >= bins_per_epoch:
                continue
            row, col = channel_to_grid(int(ch))
            clips[ci, time_bin, row, col] = 1

    return clips, {"times": [], "waveforms": []}


def extract_all_trials(tar_path: str, out_dir: str, bin_ms: int = 10) -> dict:
    """
    Extract entire tar archive, convert all H5 files to NPZ.
    Also builds the waveform database for Stage 0.

    Returns dict mapping config_key -> npz_path.
    """
    import tarfile

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}

    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".h5")]
        for member in members:
            tar.extract(member, path=str(out / "_raw"))

    raw_dir = out / "_raw"
    h5_files = sorted(raw_dir.rglob("*.h5"))

    all_wf_features = []
    all_wf_channels = []
    all_wf_clip_ids = []
    all_wf_trial_keys = []

    for h5_path in h5_files:
        name = h5_path.name
        if "baseline" in name:
            tag = "baseline_pre" if "pre_baseline" in name else "baseline_post"
            clips, _ = h5_to_baseline_clips(str(h5_path), bin_ms=bin_ms)
            npz_path = out / f"{tag}.npz"
            np.savez_compressed(str(npz_path), clips=clips, stim_channel=-1,
                                current_nA=0, pulse_width_us=0)
            manifest[tag] = str(npz_path)
        else:
            clips, stim_params, _ = h5_to_clips(str(h5_path), bin_ms=bin_ms)
            key = (f"ch{stim_params['stim_channel']:02d}"
                   f"_cur{stim_params['current_nA']}"
                   f"_pw{stim_params['pulse_width_us']}")
            npz_path = out / f"{key}.npz"
            np.savez_compressed(str(npz_path), clips=clips,
                                stim_channel=stim_params["stim_channel"],
                                current_nA=stim_params["current_nA"],
                                pulse_width_us=stim_params["pulse_width_us"])
            manifest[key] = str(npz_path)

        # Extract neighbor waveforms for Stage 0
        features, channels, clip_ids = extract_neighbor_waveforms(str(h5_path))
        all_wf_features.append(features)
        all_wf_channels.append(channels)
        all_wf_clip_ids.append(clip_ids)
        all_wf_trial_keys.extend([name] * len(channels))

    # Save waveform database
    np.savez_compressed(
        str(out / "waveform_db.npz"),
        features=np.concatenate(all_wf_features, axis=0),
        channels=np.concatenate(all_wf_channels, axis=0),
        clip_ids=np.concatenate(all_wf_clip_ids, axis=0),
        trial_keys=np.array(all_wf_trial_keys),
    )

    return manifest
```

- [ ] **Step 4: Create `daps/__init__.py` and run tests**

```bash
mkdir -p daps/tests
touch daps/__init__.py
touch daps/tests/__init__.py
python -m pytest daps/tests/test_extract.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add daps/
git commit -m "feat(daps): H5 to binary raster + 3x3 neighbor waveform extraction"
```

---

### Task 2: Stage 0 — Waveform Clustering and Electrode Characterization

**Files:**
- Create: `daps/waveform_clustering.py`
- Create: `daps/tests/test_clustering.py`

**Interfaces:**
- Consumes: `waveform_db.npz` from Task 1 (features: `(N_spikes, 675)`, channels: `(N_spikes,)`, clip_ids: `(N_spikes,)`)
- Produces:
  - `run_stage0(features, channels, train_mask, n_pca=60, n_clusters=128, seed=42) -> Stage0Result` containing:
    - `global_templates`: `(K, 675)` cluster centroids sorted by trough-to-peak duration
    - `cluster_order`: `(K,)` trough-to-peak durations (ascending: fast-spiking to regular-spiking)
    - `sort_order`: `(K,)` permutation from GMM to t2p order
    - `pca_model`: fitted PCA for transforming new spikes
    - `gmm_means/covs/weights`: GPU GMM parameters (t2p-sorted) for soft posterior computation
  - `soft_assign(features, result) -> np.ndarray` shape `(N, K)` float32 soft posteriors, rows sum to 1
  - `compute_electrode_characterization(features, channels, clip_ids, trial_keys, result) -> tuple[dict, np.ndarray]` returning:
    - `per_clip`: `dict[(trial_key, clip_id) -> (8, 8, K)]` mean soft posterior per electrode per clip (dynamic)
    - `static`: `(8, 8, K)` mean soft posterior per electrode across all clips (static, chip-level marginal)
  - `compute_cooccurrence_matrix(features, channels, result) -> (64, K)` soft co-occurrence matrix. Each row is the mean posterior for that electrode. If all rows are near-identical, cluster identity adds nothing.

**Context:**
- PCA+GMM runs on training data only. The `run_stage0` function accepts a mask of which trial_keys belong to the training set.
- Feature vector per spike: central waveform (75 dims) + 8 neighbor trough amplitudes = 83 dims. **Relative-normalized** by the central waveform's peak-to-peak amplitude before PCA.
- **Relative P2P normalization**: divide all 83 feature dims by the central waveform's peak-to-peak amplitude. Preserves relative propagation direction (neighbor ratios), removes absolute distance-from-source.
- PCA reduces 83 -> `n_pca` (default 20) before GMM. This keeps GMM tractable and avoids curse of dimensionality.
- GMM with `n_clusters=256` components (validated by BIC sweep: K=256 is the visual knee, with 188.5 effective entries and only 17/256 truly dead). GPU GMM (PyTorch, full covariance, chunked E-step) required — sklearn CPU is intractable at K=256. Fit on 500k subsample, then compute full posteriors for all spikes.
- **Soft posteriors**: all cluster operations use `predict_proba`, never `predict`. Each spike contributes its full K-dim posterior to its electrode's characterization. This prevents the "100% dominant code" collapse seen with hard assignments at low K.
- Global template library: GMM centroids (inverse-transformed from PCA space). Sorted by trough-to-peak duration of the central waveform (fast-spiking to regular-spiking).
- Per-clip electrode characterization: mean posterior across all spikes on electrode (r,c) in that clip. Zero vector for silent electrodes.
- Static (chip-level) electrode characterization: mean posterior across ALL spikes per electrode (all clips). Analogous to the spatial map in HDMEA — the long-run electrode identity.
- **Sanity check**: soft co-occurrence matrix. If every electrode shows the same mean posterior (low row variance), cluster identity adds nothing per electrode.

- [ ] **Step 1: Write tests**

```python
# daps/tests/test_clustering.py
import numpy as np
import pytest


def test_stage0_template_ordering():
    """Templates should be sorted by trough-to-peak duration (ascending)."""
    from daps.waveform_clustering import run_stage0
    rng = np.random.default_rng(42)
    # Mock waveform database: 1000 spikes, 83 dims
    features = rng.standard_normal((1000, 83)).astype(np.float32)
    channels = rng.integers(0, 64, size=1000).astype(np.int32)
    clip_ids = rng.integers(0, 3, size=1000).astype(np.int32)
    train_mask = np.ones(1000, dtype=bool)

    result = run_stage0(features, channels, train_mask, n_pca=10, n_clusters=4)
    assert result.global_templates.shape == (4, 83)
    assert result.sort_order.shape == (4,)
    # Trough-to-peak durations should be monotonically non-decreasing
    t2p = result.cluster_order
    assert all(t2p[i] <= t2p[i + 1] for i in range(len(t2p) - 1))


def test_soft_assign_posteriors():
    """Soft posteriors should be (N, K) with rows summing to 1."""
    from daps.waveform_clustering import run_stage0, soft_assign
    rng = np.random.default_rng(42)
    features = rng.standard_normal((500, 83)).astype(np.float32)
    channels = rng.integers(0, 64, size=500).astype(np.int32)
    train_mask = np.ones(500, dtype=bool)

    result = run_stage0(features, channels, train_mask, n_pca=10, n_clusters=4)
    posteriors = soft_assign(features, result)
    assert posteriors.shape == (500, 4)
    assert posteriors.min() >= 0.0
    assert np.allclose(posteriors.sum(axis=1), 1.0, atol=1e-5)


def test_electrode_characterization_shape():
    """Per-clip electrode embed should be (8, 8, n_clusters)."""
    from daps.waveform_clustering import run_stage0, compute_electrode_characterization
    rng = np.random.default_rng(42)
    features = rng.standard_normal((500, 83)).astype(np.float32)
    channels = rng.integers(0, 64, size=500).astype(np.int32)
    clip_ids = rng.integers(0, 3, size=500).astype(np.int32)
    trial_keys = np.array(["trial_0"] * 500)
    train_mask = np.ones(500, dtype=bool)

    result = run_stage0(features, channels, train_mask, n_pca=10, n_clusters=4)
    per_clip, static = compute_electrode_characterization(
        features, channels, clip_ids, trial_keys, result
    )
    assert static.shape == (8, 8, 4)
    assert static.min() >= 0.0
    # Each active electrode's composition is a mean of posteriors (each sums to 1)
    active = static.sum(axis=-1) > 0
    assert np.allclose(static[active].sum(axis=-1), 1.0, atol=1e-4)


def test_cooccurrence_sanity_check():
    """Co-occurrence matrix should be (64, n_clusters)."""
    from daps.waveform_clustering import compute_cooccurrence_matrix, run_stage0
    rng = np.random.default_rng(42)
    features = rng.standard_normal((500, 83)).astype(np.float32)
    channels = rng.integers(0, 64, size=500).astype(np.int32)
    train_mask = np.ones(500, dtype=bool)

    result = run_stage0(features, channels, train_mask, n_pca=10, n_clusters=4)
    cooc = compute_cooccurrence_matrix(features, channels, result)
    assert cooc.shape == (64, 4)
    assert cooc.min() >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest daps/tests/test_clustering.py -v
```

- [ ] **Step 3: Implement Stage 0**

```python
# daps/waveform_clustering.py
"""Stage 0: Offline waveform clustering using relative-normalized 3x3 neighborhood features.

K=256 GMM on GPU (PyTorch), soft posteriors throughout. Electrodes are
sparse weighted mixtures over the codebook, never hard-assigned.
"""
import numpy as np
import torch
from dataclasses import dataclass
from sklearn.decomposition import PCA
from pathlib import Path


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Stage0Result:
    global_templates: np.ndarray      # (K, 83) cluster centroids (relative-normalized PCA space, inverse-transformed)
    cluster_order: np.ndarray         # (K,) trough-to-peak durations (ascending: fast to regular spiking)
    sort_order: np.ndarray            # (K,) permutation from GMM order to t2p-sorted order
    pca_model: PCA
    gmm_means: np.ndarray             # (K, n_pca) in PCA space, t2p-sorted
    gmm_covs: np.ndarray              # (K, n_pca, n_pca) full covariance, t2p-sorted
    gmm_weights: np.ndarray           # (K,) mixture weights, t2p-sorted
    n_clusters: int


def _relative_normalize(features: np.ndarray) -> np.ndarray:
    """Divide all 83 dims by the central waveform peak-to-peak amplitude.
    
    Preserves propagation direction (neighbor ratios) while removing
    absolute distance-from-source.
    """
    central = features[:, :75]  # central waveform
    p2p = central.max(axis=1) - central.min(axis=1)  # (N,)
    p2p = np.maximum(p2p, 1e-8)  # avoid div-by-zero for flat waveforms
    return features / p2p[:, None]


def _trough_to_peak_duration(waveform_75: np.ndarray, fs: int = 25000) -> float:
    trough_idx = int(np.argmin(waveform_75))
    if trough_idx >= len(waveform_75) - 1:
        return float(len(waveform_75)) / fs
    peak_idx = int(np.argmax(waveform_75[trough_idx:])) + trough_idx
    return (peak_idx - trough_idx) / fs


class GMMGpu:
    """Full-covariance GMM fitted on GPU with chunked E-step to avoid OOM."""

    def __init__(self, n_components, max_iter=300, n_init=3, tol=1e-4):
        self.K = n_components
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.means_ = None   # (K, D) on CPU after fit
        self.covs_ = None    # (K, D, D) on CPU after fit
        self.weights_ = None # (K,) on CPU after fit

    def fit(self, X_np: np.ndarray):
        """Fit GMM on GPU. X_np: (N, D) float32 numpy array."""
        X = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
        N, D = X.shape
        best_ll = -float("inf")
        chunk = max(1, int(4e9 / (self.K * D * 4)))

        for init_idx in range(self.n_init):
            means, covs, weights = self._kmpp_init(X)
            prev_ll = -float("inf")
            for it in range(self.max_iter):
                resp, ll = self._e_step(X, means, covs, weights, chunk)
                means, covs, weights = self._m_step(X, resp)
                if abs(ll - prev_ll) / max(abs(ll), 1.0) < self.tol:
                    break
                prev_ll = ll
            if ll > best_ll:
                best_ll = ll
                self.means_ = means.cpu().numpy()
                self.covs_ = covs.cpu().numpy()
                self.weights_ = weights.cpu().numpy()

    def _kmpp_init(self, X):
        """K-means++ initialization."""
        N, D = X.shape
        idx = [torch.randint(N, (1,), device=X.device).item()]
        min_dist = torch.full((N,), float("inf"), device=X.device)
        for _ in range(1, self.K):
            diff = X - X[idx[-1]].unsqueeze(0)
            d2 = (diff * diff).sum(dim=1)
            min_dist = torch.minimum(min_dist, d2)
            probs = min_dist / min_dist.sum()
            idx.append(torch.multinomial(probs, 1).item())
        means = X[idx].clone()
        covs = torch.eye(D, device=X.device).unsqueeze(0).expand(self.K, -1, -1).clone()
        weights = torch.ones(self.K, device=X.device) / self.K
        return means, covs, weights

    def _e_step(self, X, means, covs, weights, chunk):
        N, D = X.shape
        log_resp = torch.empty(N, self.K, device=X.device)
        precs = torch.linalg.inv(covs)
        log_dets = torch.linalg.slogdet(covs)[1]
        log_w = torch.log(weights)
        const = -0.5 * D * 1.8378770664093453  # log(2*pi)
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            Xc = X[start:end]
            diff = Xc.unsqueeze(1) - means.unsqueeze(0)
            mahal = torch.einsum("nkd,kde,nke->nk", diff, precs, diff)
            log_resp[start:end] = log_w + const - 0.5 * log_dets.unsqueeze(0) - 0.5 * mahal
        log_norm = torch.logsumexp(log_resp, dim=1, keepdim=True)
        resp = torch.exp(log_resp - log_norm)
        ll = log_norm.mean().item()
        return resp, ll

    def predict_proba(self, X_np: np.ndarray, chunk: int = 50000) -> np.ndarray:
        """Soft posteriors for new data. Returns (N, K) float32 numpy array."""
        X = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
        means = torch.tensor(self.means_, dtype=torch.float32, device=DEVICE)
        covs = torch.tensor(self.covs_, dtype=torch.float32, device=DEVICE)
        weights = torch.tensor(self.weights_, dtype=torch.float32, device=DEVICE)
        precs = torch.linalg.inv(covs)
        log_dets = torch.linalg.slogdet(covs)[1]
        log_w = torch.log(weights)
        N, D = X.shape
        const = -0.5 * D * 1.8378770664093453
        out = np.empty((N, self.K), dtype=np.float32)
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            Xc = X[start:end]
            diff = Xc.unsqueeze(1) - means.unsqueeze(0)
            mahal = torch.einsum("nkd,kde,nke->nk", diff, precs, diff)
            log_resp = log_w + const - 0.5 * log_dets.unsqueeze(0) - 0.5 * mahal
            log_norm = torch.logsumexp(log_resp, dim=1, keepdim=True)
            resp = torch.exp(log_resp - log_norm)
            out[start:end] = resp.cpu().numpy()
        return out

    def _m_step(self, X, resp):
        Nk = resp.sum(dim=0)
        Nk = torch.clamp(Nk, min=1e-8)
        weights = Nk / X.shape[0]
        means = (resp.T @ X) / Nk.unsqueeze(1)
        D = X.shape[1]
        covs = torch.zeros(self.K, D, D, device=X.device)
        for k in range(self.K):
            diff = X - means[k].unsqueeze(0)
            covs[k] = (diff * resp[:, k:k+1]).T @ diff / Nk[k]
            covs[k] += 1e-6 * torch.eye(D, device=X.device)
        return means, covs, weights


def run_stage0(
    features: np.ndarray,
    channels: np.ndarray,
    train_mask: np.ndarray,
    n_pca: int = 20,
    n_clusters: int = 256,
    n_subsample: int = 500_000,
    seed: int = 42,
) -> Stage0Result:
    """Fit PCA + GPU GMM on relative-normalized training spikes.

    Returns Stage0Result with t2p-sorted templates and GMM parameters.
    All subsequent operations use soft posteriors from the fitted model.
    """
    rng = np.random.default_rng(seed)
    train_feats = features[train_mask]

    # Relative P2P normalization
    train_feats = _relative_normalize(train_feats)

    pca = PCA(n_components=min(n_pca, train_feats.shape[1]), random_state=seed)
    train_pca = pca.fit_transform(train_feats).astype(np.float32)

    # Subsample for GMM fitting (K=256 with full cov needs manageable N)
    if train_pca.shape[0] > n_subsample:
        idx = rng.choice(train_pca.shape[0], n_subsample, replace=False)
        sub_pca = train_pca[idx]
    else:
        sub_pca = train_pca

    gmm = GMMGpu(n_components=n_clusters, max_iter=300, n_init=3)
    gmm.fit(sub_pca)

    # Recover templates in original 83-dim space
    templates_83 = pca.inverse_transform(gmm.means_)

    # Sort by trough-to-peak duration
    t2p = np.array([_trough_to_peak_duration(t[:75]) for t in templates_83])
    sort_order = np.argsort(t2p)
    templates_83 = templates_83[sort_order]
    t2p = t2p[sort_order]

    return Stage0Result(
        global_templates=templates_83.astype(np.float32),
        cluster_order=t2p.astype(np.float32),
        sort_order=sort_order,
        pca_model=pca,
        gmm_means=gmm.means_[sort_order],
        gmm_covs=gmm.covs_[sort_order],
        gmm_weights=gmm.weights_[sort_order],
        n_clusters=n_clusters,
    )


def soft_assign(
    features: np.ndarray,
    result: Stage0Result,
    chunk: int = 50000,
) -> np.ndarray:
    """Soft posterior assignment. Returns (N, K) float32 array, t2p-sorted."""
    features = _relative_normalize(features)
    pca_feats = result.pca_model.transform(features).astype(np.float32)
    # Build a temporary GMMGpu with the stored parameters
    gmm = GMMGpu(n_components=result.n_clusters)
    gmm.means_ = result.gmm_means
    gmm.covs_ = result.gmm_covs
    gmm.weights_ = result.gmm_weights
    return gmm.predict_proba(pca_feats, chunk=chunk)


def compute_electrode_characterization(
    features: np.ndarray,
    channels: np.ndarray,
    clip_ids: np.ndarray,
    trial_keys: np.ndarray,
    result: Stage0Result,
) -> tuple[dict, np.ndarray]:
    """Compute per-clip and static electrode characterization using soft posteriors.

    Each spike contributes its full posterior vector (not a hard label).
    Per-electrode composition = mean posterior across all spikes on that electrode.

    Returns:
        per_clip: dict[(trial_key, clip_id) -> (8, 8, K)] soft cluster composition
        static: (8, 8, K) chip-level marginal (average across all clips)
    """
    from daps.extract_data import _CH_TO_GRID
    posteriors = soft_assign(features, result)  # (N, K)
    K = result.n_clusters

    per_clip = {}
    all_weight = np.zeros((8, 8, K), dtype=np.float64)
    all_count = np.zeros((8, 8), dtype=np.float64)

    unique_clips = set(zip(trial_keys, clip_ids))
    for tk, ci in unique_clips:
        mask = (trial_keys == tk) & (clip_ids == ci)
        clip_weight = np.zeros((8, 8, K), dtype=np.float64)
        clip_count = np.zeros((8, 8), dtype=np.float64)
        for feat_idx in np.where(mask)[0]:
            ch = int(channels[feat_idx])
            r, c = _CH_TO_GRID[ch]
            clip_weight[r, c] += posteriors[feat_idx]
            clip_count[r, c] += 1

        # Normalize: mean posterior per electrode (already sums to 1 per spike)
        denom = np.maximum(clip_count[:, :, None], 1.0)
        clip_comp = (clip_weight / denom).astype(np.float32)
        per_clip[(tk, ci)] = clip_comp

        all_weight += clip_weight
        all_count += clip_count

    # Static: mean posterior across all spikes per electrode
    denom = np.maximum(all_count[:, :, None], 1.0)
    static = (all_weight / denom).astype(np.float32)

    return per_clip, static


def compute_cooccurrence_matrix(
    features: np.ndarray,
    channels: np.ndarray,
    result: Stage0Result,
) -> np.ndarray:
    """Sanity check: (64, K) soft co-occurrence matrix.

    Each row is the mean posterior for that electrode. If all rows are
    near-identical, cluster identity adds nothing per electrode.
    """
    posteriors = soft_assign(features, result)
    K = result.n_clusters
    weight = np.zeros((64, K), dtype=np.float64)
    count = np.zeros(64, dtype=np.float64)
    for i, (ch, post) in enumerate(zip(channels, posteriors)):
        weight[int(ch)] += post
        count[int(ch)] += 1
    denom = np.maximum(count[:, None], 1.0)
    return (weight / denom).astype(np.float32)
```

**NOTE:** All cluster operations use **soft posteriors** (predict_proba), never hard labels. Each spike contributes its full K=256 posterior to the electrode it landed on; electrode characterization is the mean posterior across its spikes. The relative P2P normalization divides all 83 feature dims by the central waveform's peak-to-peak amplitude, preserving propagation direction while removing absolute distance-from-source. GPU GMM uses chunked E-step to stay under 4GB VRAM at K=256 with 500k subsample.

- [ ] **Step 4: Run tests**

```bash
python -m pytest daps/tests/test_clustering.py -v
```

- [ ] **Step 5: Commit**

```bash
git add daps/waveform_clustering.py daps/tests/test_clustering.py
git commit -m "feat(daps): Stage 0 waveform clustering (PCA+GMM on 3x3 neighborhood features)"
```

---

### Task 3: DAPS Dataset

**Files:**
- Create: `daps/dataset.py`
- Create: `daps/tests/test_dataset.py`

**Interfaces:**
- Consumes: NPZ files from Task 1, soft electrode characterization from Task 2 (`per_clip` dict of (8,8,K) posteriors + `static` (8,8,K) marginal), `compute_activity_ctx` from `utils/recon.py:502`
- Produces: `DapsDataset(data_dir, npz_paths, electrode_embed_dir, ...)` — a PyTorch Dataset returning:

```python
{
    "x": (1, T, 8, 8),                    # binary spike raster, float32
    "electrode_embed": (8, 8, n_clusters), # per-clip cluster composition (dynamic)
    "electrode_embed_static": (8, 8, n_clusters),  # chip-level average (for generation)
    "local_ctx": (11,),                    # 9 original stats + latency + pulse_index
    "global_ctx_stim": (4,),               # stim_row, stim_col, current_norm, pw_norm
    "global_ctx_static": (C_static,),      # chip-level constant
    "pulse_index": int,
    "stim_channel": int,
    "is_baseline": bool,
}
```

**Context:**
- Two-stream input: `x` is binary, `electrode_embed_static` is the chip-level marginal electrode characterization.
- `electrode_embed_static`: loaded once per chip, shared across all clips. This is what the encoder/decoder sees — it's the electrode identity bias, never varies per clip. At generation time, this is always available (no fallback needed).
- `electrode_embed` (dynamic): varies per clip. Routed to gct ONLY, never to the encoder. For training clips, this is the mean soft posterior over that clip's spikes. For baseline clips, this equals the static embed. At generation time, pass static to gct as E[dynamic] — the natural marginal.
- `local_ctx`: 9 original stats from `compute_activity_ctx(thw)` + response_latency (time bin of first post-stim spike, normalized to [0,1]; 0 for baseline) + pulse_index (0/1/2, normalized to [0,1]). Total: 11 dims.
- `global_ctx_stim`: `(stim_row, stim_col, current_normalized, pulse_width_normalized)`. Baseline: `(4/7, 4/7, 0, 0)` (grid center, zero intensity).
- `global_ctx_static`: constant vector per chip (placeholder for DIV, cell line encoding). For single-chip data, a ones vector of dim 4.

- [ ] **Step 1: Write test for dataset output shapes and types**

```python
# daps/tests/test_dataset.py
import numpy as np
import torch
import pytest
from pathlib import Path


def test_dataset_shapes(tmp_path):
    """DapsDataset returns correctly shaped tensors."""
    from daps.dataset import DapsDataset
    _create_mock_data(tmp_path, n_clusters=4)
    ds = DapsDataset(
        npz_paths=[str(tmp_path / "ch13_cur2250_pw120.npz")],
        electrode_embed_dir=str(tmp_path / "electrode_embeds"),
    )
    assert len(ds) == 3  # 3 pulses per trial
    sample = ds[0]
    assert sample["x"].shape == (1, 80, 8, 8)
    assert sample["x"].dtype == torch.float32
    assert sample["electrode_embed"].shape == (8, 8, 4)
    assert sample["electrode_embed_static"].shape == (8, 8, 4)
    assert sample["local_ctx"].shape == (11,)
    assert sample["global_ctx_stim"].shape == (4,)
    assert sample["global_ctx_static"].shape == (4,)
    assert isinstance(sample["pulse_index"], int)
    assert isinstance(sample["is_baseline"], bool)


def test_dataset_baseline_clip(tmp_path):
    """Baseline clips have zero stim params and is_baseline=True."""
    from daps.dataset import DapsDataset
    _create_mock_data(tmp_path, n_clusters=4, baseline=True)
    ds = DapsDataset(
        npz_paths=[str(tmp_path / "baseline_pre.npz")],
        electrode_embed_dir=str(tmp_path / "electrode_embeds"),
    )
    sample = ds[0]
    assert sample["is_baseline"] is True
    assert sample["global_ctx_stim"][2] == 0.0
    assert sample["global_ctx_stim"][3] == 0.0


def test_local_ctx_has_11_dims(tmp_path):
    """local_ctx dimensions 9 and 10 are response_latency and pulse_index."""
    from daps.dataset import DapsDataset
    _create_mock_data(tmp_path, n_clusters=4)
    ds = DapsDataset(
        npz_paths=[str(tmp_path / "ch13_cur2250_pw120.npz")],
        electrode_embed_dir=str(tmp_path / "electrode_embeds"),
    )
    sample_0 = ds[0]
    sample_2 = ds[2]
    assert sample_0["local_ctx"].shape == (11,)
    # Pulse index should differ
    assert sample_0["local_ctx"][10] != sample_2["local_ctx"][10]


def _create_mock_data(tmp_path, n_clusters, baseline=False):
    rng = np.random.default_rng(42)
    embed_dir = tmp_path / "electrode_embeds"
    embed_dir.mkdir()

    # Static electrode embed
    static = rng.dirichlet(np.ones(n_clusters), size=(8, 8)).astype(np.float32)
    np.save(str(embed_dir / "static_embed.npy"), static)

    if baseline:
        clips = rng.choice([0, 1], size=(5, 80, 8, 8), p=[0.95, 0.05]).astype(np.uint8)
        np.savez_compressed(str(tmp_path / "baseline_pre.npz"),
                            clips=clips, stim_channel=-1, current_nA=0, pulse_width_us=0)
        # Baseline per-clip embeds: just use static
        for ci in range(5):
            np.save(str(embed_dir / f"baseline_pre_clip{ci}.npy"), static)
    else:
        clips = rng.choice([0, 1], size=(3, 80, 8, 8), p=[0.95, 0.05]).astype(np.uint8)
        np.savez_compressed(str(tmp_path / "ch13_cur2250_pw120.npz"),
                            clips=clips, stim_channel=13, current_nA=2250, pulse_width_us=120)
        for ci in range(3):
            per_clip = rng.dirichlet(np.ones(n_clusters), size=(8, 8)).astype(np.float32)
            np.save(str(embed_dir / f"ch13_cur2250_pw120_clip{ci}.npy"), per_clip)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest daps/tests/test_dataset.py -v
```

- [ ] **Step 3: Implement DapsDataset**

```python
# daps/dataset.py
"""PyTorch Dataset for DAPS binary spike rasters with two-stream input."""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

CURRENT_RANGE = (500, 3750)
PW_RANGE = (40, 200)


def _normalize(val, lo, hi):
    if hi == lo:
        return 0.0
    return (val - lo) / (hi - lo)


def _compute_response_latency(clip_thw: np.ndarray) -> float:
    """First time bin with any spike, normalized to [0, 1]."""
    any_spike_per_t = clip_thw.reshape(clip_thw.shape[0], -1).any(axis=1)
    nz = np.nonzero(any_spike_per_t)[0]
    if len(nz) == 0:
        return 0.0
    return nz[0] / clip_thw.shape[0]


class DapsDataset(Dataset):
    def __init__(
        self,
        npz_paths: list[str],
        electrode_embed_dir: str,
        static_ctx_dim: int = 4,
    ):
        self.embed_dir = Path(electrode_embed_dir)
        self.static_embed = torch.from_numpy(
            np.load(str(self.embed_dir / "static_embed.npy"))
        ).float()
        self.static_ctx = torch.ones(static_ctx_dim, dtype=torch.float32)

        # Build flat index: (npz_path, pulse_index, stim_params, embed_key)
        self.index = []
        for p in npz_paths:
            d = np.load(p)
            clips = d["clips"]
            stim_ch = int(d["stim_channel"])
            cur = int(d["current_nA"])
            pw = int(d["pulse_width_us"])
            is_baseline = stim_ch < 0
            stem = Path(p).stem  # e.g. "ch13_cur2250_pw120"
            for pi in range(clips.shape[0]):
                self.index.append({
                    "path": p,
                    "pulse_index": pi,
                    "stim_channel": stim_ch,
                    "current_nA": cur,
                    "pulse_width_us": pw,
                    "is_baseline": is_baseline,
                    "embed_key": f"{stem}_clip{pi}",
                })

        from utils.recon import compute_activity_ctx
        self._compute_activity_ctx = compute_activity_ctx

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index[idx]
        d = np.load(entry["path"])
        clip = d["clips"][entry["pulse_index"]]  # (80, 8, 8)

        x = torch.from_numpy(clip).float().unsqueeze(0)  # (1, T, H, W)

        # Per-clip electrode embedding (dynamic)
        embed_path = self.embed_dir / f"{entry['embed_key']}.npy"
        if embed_path.exists():
            electrode_embed = torch.from_numpy(np.load(str(embed_path))).float()
        else:
            electrode_embed = self.static_embed.clone()

        # lct: 9 original + latency + pulse_index
        base_ctx = self._compute_activity_ctx(clip)
        latency = _compute_response_latency(clip)
        pulse_norm = entry["pulse_index"] / 2.0
        local_ctx = torch.tensor(
            list(base_ctx) + [latency, pulse_norm],
            dtype=torch.float32,
        )

        # gct stim params
        if entry["is_baseline"]:
            stim_row, stim_col = 4.0 / 7.0, 4.0 / 7.0
            cur_norm, pw_norm = 0.0, 0.0
        else:
            from daps.extract_data import channel_to_grid
            r, c = channel_to_grid(entry["stim_channel"])
            stim_row = r / 7.0
            stim_col = c / 7.0
            cur_norm = _normalize(entry["current_nA"], *CURRENT_RANGE)
            pw_norm = _normalize(entry["pulse_width_us"], *PW_RANGE)

        global_ctx_stim = torch.tensor(
            [stim_row, stim_col, cur_norm, pw_norm], dtype=torch.float32,
        )

        return {
            "x": x,
            "electrode_embed": electrode_embed,
            "electrode_embed_static": self.static_embed,
            "local_ctx": local_ctx,
            "global_ctx_stim": global_ctx_stim,
            "global_ctx_static": self.static_ctx,
            "pulse_index": entry["pulse_index"],
            "stim_channel": entry["stim_channel"],
            "is_baseline": entry["is_baseline"],
        }
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest daps/tests/test_dataset.py -v
```

- [ ] **Step 5: Commit**

```bash
git add daps/dataset.py daps/tests/test_dataset.py
git commit -m "feat(daps): DapsDataset with two-stream input and 11-dim lct"
```

---

### Task 4: HybridGCT with Electrode Characterization + Banks

**Files:**
- Create: `daps/gct.py`
- Create: `daps/tests/test_gct.py`

**Interfaces:**
- Consumes: `global_ctx_stim (B, 4)`, `electrode_embed (B, 8, 8, n_clusters)` (dynamic, per-clip — or static at generation time), adjacency rates from `DapsAdjacencyBank`, `is_baseline (B,)` from DapsDataset.
- Produces:
  - `HybridGCT(stim_dim=4, electrode_char_dim=8*8*128, adj_dim=7, out_dim=32)` module. Forward: `MLP(concat(stim_embed, elec_char_flat_dynamic, adj_flat)) -> (B, 32)`.
  - Three Stage 1 pretraining heads:
    1. **Dynamic electrode embed prediction**: predict per-clip dynamic posterior from gct (main target, analog of HDMEA site-map prediction)
    2. **Adjacency rate prediction**: predict temporal autocorrelation rates from gct
    3. **Linear stim param decoder**: reconstruct stim params from gct (invertibility constraint, same role as HDMEA spatial map head)
  - Loss weights for the 3 heads: swept (uncertainty weighting or grid search)
  - `DapsStimResponseBank` — EMA soft support map keyed by gct(stim_params), stores activation map `(8, 8)` per stim config. Analog of `GlobalContextSpatialBank`.
  - `DapsAdjacencyBank` — temporal autocorrelation at binned gaps, keyed by gct embedding with rounding. Analog of `GlobalContextAdjacencyBank`. Same DEFAULT_GAP_BINS as HDMEA.

**Context:**
- Electrode characterization enters gct as a flattened DYNAMIC per-clip vector: `electrode_char_flat = electrode_embed.reshape(B, -1)` — shape `(B, 8*8*128)`. Concatenated with stim params and adjacency rates before MLP. During training, this is the mean soft posterior for that clip; at generation time, pass the static marginal.
- HybridGCT input: `concat(stim_params, electrode_char_flat_dynamic, adj_flat)` — the dynamic characterization makes gct both clip-aware and chip-aware.
- Output is L2-normalized to match the existing codebook convention.
- Stage 1 pretraining: 3 heads predict (1) dynamic electrode embed, (2) adjacency rates, (3) stim params from the 32-d gct. Loss weights swept. Stim param decoder is handled same as HDMEA spatial map head (kept during Stage 1, behavior after Stage 1 matches HDMEA).
- `DapsStimResponseBank`: identical API to `GlobalContextSpatialBank` in `model/spatial_map.py:19`, but stores `(8, 8)` maps instead of `(H, W)` pixel maps. Keyed by a rounded tuple of the stim-params portion of gct (not the full gct including electrode char, since the bank should generalize across chips for same stim config).
- `DapsAdjacencyBank`: identical API to `GlobalContextAdjacencyBank` in `model/spatial_map.py:94`. Keyed by gct embedding with rounding (same `_key()` mechanism). Same DEFAULT_GAP_BINS: `(1,1), (2,2), (3,3), (4,6), (7,12), (13,24), (25,48)`. Per stim-condition, not per clip.

- [ ] **Step 1: Write tests**

```python
# daps/tests/test_gct.py
import torch
import pytest


def test_hybrid_gct_shape():
    from daps.gct import HybridGCT
    model = HybridGCT(stim_dim=4, static_dim=4, electrode_char_dim=8*8*8, out_dim=32)
    stim = torch.randn(8, 4)
    static = torch.ones(8, 4)
    elec_char = torch.randn(8, 8*8*8)
    is_baseline = torch.tensor([False]*4 + [True]*4)
    out = model(stim, static, elec_char, is_baseline)
    assert out.shape == (8, 32)


def test_hybrid_gct_l2_normalized():
    from daps.gct import HybridGCT
    model = HybridGCT(stim_dim=4, static_dim=4, electrode_char_dim=512, out_dim=32)
    stim = torch.randn(4, 4)
    static = torch.ones(4, 4)
    elec_char = torch.randn(4, 512)
    is_baseline = torch.tensor([False]*4)
    out = model(stim, static, elec_char, is_baseline)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_stim_response_bank_update_and_get():
    from daps.gct import DapsStimResponseBank
    bank = DapsStimResponseBank()
    gct = torch.randn(2, 32)
    target = torch.rand(2, 8, 8)
    bank.update(gct, target)
    retrieved = bank.get(gct)
    assert retrieved.shape == (2, 8, 8)


def test_adjacency_bank_update_and_get():
    from daps.gct import DapsAdjacencyBank
    bank = DapsAdjacencyBank()
    gct = torch.randn(2, 32)
    x = torch.randint(0, 2, (2, 1, 80, 8, 8)).float()
    bank.update_from_x(gct, x)
    rates = bank.get(gct)
    assert rates.shape[0] == 2
    assert rates.shape[1] == bank.num_bins
```

- [ ] **Step 2: Run tests, verify fail**

```bash
python -m pytest daps/tests/test_gct.py -v
```

- [ ] **Step 3: Implement HybridGCT and banks**

```python
# daps/gct.py
"""Hybrid global context with electrode characterization + banks for DAPS."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.spatial_map import GlobalContextSpatialBank, GlobalContextAdjacencyBank


class HybridGCT(nn.Module):
    def __init__(
        self,
        stim_dim: int = 4,
        static_dim: int = 4,
        electrode_char_dim: int = 512,  # 8*8*n_clusters
        out_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.out_dim = out_dim

        combined_in = stim_dim + electrode_char_dim
        self.f_dynamic = nn.Sequential(
            nn.Linear(combined_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.g_net = nn.Linear(stim_dim, 1)
        self.f_static = nn.Linear(static_dim, out_dim)
        self.gct_spontaneous = nn.Parameter(torch.randn(out_dim) * 0.02)

    def _compute_gate(self, stim: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.g_net(stim))

    def forward(
        self,
        stim: torch.Tensor,
        static: torch.Tensor,
        electrode_char_flat: torch.Tensor,
        is_baseline: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            stim: (B, stim_dim)
            static: (B, static_dim)
            electrode_char_flat: (B, electrode_char_dim) — flattened static electrode embed
            is_baseline: (B,) bool

        Returns:
            (B, out_dim) L2-normalized global context vector
        """
        gate = self._compute_gate(stim)
        dynamic_in = torch.cat([stim, electrode_char_flat], dim=-1)
        dynamic = self.f_dynamic(dynamic_in)
        spontaneous = self.gct_spontaneous.unsqueeze(0).expand(stim.shape[0], -1)
        static_emb = self.f_static(static)

        combined = gate * dynamic + (1 - gate) * spontaneous + static_emb
        return F.normalize(combined, dim=-1)


class DapsStimResponseBank(GlobalContextSpatialBank):
    """EMA soft support map per stim config, storing (8, 8) activation maps."""
    pass


class DapsAdjacencyBank(GlobalContextAdjacencyBank):
    """Temporal autocorrelation at binned gaps, keyed by gct."""
    pass
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest daps/tests/test_gct.py -v
```

- [ ] **Step 5: Commit**

```bash
git add daps/gct.py daps/tests/test_gct.py
git commit -m "feat(daps): HybridGCT with electrode characterization + StimResponse/Adjacency banks"
```

---

### Task 5: Model Adapter (Additive Electrode Embedding, No FiLM)

**Files:**
- Create: `daps/model_adapter.py`
- Create: `daps/tests/test_model.py`

**Interfaces:**
- Consumes: `TransformerVQVAE` from `model/vqvae.py:35`, `HybridGCT` from Task 4, `CtxEmbed` from `model/base.py:37`, `LctMapper` from `model/base.py:64`, `ActivePatchEmbed3D` from `model/base.py:1046`
- Produces: `build_daps_model(config) -> DapsTransformerVQVAE` — a subclass of TransformerVQVAE with:
  - `img_size=(80, 8, 8)`, `patch_size=(20, 2, 2)` → 64 tokens
  - `global_ctx_in_dim=32` (HybridGCT output)
  - `local_ctx_in_dim=11`
  - Additive electrode embedding at token level (no FiLM)
  - `ElectrodeEmbedProjector`: `nn.Linear(n_clusters, embed_dim)` that projects per-electrode cluster composition `(B, 8, 8, n_clusters)` to `(B, 8, 8, embed_dim)`, then pools to patch resolution `(B, h_tok, w_tok, embed_dim)` and adds to both encoder and decoder tokens.

**Context:**
- In the encoder: `active_patch_token = patch_embed(content) + pos_embed(h,w,t) + electrode_embed_static(h,w)`. The `electrode_embed_static(h,w)` is the projected STATIC (chip-level marginal) cluster composition for the 2x2 patch at position (h,w), averaged over the 4 electrodes in that patch. The encoder is context-agnostic — it sees the same electrode identity bias regardless of which clip is being processed.
- In the decoder: `blank_token + pos_embed(h,w,t) + electrode_embed_static(h,w)` for blank positions, `encoder_output + pos_embed(h,w,t) + electrode_embed_static(h,w)` for active positions.
- The electrode embedding is additive (not FiLM). This is simpler and compatible with the sparse encoder.
- No `ElectrodeCtxLoss` — the old plan's auxiliary loss predicting electrode embeddings from decoder output is dropped. The electrode embedding enters as input, not as a prediction target.
- The dynamic per-clip characterization does NOT enter the encoder/decoder — it goes to gct only (Task 4). This separation means the encoder learns a chip-invariant representation while gct handles clip-specific variation.
- The `DapsConvStem` from Task 6 replaces the standard conv stem. No FiLM is attached to it.

- [ ] **Step 1: Write tests**

```python
# daps/tests/test_model.py
import torch
import pytest


def test_daps_model_forward_shape():
    from daps.model_adapter import build_daps_model
    model = build_daps_model(n_clusters=4)
    x = torch.zeros(2, 1, 80, 8, 8)
    x[0, 0, 10, 3, 3] = 1.0
    x[1, 0, 40, 5, 5] = 1.0
    global_ctx = torch.randn(2, 32)
    electrode_embed_static = torch.randn(2, 8, 8, 4)
    out = model(x, global_ctx=global_ctx, electrode_embed_static=electrode_embed_static)
    assert "logits" in out or hasattr(out, "logits_vol_raw")


def test_daps_model_patch_count():
    from daps.model_adapter import build_daps_model
    model = build_daps_model(n_clusters=4)
    # (80/20) * (8/2) * (8/2) = 4 * 4 * 4 = 64 patches
    assert model.num_patches == 64


def test_electrode_embed_changes_output():
    """Different static electrode embeddings should produce different outputs."""
    from daps.model_adapter import build_daps_model
    model = build_daps_model(n_clusters=4)
    model.eval()
    x = torch.zeros(1, 1, 80, 8, 8)
    x[0, 0, 10, 3, 3] = 1.0
    global_ctx = torch.randn(1, 32)
    e1 = torch.randn(1, 8, 8, 4)
    e2 = torch.randn(1, 8, 8, 4) * 5
    with torch.no_grad():
        out1 = model(x, global_ctx=global_ctx, electrode_embed_static=e1)
        out2 = model(x, global_ctx=global_ctx, electrode_embed_static=e2)
    logits1 = out1.logits_vol_raw if hasattr(out1, "logits_vol_raw") else out1["logits"]
    logits2 = out2.logits_vol_raw if hasattr(out2, "logits_vol_raw") else out2["logits"]
    assert not torch.allclose(logits1, logits2, atol=1e-3)


def test_electrode_embed_projector_shape():
    from daps.model_adapter import ElectrodeEmbedProjector
    proj = ElectrodeEmbedProjector(n_clusters=256, embed_dim=64, patch_h=2, patch_w=2)
    electrode_embed = torch.randn(2, 8, 8, 8)
    out = proj(electrode_embed)
    # After pooling 2x2 patches: (B, 4, 4, embed_dim)
    assert out.shape == (2, 4, 4, 64)
```

- [ ] **Step 2: Run tests, verify fail**

```bash
python -m pytest daps/tests/test_model.py -v
```

- [ ] **Step 3: Implement model adapter**

```python
# daps/model_adapter.py
"""Adapt TransformerVQVAE for DAPS 8x8 MEA data with additive electrode embedding."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.vqvae import TransformerVQVAE


class ElectrodeEmbedProjector(nn.Module):
    """Project per-electrode cluster composition to patch-level token embeddings."""

    def __init__(self, n_clusters: int, embed_dim: int, patch_h: int = 2, patch_w: int = 2):
        super().__init__()
        self.proj = nn.Linear(n_clusters, embed_dim)
        self.patch_h = patch_h
        self.patch_w = patch_w

    def forward(self, electrode_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            electrode_embed: (B, H, W, n_clusters) per-electrode cluster composition

        Returns:
            (B, H//pH, W//pW, embed_dim) patch-level electrode embedding
        """
        B, H, W, K = electrode_embed.shape
        projected = self.proj(electrode_embed)  # (B, H, W, embed_dim)
        # Average pool to patch resolution
        pH, pW = self.patch_h, self.patch_w
        h_tok, w_tok = H // pH, W // pW
        projected = projected.reshape(B, h_tok, pH, w_tok, pW, -1)
        projected = projected.mean(dim=(2, 4))  # (B, h_tok, w_tok, embed_dim)
        return projected


def build_daps_model(
    n_clusters: int = 8,
    encoder_embed_dim: int = 64,
    encoder_depth: int = 3,
    code_dim: int = 64,
    num_codes: tuple = (32, 8, 4),
    decoder_embed_dim: int = 64,
    decoder_depth: int = 3,
    global_emb_dim: int = 32,
    local_emb_dim: int = 16,
) -> TransformerVQVAE:
    """
    Construct a TransformerVQVAE configured for DAPS 8x8 data.

    Attaches an ElectrodeEmbedProjector for additive token-level electrode embedding.
    """
    model = TransformerVQVAE(
        img_size=(80, 8, 8),
        patch_size=(20, 2, 2),
        encoder_embed_dim=encoder_embed_dim,
        encoder_depth=encoder_depth,
        code_dim=code_dim,
        num_codes=num_codes,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        global_ctx_in_dim=32,
        global_emb_dim=global_emb_dim,
        local_ctx_in_dim=11,
        local_emb_dim=local_emb_dim,
    )

    model.electrode_proj = ElectrodeEmbedProjector(
        n_clusters=n_clusters,
        embed_dim=encoder_embed_dim,
        patch_h=2,
        patch_w=2,
    )

    # Store original forward and wrap it
    _orig_forward = model.forward

    def _forward_with_electrode(self_ref, x, global_ctx=None, electrode_embed_static=None, **kwargs):
        # The STATIC electrode embedding is added inside the encoder/decoder
        # by hooking into the token pipeline. Dynamic per-clip embed goes to gct only.
        self_ref._current_electrode_embed = electrode_embed_static
        out = _orig_forward(x, global_ctx=global_ctx, **kwargs)
        self_ref._current_electrode_embed = None
        return out

    import types
    model.forward = types.MethodType(_forward_with_electrode, model)

    return model
```

**NOTE:** The forward-wrapping approach above is a simplified sketch. The actual integration depends on where `ActivePatchEmbed3D` and the decoder produce tokens. The implementor must:
1. Read `model/vqvae.py` forward method to find where encoder tokens are formed (after `ActivePatchEmbed3D`)
2. Add `+ electrode_proj(electrode_embed)` reshaped to `(B, N, D)` at that point
3. Do the same in the decoder path where blank_token + pos_embed is computed
4. The electrode embedding at `(h_tok, w_tok)` must be broadcast across temporal token positions (4 temporal patches share the same spatial electrode embed)

- [ ] **Step 4: Run tests**

```bash
python -m pytest daps/tests/test_model.py -v
```

- [ ] **Step 5: Commit**

```bash
git add daps/model_adapter.py daps/tests/test_model.py
git commit -m "feat(daps): model adapter with additive electrode embedding (no FiLM)"
```

---

### Task 6: Conv Stem for 8x8 (Single 3×3×K 3D Conv)

**Files:**
- Create: `daps/conv_stem.py`
- Create: `daps/tests/test_stem.py`

**Interfaces:**
- Consumes: input volume `(B, 1, T, 8, 8)` — binary spike raster
- Produces: `DapsConvStem(in_chans=1, out_chans=1, kernel_size=(K, 3, 3))` — single 3D conv layer with spatial 3×3 (matching the (1,1) tolerance level) and temporal K (slightly longer, flexible). Output is `(B, out_chans, T, 8, 8)` with same spatial size (padding preserves shape). Conv3d -> BatchNorm3d -> GELU.

**Context:**
- Single layer, NOT the 4-layer deep stem from v3. The sparse transformer encoder handles global spatial mixing via attention. The conv stem's job is only local feature preparation for VQ — matching the 3×3 neighborhood tolerance structure.
- Temporal kernel K can be slightly larger than spatial (e.g., K=5) since temporal dimension is longer (80 bins). Exact K is flexible.
- The stem does NOT downsample spatially. Output is same (T, 8, 8) shape.
- `out_chans=1` keeps input to `ActivePatchEmbed3D` as `(B, 1, T, H, W)` — patch dimension is `1 * 20 * 2 * 2 = 80`.
- No FiLM is attached to this stem. Electrode information enters via additive embedding at the token level (Task 5), not via spatial modulation in the stem.

- [ ] **Step 1: Write tests**

```python
# daps/tests/test_stem.py
import torch
import pytest


def test_stem_output_shape():
    from daps.conv_stem import DapsConvStem
    stem = DapsConvStem(in_chans=1, out_chans=1, n_layers=4)
    x = torch.randn(2, 1, 80, 8, 8)
    out = stem(x)
    assert out.shape == (2, 1, 80, 8, 8)


def test_stem_receptive_field():
    """After 4 layers of 3x3 spatial conv, RF covers entire 8x8 grid."""
    from daps.conv_stem import DapsConvStem
    stem = DapsConvStem(in_chans=1, out_chans=1, n_layers=4)
    x = torch.zeros(1, 1, 80, 8, 8)
    x[0, 0, 40, 0, 0] = 1.0
    x.requires_grad_(True)
    out = stem(x)
    loss = out[0, 0, 40, 7, 7].sum()
    loss.backward()
    assert x.grad[0, 0, 40, 0, 0].abs() > 0, "No gradient from (0,0) to (7,7)"


def test_stem_preserves_sparsity_pattern():
    """Mostly-zero input should produce mostly-small output."""
    from daps.conv_stem import DapsConvStem
    stem = DapsConvStem(in_chans=1, out_chans=1, n_layers=4)
    stem.eval()
    x = torch.zeros(1, 1, 80, 8, 8)
    x[0, 0, 10, 3, 3] = 1.0
    with torch.no_grad():
        out = stem(x)
    assert out.abs().mean() < 1.0, "Stem output is exploding"
```

- [ ] **Step 2: Run tests, verify fail**

```bash
python -m pytest daps/tests/test_stem.py -v
```

- [ ] **Step 3: Implement**

```python
# daps/conv_stem.py
"""Deeper 3D convolutional stem for 8x8 spatial pre-mixing."""
import torch
import torch.nn as nn


class DapsConvStem(nn.Module):
    """
    Multi-layer 3D conv stem with spatial-only kernels.

    Kernel (1, 3, 3): no temporal mixing, spatial 3x3 with padding=1.
    After n_layers=4, receptive field is (1, 9, 9) — covers entire 8x8 grid.
    """

    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 1,
        n_layers: int = 4,
        hidden_chans: int = 16,
        kernel_size: int = 3,
    ):
        super().__init__()
        padding = kernel_size // 2
        layers = []
        for i in range(n_layers):
            c_in = in_chans if i == 0 else hidden_chans
            c_out = out_chans if i == n_layers - 1 else hidden_chans
            layers.extend([
                nn.Conv3d(c_in, c_out,
                          kernel_size=(1, kernel_size, kernel_size),
                          padding=(0, padding, padding), bias=False),
                nn.BatchNorm3d(c_out),
            ])
            if i < n_layers - 1:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest daps/tests/test_stem.py -v
```

- [ ] **Step 5: Commit**

```bash
git add daps/conv_stem.py daps/tests/test_stem.py
git commit -m "feat(daps): deeper conv stem with full 8x8 receptive field"
```

---

### Task 7: Train/Test Splits

**Files:**
- Create: `daps/splits.py`
- Create: `daps/tests/test_splits.py`

**Interfaces:**
- Consumes: list of NPZ paths from Task 1
- Produces: `make_channel_holdout_split(data_dir, n_test_channels=5, n_val_channels=4, seed=42) -> dict` saved as JSON. Returns `{"train": [...], "val": [...], "test": [...]}` lists of NPZ filenames. Also `make_channel_and_current_holdout_split(...)` for harder ablation.

**Context:**
- Primary split: hold out 5 stim channels for test, 4 for val. Remaining 50 for train. All current/pulse_width combos of a held-out channel go into that split.
- Ablation split: additionally hold out 3 current levels from training set. Test set includes both unseen channels AND unseen currents.
- Baseline clips go into all splits.
- The split is deterministic given the seed.
- Stage 0 clustering runs only on spikes from training-split trials — pass the training trial keys to `run_stage0` via its `train_mask`.

- [ ] **Step 1: Write tests**

```python
# daps/tests/test_splits.py
import numpy as np
import pytest
from pathlib import Path


def test_channel_holdout_no_leak(tmp_path):
    from daps.splits import make_channel_holdout_split
    _create_mock_npzs(tmp_path, n_channels=20, n_currents=3, n_pws=2)
    split = make_channel_holdout_split(str(tmp_path), n_test_channels=3, n_val_channels=2)

    train_channels = _extract_channels(split["train"])
    val_channels = _extract_channels(split["val"])
    test_channels = _extract_channels(split["test"])

    assert train_channels.isdisjoint(test_channels)
    assert train_channels.isdisjoint(val_channels)
    assert val_channels.isdisjoint(test_channels)


def test_channel_holdout_counts(tmp_path):
    from daps.splits import make_channel_holdout_split
    _create_mock_npzs(tmp_path, n_channels=20, n_currents=3, n_pws=2)
    split = make_channel_holdout_split(str(tmp_path), n_test_channels=3, n_val_channels=2)
    assert len(_extract_channels(split["test"])) == 3
    assert len(_extract_channels(split["val"])) == 2


def test_baseline_in_all_splits(tmp_path):
    from daps.splits import make_channel_holdout_split
    _create_mock_npzs(tmp_path, n_channels=10, n_currents=2, n_pws=1)
    np.savez_compressed(str(tmp_path / "baseline_pre.npz"),
                        clips=np.zeros((5, 80, 8, 8), dtype=np.uint8),
                        stim_channel=-1, current_nA=0, pulse_width_us=0)
    split = make_channel_holdout_split(str(tmp_path), n_test_channels=2, n_val_channels=1)
    for subset in ["train", "val", "test"]:
        baselines = [p for p in split[subset] if "baseline" in p]
        assert len(baselines) > 0, f"No baseline in {subset}"


def _create_mock_npzs(tmp_path, n_channels, n_currents, n_pws):
    rng = np.random.default_rng(42)
    for ch in range(n_channels):
        for cur_idx in range(n_currents):
            for pw_idx in range(n_pws):
                cur = 500 + cur_idx * 250
                pw = 40 + pw_idx * 40
                clips = rng.choice([0, 1], size=(3, 80, 8, 8), p=[0.95, 0.05]).astype(np.uint8)
                np.savez_compressed(str(tmp_path / f"ch{ch:02d}_cur{cur}_pw{pw}.npz"),
                                    clips=clips, stim_channel=ch, current_nA=cur, pulse_width_us=pw)


def _extract_channels(paths):
    import re
    channels = set()
    for p in paths:
        m = re.search(r'ch(\d+)_', Path(p).name)
        if m:
            channels.add(int(m.group(1)))
    return channels
```

- [ ] **Step 2: Run tests, verify fail**

```bash
python -m pytest daps/tests/test_splits.py -v
```

- [ ] **Step 3: Implement**

```python
# daps/splits.py
"""Train/val/test split strategies for DAPS data."""
import json
import re
import numpy as np
from pathlib import Path


def _parse_stim_channel(filename: str) -> int | None:
    m = re.search(r'ch(\d+)_', filename)
    return int(m.group(1)) if m else None


def _parse_current(filename: str) -> int | None:
    m = re.search(r'cur(\d+)', filename)
    return int(m.group(1)) if m else None


def make_channel_holdout_split(
    data_dir: str,
    n_test_channels: int = 5,
    n_val_channels: int = 4,
    seed: int = 42,
) -> dict:
    data_dir = Path(data_dir)
    all_npz = sorted(data_dir.glob("*.npz"))

    baselines = [str(p) for p in all_npz if "baseline" in p.name]
    trials = [p for p in all_npz if "baseline" not in p.name]

    channels = sorted(set(_parse_stim_channel(p.name) for p in trials) - {None})
    rng = np.random.default_rng(seed)
    rng.shuffle(channels)

    test_ch = set(channels[:n_test_channels])
    val_ch = set(channels[n_test_channels:n_test_channels + n_val_channels])

    split = {"train": list(baselines), "val": list(baselines), "test": list(baselines)}
    for p in trials:
        ch = _parse_stim_channel(p.name)
        if ch in test_ch:
            split["test"].append(str(p))
        elif ch in val_ch:
            split["val"].append(str(p))
        else:
            split["train"].append(str(p))

    return split


def make_channel_and_current_holdout_split(
    data_dir: str,
    n_test_channels: int = 5,
    n_val_channels: int = 4,
    holdout_currents: tuple[int, ...] = (1000, 2000, 3000),
    seed: int = 42,
) -> dict:
    data_dir = Path(data_dir)
    all_npz = sorted(data_dir.glob("*.npz"))

    baselines = [str(p) for p in all_npz if "baseline" in p.name]
    trials = [p for p in all_npz if "baseline" not in p.name]

    channels = sorted(set(_parse_stim_channel(p.name) for p in trials) - {None})
    rng = np.random.default_rng(seed)
    rng.shuffle(channels)

    test_ch = set(channels[:n_test_channels])
    val_ch = set(channels[n_test_channels:n_test_channels + n_val_channels])
    holdout_cur = set(holdout_currents)

    split = {"train": list(baselines), "val": list(baselines), "test": list(baselines)}
    for p in trials:
        ch = _parse_stim_channel(p.name)
        cur = _parse_current(p.name)
        if ch in test_ch or cur in holdout_cur:
            split["test"].append(str(p))
        elif ch in val_ch:
            split["val"].append(str(p))
        else:
            split["train"].append(str(p))

    return split


def load_split(split_path: str, data_dir: str) -> tuple[list, list, list]:
    with open(split_path) as f:
        split = json.load(f)
    return split["train"], split["val"], split["test"]


def save_split(split: dict, path: str):
    with open(path, "w") as f:
        json.dump(split, f, indent=2)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest daps/tests/test_splits.py -v
```

- [ ] **Step 5: Commit**

```bash
git add daps/splits.py daps/tests/test_splits.py
git commit -m "feat(daps): channel-holdout and channel+current-holdout split strategies"
```

---

### Task 8: Training Orchestrator

**Files:**
- Create: `daps/train.py`
- Create: `daps/tests/test_train_smoke.py`

**Interfaces:**
- Consumes: `build_daps_model` from Task 5, `DapsDataset` from Task 3, `HybridGCT` from Task 4, `DapsStimResponseBank`/`DapsAdjacencyBank` from Task 4, training loops from `training/train_vqvae.py`, `training/stage3_lct.py`, `training/train_prior.py`
- Produces: `main()` function that runs all 4 stages. Saves checkpoints to `ckpts/daps/`.

**Context:**
- **Stage 1**: train `HybridGCT` with 3 pretraining heads + populate `DapsStimResponseBank` + `DapsAdjacencyBank`. Freeze encoder/decoder/VQ. Three losses: (1) dynamic electrode embed prediction (main, analog of HDMEA site-map), (2) adjacency rate prediction, (3) linear stim param reconstruction. Loss weights swept (uncertainty weighting or grid search).
  - gct input: `MLP(concat(stim_params, electrode_char_flat_dynamic, adj_flat)) → 32-d`
  - Adjacency keyed by gct embedding with rounding (same `_key()` as HDMEA). Per stim-condition.
  - StimResponseBank keyed by stim-params portion of gct (generalizes across chips).
  - Stim param decoder handled same as HDMEA spatial map head.
- **Stage 2**: train conv stem + encoder + VQ + decoder. Freeze HybridGCT and banks. Loss: reconstruction BCE + ctx_loss_soft + ctx_field.
  - Electrode embedding (dynamic per-clip) enters additively at token level.
  - 2a: continuous alpha relaxation, 2b: deterministic flatten.
  - VQ: V=961, 3-level hierarchy (32/8/4), (111)(011)(000) tolerance — same as HDMEA.
  - Conv stem: single 3×3×K 3D conv layer.
- **Stage 3**: train LctMapper. Freeze VQ-VAE. lct dimension TBD (9+ dims, exact features parked for offline design).
- **Stage 4**: train priors. 4A: motif prior (code prediction given activity mask + gct). 4B: activity prior (which tokens are active, given gct). Freeze VQ-VAE. Same architecture as HDMEA.
  - Generation: unified model. Stim params in gct, null stim = spontaneous.
  - Spontaneous clips: random crop from 30s baseline recordings, oversampled at bursty windows.
- The existing training infrastructure (`fit_vqvae`, `run_stage3_lct`, etc.) should be reused by passing DAPS-specific configs. This file wires model, dataset, and existing training functions.
- The implementor must read `main.py:700-1200` and `training/train_vqvae.py:44` to understand how stages are orchestrated.

- [ ] **Step 1: Write smoke test**

```python
# daps/tests/test_train_smoke.py
import torch
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def mock_data_dir(tmp_path):
    rng = np.random.default_rng(42)
    n_clusters = 4
    embed_dir = tmp_path / "electrode_embeds"
    embed_dir.mkdir()

    static = rng.dirichlet(np.ones(n_clusters), size=(8, 8)).astype(np.float32)
    np.save(str(embed_dir / "static_embed.npy"), static)

    for i in range(5):
        clips = rng.choice([0, 1], size=(3, 80, 8, 8), p=[0.95, 0.05]).astype(np.uint8)
        stem = f"ch{i:02d}_cur1000_pw80"
        np.savez_compressed(str(tmp_path / f"{stem}.npz"),
                            clips=clips, stim_channel=i, current_nA=1000, pulse_width_us=80)
        for ci in range(3):
            per_clip = rng.dirichlet(np.ones(n_clusters), size=(8, 8)).astype(np.float32)
            np.save(str(embed_dir / f"{stem}_clip{ci}.npy"), per_clip)
    return tmp_path


def test_single_train_step(mock_data_dir):
    from daps.model_adapter import build_daps_model
    from daps.gct import HybridGCT
    from daps.dataset import DapsDataset

    n_clusters = 4
    model = build_daps_model(
        n_clusters=n_clusters,
        encoder_embed_dim=32, encoder_depth=1,
        code_dim=32, num_codes=(8, 4, 2),
        decoder_embed_dim=32, decoder_depth=1,
    )
    gct_module = HybridGCT(electrode_char_dim=8*8*n_clusters)

    npz_paths = sorted(str(p) for p in mock_data_dir.glob("ch*.npz"))
    embed_dir = str(mock_data_dir / "electrode_embeds")
    ds = DapsDataset(npz_paths, embed_dir)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)

    batch = next(iter(loader))
    # Dynamic per-clip embed goes to gct
    electrode_char_flat = batch["electrode_embed"].reshape(
        batch["electrode_embed"].shape[0], -1
    )
    global_ctx = gct_module(
        batch["global_ctx_stim"],
        batch["global_ctx_static"],
        electrode_char_flat,
        batch["is_baseline"],
    )
    # Static embed goes to encoder/decoder
    out = model(
        batch["x"],
        global_ctx=global_ctx,
        electrode_embed_static=batch["electrode_embed_static"],
    )
    assert out is not None
```

- [ ] **Step 2: Run smoke test, verify fail**

```bash
python -m pytest daps/tests/test_train_smoke.py -v
```

- [ ] **Step 3: Implement training orchestrator**

```python
# daps/train.py
"""4-stage training orchestrator for DAPS pipeline."""
import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from daps.model_adapter import build_daps_model
from daps.gct import HybridGCT, DapsStimResponseBank, DapsAdjacencyBank
from daps.dataset import DapsDataset
from daps.splits import load_split


def make_loaders(data_dir, split_path, electrode_embed_dir, batch_size):
    train_paths, val_paths, test_paths = load_split(split_path, data_dir)
    train_ds = DapsDataset(train_paths, electrode_embed_dir)
    val_ds = DapsDataset(val_paths, electrode_embed_dir)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2),
    )


def _prepare_gct_inputs(batch, device):
    """Extract and flatten DYNAMIC electrode characterization for gct."""
    electrode_dynamic = batch["electrode_embed"].to(device)  # per-clip
    B = electrode_dynamic.shape[0]
    electrode_char_flat = electrode_dynamic.reshape(B, -1)
    return (
        batch["global_ctx_stim"].to(device),
        batch["global_ctx_static"].to(device),
        electrode_char_flat,
        batch["is_baseline"].to(device),
    )


def run_stage1(model, gct_module, stim_bank, adj_bank, train_loader, val_loader, config):
    """Train HybridGCT + banks. Freeze everything else."""
    for p in model.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(gct_module.parameters(), lr=config.get("lr", 1e-3))
    device = next(model.parameters()).device

    for epoch in range(config.get("epochs", 50)):
        gct_module.train()
        for batch in train_loader:
            optimizer.zero_grad()
            stim, static, elec_char, is_bl = _prepare_gct_inputs(batch, device)
            global_ctx = gct_module(stim, static, elec_char, is_bl)

            x = batch["x"].to(device)
            electrode_embed_static = batch["electrode_embed_static"].to(device)

            out = model(x, global_ctx=global_ctx, electrode_embed_static=electrode_embed_static)

            # Populate banks
            target_map = (x[:, 0].sum(dim=1) > 0).float()  # (B, 8, 8)
            stim_bank.update(global_ctx, target_map)
            adj_bank.update_from_x(global_ctx, x)

            # Loss: ctx_loss_soft (adapt from existing)
            # Implementation: import from utils/losses.py
            # loss.backward() through gct_module only


def run_stage2(model, gct_module, train_loader, val_loader, config):
    """Train stem + encoder + VQ + decoder. Freeze HybridGCT."""
    for p in gct_module.parameters():
        p.requires_grad_(False)
    for p in model.parameters():
        p.requires_grad_(True)
    # Reuse training/train_vqvae.py:44 fit_vqvae
    pass


def run_stage3(model, train_loader, val_loader, config):
    """Train LctMapper (in_dim=11). Everything else frozen."""
    # Reuse training/stage3_lct.py:83 run_stage3_lct
    pass


def run_stage4(model, gct_module, train_loader, val_loader, config):
    """Train activity + motif priors."""
    # Reuse training/train_prior.py
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_path", required=True)
    parser.add_argument("--electrode_embed_dir", required=True)
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ckpt_dir", default="ckpts/daps")
    args = parser.parse_args()

    model = build_daps_model(n_clusters=args.n_clusters).cuda()
    gct_module = HybridGCT(electrode_char_dim=8*8*args.n_clusters).cuda()
    stim_bank = DapsStimResponseBank()
    adj_bank = DapsAdjacencyBank()

    train_loader, val_loader = make_loaders(
        args.data_dir, args.split_path, args.electrode_embed_dir, args.batch_size,
    )

    config = {"epochs": args.epochs, "lr": args.lr, "ckpt_dir": args.ckpt_dir}

    if args.stage == 1:
        run_stage1(model, gct_module, stim_bank, adj_bank, train_loader, val_loader, config)
    elif args.stage == 2:
        run_stage2(model, gct_module, train_loader, val_loader, config)
    elif args.stage == 3:
        run_stage3(model, train_loader, val_loader, config)
    elif args.stage == 4:
        run_stage4(model, gct_module, train_loader, val_loader, config)


if __name__ == "__main__":
    main()
```

**NOTE:** `run_stage*` functions are stubs. The implementor must:
1. Read `training/train_vqvae.py:44` (`fit_vqvae`) for loss computation
2. Read `main.py:782-1200` for stage orchestration
3. Wrap DAPS-specific batch preparation around existing training step functions
4. Pass `electrode_embed` through model forward in every stage

- [ ] **Step 4: Run smoke test**

```bash
python -m pytest daps/tests/test_train_smoke.py -v
```

- [ ] **Step 5: Commit**

```bash
git add daps/train.py daps/tests/test_train_smoke.py
git commit -m "feat(daps): 4-stage training orchestrator with stub stage runners"
```

---

### Task 9: Baselines

**Files:**
- Create: `daps/baselines.py`
- Create: `daps/tests/test_baselines.py`

**Interfaces:**
- Consumes: `DapsDataset` from Task 3, split from Task 7
- Produces: 4 baseline models, each with `fit(train_data)` and `predict(...)` interface.

**Context:**
- **(A) Per-channel rate model:** For each (stim_channel, current, pulse_width, response_channel) combo, learn P(spike in any time bin). Predict by sampling Bernoulli per time bin.
- **(B) GLM:** Logistic regression per response channel. Features: stim_channel_row, stim_channel_col, current_norm, pulse_width_norm, time_bin.
- **(C) Simple conv decoder:** Small 3D transposed-conv network. Input: stim params -> dense -> reshape -> ConvTranspose3d -> (1, 80, 8, 8) logits. No VQ, no hierarchy.
- **(D) Ablation: pipeline without electrode embeddings.** Use `build_daps_model()` but pass zero electrode embeddings. Tests whether electrode embedding adds value.

- [ ] **Step 1: Write tests**

```python
# daps/tests/test_baselines.py
import numpy as np
import torch
import pytest


def test_rate_model_predicts_correct_shape():
    from daps.baselines import PerChannelRateModel
    model = PerChannelRateModel()
    rng = np.random.default_rng(42)
    train_data = []
    for _ in range(20):
        clip = rng.choice([0, 1], size=(80, 8, 8), p=[0.95, 0.05]).astype(np.float32)
        train_data.append({"clip": clip, "stim_channel": 5, "current_nA": 1000, "pulse_width_us": 80})
    model.fit(train_data)
    pred = model.predict(stim_channel=5, current_nA=1000, pulse_width_us=80, shape=(80, 8, 8))
    assert pred.shape == (80, 8, 8)
    assert pred.min() >= 0.0
    assert pred.max() <= 1.0


def test_glm_baseline_shape():
    from daps.baselines import GLMBaseline
    model = GLMBaseline()
    rng = np.random.default_rng(42)
    X = rng.standard_normal((1000, 5)).astype(np.float32)
    y = rng.choice([0, 1], size=1000).astype(np.float32)
    model.fit(X, y)
    pred = model.predict(rng.standard_normal((10, 5)).astype(np.float32))
    assert pred.shape == (10,)
    assert pred.min() >= 0.0
    assert pred.max() <= 1.0
```

- [ ] **Step 2: Run tests, verify fail**

```bash
python -m pytest daps/tests/test_baselines.py -v
```

- [ ] **Step 3: Implement baselines**

```python
# daps/baselines.py
"""Baseline models for DAPS evaluation."""
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression


class PerChannelRateModel:
    def __init__(self):
        self.rates = {}

    def fit(self, train_data: list[dict]):
        counts = defaultdict(lambda: {"spikes": 0, "bins": 0})
        for entry in train_data:
            clip = entry["clip"]
            key_prefix = (entry["stim_channel"], entry["current_nA"], entry["pulse_width_us"])
            T, H, W = clip.shape
            for r in range(H):
                for c in range(W):
                    key = key_prefix + (r, c)
                    counts[key]["spikes"] += clip[:, r, c].sum()
                    counts[key]["bins"] += T
        for key, v in counts.items():
            self.rates[key] = v["spikes"] / max(v["bins"], 1)

    def predict(self, stim_channel, current_nA, pulse_width_us, shape=(80, 8, 8)):
        T, H, W = shape
        prob_map = np.zeros(shape, dtype=np.float32)
        for r in range(H):
            for c in range(W):
                key = (stim_channel, current_nA, pulse_width_us, r, c)
                prob_map[:, r, c] = self.rates.get(key, 0.0)
        return prob_map


class GLMBaseline:
    def __init__(self, max_iter=1000):
        self.model = LogisticRegression(max_iter=max_iter, solver="lbfgs")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


class SimpleConvDecoder:
    def __init__(self, stim_dim=4, hidden=128):
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(stim_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 64 * 10),
            nn.Unflatten(1, (1, 10, 8, 8)),
            nn.ConvTranspose3d(1, 1, kernel_size=(8, 1, 1), stride=(8, 1, 1)),
        )

    def forward(self, stim_params):
        return self.net(stim_params)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest daps/tests/test_baselines.py -v
```

- [ ] **Step 5: Commit**

```bash
git add daps/baselines.py daps/tests/test_baselines.py
git commit -m "feat(daps): rate model, GLM, and conv decoder baselines"
```

---

### Task 10: Evaluation and Generation Metrics

**Files:**
- Create: `daps/evaluate.py`
- Create: `daps/tests/test_evaluate.py`

**Interfaces:**
- Consumes: trained model from Task 8, test split from Task 7, generation pipeline from `inference/sample_prior.py`
- Produces: `run_evaluation(model, gct_module, test_loader, config) -> dict` of metrics. Reuses `inference/metrics_generative.py` where possible.

**Context:**
Two parts:

1. **Reconstruction quality** (Stages 1-2): AUPRC on held-out stim channels. Binary cross-entropy. ctx_loss on test set. The output is (1, 80, 8, 8) binary logits — existing metrics from `utils/losses.py` apply directly.

2. **Generation quality** (Stage 4): Generate binary volumes from stim params alone (no ground truth input). Compare generated vs real clips using:
   - Per-channel spike rate correlation
   - Temporal structure: inter-spike interval distribution (KS test)
   - Spatial pattern: which channels are active (Jaccard similarity)
   - Conditional accuracy: does the generated pattern change correctly with stim params? (paired test across current levels)

Most metrics from `inference/metrics_generative.py` can be reused. Main adaptation: replace (120, 220) spatial metrics with 8x8 channel metrics.

- [ ] **Step 1: Write tests**

```python
# daps/tests/test_evaluate.py
import numpy as np
import pytest


def test_channel_rate_correlation():
    from daps.evaluate import channel_rate_correlation
    rng = np.random.default_rng(42)
    real = rng.choice([0, 1], size=(10, 80, 8, 8), p=[0.95, 0.05]).astype(np.float32)
    generated = real.copy()
    generated[rng.random(real.shape) < 0.02] = 1
    corr = channel_rate_correlation(real, generated)
    assert -1.0 <= corr <= 1.0
    assert corr > 0.5


def test_spatial_jaccard():
    from daps.evaluate import spatial_jaccard
    real = np.zeros((80, 8, 8), dtype=np.float32)
    real[:, 2:5, 2:5] = 1
    gen = np.zeros((80, 8, 8), dtype=np.float32)
    gen[:, 2:5, 2:5] = 1
    j = spatial_jaccard(real, gen)
    assert j == 1.0


def test_isi_ks_test():
    from daps.evaluate import isi_ks_test
    rng = np.random.default_rng(42)
    real = rng.choice([0, 1], size=(80, 8, 8), p=[0.95, 0.05]).astype(np.float32)
    gen = real.copy()
    ks = isi_ks_test(real, gen)
    assert 0.0 <= ks <= 1.0
    assert ks < 0.1  # identical distributions
```

- [ ] **Step 2: Run tests, verify fail**

```bash
python -m pytest daps/tests/test_evaluate.py -v
```

- [ ] **Step 3: Implement**

```python
# daps/evaluate.py
"""Evaluation metrics for DAPS pipeline."""
import numpy as np
from scipy import stats


def channel_rate_correlation(real, generated):
    real_rates = real.mean(axis=(0, 1)).flatten()
    gen_rates = generated.mean(axis=(0, 1)).flatten()
    if real_rates.std() == 0 or gen_rates.std() == 0:
        return 0.0
    r, _ = stats.pearsonr(real_rates, gen_rates)
    return float(r)


def spatial_jaccard(real, generated):
    real_active = real.any(axis=0).flatten()
    gen_active = generated.any(axis=0).flatten()
    intersection = (real_active & gen_active).sum()
    union = (real_active | gen_active).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def isi_ks_test(real, generated):
    real_isis = _compute_isis(real)
    gen_isis = _compute_isis(generated)
    if len(real_isis) < 2 or len(gen_isis) < 2:
        return 1.0
    ks_stat, _ = stats.ks_2samp(real_isis, gen_isis)
    return float(ks_stat)


def _compute_isis(volume):
    T, H, W = volume.shape
    isis = []
    for r in range(H):
        for c in range(W):
            spike_bins = np.nonzero(volume[:, r, c])[0]
            if len(spike_bins) > 1:
                isis.extend(np.diff(spike_bins).tolist())
    return np.array(isis, dtype=np.float64)


def conditional_accuracy(model_predictions):
    currents = sorted(model_predictions.keys())
    if len(currents) < 2:
        return 0.0
    correct = 0
    total = 0
    for i in range(len(currents)):
        for j in range(i + 1, len(currents)):
            low_rate = np.mean([c.sum() for c in model_predictions[currents[i]]])
            high_rate = np.mean([c.sum() for c in model_predictions[currents[j]]])
            if high_rate > low_rate:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def run_reconstruction_eval(model, gct_module, test_loader, n_clusters, device="cuda"):
    import torch
    from sklearn.metrics import average_precision_score

    model.eval()
    gct_module.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            electrode_embed_dynamic = batch["electrode_embed"].to(device)
            electrode_embed_static = batch["electrode_embed_static"].to(device)
            B = electrode_embed_dynamic.shape[0]
            electrode_char_flat = electrode_embed_dynamic.reshape(B, -1)  # dynamic to gct

            global_ctx = gct_module(
                batch["global_ctx_stim"].to(device),
                batch["global_ctx_static"].to(device),
                electrode_char_flat,
                batch["is_baseline"].to(device),
            )
            out = model(x, global_ctx=global_ctx, electrode_embed_static=electrode_embed_static)
            logits = out.logits_vol_raw if hasattr(out, "logits_vol_raw") else out["logits"]
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            targets = x.cpu().numpy().flatten()
            all_targets.append(targets)
            all_preds.append(probs)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    auprc = average_precision_score(all_targets, all_preds)
    bce = -np.mean(
        all_targets * np.log(all_preds + 1e-7) +
        (1 - all_targets) * np.log(1 - all_preds + 1e-7)
    )
    return {"auprc": auprc, "bce": bce}
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest daps/tests/test_evaluate.py -v
```

- [ ] **Step 5: Commit**

```bash
git add daps/evaluate.py daps/tests/test_evaluate.py
git commit -m "feat(daps): evaluation metrics (AUPRC, spatial Jaccard, ISI KS, conditional accuracy)"
```

---

## Execution Order and Dependencies

```
Task 1 (extract + neighbor waveforms)
  |
  v
Task 2 (Stage 0: PCA+GMM clustering + sanity check)
  |
  v
Task 3 (dataset, two-stream)
  |
  +--- Task 4 (HybridGCT + banks)
  |
  +--- Task 6 (conv stem)
  |      |
  |      v
  +--- Task 5 (model adapter)
  |
  v
Task 7 (splits) ------+
                       |
                       v
                 Task 8 (training orchestrator)
                       |
Task 9 (baselines) ---+
                       |
                       v
                 Task 10 (evaluation)
```

Tasks 4, 6, 7 can be developed in parallel after Task 3. Task 5 depends on Task 6. Task 8 integrates everything. Task 10 is the final evaluation layer.

## Key Architectural Differences from v1

| Aspect | v1 (2026-08-31) | v2 (revised) |
|--------|-----------------|--------------|
| Culture type | "cortical organoid" (wrong) | Dense 2D multilayer cortical culture |
| Electrode info | Static embedding from waveform PCA + amplitude + connectivity | Static (chip marginal) at encoder, dynamic (per-clip) at gct, from 3x3 neighborhood PCA+GMM |
| Electrode info pathway | FiLM at conv stem (encoder-side) | Static additive at token level (encoder+decoder); dynamic flattened into gct |
| gct input | stim_params only | stim_params + dynamic_electrode_characterization_per_clip |
| Spatial bank | Not present | DapsStimResponseBank (EMA activation map per stim config) |
| Adjacency bank | Not present | DapsAdjacencyBank (temporal autocorrelation) |
| Stage 0 | Not present | Offline PCA+GMM on 3x3 neighborhood waveforms |
| Multi-chip | Not considered | Electrode characterization varies per chip |
| Cluster ordering | N/A | Global templates sorted by trough-to-peak duration |
| Dataset output | Single stream (x only) | Two-stream (x_binary, x_electrode_embed) |
| FiLM | ElectrodeFiLM at conv stem | None |
| ElectrodeCtxLoss | Auxiliary loss predicting electrode embeds | Removed |
