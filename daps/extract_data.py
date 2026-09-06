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

# MCS 60MEA200/30iR-Ti physical layout
CORNERS = frozenset({0, 7, 56, 63})       # electrodes 11, 18, 81, 88 — absent
REFERENCE = frozenset({4})                  # electrode 15 — internal reference
EXCLUDED = CORNERS | REFERENCE              # not recording electrodes
RECORDING_CHANNELS = frozenset(range(64)) - EXCLUDED  # 59 valid positions

# Neighbor ordering: NW, N, NE, W, E, SW, S, SE
_NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def channel_to_grid(ch: int) -> tuple[int, int]:
    return _CH_TO_GRID[ch]


def get_neighbor_channels(ch: int) -> list[int | None]:
    """Return 8-element list of neighbor channel indices.
    None for out-of-bounds, corners, or reference positions."""
    r, c = _CH_TO_GRID[ch]
    neighbors = []
    for dr, dc in _NEIGHBOR_OFFSETS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 8 and 0 <= nc < 8:
            nbr = _GRID_TO_CH[(nr, nc)]
            neighbors.append(None if nbr in EXCLUDED else nbr)
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

    For each spike: central waveform (75 dims, already in uV) +
    8 neighbor trough amplitudes from raw voltage (zero-padded for borders).

    Returns:
        features: (N_spikes, 83) float32
        channels: (N_spikes,) int32
        clip_ids: (N_spikes,) int32 — epoch index (0/1/2) or -1 for out-of-epoch
    """
    with tables.open_file(h5_path, mode="r") as f:
        start_ts = f.root._v_attrs.start_timestamp
        uv_scale = f.root._v_attrs.uV_per_sample_unit

        spike_channels = f.root.spikes.col("channel")
        spike_timestamps = f.root.spikes.col("timestamp") - start_ts
        spike_waveforms = f.root.spikes.col("samples")  # (N, 75) float32, already uV
        raw_samples = f.root.samples  # (total_samples, 64) int16, on-disk
        total_raw_samples = raw_samples.shape[0]

        if hasattr(f.root, "stims") and f.root.stims.nrows > 0:
            stim_ts = np.array(
                [row["timestamp"] - start_ts for row in f.root.stims]
            )
        else:
            stim_ts = np.array([], dtype=np.int64)

        n_spikes = len(spike_channels)
        features = np.zeros((n_spikes, 83), dtype=np.float32)
        channels_out = spike_channels.astype(np.int32)
        clip_ids = np.full(n_spikes, -1, dtype=np.int32)

        for i in range(n_spikes):
            ch = int(spike_channels[i])
            t = int(spike_timestamps[i])

            features[i, :75] = spike_waveforms[i]

            for ei, st in enumerate(stim_ts):
                epoch_end = st + int(0.8 * fs)
                if st <= t < epoch_end:
                    clip_ids[i] = ei
                    break

            neighbors = get_neighbor_channels(ch)
            t_start = max(0, t - waveform_half)
            t_end = min(total_raw_samples, t + waveform_half + 1)
            if t_end > t_start:
                raw_snippet = np.array(
                    raw_samples[t_start:t_end, :], dtype=np.float32
                ) * uv_scale

                for ni, nbr_ch in enumerate(neighbors):
                    if nbr_ch is not None:
                        features[i, 75 + ni] = raw_snippet[:, nbr_ch].min()

    return features, channels_out, clip_ids


def h5_to_clips(
    h5_path: str,
    bin_ms: int = 10,
    fs: int = 25000,
    epoch_sec: float = 0.8,
) -> tuple[np.ndarray, dict]:
    """
    Extract binary spike rasters from a DAPS HDF5 trial file.

    Returns:
        clips: (n_stims, T_bins, 8, 8) uint8 binary raster
        stim_params: dict with stim_channel, current_nA, pulse_width_us
    """
    samples_per_bin = int(fs * bin_ms / 1000)
    bins_per_epoch = int(epoch_sec * 1000 / bin_ms)

    with tables.open_file(h5_path, mode="r") as f:
        attrs = f.root._v_attrs
        start_ts = attrs.start_timestamp
        app = attrs.application

        stim_ts = np.array(
            [row["timestamp"] - start_ts for row in f.root.stims]
        )
        n_stims = len(stim_ts)

        spike_channels = f.root.spikes.col("channel")
        spike_timestamps = f.root.spikes.col("timestamp") - start_ts

    clips = np.zeros((n_stims, bins_per_epoch, 8, 8), dtype=np.uint8)

    for ei, st in enumerate(stim_ts):
        epoch_end = st + int(epoch_sec * fs)
        mask = (spike_timestamps >= st) & (spike_timestamps < epoch_end)
        ep_channels = spike_channels[mask]
        ep_times = spike_timestamps[mask] - st

        for ch, t in zip(ep_channels, ep_times):
            time_bin = int(t / samples_per_bin)
            if time_bin >= bins_per_epoch:
                continue
            row, col = channel_to_grid(int(ch))
            clips[ei, time_bin, row, col] = 1

    stim_params = {
        "stim_channel": int(app["stim_channel"]),
        "current_nA": int(round(app["stim_current_A"] * 1e9)),
        "pulse_width_us": int(round(app["pulse_width_sec"] * 1e6)),
    }

    return clips, stim_params


def h5_to_baseline_clips(
    h5_path: str,
    bin_ms: int = 10,
    fs: int = 25000,
    epoch_sec: float = 0.8,
) -> np.ndarray:
    """Extract non-overlapping binary clips from a baseline recording."""
    samples_per_bin = int(fs * bin_ms / 1000)
    bins_per_epoch = int(epoch_sec * 1000 / bin_ms)
    samples_per_epoch = int(epoch_sec * fs)

    with tables.open_file(h5_path, mode="r") as f:
        start_ts = f.root._v_attrs.start_timestamp
        total_samples = f.root.samples.shape[0]
        n_clips = total_samples // samples_per_epoch

        spike_channels = f.root.spikes.col("channel")
        spike_timestamps = f.root.spikes.col("timestamp") - start_ts

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

    return clips
