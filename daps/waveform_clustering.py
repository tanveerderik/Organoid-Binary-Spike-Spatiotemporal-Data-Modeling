"""Stage 0: Offline waveform clustering using 3x3 neighborhood features."""
import numpy as np
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


@dataclass
class Stage0Result:
    global_templates: np.ndarray      # (n_clusters, 83)
    cluster_order: np.ndarray         # (n_clusters,) trough-to-peak durations in seconds
    pca_model: PCA
    gmm_model: GaussianMixture
    n_clusters: int
    sort_order: np.ndarray            # permutation mapping GMM labels to sorted labels


def _trough_to_peak_duration(waveform_75: np.ndarray, fs: int = 25000) -> float:
    """Compute trough-to-peak duration from the central waveform (dims 0-74)."""
    trough_idx = int(np.argmin(waveform_75))
    if trough_idx >= len(waveform_75) - 1:
        return float(len(waveform_75)) / fs
    peak_idx = int(np.argmax(waveform_75[trough_idx:])) + trough_idx
    return (peak_idx - trough_idx) / fs


def run_stage0(
    features: np.ndarray,
    channels: np.ndarray,
    train_mask: np.ndarray,
    n_pca: int = 20,
    n_clusters: int = 8,
    seed: int = 42,
) -> Stage0Result:
    """
    Fit PCA + GMM on training spikes.

    Args:
        features: (N, 83) all spikes' neighborhood feature vectors
        channels: (N,) electrode IDs
        train_mask: (N,) bool
        n_pca: number of PCA components
        n_clusters: number of GMM clusters
    """
    train_feats = features[train_mask]

    pca = PCA(n_components=min(n_pca, train_feats.shape[1]), random_state=seed)
    train_pca = pca.fit_transform(train_feats)

    gmm = GaussianMixture(
        n_components=n_clusters, covariance_type="full",
        random_state=seed, max_iter=200, n_init=3, reg_covar=1e-4,
    )
    gmm.fit(train_pca.astype(np.float64))

    templates_pca = gmm.means_  # (n_clusters, n_pca)
    templates_83 = pca.inverse_transform(templates_pca)  # (n_clusters, 83)

    t2p = np.array([_trough_to_peak_duration(t[:75]) for t in templates_83])
    sort_order = np.argsort(t2p)
    templates_83 = templates_83[sort_order]
    t2p = t2p[sort_order]

    return Stage0Result(
        global_templates=templates_83.astype(np.float32),
        cluster_order=t2p.astype(np.float32),
        pca_model=pca,
        gmm_model=gmm,
        n_clusters=n_clusters,
        sort_order=sort_order,
    )


def assign_clusters(features: np.ndarray, result: Stage0Result) -> np.ndarray:
    """Assign each spike to a cluster. Returns (N,) int labels in sorted order."""
    pca_feats = result.pca_model.transform(features)
    raw_labels = result.gmm_model.predict(pca_feats)
    inv_perm = np.argsort(result.sort_order)
    return inv_perm[raw_labels]


def compute_electrode_characterization(
    features: np.ndarray,
    channels: np.ndarray,
    clip_ids: np.ndarray,
    trial_keys: np.ndarray,
    result: Stage0Result,
) -> tuple[dict, np.ndarray]:
    """
    Compute per-clip (dynamic) and static electrode characterization.

    Returns:
        per_clip: dict[(trial_key, clip_id) -> (8, 8, n_clusters)] normalized cluster composition
        static: (8, 8, n_clusters) average across all clips — the chip-level marginal E[dynamic]
    """
    from daps.extract_data import _CH_TO_GRID
    labels = assign_clusters(features, result)
    K = result.n_clusters

    per_clip = {}
    all_counts = np.zeros((8, 8, K), dtype=np.float64)

    unique_clips = set(zip(trial_keys, clip_ids))
    for tk, ci in unique_clips:
        mask = (trial_keys == tk) & (clip_ids == ci)
        clip_counts = np.zeros((8, 8, K), dtype=np.float64)
        for feat_idx in np.where(mask)[0]:
            ch = int(channels[feat_idx])
            r, c = _CH_TO_GRID[ch]
            clip_counts[r, c, labels[feat_idx]] += 1

        totals = clip_counts.sum(axis=-1, keepdims=True)
        clip_comp = np.where(totals > 0, clip_counts / totals, 0.0)
        per_clip[(tk, ci)] = clip_comp.astype(np.float32)

        all_counts += clip_counts

    static_totals = all_counts.sum(axis=-1, keepdims=True)
    static = np.where(static_totals > 0, all_counts / static_totals, 0.0)

    return per_clip, static.astype(np.float32)


def compute_cooccurrence_matrix(
    features: np.ndarray,
    channels: np.ndarray,
    result: Stage0Result,
) -> np.ndarray:
    """
    Sanity check: (64, n_clusters) co-occurrence matrix.
    If all rows are identical, cluster identity adds nothing per electrode.
    """
    labels = assign_clusters(features, result)
    K = result.n_clusters
    cooc = np.zeros((64, K), dtype=np.float64)
    for ch, lbl in zip(channels, labels):
        cooc[int(ch), lbl] += 1
    totals = cooc.sum(axis=1, keepdims=True)
    cooc = np.where(totals > 0, cooc / totals, 0.0)
    return cooc.astype(np.float32)
