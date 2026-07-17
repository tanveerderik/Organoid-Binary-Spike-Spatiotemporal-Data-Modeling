"""
Project-wide immutable constants and small validation helpers.

Keep canonical feature ordering and adjacency-bin definitions here so training,
inference, metrics, reports and visualization cannot silently disagree.
"""

from typing import Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Local activity-context schema
# ---------------------------------------------------------------------------

ACTIVITY_CTX_NAMES: Tuple[str, ...] = (
    "log_mean_firing_density",
    "var_x",
    "var_y",
    "var_t",
    "cov_xy",
    "cov_xt",
    "cov_yt",
    "active_site_ratio",
    "temporal_trend",
)

ACTIVITY_CTX_DIM: int = len(ACTIVITY_CTX_NAMES)

ACTIVITY_CTX_INDEX = {
    name: index
    for index, name in enumerate(ACTIVITY_CTX_NAMES)
}


# ---------------------------------------------------------------------------
# Short-gap adjacency schema
# ---------------------------------------------------------------------------

DEFAULT_GAP_BINS: Tuple[Tuple[int, int], ...] = (
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 6),
    (7, 12),
    (13, 24),
    (25, 48),
)


def normalize_gap_bins(
    gap_bins: Optional[
        Sequence[Sequence[int]]
    ] = None,
) -> Tuple[Tuple[int, int], ...]:
    """
    Validate and normalize short-gap bins.

    Parameters
    ----------
    gap_bins:
        Sequence of (lower, upper) inclusive temporal-gap ranges.
        If None, DEFAULT_GAP_BINS is used.

    Returns
    -------
    tuple of tuple
        Immutable normalized bins.

    Requirements
    ------------
    - Every bin has exactly two values.
    - Gaps start at 1 or greater.
    - lower <= upper.
    - Bins are ordered.
    - Bins do not overlap.
    """
    if gap_bins is None:
        gap_bins = DEFAULT_GAP_BINS

    normalized = []

    for index, gap_bin in enumerate(gap_bins):
        if len(gap_bin) != 2:
            raise ValueError(
                f"gap_bins[{index}] must contain exactly "
                f"(lower, upper), got {gap_bin!r}"
            )

        lower = int(gap_bin[0])
        upper = int(gap_bin[1])

        if lower < 1:
            raise ValueError(
                f"gap_bins[{index}] has lower={lower}; "
                "temporal gaps must start at 1."
            )

        if upper < lower:
            raise ValueError(
                f"gap_bins[{index}] has invalid range "
                f"({lower}, {upper})."
            )

        if normalized:
            previous_lower, previous_upper = normalized[-1]

            if lower <= previous_upper:
                raise ValueError(
                    "Gap bins must be ordered and non-overlapping. "
                    f"Found {(previous_lower, previous_upper)} "
                    f"followed by {(lower, upper)}."
                )

        normalized.append(
            (lower, upper)
        )

    if not normalized:
        raise ValueError(
            "gap_bins cannot be empty."
        )

    return tuple(normalized)


def max_gap_from_bins(
    gap_bins: Optional[
        Sequence[Sequence[int]]
    ] = None,
) -> int:
    """
    Return the largest upper gap bound.

    This is the temporal search/shift distance, not the number of bins.
    """
    normalized = normalize_gap_bins(
        gap_bins
    )

    return max(
        upper
        for _, upper in normalized
    )


def gap_bin_label(
    gap_bin: Sequence[int],
) -> str:
    """
    Format a bin for plots and reports.

    Examples
    --------
    (1, 1) -> "1"
    (4, 6) -> "4-6"
    """
    if len(gap_bin) != 2:
        raise ValueError(
            f"gap_bin must contain exactly two values, "
            f"got {gap_bin!r}"
        )

    lower = int(gap_bin[0])
    upper = int(gap_bin[1])

    if lower == upper:
        return str(lower)

    return f"{lower}-{upper}"


def gap_bin_labels(
    gap_bins: Optional[
        Sequence[Sequence[int]]
    ] = None,
) -> Tuple[str, ...]:
    """
    Return display labels for every bin.
    """
    normalized = normalize_gap_bins(
        gap_bins
    )

    return tuple(
        gap_bin_label(gap_bin)
        for gap_bin in normalized
    )