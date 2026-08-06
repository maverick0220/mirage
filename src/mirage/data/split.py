from __future__ import annotations


def chronological_slices(
    n_samples: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    embargo: int = 0,
) -> dict[str, slice]:
    """Strict chronological split with an optional embargo band between segments.

    ``embargo`` drops that many samples from the tail of every split and prepends
    the same margin to the next split, so that windows anchored near a boundary
    never read data belonging to the following segment. The implementation plan
    (section 1.3) requires embargo >= lookback + max_lag + counterfactual horizon.
    """
    if n_samples < 3:
        raise ValueError("At least three samples are required")
    if not (0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("Invalid chronological split ratios")
    embargo = max(0, int(embargo))
    train_raw_end = max(1, int(n_samples * train_ratio))
    val_raw_end = max(train_raw_end + 1, int(n_samples * (train_ratio + val_ratio)))
    train_end = max(1, train_raw_end - embargo)
    val_start = min(train_raw_end + embargo, val_raw_end)
    val_end = max(val_start + 1, val_raw_end - embargo)
    test_start = min(val_raw_end + embargo, n_samples)
    if val_start >= n_samples or val_end > n_samples or test_start >= n_samples:
        raise ValueError("Embargo too large for the available sample count")
    return {
        "train": slice(0, train_end),
        "validation": slice(val_start, val_end),
        "test": slice(test_start, n_samples),
    }


