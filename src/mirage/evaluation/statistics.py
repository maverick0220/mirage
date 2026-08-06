from __future__ import annotations

import numpy as np
from scipy import stats


def paired_bootstrap_interval(
    values: np.ndarray, confidence: float = 0.95, samples: int = 2000, seed: int = 2026
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(array, size=len(array), replace=True).mean() for _ in range(samples)])
    alpha = (1 - confidence) / 2
    return {
        "mean": float(array.mean()),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1 - alpha)),
    }


def wilcoxon_paired(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    statistic, p_value = stats.wilcoxon(left, right)
    difference = np.asarray(left) - np.asarray(right)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "median_difference": float(np.median(difference)),
    }

