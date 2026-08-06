from __future__ import annotations

import numpy as np


def aggregate_local_scores(local_scores: np.ndarray, method: str = "mean") -> np.ndarray:
    values = np.asarray(local_scores, dtype=np.float64)
    if method == "mean":
        return values.mean(axis=-1)
    if method == "max":
        return values.max(axis=-1)
    if method == "topk":
        if values.shape[-1] == 0:
            raise ValueError("topk aggregation requires at least one variable")
        k = max(1, min(3, values.shape[-1]))
        return np.partition(values, -k, axis=-1)[..., -k:].mean(axis=-1)
    raise ValueError(f"Unsupported aggregation: {method}")

