from __future__ import annotations

import numpy as np

from mirage.baselines.base import AnomalyBaseline


class RobustZScore(AnomalyBaseline):
    name = "robust_zscore"

    def __init__(self) -> None:
        self.median_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, train_values: np.ndarray) -> "RobustZScore":
        values = np.asarray(train_values, dtype=float)
        self.median_ = np.nanmedian(values, axis=0)
        mad = np.nanmedian(np.abs(values - self.median_), axis=0)
        self.scale_ = np.where(mad < 1e-8, 1.0, 1.4826 * mad)
        return self

    def local_score(self, values: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.scale_ is None:
            raise RuntimeError("Baseline has not been fit")
        return np.abs((np.asarray(values) - self.median_) / self.scale_)

    def score(self, values: np.ndarray) -> np.ndarray:
        return self.local_score(values).mean(axis=-1)

