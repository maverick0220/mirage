from __future__ import annotations

import numpy as np

from mirage.baselines.base import CausalDiscoveryBaseline


class LaggedCorrelation(CausalDiscoveryBaseline):
    name = "lagged_correlation"

    def __init__(self, max_lag: int = 3, threshold: float = 0.2) -> None:
        self.max_lag = max_lag
        self.threshold = threshold
        self._adjacency: np.ndarray | None = None

    def fit(self, train_values: np.ndarray, variable_names: list[str]) -> "LaggedCorrelation":
        values = np.asarray(train_values, dtype=float)
        d = values.shape[1]
        result = np.zeros((self.max_lag + 1, d, d), dtype=np.float32)
        for lag in range(1, self.max_lag + 1):
            left = values[:-lag]
            right = values[lag:]
            # 常数列（stddev=0）会让 corrcoef 产生 NaN 并刷 divide 警告；
            # 抑制警告，NaN 由下方 nan_to_num 统一转 0（无相关）。
            with np.errstate(invalid="ignore", divide="ignore"):
                correlation = np.corrcoef(left.T, right.T)[:d, d:]
            correlation = np.nan_to_num(correlation)
            correlation[np.abs(correlation) < self.threshold] = 0
            result[lag] = correlation
        self._adjacency = result
        return self

    def adjacency(self) -> np.ndarray:
        if self._adjacency is None:
            raise RuntimeError("Baseline has not been fit")
        return self._adjacency

