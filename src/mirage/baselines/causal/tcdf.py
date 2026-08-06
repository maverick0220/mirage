"""Compact TCDF-style causal discovery: per-target regression with lag selection.

The full TCDF uses causal convolutions plus a permutation-based causal
validation step; here the convolutional stage is condensed into a per-target
multi-lag regression with L1 selection (the causal-validation stage is omitted
and documented). Edge presence = |coefficient| above threshold, mirroring the
attention-weight interpretation of TCDF.
"""

from __future__ import annotations

import numpy as np

from mirage.baselines.base import CausalDiscoveryBaseline


class TCDFAdapter(CausalDiscoveryBaseline):
    name = "tcdf"

    def __init__(
        self,
        max_lag: int = 3,
        threshold: float = 0.05,
        l1_ratio: float = 0.01,
        seed: int = 2026,
    ) -> None:
        self.max_lag = max_lag
        self.threshold = threshold
        self.l1_ratio = l1_ratio
        self.seed = seed
        self._adjacency: np.ndarray | None = None

    def fit(self, train_values: np.ndarray, variable_names: list[str]) -> "TCDFAdapter":
        from sklearn.linear_model import Lasso

        rng = np.random.default_rng(self.seed)
        values = np.asarray(train_values, dtype=np.float64)
        L, d = self.max_lag, values.shape[1]
        if len(values) <= L + 1:
            raise ValueError("train series must be longer than max_lag + 1")
        target = values[L:]
        adjacency = np.zeros((L + 1, d, d), dtype=np.float32)
        for target_index in range(d):
            design = []
            for lag in range(1, L + 1):
                for source in range(d):
                    design.append((lag, source))
            x = np.column_stack([values[L - lag : -lag, source] for lag, source in design])
            y = target[:, target_index]
            scale = np.std(x, axis=0)
            scale[scale < 1e-9] = 1.0
            model = Lasso(alpha=self.l1_ratio, max_iter=2000, random_state=int(rng.integers(0, 2**31)))
            model.fit(x / scale, y)
            coefficients = model.coef_ * (1.0 / scale)
            for position, (lag, source) in enumerate(design):
                if abs(coefficients[position]) > self.threshold:
                    adjacency[lag, source, target_index] = float(coefficients[position])
        self._adjacency = adjacency
        return self

    def adjacency(self) -> np.ndarray:
        if self._adjacency is None:
            raise RuntimeError("Baseline has not been fit")
        return self._adjacency
