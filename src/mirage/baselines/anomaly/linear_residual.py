from __future__ import annotations

import numpy as np

from mirage.baselines.base import AnomalyBaseline


class LinearResidualDetector(AnomalyBaseline):
    name = "linear_residual"

    def __init__(self, ridge: float = 1e-3) -> None:
        self.ridge = ridge
        self.coefficients_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(
        self, train_values: np.ndarray, variable_names: list[str] | None = None
    ) -> "LinearResidualDetector":
        values = np.asarray(train_values, dtype=float)
        left = np.column_stack([np.ones(len(values) - 1), values[:-1]])
        right = values[1:]
        regularizer = self.ridge * np.eye(left.shape[1])
        regularizer[0, 0] = 0
        self.coefficients_ = np.linalg.solve(left.T @ left + regularizer, left.T @ right)
        residual = right - left @ self.coefficients_
        self.scale_ = np.std(residual, axis=0) + 1e-6
        return self

    def local_score(self, values: np.ndarray) -> np.ndarray:
        if self.coefficients_ is None or self.scale_ is None:
            raise RuntimeError("Baseline has not been fit")
        array = np.asarray(values, dtype=float)
        left = np.column_stack([np.ones(len(array) - 1), array[:-1]])
        residual = np.abs(array[1:] - left @ self.coefficients_) / self.scale_
        return np.vstack([np.zeros((1, array.shape[1])), residual])

    def score(self, values: np.ndarray) -> np.ndarray:
        return self.local_score(values).mean(axis=-1)

