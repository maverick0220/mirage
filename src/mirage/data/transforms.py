from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RobustScaler:
    median_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "RobustScaler":
        array = np.asarray(values, dtype=np.float64)
        self.median_ = np.nanmedian(array, axis=0)
        q25, q75 = np.nanpercentile(array, [25, 75], axis=0)
        scale = q75 - q25
        self.scale_ = np.where(scale < 1e-8, 1.0, scale)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.scale_ is None:
            raise RuntimeError("RobustScaler must be fit on training data")
        array = np.asarray(values, dtype=np.float64)
        transformed = (array - self.median_) / self.scale_
        return np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.scale_ is None:
            raise RuntimeError("RobustScaler must be fit on training data")
        return np.asarray(values) * self.scale_ + self.median_

