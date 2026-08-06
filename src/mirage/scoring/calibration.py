from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RegimeConditionalCalibrator:
    quantile: float = 0.99
    minimum_samples: int = 20
    thresholds_: dict[int, float] = field(default_factory=dict)
    global_threshold_: float | None = None

    def fit(self, scores: np.ndarray, regimes: np.ndarray | None = None) -> "RegimeConditionalCalibrator":
        values = np.asarray(scores, dtype=float)
        self.global_threshold_ = float(np.quantile(values, self.quantile))
        if regimes is not None:
            labels = np.asarray(regimes, dtype=int)
            for regime in np.unique(labels):
                selected = values[labels == regime]
                if len(selected) >= self.minimum_samples:
                    self.thresholds_[int(regime)] = float(np.quantile(selected, self.quantile))
        return self

    def threshold(self, regimes: np.ndarray | None = None) -> np.ndarray | float:
        if self.global_threshold_ is None:
            raise RuntimeError("Calibrator has not been fit")
        if regimes is None:
            return self.global_threshold_
        return np.array(
            [self.thresholds_.get(int(regime), self.global_threshold_) for regime in regimes]
        )

    def predict(self, scores: np.ndarray, regimes: np.ndarray | None = None) -> np.ndarray:
        return np.asarray(scores) > self.threshold(regimes)

