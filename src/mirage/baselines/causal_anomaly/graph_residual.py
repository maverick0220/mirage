from __future__ import annotations

import numpy as np

from mirage.baselines.anomaly.linear_residual import LinearResidualDetector
from mirage.baselines.causal.lagged_correlation import LaggedCorrelation


class GraphResidualDetector:
    """Transparent two-stage internal reference, not a named published baseline.

    The learned lagged-correlation graph is actually used at scoring time: for
    every variable, its causal parents (from the graph) are regressed out with
    OLS over the whole sequence and the absolute residuals are the local scores.
    A variable whose parents no longer explain it (mechanism violation) scores
    high; variables without parents receive zero score.
    """

    name = "graph_residual_reference"

    def __init__(self, max_lag: int = 1, threshold: float = 0.2) -> None:
        self.graph = LaggedCorrelation(max_lag, threshold)
        self.detector = LinearResidualDetector()

    def fit(self, train_values: np.ndarray, variable_names: list[str]) -> "GraphResidualDetector":
        self.graph.fit(train_values, variable_names)
        self.detector.fit(train_values)
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2:
            raise ValueError("values must be [time, variable]")
        adjacency = self.graph.adjacency()
        max_lag = adjacency.shape[0] - 1
        residuals = np.zeros_like(values)
        for target in range(values.shape[1]):
            design = [
                (lag, parent)
                for lag in range(1, max_lag + 1)
                for parent in np.where(np.abs(adjacency[lag, :, target]) > 1e-12)[0]
            ]
            if not design:
                continue
            rows = []
            y = []
            for time in range(max_lag, len(values)):
                rows.append([values[time - lag, parent] for lag, parent in design])
                y.append(values[time, target])
            features = np.asarray(rows, dtype=float)
            target_values = np.asarray(y, dtype=float)
            beta, *_ = np.linalg.lstsq(features, target_values, rcond=None)
            predicted = features @ beta
            residuals[max_lag:, target] = np.abs(target_values - predicted)
        return residuals
