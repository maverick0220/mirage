import numpy as np

from mirage.baselines.anomaly import LinearResidualDetector, RobustZScore
from mirage.baselines.causal import LaggedCorrelation


def test_internal_baselines_return_contract_shapes():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(200, 4))
    values[1:, 1] += 0.8 * values[:-1, 0]
    graph = LaggedCorrelation(max_lag=2, threshold=0.1).fit(values, list("abcd"))
    assert graph.adjacency().shape == (3, 4, 4)
    for detector in (RobustZScore(), LinearResidualDetector()):
        scores = detector.fit(values[:120]).score(values[120:])
        assert scores.shape == (80,)
        assert np.isfinite(scores).all()

