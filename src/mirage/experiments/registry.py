from __future__ import annotations

from mirage.baselines.anomaly import LinearResidualDetector, RobustZScore
from mirage.baselines.causal import LaggedCorrelation
from mirage.baselines.causal.external_adapters import (
    CDANsAdapter,
    DYNOTEARSAdapter,
    PCMCIPlusAdapter,
    PCMCIomegaAdapter,
    TCDFAdapter,
)


CAUSAL_BASELINES = {
    "lagged_correlation": LaggedCorrelation,
    "tigramite_pcmciplus": PCMCIPlusAdapter,
    "dynotears_causalnex": DYNOTEARSAdapter,
    "tcdf": TCDFAdapter,
    "pcmci_omega": PCMCIomegaAdapter,
    "cdans": CDANsAdapter,
}

ANOMALY_BASELINES = {
    "robust_zscore": RobustZScore,
    "linear_residual": LinearResidualDetector,
}


def create_baseline(name: str):
    if name in CAUSAL_BASELINES:
        return CAUSAL_BASELINES[name]()
    if name in ANOMALY_BASELINES:
        return ANOMALY_BASELINES[name]()
    raise KeyError(f"Unknown baseline: {name}")

