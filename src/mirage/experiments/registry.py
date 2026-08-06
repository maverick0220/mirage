from __future__ import annotations

from mirage.baselines.anomaly import LinearResidualDetector, RobustZScore
from mirage.baselines.anomaly.neural import NeuralAnomalyBaseline
from mirage.baselines.causal import LaggedCorrelation
from mirage.baselines.causal.dynotears import DYNOTEARSAdapter
from mirage.baselines.causal.external_adapters import (
    CDANsAdapter,
    PCMCIPlusAdapter,
    PCMCIomegaAdapter,
)
from mirage.baselines.causal.tcdf import TCDFAdapter

# Causality baselines producing [lag, source, target] adjacency tensors.
CAUSAL_BASELINES = {
    "lagged_correlation": LaggedCorrelation,
    "tigramite_pcmciplus": PCMCIPlusAdapter,
    "dynotears_causalnex": DYNOTEARSAdapter,
    "tcdf": TCDFAdapter,
    "pcmci_omega": PCMCIomegaAdapter,
    "cdans": CDANsAdapter,
}

# Statistical anomaly baselines producing scores directly.
ANOMALY_BASELINES = {
    "robust_zscore": RobustZScore,
    "linear_residual": LinearResidualDetector,
}

# Neural anomaly baselines (window-to-next-step predictors, MSE scores).
NEURAL_ANOMALY_BASELINES = [
    "lstm",
    "transformer",
    "anomaly_transformer",
    "tranad",
    "dlinear",
    "gdn",
    "omnianomaly",
    "mtad_gat",
]

ANOMALY_ALIASES = {
    "graph_residual_reference": "graph_residual",
}


def is_neural_anomaly(name: str) -> bool:
    return name.lower().replace("-", "").replace("_", "") in {
        item.lower().replace("-", "").replace("_", "") for item in NEURAL_ANOMALY_BASELINES
    }


def create_baseline(name: str, **kwargs):
    """Build a baseline instance by name. Neural baselines accept training kwargs."""
    if name == "pcmci_omega":
        return PCMCIomegaAdapter()
    if name == "cdans":
        return CDANsAdapter()
    if name in CAUSAL_BASELINES:
        return CAUSAL_BASELINES[name](**kwargs)
    if name in ANOMALY_BASELINES:
        return ANOMALY_BASELINES[name]()
    if name in ANOMALY_ALIASES:
        alias = ANOMALY_ALIASES[name]
        if alias == "graph_residual":
            from mirage.baselines.causal_anomaly.graph_residual import GraphResidualDetector

            return GraphResidualDetector(max_lag=int(kwargs.get("max_lag", 1)))
    normalized = name.lower().replace("-", "").replace("_", "")
    for candidate in NEURAL_ANOMALY_BASELINES:
        if normalized == candidate.lower().replace("-", "").replace("_", ""):
            return NeuralAnomalyBaseline(model_name=candidate, **kwargs)
    raise KeyError(f"Unknown baseline: {name}")
