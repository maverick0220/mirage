from __future__ import annotations

from pathlib import Path

import numpy as np

from mirage.baselines.base import AnomalyBaseline
from mirage.baselines.external import require_external


class PublishedAnomalyAdapter(AnomalyBaseline):
    vendor_directory: str = ""

    def __init__(self, vendor_root: str | Path = "vendor") -> None:
        self.vendor_root = Path(vendor_root)

    def fit(self, train_values: np.ndarray) -> "PublishedAnomalyAdapter":
        require_external(self.name, vendor_path=self.vendor_root / self.vendor_directory)
        raise NotImplementedError(
            f"'{self.name}' is executed in its pinned upstream environment and imported through the "
            "experiment result schema; no internal proxy is used."
        )

    def score(self, values: np.ndarray) -> np.ndarray:
        raise RuntimeError("No published-method result has been imported")


def _adapter(name: str, directory: str):
    return type(name, (PublishedAnomalyAdapter,), {"name": directory, "vendor_directory": directory})


OmniAnomalyAdapter = _adapter("OmniAnomalyAdapter", "omnianomaly")
MTADGATAdapter = _adapter("MTADGATAdapter", "mtad_gat")
GDNAdapter = _adapter("GDNAdapter", "gdn")
TranADAdapter = _adapter("TranADAdapter", "tranad")
AnomalyTransformerAdapter = _adapter("AnomalyTransformerAdapter", "anomaly_transformer")

