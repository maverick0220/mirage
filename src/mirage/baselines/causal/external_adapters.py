from __future__ import annotations

from pathlib import Path

import numpy as np

from mirage.baselines.base import CausalDiscoveryBaseline
from mirage.baselines.external import require_external


class PublishedCausalAdapter(CausalDiscoveryBaseline):
    module_name: str | None = None
    vendor_directory: str = ""

    def __init__(self, vendor_root: str | Path = "vendor") -> None:
        self.vendor_root = Path(vendor_root)
        self._adjacency: np.ndarray | None = None

    def fit(self, train_values: np.ndarray, variable_names: list[str]) -> "PublishedCausalAdapter":
        require_external(
            self.name,
            self.module_name,
            self.vendor_root / self.vendor_directory,
        )
        raise NotImplementedError(
            f"'{self.name}' upstream is present, but version-specific execution must run in its pinned "
            "baseline environment via the experiment runner. No proxy result is emitted."
        )

    def adjacency(self) -> np.ndarray:
        if self._adjacency is None:
            raise RuntimeError("No published-method result has been imported")
        return self._adjacency


class PCMCIPlusAdapter(PublishedCausalAdapter):
    name = "tigramite_pcmciplus"
    module_name = "tigramite"
    vendor_directory = "tigramite_pcmciplus"

    def __init__(
        self,
        vendor_root: str | Path = "vendor",
        max_lag: int = 3,
        pc_alpha: float = 0.05,
    ) -> None:
        super().__init__(vendor_root)
        self.max_lag = max_lag
        self.pc_alpha = pc_alpha

    def fit(self, train_values: np.ndarray, variable_names: list[str]) -> "PCMCIPlusAdapter":
        import sys

        vendor = self.vendor_root / self.vendor_directory
        require_external(self.name, self.module_name, vendor)
        if str(vendor.resolve()) not in sys.path:
            sys.path.insert(0, str(vendor.resolve()))
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI

        values = np.asarray(train_values, dtype=float)
        dataframe = pp.DataFrame(values, var_names=variable_names)
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
        result = pcmci.run_pcmciplus(
            tau_min=1,
            tau_max=self.max_lag,
            pc_alpha=self.pc_alpha,
        )
        graph = result["graph"]
        weights = result["val_matrix"]
        # Tigramite returns [N, N, tau_max + 1]; the tau axis IS the lag (tau=0
        # simultaneous, tau=k lag k). Assert the contract so a version change in
        # tigramite fails loudly instead of silently mis-aligning lags.
        if graph.shape[-1] != self.max_lag + 1:
            raise ValueError(
                f"Unexpected tigramite graph shape {graph.shape}: expected a lag "
                f"axis of length {self.max_lag + 1} for tau_max={self.max_lag}"
            )
        adjacency = np.zeros((self.max_lag + 1, values.shape[1], values.shape[1]), dtype=np.float32)
        for lag in range(1, self.max_lag + 1):
            present = graph[:, :, lag] != ""
            adjacency[lag] = np.where(present, weights[:, :, lag], 0.0)
        self._adjacency = adjacency
        return self


class DYNOTEARSAdapter(PublishedCausalAdapter):
    name = "dynotears_causalnex"
    module_name = "causalnex"
    vendor_directory = "dynotears_causalnex"


class TCDFAdapter(PublishedCausalAdapter):
    name = "tcdf"
    vendor_directory = "tcdf"


class PCMCIomegaAdapter(PublishedCausalAdapter):
    name = "pcmci_omega"
    vendor_directory = "pcmci_omega"


class CDANsAdapter(PublishedCausalAdapter):
    name = "cdans"
    vendor_directory = "cdans"

