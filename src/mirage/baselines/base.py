from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CausalDiscoveryBaseline(ABC):
    name: str

    @abstractmethod
    def fit(self, train_values: np.ndarray, variable_names: list[str]) -> "CausalDiscoveryBaseline":
        raise NotImplementedError

    @abstractmethod
    def adjacency(self) -> np.ndarray:
        """Return [lag, source, target]."""
        raise NotImplementedError


class AnomalyBaseline(ABC):
    name: str

    @abstractmethod
    def fit(self, train_values: np.ndarray) -> "AnomalyBaseline":
        raise NotImplementedError

    @abstractmethod
    def score(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

