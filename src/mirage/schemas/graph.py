from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np


@dataclass
class DynamicCausalGraph:
    """Lagged causal tensor with axes [graph_type, regime, lag, source, target]."""

    weights: np.ndarray
    variable_names: tuple[str, ...]
    graph_types: tuple[str, ...] = ("shared", "effective")

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=np.float32)
        if self.weights.ndim != 5:
            raise ValueError("weights must have shape [graph_type, regime, lag, source, target]")
        if self.weights.shape[0] != len(self.graph_types):
            raise ValueError("graph_types do not match weights")
        if self.weights.shape[-1] != len(self.variable_names) or self.weights.shape[-2] != len(
            self.variable_names
        ):
            raise ValueError("variable_names do not match graph dimensions")

    @property
    def n_regimes(self) -> int:
        return self.weights.shape[1]

    @property
    def max_lag(self) -> int:
        return self.weights.shape[2] - 1

    def adjacency(
        self, regime: int, graph_type: str = "effective", aggregate: str = "max"
    ) -> np.ndarray:
        index = self.graph_types.index(graph_type)
        values = self.weights[index, regime]
        if aggregate == "max":
            selected = np.take_along_axis(
                values, np.abs(values).argmax(axis=0, keepdims=True), axis=0
            )[0]
        elif aggregate == "sum":
            selected = values.sum(axis=0)
        else:
            raise ValueError(f"Unsupported aggregate: {aggregate}")
        return selected

    def to_networkx(
        self, regime: int, graph_type: str = "effective", threshold: float = 0.05
    ) -> nx.DiGraph:
        graph = nx.DiGraph(regime=regime, graph_type=graph_type)
        graph.add_nodes_from(self.variable_names)
        index = self.graph_types.index(graph_type)
        for lag, matrix in enumerate(self.weights[index, regime]):
            for source, target in zip(*np.where(np.abs(matrix) >= threshold), strict=False):
                graph.add_edge(
                    self.variable_names[source],
                    self.variable_names[target],
                    weight=float(matrix[source, target]),
                    lag=lag,
                )
        return graph

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            weights=self.weights,
            variable_names=np.array(self.variable_names),
            graph_types=np.array(self.graph_types),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "DynamicCausalGraph":
        payload = np.load(path, allow_pickle=False)
        return cls(
            weights=payload["weights"],
            variable_names=tuple(payload["variable_names"].tolist()),
            graph_types=tuple(payload["graph_types"].tolist()),
        )

    @classmethod
    def from_single_graph(
        cls, weights: np.ndarray, variable_names: Sequence[str], label: str = "effective"
    ) -> "DynamicCausalGraph":
        array = np.asarray(weights, dtype=np.float32)
        if array.ndim == 3:
            array = array[None, None]
        elif array.ndim == 4:
            array = array[None]
        return cls(array, tuple(variable_names), (label,))

