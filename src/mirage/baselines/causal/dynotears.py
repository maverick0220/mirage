"""Compact NOTEARS-style differentiable DAG learning (DYNOTEARS backbone).

Learns X_t = X_t @ A0 + sum_l X_{t-l} @ Al + E with a DAG constraint on the
lag-0 slice A0 (NOTEARS h(A) = trace(expm(A∘A)) - d), matching the DYNOTEARS
design of acyclic instantaneous graph + unconstrained lagged effects.
"""

from __future__ import annotations

import numpy as np
import torch

from mirage.baselines.base import CausalDiscoveryBaseline


class DYNOTEARSAdapter(CausalDiscoveryBaseline):
    name = "dynotears_causalnex"

    def __init__(
        self,
        max_lag: int = 3,
        lambda1: float = 0.01,
        rho_init: float = 1.0,
        alpha: float = 0.0,
        epochs: int = 300,
        learning_rate: float = 1e-2,
        seed: int = 2026,
    ) -> None:
        self.max_lag = max_lag
        self.lambda1 = lambda1
        self.rho_init = rho_init
        self.alpha = alpha
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self._adjacency: np.ndarray | None = None

    def _h(self, matrix: torch.Tensor) -> torch.Tensor:
        product = matrix * matrix
        exponent = torch.linalg.matrix_exp(product)
        return exponent.trace() - product.shape[0]

    def fit(self, train_values: np.ndarray, variable_names: list[str]) -> "DYNOTEARSAdapter":
        torch.manual_seed(self.seed)
        values = np.asarray(train_values, dtype=np.float64)
        L, d = self.max_lag, values.shape[1]
        if len(values) <= L + 1:
            raise ValueError("train series must be longer than max_lag + 1")
        target = torch.from_numpy(values[L:]).float()  # [T-L, D]
        blocks = [target]
        for lag in range(1, L + 1):
            blocks.append(torch.from_numpy(values[L - lag : -lag]).float())
        matrices = torch.stack(blocks, dim=0)  # [L+1, T-L, D]
        # Parameters: A0 (lag 0, DAG-constrained) and A1..AL (lagged, free).
        weights = torch.zeros(L + 1, d, d, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([weights], lr=self.learning_rate)
        rho = self.rho_init
        h_value = torch.tensor(1.0)
        for step in range(self.epochs):
            optimizer.zero_grad()
            prediction = torch.einsum("ltn,lnd->td", matrices, weights)
            mse = ((target - prediction) ** 2).mean()
            sparsity = weights.abs().mean()
            h_value = self._h(weights[0])
            lagrangian = mse + self.lambda1 * sparsity + rho * h_value + 0.5 * self.alpha * h_value * h_value
            lagrangian.backward()
            optimizer.step()
            if step % 50 == 0 and h_value.item() > 1e-4:
                rho *= 2.0
        with torch.no_grad():
            # Threshold tiny weights to zero for a clean adjacency.
            adjacency = torch.clamp(weights.detach().abs() - 1e-3, min=0.0)
            adjacency = torch.where(adjacency > 1e-2, weights.detach().abs(), torch.zeros_like(adjacency))
        self._adjacency = adjacency.numpy().astype(np.float32)
        return self

    def adjacency(self) -> np.ndarray:
        if self._adjacency is None:
            raise RuntimeError("Baseline has not been fit")
        return self._adjacency
