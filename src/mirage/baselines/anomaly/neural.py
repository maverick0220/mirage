"""Unified training/scoring loop for the neural anomaly-detection baselines."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mirage.baselines.anomaly.neural_models import build_neural_predictor
from mirage.utils import seed_everything


class NeuralAnomalyBaseline:
    """Window-to-next-step predictor whose anomaly score is the prediction MSE.

    Aligns with MIRAGE scoring semantics: a window [t-W, t) predicts t, and the
    score at time t is the per-window prediction error (optionally top-q over
    non-context variables, mirroring the MIRAGE device score).
    """

    name = "neural"

    def __init__(
        self,
        model_name: str,
        window_size: int = 32,
        epochs: int = 10,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        seed: int = 2026,
        device: str | None = None,
        context_indices: list[int] | None = None,
        score_topq: float = 0.5,
        **model_kwargs,
    ) -> None:
        self.name = model_name
        self.window_size = int(window_size)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_kwargs = model_kwargs
        self.context_indices = context_indices or []
        self.score_topq = float(score_topq)
        self._model: nn.Module | None = None
        self._n_features: int | None = None

    def _windows(self, values: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        data = np.asarray(values, dtype=np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(data, self.window_size, axis=0)[:-1]
        targets = data[self.window_size :]
        return (
            torch.from_numpy(np.ascontiguousarray(windows)),
            torch.from_numpy(np.ascontiguousarray(targets)),
        )

    def _device_score(self, errors: torch.Tensor) -> torch.Tensor:
        if self.context_indices:
            keep = [i for i in range(errors.shape[-1]) if i not in self.context_indices]
            if keep:
                errors = errors[:, keep]
        k = max(1, min(errors.shape[-1], int(self.score_topq * errors.shape[-1])))
        return torch.topk(errors, k, dim=-1).values.mean(dim=-1)

    def fit(self, train_values: np.ndarray, variable_names: list[str] | None = None) -> "NeuralAnomalyBaseline":
        values = np.asarray(train_values, dtype=np.float32)
        if len(values) <= self.window_size:
            raise ValueError("train series must be longer than window_size")
        self._n_features = values.shape[1]
        seed_everything(self.seed)
        self._model = build_neural_predictor(
            self.name, self._n_features, self.window_size, **self.model_kwargs
        ).to(self.device)
        windows, targets = self._windows(values)
        loader = DataLoader(
            TensorDataset(windows, targets),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self._model.train()
        for _ in range(self.epochs):
            for batch_windows, batch_targets in loader:
                batch_windows = batch_windows.to(self.device)
                batch_targets = batch_targets.to(self.device)
                optimizer.zero_grad()
                if self.name.lower().replace("-", "").replace("_", "") == "omnianomaly":
                    reconstruction, mean, logvar = self._model(batch_windows)
                    reconstruction = reconstruction[:, -1]
                    mse = ((reconstruction - batch_targets) ** 2).mean()
                    kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).mean()
                    loss = mse + 0.1 * kl
                else:
                    prediction = self._model(batch_windows)
                    loss = ((prediction - batch_targets) ** 2).mean()
                loss.backward()
                optimizer.step()
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Baseline has not been fit")
        self._model.eval()
        windows, targets = self._windows(np.asarray(values, dtype=np.float32))
        errors = []
        with torch.no_grad():
            for batch_windows, batch_targets in DataLoader(
                TensorDataset(windows, targets), batch_size=self.batch_size, shuffle=False
            ):
                batch_windows = batch_windows.to(self.device)
                batch_targets = batch_targets.to(self.device)
                if self.name.lower().replace("-", "").replace("_", "") == "omnianomaly":
                    reconstruction, mean, logvar = self._model(batch_windows)
                    reconstruction = reconstruction[:, -1]
                    mse = ((reconstruction - batch_targets) ** 2)
                    kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=-1)
                    errors.append((mse + 0.1 * kl).cpu())
                else:
                    prediction = self._model(batch_windows)
                    errors.append(((prediction - batch_targets) ** 2).cpu())
        stacked = torch.cat(errors, dim=0)
        return self._device_score(stacked).numpy()
