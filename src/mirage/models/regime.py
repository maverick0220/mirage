from __future__ import annotations

import torch
from torch import nn


class RegimeEncoder(nn.Module):
    def __init__(
        self,
        n_variables: int,
        n_regimes: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_regimes = n_regimes
        self.temperature = temperature
        self.encoder = nn.Sequential(
            nn.Conv1d(n_variables, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden_dim, n_regimes)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if history.ndim != 3:
            raise ValueError("history must be [batch, window, variable]")
        features = self.encoder(history.transpose(1, 2)).squeeze(-1)
        logits = self.head(features)
        probabilities = torch.softmax(logits / max(self.temperature, 1e-4), dim=-1)
        return probabilities, logits

