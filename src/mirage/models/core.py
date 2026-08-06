from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mirage.models.distributions import student_t_nll
from mirage.models.dual_graph import DualGraphParameterization
from mirage.models.mechanism import LocalMechanismNetwork
from mirage.models.regime import RegimeEncoder


@dataclass
class MIRAGEOutput:
    location: torch.Tensor
    scale: torch.Tensor
    local_nll: torch.Tensor
    regime_probabilities: torch.Tensor
    regime_logits: torch.Tensor
    effective_graph: torch.Tensor


class MIRAGECore(nn.Module):
    def __init__(
        self,
        n_variables: int,
        n_regimes: int = 3,
        max_lag: int = 3,
        hidden_dim: int = 64,
        regime_temperature: float = 1.0,
        student_t_df: float = 5.0,
        allowed_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.n_variables = n_variables
        self.student_t_df = student_t_df
        self.regime_encoder = RegimeEncoder(
            n_variables, n_regimes, hidden_dim, regime_temperature
        )
        self.graph = DualGraphParameterization(
            n_variables, n_regimes, max_lag, allowed_mask=allowed_mask
        )
        self.mechanisms = LocalMechanismNetwork(n_variables, max_lag, hidden_dim)

    def forward(self, history: torch.Tensor, target: torch.Tensor) -> MIRAGEOutput:
        regime_probabilities, regime_logits = self.regime_encoder(history)
        effective_graph = self.graph.effective_graph(regime_probabilities)
        location, scale = self.mechanisms(history, target, effective_graph)
        local_nll = student_t_nll(target, location, scale, self.student_t_df)
        return MIRAGEOutput(
            location=location,
            scale=scale,
            local_nll=local_nll,
            regime_probabilities=regime_probabilities,
            regime_logits=regime_logits,
            effective_graph=effective_graph,
        )

