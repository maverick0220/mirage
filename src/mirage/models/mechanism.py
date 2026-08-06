from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as functional


class LocalMechanismNetwork(nn.Module):
    """Target-wise nonlinear mechanism driven by lagged causal parent messages.

    The graph tensor has axes [batch, lag, source, target] with lag 0 being the
    instantaneous slice. Lag-0 messages are built from the *target* values of the
    other variables (masked so no self-edge contributes); lagged messages are
    built from history.
    """

    def __init__(self, n_variables: int, max_lag: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.n_variables = n_variables
        self.max_lag = max_lag
        # Inputs: lag-0 parent messages + lag-1..L parent messages + own lag-1 state.
        self.network = nn.Sequential(
            nn.Linear(max_lag + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.target_bias = nn.Parameter(torch.zeros(n_variables))

    def forward(
        self, history: torch.Tensor, target: torch.Tensor, graph: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history.shape[1] < self.max_lag:
            raise ValueError("History is shorter than max_lag")
        lagged = torch.stack(
            [history[:, -lag, :] for lag in range(1, self.max_lag + 1)], dim=1
        )
        # [B, L, D] messages over lag 1..L
        parent_messages = torch.einsum("bls,blst->blt", lagged, graph[:, 1:])
        # [B, D] simultaneous parent messages from the target step (lag 0).
        instantaneous = torch.einsum("bs,bst->bt", target, graph[:, 0])
        own_state = history[:, -1, :].unsqueeze(-1)
        features = torch.cat(
            [parent_messages.transpose(1, 2), instantaneous.unsqueeze(-1), own_state],
            dim=-1,
        )
        parameters = self.network(features)
        location = parameters[..., 0] + self.target_bias
        scale = functional.softplus(parameters[..., 1]) + 1e-4
        return location, scale
