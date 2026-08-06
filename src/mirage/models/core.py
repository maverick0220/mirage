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
    """Role-aware dual-graph core (plant + controller) or a single-graph variant.

    When ``plant_mask`` is provided the model learns TWO graphs:
    - the plant graph drives process/output/feedback/setpoint targets;
    - the controller graph drives actuator-command targets.
    The effective graph passed to the mechanism network is assembled by routing
    each target variable to its own graph's columns. When ``plant_mask`` is None
    a single shared graph is used (backward-compatible behaviour).
    """

    def __init__(
        self,
        n_variables: int,
        n_regimes: int = 3,
        max_lag: int = 3,
        hidden_dim: int = 64,
        regime_temperature: float = 1.0,
        student_t_df: float = 5.0,
        allowed_mask: torch.Tensor | None = None,
        plant_mask: torch.Tensor | None = None,
        plant_allowed_mask: torch.Tensor | None = None,
        controller_allowed_mask: torch.Tensor | None = None,
        plant_prior_expected: torch.Tensor | None = None,
        plant_prior_sign: torch.Tensor | None = None,
        plant_prior_confidence: torch.Tensor | None = None,
        controller_prior_expected: torch.Tensor | None = None,
        controller_prior_sign: torch.Tensor | None = None,
        controller_prior_confidence: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.n_variables = n_variables
        self.student_t_df = student_t_df
        self.regime_encoder = RegimeEncoder(
            n_variables, n_regimes, hidden_dim, regime_temperature
        )
        if plant_mask is None:
            self.plant_mask: torch.Tensor | None = None
            self.graph = DualGraphParameterization(
                n_variables, n_regimes, max_lag, allowed_mask=allowed_mask
            )
        else:
            self.register_buffer("plant_mask", torch.as_tensor(plant_mask, dtype=torch.bool))
            self.plant_graph = DualGraphParameterization(
                n_variables, n_regimes, max_lag, allowed_mask=plant_allowed_mask
            )
            self.controller_graph = DualGraphParameterization(
                n_variables, n_regimes, max_lag, allowed_mask=controller_allowed_mask
            )
            size = (n_variables, n_variables)
            for name, value in {
                "plant_prior_expected": plant_prior_expected,
                "plant_prior_sign": plant_prior_sign,
                "plant_prior_confidence": plant_prior_confidence,
                "controller_prior_expected": controller_prior_expected,
                "controller_prior_sign": controller_prior_sign,
                "controller_prior_confidence": controller_prior_confidence,
            }.items():
                self.register_buffer(
                    name,
                    torch.zeros(*size) if value is None else value.float(),
                )
        self.mechanisms = LocalMechanismNetwork(n_variables, max_lag, hidden_dim)

    def _effective_graph(self, regime_probabilities: torch.Tensor) -> torch.Tensor:
        if self.plant_mask is None:
            return self.graph.effective_graph(regime_probabilities)
        plant = self.plant_graph.effective_graph(regime_probabilities)
        controller = self.controller_graph.effective_graph(regime_probabilities)
        # [1, 1, 1, D] broadcast over the TARGET dimension of [B, L+1, S, T].
        mask = self.plant_mask[None, None, None, :]
        return torch.where(mask, plant, controller)

    def forward(self, history: torch.Tensor, target: torch.Tensor) -> MIRAGEOutput:
        regime_probabilities, regime_logits = self.regime_encoder(history)
        effective_graph = self._effective_graph(regime_probabilities)
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

    def regularization(
        self,
        prior_expected: torch.Tensor | None = None,
        prior_sign: torch.Tensor | None = None,
        prior_confidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.plant_mask is None:
            return self.graph.regularization(prior_expected, prior_sign, prior_confidence)
        plant = self.plant_graph.regularization(
            self.plant_prior_expected, self.plant_prior_sign, self.plant_prior_confidence
        )
        controller = self.controller_graph.regularization(
            self.controller_prior_expected,
            self.controller_prior_sign,
            self.controller_prior_confidence,
        )
        return {key: (plant[key] + controller[key]) * 0.5 for key in plant}

    def shared_graphs(self) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        """Return (plant_shared, controller_shared, merged_shared)."""
        if self.plant_mask is None:
            return None, None, self.graph.shared_graph()
        plant = self.plant_graph.shared_graph()
        controller = self.controller_graph.shared_graph()
        mask = self.plant_mask[None, None, :]
        return plant, controller, torch.where(mask, plant, controller)

    def regime_graphs(self) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        """Return (plant_regimes, controller_regimes, merged_regimes) [K, L+1, D, D]."""
        if self.plant_mask is None:
            return None, None, self.graph.regime_graphs()
        plant = self.plant_graph.regime_graphs()
        controller = self.controller_graph.regime_graphs()
        mask = self.plant_mask[None, None, None, :]
        return plant, controller, torch.where(mask, plant, controller)
