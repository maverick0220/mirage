from __future__ import annotations

import torch
import torch.nn.functional as functional


def regime_balance_loss(probabilities: torch.Tensor) -> torch.Tensor:
    mean_probability = probabilities.mean(dim=0)
    target = torch.full_like(mean_probability, 1.0 / mean_probability.numel())
    return torch.sum(mean_probability * (torch.log(mean_probability + 1e-8) - torch.log(target)))


def regime_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities * torch.log(probabilities + 1e-8)).sum(dim=-1).mean()


def regime_supervision_loss(
    logits: torch.Tensor, regimes: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy against weak regime labels, when the dataset provides them.

    Uses the raw logits (not the temperature-scaled probabilities) so the CE
    gradient is well behaved.
    """
    return functional.cross_entropy(logits, regimes)


def total_loss(
    reconstruction: torch.Tensor,
    regularization: dict[str, torch.Tensor],
    regime_probabilities: torch.Tensor,
    prior_weight: float,
    sparsity_weight: float,
    delta_weight: float,
    acyclicity_weight: float,
    balance_weight: float = 0.01,
    regime_loss: torch.Tensor | None = None,
    regime_supervision_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    components = {
        "nll": reconstruction.mean(),
        "prior": regularization["prior"],
        "sparsity": regularization["sparsity"],
        "delta": regularization["delta"],
        "acyclicity": regularization["acyclicity"],
        "balance": regime_balance_loss(regime_probabilities),
        "regime_supervision": (
            regime_loss if regime_loss is not None else reconstruction.new_zeros(())
        ),
    }
    value = (
        components["nll"]
        + prior_weight * components["prior"]
        + sparsity_weight * components["sparsity"]
        + delta_weight * components["delta"]
        + acyclicity_weight * components["acyclicity"]
        + balance_weight * components["balance"]
        + regime_supervision_weight * components["regime_supervision"]
    )
    return value, components
