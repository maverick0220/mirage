from __future__ import annotations

from torch import nn


def set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def select_alternating_phase(model: nn.Module, epoch: int, graph_epochs: int = 1, mechanism_epochs: int = 2) -> str:
    """Optional block-coordinate phase switch used in long experiments."""
    cycle = graph_epochs + mechanism_epochs
    graph_phase = epoch % cycle < graph_epochs
    set_trainable(model.graph, graph_phase)
    set_trainable(model.regime_encoder, True)
    set_trainable(model.mechanisms, not graph_phase)
    return "graph" if graph_phase else "mechanism"

