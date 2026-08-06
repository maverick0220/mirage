from __future__ import annotations

from torch import nn


def set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def _graph_modules(model: nn.Module) -> list[nn.Module]:
    if hasattr(model, "plant_graph"):
        return [model.plant_graph, model.controller_graph]
    return [model.graph]


def select_alternating_phase(model: nn.Module, epoch: int, graph_epochs: int = 1, mechanism_epochs: int = 2) -> str:
    """Optional block-coordinate phase switch used in long experiments.

    Works for both the single-graph and the dual plant/controller configuration.
    """
    cycle = graph_epochs + mechanism_epochs
    graph_phase = epoch % cycle < graph_epochs
    for module in _graph_modules(model):
        set_trainable(module, graph_phase)
    set_trainable(model.regime_encoder, True)
    set_trainable(model.mechanisms, not graph_phase)
    return "graph" if graph_phase else "mechanism"
