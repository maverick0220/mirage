from __future__ import annotations

import numpy as np

from mirage.schemas import VariableRole, VariableSpec


def role_allowed_mask(variables: list[VariableSpec]) -> np.ndarray:
    """Compile physically plausible source→target role constraints."""

    roles = [variable.role for variable in variables]
    size = len(roles)
    mask = np.ones((size, size), dtype=np.float32)
    np.fill_diagonal(mask, 1.0)
    for source, source_role in enumerate(roles):
        for target, target_role in enumerate(roles):
            if target_role == VariableRole.CONTEXT and source_role != VariableRole.CONTEXT:
                mask[source, target] = 0.0
            if target_role == VariableRole.SETPOINT and source_role not in {
                VariableRole.CONTEXT,
                VariableRole.SETPOINT,
            }:
                mask[source, target] = 0.0
            if source_role == VariableRole.OUTPUT and target_role == VariableRole.ACTUATOR_COMMAND:
                mask[source, target] = 1.0
            if source_role == VariableRole.ACTUATOR_COMMAND and target_role == VariableRole.CONTEXT:
                mask[source, target] = 0.0
    return mask


# Roles whose target variables belong to the controller graph (paper: f^C).
_CONTROLLER_TARGET_ROLES = {VariableRole.ACTUATOR_COMMAND}


def role_graph_assignment(variables: list[VariableSpec]) -> tuple[np.ndarray, np.ndarray]:
    """Split variables into plant / controller graphs by target role.

    Controller targets are actuator commands (the controller's output); everything
    else (context, setpoints, feedback, process states, outputs) is a plant node.
    Returns (plant_mask, controller_mask), boolean arrays over variables.
    """
    roles = [variable.role for variable in variables]
    controller_mask = np.array([role in _CONTROLLER_TARGET_ROLES for role in roles])
    plant_mask = ~controller_mask
    return plant_mask, controller_mask


def plant_allowed_mask(variables: list[VariableSpec]) -> np.ndarray:
    """Role mask restricted to plant targets (controller columns zeroed)."""
    mask = role_allowed_mask(variables)
    _, controller_mask = role_graph_assignment(variables)
    mask[:, controller_mask] = 0.0
    return mask


def controller_allowed_mask(variables: list[VariableSpec]) -> np.ndarray:
    """Role mask restricted to controller targets (plant columns zeroed)."""
    mask = role_allowed_mask(variables)
    plant_mask, _ = role_graph_assignment(variables)
    mask[:, plant_mask] = 0.0
    return mask

