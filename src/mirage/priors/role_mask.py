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

