from __future__ import annotations

import numpy as np

from mirage.priors.role_mask import role_allowed_mask
from mirage.schemas import MechanismPriorSpec, VariableSpec


def compile_mechanism_prior(
    variables: list[VariableSpec],
    expected_edges: list[tuple[str, str, int, float]] | None = None,
) -> MechanismPriorSpec:
    names = tuple(variable.name for variable in variables)
    lookup = {name: index for index, name in enumerate(names)}
    allowed = role_allowed_mask(variables)
    expected = np.zeros_like(allowed)
    signs = np.zeros_like(allowed)
    confidence = np.zeros_like(allowed)
    for source, target, sign, weight in expected_edges or []:
        if source not in lookup or target not in lookup:
            raise KeyError(f"Unknown prior edge: {source} -> {target}")
        source_index, target_index = lookup[source], lookup[target]
        expected[source_index, target_index] = 1.0
        signs[source_index, target_index] = float(np.sign(sign))
        confidence[source_index, target_index] = float(np.clip(weight, 0, 1))
    return MechanismPriorSpec(allowed, expected, signs, confidence, names)

