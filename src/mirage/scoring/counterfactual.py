from __future__ import annotations

from collections.abc import Callable

import numpy as np


def counterfactual_recovery(
    window: np.ndarray,
    variable_index: int,
    reference_value: float,
    score_function: Callable[[np.ndarray], float],
) -> dict[str, float]:
    observed = np.asarray(window, dtype=np.float32)
    factual_score = float(score_function(observed))
    counterfactual = observed.copy()
    counterfactual[-1, variable_index] = reference_value
    counterfactual_score = float(score_function(counterfactual))
    recovery = max(0.0, factual_score - counterfactual_score)
    return {
        "factual_score": factual_score,
        "counterfactual_score": counterfactual_score,
        "recovery": recovery,
    }

