from __future__ import annotations

import numpy as np

from mirage.schemas import MechanismPriorSpec


def prior_conflict_score(adjacency: np.ndarray, prior: MechanismPriorSpec) -> float:
    matrix = np.asarray(adjacency)
    if matrix.ndim == 3:
        matrix = np.max(np.abs(matrix), axis=0) * np.sign(matrix.sum(axis=0))
    forbidden = np.abs(matrix) * (1 - prior.allowed_mask)
    sign_conflict = (
        (np.sign(matrix) != prior.sign_matrix)
        & (prior.sign_matrix != 0)
        & (np.abs(matrix) > 1e-6)
    )
    numerator = forbidden.sum() + (np.abs(matrix) * sign_conflict * prior.confidence).sum()
    return float(numerator / (np.abs(matrix).sum() + 1e-8))

