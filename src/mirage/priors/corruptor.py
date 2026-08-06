from __future__ import annotations

import numpy as np

from mirage.schemas import MechanismPriorSpec


def corrupt_prior(
    prior: MechanismPriorSpec,
    edge_flip_rate: float,
    sign_flip_rate: float = 0.0,
    seed: int = 2026,
) -> MechanismPriorSpec:
    rng = np.random.default_rng(seed)
    expected = prior.expected_mask.copy()
    edge_flips = rng.random(expected.shape) < edge_flip_rate
    np.fill_diagonal(edge_flips, False)
    expected[edge_flips] = 1.0 - expected[edge_flips]
    signs = prior.sign_matrix.copy()
    sign_flips = (rng.random(signs.shape) < sign_flip_rate) & (signs != 0)
    signs[sign_flips] *= -1
    confidence = prior.confidence.copy()
    confidence[edge_flips] = np.minimum(confidence[edge_flips], 0.25)
    return MechanismPriorSpec(
        prior.allowed_mask.copy(), expected, signs, confidence, prior.variable_names
    )

