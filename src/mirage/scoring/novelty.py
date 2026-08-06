from __future__ import annotations

import numpy as np


def regime_novelty_score(regime_probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(regime_probabilities, dtype=float), 1e-12, 1.0)
    n_regimes = probabilities.shape[-1]
    if n_regimes <= 1:
        # A single regime carries no novelty information: entropy is undefined
        # (log 1 = 0), so fall back to the uncertainty term only.
        return (1.0 - probabilities.max(axis=-1)).astype(float)
    normalized_entropy = -(probabilities * np.log(probabilities)).sum(axis=-1) / np.log(
        n_regimes
    )
    uncertainty = 1.0 - probabilities.max(axis=-1)
    return 0.5 * normalized_entropy + 0.5 * uncertainty

