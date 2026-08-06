from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MechanismPriorSpec:
    allowed_mask: np.ndarray
    expected_mask: np.ndarray
    sign_matrix: np.ndarray
    confidence: np.ndarray
    variable_names: tuple[str, ...]

    def __post_init__(self) -> None:
        size = len(self.variable_names)
        for name in ("allowed_mask", "expected_mask", "sign_matrix", "confidence"):
            value = np.asarray(getattr(self, name))
            if value.shape != (size, size):
                raise ValueError(f"{name} must be [{size}, {size}]")
            setattr(self, name, value.astype(np.float32))
        if np.any((self.allowed_mask < 0) | (self.allowed_mask > 1)):
            raise ValueError("allowed_mask must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable_names": list(self.variable_names),
            "allowed_mask": self.allowed_mask.tolist(),
            "expected_mask": self.expected_mask.tolist(),
            "sign_matrix": self.sign_matrix.tolist(),
            "confidence": self.confidence.tolist(),
        }

