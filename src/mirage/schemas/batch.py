from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TimeSeriesBatch:
    """A batch with values shaped [batch, window, variable]."""

    values: torch.Tensor
    target: torch.Tensor
    timestamps: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    regimes: torch.Tensor | None = None

    def validate(self) -> "TimeSeriesBatch":
        if self.values.ndim != 3:
            raise ValueError("values must be [batch, window, variable]")
        if self.target.shape != (self.values.shape[0], self.values.shape[2]):
            raise ValueError("target must be [batch, variable]")
        return self

