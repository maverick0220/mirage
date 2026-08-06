from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class IndustrialWindowDataset(Dataset):
    """Chronological sliding windows returning history and next-step target."""

    def __init__(
        self,
        values: np.ndarray,
        window_size: int,
        labels: np.ndarray | None = None,
        regimes: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
    ) -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.window_size = int(window_size)
        self.labels = None if labels is None else np.asarray(labels)
        self.regimes = None if regimes is None else np.asarray(regimes)
        self.timestamps = None if timestamps is None else np.asarray(timestamps)
        if self.values.ndim != 2:
            raise ValueError("values must be [time, variable]")
        if len(self.values) <= self.window_size:
            raise ValueError("series must be longer than window_size")

    def __len__(self) -> int:
        return len(self.values) - self.window_size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        target_index = index + self.window_size
        item = {
            "values": torch.from_numpy(self.values[index:target_index]),
            "target": torch.from_numpy(self.values[target_index]),
            "index": torch.tensor(target_index, dtype=torch.long),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[target_index], dtype=torch.float32)
        if self.regimes is not None:
            item["regime"] = torch.tensor(self.regimes[target_index], dtype=torch.long)
        if self.timestamps is not None:
            item["timestamp"] = torch.tensor(self.timestamps[target_index], dtype=torch.long)
        return item

