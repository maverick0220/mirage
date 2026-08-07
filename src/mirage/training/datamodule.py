from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from mirage.data.transforms import RobustScaler
from mirage.data.window import IndustrialWindowDataset


class IndustrialDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path,
        window_size: int,
        batch_size: int = 128,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.window_size = int(window_size)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.scaler = RobustScaler()
        self.feature_names: list[str] = []
        self.datasets: dict[str, IndustrialWindowDataset] = {}

    def prepare_data(self) -> None:
        missing = [
            str(self.data_dir / f"{split}.parquet")
            for split in ("train", "validation", "test")
            if not (self.data_dir / f"{split}.parquet").exists()
        ]
        if missing:
            raise FileNotFoundError(f"Prepared data are missing: {missing}")

    @staticmethod
    def _feature_columns(frame: pd.DataFrame) -> list[str]:
        return [
            str(column)
            for column in frame.select_dtypes(include=[np.number]).columns
            if not str(column).startswith("__")
        ]

    def setup(self, stage: str | None = None) -> None:
        if self.datasets:
            return
        frames = {
            split: pd.read_parquet(self.data_dir / f"{split}.parquet")
            for split in ("train", "validation", "test")
        }
        self.feature_names = self._feature_columns(frames["train"])
        if not self.feature_names:
            raise ValueError("No numeric feature columns found")
        self.scaler.fit(frames["train"][self.feature_names].to_numpy())
        for split, frame in frames.items():
            values = self.scaler.transform(frame[self.feature_names].to_numpy())
            labels = frame["__label"].to_numpy() if "__label" in frame else None
            regimes = frame["__regime"].to_numpy() if "__regime" in frame else None
            if "timestamp" in frame:
                parsed = pd.to_datetime(frame["timestamp"], errors="coerce")
                timestamps = parsed.astype("int64").to_numpy()
            else:
                timestamps = np.arange(len(frame), dtype=np.int64)
            self.datasets[split] = IndustrialWindowDataset(
                values, self.window_size, labels, regimes, timestamps
            )

    @property
    def n_variables(self) -> int:
        return len(self.feature_names)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "median": None if self.scaler.median_ is None else self.scaler.median_.tolist(),
            "scale": None if self.scaler.scale_ is None else self.scaler.scale_.tolist(),
        }

    def save_state(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

