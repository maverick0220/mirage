from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np


class GraphSnapshotCallback(L.Callback):
    def __init__(self, output_dir: str | Path, every_n_epochs: int = 1) -> None:
        self.output_dir = Path(output_dir)
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        graph = pl_module.model.graph.regime_graphs().detach().cpu().numpy()
        np.savez_compressed(self.output_dir / f"epoch-{trainer.current_epoch:04d}.npz", weights=graph)

