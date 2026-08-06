from pathlib import Path

import lightning as L

from mirage.data.sources.synthetic import ClosedLoopSCMGenerator, SyntheticSCMConfig
from mirage.training.datamodule import IndustrialDataModule
from mirage.training.lightning_module import MIRAGELightningModule


def test_one_batch_training_and_prediction(tmp_path):
    data_dir = tmp_path / "data"
    ClosedLoopSCMGenerator(
        SyntheticSCMConfig(seed=5, n_steps=260, n_variables=6, n_regimes=2, max_lag=2)
    ).prepare(data_dir)
    data = IndustrialDataModule(data_dir, window_size=12, batch_size=16)
    data.prepare_data()
    data.setup()
    module = MIRAGELightningModule(6, n_regimes=2, max_lag=2, hidden_dim=8)
    trainer = L.Trainer(
        default_root_dir=tmp_path / "run",
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
    )
    trainer.fit(module, datamodule=data)
    output = trainer.predict(module, dataloaders=data.test_dataloader(), return_predictions=True)
    assert output and output[0]["local_score"].shape[1] == 6

