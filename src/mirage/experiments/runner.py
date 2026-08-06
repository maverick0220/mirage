from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from mirage.evaluation.event_metrics import event_detection_metrics, pointwise_metrics
from mirage.evaluation.graph_metrics import graph_recovery_metrics
from mirage.experiments.artifact_store import ArtifactStore
from mirage.experiments.result_schema import ExperimentResult
from mirage.priors import compile_mechanism_prior
from mirage.schemas import DynamicCausalGraph, VariableRole, VariableSpec
from mirage.scoring.calibration import RegimeConditionalCalibrator
from mirage.training.callbacks import GraphSnapshotCallback
from mirage.training.datamodule import IndustrialDataModule
from mirage.training.lightning_module import MIRAGELightningModule
from mirage.utils import dump_json, environment_snapshot, load_yaml, seed_everything


def _load_variables(data_dir: Path, feature_names: list[str]) -> list[VariableSpec]:
    path = data_dir / "variables.json"
    if not path.exists():
        return [VariableSpec(name) for name in feature_names]
    values = json.loads(path.read_text(encoding="utf-8"))
    lookup = {value["name"]: VariableSpec.from_dict(value) for value in values}
    return [lookup.get(name, VariableSpec(name)) for name in feature_names]


def _combine_predictions(outputs: list[dict[str, torch.Tensor]]) -> dict[str, np.ndarray]:
    if not outputs:
        raise RuntimeError(
            "Prediction produced no outputs; the validation/test dataloaders are empty"
        )
    keys = set.intersection(*(set(output) for output in outputs))
    return {
        key: torch.cat([output[key].detach().cpu() for output in outputs], dim=0).numpy()
        for key in keys
    }


def _prediction_frame(values: dict[str, np.ndarray], names: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({"index": values["index"], "score": values["score"]})
    if "label" in values:
        frame["label"] = values["label"]
    if "regime_probability" in values:
        frame["regime"] = values["regime_probability"].argmax(axis=1)
        frame["regime_confidence"] = values["regime_probability"].max(axis=1)
    for index, name in enumerate(names):
        frame[f"local::{name}"] = values["local_score"][:, index]
    return frame


def train_experiment(config_path: str | Path) -> ExperimentResult:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    model_config_path = Path(config["model_config"])
    model_config = load_yaml(model_config_path)
    seed = int(config.get("seed", 2026))
    seed_everything(seed, bool(config.get("deterministic", True)))
    L.seed_everything(seed, workers=True)
    data_dir = Path(config["data_dir"])
    run_dir = Path(config["run_dir"])
    store = ArtifactStore(run_dir)
    store.copy_config(config_path, "experiment.yaml")
    store.copy_config(model_config_path, "model.yaml")

    data_module = IndustrialDataModule(
        data_dir=data_dir,
        window_size=int(model_config["window_size"]),
        batch_size=int(config.get("batch_size", 128)),
        num_workers=int(config.get("num_workers", 0)),
    )
    data_module.prepare_data()
    data_module.setup("fit")
    variables = _load_variables(data_dir, data_module.feature_names)
    prior = compile_mechanism_prior(variables)
    window_size = int(model_config["window_size"])
    max_lag = int(model_config.get("max_lag", 3))
    if window_size < max_lag:
        raise ValueError(
            f"window_size ({window_size}) must be >= max_lag ({max_lag}); "
            "otherwise the mechanism network indexes out of the history window"
        )
    context_indices = [
        index
        for index, variable in enumerate(variables)
        if variable.role == VariableRole.CONTEXT
    ]
    module = MIRAGELightningModule(
        n_variables=data_module.n_variables,
        n_regimes=int(model_config.get("n_regimes", 3)),
        max_lag=max_lag,
        hidden_dim=int(model_config.get("hidden_dim", 64)),
        regime_temperature=float(model_config.get("regime_temperature", 1.0)),
        student_t_df=float(model_config.get("student_t_df", 5.0)),
        prior_weight=float(model_config.get("prior_weight", 0.05)),
        sparsity_weight=float(model_config.get("sparsity_weight", 0.001)),
        delta_weight=float(model_config.get("delta_weight", 0.001)),
        acyclicity_weight=float(model_config.get("acyclicity_weight", 0.0)),
        balance_weight=float(model_config.get("balance_weight", 0.01)),
        regime_supervision_weight=float(model_config.get("regime_supervision_weight", 0.0)),
        alternating=bool(model_config.get("alternating", False)),
        score_topq=float(model_config.get("score_topq", 0.5)),
        learning_rate=float(model_config.get("learning_rate", 1e-3)),
        weight_decay=float(model_config.get("weight_decay", 1e-4)),
        allowed_mask=torch.from_numpy(prior.allowed_mask),
        prior_expected=torch.from_numpy(prior.expected_mask),
        prior_sign=torch.from_numpy(prior.sign_matrix),
        prior_confidence=torch.from_numpy(prior.confidence),
        context_indices=context_indices,
    )
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best-{epoch:03d}",
        monitor="validation/loss",
        mode="min",
        save_last=True,
    )
    callbacks = [checkpoint, LearningRateMonitor(logging_interval="epoch"), GraphSnapshotCallback(run_dir / "graphs")]
    if int(config.get("max_epochs", 1)) > 2:
        callbacks.append(EarlyStopping(monitor="validation/loss", mode="min", patience=10))
    trainer = L.Trainer(
        default_root_dir=run_dir,
        max_epochs=int(config.get("max_epochs", 50)),
        accelerator=config.get("accelerator", "auto"),
        devices=config.get("devices", 1),
        deterministic=bool(config.get("deterministic", True)),
        fast_dev_run=bool(config.get("fast_dev_run", False)),
        limit_train_batches=config.get("limit_train_batches", 1.0),
        limit_val_batches=config.get("limit_val_batches", 1.0),
        gradient_clip_val=1.0,
        logger=CSVLogger(save_dir=run_dir, name="logs"),
        callbacks=callbacks,
        enable_progress_bar=False,
        log_every_n_steps=1,
    )
    trainer.fit(module, datamodule=data_module)
    # Evaluate with the BEST checkpoint (validation-loss-optimal), not the last
    # epoch's weights; otherwise the ModelCheckpoint callback is decorative.
    if checkpoint.best_model_path and Path(checkpoint.best_model_path).exists():
        payload = torch.load(checkpoint.best_model_path, map_location="cpu")
        module.load_state_dict(payload["state_dict"])
    validation_outputs = trainer.predict(module, dataloaders=data_module.val_dataloader())
    test_outputs = trainer.predict(module, dataloaders=data_module.test_dataloader())
    validation = _combine_predictions(validation_outputs)
    test = _combine_predictions(test_outputs)
    calibration = RegimeConditionalCalibrator(quantile=0.99)
    validation_regime = validation["regime_probability"].argmax(axis=1)
    calibration.fit(validation["score"], validation_regime)
    test_regime = test["regime_probability"].argmax(axis=1)
    thresholds = calibration.threshold(test_regime)
    test_frame = _prediction_frame(test, data_module.feature_names)
    test_frame["threshold"] = thresholds
    test_frame["prediction"] = test_frame["score"].to_numpy() >= thresholds
    score_path = run_dir / "predictions" / "test_scores.parquet"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    test_frame.to_parquet(score_path, index=False)
    data_module.save_state(run_dir / "data_state.json")
    # Store the learned graphs in the SAME [graph_type, regime, lag, source, target]
    # layout as the truth graph, so structural metrics are directly comparable.
    shared = module.model.graph.shared_graph().detach().cpu().numpy()
    regime = module.model.graph.regime_graphs().detach().cpu().numpy()
    shared_by_regime = np.repeat(shared[None], module.model.graph.n_regimes, axis=0)
    learned = DynamicCausalGraph(
        np.stack([shared_by_regime, regime], axis=0),
        tuple(data_module.feature_names),
        ("shared", "effective"),
    )
    learned.save(run_dir / "learned_graphs.npz")
    truth_graph = data_dir / "truth_graph.npz"
    if truth_graph.exists():
        shutil.copy2(truth_graph, run_dir / "truth_graph.npz")
    metrics: dict[str, float] = {}
    if "label" in test:
        # Use the per-row regime thresholds everywhere (consistent with the
        # `prediction` column stored in test_scores.parquet).
        metrics.update(pointwise_metrics(test["label"], test["score"], thresholds))
        metrics.update(event_detection_metrics(test["label"], test["score"], thresholds))
    artifacts = {
        "scores": str(score_path),
        "graphs": str(run_dir / "learned_graphs.npz"),
        "checkpoint": str(checkpoint.best_model_path or checkpoint.last_model_path),
    }
    result = ExperimentResult(
        run_name=str(config.get("name", config_path.stem)),
        status="completed",
        metrics=metrics,
        artifacts=artifacts,
        metadata={
            "features": data_module.feature_names,
            "calibration_global_threshold": calibration.global_threshold_,
            "environment": environment_snapshot(),
        },
    )
    dump_json(result.to_dict(), run_dir / "result.json")
    store.manifest()
    return result


def evaluate_run(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir)
    result_path = path / "result.json"
    scores_path = path / "predictions" / "test_scores.parquet"
    if not result_path.exists() or not scores_path.exists():
        raise FileNotFoundError("Run is incomplete: result.json or test_scores.parquet is missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    frame = pd.read_parquet(scores_path)
    verification = {
        "rows": int(len(frame)),
        "finite_scores": bool(np.isfinite(frame["score"]).all()),
        "prediction_count": int(frame["prediction"].sum()),
        "metrics": result.get("metrics", {}),
    }
    truth_path = path / "truth_graph.npz"
    learned_path = path / "learned_graphs.npz"
    if truth_path.exists() and learned_path.exists():
        truth = DynamicCausalGraph.load(truth_path)
        learned = DynamicCausalGraph.load(learned_path)
        if truth.weights.shape == learned.weights.shape:
            verification["graph_metrics"] = graph_recovery_metrics(
                truth.weights, learned.weights
            )
        else:
            verification["graph_metrics"] = {
                "error": (
                    f"shape mismatch: truth {truth.weights.shape} "
                    f"vs learned {learned.weights.shape}"
                )
            }
    dump_json(verification, path / "evaluation_verification.json")
    return verification

