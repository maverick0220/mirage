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
from lightning.pytorch.strategies import DDPStrategy

from mirage.evaluation.event_metrics import event_detection_metrics, pointwise_metrics
from mirage.evaluation.graph_metrics import graph_recovery_metrics
from mirage.evaluation.open_world import open_world_metrics
from mirage.evaluation.root_cause_metrics import root_cause_metrics
from mirage.experiments.artifact_store import ArtifactStore
from mirage.experiments.result_schema import ExperimentResult
from mirage.priors import compile_mechanism_prior
from mirage.priors.corruptor import corrupt_prior
from mirage.priors.role_mask import (
    controller_allowed_mask,
    plant_allowed_mask,
    role_graph_assignment,
)
from mirage.schemas import DynamicCausalGraph, MechanismPriorSpec, VariableRole, VariableSpec
from mirage.scoring.calibration import RegimeConditionalCalibrator
from mirage.training.callbacks import EpochSummaryCallback, GraphSnapshotCallback
from mirage.training.datamodule import IndustrialDataModule
from mirage.training.lightning_module import MIRAGELightningModule
from mirage.utils import dump_json, environment_snapshot, load_yaml, seed_everything


def _compile_prior(
    variables: list[VariableSpec],
    mode: str = "soft",
    corruption_rate: float = 0.0,
    seed: int = 2026,
) -> MechanismPriorSpec:
    """Compile a mechanism prior under none / hard / soft knowledge modes.

    - none: unconstrained (all edges allowed, no expectations)
    - hard: expected edges become hard mask restrictions
    - soft: the default prior, optionally corrupted at a given rate
    """
    size = len(variables)
    base = compile_mechanism_prior(variables)
    if mode == "none":
        return MechanismPriorSpec(
            np.ones((size, size), dtype=np.float32),
            np.zeros((size, size), dtype=np.float32),
            np.zeros((size, size), dtype=np.float32),
            np.zeros((size, size), dtype=np.float32),
            tuple(variable.name for variable in variables),
        )
    if mode == "hard":
        allowed = np.maximum(base.expected_mask, np.eye(size))
        return MechanismPriorSpec(
            allowed.astype(np.float32),
            base.expected_mask,
            base.sign_matrix,
            base.confidence,
            base.variable_names,
        )
    if corruption_rate > 0:
        return corrupt_prior(base, edge_flip_rate=corruption_rate, seed=seed)
    return base


def _masked_to_targets(spec: MechanismPriorSpec, mask: np.ndarray) -> MechanismPriorSpec:
    """Zero the prior matrices on the target columns not owned by this graph."""
    return MechanismPriorSpec(
        spec.allowed_mask * mask,
        spec.expected_mask * mask,
        spec.sign_matrix * mask,
        spec.confidence * mask,
        spec.variable_names,
    )


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
    if "regime_truth" in values:
        frame["regime_truth"] = values["regime_truth"]
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
    prior_mode = str(config.get("prior_mode", "soft"))
    corruption_rate = float(config.get("prior_corruption_rate", 0.0))
    prior = _compile_prior(variables, prior_mode, corruption_rate, seed)
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
    plant_mask_np, _ = role_graph_assignment(variables)
    dual_graph = bool(config.get("dual_graph", True))
    if dual_graph and not (plant_mask_np.any() and (~plant_mask_np).any()):
        dual_graph = False  # nothing to split: fall back to a single graph
    if dual_graph:
        plant_allowed = plant_allowed_mask(variables)
        controller_allowed = controller_allowed_mask(variables)
        plant_prior = _masked_to_targets(prior, plant_allowed)
        controller_prior = _masked_to_targets(prior, controller_allowed)
    else:
        plant_allowed = controller_allowed = None
        plant_prior = controller_prior = prior
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
        plant_mask=torch.from_numpy(plant_mask_np) if dual_graph else None,
        plant_allowed_mask=torch.from_numpy(plant_allowed) if dual_graph else None,
        controller_allowed_mask=torch.from_numpy(controller_allowed) if dual_graph else None,
        plant_prior_expected=torch.from_numpy(plant_prior.expected_mask) if dual_graph else None,
        plant_prior_sign=torch.from_numpy(plant_prior.sign_matrix) if dual_graph else None,
        plant_prior_confidence=torch.from_numpy(plant_prior.confidence) if dual_graph else None,
        controller_prior_expected=torch.from_numpy(controller_prior.expected_mask) if dual_graph else None,
        controller_prior_sign=torch.from_numpy(controller_prior.sign_matrix) if dual_graph else None,
        controller_prior_confidence=torch.from_numpy(controller_prior.confidence) if dual_graph else None,
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
    # 每 epoch 打印一行 train/validation loss 摘要（nohup 后台跑也能看进度）
    callbacks.append(EpochSummaryCallback())
    # 默认 "auto"（Lightning 内置默认）；显式传 None 会被 Lightning 校验拒绝
    strategy = config.get("strategy", "auto")
    if strategy == "ddp_gloo":
        # NCCL 在部分虚拟化 / GPU 直通环境不可用（Duplicate GPU 判定、
        # P2P 传输挂起导致 broadcast 超时）。gloo 走 TCP 通信、不依赖
        # 卡间 P2P；25.7K 参数量级下通信开销可忽略。yaml 配
        # `strategy: ddp_gloo` 即启用。
        strategy = DDPStrategy(process_group_backend="gloo", find_unused_parameters=True)
    elif strategy in ("auto", None):
        # GPU 上 Lightning 单卡/多卡默认走 DDP；MIRAGE 的图掩码路由
        # （torch.where 按 target 拆分 plant/controller 图）在部分
        # Lightning/torch 版本下会触发 DDP "unused parameters" 严格检查而
        # 崩溃（如 V100 + root 容器环境）。find_unused_parameters=True 对
        # 单进程 DDP 零开销，多卡仅一次性建桶成本。
        if config.get("accelerator", "auto") in ("auto", "gpu", "cuda"):
            strategy = "ddp_find_unused_parameters_true"
    trainer = L.Trainer(
        default_root_dir=run_dir,
        max_epochs=int(config.get("max_epochs", 50)),
        accelerator=config.get("accelerator", "auto"),
        devices=config.get("devices", 1),
        strategy=strategy,
        deterministic=bool(config.get("deterministic", True)),
        fast_dev_run=bool(config.get("fast_dev_run", False)),
        limit_train_batches=config.get("limit_train_batches", 1.0),
        limit_val_batches=config.get("limit_val_batches", 1.0),
        gradient_clip_val=1.0,
        logger=CSVLogger(save_dir=run_dir, name="logs"),
        callbacks=callbacks,
        enable_progress_bar=bool(config.get("enable_progress_bar", True)),
        log_every_n_steps=int(config.get("log_every_n_steps", 50)),
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
    def _to_truth_layout(shared: np.ndarray, regime: np.ndarray) -> np.ndarray:
        if hasattr(module.model, "plant_graph"):
            n_regimes = module.model.plant_graph.n_regimes
        else:
            n_regimes = module.model.graph.n_regimes
        shared_by_regime = np.repeat(shared[None], n_regimes, axis=0)
        return np.stack([shared_by_regime, regime], axis=0)

    plant, controller, merged = module.model.regime_graphs()
    shared_plant, shared_ctrl, shared_merged = module.model.shared_graphs()
    plant = plant.detach().cpu().numpy() if plant is not None else None
    controller = controller.detach().cpu().numpy() if controller is not None else None
    merged = merged.detach().cpu().numpy()
    shared_plant = shared_plant.detach().cpu().numpy() if shared_plant is not None else None
    shared_ctrl = shared_ctrl.detach().cpu().numpy() if shared_ctrl is not None else None
    shared_merged = shared_merged.detach().cpu().numpy()
    learned_merged = DynamicCausalGraph(
        _to_truth_layout(shared_merged, merged),
        tuple(data_module.feature_names),
        ("shared", "effective"),
    )
    learned_merged.save(run_dir / "learned_graphs.npz")
    if plant is not None:
        names_tuple = tuple(data_module.feature_names)
        DynamicCausalGraph(
            _to_truth_layout(shared_plant, plant), names_tuple, ("shared", "effective")
        ).save(run_dir / "learned_graphs_plant.npz")
        DynamicCausalGraph(
            _to_truth_layout(shared_ctrl, controller),
            names_tuple,
            ("shared", "effective"),
        ).save(run_dir / "learned_graphs_controller.npz")
        np.savez_compressed(
            run_dir / "graph_assignment.npz",
            plant_mask=np.array(plant_mask_np),
            variable_names=np.array(data_module.feature_names),
        )
    truth_graph = data_dir / "truth_graph.npz"
    if truth_graph.exists():
        shutil.copy2(truth_graph, run_dir / "truth_graph.npz")
    events_source = data_dir / "events.json"
    if events_source.exists():
        shutil.copy2(events_source, run_dir / "events.json")
    train_frame = pd.read_parquet(data_dir / "train.parquet")
    train_regimes = (
        sorted(train_frame["__regime"].unique().tolist())
        if "__regime" in train_frame
        else []
    )
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
            "train_regimes": train_regimes,
            "dual_graph": dual_graph,
            "prior_mode": prior_mode,
            "prior_corruption_rate": corruption_rate,
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
    # --- Structural (graph) metrics: overall + plant/controller split ---
    truth_path = path / "truth_graph.npz"
    learned_path = path / "learned_graphs.npz"
    if truth_path.exists() and learned_path.exists():
        truth = DynamicCausalGraph.load(truth_path)
        learned = DynamicCausalGraph.load(learned_path)
        if truth.weights.shape == learned.weights.shape:
            verification["graph_metrics"] = graph_recovery_metrics(
                truth.weights, learned.weights
            )
            assignment = path / "graph_assignment.npz"
            if assignment.exists():
                plant_mask = np.load(assignment, allow_pickle=False)["plant_mask"].astype(bool)
                plant_truth = truth.weights[..., plant_mask]
                plant_learned = DynamicCausalGraph.load(path / "learned_graphs_plant.npz").weights
                verification["graph_metrics_plant"] = graph_recovery_metrics(
                    plant_truth, plant_learned[..., plant_mask]
                )
                controller_mask = ~plant_mask
                controller_truth = truth.weights[..., controller_mask]
                controller_learned = DynamicCausalGraph.load(
                    path / "learned_graphs_controller.npz"
                ).weights
                verification["graph_metrics_controller"] = graph_recovery_metrics(
                    controller_truth, controller_learned[..., controller_mask]
                )
        else:
            verification["graph_metrics"] = {
                "error": (
                    f"shape mismatch: truth {truth.weights.shape} "
                    f"vs learned {learned.weights.shape}"
                )
            }
    # --- Open-world metrics: known vs unseen regimes ---
    if "regime_truth" in frame.columns and result.get("metadata", {}).get("train_regimes"):
        train_regimes = set(result["metadata"]["train_regimes"])
        known = frame["regime_truth"].isin(train_regimes).to_numpy()
        novelty = frame["regime_confidence"].to_numpy()
        verification["open_world_metrics"] = open_world_metrics(known, novelty)
        unknown_mask = ~known
        if known.any() and unknown_mask.any():
            known_far = frame.loc[known, "prediction"].mean()
            unknown_far = frame.loc[unknown_mask, "prediction"].mean()
            verification["open_world_metrics"]["false_alarm_rate_known"] = float(known_far)
            verification["open_world_metrics"]["false_alarm_rate_unknown"] = float(unknown_far)
    # --- Root-cause metrics from stored events ---
    events_path = path / "events.json"
    feature_names = result.get("metadata", {}).get("features", [])
    if events_path.exists() and feature_names:
        payload = json.loads(events_path.read_text(encoding="utf-8"))
        test_start = int(payload.get("test_start", 0))
        events = payload.get("events", [])
        if events and any(column.startswith("local::") for column in frame.columns):
            local_columns = [f"local::{name}" for name in feature_names]
            local_columns = [column for column in local_columns if column in frame.columns]
            rankings: list[list[str]] = []
            truths: list[str] = []
            for event in events:
                start = event["start_index"] - test_start
                stop = event["end_index"] - test_start
                if start < 0 or stop < start:
                    continue
                window_start = int(frame["index"].min())
                if stop < window_start or start > int(frame["index"].max()):
                    continue
                rows = frame.loc[
                    (frame["index"] >= max(start, window_start)) & (frame["index"] <= stop)
                ]
                if rows.empty:
                    continue
                means = rows[local_columns].mean(axis=0)
                ranking = [name.split("::", 1)[1] for name in means.sort_values(ascending=False).index]
                rankings.append(ranking)
                truths.append(event["root_cause"])
            if rankings:
                verification["root_cause_metrics"] = root_cause_metrics(rankings, truths)
    dump_json(verification, path / "evaluation_verification.json")
    return verification

