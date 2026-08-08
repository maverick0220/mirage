from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mirage.data.sources.base import SourceBundle, TimeSeriesSource
from mirage.data.split import chronological_slices
from mirage.schemas import DynamicCausalGraph, EventRecord, VariableRole, VariableSpec
from mirage.utils import dump_json


@dataclass
class SyntheticSCMConfig:
    seed: int = 2026
    n_steps: int = 12000
    n_variables: int = 12
    n_regimes: int = 3
    max_lag: int = 3
    sample_period_seconds: int = 5
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    anomaly_rate: float = 0.035
    split_embargo: int = 0
    train_regimes: tuple[int, ...] | None = None
    regime_delta_scale: float = 0.2


class ClosedLoopSCMGenerator(TimeSeriesSource):
    """Stable closed-loop nonlinear SCM with regime switching and event truth."""

    def __init__(self, config: SyntheticSCMConfig) -> None:
        if config.n_variables < 6:
            raise ValueError("n_variables must be at least 6 for the closed-loop roles")
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._bundle: SourceBundle | None = None
        self._truth: DynamicCausalGraph | None = None

    def _variables(self) -> list[VariableSpec]:
        fixed = [
            ("load_context", VariableRole.CONTEXT),
            ("setpoint", VariableRole.SETPOINT),
            ("actuator_command", VariableRole.ACTUATOR_COMMAND),
            ("actuator_feedback", VariableRole.ACTUATOR_FEEDBACK),
        ]
        remaining = []
        for index in range(self.config.n_variables - len(fixed)):
            role = VariableRole.OUTPUT if index >= self.config.n_variables - len(fixed) - 2 else VariableRole.PROCESS
            remaining.append((f"process_{index}", role))
        return [VariableSpec(name, role, unit="a.u.", subsystem="closed_loop") for name, role in fixed + remaining]

    def _graphs(self, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        d = len(names)
        lags = self.config.max_lag + 1
        shared = np.zeros((lags, d, d), dtype=np.float32)
        # Coefficients below are EXACTLY the ones used by the simulation loop in
        # load() (no separate exogenous terms on these edges), so the stored truth
        # graph matches the data-generation mechanism.
        shared[1, 0, 1] = 0.55
        shared[1, d - 1, 2] = -0.28
        shared[1, 1, 2] = 0.55
        shared[1, 2, 3] = 0.70
        shared[1, 3, 4] = 0.42
        for index in range(4, d):
            shared[1, index, index] = 0.30
            if index + 1 < d:
                shared[1, index, index + 1] = 0.32
        shared[1, 0, 4] = 0.22
        if self.config.max_lag >= 2:
            shared[2, 3, min(5, d - 1)] = 0.16
        deltas = np.zeros((self.config.n_regimes, lags, d, d), dtype=np.float32)
        # regime 间系数差异幅度：0.08 太小——unseen regime（训练未见工况）与
        # 已知工况分布高度重叠，模型无法区分，novelty AUROC 会停留在 0.5
        # （掷硬币）。加大到 0.2 让工况差异可检测，open-world 留出才有意义。
        delta_scale = float(getattr(self.config, "regime_delta_scale", 0.2))
        for regime in range(self.config.n_regimes):
            scale = (regime - (self.config.n_regimes - 1) / 2) * delta_scale
            deltas[regime, 1, 2, 3] = scale
            deltas[regime, 1, 3, 4] = -scale / 2
            deltas[regime, 1, 0, 4] = scale / 3
        return shared, deltas

    def _reachable_indices(
        self, root: int, shared: np.ndarray, deltas: np.ndarray
    ) -> list[int]:
        """Variables causally reachable from `root` through any regime graph."""
        reach = {int(root)}
        changed = True
        while changed:
            changed = False
            for regime in range(deltas.shape[0]):
                adjacency = shared + deltas[regime]
                for lag in range(1, adjacency.shape[0]):
                    for source in list(reach):
                        for target in np.where(np.abs(adjacency[lag, source]) > 1e-8)[0].tolist():
                            if target not in reach:
                                reach.add(target)
                                changed = True
        return sorted(reach)

    def _event_plan(self, test_start: int) -> list[dict]:
        """Fault plan whose injected step budget matches ``anomaly_rate`` and that
        stays clear of the split boundary and of the window prefix (events whose
        labels would fall outside the evaluated target range are silently lost)."""
        cfg = self.config
        kinds = ("bias", "drift", "stuck", "variance")
        budget = max(2, int(cfg.n_steps * cfg.anomaly_rate))
        test_len = cfg.n_steps - test_start
        guard = min(max(48, cfg.max_lag * 8), max(8, test_len // 4))
        event_count = max(1, min(16, budget // 20))
        duration = min(max(20, budget // event_count), 60)
        low = test_start + guard
        high = cfg.n_steps - duration - guard
        if high < low:
            duration = max(8, min(duration, cfg.n_steps - test_start - 2 * guard))
            high = cfg.n_steps - duration - guard
        span = max(1, high - low)
        event_count = max(1, min(event_count, max(1, span // (duration + 4))))
        starts = np.clip(
            np.linspace(max(low, test_start + 8), max(high, low), event_count, dtype=int),
            test_start + 8,
            cfg.n_steps - duration - 1,
        )
        plan = []
        for index, start in enumerate(starts):
            variable = 2 + index % max(1, cfg.n_variables - 2)
            plan.append(
                {
                    "start": int(start),
                    "stop": int(min(cfg.n_steps, start + duration)),
                    "variable": variable,
                    "kind": kinds[index % len(kinds)],
                    "magnitude": float(0.8 + 0.2 * (index % 3)),
                }
            )
        return plan

    def load(self) -> SourceBundle:
        if self._bundle is not None:
            return self._bundle
        cfg = self.config
        variables = self._variables()
        names = [variable.name for variable in variables]
        shared, deltas = self._graphs(names)
        split = chronological_slices(
            cfg.n_steps, cfg.train_ratio, cfg.val_ratio, cfg.split_embargo
        )
        test_start = split["test"].start or 0
        d = cfg.n_variables
        values = np.zeros((cfg.n_steps, d), dtype=np.float32)
        regimes = np.zeros(cfg.n_steps, dtype=np.int64)
        segment = max(1, cfg.n_steps // (cfg.n_regimes * 3))
        # Open-world protocol: the training/validation segment cycles only through
        # `train_regimes`; the test segment cycles through ALL regimes, so it
        # contains regimes never seen during training (unseen-regime holdout).
        all_regimes = list(range(cfg.n_regimes))
        train_regimes = list(cfg.train_regimes or all_regimes)
        for time in range(test_start):
            regimes[time] = train_regimes[(time // segment) % len(train_regimes)]
        # 关键：test 段必须覆盖全部 regime（含训练未见工况）。全局 segment
        # （n_steps//(K*3)）在 test 段只容纳 1~2 个块，unseen regime 进不了
        # test，novelty AUROC 会全 NaN。test 段按自身长度分 K 块，保证每个
        # regime（含 unseen）都出现且首块满长。
        test_len = cfg.n_steps - test_start
        test_segment = max(1, test_len // len(all_regimes))
        for time in range(test_start, cfg.n_steps):
            regimes[time] = all_regimes[
                ((time - test_start) // test_segment) % len(all_regimes)
            ]
        noise_scale = np.linspace(0.035, 0.07, d)
        values[: cfg.max_lag + 1] = self.rng.normal(0, 0.05, (cfg.max_lag + 1, d))

        plan = self._event_plan(test_start)
        additive = np.zeros_like(values)
        stuck_events: list[tuple[int, int, int]] = []
        for event in plan:
            start, stop, variable = event["start"], event["stop"], event["variable"]
            kind, magnitude = event["kind"], event["magnitude"]
            if kind == "bias":
                additive[start:stop, variable] += magnitude
            elif kind == "drift":
                additive[start:stop, variable] += np.linspace(0.0, magnitude, stop - start)
            elif kind == "variance":
                additive[start:stop, variable] += self.rng.normal(
                    0.0, magnitude, stop - start
                )
            elif kind == "stuck":
                stuck_events.append((start, stop, variable))

        # Simulation loop: faults are injected at the mechanism level (inside the
        # loop), so the disturbance propagates downstream through the adjacency,
        # matching the "inject on the SCM and re-simulate" protocol of the plan.
        for time in range(cfg.max_lag + 1, cfg.n_steps):
            regime = regimes[time]
            adjacency = shared + deltas[regime]
            raw = np.zeros(d)
            for lag in range(1, cfg.max_lag + 1):
                raw += values[time - lag] @ adjacency[lag]
            raw[0] += 0.25 * np.sin(2 * np.pi * time / 600) + 0.08 * np.sin(
                2 * np.pi * time / 97
            )
            values[time] = np.tanh(raw) + self.rng.standard_t(7, d) * noise_scale
            for start, stop, variable in stuck_events:
                if start <= time < stop:
                    values[time, variable] = values[time - 1, variable]
            if additive[time].any():
                values[time] += additive[time]

        events: list[EventRecord] = []
        for event in plan:
            reach = self._reachable_indices(event["variable"], shared, deltas)
            affected = tuple(names[index] for index in reach)
            events.append(
                EventRecord(
                    event_id=f"synthetic-{event['kind']}-{event['start']}",
                    start_index=event["start"],
                    end_index=event["stop"] - 1,
                    event_type=event["kind"],
                    root_cause=names[event["variable"]],
                    severity=event["magnitude"],
                    regime=int(regimes[event["start"]]),
                    affected_variables=affected,
                )
            )

        labels = np.zeros(cfg.n_steps, dtype=np.int8)
        for event in events:
            labels[event.start_index : event.end_index + 1] = 1
        timestamps = pd.date_range(
            "2025-01-01", periods=cfg.n_steps, freq=f"{cfg.sample_period_seconds}s"
        )
        frame = pd.DataFrame(values, columns=names)
        frame.insert(0, "timestamp", timestamps)
        frame["__regime"] = regimes
        frame["__label"] = labels
        effective = shared[None] + deltas
        shared_by_regime = np.repeat(shared[None], cfg.n_regimes, axis=0)
        self._truth = DynamicCausalGraph(
            np.stack([shared_by_regime, effective], axis=0), tuple(names)
        )
        self._bundle = SourceBundle(
            frame=frame,
            variables=variables,
            events=events,
            metadata={"generator": "ClosedLoopSCMGenerator", **asdict(cfg)},
        )
        return self._bundle

    def prepare(self, output_dir: str | Path) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        bundle = self.load()
        slices = chronological_slices(
            len(bundle.frame),
            self.config.train_ratio,
            self.config.val_ratio,
            self.config.split_embargo,
        )
        paths: dict[str, Path] = {}
        for name, selection in slices.items():
            path = output / f"{name}.parquet"
            bundle.frame.iloc[selection].reset_index(drop=True).to_parquet(path, index=False)
            paths[name] = path
        assert self._truth is not None
        paths["truth_graph"] = self._truth.save(output / "truth_graph.npz")
        paths["events"] = dump_json(
            {
                "test_start": int(slices["test"].start),
                "events": [event.to_dict() for event in bundle.events],
            },
            output / "events.json",
        )
        paths["variables"] = dump_json(
            [variable.to_dict() for variable in bundle.variables], output / "variables.json"
        )
        manifest = {
            "schema_version": "1.0",
            "metadata": bundle.metadata,
            "files": {name: str(path.name) for name, path in paths.items()},
        }
        paths["manifest"] = dump_json(manifest, output / "manifest.json")
        return paths


def config_from_mapping(value: dict) -> SyntheticSCMConfig:
    fields = SyntheticSCMConfig.__dataclass_fields__
    return SyntheticSCMConfig(**{key: value[key] for key in fields if key in value})

