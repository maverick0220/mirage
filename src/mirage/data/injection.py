from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mirage.schemas import EventRecord


@dataclass(frozen=True)
class InjectionSpec:
    start: int
    duration: int
    variable: int
    kind: str = "bias"
    magnitude: float = 2.0


def inject_fault(values: np.ndarray, spec: InjectionSpec, name: str) -> tuple[np.ndarray, EventRecord]:
    result = np.asarray(values, dtype=np.float32).copy()
    stop = min(len(result), spec.start + spec.duration)
    if spec.kind == "bias":
        result[spec.start:stop, spec.variable] += spec.magnitude
    elif spec.kind == "drift":
        ramp = np.linspace(0, spec.magnitude, stop - spec.start)
        result[spec.start:stop, spec.variable] += ramp
    elif spec.kind == "stuck":
        result[spec.start:stop, spec.variable] = result[max(0, spec.start - 1), spec.variable]
    elif spec.kind == "variance":
        rng = np.random.default_rng(spec.start + spec.variable)
        result[spec.start:stop, spec.variable] += rng.normal(
            0, spec.magnitude, stop - spec.start
        )
    else:
        raise ValueError(f"Unsupported fault kind: {spec.kind}")
    return result, EventRecord(
        event_id=f"synthetic-{spec.kind}-{spec.start}",
        start_index=spec.start,
        end_index=stop - 1,
        event_type=spec.kind,
        root_cause=name,
        severity=spec.magnitude,
        affected_variables=(name,),
    )

