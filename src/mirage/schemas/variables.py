from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class VariableRole(str, Enum):
    CONTEXT = "context"
    SETPOINT = "setpoint"
    ACTUATOR_COMMAND = "actuator_command"
    ACTUATOR_FEEDBACK = "actuator_feedback"
    PROCESS = "process"
    OUTPUT = "output"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class VariableSpec:
    name: str
    role: VariableRole = VariableRole.UNKNOWN
    unit: str | None = None
    description: str | None = None
    subsystem: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["role"] = self.role.value
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "VariableSpec":
        return cls(**{**value, "role": VariableRole(value.get("role", "unknown"))})

