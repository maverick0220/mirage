from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    start_index: int
    end_index: int
    event_type: str
    root_cause: str | None = None
    regime: int | None = None
    severity: float = 1.0
    affected_variables: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.start_index < 0 or self.end_index < self.start_index:
            raise ValueError("Invalid event interval")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["affected_variables"] = list(self.affected_variables)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EventRecord":
        affected = value.get("affected_variables") or ()
        return cls(
            **{**value, "affected_variables": tuple(affected)}
        )

