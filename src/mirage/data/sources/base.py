from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from mirage.schemas import EventRecord, VariableSpec


@dataclass
class SourceBundle:
    frame: pd.DataFrame
    variables: list[VariableSpec]
    events: list[EventRecord] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class TimeSeriesSource(ABC):
    @abstractmethod
    def load(self) -> SourceBundle:
        raise NotImplementedError

    @abstractmethod
    def prepare(self, output_dir: str | Path) -> dict[str, Path]:
        raise NotImplementedError

