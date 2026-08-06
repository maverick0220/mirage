from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

import torch


@contextmanager
def runtime_meter(output: dict, key: str) -> Iterator[None]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    output[key] = perf_counter() - start


def peak_memory_bytes() -> int:
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0

