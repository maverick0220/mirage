from __future__ import annotations

import importlib.util
from pathlib import Path


class MissingExternalBaselineError(RuntimeError):
    pass


def require_external(name: str, module: str | None = None, vendor_path: str | Path | None = None) -> None:
    module_available = module is not None and importlib.util.find_spec(module) is not None
    path_available = vendor_path is not None and Path(vendor_path).exists()
    if not module_available and not path_available:
        raise MissingExternalBaselineError(
            f"External baseline '{name}' is not installed. Run "
            "`python scripts/fetch_baselines.py --name " + name + "` and follow its pinned environment instructions. "
            "MIRAGE never silently substitutes a proxy for a named published method."
        )

