from pathlib import Path

import numpy as np
import pytest

from mirage.baselines.causal.pcmci_plus import PCMCIPlusAdapter


@pytest.mark.contract
def test_pcmci_plus_runs_from_pinned_vendor_source():
    if not Path("vendor/tigramite_pcmciplus").exists():
        pytest.skip("Vendored Tigramite is not present")
    rng = np.random.default_rng(10)
    values = rng.normal(size=(160, 3))
    values[1:, 1] += 0.7 * values[:-1, 0]
    adapter = PCMCIPlusAdapter(max_lag=2, pc_alpha=0.1).fit(values, ["x", "y", "z"])
    assert adapter.adjacency().shape == (3, 3, 3)
    assert np.isfinite(adapter.adjacency()).all()

