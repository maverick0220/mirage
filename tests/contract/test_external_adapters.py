import numpy as np
import pytest

from mirage.baselines.causal.tcdf import TCDFAdapter
from mirage.baselines.external import MissingExternalBaselineError


@pytest.mark.contract
def test_local_tcdf_is_implemented():
    values = np.random.RandomState(0).randn(40, 3)
    adapter = TCDFAdapter(max_lag=1).fit(values, ["x", "y", "z"])
    assert adapter.adjacency().shape == (2, 3, 3)


@pytest.mark.contract
def test_external_published_baselines_never_fallback(tmp_path):
    from mirage.baselines.causal.external_adapters import CDANsAdapter

    with pytest.raises(MissingExternalBaselineError, match="never silently"):
        CDANsAdapter(vendor_root=tmp_path).fit(np.zeros((10, 2)), ["x", "y"])
