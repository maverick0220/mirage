import pytest

from mirage.baselines.causal.tcdf import TCDFAdapter
from mirage.baselines.external import MissingExternalBaselineError


@pytest.mark.contract
def test_missing_external_baseline_never_falls_back(tmp_path):
    with pytest.raises(MissingExternalBaselineError, match="never silently"):
        TCDFAdapter(vendor_root=tmp_path).fit([[0.0], [1.0]], ["x"])

