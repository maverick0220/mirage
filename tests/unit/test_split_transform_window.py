import numpy as np

from mirage.data.split import chronological_slices
from mirage.data.transforms import RobustScaler
from mirage.data.window import IndustrialWindowDataset


def test_chronological_split_has_no_overlap():
    slices = chronological_slices(100, 0.6, 0.2)
    assert slices["train"] == slice(0, 60)
    assert slices["validation"] == slice(60, 80)
    assert slices["test"] == slice(80, 100)


def test_scaler_is_fit_only_from_given_values():
    train = np.arange(20).reshape(10, 2)
    scaler = RobustScaler().fit(train)
    before = scaler.median_.copy()
    transformed = scaler.transform(np.full((3, 2), 10_000))
    np.testing.assert_allclose(scaler.median_, before)
    assert transformed.min() > 100


def test_window_shape_and_target_alignment():
    values = np.arange(60, dtype=np.float32).reshape(20, 3)
    dataset = IndustrialWindowDataset(values, 5)
    item = dataset[2]
    assert item["values"].shape == (5, 3)
    np.testing.assert_allclose(item["target"].numpy(), values[7])

