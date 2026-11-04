import numpy as np
import pytest

from preprocessing_core.datasets import BaseDataset
from preprocessing_core.transforms import (
    fill_nan,
    register_builtin_transforms,
    repair_rolling_outliers,
    standard_scale,
)
from preprocessing_core.validation import ConfigError, PipelineError


def test_register_builtin_transforms() -> None:
    register_builtin_transforms(replace=True)


def test_fill_nan_replaces_only_data_nans() -> None:
    dataset = BaseDataset(
        data=np.array([[[1.0]], [[np.nan]]]),
        timestamps=np.array([0.0, 1.0]),
    )

    result = fill_nan(dataset, state={}, fit=False, value=-1.0)

    assert result.data[:, 0, 0].tolist() == [1.0, -1.0]
    assert result.timestamps.tolist() == [0.0, 1.0]


def test_standard_scale_fits_channel_feature_state() -> None:
    data = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 14.0], [4.0, 28.0]],
            [[5.0, 18.0], [6.0, 36.0]],
        ]
    )
    dataset = BaseDataset(data=data)
    state = {}

    result = standard_scale(dataset, state=state, fit=True)

    assert state["mean"].shape == (2, 2)
    assert state["std"].shape == (2, 2)
    np.testing.assert_allclose(np.mean(result.data, axis=0), np.zeros((2, 2)))
    np.testing.assert_allclose(np.std(result.data, axis=0), np.ones((2, 2)))


def test_standard_scale_requires_state_when_not_fit() -> None:
    with pytest.raises(PipelineError, match="requires fitted"):
        standard_scale(BaseDataset(data=np.ones((2, 1, 1))), state={}, fit=False)


def test_repair_rolling_outliers_uses_trailing_history() -> None:
    dataset = BaseDataset(data=np.array([1.0, 1.0, 1.0, 10.0]).reshape(4, 1, 1))

    result = repair_rolling_outliers(
        dataset,
        state={},
        fit=False,
        window=3,
        sigma=2.0,
        min_periods=3,
    )

    assert result.data[:, 0, 0].tolist() == [1.0, 1.0, 1.0, 10.0]


def test_repair_rolling_outliers_repairs_nonzero_sigma_outlier() -> None:
    dataset = BaseDataset(data=np.array([1.0, 2.0, 1.0, 20.0]).reshape(4, 1, 1))

    result = repair_rolling_outliers(
        dataset,
        state={},
        fit=False,
        window=3,
        sigma=2.0,
        min_periods=3,
    )

    assert result.data[3, 0, 0] == pytest.approx(4.0 / 3.0)


def test_repair_rolling_outliers_validates_params() -> None:
    with pytest.raises(ConfigError, match="window"):
        repair_rolling_outliers(
            BaseDataset(data=np.ones((3, 1, 1))),
            state={},
            fit=False,
            window=0,
        )
