import numpy as np
import pytest

from preprocessing_core.datasets import BaseDataset
from preprocessing_core.selection import select_channels, select_time_rows
from preprocessing_core.validation import PipelineError


def make_dataset() -> BaseDataset:
    return BaseDataset(
        data=np.arange(24).reshape(4, 3, 2),
        timestamps=np.array([10, 11, 12, 13]),
        labels=np.array([0, 1, 0, 1]),
        channel_labels=np.arange(12).reshape(4, 3),
        channel_names=["a", "b", "c"],
        feature_names=["x", "y"],
        metadata={"aligned_time": {"file_id": np.array([1, 1, 2, 2])}},
    )


def test_select_time_rows_slices_aligned_fields() -> None:
    selected = select_time_rows(make_dataset(), np.array([0, 2]))

    assert selected.data.shape == (2, 3, 2)
    assert selected.timestamps.tolist() == [10, 12]
    assert selected.labels.tolist() == [0, 0]
    assert selected.channel_labels.shape == (2, 3)
    assert selected.metadata["aligned_time"]["file_id"].tolist() == [1, 2]


def test_select_time_rows_accepts_boolean_mask() -> None:
    selected = select_time_rows(make_dataset(), np.array([True, False, True, False]))

    assert selected.timestamps.tolist() == [10, 12]


def test_select_channels_slices_channel_fields() -> None:
    selected = select_channels(make_dataset(), np.array([0, 2]))

    assert selected.data.shape == (4, 2, 2)
    assert selected.channel_labels.shape == (4, 2)
    assert selected.channel_names == ["a", "c"]


def test_invalid_selector_fails() -> None:
    with pytest.raises(PipelineError, match="out-of-range"):
        select_time_rows(make_dataset(), np.array([9]))
