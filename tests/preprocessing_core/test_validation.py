import numpy as np
import pytest

from preprocessing_core.validation import (
    ConfigError,
    PipelineError,
    validate_split_indices,
    validate_state_is_serializable,
)


def test_validate_split_indices_rejects_overlap() -> None:
    indices = {
        "train": np.array([0, 1]),
        "val": np.array([1, 2]),
        "test": np.array([3]),
    }

    with pytest.raises(ConfigError, match="overlaps"):
        validate_split_indices(indices, n_rows=4)


def test_validate_split_indices_rejects_empty_split() -> None:
    indices = {
        "train": np.array([0]),
        "val": np.array([], dtype=int),
        "test": np.array([1]),
    }

    with pytest.raises(ConfigError, match="empty"):
        validate_split_indices(indices, n_rows=2)


def test_serializable_numpy_state_passes() -> None:
    validate_state_is_serializable({"mean": np.array([1.0]), "count": 1}, context="scale")


def test_non_serializable_state_fails() -> None:
    with pytest.raises(PipelineError, match="non-serializable"):
        validate_state_is_serializable({"fn": lambda value: value}, context="scale")
